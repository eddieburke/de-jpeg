import os
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import TinyJPEGRestorer
from utils import (
    pad_to_multiple,
    crop_back,
    pil_to_tensor,
    tensor_to_pil,
    make_comparison,
)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_for_inference(path, device, compile_model=False, log_fn=None, use_ema=True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    # HARDCODED to match your training code exactly
    model = TinyJPEGRestorer(64, 256, 2).to(device)

    # Conditionally load EMA or Full weights based on UI toggle
    if use_ema and "ema" in ckpt:
        state = ckpt["ema"]
        if log_fn: log_fn("Loaded weights: EMA (Exponential Moving Average)")
    else:
        state = ckpt.get("model", ckpt)
        if log_fn: log_fn("Loaded weights: Full Model (Standard)")

    # Strip module prefixes from DDP or EMA wrap
    state = {k.replace("module.", ""): v for k, v in state.items() if "n_averaged" not in k}
    
    # strict=True ensures we catch architecture mismatches
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, ckpt

def get_checkpoint_info(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        step = ckpt.get("step", 0)
        has_ema = "ema" in ckpt
        return {
            "model_config": {"base_channels": 64, "emb_dim": 256, "depth": 2},
            "args": {},
            "step": step,
            "has_ema": has_ema,
        }
    except Exception:
        return None

@torch.no_grad()
def sample_restore(model, cond, q, steps):
    """ EDM Sampling Logic (Matches your Training Engine) """
    device = cond.device
    b = cond.shape[0]

    if q.shape[0] != b:
        q = q.expand(b, *q.shape[1:])

    sigma_min, sigma_max, rho, sigma_data = 0.002, 3.0, 7.0, 0.5
    ramp = torch.linspace(0, 1, steps, device=device)
    sigmas = (sigma_max**(1/rho) + ramp * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

    x = cond + torch.randn_like(cond) * sigmas[0]

    for i in range(steps):
        s_curr, s_next = sigmas[i], sigmas[i+1]
        c_in = 1.0 / (s_curr ** 2 + sigma_data ** 2).sqrt()
        c_out = (s_curr * sigma_data) / (s_curr ** 2 + sigma_data ** 2).sqrt()
        c_skip = sigma_data ** 2 / (s_curr ** 2 + sigma_data ** 2)
        c_noise = s_curr.log() / 4.0

        with torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu", dtype=torch.float16, enabled=device.type=="cuda"):
            pred = model(x * c_in, cond, c_noise.expand(b), q)
            
        x0_pred = (c_skip * x + c_out * pred).clamp(0.0, 1.0)

        d_curr = (x - x0_pred) / s_curr
        x = x + d_curr * (s_next - s_curr)

    # Since the new model doesn't generate a gate mask natively, we return a blank tensor
    dummy_gate = torch.zeros(b, 1, cond.shape[2], cond.shape[3], device=device)
    return x.clamp(0, 1), dummy_gate

@torch.no_grad()
def infer_tiled(model, img, q_tensor, steps, tile_size=0, tile_overlap=64):
    b, c, h, w = img.shape
    img_pad, pads = pad_to_multiple(img, 16)
    if q_tensor.shape[0] != b:
        q_tensor = q_tensor.expand(b, *q_tensor.shape[1:]) if q_tensor.shape[0] == 1 else q_tensor[:b]

    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        r, g = sample_restore(model, img_pad, q_tensor, steps)
        return crop_back(r, pads), crop_back(g, pads)

    tile_overlap = max(0, min(tile_overlap, tile_size - 1))
    stride = max(1, tile_size - tile_overlap)
    _, _, ph, pw = img_pad.shape
    bp = torch.zeros(b, 3, ph, pw, device=img.device)
    bg = torch.zeros(b, 1, ph, pw, device=img.device)
    ws = torch.zeros(b, 1, ph, pw, device=img.device)

    ramp = torch.linspace(0, 1, tile_overlap, device=img.device) if tile_overlap > 0 else torch.empty(0, device=img.device)
    ramp_flip = ramp.flip(0) if tile_overlap > 0 else torch.empty(0, device=img.device)

    for y in range(0, ph, stride):
        for x in range(0, pw, stride):
            y2, x2 = min(ph, y + tile_size), min(pw, x + tile_size)
            ti = img_pad[:, :, y:y2, x:x2]

            tr, tg = sample_restore(model, ti, q_tensor, steps)

            wh = torch.ones(y2 - y, device=img.device)
            ww = torch.ones(x2 - x, device=img.device)

            if tile_overlap > 0:
                if y > 0: wh[:min(tile_overlap, y2 - y)] *= ramp[:min(tile_overlap, y2 - y)]
                if y + stride < ph: wh[-(y2 - (y + stride)):] *= ramp_flip[:y2 - (y + stride)]
                if x > 0: ww[:min(tile_overlap, x2 - x)] *= ramp[:min(tile_overlap, x2 - x)]
                if x + stride < pw: ww[-(x2 - (x + stride)):] *= ramp_flip[:x2 - (x + stride)]

            win = (wh.unsqueeze(1) * ww.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            bp[:, :, y:y2, x:x2] += tr * win
            bg[:, :, y:y2, x:x2] += tg * win
            ws[:, :, y:y2, x:x2] += win

    return crop_back(bp / ws.clamp(min=1e-8), pads), crop_back(bg / ws.clamp(min=1e-8), pads)

@torch.no_grad()
def infer_stacked_ensemble(
    model, img, base_quality, steps, tile, overlap, num_passes=3, quality_jitter=2.0, use_tta=True
):
    device = img.device
    avg_pred = torch.zeros_like(img)
    avg_gate = torch.zeros(img.shape[0], 1, img.shape[2], img.shape[3], device=device)
    
    for i in range(num_passes):
        sq = max(1.0, min(100.0, base_quality + (torch.rand(1).item() * 2 - 1) * quality_jitter)) if num_passes > 1 else base_quality
        qt = torch.tensor([sq / 100.0], device=device)
        fh, fv = use_tta and (i % 2 == 1), use_tta and (i % 4 >= 2)
        
        pi = img
        if fh: pi = torch.flip(pi, [3])
        if fv: pi = torch.flip(pi, [2])

        p, g = infer_tiled(model, pi, qt, steps, tile, overlap)

        if fv: p, g = torch.flip(p, [2]), torch.flip(g, [2])
        if fh: p, g = torch.flip(p, [3]), torch.flip(g, [3])

        avg_pred.add_(p)
        avg_gate.add_(g)

    return avg_pred.div_(num_passes).clamp_(0, 1), avg_gate.div_(num_passes).clamp_(0, 1)

def run_inference(args: dict, progress_callback=None, log_callback=None):
    device = get_device()
    def log(msg): log_callback(msg) if log_callback else None

    try:
        log(f"Using device: {device}")
        if progress_callback: progress_callback(5)

        use_ema = args.get("use_ema", True)
        model, ckpt = load_model_for_inference(
            args["weights"], device, compile_model=args.get("compile", False), log_fn=log, use_ema=use_ema
        )

        effective_steps = args.get("steps", 20)
        log(f"Model loaded (step {ckpt.get('step', '?')})")
        if progress_callback: progress_callback(10)

        img_t = pil_to_tensor(Image.open(args["input"])).unsqueeze(0).to(device)
        _, _, ih, iw = img_t.shape
        log(f"Image size: {iw}x{ih}")
        if progress_callback: progress_callback(15)

        passes = args.get("passes", 1)
        use_tta = args.get("tta", True)
        log(f"Running EDM inference ({effective_steps} steps, {passes} passes, TTA={'on' if use_tta else 'off'})...")
        if progress_callback: progress_callback(20)

        p, g = infer_stacked_ensemble(
            model, img_t, float(args.get("quality", 75)), effective_steps,
            args.get("tile", 0), args.get("overlap", 32), num_passes=passes,
            quality_jitter=args.get("q_jitter", 2.0), use_tta=use_tta,
        )

        if progress_callback: progress_callback(90)

        out_path = args.get("output", "output.png")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tensor_to_pil(p[0]).save(out_path)
        log(f"Saved to: {out_path}")

        comparison_path = None
        if args.get("save_comparison", False):
            comp = make_comparison(img_t[0], p[0])
            stem, ext = os.path.splitext(out_path)
            comparison_path = f"{stem}_comparison{ext}"
            tensor_to_pil(comp).save(comparison_path)
            log(f"Saved comparison: {comparison_path}")

        if progress_callback: progress_callback(100)

        result = {
            "src": img_t[0].detach().cpu(),
            "pred": p[0].detach().cpu(),
            "gate": g[0].detach().cpu(),
            "output_path": out_path,
            "comparison_path": comparison_path,
        }

        del model, img_t, p, g
        if device.type == "cuda": torch.cuda.empty_cache()

        return result

    except Exception as e:
        log(f"Error: {e}")
        if device.type == "cuda": torch.cuda.empty_cache()
        raise