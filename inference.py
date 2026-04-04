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

def get_auto_steps(quality_val: float) -> int:
    """ Dynamically calculates necessary EDM sampling steps based on target quality. """
    return max(8, int(25 - (quality_val / 100.0) * 15))

def detect_architecture(ckpt):
    """ Intelligently determines the model architecture from the checkpoint """
    # 1. Try to read explicit config if the new trainer saved it
    cfg = ckpt.get("config", {})
    if "base_channels" in cfg:
        return cfg["base_channels"], cfg.get("emb_dim", 256), cfg.get("depth", 3)
        
    # 2. Fallback: Physically inspect the tensor shapes inside the state_dict
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    clean_sd = {k.replace("module.", ""): v for k, v in state.items()}
    
    base_ch, emb_dim, depth = 64, 256, 2 # absolute defaults
    
    # Read base channels from input convolution
    if "in_conv.weight" in clean_sd:
        base_ch = clean_sd["in_conv.weight"].shape[0]
        
    # Read embedding dimension from time embedding MLP
    if "time_emb.mlp.0.weight" in clean_sd:
        emb_dim = clean_sd["time_emb.mlp.0.weight"].shape[1]
        
    # Count the depth dynamically by looking for down1 blocks
    down1_indices = [int(k.split('.')[1]) for k in clean_sd.keys() 
                     if k.startswith("down1.") and k.split('.')[1].isdigit()]
    if down1_indices:
        depth = max(down1_indices) + 1
        
    return base_ch, emb_dim, depth

def load_model_for_inference(path, device, use_ema=True, log_fn=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    # Auto-detect architecture to prevent shape mismatch crashes
    base_ch, emb_dim, depth = detect_architecture(ckpt)
    if log_fn: 
        log_fn(f"Auto-detected Architecture: {base_ch} Channels, {emb_dim} Emb, Depth {depth}")
    
    model = TinyJPEGRestorer(base_channels=base_ch, emb_dim=emb_dim, depth=depth).to(device)

    if use_ema and "ema" in ckpt:
        state = ckpt["ema"]
        if log_fn: log_fn("Loaded weights: EMA")
    else:
        state = ckpt.get("model", ckpt)
        if log_fn: log_fn("Loaded weights: Standard Model")

    # Strip PyTorch distributed/EMA prefix artifacts
    state = {k.replace("module.", ""): v for k, v in state.items() if "n_averaged" not in k}
    
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt

def get_checkpoint_info(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        bc, ed, dp = detect_architecture(ckpt)
        return {
            "step": ckpt.get("step", 0), 
            "has_ema": "ema" in ckpt,
            "base_channels": bc,
            "depth": dp
        }
    except Exception:
        return None

@torch.no_grad()
def sample_restore(model, cond, q, steps):
    device = cond.device
    b = cond.shape[0]

    if q.shape[0] != b:
        q = q.expand(b)
    q = q.view(-1, 1)

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

    return x.clamp(0, 1)

@torch.no_grad()
def infer_tiled_batched(model, img, q_tensor, steps, tile_size=512, tile_overlap=32, batch_size=4):
    b, c, h, w = img.shape
    img_pad, pads = pad_to_multiple(img, 16, buffer=tile_overlap)
    
    if tile_size <= 0 or (img_pad.shape[2] <= tile_size and img_pad.shape[3] <= tile_size):
        return crop_back(sample_restore(model, img_pad, q_tensor, steps), pads)

    tile_overlap = max(0, min(tile_overlap, tile_size - 1))
    stride = max(1, tile_size - tile_overlap)
    _, _, ph, pw = img_pad.shape
    
    bp = torch.zeros(b, 3, ph, pw, device=img.device)
    ws = torch.zeros(b, 1, ph, pw, device=img.device)

    ramp = torch.linspace(0, 1, tile_overlap, device=img.device) if tile_overlap > 0 else torch.empty(0, device=img.device)
    ramp_flip = ramp.flip(0) if tile_overlap > 0 else torch.empty(0, device=img.device)

    coords = []
    for y in range(0, ph, stride):
        for x in range(0, pw, stride):
            coords.append((y, min(ph, y + tile_size), x, min(pw, x + tile_size)))

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i+batch_size]
        tiles = torch.cat([img_pad[:, :, y:y2, x:x2] for y, y2, x, x2 in batch_coords], dim=0)
        qt = q_tensor.repeat(len(batch_coords), 1)
        
        restored_tiles = sample_restore(model, tiles, qt, steps)

        for j, (y, y2, x, x2) in enumerate(batch_coords):
            tr = restored_tiles[j:j+1]
            wh = torch.ones(y2 - y, device=img.device)
            ww = torch.ones(x2 - x, device=img.device)

            if tile_overlap > 0:
                if y > 0: wh[:min(tile_overlap, y2 - y)] *= ramp[:min(tile_overlap, y2 - y)]
                if y + stride < ph: wh[-(y2 - (y + stride)):] *= ramp_flip[:y2 - (y + stride)]
                if x > 0: ww[:min(tile_overlap, x2 - x)] *= ramp[:min(tile_overlap, x2 - x)]
                if x + stride < pw: ww[-(x2 - (x + stride)):] *= ramp_flip[:x2 - (x + stride)]

            win = (wh.unsqueeze(1) * ww.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            bp[:, :, y:y2, x:x2] += tr * win
            ws[:, :, y:y2, x:x2] += win

    return crop_back(bp / ws.clamp(min=1e-8), pads)

@torch.no_grad()
def infer_stacked_ensemble(model, img, base_quality, tile, overlap, batch_size=4, num_passes=1, use_tta=True):
    device = img.device
    avg_pred = torch.zeros_like(img)
    steps = get_auto_steps(base_quality)
    
    for i in range(num_passes):
        sq = max(1.0, min(100.0, base_quality + (torch.rand(1).item() * 2 - 1) * 2.0)) if num_passes > 1 else base_quality
        qt = torch.tensor([sq / 100.0], device=device)
        
        fh, fv = use_tta and (i % 2 == 1), use_tta and (i % 4 >= 2)
        pi = img
        if fh: pi = torch.flip(pi, [3])
        if fv: pi = torch.flip(pi, [2])

        p = infer_tiled_batched(model, pi, qt, steps, tile, overlap, batch_size)

        if fv: p = torch.flip(p, [2])
        if fh: p = torch.flip(p, [3])

        avg_pred.add_(p)

    return avg_pred.div_(num_passes).clamp_(0, 1)

def run_inference(args: dict, progress_callback=None, log_callback=None):
    device = get_device()
    def log(msg): log_callback(msg) if log_callback else None

    try:
        if progress_callback: progress_callback(10)
        
        model, ckpt = load_model_for_inference(
            args["weights"], device, use_ema=args.get("use_ema", True), log_fn=log
        )
        
        log(f"Checkpoint Loaded (Step {ckpt.get('step', '?')})")
        if progress_callback: progress_callback(20)

        img_t = pil_to_tensor(Image.open(args["input"])).unsqueeze(0).to(device)
        log(f"Image loaded: {img_t.shape[3]}x{img_t.shape[2]}")
        if progress_callback: progress_callback(30)

        auto_steps = get_auto_steps(args.get("quality", 75))
        passes = args.get("passes", 1)
        use_tta = args.get("tta", True)
        
        log(f"Processing... (Steps: {auto_steps}, Passes: {passes}, TTA: {'Yes' if use_tta else 'No'})")
        if progress_callback: progress_callback(50)

        p = infer_stacked_ensemble(
            model, img_t, float(args.get("quality", 75)), 
            args.get("tile", 0), args.get("overlap", 32), args.get("batch_size", 4),
            num_passes=passes, use_tta=use_tta
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
        
        result = {"src": img_t[0].detach().cpu(), "pred": p[0].detach().cpu(), "output_path": out_path}

        del model, img_t, p
        if device.type == "cuda": torch.cuda.empty_cache()
        return result

    except Exception as e:
        log(f"Error: {e}")
        if device.type == "cuda": torch.cuda.empty_cache()
        raise