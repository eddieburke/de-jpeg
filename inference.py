import os
import sys
import math
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import TinyJPEGRestorer
from utils import (
    pad_to_multiple,
    crop_back,
    pil_to_tensor,
    tensor_to_pil,
    make_comparison,
    estimate_jpeg_quality,
)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_architecture(ckpt):
    cfg = ckpt.get("config", {})
    if "base_channels" in cfg:
        return cfg["base_channels"], cfg.get("emb_dim", 256), cfg.get("depth", 3)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    clean_sd = {k.replace("module.", ""): v for k, v in state.items()}
    base_ch, emb_dim, depth = 64, 256, 2 
    if "in_conv.weight" in clean_sd:
        base_ch = clean_sd["in_conv.weight"].shape[0]
    if "time_emb.mlp.0.weight" in clean_sd:
        emb_dim = clean_sd["time_emb.mlp.0.weight"].shape[1]
    down1_indices = [int(k.split('.')[1]) for k in clean_sd.keys() if k.startswith("down1.") and k.split('.')[1].isdigit()]
    if down1_indices:
        depth = max(down1_indices) + 1
    return base_ch, emb_dim, depth

def get_checkpoint_info(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        base_ch, emb_dim, depth = detect_architecture(ckpt)
        return {
            "step": ckpt.get("step", "?") if isinstance(ckpt, dict) else "?",
            "base_channels": base_ch,
            "depth": depth,
            "has_ema": "ema" in ckpt if isinstance(ckpt, dict) else False
        }
    except Exception:
        return None

def load_model_for_inference(path, device, use_ema=True, log_fn=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    base_ch, emb_dim, depth = detect_architecture(ckpt)
    if log_fn: log_fn(f"Auto-detected Architecture: {base_ch} Channels, {emb_dim} Emb, Depth {depth}")
    model = TinyJPEGRestorer(base_channels=base_ch, emb_dim=emb_dim, depth=depth).to(device)
    if use_ema and "ema" in ckpt:
        state = ckpt["ema"]
        if log_fn: log_fn("Loaded weights: EMA")
    else:
        state = ckpt.get("model", ckpt)
        if log_fn: log_fn("Loaded weights: Standard Model")
    state = {k.replace("module.", ""): v for k, v in state.items() if "n_averaged" not in k}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt

@torch.no_grad()
def sample_restore(model, cond, q, steps):
    device = cond.device
    b = cond.shape[0]
    if q.shape[0] != b: q = q.expand(b)
    q = q.view(-1, 1)
    sigma_min, sigma_max, rho, sigma_data = 0.002, 3.0, 7.0, 0.5
    # Automatically tie starting noise to the number of steps based on the baseline 20-step schedule size
    # A 20-step schedule takes 19 intervals. This ensures step size remains constant, bringing it in parity
    # with the training engine's default scheduling.
    t_start = 1.0 - (steps - 1) / 19.0 if steps > 1 else 1.0
    ramp = torch.linspace(t_start, 1.0, steps, device=device)
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
def infer_tiled_batched(model, img, q_tensor, steps, tile_size=512, tile_overlap=32, batch_size=4, progress_fn=None):
    b, c, h, w = img.shape
    tile_overlap = max(0, (tile_overlap // 16) * 16)
    if tile_size > 0:
        tile_size = max(16, (tile_size // 16) * 16)
        tile_overlap = min(tile_overlap, tile_size - 16)
    img_pad, pads = pad_to_multiple(img, 16, buffer=tile_overlap)
    if tile_size <= 0 or (img_pad.shape[2] <= tile_size and img_pad.shape[3] <= tile_size):
        return crop_back(sample_restore(model, img_pad, q_tensor, steps), pads)
    _, _, ph, pw = img_pad.shape
    th, tw = min(tile_size, ph), min(tile_size, pw)
    stride_y, stride_x = max(16, th - tile_overlap), max(16, tw - tile_overlap)
    bp = torch.zeros(b, 3, ph, pw, device=img.device)
    ws = torch.zeros(b, 1, ph, pw, device=img.device)
    ramp_y = torch.linspace(0, 1, tile_overlap, device=img.device) if tile_overlap > 0 else None
    ramp_y_flip = ramp_y.flip(0) if tile_overlap > 0 else None
    ramp_x = torch.linspace(0, 1, tile_overlap, device=img.device) if tile_overlap > 0 else None
    ramp_x_flip = ramp_x.flip(0) if tile_overlap > 0 else None
    coords = []
    for y in range(0, ph, stride_y):
        y_s = min(y, ph - th); y_e = y_s + th
        for x in range(0, pw, stride_x):
            x_s = min(x, pw - tw); x_e = x_s + tw
            if (y_s, y_e, x_s, x_e) not in coords: coords.append((y_s, y_e, x_s, x_e))
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i+batch_size]
        tiles = torch.cat([img_pad[:, :, y:y2, x:x2] for y, y2, x, x2 in batch], dim=0)
        qt = q_tensor.repeat(len(batch) * b, 1)
        res_tiles = sample_restore(model, tiles, qt, steps)
        for j, (y, y2, x, x2) in enumerate(batch):
            tr = res_tiles[j*b:(j+1)*b]
            wh, ww = torch.ones(th, device=img.device), torch.ones(tw, device=img.device)
            if tile_overlap > 0:
                if y > 0: wh[:tile_overlap] *= ramp_y
                if y2 < ph: wh[-tile_overlap:] *= ramp_y_flip
                if x > 0: ww[:tile_overlap] *= ramp_x
                if x2 < pw: ww[-tile_overlap:] *= ramp_x_flip
            win = (wh.unsqueeze(1) * ww.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            bp[:, :, y:y2, x:x2] += tr * win
            ws[:, :, y:y2, x:x2] += win
        if progress_fn: progress_fn(i + len(batch), len(coords))
    return crop_back(bp / ws.clamp(min=1e-8), pads)

@torch.no_grad()
def infer_stacked_ensemble(model, img, base_quality, steps, tile, overlap, batch_size=4, num_passes=1, use_tta=True, progress_fn=None, log_fn=None):
    device = img.device
    avg_pred = torch.zeros_like(img)

    
    # Combined Large-Scale Jitter (4/8 multiples) + 1-pixel shifts
    # Decouples from U-Net grid AND gets the anti-aliasing benefits of 1px shifts
    shifts = [
        (0,0), (1,1), (0,1), (1,0),  # Baseline 1-pixel shift cluster
        (4,4), (5,5), (4,5), (5,4),  # Large center shift + 1px cluster
        (2,6), (3,7), (2,7), (3,6),  # Offset pair 1 + 1px cluster
        (6,2), (7,3), (6,3), (7,2)   # Offset pair 2 + 1px cluster
    ]
    
    for i in range(num_passes):
        if log_fn: log_fn(f"Pass {i+1}/{num_passes}")
        sq = max(1.0, min(100.0, base_quality + (torch.rand(1).item() * 2 - 1) * 2.0)) if num_passes > 1 else base_quality
        qt = torch.tensor([sq / 100.0], device=device)
        
        # Determine augmentation for this pass
        sx, sy = shifts[i % len(shifts)] if use_tta else (0, 0)
        fh = use_tta and (i % 2 == 1)
        
        pi = img
        if sx > 0 or sy > 0: pi = F.pad(pi, (sx, 0, sy, 0), mode="reflect")
        if fh: pi = torch.flip(pi, [3])

        def tile_progress(curr, total):
            if progress_fn: progress_fn(int(((i / num_passes) + (curr / total / num_passes)) * 100))

        p = infer_tiled_batched(model, pi, qt, steps, tile, overlap, batch_size, progress_fn=tile_progress)

        if fh: p = torch.flip(p, [3])
        if sx > 0 or sy > 0: p = p[..., sy:, sx:]

        avg_pred.add_(p)

    return avg_pred.div_(num_passes).clamp_(0, 1)

def run_inference(args: dict, progress_callback=None, batch_progress_callback=None, log_callback=None):
    device = get_device()
    def log(msg): log_callback(msg) if log_callback else None
    
    try:
        model, ckpt = load_model_for_inference(args["weights"], device, use_ema=args.get("use_ema", True), log_fn=log)
        
        input_path = args["input"]
        if os.path.isdir(input_path):
            valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
            files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                     if os.path.splitext(f.lower())[1] in valid_exts]
            if not files:
                raise ValueError(f"No valid images found in directory: {input_path}")
            is_batch = True
        else:
            files = [input_path]
            is_batch = False
            
        total_files = len(files)
        os.makedirs(args.get("output_dir", "."), exist_ok=True)
        
        last_result = None
        for idx, fpath in enumerate(files):
            if is_batch: log(f"Processing {idx+1}/{total_files}: {os.path.basename(fpath)}")
            
            img_t = pil_to_tensor(Image.open(fpath).convert("RGB")).unsqueeze(0).to(device)
            
            file_quality = float(args.get("quality", 75))
            if args.get("auto_quality", False):
                est_q = estimate_jpeg_quality(fpath)
                if est_q is not None:
                    file_quality = float(est_q)
                    log(f"Auto-detected quality {est_q} for {os.path.basename(fpath)}")
                else:
                    log(f"Could not auto-detect quality for {os.path.basename(fpath)}, falling back to {int(file_quality)}")

            p = infer_stacked_ensemble(model, img_t, file_quality, args.get("steps", 20), args.get("tile", 0),  
                                       args.get("overlap", 32), args.get("batch_size", 4),
                                       num_passes=args.get("passes", 1), use_tta=args.get("tta", True),
                                       progress_fn=progress_callback, log_fn=log)
                                       
            if is_batch:
                out_path = os.path.join(args.get("output_dir", "."), f"{os.path.splitext(os.path.basename(fpath))[0]}_restored.png")
            else:
                out_path = os.path.join(args.get("output_dir", "."), args.get("output_name", "restored.png"))
                
            tensor_to_pil(p[0]).save(out_path)
            
            if args.get("save_comparison", False):
                comp = make_comparison(img_t[0], p[0])
                tensor_to_pil(comp).save(f"{os.path.splitext(out_path)[0]}_comparison.png")
                
            last_result = {"src": img_t[0].cpu(), "pred": p[0].cpu(), "output_path": out_path}
            
            if batch_progress_callback: 
                batch_progress_callback(int(((idx + 1) / total_files) * 100))
                
        if progress_callback: progress_callback(100)
        return last_result
        
    except Exception as e:
        log(f"Error: {e}"); raise