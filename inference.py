import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import (
    DiffusionSchedule,
    TinyJPEGRestorer,
    extract,
)
from utils import (
    pad_to_multiple,
    crop_back,
    pil_to_tensor,
    tensor_to_pil,
    make_comparison,
)

_schedule_cache: dict[int, DiffusionSchedule] = {}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_schedule(steps: int, device: torch.device) -> DiffusionSchedule:
    key = steps
    cached = _schedule_cache.get(key)
    if cached is not None and cached.sqrt_ab.device == device:
        return cached
    sched = DiffusionSchedule(steps, device)
    _schedule_cache[key] = sched
    return sched


def load_model_for_inference(path, device, compile_model=False, log_fn=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    args = ckpt.get("args", {})
    bc = cfg.get("base_channels", args.get("base_channels", 64))
    ed = cfg.get("emb_dim", args.get("emb_dim", 256))
    dp = cfg.get("depth", args.get("depth", 3))

    model = TinyJPEGRestorer(bc, ed, dp).to(device)

    state = ckpt.get("ema", {}).get("shadow", ckpt["model"])
    model.load_state_dict(state, strict=False)
    model.eval()

    if compile_model and hasattr(torch, "compile"):
        model = _try_compile(model, device, log_fn)

    return model, ckpt


def _try_compile(model, device, log_fn=None):
    dummy = torch.zeros(1, 3, 8, 8, device=device)
    q = torch.tensor([0.5], device=device)
    t = torch.zeros(1, dtype=torch.long, device=device)

    for backend, label in [
        (None, "inductor"),
        ("aot_eager", "aot_eager"),
    ]:
        try:
            compiled = (
                torch.compile(model, backend=backend)
                if backend
                else torch.compile(model)
            )
            with torch.no_grad():
                compiled(dummy, dummy, t, q)
            if log_fn:
                log_fn(f"Model compiled with {label} backend")
            return compiled
        except Exception:
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            continue

    if log_fn:
        log_fn("torch.compile unavailable, using eager mode")
    return model


def get_checkpoint_info(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_config", {})
        args = ckpt.get("args", {})
        step = ckpt.get("step", 0)
        has_ema = "ema" in ckpt and ckpt["ema"].get("shadow")
        return {
            "model_config": cfg,
            "args": args,
            "step": step,
            "has_ema": has_ema,
        }
    except Exception:
        return None


@torch.no_grad()
def sample_restore(model, cond, q, schedule, noise_strength=0.0):
    steps, b = schedule.steps, cond.shape[0]
    if q.shape[0] != b:
        if q.shape[0] == 1:
            q = q.expand(b, *q.shape[1:])
        else:
            raise ValueError(f"Batch size mismatch: cond={b}, q={q.shape[0]}")
    T = steps - 1
    t_T = torch.full((b,), T, device=cond.device, dtype=torch.long)
    x = (
        extract(schedule.sqrt_ab, t_T, cond) * cond
        + extract(schedule.sqrt_1mab, t_T, cond)
        * torch.randn_like(cond)
        * noise_strength
    )
    for i in reversed(range(steps)):
        t = torch.full((b,), i, device=x.device, dtype=torch.long)
        pred_eps, gate_logits = model(x, cond, t, q)
        sqrt_ab, sqrt_1mab = (
            extract(schedule.sqrt_ab, t, x),
            extract(schedule.sqrt_1mab, t, x),
        )
        x0_pred = ((x - sqrt_1mab * pred_eps) / (sqrt_ab + 1e-8)).clamp(0, 1)
        if i == 0:
            return x0_pred, gate_logits
        eps_derived = (x - sqrt_ab * x0_pred) / (sqrt_1mab + 1e-8)
        t_prev = t - 1
        sqrt_ab_prev = extract(schedule.sqrt_ab, t_prev, x)
        alpha_t = sqrt_ab**2
        alpha_prev = sqrt_ab_prev**2
        sigma_t = noise_strength * torch.sqrt(
            ((1 - alpha_prev) / (1 - alpha_t)) * (1 - alpha_t / alpha_prev)
        )
        dir_xt = torch.sqrt((1 - alpha_prev - sigma_t**2).clamp(min=0)) * eps_derived
        x = sqrt_ab_prev * x0_pred + dir_xt + sigma_t * torch.randn_like(x)
    return x.clamp(0, 1), gate_logits


@torch.no_grad()
def infer_tiled(model, img, q_tensor, schedule, noise, tile_size=0, tile_overlap=64):
    b, c, h, w = img.shape
    img_pad, pads = pad_to_multiple(img, 16)
    if q_tensor.shape[0] != b:
        if q_tensor.shape[0] == 1:
            q_tensor = q_tensor.expand(b, *q_tensor.shape[1:])
        else:
            q_tensor = q_tensor[:b]

    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        r, g = sample_restore(model, img_pad, q_tensor, schedule, noise)
        return crop_back(r, pads), crop_back(g, pads)

    tile_overlap = max(0, min(tile_overlap, tile_size - 1))
    stride = max(1, tile_size - tile_overlap)
    _, _, ph, pw = img_pad.shape
    bp = torch.zeros(b, 3, ph, pw, device=img.device)
    bg = torch.zeros(b, 1, ph, pw, device=img.device)
    ws = torch.zeros(b, 1, ph, pw, device=img.device)

    if tile_overlap > 0:
        ramp = torch.linspace(0, 1, tile_overlap, device=img.device)
        ramp_flip = ramp.flip(0)
    else:
        ramp = torch.empty(0, device=img.device)
        ramp_flip = torch.empty(0, device=img.device)

    for y in range(0, ph, stride):
        for x in range(0, pw, stride):
            y2, x2 = min(ph, y + tile_size), min(pw, x + tile_size)
            ti = img_pad[:, :, y:y2, x:x2]

            tr, tg = sample_restore(model, ti, q_tensor, schedule, noise)

            wh = torch.ones(y2 - y, device=img.device)
            ww = torch.ones(x2 - x, device=img.device)

            if tile_overlap > 0:
                if y > 0:
                    ov_y = min(tile_overlap, y2 - y)
                    wh[:ov_y] *= ramp[:ov_y]
                if y + stride < ph:
                    ov_y = y2 - (y + stride)
                    if ov_y > 0:
                        wh[-ov_y:] *= ramp_flip[:ov_y]

                if x > 0:
                    ov_x = min(tile_overlap, x2 - x)
                    ww[:ov_x] *= ramp[:ov_x]
                if x + stride < pw:
                    ov_x = x2 - (x + stride)
                    if ov_x > 0:
                        ww[-ov_x:] *= ramp_flip[:ov_x]

            win = (wh.unsqueeze(1) * ww.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            bp[:, :, y:y2, x:x2] += tr * win
            bg[:, :, y:y2, x:x2] += tg * win
            ws[:, :, y:y2, x:x2] += win

    return crop_back(bp / ws.clamp(min=1e-8), pads), crop_back(
        bg / ws.clamp(min=1e-8), pads
    )


@torch.no_grad()
def infer_stacked_ensemble(
    model,
    img,
    base_quality,
    schedule,
    base_noise,
    tile,
    overlap,
    num_passes=3,
    quality_jitter=2.0,
    use_tta=True,
):
    device = img.device
    avg_pred = torch.zeros_like(img)
    avg_gate = torch.zeros(img.shape[0], 1, img.shape[2], img.shape[3], device=device)
    for i in range(num_passes):
        if num_passes > 1:
            jitter = (torch.rand(1).item() * 2 - 1) * quality_jitter
            sq = max(1.0, min(100.0, base_quality + jitter))
        else:
            sq = base_quality
        qt = torch.tensor([sq / 100.0], device=device)
        fh, fv = use_tta and (i % 2 == 1), use_tta and (i % 4 >= 2)
        pi = img
        if fh:
            pi = torch.flip(pi, [3])
        if fv:
            pi = torch.flip(pi, [2])

        p, g = infer_tiled(model, pi, qt, schedule, base_noise, tile, overlap)

        if fv:
            p, g = torch.flip(p, [2]), torch.flip(g, [2])
        if fh:
            p, g = torch.flip(p, [3]), torch.flip(g, [3])

        avg_pred.add_(p)
        avg_gate.add_(g)

    avg_pred.div_(num_passes).clamp_(0, 1)
    avg_gate.div_(num_passes).clamp_(0, 1)
    return avg_pred, avg_gate


def run_inference(args: dict, progress_callback=None, log_callback=None):
    device = get_device()

    def log(msg):
        if log_callback:
            log_callback(msg)

    try:
        log(f"Using device: {device}")
        log("Loading checkpoint...")
        if progress_callback:
            progress_callback(5)

        model, ckpt = load_model_for_inference(
            args["weights"],
            device,
            compile_model=args.get("compile", False),
            log_fn=log,
        )

        ckpt_args = ckpt.get("args", {})
        effective_steps = args.get("steps", ckpt_args.get("diffusion_steps", 8))

        log(f"Model loaded (step {ckpt.get('step', '?')})")
        log("Loading image...")
        if progress_callback:
            progress_callback(10)

        img_t = pil_to_tensor(Image.open(args["input"])).unsqueeze(0).to(device)
        _, _, ih, iw = img_t.shape
        log(f"Image size: {iw}x{ih}")
        if progress_callback:
            progress_callback(15)

        passes = args.get("passes", 1)
        use_tta = args.get("tta", True)

        log(
            f"Running diffusion inference ({effective_steps} steps, {passes} passes, TTA={'on' if use_tta else 'off'})..."
        )
        if progress_callback:
            progress_callback(20)

        p, g = infer_stacked_ensemble(
            model,
            img_t,
            float(args.get("quality", 75)),
            _get_schedule(effective_steps, device),
            args.get("noise", 0.05),
            args.get("tile", 0),
            args.get("overlap", 32),
            num_passes=passes,
            quality_jitter=args.get("q_jitter", 2.0),
            use_tta=use_tta,
        )

        if progress_callback:
            progress_callback(90)

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

        if progress_callback:
            progress_callback(100)

        result = {
            "src": img_t[0].detach().cpu(),
            "pred": p[0].detach().cpu(),
            "gate": g[0].detach().cpu(),
            "output_path": out_path,
            "comparison_path": comparison_path,
        }

        del model, img_t, p, g
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        log(f"Error: {e}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise
