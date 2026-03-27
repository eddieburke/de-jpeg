import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


def pad_to_multiple(x: torch.Tensor, m: int = 16, b: int = 0):
    h, w = x.shape[-2:]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    padded = F.pad(x, (b, pw + b, b, ph + b), mode="reflect")
    return padded, (ph, pw, b)


def crop_back(x: torch.Tensor, pads) -> torch.Tensor:
    ph, pw, b = pads
    h, w = x.shape[-2:]
    return x[..., b : h - ph - b, b : w - pw - b]


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = ImageOps.exif_transpose(img).convert("RGB")
    return torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.0


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = x.detach().clamp(0, 1).mul(255).add(0.5).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGB")


def make_comparison(original: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
    if original.ndim == 4:
        original = original[0]
    if restored.ndim == 4:
        restored = restored[0]
    _, h, w = original.shape
    _, rh, rw = restored.shape
    mh, mw = max(h, rh), w + rw
    canvas = torch.zeros(3, mh, mw, device=original.device)
    canvas[:, :h, :w] = original
    canvas[:, :rh, w : w + rw] = restored
    line_x = w
    canvas[:, :mh, max(0, line_x - 1) : min(mw, line_x + 1)] = 1.0
    return canvas


def make_heatmap_3ch(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    return torch.cat(
        [
            (1.5 * x).clamp(0, 1),
            (1.0 - (x - 0.5).abs() * 2.0).clamp(0, 1),
            (1.5 * (1.0 - x)).clamp(0, 1),
        ],
        dim=0,
    )
