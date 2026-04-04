import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

def pad_to_multiple(x: torch.Tensor, m: int = 16, buffer: int = 32) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """
    Pads the tensor symmetrically with a reflection buffer to avoid CNN boundary 
    artifacts, ensuring the final dimensions are a multiple of `m`.
    """
    h, w = x.shape[-2:]
    
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    
    total_ph = ph + buffer * 2
    total_pw = pw + buffer * 2
    
    pt = total_ph // 2
    pb = total_ph - pt
    pl = total_pw // 2
    pr = total_pw - pl
    
    return F.pad(x, (pl, pr, pt, pb), mode="reflect"), (pt, pb, pl, pr)

def crop_back(x: torch.Tensor, pads: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Strips away the padding and the reflection buffer to return the original dimensions.
    """
    pt, pb, pl, pr = pads
    return x[..., pt : x.shape[-2] - pb, pl : x.shape[-1] - pr]

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = ImageOps.exif_transpose(img).convert("RGB")
    return torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.0

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = x.detach().clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGB")

def make_comparison(original: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
    if original.ndim == 4: original = original[0]
    if restored.ndim == 4: restored = restored[0]
    
    _, h, w = original.shape
    _, rh, rw = restored.shape
    mh, mw = max(h, rh), w + rw
    
    canvas = torch.zeros(3, mh, mw, device=original.device)
    canvas[:, :h, :w] = original
    canvas[:, :rh, w : w + rw] = restored
    
    line_x = w
    canvas[:, :mh, max(0, line_x - 1) : min(mw, line_x + 1)] = 1.0
    return canvas