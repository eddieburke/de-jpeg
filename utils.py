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

def estimate_jpeg_quality(img_path: str) -> int:
    """
    Reverse-engineers the JPEG Quality Factor (0-100) by inspecting 
    the luminance quantization table embedded within the file structure.
    """
    try:
        with Image.open(img_path) as img:
            if img.format != 'JPEG' or not getattr(img, 'quantization', None):
                return None
            
            q = img.quantization.get(0)
            if not q: return None
            
            # IJG standard luminance baseline Q-table for quality=50
            std_q = [
                16, 11, 10, 16, 24, 40, 51, 61,
                12, 12, 14, 19, 26, 58, 60, 55,
                14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62,
                18, 22, 37, 56, 68, 109, 103, 77,
                24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101,
                72, 92, 95, 98, 112, 100, 103, 99
            ]
            
            qualities = []
            for i in range(64):
                if std_q[i] == 0: continue
                val = q[i]
                
                # Reverse scaling mapping relative to Q50
                scale = (val * 100) / std_q[i]
                if scale == 0: continue
                
                if scale <= 100:
                    quality = 100 - scale / 2.0
                else:
                    quality = 5000.0 / scale
                qualities.append(quality)
                
            if not qualities: return None
            return max(1, min(100, int(sum(qualities)/len(qualities))))
    except Exception:
        return None