import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(F.pad(emb, (0, max(0, self.dim - emb.shape[1]))))

class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, c_out * 2)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class TinyJPEGRestorer(nn.Module):
    def __init__(self, base_channels=128, emb_dim=256, depth=3):
        super().__init__()
        self.time_emb = TimeEmbedding(emb_dim)
        self.q_emb = nn.Linear(1, emb_dim)
        b = base_channels
        self.in_conv = nn.Conv2d(6, b, 3, padding=1)
        self.down1 = nn.ModuleList([ResBlock(b, b, emb_dim) for _ in range(depth)])
        self.down2 = nn.ModuleList([nn.Conv2d(b, b*2, 4, 2, 1)] + [ResBlock(b*2, b*2, emb_dim) for _ in range(depth)])
        self.down3 = nn.ModuleList([nn.Conv2d(b*2, b*4, 4, 2, 1)] + [ResBlock(b*4, b*4, emb_dim) for _ in range(depth)])
        self.mid = ResBlock(b*4, b*4, emb_dim)
        self.up1 = nn.ModuleList([nn.ConvTranspose2d(b*4, b*2, 4, 2, 1)] + [ResBlock(b*4, b*2, emb_dim)] + [ResBlock(b*2, b*2, emb_dim) for _ in range(depth-1)])
        self.up2 = nn.ModuleList([nn.ConvTranspose2d(b*2, b, 4, 2, 1)] + [ResBlock(b*2, b, emb_dim)] + [ResBlock(b, b, emb_dim) for _ in range(depth-1)])
        self.out_conv = nn.Conv2d(b, 3, 3, padding=1)

    def forward(self, xt, cond, t, q):
        emb = self.time_emb(t) + self.q_emb(q.view(-1, 1))
        x = self.in_conv(torch.cat([xt, cond], dim=1))
        skips = []
        for blk in self.down1: x = blk(x, emb)
        skips.append(x)
        for blk in self.down2: x = blk(x, emb) if isinstance(blk, ResBlock) else blk(x)
        skips.append(x)
        for blk in self.down3: x = blk(x, emb) if isinstance(blk, ResBlock) else blk(x)
        x = self.mid(x, emb)
        for i, blk in enumerate(self.up1):
            if i == 1: x = torch.cat([x, skips.pop()], dim=1)
            x = blk(x, emb) if isinstance(blk, ResBlock) else blk(x)
        for i, blk in enumerate(self.up2):
            if i == 1: x = torch.cat([x, skips.pop()], dim=1)
            x = blk(x, emb) if isinstance(blk, ResBlock) else blk(x)
        return self.out_conv(x)