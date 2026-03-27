import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


def extract(v: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return v.gather(0, t).view(x.shape[0], 1, 1, 1)


class DiffusionSchedule:
    def __init__(self, steps: int, device: torch.device):
        betas = torch.linspace(1e-4, 0.02, steps, device=device)
        alphabar = torch.cumprod(1.0 - betas, dim=0)
        self.steps = steps
        self.sqrt_ab = torch.sqrt(alphabar)
        self.sqrt_1mab = torch.sqrt(1.0 - alphabar)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(1, half - 1)
        )
        emb = torch.cat(
            [
                torch.sin(t.float().unsqueeze(1) * freq),
                torch.cos(t.float().unsqueeze(1) * freq),
            ],
            dim=1,
        )
        return self.mlp(F.pad(emb, (0, max(0, self.dim - emb.shape[1]))))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, expansion: int = 4):
        super().__init__()
        hidden = cout * expansion
        self.norm1 = nn.GroupNorm(safe_groups(cin), cin)
        self.pw1 = nn.Conv2d(cin, hidden, 1)
        self.dw = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden, padding_mode="reflect"
        )
        self.norm2 = nn.GroupNorm(safe_groups(hidden), hidden)
        self.pw2 = nn.Conv2d(hidden, cout, 1)
        self.emb = nn.Linear(emb_dim, hidden * 2)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x, emb):
        h = self.pw1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=1)
        h = self.dw(h)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.pw2(F.silu(h))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(safe_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, emb=None):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)
        context = torch.bmm(k, v.transpose(1, 2))
        out = torch.bmm(context, q).view(b, c, h, w)
        return x + self.proj(out)


class TinyJPEGRestorer(nn.Module):
    def __init__(self, base_channels=64, emb_dim=192, depth=3):
        super().__init__()
        self.base_channels, self.emb_dim, self.depth = base_channels, emb_dim, depth
        b = base_channels
        self.time = TimeEmbedding(emb_dim)
        self.qemb = nn.Sequential(
            nn.Linear(1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.in_conv = nn.Conv2d(6, b, 3, padding=1, padding_mode="reflect")

        self.enc1 = nn.ModuleList([ResBlock(b, b, emb_dim) for _ in range(depth)])
        self.down1 = nn.Conv2d(b, b * 2, 4, stride=2, padding=1)
        self.enc2 = nn.ModuleList(
            [ResBlock(b * 2, b * 2, emb_dim) for _ in range(depth)]
        )
        self.down2 = nn.Conv2d(b * 2, b * 4, 4, stride=2, padding=1)
        self.enc3 = nn.ModuleList(
            [ResBlock(b * 4, b * 4, emb_dim) for _ in range(depth)]
        )

        self.mid = nn.ModuleList(
            [ResBlock(b * 4, b * 4, emb_dim) for _ in range(depth)]
        )
        self.mid_attn = AttentionBlock(b * 4)

        self.up1 = nn.ConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1)
        self.dec1 = nn.ModuleList(
            [ResBlock(b * 4, b * 2, emb_dim)]
            + [ResBlock(b * 2, b * 2, emb_dim) for _ in range(depth - 1)]
        )
        self.up2 = nn.ConvTranspose2d(b * 2, b, 4, stride=2, padding=1)
        self.dec2 = nn.ModuleList(
            [ResBlock(b * 2, b, emb_dim)]
            + [ResBlock(b, b, emb_dim) for _ in range(depth - 1)]
        )

        self.out_eps = nn.Conv2d(b, 3, 3, padding=1, padding_mode="reflect")
        gh = max(8, b // 2)
        self.out_gate = nn.Sequential(
            nn.Conv2d(b, gh, 3, padding=1, padding_mode="reflect"),
            nn.SiLU(),
            nn.Conv2d(gh, 1, 1),
        )

    def _run_blocks(self, blocks, x, emb):
        for blk in blocks:
            x = blk(x, emb)
        return x

    def forward(self, xt, cond, t, q):
        emb = self.time(t) + self.qemb(q[:, None])
        h_in = torch.cat([xt, cond], dim=1)
        h1 = self._run_blocks(self.enc1, self.in_conv(h_in), emb)
        h2 = self._run_blocks(self.enc2, self.down1(h1), emb)
        h3 = self._run_blocks(self.enc3, self.down2(h2), emb)
        mid = self.mid_attn(self._run_blocks(self.mid, h3, emb))
        u1 = self._run_blocks(
            self.dec1, torch.cat([self.up1(mid, output_size=h2.shape), h2], dim=1), emb
        )
        u2 = self._run_blocks(
            self.dec2, torch.cat([self.up2(u1, output_size=h1.shape), h1], dim=1), emb
        )
        return self.out_eps(F.silu(u2)), self.out_gate(u2)


class TinyJPEGStudentNet(nn.Module):
    def __init__(self, base_channels: int = 64, emb_dim: int = 192, depth: int = 3):
        super().__init__()
        self.base_channels, self.emb_dim, self.depth = base_channels, emb_dim, depth
        b = base_channels
        self.qemb = nn.Sequential(
            nn.Linear(1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.in_conv = nn.Conv2d(6, b, 3, padding=1, padding_mode="reflect")

        self.enc1 = nn.ModuleList([ResBlock(b, b, emb_dim) for _ in range(depth)])
        self.down1 = nn.Conv2d(b, b * 2, 4, stride=2, padding=1)
        self.enc2 = nn.ModuleList(
            [ResBlock(b * 2, b * 2, emb_dim) for _ in range(depth)]
        )
        self.down2 = nn.Conv2d(b * 2, b * 4, 4, stride=2, padding=1)
        self.enc3 = nn.ModuleList(
            [ResBlock(b * 4, b * 4, emb_dim) for _ in range(depth)]
        )

        self.mid = nn.ModuleList(
            [ResBlock(b * 4, b * 4, emb_dim) for _ in range(depth)]
        )
        self.mid_attn = AttentionBlock(b * 4)

        self.up1 = nn.ConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1)
        self.dec1 = nn.ModuleList(
            [ResBlock(b * 4, b * 2, emb_dim)]
            + [ResBlock(b * 2, b * 2, emb_dim) for _ in range(depth - 1)]
        )
        self.up2 = nn.ConvTranspose2d(b * 2, b, 4, stride=2, padding=1)
        self.dec2 = nn.ModuleList(
            [ResBlock(b * 2, b, emb_dim)]
            + [ResBlock(b, b, emb_dim) for _ in range(depth - 1)]
        )

        self.out_eps = nn.Conv2d(b, 3, 3, padding=1, padding_mode="reflect")
        gh = max(8, b // 2)
        self.out_gate = nn.Sequential(
            nn.Conv2d(b, gh, 3, padding=1, padding_mode="reflect"),
            nn.SiLU(),
            nn.Conv2d(gh, 1, 1),
        )

    def _run_blocks(self, blocks, x, emb):
        for blk in blocks:
            x = blk(x, emb)
        return x

    def forward(self, jpeg_img: torch.Tensor, q: torch.Tensor):
        emb = self.qemb(q[:, None])
        h_in = torch.cat([jpeg_img, jpeg_img], dim=1)
        h1 = self._run_blocks(self.enc1, self.in_conv(h_in), emb)
        h2 = self._run_blocks(self.enc2, self.down1(h1), emb)
        h3 = self._run_blocks(self.enc3, self.down2(h2), emb)
        mid = self.mid_attn(self._run_blocks(self.mid, h3, emb))
        u1 = self._run_blocks(
            self.dec1, torch.cat([self.up1(mid, output_size=h2.shape), h2], dim=1), emb
        )
        u2 = self._run_blocks(
            self.dec2, torch.cat([self.up2(u1, output_size=h1.shape), h1], dim=1), emb
        )
        eps_pred = self.out_eps(F.silu(u2))
        gate_logits = self.out_gate(u2)
        return (jpeg_img + eps_pred).clamp(0, 1), gate_logits
