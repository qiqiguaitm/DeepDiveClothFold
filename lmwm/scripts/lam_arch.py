"""LAM architecture variants for the capacity ladder: cnn | convattn | transformer.

Each build_* returns a module matching the interfaces used in optimize_subgoal.py:
  inverse(gt, gf) -> code(B,cd) ; forward(gt, code) -> grid(B,din,16,16) ; predm(gt) -> code(B,cd)
grids are (B, din, 16, 16). cnn = existing light CNN (from train_lawm_patch/optimize_subgoal);
transformer/convattn add global attention over the 256 grid tokens, scaled by (width, depth).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _heads(w):
    return max(4, w // 64)


class AttnBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim); self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.n1(x); a, _ = self.attn(h, h, h, need_weights=False); x = x + a
        return x + self.ff(self.n2(x))


# ---------------- TRANSFORMER (tokens over the 16x16 grid) ----------------
class TFInverse(nn.Module):
    def __init__(self, din, code_dim, width, depth):
        super().__init__()
        self.inp = nn.Linear(din, width)
        self.pos = nn.Parameter(torch.zeros(1, 256, width))
        self.frame = nn.Parameter(torch.zeros(2, 1, width))
        self.q = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([AttnBlock(width, _heads(width)) for _ in range(depth)])
        self.out = nn.Linear(width, code_dim); self.ln = nn.LayerNorm(code_dim)

    def _tok(self, g):
        return self.inp(g.flatten(2).transpose(1, 2)) + self.pos

    def forward(self, gt, gf):
        a = self._tok(gt) + self.frame[0]; b = self._tok(gf) + self.frame[1]
        x = torch.cat([self.q.expand(a.shape[0], -1, -1), a, b], 1)
        for bl in self.blocks:
            x = bl(x)
        return self.ln(self.out(x[:, 0]))


class TFForward(nn.Module):
    def __init__(self, din, code_dim, width, depth):
        super().__init__()
        self.inp = nn.Linear(din, width); self.pos = nn.Parameter(torch.zeros(1, 256, width))
        self.code = nn.Linear(code_dim, width)
        self.blocks = nn.ModuleList([AttnBlock(width, _heads(width)) for _ in range(depth)])
        self.out = nn.Linear(width, din)

    def forward(self, gt, code):
        b = gt.shape[0]
        x = self.inp(gt.flatten(2).transpose(1, 2)) + self.pos
        x = torch.cat([self.code(code)[:, None, :], x], 1)
        for bl in self.blocks:
            x = bl(x)
        o = self.out(x[:, 1:])                                  # (B,256,din)
        return o.transpose(1, 2).reshape(b, -1, 16, 16)


class TFPredM(nn.Module):
    def __init__(self, din, code_dim, width, depth):
        super().__init__()
        self.inp = nn.Linear(din, width); self.pos = nn.Parameter(torch.zeros(1, 256, width))
        self.q = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([AttnBlock(width, _heads(width)) for _ in range(depth)])
        self.out = nn.Linear(width, code_dim); self.ln = nn.LayerNorm(code_dim)

    def forward(self, gt):
        x = self.inp(gt.flatten(2).transpose(1, 2)) + self.pos
        x = torch.cat([self.q.expand(x.shape[0], -1, -1), x], 1)
        for bl in self.blocks:
            x = bl(x)
        return self.ln(self.out(x[:, 0]))


# ---------------- CONVATTN (conv stem + attention blocks) ----------------
class _ConvStem(nn.Module):
    def __init__(self, cin, width):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(cin, width, 3, 1, 1), nn.GroupNorm(8, width), nn.GELU(),
                               nn.Conv2d(width, width, 3, 1, 1), nn.GroupNorm(8, width), nn.GELU())

    def forward(self, x):
        return self.c(x)


class CAInverse(nn.Module):
    def __init__(self, din, code_dim, width, depth):
        super().__init__()
        self.stem = _ConvStem(2 * din, width)
        self.blocks = nn.ModuleList([AttnBlock(width, _heads(width)) for _ in range(depth)])
        self.out = nn.Linear(width, code_dim); self.ln = nn.LayerNorm(code_dim)

    def forward(self, gt, gf):
        x = self.stem(torch.cat([gt, gf], 1)).flatten(2).transpose(1, 2)
        for bl in self.blocks:
            x = bl(x)
        return self.ln(self.out(x.mean(1)))


class CAForward(nn.Module):
    def __init__(self, din, code_dim, width, depth):
        super().__init__()
        self.stem = _ConvStem(din + code_dim, width)
        self.blocks = nn.ModuleList([AttnBlock(width, _heads(width)) for _ in range(depth)])
        self.out = nn.Conv2d(width, din, 3, 1, 1)

    def forward(self, gt, code):
        c = code[:, :, None, None].expand(-1, -1, gt.shape[2], gt.shape[3])
        x = self.stem(torch.cat([gt, c], 1))
        b, w, h, ww = x.shape
        t = x.flatten(2).transpose(1, 2)
        for bl in self.blocks:
            t = bl(t)
        x = t.transpose(1, 2).reshape(b, w, h, ww)
        return self.out(x)


class CAPredM(nn.Module):
    def __init__(self, din, code_dim, width, depth):
        super().__init__()
        self.stem = _ConvStem(din, width)
        self.blocks = nn.ModuleList([AttnBlock(width, _heads(width)) for _ in range(depth)])
        self.out = nn.Linear(width, code_dim); self.ln = nn.LayerNorm(code_dim)

    def forward(self, gt):
        x = self.stem(gt).flatten(2).transpose(1, 2)
        for bl in self.blocks:
            x = bl(x)
        return self.ln(self.out(x.mean(1)))


def build_lam(arch, din, code_dim, width, depth):
    """Return (inverse, forward, predm) modules for the chosen arch."""
    if arch == "cnn":
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_lawm_patch import InverseEnc, ForwardDec
        from optimize_subgoal import PredM
        return InverseEnc(din, code_dim), ForwardDec(din, code_dim), PredM(din, code_dim)
    if arch == "transformer":
        return (TFInverse(din, code_dim, width, depth), TFForward(din, code_dim, width, depth),
                TFPredM(din, code_dim, width, depth))
    if arch == "convattn":
        return (CAInverse(din, code_dim, width, depth), CAForward(din, code_dim, width, depth),
                CAPredM(din, code_dim, width, depth))
    raise ValueError(arch)


def nparams(m):
    return sum(p.numel() for p in m.parameters())
