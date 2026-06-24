"""Centroid decoder: train a small CNN that maps a patch grid (dim,P,P) → 128×128 RGB,
so cluster-centroid grids can be visualized as synthetic prototype frames.

Ported from crave_decoder_scale_ablation.{Dec,train_dec}. The decoder is fixed 16→128
(three 2× upsamples), which is why every encoder is configured to emit a 16×16 grid.
"""
from __future__ import annotations

import numpy as np

P = 16  # patch-grid side the decoder expects (encoders pin res so this holds)


def _build_dec(nn, din, dec="small"):
    def up(i, o):
        return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))

    def cb(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True))

    if dec == "tiny":
        head = nn.Sequential(nn.Conv2d(din, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        net = nn.Sequential(up(128, 64), up(64, 32), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())
    elif dec == "small":
        head = nn.Sequential(nn.Conv2d(din, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        net = nn.Sequential(up(256, 128), up(128, 64), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())
    elif dec == "medium":
        head = nn.Sequential(nn.Conv2d(din, 384, 1), nn.BatchNorm2d(384), nn.ReLU(True))
        net = nn.Sequential(up(384, 192), up(192, 96), nn.ConvTranspose2d(96, 3, 4, 2, 1), nn.Tanh())
    elif dec == "big":
        head = nn.Sequential(nn.Conv2d(din, 512, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        net = nn.Sequential(up(512, 384), up(384, 192), up(192, 96), cb(96, 96), nn.Conv2d(96, 3, 3, 1, 1), nn.Tanh())
    elif dec == "xl":
        head = nn.Sequential(nn.Conv2d(din, 768, 1), nn.BatchNorm2d(768), nn.ReLU(True))
        net = nn.Sequential(up(768, 512), up(512, 384), up(384, 256), cb(256, 256), cb(256, 128), cb(128, 128),
                            nn.Conv2d(128, 3, 3, 1, 1), nn.Tanh())
    else:
        raise ValueError(dec)
    return head, net


def make_decoder(din, dec="small"):
    """Return a torch.nn.Module decoder (din-channel grid → 3×128×128)."""
    import torch.nn as nn

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.head, self.net = _build_dec(nn, din, dec)

        def forward(self, g):
            return self.net(self.head(g))

    return Dec()


def train_dec(grids, imgs, din, dec="small", epochs=55, device="cuda"):
    """Train a centroid decoder; returns a `decode(grid_np)->uint8 images` closure.

    grids: (N, din, P, P) ; imgs: (N, 128, 128, 3) uint8 targets.
    """
    import torch
    mu = grids.mean(axis=(0, 2, 3), dtype=np.float32)
    sd = grids.astype(np.float32).std(axis=(0, 2, 3)) + 1e-4
    muT = torch.from_numpy(mu).view(1, din, 1, 1).to(device)
    sdT = torch.from_numpy(sd).view(1, din, 1, 1).to(device)
    Y = torch.from_numpy(imgs.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous().to(device)
    Gg = torch.from_numpy(grids.astype(np.float32)).to(device)
    D = make_decoder(din, dec).to(device)
    opt = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    n = len(grids); bs = 64
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for b in range(0, n, bs):
            bi = perm[b:b + bs]
            x = (Gg[bi] - muT) / sdT
            pred = D(x)
            loss = (pred - Y[bi]).abs().mean() + 0.5 * ((pred - Y[bi]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    D.eval()

    def decode(gnp):
        with torch.no_grad():
            x = (torch.from_numpy(np.atleast_3d(gnp).astype(np.float32)).to(device).view(-1, din, P, P) - muT) / sdT
            o = D(x).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    del Gg, Y
    torch.cuda.empty_cache()
    return decode
