#!/usr/bin/env python
"""Conditional flow-matching PIXEL decoder (DINO-SAE / PiD style) for gf3.

Rectified flow in image space, conditioned on the pooled DINOv3-H latent: samples a
SHARP image consistent with the latent (no L1 mean-blur, no arbitrary GAN hallucination).
Encodes (latent, frame) pairs on gf3 via the pure-torch gated DINOv3-H encoder.

Launch one per GPU with different --seed/--cond_w for a fast sweep, or DDP for one model.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov3h_gated import DINOv3HGated  # noqa: E402


def temb(t, d=128):
    half = d // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class FiLM(nn.Module):
    def __init__(self, cond, ch):
        super().__init__(); self.f = nn.Linear(cond, ch * 2)

    def forward(self, x, c):
        s, b = self.f(c).chunk(2, -1)
        return x * (1 + s[:, :, None, None]) + b[:, :, None, None]


class Block(nn.Module):
    def __init__(self, ci, co, cond):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, 1, 1); self.c2 = nn.Conv2d(co, co, 3, 1, 1)
        self.n1 = nn.GroupNorm(8, co); self.n2 = nn.GroupNorm(8, co); self.film = FiLM(cond, co)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, c):
        h = F.silu(self.n1(self.c1(x))); h = self.film(h, c); h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, cond_in=1280, base=96):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(cond_in + 128, 256), nn.SiLU(), nn.Linear(256, 256))
        C = 256
        self.in_conv = nn.Conv2d(3, base, 3, 1, 1)
        self.d1 = Block(base, base, C); self.d2 = Block(base, base * 2, C); self.d3 = Block(base * 2, base * 4, C)
        self.mid = Block(base * 4, base * 4, C)
        self.u3 = Block(base * 8, base * 2, C); self.u2 = Block(base * 4, base, C); self.u1 = Block(base * 2, base, C)
        self.out = nn.Conv2d(base, 3, 3, 1, 1)
        self.pool = nn.AvgPool2d(2); self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t, lat):
        c = self.cond(torch.cat([lat, temb(t)], -1))
        h0 = self.in_conv(x)
        h1 = self.d1(h0, c); h2 = self.d2(self.pool(h1), c); h3 = self.d3(self.pool(h2), c)
        m = self.mid(self.pool(h3), c)
        u3 = self.u3(torch.cat([self.up(m), h3], 1), c)
        u2 = self.u2(torch.cat([self.up(u3), h2], 1), c)
        u1 = self.u1(torch.cat([self.up(u2), h1], 1), c)
        return self.out(u1)


def build_pairs(root, camera, enc, n, res, dev):
    cs = int(json.loads((root / "meta/info.json").read_text())["chunks_size"]) if (root / "meta/info.json").exists() else 1000
    vids = sorted(glob.glob(str(root / f"videos/chunk-*/{camera}/episode_*.mp4")))
    rng = np.random.default_rng(0); rng.shuffle(vids)
    imgs = []
    for v in vids:
        cap = cv2.VideoCapture(v); i = 0
        while len(imgs) < n:
            okf, im = cap.read()
            if not okf:
                break
            if i % 10 == 0:
                imgs.append(cv2.resize(im[:, :, ::-1], (256, 256)))
            i += 1
        cap.release()
        if len(imgs) >= n:
            break
    imgs = np.stack(imgs[:n])
    lat = enc.encode_pooled(imgs).astype(np.float32); lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-8
    tgt = np.stack([cv2.resize(im, (res, res)) for im in imgs])
    return lat, tgt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--dino_path", default="temp/lmwm_p0/dinov3h")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ode_steps", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="temp/lmwm_p0/flow_decoder.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cache = Path(f"temp/lmwm_p0/flow_pairs_n{args.n}_r{args.res}.npz")
    if cache.exists():
        d = np.load(cache); lat, tgt = d["lat"], d["tgt"]; print(f"loaded cached pairs {cache}", flush=True)
    else:
        enc = DINOv3HGated(args.dino_path, device=str(dev))             # pure-torch gated DINOv3-H
        print("building (latent, frame) pairs ...", flush=True)
        lat, tgt = build_pairs_enc(args.dataset_root, args.camera, lambda im: _batched_pooled(enc, im), args.n, args.res)
        cache.parent.mkdir(parents=True, exist_ok=True); np.savez(cache, lat=lat, tgt=tgt); del enc; torch.cuda.empty_cache()
    X = torch.from_numpy(lat); Y = torch.from_numpy(tgt.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous()
    ntr = max(1, len(X) - min(512, len(X) // 5))
    vsel = np.arange(ntr, len(X))[:8]

    @torch.no_grad()
    def sample(latents):
        x = torch.randn(len(latents), 3, args.res, args.res, device=dev)
        for k in range(args.ode_steps):
            t = torch.full((len(latents),), k / args.ode_steps, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = net(x, t, latents)
            x = x + (1.0 / args.ode_steps) * v.float()
        return np.clip((x.cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    def save_viz(tag):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        samp = sample(X[vsel].to(dev)); real = np.clip((Y[vsel].numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
        fig, ax = plt.subplots(2, len(vsel), figsize=(len(vsel) * 1.5, 3.2))
        for j in range(len(vsel)):
            ax[0, j].imshow(real[j]); ax[0, j].axis("off"); ax[1, j].imshow(samp[j]); ax[1, j].axis("off")
        fig.suptitle(f"flow decoder {tag} | top real, bottom flow-sample", fontsize=10); fig.tight_layout()
        fig.savefig(str(Path(args.out).with_suffix("")) + f"_sample.png", dpi=110); plt.close(fig)
    print(f"{len(X)} pairs", flush=True)

    net = UNet(1280, args.base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000))
    for s in range(args.steps):
        bi = torch.randint(0, ntr, (args.bs,))
        x1 = Y[bi].to(dev); lb = X[bi].to(dev)
        x0 = torch.randn_like(x1); t = torch.rand(len(x1), device=dev)
        xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = net(xt, t, lb); loss = F.mse_loss(v, x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 1000 == 0:
            print(f"step {s} loss {loss.item():.4f}", flush=True)
        if s > 0 and s % args.save_every == 0:
            net.eval()
            torch.save({"model": net.state_dict(), "base": args.base, "res": args.res, "step": s}, args.out)
            save_viz(f"step{s}")
            print(f"[save+viz @ step {s}]", flush=True); net.train()

    net.eval(); torch.save({"model": net.state_dict(), "base": args.base, "res": args.res, "step": args.steps}, args.out)
    save_viz("final")
    print(f"saved {args.out}", flush=True)


def _batched_pooled(enc, imgs, bs=64):
    out = []
    for s in range(0, len(imgs), bs):
        g = enc.encode_grid(imgs[s:s + bs])            # (b,256,1280) via gated
        out.append(g.mean(1))
    return np.concatenate(out)


def build_pairs_enc(root, camera, encode_pooled, n, res):
    vids = sorted(glob.glob(str(root / f"videos/chunk-*/{camera}/episode_*.mp4")))
    rng = np.random.default_rng(0); rng.shuffle(vids)
    imgs = []
    for v in vids:
        cap = cv2.VideoCapture(v); i = 0
        while len(imgs) < n:
            okf, im = cap.read()
            if not okf:
                break
            if i % 10 == 0:
                imgs.append(cv2.resize(im[:, :, ::-1], (256, 256)))
            i += 1
        cap.release()
        if len(imgs) >= n:
            break
    imgs = np.stack(imgs[:n])
    lat = encode_pooled(imgs).astype(np.float32); lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-8
    tgt = np.stack([cv2.resize(im, (res, res)) for im in imgs])
    return lat, tgt


if __name__ == "__main__":
    main()
