#!/usr/bin/env python
"""SHARP + temporally-consistent decoder from DINOv3-base *GRID* features (768x16x16).

The pooled decoder (train_dinov3base_video_decoder.py) is blurry because pooled 768D
carries NO spatial layout — one vector per frame -> the decode is a soft prototype.
This uses the DINOv3-base 16x16 patch GRID (spatial tokens preserved) -> the decoder
reconstructs spatial detail sharply. Temporal consistency is kept via the same
decoded-motion == real-motion loss.

  frame -> DINOv3-base encode_grid -> (768,16,16) -> make_decoder("big") -> 3x128x128
  loss over clip = recon + tc_weight*|dD-dI| + gdl_weight*grad-diff

Run in the CRAVE env (DINOv3-capable): /home/tim/miniconda3/envs/srpo/bin/python

Usage:
    /home/tim/miniconda3/envs/srpo/bin/python lmvla/lmwm/scripts/train_dinov3base_grid_decoder.py \
        --feature_dir lmvla/crave/data/kai_dinov3base \
        --dataset_root kai0/data/Task_A/kai0_base \
        --clip 6 --n_clips 1600 --epochs 60 --dec big \
        --out lmvla/lmwm/checkpoints/dinov3base_decoder/kai_grid_dec.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

REPO = Path("/home/tim/workspace/deepdive_kai0")
sys.path.insert(0, str(REPO / "lmvla/crave/src"))
from crave.decoding import make_decoder  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


def load_index(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    return idx["E"].astype(np.int64), idx["FR"].astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", required=True, type=Path)          # only for E/FR frame index
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--clip", type=int, default=6)
    ap.add_argument("--n_clips", type=int, default=1600)
    ap.add_argument("--dec", default="big", choices=["small", "medium", "big", "xl"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tc_weight", type=float, default=0.5)
    ap.add_argument("--gdl_weight", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    dev = args.device
    T = args.clip
    E, FR = load_index(args.feature_dir)
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    rng = np.random.default_rng(args.seed)

    # group by (ep, FR) into contiguous ranges; sample clips of T consecutive frames
    order = np.lexsort((FR, E)); E, FR = E[order], FR[order]
    eps, starts = np.unique(E, return_index=True); ends = np.append(starts[1:], len(E))
    valid = [(e, s, en) for e, s, en in zip(eps, starts, ends) if en - s >= T]
    chosen = []
    for _ in range(args.n_clips):
        e, s, en = valid[rng.integers(len(valid))]
        st = rng.integers(s, en - T + 1)
        chosen.append((int(e), int(st)))

    # read frames (224 for encoder, 128 target), grouped by episode
    enc_imgs = np.zeros((args.n_clips, T, 224, 224, 3), np.uint8)
    tgt = np.zeros((args.n_clips, T, 128, 128, 3), np.uint8)
    by_ep: dict[int, list[tuple[int, int]]] = {}
    for ci, (e, st) in enumerate(chosen):
        by_ep.setdefault(e, []).append((ci, st))
    done = 0
    for e, items in by_ep.items():
        mp4 = args.dataset_root / f"videos/chunk-{e // cs:03d}/{args.camera}/episode_{e:06d}.mp4"
        cap = cv2.VideoCapture(str(mp4))
        for ci, st in items:
            for t in range(T):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(FR[st + t])); ok, fr = cap.read()
                if ok:
                    rgb = fr[:, :, ::-1]
                    enc_imgs[ci, t] = cv2.resize(rgb, (224, 224))
                    tgt[ci, t] = cv2.resize(rgb, (128, 128))
        cap.release(); done += len(items)
        if done % 300 < len(items):
            print(f"  read {done}/{args.n_clips} clips", flush=True)

    # encode DINOv3-base grid (768,16,16) for all frames
    print("encoding DINOv3-base grid ...", flush=True)
    enc = load_encoder("dinov3-base", device=dev)
    flat = enc_imgs.reshape(-1, 224, 224, 3)
    G = np.zeros((len(flat), 768, 16, 16), np.float16)
    for b in range(0, len(flat), 256):
        G[b:b + 256] = enc.encode_grid(flat[b:b + 256], bs=64)
        if b % 4096 == 0:
            print(f"  encoded {b}/{len(flat)}", flush=True)
    enc.unload()
    G = G.reshape(args.n_clips, T, 768, 16, 16)
    # l2-normalize each patch token (channel dim) — matches how CRAVE uses grids
    Gn = G.astype(np.float32)
    Gn = Gn / (np.linalg.norm(Gn, axis=2, keepdims=True) + 1e-8)

    X = torch.from_numpy(Gn)                                             # (N,T,768,16,16)
    Y = torch.from_numpy(tgt.astype(np.float32) / 127.5 - 1).permute(0, 1, 4, 2, 3).contiguous()
    dec = make_decoder(768, args.dec).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    n, bs = len(X), 8
    n_val = max(24, n // 10); perm0 = torch.randperm(n); val_i, tr_i = perm0[:n_val], perm0[n_val:]

    def run(xc):                                                         # (B,T,768,16,16) -> (B,T,3,128,128)
        B = xc.shape[0]
        d = dec(xc.reshape(B * T, 768, 16, 16))
        return d.view(B, T, 3, 128, 128)

    def losses(pred, y):
        recon = (pred - y).abs().mean() + 0.5 * ((pred - y) ** 2).mean()
        tc = (torch.diff(pred, dim=1) - torch.diff(y, dim=1)).abs().mean()
        gdl = ((pred[..., 1:] - pred[..., :-1]) - (y[..., 1:] - y[..., :-1])).abs().mean() \
            + ((pred[..., 1:, :] - pred[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])).abs().mean()
        return recon, tc, gdl

    for ep in range(args.epochs):
        dec.train(); p = tr_i[torch.randperm(len(tr_i))]
        for b in range(0, len(p), bs):
            bi = p[b:b + bs]; x, y = X[bi].to(dev), Y[bi].to(dev)
            pred = run(x); recon, tc, gdl = losses(pred, y)
            loss = recon + args.tc_weight * tc + args.gdl_weight * gdl
            opt.zero_grad(); loss.backward(); opt.step()
        if ep == 0 or (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            dec.eval()
            with torch.no_grad():
                vp = run(X[val_i].to(dev)); vr, vtc, _ = losses(vp, Y[val_i].to(dev))
            print(f"epoch {ep + 1}/{args.epochs}  val_recon={vr.item():.4f}  val_tc={vtc.item():.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dec.state_dict(), "dec": args.dec, "din": 768,
                "meta": {"clip": T, "n_clips": args.n_clips, "epochs": args.epochs,
                         "val_recon": vr.item(), "val_tc": vtc.item(),
                         "input": "l2-normalized DINOv3-base GRID 768x16x16"}}, args.out)
    print(f"saved {args.out}  val_recon={vr.item():.4f} val_tc={vtc.item():.4f}")


if __name__ == "__main__":
    main()
