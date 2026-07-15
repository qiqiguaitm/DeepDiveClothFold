#!/usr/bin/env python
"""Train a TEMPORALLY-CONSISTENT video decoder for DINOv3-base POOLED features (768D).

Unlike the per-frame `train_dinov3h_decoder.py` (each frame decoded independently ->
flickers when played as video), this decodes a *clip* of consecutive pooled features
with temporal context + a temporal-consistency loss, so a decoded episode plays as a
coherent VIDEO STREAM, not a slideshow of independent frames.

Design (iterable):
  1. Temporal context encoder: Conv1d over TIME on the pooled-feature sequence
     (kernel `tk`) -> each frame's decode is aware of its neighbours -> smooth.
  2. Per-frame image head: context(ctx) -> fc -> (512,4,4) -> 5x upsample -> 3xRxR.
  3. Losses over the clip:
       recon      = |D-I| + 0.5|D-I|^2
       temporal   = | (D_t - D_{t-1}) - (I_t - I_{t-1}) |   <- decoded motion == real motion
       gdl (opt)  = gradient-difference (spatial sharpness)
     The temporal term enforces coherence WITHOUT over-smoothing real motion (it matches
     the ground-truth inter-frame delta, not a zero-motion prior).

Pooled decoding is a soft "readable prototype" (pooled carries no spatial grid); the win
here is temporal coherence of that prototype across a video.

Usage:
    python lmwm/scripts/train_dinov3base_video_decoder.py \
        --feature_dir lmvla/crave/data/kai_dinov3base \
        --dataset_root kai0/data/Task_A/kai0_base \
        --clip 8 --n_clips 3000 --epochs 60 --tc_weight 1.0 \
        --out lmwm/checkpoints/dinov3base_decoder/kai_video_dec.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


def load_features(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    shards = sorted(s for s in feature_dir.glob("shard_*.npz") if "_bak" not in s.name)
    dim = int(np.load(shards[0])["feat"].shape[1])
    feat = np.zeros((n, dim), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in shards:
        z = np.load(shard)
        g = z["gidx"].astype(np.int64)
        feat[g] = z["feat"]
        valid[g] = z["valid"].astype(bool)
    return e[valid], fr[valid], feat[valid], dim


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class TemporalPooledDecoder(nn.Module):
    """clip of pooled features (B,T,din) -> clip of images (B,T,3,R,R), temporally aware."""

    def __init__(self, din: int = 768, res: int = 128, ctx: int = 512, tk: int = 5):
        super().__init__()
        pad = tk // 2
        self.tenc = nn.Sequential(
            nn.Conv1d(din, ctx, tk, padding=pad), nn.GELU(),
            nn.Conv1d(ctx, ctx, tk, padding=pad), nn.GELU(),
        )
        self.fc = nn.Sequential(nn.Linear(ctx, 512 * 4 * 4), nn.GELU())

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))

        self.net = nn.Sequential(up(512, 256), up(256, 128), up(128, 64), up(64, 32),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())  # 4->128

    def forward(self, feats):                                                    # (B,T,din)
        B, T, D = feats.shape
        h = self.tenc(feats.transpose(1, 2)).transpose(1, 2)                     # (B,T,ctx)
        img = self.net(self.fc(h.reshape(B * T, -1)).view(B * T, 512, 4, 4))     # (B*T,3,R,R)
        return img.view(B, T, 3, img.shape[-2], img.shape[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--clip", type=int, default=8, help="consecutive frames per training clip")
    ap.add_argument("--n_clips", type=int, default=3000)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--tk", type=int, default=5, help="temporal conv kernel (context window)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tc_weight", type=float, default=1.0, help="temporal-consistency loss weight")
    ap.add_argument("--gdl_weight", type=float, default=0.3, help="spatial gradient-difference (sharpness)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    E, FR, F, din = load_features(args.feature_dir)
    chunks_size = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    rng = np.random.default_rng(args.seed)
    T, R = args.clip, args.res

    # order frames by (episode, FR); build per-episode contiguous index ranges
    order = np.lexsort((FR, E))
    E, FR, F = E[order], FR[order], F[order]
    eps, starts = np.unique(E, return_index=True)
    ends = np.append(starts[1:], len(E))

    # sample clips = (episode, local_start) with T consecutive frames present
    clips = []
    for e, s, en in zip(eps, starts, ends):
        n_ep = en - s
        if n_ep < T:
            continue
        clips.append((e, s, en))
    n_clips = min(args.n_clips, sum((en - s - T + 1) for _e, s, en in clips))
    chosen = []
    for _ in range(n_clips):
        e, s, en = clips[rng.integers(len(clips))]
        st = rng.integers(s, en - T + 1)
        chosen.append((int(e), int(st)))

    # read real frames for chosen clips, grouped by episode (open each video once)
    Xf = np.zeros((n_clips, T, din), np.float32)
    Yi = np.zeros((n_clips, T, R, R, 3), np.uint8)
    by_ep: dict[int, list[tuple[int, int]]] = {}
    for ci, (e, st) in enumerate(chosen):
        by_ep.setdefault(e, []).append((ci, st))
    done = 0
    for e, items in by_ep.items():
        mp4 = args.dataset_root / f"videos/chunk-{e // chunks_size:03d}/{args.camera}/episode_{e:06d}.mp4"
        cap = cv2.VideoCapture(str(mp4))
        for ci, st in items:
            Xf[ci] = l2(F[st:st + T].astype(np.float32))
            for t in range(T):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(FR[st + t]))
                ok, fr = cap.read()
                if ok:
                    Yi[ci, t] = cv2.resize(fr[:, :, ::-1], (R, R))
        cap.release()
        done += len(items)
        if done % 500 < len(items):
            print(f"  read {done}/{n_clips} clips", flush=True)

    X = torch.from_numpy(Xf)
    Y = torch.from_numpy(Yi.astype(np.float32) / 127.5 - 1).permute(0, 1, 4, 2, 3).contiguous()  # (N,T,3,R,R)
    dec = TemporalPooledDecoder(din=din, res=R, ctx=args.ctx, tk=args.tk).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    n, bs = len(X), 16
    n_val = max(32, n // 10)
    perm0 = torch.randperm(n)
    val_i, tr_i = perm0[:n_val], perm0[n_val:]

    def losses(pred, y):
        recon = (pred - y).abs().mean() + 0.5 * ((pred - y) ** 2).mean()
        dD = pred[:, 1:] - pred[:, :-1]
        dI = y[:, 1:] - y[:, :-1]
        tc = (dD - dI).abs().mean()                                              # decoded motion == real motion
        gdl = ((pred[..., 1:] - pred[..., :-1]) - (y[..., 1:] - y[..., :-1])).abs().mean() \
            + ((pred[..., 1:, :] - pred[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])).abs().mean()
        return recon, tc, gdl

    for ep in range(args.epochs):
        dec.train()
        p = tr_i[torch.randperm(len(tr_i))]
        for b in range(0, len(p), bs):
            bi = p[b:b + bs]
            x, y = X[bi].to(device), Y[bi].to(device)
            pred = dec(x)
            recon, tc, gdl = losses(pred, y)
            loss = recon + args.tc_weight * tc + args.gdl_weight * gdl
            opt.zero_grad(); loss.backward(); opt.step()
        if ep == 0 or (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            dec.eval()
            with torch.no_grad():
                vp = dec(X[val_i].to(device)); vy = Y[val_i].to(device)
                vr, vtc, _ = losses(vp, vy)
            print(f"epoch {ep + 1}/{args.epochs}  val_recon={vr.item():.4f}  val_tc={vtc.item():.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dec.state_dict(), "res": R, "din": din, "ctx": args.ctx, "tk": args.tk,
                "meta": {"clip": T, "n_clips": n_clips, "epochs": args.epochs,
                         "val_recon": vr.item(), "val_tc": vtc.item(), "tc_weight": args.tc_weight,
                         "feature_dir": str(args.feature_dir),
                         "input": "l2-normalized pooled DINOv3-base 768D, temporal Conv1d context"}},
               args.out)
    print(f"saved {args.out}  val_recon={vr.item():.4f} val_tc={vtc.item():.4f}")


if __name__ == "__main__":
    main()
