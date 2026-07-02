#!/usr/bin/env python
"""Train a DINOv3-H pooled-feature -> image decoder.

CRAVE's existing decoder consumes DINOv2-large patch grids (16x16), but LMWM lives
in DINOv3-H *pooled* 1280D space (that is what its greedy_proto subgoal head
predicts). No decoder exists for that space, so this trains one:

    l2-normalized pooled DINOv3-H (1280) -> fc -> (512,4,4) -> 5x upsample -> 3x128x128

Trained on (cached pooled feature, real frame) pairs sampled across episodes.
Pooled decoding is intentionally a smooth "readable prototype" (CRAVE showed pooled
decodes are soft); pair it with medoid retrieval for sharp exemplars.

Usage:
    python lmwm/scripts/train_dinov3h_decoder.py \
        --feature_dir temp/crave_full_dinov3h \
        --dataset_root kai0/data/Task_A/kai0_base \
        --n_pairs 16000 --epochs 60 --out lmwm/checkpoints/dinov3h_decoder/dec.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


def load_features(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.load(feature_dir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        g = z["gidx"].astype(np.int64)
        feat[g] = z["feat"]
        valid[g] = z["valid"].astype(bool)
    return e[valid], fr[valid], feat[valid]


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class PooledDecoder(nn.Module):
    """1280-D pooled feature -> 3x128x128 image."""

    def __init__(self, din: int = 1280, res: int = 128) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(din, 512 * 4 * 4), nn.GELU())

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))

        self.net = nn.Sequential(up(512, 256), up(256, 128), up(128, 64), up(64, 32),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())  # 4->8->16->32->64->128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(x).view(-1, 512, 4, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_pairs", type=int, default=16000)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    E, FR, F = load_features(args.feature_dir)
    chunks_size = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    rng = np.random.default_rng(args.seed)

    # Sample pairs, grouped by episode so each video is opened once.
    n_pairs = min(args.n_pairs, len(F))
    sel = rng.choice(len(F), n_pairs, replace=False)
    by_ep: dict[int, list[int]] = {}
    for i in sel:
        by_ep.setdefault(int(E[i]), []).append(int(i))

    R = args.res
    imgs = np.zeros((n_pairs, R, R, 3), dtype=np.uint8)
    feats = l2(F[sel].astype(np.float32))
    pos = {int(i): k for k, i in enumerate(sel)}
    done = 0
    for ep, idxs in by_ep.items():
        mp4 = args.dataset_root / f"videos/chunk-{ep // chunks_size:03d}/{args.camera}/episode_{ep:06d}.mp4"
        cap = cv2.VideoCapture(str(mp4))
        for gi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(FR[gi]))
            ok, fr = cap.read()
            if ok:
                imgs[pos[gi]] = cv2.resize(fr[:, :, ::-1], (R, R))
            done += 1
        cap.release()
        if done % 4000 < len(idxs):
            print(f"  read {done}/{n_pairs}", flush=True)

    # Train.
    Y = torch.from_numpy(imgs.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous()
    X = torch.from_numpy(feats)
    dec = PooledDecoder(din=X.shape[1], res=R).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    n, bs = len(X), 128
    n_val = max(256, n // 10)
    perm0 = torch.randperm(n)
    val_i, tr_i = perm0[:n_val], perm0[n_val:]
    for ep in range(args.epochs):
        dec.train()
        p = tr_i[torch.randperm(len(tr_i))]
        for b in range(0, len(p), bs):
            bi = p[b:b + bs]
            x, y = X[bi].to(device), Y[bi].to(device)
            pred = dec(x)
            loss = (pred - y).abs().mean() + 0.5 * ((pred - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep == 0 or (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            dec.eval()
            with torch.no_grad():
                vp = dec(X[val_i].to(device))
                vl = (vp - Y[val_i].to(device)).abs().mean().item()
            print(f"epoch {ep + 1}/{args.epochs}  val_L1={vl:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dec.state_dict(), "res": R, "din": int(X.shape[1]),
                "meta": {"n_pairs": n_pairs, "epochs": args.epochs, "val_L1": vl,
                         "feature_dir": str(args.feature_dir), "input": "l2-normalized pooled DINOv3-H 1280D"}},
               args.out)
    print(f"saved {args.out}  val_L1={vl:.4f}")


if __name__ == "__main__":
    main()
