#!/usr/bin/env python
"""Phase C — augment existing next-unique pairs with a short frame-history window.

Takes an existing pairs `.npz` and, for each row, replaces `current` (a single
DINOv3-H frame feature) with the concatenation of the current frame and the
previous ``H-1`` frames at a fixed stride, gathered from the raw per-frame
feature cache. All other fields (milestones, real future, episode_id, t) are
copied verbatim, so the episode split and the real-future targets are byte-for-byte
identical to the single-frame dataset -- an apples-to-apples Phase C comparison.

Frames earlier than available are padded by repeating the earliest frame.

Usage:
    python lmwm/scripts/export_history_pairs.py \
        --pairs lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique.npz \
        --feature_dir temp/crave_full_dinov3h \
        --history 4 --stride 2 \
        --out lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_hist4s2.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_full_features(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.load(feature_dir / "index.npz")
    e = idx["E"].astype(np.int64)
    fr = idx["FR"].astype(np.int64)
    n = int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        gidx = z["gidx"].astype(np.int64)
        feat[gidx] = z["feat"]
        valid[gidx] = z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32)
    fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], fv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--history", type=int, default=4, help="number of frames incl. current")
    ap.add_argument("--stride", type=int, default=2, help="frame stride between history taps")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    E, FR, F = load_full_features(args.feature_dir)
    D = F.shape[1]
    H, S = int(args.history), int(args.stride)

    # Per-episode: sorted-by-FR global row indices, and FR -> position map.
    ep_order: dict[int, np.ndarray] = {}
    ep_pos: dict[int, dict[int, int]] = {}
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]
        order = loc[np.argsort(FR[loc])]
        ep_order[int(ep)] = order
        ep_pos[int(ep)] = {int(fr): p for p, fr in enumerate(FR[order])}

    z = dict(np.load(args.pairs))
    eps = z["episode_id"].astype(np.int64)
    ts = z["t"].astype(np.int64)  # FR of the current frame
    n_rows = len(eps)

    hist = np.empty((n_rows, H * D), dtype=np.float16)
    missing = 0
    for i in range(n_rows):
        ep = int(eps[i])
        order = ep_order[ep]
        pos = ep_pos[ep].get(int(ts[i]))
        if pos is None:
            missing += 1
            pos = 0
        taps = []
        for h in range(H - 1, -1, -1):  # oldest -> current
            p = max(0, pos - h * S)
            taps.append(F[order[p]])
        hist[i] = np.concatenate(taps).astype(np.float16)
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1}/{n_rows}", flush=True)

    z["current"] = hist  # (n_rows, H*D)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **z)

    meta = {
        "source_pairs": str(args.pairs),
        "feature_dir": str(args.feature_dir),
        "history": H,
        "stride": S,
        "current_dim": int(H * D),
        "frame_dim": int(D),
        "num_pairs": int(n_rows),
        "rows_with_missing_frame_lookup": int(missing),
        "note": "current = concat of [t-(H-1)*S ... t-S, t] frame features; other fields copied from source pairs (identical episode split + real-future targets)",
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
