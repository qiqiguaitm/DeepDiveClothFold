#!/usr/bin/env python
"""Augment next-unique pairs with the NEXT-STAGE episode-local medoid latent.

For each existing pair (episode, current frame at FR=t), find the stage containing
t, take the next stage, and store that stage's medoid latent (the real frame in
that stage closest to its milestone centroid). This is the continuous,
episode-real target for the proto/subgoal head (vs the fixed global centroid).

Row-aligned to the source pairs so the episode split and real-future milestone
targets are byte-identical -> clean A/B against the centroid-target model.

Usage:
    python lmwm/scripts/export_episode_medoid_pairs.py \
        --pairs lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique.npz \
        --feature_dir temp/crave_full_dinov3h \
        --graph_npz lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz \
        --out lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_medoid.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_full(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        gi = z["gidx"].astype(np.int64)
        feat[gi] = z["feat"]; valid[gi] = z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32)
    fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], fv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--graph_npz", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    proto = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    E, FR, F = load_full(args.feature_dir)
    assign = np.empty(len(F), dtype=np.int64)
    for i in range(0, len(F), 32768):
        assign[i:i + 32768] = (F[i:i + 32768] @ proto.T).argmax(1)

    # Per episode: stage runs; per stage medoid latent + milestone; FR -> stage index.
    ep_stage_med: dict[int, np.ndarray] = {}
    ep_stage_m: dict[int, np.ndarray] = {}
    ep_fr_to_stage: dict[int, dict[int, int]] = {}
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]
        order = loc[np.argsort(FR[loc])]
        seq = assign[order]
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        meds, ms, fr2s = [], [], {}
        for si, (s, e) in enumerate(zip(st, en)):
            m = int(seq[s]); sub = F[order[s:e]]
            meds.append(sub[(sub @ proto[m]).argmax()]); ms.append(m)
            for p in range(s, e):
                fr2s[int(FR[order[p]])] = si
        ep_stage_med[int(ep)] = np.stack(meds)
        ep_stage_m[int(ep)] = np.array(ms, dtype=np.int64)
        ep_fr_to_stage[int(ep)] = fr2s

    z = dict(np.load(args.pairs))
    eps = z["episode_id"].astype(np.int64)
    ts = z["t"].astype(np.int64)
    fut_m = z["future_milestone"].astype(np.int64)
    n_rows = len(eps)
    D = F.shape[1]
    next_med = np.zeros((n_rows, D), dtype=np.float16)
    miss, mism = 0, 0
    for i in range(n_rows):
        ep = int(eps[i]); si = ep_fr_to_stage[ep].get(int(ts[i]))
        stages_m = ep_stage_m[ep]
        if si is None or si + 1 >= len(stages_m):
            miss += 1
            continue
        next_med[i] = ep_stage_med[ep][si + 1]
        if int(stages_m[si + 1]) != int(fut_m[i]):
            mism += 1  # sanity: next-stage milestone should equal next-unique target

    z["next_medoid"] = next_med
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **z)
    meta = {
        "source_pairs": str(args.pairs),
        "num_rows": n_rows,
        "rows_missing_next_stage": int(miss),
        "next_stage_milestone_mismatch_vs_future_milestone": int(mism),
        "note": "next_medoid = next-stage episode-local medoid latent (real frame closest to its centroid)",
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
