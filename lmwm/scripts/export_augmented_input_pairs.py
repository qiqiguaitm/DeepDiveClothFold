#!/usr/bin/env python
"""Augment the LMWM input: current = [frame(1280); prev-milestone one-hot(M+1); state(14, z-scored)].

Row-aligned to the medoid pairs (keeps next_medoid, future_milestone, split), so
the trainer runs unchanged with a larger in_dim. Proven to lift real-next-milestone
top1 0.382 -> 0.434 (frame -> frame+path+state) in probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_full(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16); valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard); gi = z["gidx"].astype(np.int64)
        feat[gi] = z["feat"]; valid[gi] = z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32); fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], fv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path, help="medoid pairs (episode_id, t, current, next_medoid)")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    proto = np.load(args.graph_npz)["prototype_table"].astype(np.float32); num_m = len(proto)
    E, FR, F_ = load_full(args.feature_dir)
    assign = np.empty(len(F_), dtype=np.int64)
    for i in range(0, len(F_), 32768):
        assign[i:i + 32768] = (F_[i:i + 32768] @ proto.T).argmax(1)
    START = num_m
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])

    # per episode: FR -> stage index, stage milestones (for prev-milestone lookup)
    ep_fr_stage: dict[int, dict[int, int]] = {}
    ep_stage_m: dict[int, list[int]] = {}
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]; seq = assign[order]
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        fr2s = {}
        for si, (s, e) in enumerate(zip(st, en)):
            for p in range(s, e):
                fr2s[int(FR[order[p]])] = si
        ep_fr_stage[int(ep)] = fr2s
        ep_stage_m[int(ep)] = [int(seq[s]) for s in st]

    z = dict(np.load(args.pairs))
    eps = z["episode_id"].astype(np.int64); ts = z["t"].astype(np.int64); frame = z["current"].astype(np.float32)
    n = len(eps)

    prev_oh = np.zeros((n, num_m + 1), np.float32)
    state = np.zeros((n, 14), np.float32)
    # state join per episode
    for ep in np.unique(eps):
        pq = args.dataset_root / f"data/chunk-{ep // cs:03d}/episode_{ep:06d}.parquet"
        df = pd.read_parquet(pq, columns=["observation.state"])
        arr = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        m = np.where(eps == ep)[0]
        fi = np.clip(ts[m], 0, len(arr) - 1)
        state[m] = arr[fi]
        stages = ep_stage_m[int(ep)]; fr2s = ep_fr_stage[int(ep)]
        for i in m:
            si = fr2s.get(int(ts[i]))
            prev_oh[i, (stages[si - 1] if (si is not None and si >= 1) else START)] = 1.0
    state = (state - state.mean(0)) / (state.std(0) + 1e-6)

    aug = np.concatenate([frame, prev_oh, state.astype(np.float32)], axis=1)  # 1280 + (M+1) + 14
    z["current"] = aug.astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **z)
    meta = {"source_pairs": str(args.pairs), "num_rows": int(n), "in_dim": int(aug.shape[1]),
            "layout": "[frame:1280][prev_milestone_onehot:%d][state_zscore:14]" % (num_m + 1)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__":
    main()
