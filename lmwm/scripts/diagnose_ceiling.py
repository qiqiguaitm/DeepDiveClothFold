#!/usr/bin/env python
"""Is LMWM's next-milestone ceiling intrinsic, or label-jitter / aliasing?

Bias-free count-based diagnostic (no neural training): estimate P(next|context)
on train episodes with add-alpha smoothing, evaluate NLL/top1 of the real next
milestone on held-out episodes. Compares:
  - RAW vs temporally-SMOOTHED milestone labels  (tests H1: label jitter)
  - context = current-milestone  vs  current-milestone + time-bin  (tests H2:
    cluster-revisit aliasing that absolute trajectory position could resolve)

If smoothing and/or time-conditioning materially lower held-out NLL, the ceiling
is NOT intrinsic task ambiguity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_full(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, t, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), idx["T"].astype(np.float64), int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        g = z["gidx"].astype(np.int64)
        feat[g] = z["feat"]; valid[g] = z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32)
    fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], t[valid], fv


def mode_filter(seq: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return seq
    h = w // 2
    out = seq.copy()
    for i in range(len(seq)):
        lo, hi = max(0, i - h), min(len(seq), i + h + 1)
        vals, cnts = np.unique(seq[lo:hi], return_counts=True)
        out[i] = vals[cnts.argmax()]
    return out


def next_unique_pairs(seq: np.ndarray, tfrac: np.ndarray):
    """Return (cur, next, tfrac_at_cur) for each position that has a later change."""
    out = []
    nxt = -1
    for i in range(len(seq) - 2, -1, -1):
        if seq[i + 1] != seq[i]:
            nxt = seq[i + 1]
        if nxt != -1:
            out.append((int(seq[i]), int(nxt), float(tfrac[i])))
    return out


def eval_estimator(train, val, num_m, n_tbins, use_time, alpha=0.5):
    """train/val: lists of (cur, nxt, tfrac). Fit P(nxt|context) on train, eval val NLL/top1."""
    def ctx(cur, tf):
        return (cur, min(n_tbins - 1, int(tf * n_tbins))) if use_time else cur
    tables = defaultdict(Counter)
    for cur, nxt, tf in train:
        tables[ctx(cur, tf)][nxt] += 1
    nll, hit, tot = 0.0, 0, 0
    logM = np.log(num_m)
    for cur, nxt, tf in val:
        c = tables.get(ctx(cur, tf))
        if not c:
            nll += logM  # unseen context -> uniform
            tot += 1
            continue
        total = sum(c.values()) + alpha * num_m
        p = (c.get(nxt, 0) + alpha) / total
        nll += -np.log(p)
        if c.most_common(1)[0][0] == nxt:
            hit += 1
        tot += 1
    return {"nll": nll / tot, "top1": hit / tot, "n": tot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", default="lmwm/outputs/ceiling_diag/summary.json", type=Path)
    ap.add_argument("--n_tbins", type=int, default=6)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    g = np.load(args.graph_npz)
    proto = g["prototype_table"].astype(np.float32)
    num_m = len(proto)
    E, FR, T, F = load_full(args.feature_dir)
    assign = np.empty(len(F), dtype=np.int64)
    for i in range(0, len(F), 32768):
        assign[i:i + 32768] = (F[i:i + 32768] @ proto.T).argmax(1)

    eps = np.unique(E)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(eps)
    n_val = max(1, int(round(len(eps) * args.val_ratio)))
    val_eps = set(eps[:n_val].tolist())

    # Build per-episode sequences; compute raw + smoothed next-unique pairs.
    results = {}
    for w in (1, 3, 5, 9):
        train_pairs, val_pairs, comp_lens = [], [], []
        for ep in eps:
            loc = np.where(E == ep)[0]
            order = loc[np.argsort(FR[loc])]
            seq = assign[order]
            if len(seq) < 3:
                continue
            tnorm = T[order]
            tf = (tnorm - tnorm.min()) / (tnorm.max() - tnorm.min() + 1e-9)
            sm = mode_filter(seq, w)
            comp = int((np.diff(sm) != 0).sum() + 1)
            comp_lens.append(comp)
            pairs = next_unique_pairs(sm, tf)
            (val_pairs if ep in val_eps else train_pairs).extend(pairs)
        cond_cur = eval_estimator(train_pairs, val_pairs, num_m, args.n_tbins, use_time=False)
        cond_cur_t = eval_estimator(train_pairs, val_pairs, num_m, args.n_tbins, use_time=True)
        results[f"smooth_w{w}"] = {
            "mean_compressed_len": float(np.mean(comp_lens)),
            "context=current_milestone": cond_cur,
            "context=current_milestone+time_bin": cond_cur_t,
        }
        print(f"[w={w}] comp_len={np.mean(comp_lens):.1f}  "
              f"cur: nll={cond_cur['nll']:.3f} top1={cond_cur['top1']:.3f}  "
              f"cur+time: nll={cond_cur_t['nll']:.3f} top1={cond_cur_t['top1']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"num_milestones": num_m, "results": results}, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
