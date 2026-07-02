#!/usr/bin/env python
"""Non-parametric ceiling probe: how well can the DINOv3-H frame feature predict
the real next milestone, independent of our MLP?

For each held-out frame (with a next-unique milestone), retrieve its k nearest
train frames by cosine and soft-vote their next-milestone distribution. This is a
near-optimal estimator for the given representation+labels, so it upper-bounds
what any model on these features can achieve. If kNN >> our neural 0.383 top1,
the MLP is under-fitting (headroom exists); if kNN ~= 0.383, the representation /
labels are the limit (need a reframe, not more tuning).

Optionally append normalized episode time to the feature (tests whether absolute
trajectory position helps the continuous predictor too).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_full(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, t, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), idx["T"].astype(np.float64), int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        gi = z["gidx"].astype(np.int64)
        feat[gi] = z["feat"]; valid[gi] = z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32)
    fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], t[valid], fv


def next_unique_targets(seq: np.ndarray) -> np.ndarray:
    nu = np.full(len(seq), -1, dtype=np.int64)
    nxt = -1
    for i in range(len(seq) - 2, -1, -1):
        if seq[i + 1] != seq[i]:
            nxt = seq[i + 1]
        nu[i] = nxt
    return nu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", default="lmwm/outputs/ceiling_diag/knn_summary.json", type=Path)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    g = np.load(args.graph_npz)
    proto = g["prototype_table"].astype(np.float32)
    num_m = len(proto)
    E, FR, T, F = load_full(args.feature_dir)
    assign = np.empty(len(F), dtype=np.int64)
    for i in range(0, len(F), 32768):
        assign[i:i + 32768] = (F[i:i + 32768] @ proto.T).argmax(1)

    # Per-episode next-unique target + normalized time, keep only frames with a target.
    gi_all, tgt_all, tfrac_all = [], [], []
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]
        order = loc[np.argsort(FR[loc])]
        nu = next_unique_targets(assign[order])
        tn = T[order]; tf = (tn - tn.min()) / (tn.max() - tn.min() + 1e-9)
        keep = nu >= 0
        gi_all.append(order[keep]); tgt_all.append(nu[keep]); tfrac_all.append(tf[keep])
    gi = np.concatenate(gi_all); tgt = np.concatenate(tgt_all).astype(np.int64); tfrac = np.concatenate(tfrac_all)

    rng = np.random.default_rng(args.seed)
    eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * args.val_ratio)))].tolist())
    is_val = np.array([E[i] in val_eps for i in gi])

    summary = {"num_milestones": num_m, "neural_ref": {"top1": 0.383, "nll": 1.98}, "variants": {}}
    for use_time in (False, True):
        base = F[gi]
        if use_time:
            base = np.concatenate([base, (tfrac[:, None] * 0.5).astype(np.float32)], axis=1)  # small time channel
            base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-8)
        keys = torch.from_numpy(base[~is_val]).to(device)
        key_tgt = torch.from_numpy(tgt[~is_val]).to(device)
        q = torch.from_numpy(base[is_val]).to(device)
        q_tgt = tgt[is_val]
        for k in (1, 10, 50):
            top1 = 0; nll = 0.0; tot = 0
            for s in range(0, len(q), 1024):
                qb = q[s:s + 1024]
                sim = qb @ keys.T
                _, nn_idx = sim.topk(k, dim=1)
                nn_lbl = key_tgt[nn_idx]  # (b,k)
                # soft vote -> distribution
                votes = torch.zeros(len(qb), num_m, device=device)
                votes.scatter_add_(1, nn_lbl, torch.ones_like(nn_lbl, dtype=torch.float))
                p = (votes + 0.1) / (votes.sum(1, keepdim=True) + 0.1 * num_m)
                yb = torch.from_numpy(q_tgt[s:s + 1024]).to(device)
                top1 += int((p.argmax(1) == yb).sum())
                nll += float(-torch.log(p[torch.arange(len(qb)), yb]).sum())
                tot += len(qb)
            key = f"knn_k{k}" + ("_time" if use_time else "")
            summary["variants"][key] = {"top1": top1 / tot, "nll": nll / tot, "n": tot}
            print(f"{key}: top1={top1/tot:.4f} nll={nll/tot:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
