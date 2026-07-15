#!/usr/bin/env python
"""Validate a new-method kai0 milestone spec vs supervised stage_progress_gt.

Reports:
  - step readout corr : value = Pord[nearest milestone]  (crude, no smoothing)
  - double-anchor Viterbi corr : milestone emissions + start/end anchors, forced 0->1
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/tim/workspace/deepdive_kai0")
LAM = 16.0


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def load_feats(fdir):
    idx = np.load(fdir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    feat = np.zeros((n, 1280), np.float16); valid = np.zeros(n, bool)
    for sh in sorted(fdir.glob("shard_*.npz")):
        if "_bak" in sh.name:
            continue
        z = np.load(sh); g = z["gidx"].astype(np.int64); feat[g] = z["feat"]; valid[g] = z["valid"].astype(bool)
    return e[valid], fr[valid], l2(feat[valid].astype(np.float32))


def viterbi_anchored(Fq, Ctgt, vals):
    """Double-anchor Viterbi over milestone emissions in 1280D. vals ascending in [0,1]."""
    bins = np.unique(np.concatenate([[0.0], vals, [1.0]])); nb = len(bins)
    cbn = [int(np.searchsorted(bins, v)) for v in vals]
    pen = LAM * np.abs(bins[:, None] - bins[None])
    de = np.linalg.norm(Fq[:, None] - Ctgt[None], axis=2)
    em = np.full((len(Fq), nb), 1e3)
    for ti in range(len(vals)):
        em[:, cbn[ti]] = np.minimum(em[:, cbn[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    cost[nb - 1] -= 2; s = int(cost.argmin()); path = np.zeros(len(Fq), int); path[-1] = s
    for j in range(len(Fq) - 2, -1, -1):
        s = BP[j + 1][s]; path[j] = s
    return bins[path]


def main():
    fdir = REPO / "temp/crave_full_dinov3h"
    spec = np.load(REPO / "temp/newcrave_specs/kai0_milestones_newmethod.npz")
    C = l2(spec["C"].astype(np.float32)); Pord = spec["Pord"].astype(np.float32)
    E, FR, F = load_feats(fdir)

    # GT per frame from kai0_advantage
    gt_by_ep = {}
    for f in sorted(glob.glob(str(REPO / "kai0/data/Task_A/kai0_advantage/data/**/*.parquet"), recursive=True)):
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "stage_progress_gt"])
        ep = int(df["episode_index"].iloc[0])
        gt_by_ep[ep] = df.sort_values("frame_index")["stage_progress_gt"].to_numpy().astype(np.float32)

    step_vals, vit_vals, gts = [], [], []
    per_ep_step, per_ep_vit = [], []
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        Fq = F[order]; fr = FR[order]
        assign = (Fq @ C.T).argmax(1)
        sv = Pord[assign]
        vv = viterbi_anchored(Fq, C, Pord)
        g = gt_by_ep.get(int(ep))
        if g is None:
            continue
        gg = g[np.minimum(fr, len(g) - 1)]
        step_vals.append(sv); vit_vals.append(vv); gts.append(gg)
        if np.std(gg) > 1e-6:
            per_ep_step.append(np.corrcoef(sv, gg)[0, 1])
            per_ep_vit.append(np.corrcoef(vv, gg)[0, 1])
    SV = np.concatenate(step_vals); VV = np.concatenate(vit_vals); GT = np.concatenate(gts)
    monos = [np.mean(np.diff(v) >= -1e-6) for v in vit_vals]
    print(f"kai0 new-method spec: M={len(Pord)} milestones, {len(np.unique(E))} eps")
    print(f"  GLOBAL corr(step,GT) = {np.corrcoef(SV, GT)[0,1]:.3f}")
    print(f"  GLOBAL corr(viterbi,GT) = {np.corrcoef(VV, GT)[0,1]:.3f}")
    print(f"  per-ep mean corr(step) = {np.nanmean(per_ep_step):.3f}")
    print(f"  per-ep mean corr(viterbi) = {np.nanmean(per_ep_vit):.3f}   (doc double-anchor target 0.943)")
    print(f"  viterbi monotone = {np.mean(monos):.3f}")
    print(f"  Pord range [{Pord.min():.2f},{Pord.max():.2f}], inversions={int((np.diff(Pord)<0).sum())}")


if __name__ == "__main__":
    main()
