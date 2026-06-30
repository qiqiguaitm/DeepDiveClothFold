"""Sweep cluster count K (full CRAVE selection) → milestone count M and the coverage of the
selected milestones. Shows the milestone-number ↔ coverage tradeoff on FULL kai0_base @3Hz.

Run: CUDA_VISIBLE_DEVICES=0 PY crave/experiments/milestone_count_sweep.py
Out: temp/crave_full_dinov3h/milestone_count_sweep.json  (consumed by the combined compare fig)
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crave.config import REPO, resolve_dataset
from crave.utils import L2, otsu

ENC_DIR = REPO / "temp/crave_full_dinov3h"; DIM = 1280
K_LIST = [24, 48, 64, 96, 160, 256, 318]


def select_full(cov, tstd, tpos, K0):
    tau_cov = otsu(cov); vt = tstd[tstd < 9]; tau_pur = float(np.percentile(vt, 60)) if len(vt) else 9.0
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    gap = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= gap: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    return sel


def main():
    idx = np.load(ENC_DIR / "index.npz"); E, T, N = idx["E"], idx["T"], int(idx["n"])
    feat = np.zeros((N, DIM), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(ENC_DIR / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; F = L2(feat[vi].astype(np.float32)); Tv, Ev = T[vi], E[vi]; ne = len(set(E.tolist()))
    fit_idx = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)

    from sklearn.cluster import MiniBatchKMeans
    rows = []
    for K0 in K_LIST:
        km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit_idx]); lab = km.predict(F)
        tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
        cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
        tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
        sel = select_full(cov, tstd, tpos, K0); cs = cov[sel]
        rows.append({"K": K0, "M": len(sel), "cov_med": float(np.median(cs)), "cov_min": float(cs.min()),
                     "cov_max": float(cs.max()), "cov_mean": float(cs.mean()),
                     "allcov_med": float(np.median(cov))})
        print(f"K={K0:>3} -> M={len(sel):>2}  milestone-cov med={np.median(cs):.2f} min={cs.min():.2f} max={cs.max():.2f}", flush=True)
    json.dump(rows, open(ENC_DIR / "milestone_count_sweep.json", "w"), indent=2)
    print("SAVED milestone_count_sweep.json", flush=True)


if __name__ == "__main__":
    main()
