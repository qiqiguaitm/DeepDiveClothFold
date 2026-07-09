#!/usr/bin/env python
"""最终 AE 标签: DINOv3-base PCA→128D, τ=0.20(abs), Viterbi λ=16, no smooth.
输出 temp/crave_ae_labels/final/ + crave_stage_A/B 数据集.
"""
import glob, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

REPO = Path("/home/tim/workspace/deepdive_kai0")
SRC = REPO / "temp/crave_d3b_pca128"
LAB_DIR = REPO / "temp/crave_ae_labels/final"; LAB_DIR.mkdir(parents=True, exist_ok=True)
CSQ = 1000; TAU = 0.20; LAM = 16.0

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ---- Step 1: Load milestones + re-filter with τ=0.20 ----
print("loading milestones...", flush=True)
d = np.load(SRC / "milestones.npz")
C_all = d["C"]; Pord_all = d["Pord"]; covs_all = d["covs"]; tstds_all = d["tstds"]
tau_orig = d["tau"]; K0 = int(d["K0"])

# Re-filter
idx = np.where(covs_all >= TAU)[0]
covs_sel = covs_all[idx]; tstds_sel = tstds_all[idx]
# Re-sort by mean progress (recompute Pord)
# We need the original cluster positions from the overclustered km labels
# Since we have C (centroid), we can just reorder by the saved Pord
order = np.argsort(Pord_all[idx])
idx = idx[order]
Cs = C_all[idx]; M = len(idx)
# Re-isotonic
Pord = np.asarray(IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pord_all[idx]), dtype=np.float64)
print(f"τ={TAU}: {M} milestones (from {len(covs_all)} at τ={tau_orig:.3f})", flush=True)
print(f"  coverage: mean={covs_all[idx].mean():.3f} med={np.median(covs_all[idx]):.3f}")
print(f"  Tstd:     mean={tstds_all[idx].mean():.4f} med={np.median(tstds_all[idx]):.4f}")
print(f"  Pord:     [{Pord[0]:.3f}, {Pord[-1]:.3f}]", flush=True)

# ---- Step 2: Viterbi per-ep ----
bins = np.unique(np.concatenate([[0.0], Pord, [1.0]]))
nb = len(bins); cb = [int(np.searchsorted(bins, p)) for p in Pord]
pen = LAM * np.abs(bins[:, None] - bins[None])

# Load features and run Viterbi
FEAT_DIR = SRC / "feats"
from crave.config import resolve_dataset; from crave.data import kai0
cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)

# Load PCA
pca_mean = d["pca_mean"]; pca_components = d["pca_components"]

def vit_value(Fq_pca):
    d_em = np.linalg.norm(Fq_pca[:, None] - Cs[None], axis=2)
    em = np.full((len(Fq_pca), nb), 1e3)
    for ci in range(M): em[:, cb[ci]] = np.minimum(em[:, cb[ci]], d_em[:, ci])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]
    BP = np.zeros((len(Fq_pca), nb), int)
    for j in range(1, len(Fq_pca)):
        tr = cost[None, :] + pen; k = tr.argmin(1)
        cost = em[j] + tr[np.arange(nb), k]; BP[j] = k
    cost[nb - 1] -= 2
    s = int(cost.argmin()); path = np.zeros(len(Fq_pca), int); path[-1] = s
    for j in range(len(Fq_pca) - 2, -1, -1): s = BP[j + 1][s]; path[j] = s
    return bins[path]

all_eps = sorted(int(p.stem[2:]) for p in FEAT_DIR.glob("ep*.npy"))
print(f"Viterbi per-ep (λ={LAM}, {len(all_eps)} eps)...", flush=True)
n_done = 0; n_err = 0; t0 = time.time()
dd_all = []
for e in all_eps:
    try:
        fp_out = LAB_DIR / f"ep{e}.npy"
        if fp_out.exists(): n_done += 1; continue
        # Load raw features
        feat_raw = np.load(FEAT_DIR / f"ep{e}.npy").astype(np.float32)
        # PCA transform
        feat_pca = l2((l2(feat_raw) - pca_mean) @ pca_components.T)
        # Viterbi
        v3 = vit_value(feat_pca)
        # Interpolate to 30Hz
        st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                       columns=["observation.state"])["observation.state"].to_numpy())
        n30 = len(st)
        xi = np.linspace(0, 1, len(v3)); xo = np.linspace(0, 1, n30)
        v30 = np.interp(xo, xi, v3)
        # per-ep min-max norm01
        lo, hi = float(v30.min()), float(v30.max())
        if hi > lo + 1e-6: v30 = (v30 - lo) / (hi - lo)
        np.save(fp_out, v30.astype(np.float32)); n_done += 1
        dd_all.append(float((np.maximum.accumulate(v30) - v30).max()))
        if n_done % 500 == 0:
            el = time.time() - t0; eta = el / n_done * (len(all_eps) - n_done) / 60
            print(f"  [{n_done}/{len(all_eps)}] ep{e} ({el/60:.0f}min, ~{eta:.0f}min left)", flush=True)
    except Exception as ex:
        n_err += 1
        if n_err <= 5: print(f"  ERR ep{e}: {ex}", flush=True)

print(f"DONE {n_done} eps, {n_err} errors in {(time.time()-t0)/60:.1f}min", flush=True)
print(f"max drawdown: mean={np.mean(dd_all):.3f} max={max(dd_all):.3f}", flush=True)
print(f"Labels: {LAB_DIR}", flush=True)
