#!/usr/bin/env python
"""最终: img⊕pos 聚类 + 峰值(mode)多发射 Viterbi.
每簇用 robust-mode(峰,有谷检验)作为进度值; 双峰簇同时发射两个 bin, Viterbi 用路径选对.
"""
import glob, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.ndimage import gaussian_filter1d
from crave.config import resolve_dataset
from crave.data import kai0
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0")
MIN_COV = 0.50; LAM = 16.0; CSQ = 1000
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ---- load img⊕pos ----
d = np.load(REPO / "temp/crave_d3b_pca128/milestones.npz"); pca_m = d["pca_mean"]; pca_c = d["pca_components"]
FEAT = REPO / "temp/crave_d3b_pca128/feats"
eps = sorted(int(p.stem[2:]) for p in FEAT.glob("ep*.npy")); NC = len(eps)
cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E_idx, FR_idx = zf["E"], zf["FR"]
print(f"loading img⊕pos for {NC} eps...", flush=True)
imgF = []; T = []; Ev = []; ST = []; nlist = []
for e in eps:
    f = np.load(FEAT / f"ep{e}.npy").astype(np.float32); fq = l2((l2(f) - pca_m) @ pca_c.T); n = len(fq); nlist.append((e, n))
    imgF.append(fq); T.append(np.linspace(0, 1, n)); Ev.append(np.full(n, e))
    loc = np.where(E_idx == e)[0]; o = np.argsort(FR_idx[loc]); fr = FR_idx[loc][o]
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy())
    ST.append(st[np.minimum(fr[:n], len(st) - 1)])
imgF = np.concatenate(imgF); T = np.concatenate(T); Ev = np.concatenate(Ev); ST = np.concatenate(ST).astype(np.float32)
SMU, SSD = ST.mean(0), ST.std(0) + 1e-8
pos = l2((ST - SMU) / SSD)
jointF = np.concatenate([imgF, pos], 1)

# ---- K0 saturation ----
K0s = list(range(5, 50, 3)) + [60, 80]; ks = []
for K0 in K0s:
    km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=8192, max_iter=30).fit(l2(jointF)); lb = km.labels_
    ks.append(sum(1 for k in range(K0) if (lb == k).sum() >= 20 and len(set(Ev[lb == k].tolist())) / NC >= MIN_COV))
sat = K0s[ks.index(max(ks))]
print(f"SAT K0={sat} max_K={max(ks)}", flush=True)

# ---- final clustering ----
km = KMeans(sat, n_init=3, random_state=0).fit(l2(jointF)); labs = km.labels_
sel = []
for k in range(sat):
    mk = labs == k; nf = mk.sum()
    if nf < 20: continue
    cov = len(set(Ev[mk].tolist())) / NC
    if cov >= MIN_COV:
        sel.append({"k": k, "T": T[mk], "cov": cov, "C": km.cluster_centers_[k]})
M = len(sel)

def robust_modes(Tc, nbins=30, smooth=1.2, min_peak=0.10, merge_d=0.10, valley_ratio=0.6):
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1)); h = h.astype(float) / h.sum()
    hs = gaussian_filter1d(h, smooth); c = (ed[:-1] + ed[1:]) / 2
    peaks = [i for i in range(nbins) if hs[i] >= hs[max(0, i-1)] and hs[i] >= hs[min(nbins-1, i+1)] and hs[i] >= min_peak * hs.max()]
    merged = []
    for p in peaks:
        if merged and abs(c[p] - c[merged[-1]]) < merge_d:
            if hs[p] > hs[merged[-1]]: merged[-1] = p
        else: merged.append(p)
    final = [merged[0]] if merged else []
    for p in merged[1:]:
        valley = hs[final[-1]:p+1].min()
        if valley < valley_ratio * min(hs[final[-1]], hs[p]): final.append(p)
        elif hs[p] > hs[final[-1]]: final[-1] = p
    return [float(c[p]) for p in final]

# ---- build emission targets: (value, centroid) ----
targets = []  # list of (value, centroid, cluster_idx, is_multi)
for i, s in enumerate(sel):
    modes = robust_modes(s["T"])
    for mv in modes:
        targets.append((mv, s["C"], i, len(modes) > 1))
n_multi = sum(1 for s in sel if len(robust_modes(s["T"])) > 1)
print(f"{M} milestones, {n_multi} multi-modal, {len(targets)} emission targets", flush=True)

vals = np.array([t[0] for t in targets]); Ctgt = np.array([t[1] for t in targets], dtype=np.float32)
bins = np.unique(np.concatenate([[0.0], np.sort(vals), [1.0]])); nb = len(bins)
cbn = [int(np.searchsorted(bins, v)) for v in vals]
pen = LAM * np.abs(bins[:, None] - bins[None])

def vit(Fq):
    de = np.linalg.norm(Fq[:, None] - Ctgt[None], axis=2)  # (n, n_targets)
    em = np.full((len(Fq), nb), 1e3)
    for ti in range(len(vals)): em[:, cbn[ti]] = np.minimum(em[:, cbn[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(nb), k]; BP[j] = k
    cost[nb - 1] -= 2; s = int(cost.argmin()); path = np.zeros(len(Fq), int); path[-1] = s
    for j in range(len(Fq) - 2, -1, -1): s = BP[j + 1][s]; path[j] = s
    return bins[path]

# save spec
np.savez(REPO / "temp/crave_final_multimode.npz",
         vals=vals, Ctgt=Ctgt, sat_k0=sat, min_cov=MIN_COV, M=M,
         pca_mean=pca_m, pca_components=pca_c, SMU=SMU, SSD=SSD,
         cluster_idx=np.array([t[2] for t in targets]))
print("spec saved", flush=True)

# ---- labels ----
LAB = REPO / "temp/crave_ae_labels/final"; LAB.mkdir(parents=True, exist_ok=True)
t0 = time.time(); ptr = 0; n_done = 0; dd = []
for (e, n) in nlist:
    Fq = jointF[ptr:ptr + n]; ptr += n
    v3 = vit(Fq)
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy())
    n30 = len(st); v30 = np.interp(np.linspace(0, 1, n30), np.linspace(0, 1, n), v3)
    lo, hi = v30.min(), v30.max()
    if hi > lo + 1e-6: v30 = (v30 - lo) / (hi - lo)
    np.save(LAB / f"ep{e}.npy", v30.astype(np.float32)); n_done += 1
    dd.append(float((np.maximum.accumulate(v30) - v30).max()))
    if n_done % 1000 == 0: print(f"  [{n_done}/{len(nlist)}]", flush=True)
print(f"DONE {n_done} eps dd_mean={np.mean(dd):.3f} dd_max={max(dd):.3f} ({(time.time()-t0)/60:.1f}min)", flush=True)
