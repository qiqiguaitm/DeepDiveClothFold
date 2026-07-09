#!/usr/bin/env python
"""当前配置(img⊕pos, PCA128)下重跑底层过聚类器对比:
KMeans / GMM(diag) / HDBSCAN / BayesianGMM / MiniBatchKMeans.
统一 overcluster+coverage(min_cov=0.5) 框架, 比较 milestone T-std + coverage + K_eff + 别名.
"""
import glob, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans, HDBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.ndimage import gaussian_filter1d
from crave.config import resolve_dataset
from crave.data import kai0
REPO = Path("/home/tim/workspace/deepdive_kai0"); MIN_COV = 0.50; K0 = 32
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ---- load img⊕pos (subset 1500 eps for speed on heavy methods) ----
d = np.load(REPO / "temp/crave_d3b_pca128/milestones.npz"); pca_m = d["pca_mean"]; pca_c = d["pca_components"]
FEAT = REPO / "temp/crave_d3b_pca128/feats"
rng = np.random.RandomState(42)
alleps = sorted(int(p.stem[2:]) for p in FEAT.glob("ep*.npy"))
eps = sorted(rng.choice(alleps, 1500, replace=False)); NC = len(eps)
cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E_idx, FR_idx = zf["E"], zf["FR"]
imgF = []; T = []; Ev = []; ST = []
print(f"loading img⊕pos for {NC} eps...", flush=True)
for e in eps:
    f = np.load(FEAT / f"ep{e}.npy").astype(np.float32); fq = l2((l2(f) - pca_m) @ pca_c.T); n = len(fq)
    imgF.append(fq); T.append(np.linspace(0, 1, n)); Ev.append(np.full(n, e))
    loc = np.where(E_idx == e)[0]; o = np.argsort(FR_idx[loc]); fr = FR_idx[loc][o]
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy())
    ST.append(st[np.minimum(fr[:n], len(st) - 1)])
imgF = np.concatenate(imgF); T = np.concatenate(T); Ev = np.concatenate(Ev); ST = np.concatenate(ST).astype(np.float32)
pos = l2((ST - ST.mean(0)) / (ST.std(0) + 1e-8))
F = l2(np.concatenate([imgF, pos], 1))
print(f"joint {F.shape}", flush=True)

def robust_modes(Tc, nbins=30):
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1)); h = h.astype(float) / h.sum()
    hs = gaussian_filter1d(h, 1.2); c = (ed[:-1] + ed[1:]) / 2
    peaks = [i for i in range(nbins) if hs[i] >= hs[max(0, i-1)] and hs[i] >= hs[min(nbins-1, i+1)] and hs[i] >= 0.10 * hs.max()]
    merged = []
    for p in peaks:
        if merged and abs(c[p] - c[merged[-1]]) < 0.10:
            if hs[p] > hs[merged[-1]]: merged[-1] = p
        else: merged.append(p)
    final = [merged[0]] if merged else []
    for p in merged[1:]:
        valley = hs[final[-1]:p+1].min()
        if valley < 0.6 * min(hs[final[-1]], hs[p]): final.append(p)
        elif hs[p] > hs[final[-1]]: final[-1] = p
    return len(final)

def evaluate(labs, name, K_found):
    sel = []
    for k in range(K_found):
        mk = labs == k; nf = mk.sum()
        if nf < 20: continue
        cov = len(set(Ev[mk].tolist())) / NC
        if cov >= MIN_COV:
            sel.append({"T": T[mk], "cov": cov})
    if not sel:
        print(f"{name:>22s}: NO milestones pass coverage>={MIN_COV}", flush=True); return
    tstds = [s["T"].std() for s in sel]; covs = [s["cov"] for s in sel]
    n_multi = sum(1 for s in sel if robust_modes(s["T"]) > 1)
    print(f"{name:>22s}: K_eff={len(sel):2d} Tstd={np.mean(tstds):.4f} cov={np.mean(covs):.3f} 多峰簇={n_multi}", flush=True)

print(f"\n=== 底层过聚类器对比 (img⊕pos, K0={K0}, min_cov={MIN_COV}) ===", flush=True)
# 1. KMeans
t0 = time.time(); km = KMeans(K0, n_init=3, random_state=0).fit(F)
evaluate(km.labels_, "KMeans", K0); print(f"   ({time.time()-t0:.0f}s)", flush=True)
# 2. MiniBatchKMeans
t0 = time.time(); mbk = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=8192, max_iter=50).fit(F)
evaluate(mbk.labels_, "MiniBatchKMeans", K0); print(f"   ({time.time()-t0:.0f}s)", flush=True)
# 3. GMM diag
t0 = time.time(); gmm = GaussianMixture(K0, covariance_type="diag", n_init=1, reg_covar=1e-4, max_iter=80, random_state=0).fit(F)
evaluate(gmm.predict(F), "GMM(diag)", K0); print(f"   ({time.time()-t0:.0f}s)", flush=True)
# 4. BayesianGMM (adaptive K)
t0 = time.time(); bgmm = BayesianGaussianMixture(n_components=K0, covariance_type="diag", weight_concentration_prior=1e-2,
                                                 n_init=1, max_iter=120, random_state=0).fit(F)
eff = int((bgmm.weights_ > 0.01).sum())
evaluate(bgmm.predict(F), f"BayesianGMM(eff{eff})", K0); print(f"   ({time.time()-t0:.0f}s)", flush=True)
# 5. HDBSCAN (adaptive)
for mcs in [500, 1000]:
    t0 = time.time(); hdb = HDBSCAN(min_cluster_size=mcs, min_samples=20, metric="euclidean").fit(F)
    kf = hdb.labels_.max() + 1; noise = (hdb.labels_ == -1).mean()
    if kf < 1: print(f"{'HDBSCAN mcs='+str(mcs):>22s}: collapsed (K={kf}, noise={noise:.0%})", flush=True); continue
    evaluate(hdb.labels_, f"HDBSCAN mcs={mcs}", kf); print(f"   noise={noise:.0%} ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
