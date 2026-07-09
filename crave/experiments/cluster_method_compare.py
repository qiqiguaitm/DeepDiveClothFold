#!/usr/bin/env python
"""聚类方法对比:KMeans vs GMM vs WeightedKMeans vs HDBSCAN vs BayesianGMM。
分析 coverage-density 关联 + 时间纯度 + 自适应K。
"""
import sys, glob, time, numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans, HDBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0")
K = 48; NEPS = 200; rng = np.random.RandomState(42)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def load(idx_path, shard_dir, dim):
    zf = np.load(idx_path); E, FR, T = zf["E"], zf["FR"], zf["T"]; N = int(zf["n"])
    feat = np.zeros((N, dim), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(shard_dir / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    return E, FR, T, feat, valid

E, FR, T, d2, v2 = load(REPO / "temp/crave_full/index_dino.npz", REPO / "temp/crave_full/dino", 1024)
vi = np.where(v2)[0]; Ev, Tv = E[vi], T[vi]; Fv = l2(d2[vi].astype(np.float32))
samp = sorted(rng.choice(sorted(set(Ev.tolist())), NEPS, replace=False))
m = np.isin(Ev, samp); Fvs, Tvs, Evs = Fv[m], Tv[m], Ev[m]
print(f"{len(Fvs)} frames, {NEPS} eps", flush=True)

def cluster_metrics(labs, K_found):
    rows = []
    for k in range(K_found):
        mk = (labs == k)
        if mk.sum() < 5: continue
        cov = len(set(Evs[mk].tolist())) / NEPS
        c = Fvs[mk].mean(0); compact = float(np.linalg.norm(Fvs[mk] - c, axis=1).mean())
        t_std = float(np.nanstd(Tvs[mk]))
        rows.append((k, cov, compact, t_std, mk.sum()))
    if not rows: return None
    covs = np.array([r[1] for r in rows]); comps = np.array([r[2] for r in rows])
    tstds = np.array([r[3] for r in rows]); sizes = np.array([r[4] for r in rows])
    return {"covs": covs, "comps": comps, "tstds": tstds, "sizes": sizes,
            "mean_tstd": np.mean(tstds), "median_tstd": np.median(tstds),
            "cov_compact_corr": np.corrcoef(covs, comps)[0, 1] if len(covs) > 2 else 0,
            "K_found": K_found}

results = {}

# 1. KMeans
t0 = time.time()
km = KMeans(K, n_init=3, random_state=0).fit(Fvs)
results["KMeans K=48"] = cluster_metrics(km.labels_, K)
print(f"KMeans: K={K} mean_Tstd={results['KMeans K=48']['mean_tstd']:.4f} ({time.time()-t0:.1f}s)", flush=True)

# 2. GMM (diag covariance)
t0 = time.time()
gmm = GaussianMixture(K, covariance_type="diag", n_init=3, random_state=0, max_iter=100).fit(Fvs)
results["GMM K=48"] = cluster_metrics(gmm.predict(Fvs), K)
print(f"GMM: mean_Tstd={results['GMM K=48']['mean_tstd']:.4f} ({time.time()-t0:.1f}s)", flush=True)

# 3. Weighted KMeans (per-episode equal weight)
t0 = time.time()
ep_wts = np.ones(len(Fvs))
for e in samp:
    em = (Evs == e); ep_wts[em] = 1.0 / max(em.sum(), 1)
ep_wts = ep_wts / ep_wts.sum() * len(Fvs)
wkm = MiniBatchKMeans(K, n_init=3, random_state=0, batch_size=2048, max_iter=50).fit(Fvs, sample_weight=ep_wts)
results["WeightedKMeans K=48"] = cluster_metrics(wkm.labels_, K)
print(f"WeightedKMeans: mean_Tstd={results['WeightedKMeans K=48']['mean_tstd']:.4f} ({time.time()-t0:.1f}s)", flush=True)

# 4. HDBSCAN (adaptive K, density-based)
t0 = time.time()
# min_cluster_size: roughly NEPS/2 = 100 (a cluster should appear in at least half the eps)
hdb = HDBSCAN(min_cluster_size=50, min_samples=10, metric="euclidean", n_jobs=-1).fit(Fvs)
labs_hdb = hdb.labels_
K_hdb = len(set(labs_hdb)) - (1 if -1 in labs_hdb else 0)
results[f"HDBSCAN K={K_hdb}"] = cluster_metrics(labs_hdb, max(labs_hdb) + 1)
if results[f"HDBSCAN K={K_hdb}"]:
    print(f"HDBSCAN: K={K_hdb} mean_Tstd={results[f'HDBSCAN K={K_hdb}']['mean_tstd']:.4f} noise={np.mean(labs_hdb==-1):.1%} ({time.time()-t0:.1f}s)", flush=True)
else:
    print(f"HDBSCAN: all noise or single cluster ({time.time()-t0:.1f}s)", flush=True)

# 5. Bayesian GMM (adaptive K via Dirichlet process)
t0 = time.time()
bgmm = BayesianGaussianMixture(n_components=K, covariance_type="diag", n_init=1,
                                weight_concentration_prior=1e-3, random_state=0, max_iter=200).fit(Fvs)
labs_bgmm = bgmm.predict(Fvs)
# effective K = components with non-negligible weight
eff_K = int(np.sum(bgmm.weights_ > 0.01))
results[f"BayesianGMM K={eff_K}"] = cluster_metrics(labs_bgmm, K)
print(f"BayesianGMM: eff_K={eff_K} mean_Tstd={results[f'BayesianGMM K={eff_K}']['mean_tstd']:.4f} ({time.time()-t0:.1f}s)", flush=True)

# ---- figure ----
names = list(results.keys())
fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 5))
for ax, name in zip(axs, names):
    r = results[name]
    sc = ax.scatter(r["covs"], r["comps"], c=r["tstds"], s=np.clip(r["sizes"] / 5, 20, 250),
                     cmap="RdYlGn_r", alpha=.7, edgecolors="k", linewidth=.3)
    ax.set_xlabel("cross-episode coverage"); ax.set_ylabel("intra-cluster compactness (↓tight)")
    ax.set_title(f"{name}\ncov-compact corr={r['cov_compact_corr']:.3f}\nmean_Tstd={r['mean_tstd']:.4f}")
    ax.grid(alpha=.25); plt.colorbar(sc, ax=ax, label="T-std")
fig.suptitle(f"聚类方法对比 + coverage-density 关联 · {NEPS}ep · DINOv2 · 含自适应K方法", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .94])
out = "crave/docs/visualization/encoders/cluster_method_compare.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=120)
print("SAVED", out, flush=True)

# print summary table
print("\n===== SUMMARY =====")
for name in names:
    r = results[name]
    print(f"{name:25s} K={r['K_found']:3d}  mean_Tstd={r['mean_tstd']:.4f}  cov-compact_corr={r['cov_compact_corr']:.3f}")
