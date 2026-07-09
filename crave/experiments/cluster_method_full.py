#!/usr/bin/env python
"""全聚类方法对比 + 自适应K测试: KMeans / GMM(diag) / WeightedKMeans / HDBSCAN。
统一在 DINOv3-H 1280D 上,50 ep, 比较时间纯度 + coverage + 自适应K行为。
"""
import numpy as np, glob, time
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0")
NEPS = 50; rng = np.random.RandomState(42)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ---- load data once ----
print("loading...", flush=True)
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz")
E, T = zf["E"], zf["T"]; N = int(zf["n"])
feat = np.zeros((N, 1280), np.float16); valid = np.zeros(N, bool)
for f in sorted(glob.glob(str(REPO / "temp/crave_full_dinov3h/shard_*.npz"))):
    z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
vi = np.where(valid)[0]; Fv = l2(feat[vi].astype(np.float32)); Ev, Tv = E[vi], T[vi]
samp = sorted(rng.choice(sorted(set(Ev.tolist())), NEPS, replace=False))
m = np.isin(Ev, samp); Fvs, Tvs, Evs = np.ascontiguousarray(Fv[m]), Tv[m], Ev[m]
print(f"{len(Fvs)} frames, {NEPS} eps", flush=True)

# ---- metrics helper ----
def metrics(labs, K_found, name):
    tstds, covs, sizes = [], [], []
    noise_frac = float(np.mean(labs == -1))
    for k in range(K_found):
        mk = (labs == k)
        if mk.sum() < 5: continue
        tstds.append(float(np.nanstd(Tvs[mk])))
        covs.append(len(set(Evs[mk].tolist())) / NEPS)
        sizes.append(mk.sum())
    if not tstds: return None
    sizes_arr = np.array(sizes)
    gini = 1 - 2 * np.sum((np.arange(len(sizes_arr)) + 1) * np.sort(sizes_arr)) / (len(sizes_arr) * sizes_arr.sum() + 1e-9)
    return {"name": name, "K_found": K_found, "K_used": len(tstds),
            "mean_tstd": np.mean(tstds), "median_tstd": np.median(tstds),
            "mean_cov": np.mean(covs), "noise": noise_frac,
            "gini": gini, "tstds": tstds, "covs": covs}

all_results = []

# ---- 1. KMeans (K=20) ----
K = 20
t0 = time.time()
km = KMeans(K, n_init=3, random_state=0).fit(Fvs)
r = metrics(km.labels_, K, f"KMeans K={K}")
el = time.time() - t0
print(f"  {r['name']}: mean_Tstd={r['mean_tstd']:.4f} cov={r['mean_cov']:.3f} gini={r['gini']:.3f} ({el:.0f}s)", flush=True)
all_results.append(r)

# ---- 2. GMM(diag) K=20 ----
t0 = time.time()
gmm = GaussianMixture(K, covariance_type="diag", n_init=1, reg_covar=1e-4,
                       max_iter=50, random_state=0).fit(Fvs)
r = metrics(gmm.predict(Fvs), K, f"GMM(diag) K={K}")
el = time.time() - t0
print(f"  {r['name']}: mean_Tstd={r['mean_tstd']:.4f} cov={r['mean_cov']:.3f} gini={r['gini']:.3f} ({el:.0f}s)", flush=True)
all_results.append(r)

# ---- 3. WeightedKMeans (per-episode equal weight) ----
t0 = time.time()
ep_wts = np.ones(len(Fvs))
for e in samp:
    em = (Evs == e); ep_wts[em] = 1.0 / max(em.sum(), 1)
ep_wts = ep_wts / ep_wts.sum() * len(Fvs)
wkm = MiniBatchKMeans(K, n_init=3, random_state=0, batch_size=2048,
                       max_iter=50).fit(Fvs, sample_weight=ep_wts)
r = metrics(wkm.labels_, K, f"WeightedKMeans K={K}")
el = time.time() - t0
print(f"  {r['name']}: mean_Tstd={r['mean_tstd']:.4f} cov={r['mean_cov']:.3f} gini={r['gini']:.3f} ({el:.0f}s)", flush=True)
all_results.append(r)

# ---- 4. HDBSCAN (adaptive K) ----
for mcs in [20, 50, 100]:  # min_cluster_size sweep
    t0 = time.time()
    hdb = HDBSCAN(min_cluster_size=mcs, min_samples=5, metric="euclidean").fit(Fvs)
    labs_hdb = hdb.labels_
    K_hdb = len(set(labs_hdb)) - (1 if -1 in labs_hdb else 0)
    r = metrics(labs_hdb, max(labs_hdb) + 1, f"HDBSCAN mcs={mcs} K={K_hdb}")
    el = time.time() - t0
    if r:
        print(f"  {r['name']}: mean_Tstd={r['mean_tstd']:.4f} cov={r['mean_cov']:.3f} noise={r['noise']:.1%} ({el:.0f}s)", flush=True)
        all_results.append(r)
    else:
        print(f"  HDBSCAN mcs={mcs}: all noise or <5/cluster ({el:.0f}s)", flush=True)

# ---- 5. Overcluster + Otsu (current CRAVE) ----
K0 = int(np.clip(round(0.55 * np.sqrt(len(Fvs))), 48, 128))
t0 = time.time()
km_oc = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=4096, max_iter=30).fit(Fvs)
labs_oc = km_oc.labels_
# Otsu filter on coverage
covs_oc = []
for k in range(K0):
    mk = (labs_oc == k)
    if mk.sum() > 4:
        covs_oc.append(len(set(Evs[mk].tolist())) / NEPS)
covs_oc = np.array(covs_oc)
if len(covs_oc) > 3:
    # simple Otsu: threshold = mean between two gaussians
    tau = float(np.median(covs_oc) + 0.5 * np.std(covs_oc))
    sel = [k for k in range(K0) if (labs_oc == k).sum() > 4 and covs_oc[k] >= tau]
    K_eff = len(sel)
    r = metrics(labs_oc, K0, f"Overcluster(K0={K0})+Otsu K={K_eff}")
    el = time.time() - t0
    print(f"  {r['name']}: mean_Tstd={r['mean_tstd']:.4f} cov={r['mean_cov']:.3f} ({el:.0f}s)", flush=True)
    all_results.append(r)

# ---- Figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
names = [r["name"] for r in all_results]
cols = plt.cm.tab10(np.linspace(0, 1, len(names)))

# Bar: mean_Tstd
means_t = [r["mean_tstd"] for r in all_results]
bars = ax1.bar(range(len(names)), means_t, color=cols, alpha=.85)
ax1.set_xticks(range(len(names))); ax1.set_xticklabels([n[:28] for n in names], rotation=20, ha="right", fontsize=8)
ax1.set_ylabel("mean per-cluster T-std (↓好)"); ax1.set_title("时间纯度对比"); ax1.grid(alpha=.25, axis="y")
for b, v in zip(bars, means_t): ax1.text(b.get_x() + b.get_width() / 2, v + .002, f"{v:.4f}", ha="center", fontsize=7)

# Scatter: K vs mean_tstd
for i, r in enumerate(all_results):
    ax2.scatter(r["K_used"], r["mean_tstd"], s=120, c=[cols[i]], edgecolors="k", linewidth=.5, zorder=3)
    ax2.annotate(r["name"][:20], (r["K_used"], r["mean_tstd"]), fontsize=7, xytext=(5, 5), textcoords="offset points")
ax2.set_xlabel("effective K"); ax2.set_ylabel("mean per-cluster T-std"); ax2.set_title("K vs 时间纯度 (理想: 右下角)"); ax2.grid(alpha=.25)

fig.suptitle(f"聚类方法全对比 · {NEPS} ep · DINOv3-H 1280D · 含自适应K方法", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .94])
out = "crave/docs/visualization/encoders/cluster_method_full.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=120)
print("SAVED", out, flush=True)

# Summary table
print("\n===== SUMMARY =====", flush=True)
print(f"{'Method':<35s} {'K_found':>7s} {'T-std':>8s} {'mean_cov':>8s} {'Gini':>6s} {'noise':>6s}", flush=True)
for r in all_results:
    print(f"{r['name']:<35s} {r['K_used']:>7d} {r['mean_tstd']:>8.4f} {r['mean_cov']:>8.3f} {r['gini']:>6.3f} {r['noise']:>6.1%}", flush=True)
