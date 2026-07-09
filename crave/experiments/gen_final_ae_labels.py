#!/usr/bin/env python
"""最终 AE 标签生成: DINOv2-L→PCA256→Overcluster+Otsu(milestones)→完整对称Viterbi→per-ep label。
存 temp/crave_ae_labels/final/ep{e}.npy (native-fps, 0→1 norm01)。
"""
import sys, glob, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crave.utils import mkp
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0")
OUT = REPO / "temp/crave_ae_labels/final"; OUT.mkdir(parents=True, exist_ok=True)
NEPS_TOTAL = 3055; CSQ = 1000
rng = np.random.RandomState(42)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ---- Step 1: Load all 3Hz shard features (DINOv2-large) ----
print("loading DINOv2-large shards...", flush=True)
zf = np.load(REPO / "temp/crave_full/index_dino.npz")
E, FR, T = zf["E"], zf["FR"], zf["T"]; N = int(zf["n"])
feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
for f in sorted(glob.glob(str(REPO / "temp/crave_full/dino/shard_*.npz"))):
    z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
vi = np.where(valid)[0]; Ev, FRv, Tv = E[vi], FR[vi], T[vi]
ep_list = sorted(set(Ev.tolist()))
print(f"  {len(vi)} valid frames, {len(ep_list)} eps", flush=True)

# ---- Step 2: PCA to 256D ----
print("PCA 1024→256D...", flush=True)
img = l2(feat[vi].astype(np.float32))
pca = PCA(n_components=256, random_state=0).fit(img)
img_pca = l2(pca.transform(img))
np.savez(REPO / "temp/crave_pca256.npz", mean=pca.mean_, components=pca.components_)
print(f"  PCA256 explained variance: {pca.explained_variance_ratio_.sum():.3f}", flush=True)

# ---- Step 3: Overcluster + Otsu on ALL eps (global milestones) ----
print("Overcluster + Otsu (global, all eps)...", flush=True)
K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 64, 320))
km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=8192, max_iter=30).fit(img_pca)
labs = km.labels_
# Per-cluster coverage
print(f"  K0={K0}, computing coverage...", flush=True)
clusters = []
for k in range(K0):
    mk = (labs == k)
    if mk.sum() < 5: continue
    cov = len(set(Ev[mk].tolist())) / len(ep_list)
    tpos = float(Tv[mk].mean())
    tstd = float(Tv[mk].std())
    clusters.append({"k": k, "cov": cov, "tpos": tpos, "tstd": tstd, "size": mk.sum()})
# Otsu threshold
covs = np.array([c["cov"] for c in clusters])
tau = float(np.median(covs) + 0.5 * np.std(covs))
sel = sorted([c for c in clusters if c["cov"] >= tau], key=lambda c: c["tpos"])
M = len(sel)
print(f"  τ={tau:.3f}, K_eff={M} milestones (from {K0} overclustered)", flush=True)
# Milestone centroids + progress values
Cs = np.array([km.cluster_centers_[c["k"]] for c in sel], dtype=np.float32)
Cs = l2(Cs)
Pord_raw = np.array([c["tpos"] for c in sel], dtype=np.float64)
from sklearn.isotonic import IsotonicRegression
Pord = np.asarray(IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pord_raw), dtype=np.float64)
print(f"  Pord range: [{Pord[0]:.3f}, {Pord[-1]:.3f}]", flush=True)
np.savez(REPO / "temp/crave_final_milestones.npz", C=Cs, Pord=Pord, tau=tau, K0=K0, M=M,
         sel_k=[c["k"] for c in sel], covs=[c["cov"] for c in sel])

# ---- Step 4: Per-ep Viterbi value (3Hz native, λ=16 fixed) ----
LAM = 16.0  # fixed architecture constant
bins = np.unique(np.concatenate([[0.0], Pord, [1.0]]))
nb = len(bins)
cb = [int(np.searchsorted(bins, p)) for p in Pord]
pen = LAM * np.abs(bins[:, None] - bins[None])

print(f"Viterbi per-ep (λ={LAM}, {nb} bins)...", flush=True)
n_done = 0; n_err = 0; t0 = time.time()
from crave.config import resolve_dataset; from crave.data import kai0
cfg = resolve_dataset("kai0_base"); cs2 = kai0.chunks_size(cfg.root); DS = Path(cfg.root)

def vit_value(Fq):
    """Full symmetric Viterbi, returns per-frame value [0,1]."""
    d = np.linalg.norm(Fq[:, None] - Cs[None], axis=2)  # (n_frames, M)
    em = np.full((len(Fq), nb), 1e3)
    for ci in range(M): em[:, cb[ci]] = np.minimum(em[:, cb[ci]], d[:, ci])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]
    BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; k = tr.argmin(1)
        cost = em[j] + tr[np.arange(nb), k]; BP[j] = k
    cost[nb - 1] -= 2  # end bonus
    s = int(cost.argmin()); path = np.zeros(len(Fq), int); path[-1] = s
    for j in range(len(Fq) - 2, -1, -1): s = BP[j + 1][s]; path[j] = s
    return bins[path]

for ei, e in enumerate(ep_list):
    try:
        fp = OUT / f"ep{e}.npy"
        if fp.exists(): continue
        loc = np.where(Ev == e)[0]; o = np.argsort(FRv[loc]); loc = loc[o]
        Fq = img_pca[loc]  # 3Hz features for this ep
        v3 = vit_value(Fq)  # 3Hz native value

        # Interpolate to native fps (30Hz)
        st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                       columns=["observation.state"])["observation.state"].to_numpy())
        n30 = len(st)
        xi = np.linspace(0, 1, len(v3)); xo = np.linspace(0, 1, n30)
        v30 = np.interp(xo, xi, v3)

        # per-ep min-max norm01 (all kai0_base are successful demos)
        lo, hi = float(v30.min()), float(v30.max())
        if hi > lo + 1e-6: v30 = (v30 - lo) / (hi - lo)
        else: v30 = np.linspace(0, 1, n30).astype(np.float32)

        np.save(fp, v30.astype(np.float32)); n_done += 1
        if n_done % 200 == 0:
            el = time.time() - t0
            print(f"  [{n_done}/{len(ep_list)}] ep{e} n={n30} v∈[{v30.min():.2f},{v30.max():.2f}] ({el/60:.0f}min)", flush=True)
    except Exception as ex:
        n_err += 1
        if n_err <= 5: print(f"  ERR ep{e}: {ex}", flush=True)

print(f"\nDONE {n_done} eps, {n_err} errors in {(time.time()-t0)/60:.1f}min", flush=True)
print(f"Labels: {OUT}/", flush=True)
