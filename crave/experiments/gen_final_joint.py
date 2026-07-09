#!/usr/bin/env python
"""最终 milestone + AE 标签: DINOv3-base PCA128 ⊕ proprio(pos+vel) 全量 kai0_base.
K0 自适应(saturation), min_cov=0.50, Viterbi λ=16 no smooth.
"""
import glob, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.isotonic import IsotonicRegression
from crave.utils import mkp
from crave.config import resolve_dataset
from crave.data import kai0
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0")
MIN_COV = 0.50; LAM = 16.0; CSQ = 1000
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ---- Load image PCA128 + proprio ----
d = np.load(REPO / "temp/crave_d3b_pca128/milestones.npz")
pca_m = d["pca_mean"]; pca_c = d["pca_components"]
FEAT = REPO / "temp/crave_d3b_pca128/feats"
eps = sorted(int(p.stem[2:]) for p in FEAT.glob("ep*.npy")); NC = len(eps)
cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E_idx, FR_idx = zf["E"], zf["FR"]

print(f"loading img⊕proprio for {NC} eps...", flush=True)
imgF = []; T_raw = []; E_raw = []; P = []; nlist = []
for e in eps:
    f = np.load(FEAT / f"ep{e}.npy").astype(np.float32); fq = l2((l2(f) - pca_m) @ pca_c.T)
    n = len(fq); nlist.append((e, n))
    imgF.append(fq); T_raw.append(np.linspace(0, 1, n)); E_raw.append(np.full(n, e))
    loc = np.where(E_idx == e)[0]; o = np.argsort(FR_idx[loc]); fr = FR_idx[loc][o]
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy())
    pm = mkp(st)  # [state, Δstate] = pos+vel (temporal)
    P.append(pm[np.minimum(fr[:n], len(pm) - 1)])
imgF = np.concatenate(imgF); T_raw = np.concatenate(T_raw); E_raw = np.concatenate(E_raw)
P = np.concatenate(P).astype(np.float32)
PMU, PSD = P.mean(0), P.std(0) + 1e-8; Pn = l2((P - PMU) / PSD)
jointF = np.concatenate([imgF, Pn], 1)
print(f"joint feature {jointF.shape} (img128 ⊕ proprio28)", flush=True)

# ---- K0 saturation (find max K_eff) ----
print("K0 saturation sweep...", flush=True)
K0s = list(range(5, 50, 3)) + [60, 80]
ks = []
for K0 in K0s:
    km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=8192, max_iter=30).fit(l2(jointF))
    labs = km.labels_
    ke = sum(1 for k in range(K0) if (labs == k).sum() >= 20
             and len(set(E_raw[labs == k].tolist())) / NC >= MIN_COV)
    ks.append(ke)
max_k = max(ks); sat_k0 = K0s[ks.index(max_k)]
print(f"  K_eff curve: {list(zip(K0s, ks))}", flush=True)
print(f"  SAT: K0={sat_k0} max_K_eff={max_k}", flush=True)

# ---- Final clustering at sat_K0 ----
km = KMeans(sat_k0, n_init=3, random_state=0).fit(l2(jointF)); labs = km.labels_
sel = []
for k in range(sat_k0):
    mk = labs == k; nf = mk.sum()
    if nf < 20: continue
    cov = len(set(E_raw[mk].tolist())) / NC
    if cov >= MIN_COV:
        sel.append({"k": k, "tstd": float(np.nanstd(T_raw[mk])), "tpos": float(T_raw[mk].mean()),
                    "cov": cov, "nf": nf})
sel = sorted(sel, key=lambda c: c["tpos"]); M = len(sel)
Cs = l2(np.array([km.cluster_centers_[c["k"]] for c in sel], dtype=np.float32))
Pord = np.asarray(IsotonicRegression(increasing=True).fit_transform(
    np.arange(M), np.array([c["tpos"] for c in sel])), dtype=np.float64)
covs = np.array([c["cov"] for c in sel]); tstds = np.array([c["tstd"] for c in sel])
print(f"\nFINAL: K0={sat_k0} M={M} milestones")
print(f"  cov  mean={covs.mean():.3f} range=[{covs.min():.3f},{covs.max():.3f}]")
print(f"  Tstd mean={tstds.mean():.4f} range=[{tstds.min():.4f},{tstds.max():.4f}]")
print(f"  Pord [{Pord[0]:.3f}, {Pord[-1]:.3f}]", flush=True)

np.savez(REPO / "temp/crave_final_milestones_joint.npz", C=Cs, Pord=Pord, M=M, sat_k0=sat_k0,
         min_cov=MIN_COV, covs=covs, tstds=tstds, pca_mean=pca_m, pca_components=pca_c,
         PMU=PMU, PSD=PSD)
print("milestones saved -> crave_final_milestones_joint.npz", flush=True)

# ---- Viterbi labels ----
bins = np.unique(np.concatenate([[0.], Pord, [1.]])); nb = len(bins)
cb = [int(np.searchsorted(bins, p)) for p in Pord]; pen = LAM * np.abs(bins[:, None] - bins[None])
def vit(Fq):
    de = np.linalg.norm(Fq[:, None] - Cs[None], axis=2); em = np.full((len(Fq), nb), 1e3)
    for ci in range(M): em[:, cb[ci]] = np.minimum(em[:, cb[ci]], de[:, ci])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(nb), k]; BP[j] = k
    cost[nb - 1] -= 2; s = int(cost.argmin()); path = np.zeros(len(Fq), int); path[-1] = s
    for j in range(len(Fq) - 2, -1, -1): s = BP[j + 1][s]; path[j] = s
    return bins[path]

LAB = REPO / "temp/crave_ae_labels/final"; LAB.mkdir(parents=True, exist_ok=True)
# rebuild per-ep joint features from concatenated arrays
print("Viterbi per-ep...", flush=True)
t0 = time.time(); ptr = 0; n_done = 0; dd_all = []
for (e, n) in nlist:
    Fq = jointF[ptr:ptr + n]; ptr += n
    v3 = vit(Fq)
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy())
    n30 = len(st); v30 = np.interp(np.linspace(0, 1, n30), np.linspace(0, 1, n), v3)
    lo, hi = v30.min(), v30.max()
    if hi > lo + 1e-6: v30 = (v30 - lo) / (hi - lo)
    np.save(LAB / f"ep{e}.npy", v30.astype(np.float32)); n_done += 1
    dd_all.append(float((np.maximum.accumulate(v30) - v30).max()))
    if n_done % 1000 == 0: print(f"  [{n_done}/{len(nlist)}]", flush=True)
print(f"DONE {n_done} eps dd_mean={np.mean(dd_all):.3f} dd_max={max(dd_all):.3f} ({(time.time()-t0)/60:.1f}min)", flush=True)
print(f"Labels: {LAB}", flush=True)
