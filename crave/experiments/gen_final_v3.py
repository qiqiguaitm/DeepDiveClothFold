#!/usr/bin/env python
"""最终 v3: img⊕pos + BayesianGMM + 【中位数值 + per-mode 覆盖率筛选】+ 多发射 Viterbi.
修正: ① value 用 median(保序, 消除 mode 倒挂); ② 每个 mode 单独要 coverage≥0.5(非总和凑数).
"""
import glob, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.mixture import BayesianGaussianMixture
from scipy.ndimage import gaussian_filter1d
from crave.config import resolve_dataset
from crave.data import kai0
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
jointF = np.concatenate([imgF, l2((ST - SMU) / SSD)], 1)

# ---- BayesianGMM ----
print("BayesianGMM fitting...", flush=True); t0 = time.time()
bgmm = BayesianGaussianMixture(n_components=40, covariance_type="diag", weight_concentration_prior=1e-2,
                               n_init=1, max_iter=150, random_state=0).fit(l2(jointF))
labs = bgmm.predict(l2(jointF))
print(f"  eff_components={(bgmm.weights_>0.01).sum()} ({time.time()-t0:.0f}s)", flush=True)

def mode_split(Tc, nbins=30):
    """返回 [(median_value, member_bool_within_Tc), ...] — 谷分裂 + 每段中位数."""
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1)); h = h.astype(float) / (h.sum() + 1e-9)
    hs = gaussian_filter1d(h, 1.2); c = (ed[:-1] + ed[1:]) / 2
    peaks = [i for i in range(nbins) if hs[i] >= hs[max(0, i-1)] and hs[i] >= hs[min(nbins-1, i+1)] and hs[i] >= 0.10 * hs.max()]
    merged = []
    for p in peaks:
        if merged and abs(c[p] - c[merged[-1]]) < 0.10:
            if hs[p] > hs[merged[-1]]: merged[-1] = p
        else: merged.append(p)
    final = [merged[0]] if merged else [int(np.argmax(hs))]
    for p in merged[1:]:
        valley = hs[final[-1]:p+1].min()
        if valley < 0.6 * min(hs[final[-1]], hs[p]): final.append(p)
        elif hs[p] > hs[final[-1]]: final[-1] = p
    if len(final) <= 1:
        return [(float(np.median(Tc)), np.ones(len(Tc), bool))]
    cuts = []
    for a, b in zip(final[:-1], final[1:]):
        cuts.append(c[a + int(np.argmin(hs[a:b+1]))])
    edges = [0.0] + cuts + [1.0]; out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (Tc >= lo) & (Tc < hi)
        if msk.sum() >= 5: out.append((float(np.median(Tc[msk])), msk))
    return out if out else [(float(np.median(Tc)), np.ones(len(Tc), bool))]

# ---- 候选质心 = bgmm 各 component (>=20 帧) ----
cand=[]
for k in range(40):
    mk=labs==k
    if mk.sum()<20: continue
    cand.append(l2(jointF)[mk].mean(0))
cand=np.array(cand,dtype=np.float32)
# 最近质心分配(和 Viterbi 一致)
assign=np.empty(len(jointF),int)
Jn=l2(jointF)
for i in range(0,len(Jn),20000):
    assign[i:i+20000]=np.linalg.norm(Jn[i:i+20000,None]-cand[None],axis=2).argmin(1)

# ---- per-candidate: mode-split + per-mode median/coverage (全在最近质心分配上) ----
targets=[]; n_dropped_mode=0; kept=0
for ki in range(len(cand)):
    mk=assign==ki; nf=mk.sum()
    if nf<20: continue
    Tc=T[mk]; Ec=Ev[mk]
    ms=mode_split(Tc)  # [(median, submask)]
    valid=[]
    for mv,sub in ms:
        cov_mode=len(set(Ec[sub].tolist()))/NC
        med=float(np.median(Tc[sub]))
        if cov_mode>=MIN_COV: valid.append(med)
        else: n_dropped_mode+=1
    if not valid: continue
    kept+=1
    for med in valid: targets.append((med,cand[ki],ki,len(valid)>1))
M=len(set(t[2] for t in targets))
print(f"kept {kept} clusters, {len(targets)} emission targets ({M} milestones), dropped {n_dropped_mode} weak modes",flush=True)
targets.sort(key=lambda t:t[0])
vals=np.array([t[0] for t in targets]); Ctgt=np.array([t[1] for t in targets],dtype=np.float32)
print(f"milestone values(median,最近质心): {[round(v,2) for v in vals]}",flush=True)

bins = np.unique(np.concatenate([[0.0], vals, [1.0]])); nb = len(bins)
cbn = [int(np.searchsorted(bins, v)) for v in vals]; pen = LAM * np.abs(bins[:, None] - bins[None])
def vit(Fq):
    de = np.linalg.norm(Fq[:, None] - Ctgt[None], axis=2); em = np.full((len(Fq), nb), 1e3)
    for ti in range(len(vals)): em[:, cbn[ti]] = np.minimum(em[:, cbn[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    cost[nb - 1] -= 2; s = int(cost.argmin()); path = np.zeros(len(Fq), int); path[-1] = s
    for j in range(len(Fq) - 2, -1, -1): s = BP[j + 1][s]; path[j] = s
    return bins[path]

np.savez(REPO / "temp/crave_final_v3b.npz", vals=vals, Ctgt=Ctgt, M=M, min_cov=MIN_COV,
         cluster_idx=np.array([t[2] for t in targets]), pca_mean=pca_m, pca_components=pca_c, SMU=SMU, SSD=SSD)
print("spec saved -> crave_final_v3b.npz", flush=True)

LAB = REPO / "temp/crave_ae_labels/final"; LAB.mkdir(parents=True, exist_ok=True)
t0 = time.time(); ptr = 0; nd = 0; dd = []
for (e, n) in nlist:
    Fq = jointF[ptr:ptr + n]; ptr += n; v3 = vit(Fq)
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy())
    n30 = len(st); v30 = np.interp(np.linspace(0, 1, n30), np.linspace(0, 1, n), v3)
    lo, hi = v30.min(), v30.max()
    if hi > lo + 1e-6: v30 = (v30 - lo) / (hi - lo)
    np.save(LAB / f"ep{e}.npy", v30.astype(np.float32)); nd += 1; dd.append(float((np.maximum.accumulate(v30) - v30).max()))
    if nd % 1000 == 0: print(f"  [{nd}/{len(nlist)}]", flush=True)
allv = [np.load(LAB / f"ep{e}.npy") for (e, _) in nlist]
ends = np.array([v[-5:].mean() for v in allv]); monos = np.array([(np.diff(v) >= -1e-6).mean() for v in allv])
print(f"DONE {nd} eps dd_mean={np.mean(dd):.3f} 终值>0.9={np.mean(ends>0.9):.0%} 单调={monos.mean():.1%} ({(time.time()-t0)/60:.1f}min)", flush=True)
