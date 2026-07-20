#!/usr/bin/env python
"""A线复议(用户要求): 全帧率(30Hz, stride=1, 零降采样)重跑 So400m-mean v1 value 读出。
与 3Hz 版(corr 0.578)唯一差别 = 帧率: PCA拟合/BGMM/覆盖/Viterbi/评估 全部在 30Hz 全帧, 无插值。
依据: CRAVE STATUS 旧结论 "30Hz 原生 >> 3Hz(aliasing), 高频采样是最强平滑器"。
输入: kai0_aligned_urvc/so400m-mean_s1/ep*.npz(fp16)   输出: kai0_so400m_value_labels_30hz/
参照: 生产 DINOv3(3Hz拟合+30Hz插值) corr 0.948。环境: conda:srpo, RAM 463G(全帧 float64 拟合可行)。
"""
import glob, json, os, platform, subprocess, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from scipy.ndimage import gaussian_filter1d

REPO = "/vePFS/tim/workspace/deepdive_kai0"
FEAT = f"{REPO}/lmvla/lmwm/data/kai0_aligned_urvc/so400m-mean_s1"
OUTD = f"{REPO}/lmvla/lmwm/data/kai0_so400m_value_labels_30hz"
CSQ, MIN_COV, LAM = 1000, 0.50, 16.0
DS = lambda e: f"{REPO}/kai0/data/Task_A/kai0_base/data/chunk-{e//CSQ:03d}/episode_{e:06d}.parquet"
GTP = lambda e: f"{REPO}/kai0/data/Task_A/kai0_advantage/data/chunk-{e//CSQ:03d}/episode_{e:06d}.parquet"
os.makedirs(OUTD, exist_ok=True)
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

files = sorted(glob.glob(f"{FEAT}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
print(f"loading {len(files)} eps (30Hz 全帧)...", flush=True)
imgL, stL, nlist = [], [], []
for i, f in enumerate(files):
    e = int(os.path.basename(f)[2:-4]); d = np.load(f)
    p = d["pooled"].astype(np.float32); fidx = d["fidx"]
    st = np.stack(pd.read_parquet(DS(e), columns=["observation.state"])["observation.state"].to_numpy())
    assert abs(len(p) - len(st)) <= 1, f"ep{e}: feat {len(p)} vs state {len(st)}"
    m = min(len(p), len(st))
    imgL.append(p[:m]); stL.append(st[:m].astype(np.float32)); nlist.append((e, m))
    if i % 500 == 0: print(f"  load {i}/{len(files)}", flush=True)
img = np.concatenate(imgL); ST = np.concatenate(stL); del imgL, stL
T = np.concatenate([np.linspace(0, 1, n) for _, n in nlist]); Ev = np.concatenate([np.full(n, e) for e, n in nlist])
NC_ = len(nlist)
print(f"frames={len(img)} ({len(img)/1e6:.2f}M)", flush=True)

t0 = time.time()
img = l2(img)
pca = PCA(n_components=128, svd_solver="randomized", random_state=0).fit(img)   # 全帧拟合, 零降采样
fq = np.empty((len(img), 128), np.float32)
for i in range(0, len(img), 200000):
    fq[i:i+200000] = pca.transform(img[i:i+200000]).astype(np.float32)
fq = l2(fq); del img
print(f"PCA done ({time.time()-t0:.0f}s)", flush=True)
SMU, SSD = ST.mean(0), ST.std(0) + 1e-8
Jn = l2(np.concatenate([fq, l2((ST - SMU) / SSD)], 1)).astype(np.float32); del fq, ST

t0 = time.time()
bgmm = BayesianGaussianMixture(n_components=40, covariance_type="diag", weight_concentration_prior=1e-2,
                               n_init=1, max_iter=150, random_state=0).fit(Jn)
labs = bgmm.predict(Jn)
print(f"BGMM eff={(bgmm.weights_>0.01).sum()} ({time.time()-t0:.0f}s)", flush=True)

def mode_split(Tc, nbins=30):
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1)); h = h.astype(float)/(h.sum()+1e-9)
    hs = gaussian_filter1d(h, 1.2); c = (ed[:-1]+ed[1:])/2
    peaks = [i for i in range(nbins) if hs[i] >= hs[max(0,i-1)] and hs[i] >= hs[min(nbins-1,i+1)] and hs[i] >= 0.10*hs.max()]
    merged = []
    for p in peaks:
        if merged and abs(c[p]-c[merged[-1]]) < 0.10:
            if hs[p] > hs[merged[-1]]: merged[-1] = p
        else: merged.append(p)
    final = [merged[0]] if merged else [int(np.argmax(hs))]
    for p in merged[1:]:
        valley = hs[final[-1]:p+1].min()
        if valley < 0.6*min(hs[final[-1]], hs[p]): final.append(p)
        elif hs[p] > hs[final[-1]]: final[-1] = p
    if len(final) <= 1: return [(float(np.median(Tc)), np.ones(len(Tc), bool))]
    cuts = [c[a+int(np.argmin(hs[a:b+1]))] for a, b in zip(final[:-1], final[1:])]
    edges = [0.0]+cuts+[1.0]; out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (Tc >= lo) & (Tc < hi)
        if msk.sum() >= 5: out.append((float(np.median(Tc[msk])), msk))
    return out if out else [(float(np.median(Tc)), np.ones(len(Tc), bool))]

# 30Hz 帧数≈3Hz×10 → 帧数门槛同比例 ×10(200), 保配方等价而非更松
cand = np.array([Jn[labs == k].mean(0) for k in range(40) if (labs == k).sum() >= 200], np.float32)
assign = np.empty(len(Jn), int)
for i in range(0, len(Jn), 20000):
    assign[i:i+20000] = np.linalg.norm(Jn[i:i+20000, None]-cand[None], axis=2).argmin(1)
targets, n_drop = [], 0
for ki in range(len(cand)):
    mk = assign == ki
    if mk.sum() < 200: continue
    for mv, sub in mode_split(T[mk]):
        if len(set(Ev[mk][sub].tolist())) / NC_ >= MIN_COV: targets.append((float(np.median(T[mk][sub])), cand[ki]))
        else: n_drop += 1
targets.sort(key=lambda t: t[0])
vals = np.array([t[0] for t in targets]); Ctgt = np.array([t[1] for t in targets], np.float32)
print(f"M={len(vals)} milestones (dropped {n_drop}): {[round(v,2) for v in vals]}", flush=True)

bins = np.unique(np.concatenate([[0.0], vals, [1.0]])); nb = len(bins)
cbn = [int(np.searchsorted(bins, v)) for v in vals]; pen = LAM*np.abs(bins[:, None]-bins[None])
def vit(Fq):
    de = np.linalg.norm(Fq[:, None]-Ctgt[None], axis=2); em = np.full((len(Fq), nb), 1e3)
    for ti in range(len(vals)): em[:, cbn[ti]] = np.minimum(em[:, cbn[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :]+pen; kk = tr.argmin(1); cost = em[j]+tr[np.arange(nb), kk]; BP[j] = kk
    cost[nb-1] -= 2; s = int(cost.argmin()); path = np.zeros(len(Fq), int); path[-1] = s
    for j in range(len(Fq)-2, -1, -1): s = BP[j+1][s]; path[j] = s
    return bins[path]

ptr = 0; cs, dds, ends = [], [], []
for idx, (e, n) in enumerate(nlist):
    v30 = vit(Jn[ptr:ptr+n]); ptr += n                     # 30Hz 原生, 无插值
    gt = pd.read_parquet(GTP(e), columns=["stage_progress_gt"])["stage_progress_gt"].values.astype(np.float32)[:n]
    lo, hi = v30.min(), v30.max()
    if hi > lo+1e-6: v30 = (v30-lo)/(hi-lo)
    np.save(f"{OUTD}/ep{e}.npy", v30.astype(np.float32))
    if np.std(v30) > 1e-6 and np.std(gt) > 1e-6: cs.append(np.corrcoef(v30, gt)[0, 1])
    dds.append((np.maximum.accumulate(v30)-v30).max()); ends.append(v30[-5:].mean())
    if idx % 500 == 0: print(f"  vit {idx}/{NC_}", flush=True)
print(f"\n[So400m-mean 30Hz全帧 v1管线] corr(med)={np.median(cs):.3f}  drawdown(med)={np.median(dds):.4f}  end={np.mean(ends):.3f}  M={len(vals)}")
print("对照: 3Hz版 So400m=0.578(特化最佳0.634) | 生产 DINOv3(3Hz拟合)=0.948")

np.savez(f"{OUTD}/_spec.npz", vals=vals, Ctgt=Ctgt, pca_mean=pca.mean_, pca_components=pca.components_, SMU=SMU, SSD=SSD)
gh = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
import torch, transformers, sklearn, scipy
json.dump({"script": "lmvla/lmwm/scripts/gen_so400m_value_labels_30hz.py", "script_git_hash": gh,
           "env": "conda:srpo", "python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "sklearn": sklearn.__version__,
           "numpy": np.__version__, "scipy": scipy.__version__,
           "feat": "so400m-mean_s1(30Hz全帧, fp16存fp32算)", "recipe": "gen_final_v3 同配方@30Hz, 帧数门槛×10等价缩放", "date": "2026-07-21"},
          open(f"{OUTD}/_env.json", "w"), ensure_ascii=False, indent=1)
print("labels + _spec + _env ->", OUTD)
