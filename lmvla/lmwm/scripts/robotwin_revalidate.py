#!/usr/bin/env python
"""robotwin V0/V1 跨本体复核 —— 同一套超参(和 LIBERO/kai0 完全一致)跑 robotwin 72 个 ≥10ep 任务。
V0: r 场是否非退化(std>0)。 V1: r-低谷边界数涌现 + 跨-ep 稳定(真 recall vs 随机)。
证明"同一 kNN/σ/阈值"跨本体(aloha 双臂)成立 = C1 普适。
Out: assets/robotwin_revalidate.png + 打印汇总
"""
import os, glob, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = "/vePFS/tim/workspace/deepdive_kai0"
RFEAT = f"{REPO}/lmvla/lmwm/data/robotwin_dinov3base"; RROOT = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
NB = 50; DELTA = 0.08; THR = 0.03; rng = np.random.RandomState(0)
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)

def recur(gd):
    eps = list(gd); F = l2(np.concatenate([gd[e] for e in eps]).astype(np.float32))
    ep = np.concatenate([np.full(len(gd[e]), i) for i, e in enumerate(eps)]); ne = len(eps)
    D = cdist(F, F); dmin = np.full((len(F), ne), 1e9, np.float32)
    for j in range(ne): dmin[:, j] = D[:, ep == j].min(1)
    other = ep[:, None] != np.arange(ne)[None]; sig = np.median(dmin[other])
    r = (np.exp(-dmin**2/(2*sig*sig))*other).sum(1)/(ne-1)
    out = {}; off = 0
    for e in eps: n = len(gd[e]); out[e] = r[off:off+n]; off += n
    return out, r

def valleys(c): return find_peaks(-gaussian_filter1d(c, 1.4), prominence=THR, distance=4)[0]

def v1(gd, rr):
    eps = list(gd); B = np.zeros((len(eps), NB))
    for i, e in enumerate(eps):
        t = np.linspace(0, 1, len(gd[e])); B[i] = np.interp(np.linspace(0, 1, NB), t, rr[e])
    med = np.median(B, 0); vb = valleys(med)/(NB-1)
    ep_v = [valleys(B[i])/(NB-1) for i in range(len(B))]
    def recall(bnds):
        if len(bnds) == 0: return np.nan
        return float(np.mean([np.mean([np.any(np.abs(np.array(vv)-b) <= DELTA) if len(vv) else False for vv in ep_v]) for b in bnds]))
    real = recall(vb); rand = float(np.mean([recall(rng.uniform(0.1, 0.9, len(vb))) for _ in range(20)])) if len(vb) else np.nan
    return len(vb), real, rand

def main():
    import pyarrow.parquet as pq
    from collections import defaultdict
    t0 = time.time()
    cached = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(f"{RFEAT}/ep*.npz"))
    t2e = defaultdict(list)
    for e in cached:
        fs = glob.glob(f"{RROOT}/data/chunk-*/episode_{e:06d}.parquet")
        if fs: t2e[int(pq.read_table(fs[0], columns=["task_index"]).column("task_index")[0].as_py())].append(e)
    big = sorted([(t, es) for t, es in t2e.items() if len(es) >= 10], key=lambda x: -len(x[1]))
    print(f"robotwin: {len(big)} tasks with >=10ep ({time.time()-t0:.0f}s)", flush=True)
    stds, nbs, reals, rands = [], [], [], []
    for t, eps in big:
        gd = {e: np.load(f"{RFEAT}/ep{e}.npz")["pooled"].astype(np.float32) for e in eps}
        rr, rall = recur(gd); nb, real, rand = v1(gd, rr)
        stds.append(float(rall.std())); nbs.append(nb); reals.append(real); rands.append(rand)
        print(f"  t{t}({len(eps)}ep): r_std={rall.std():.3f} boundaries={nb} recall={real if real==real else float('nan'):.2f} vs rand={rand if rand==rand else float('nan'):.2f}", flush=True)
    stds = np.array(stds); nbs = np.array(nbs); reals = np.array(reals); rands = np.array(rands); ok = ~np.isnan(reals)
    print(f"\n[SUMMARY robotwin {len(big)} tasks] (同 LIBERO/kai0 超参)")
    print(f"  V0 r_std: median={np.median(stds):.3f} min={stds.min():.3f} | 全部>0.02: {np.mean(stds>0.02)*100:.0f}% (非退化)")
    print(f"  V1 boundaries: median={int(np.median(nbs))} dist={np.bincount(nbs)}")
    print(f"  V1 cross-ep: real recall median={np.nanmedian(reals):.2f} vs random={np.nanmedian(rands):.2f} | real>random on {np.mean(reals[ok]>rands[ok])*100:.0f}%", flush=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(stds, bins=20, color="#7c3aed", alpha=.8); ax[0].axvline(0.02, color="r", ls="--"); ax[0].set_title(f"V0 r_std (median {np.median(stds):.3f}, all>0.02={np.mean(stds>0.02)*100:.0f}%)", fontsize=9); ax[0].set_xlabel("r std")
    ax[1].hist(nbs, bins=np.arange(-0.5, nbs.max()+1.5), color="#7c3aed", alpha=.8); ax[1].set_title(f"V1 boundary count (median {int(np.median(nbs))})", fontsize=9); ax[1].set_xlabel("# boundaries")
    ax[2].scatter(rands[ok], reals[ok], c="#7c3aed", s=25, alpha=.7); ax[2].plot([0,1],[0,1],'--',color="#888"); ax[2].set_xlim(0,1); ax[2].set_ylim(0,1)
    ax[2].set_xlabel("random recall"); ax[2].set_ylabel("real recall"); ax[2].set_title(f"V1 cross-ep stable (real {np.nanmedian(reals):.2f} vs rand {np.nanmedian(rands):.2f})", fontsize=9); ax[2].grid(alpha=.2)
    fig.suptitle(f"robotwin2.0 cross-embodiment re-validation — SAME hyperparams as LIBERO/kai0, {len(big)} tasks", fontsize=12)
    fig.tight_layout(); out = f"{REPO}/lmvla/lmwm/docs/assets/robotwin_revalidate.png"; fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"SAVED {out} ({time.time()-t0:.0f}s)\nRTWREVAL_DONE", flush=True)

if __name__ == "__main__":
    main()
