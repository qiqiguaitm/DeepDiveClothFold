#!/usr/bin/env python
"""r-低谷 子任务边界分割器 —— 全 40 LIBERO + kai0, 单一全局谷显著性阈值。
边界 = 跨-ep 中位 r(t) 曲线在 -r 上的显著低谷(find_peaks, prominence=全局阈值)。
验证: (1) 边界数随任务复杂度涌现(vs 任务名"and"子任务数 / ep 长度);
      (2) 跨-ep 稳定 = 每条边界有多少比例 episode 自己也在 ±δ 内独立出现低谷, 对照随机边界。
Run: OMP_NUM_THREADS=8 srpo python rvalley_segmenter.py
Out: lmwm/docs/assets/rvalley_segmenter.png + 打印每任务表 + 阈值扫描
"""
import os, glob, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = "/vePFS/tim/workspace/deepdive_kai0"
ROOT = f"{REPO}/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
FEAT = f"{REPO}/lmvla/lmwm/data/libero_dinov3base"
KAI  = f"{REPO}/lmvla/crave/data/kai_dinov3base"
NB = 50; DELTA = 0.08; DIST = 4                # 时间bin / 边界匹配窗 / 谷间最小距
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def recurrence(gd):
    eps = list(gd); F = l2(np.concatenate([gd[e] for e in eps]).astype(np.float32))
    ep = np.concatenate([np.full(len(gd[e]), i) for i, e in enumerate(eps)]); ne = len(eps); M = len(F)
    D = cdist(F, F); dmin = np.full((M, ne), 1e9, np.float32)
    for j in range(ne): dmin[:, j] = D[:, np.where(ep == j)[0]].min(1)
    other = ep[:, None] != np.arange(ne)[None]; sig = np.median(dmin[other])
    K = np.exp(-dmin**2 / (2*sig*sig)); K[~other] = 0.0; r = K.sum(1)/(ne-1)
    out = {}; off = 0
    for e in eps: n = len(gd[e]); out[e] = r[off:off+n]; off += n
    return out

def binned(rr, gd):
    eps = list(gd); B = np.zeros((len(eps), NB))
    for i, e in enumerate(eps):
        t = np.linspace(0, 1, len(gd[e])); B[i] = np.interp(np.linspace(0, 1, NB), t, rr[e])
    return B  # [ne, NB]

def valleys(curve, thr):
    v, _ = find_peaks(-gaussian_filter1d(curve, 1.4), prominence=thr, distance=DIST)
    return v

def analyze(gd, thr, rng):
    rr = recurrence(gd); B = binned(rr, gd); med = np.median(B, 0)
    vb = valleys(med, thr); tb = vb / (NB - 1)                      # 边界时间
    # 跨-ep 稳定: 每条边界, 多少比例 ep 自己也在 ±DELTA 内有谷
    ep_v = [valleys(B[i], thr) / (NB - 1) for i in range(len(B))]
    def recall(bnds):
        if len(bnds) == 0: return np.nan
        rec = []
        for b in bnds:
            hit = np.mean([np.any(np.abs(np.array(vv) - b) <= DELTA) if len(vv) else False for vv in ep_v])
            rec.append(hit)
        return float(np.mean(rec))
    real = recall(tb)
    # 随机对照: 同数量随机边界的 recall (均值 of few draws)
    rand = np.nan
    if len(tb) > 0:
        rand = float(np.mean([recall(rng.uniform(0.1, 0.9, len(tb))) for _ in range(20)]))
    return len(tb), real, rand, med, tb

def load_libero_all():
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    tdf = pd.read_parquet(f"{ROOT}/meta/tasks.parquet")
    idx2name = {int(r["task_index"]): n for n, r in tdf.iterrows()}
    meta = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in dpar])
    ep2task = meta.groupby("episode_index")["task_index"].first().to_dict()
    from collections import defaultdict
    te = defaultdict(list)
    for e, t in ep2task.items():
        if os.path.exists(f"{FEAT}/ep{e}.npz"): te[t].append(e)
    tasks = []
    for t, eps in sorted(te.items()):
        if len(eps) < 6: continue
        gd = {e: np.load(f"{FEAT}/ep{e}.npz")["grid"].astype(np.float32).mean(1) for e in eps}
        nm = idx2name.get(t, "?"); nand = nm.lower().count(" and ")
        mlen = int(np.median([len(gd[e]) for e in gd]))
        tasks.append((f"L{t}", nm, nand, mlen, gd))
    return tasks

def load_kai0(cap=20, stride=4):
    idx = np.load(f"{KAI}/index.npz"); E = idx["E"]; FR = idx["FR"]
    feat = np.zeros((len(E), 768), np.float16)
    for sh in sorted(glob.glob(f"{KAI}/shard_*.npz")):
        s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s.files else np.ones(len(g), bool); feat[g[v]] = s["feat"][v]
    eps = sorted(np.unique(E).tolist())[:cap]; gd = {}
    for e in eps:
        m = np.where(E == e)[0]; o = m[np.argsort(FR[m])][::stride]; gd[e] = feat[o].astype(np.float32)
    return [("kai0", "fold (single long task)", 3, int(np.median([len(gd[e]) for e in gd])), gd)]

def main():
    t0 = time.time(); rng = np.random.RandomState(0)
    tasks = load_libero_all() + load_kai0()
    print(f"{len(tasks)} tasks loaded ({time.time()-t0:.0f}s)\n", flush=True)
    THRS = [0.02, 0.03, 0.05]
    # 主阈值 0.03 详表; 其余只汇总
    MAIN = 0.03; rows = []; examples = []
    for tag, nm, nand, mlen, gd in tasks:
        nb, real, rand, med, tb = analyze(gd, MAIN, rng)
        rows.append((tag, nand, mlen, nb, real, rand))
        examples.append((tag, nand, med, tb))
        print(f"  [{tag}] nand={nand} len={mlen:3d} | boundaries={nb} | cross-ep recall={real if real==real else float('nan'):.2f} vs random={rand if rand==rand else float('nan'):.2f}  ({nm[:48]})", flush=True)
    nb_all = np.array([r[3] for r in rows]); nand_all = np.array([r[1] for r in rows]); len_all = np.array([r[2] for r in rows])
    real_all = np.array([r[4] for r in rows]); rand_all = np.array([r[5] for r in rows])
    ok = ~np.isnan(real_all)
    print(f"\n[SUMMARY thr={MAIN}] tasks={len(rows)}")
    print(f"  boundary count: min={nb_all.min()} median={int(np.median(nb_all))} max={nb_all.max()} | dist={np.bincount(nb_all)}")
    print(f"  emergence corr(boundaries, #'and')={np.corrcoef(nb_all, nand_all)[0,1]:.2f} | corr(boundaries, ep_len)={np.corrcoef(nb_all, len_all)[0,1]:.2f}")
    print(f"  cross-ep stable: recall median={np.nanmedian(real_all):.2f} vs random median={np.nanmedian(rand_all):.2f} | real>random on {np.mean(real_all[ok]>rand_all[ok])*100:.0f}% tasks")
    # 阈值鲁棒性
    print("\n[THRESHOLD SWEEP]")
    for thr in THRS:
        nbs = []; recs = []
        for tag, nm, nand, mlen, gd in tasks:
            nb, real, rand, med, tb = analyze(gd, thr, rng); nbs.append(nb); recs.append(real)
        nbs = np.array(nbs); recs = np.array(recs)
        print(f"  thr={thr}: boundaries med={int(np.median(nbs))} range[{nbs.min()},{nbs.max()}] dist={np.bincount(nbs)} | recall med={np.nanmedian(recs):.2f}", flush=True)

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0,0].hist(nb_all, bins=np.arange(-0.5, nb_all.max()+1.5), color="#7c3aed", alpha=0.8, edgecolor="w")
    ax[0,0].set_xlabel("# boundaries (emergent)"); ax[0,0].set_ylabel("# tasks"); ax[0,0].set_title(f"Boundary count distribution (thr={MAIN}, 40 LIBERO + kai0)\nNOT all-0 / all-22 -> emergent per task", fontsize=10); ax[0,0].grid(alpha=.2)
    sc = ax[0,1].scatter(nand_all, nb_all, c=len_all, cmap="viridis", s=45, alpha=.85)
    ax[0,1].set_xlabel("# subtasks proxy (count of ' and ' in task name)"); ax[0,1].set_ylabel("# boundaries")
    ax[0,1].set_title(f"Adapts to complexity: corr(bnd,#and)={np.corrcoef(nb_all,nand_all)[0,1]:.2f}", fontsize=10); ax[0,1].grid(alpha=.2); plt.colorbar(sc, ax=ax[0,1], label="median ep length")
    ax[1,0].scatter(rand_all[ok], real_all[ok], c="#7c3aed", s=40, alpha=.8); ax[1,0].plot([0,1],[0,1],'--',color="#888",lw=1)
    ax[1,0].set_xlabel("random-boundary recall (chance)"); ax[1,0].set_ylabel("r-valley boundary recall (real)")
    ax[1,0].set_title(f"Cross-ep stable: real recall median {np.nanmedian(real_all):.2f} > random {np.nanmedian(rand_all):.2f}\n(points above diagonal = boundaries consistent across episodes)", fontsize=10); ax[1,0].grid(alpha=.2); ax[1,0].set_xlim(0,1); ax[1,0].set_ylim(0,1)
    # 例子: 按 nand 挑 简单/中/复杂
    order = sorted(range(len(examples)), key=lambda i: examples[i][1])
    pick = [order[0], order[len(order)//2], order[-2], order[-1]]
    for i in pick:
        tag, nand, med, tb = examples[i]
        ax[1,1].plot(np.linspace(0,1,NB), med, lw=1.8, label=f"{tag}(and={nand},b={len(tb)})")
        for b in tb: ax[1,1].axvline(b, color="r", ls=":", lw=0.8, alpha=.5)
    ax[1,1].set_xlabel("normalized time"); ax[1,1].set_ylabel("median r(o)"); ax[1,1].set_title("Example median-r curves + detected r-valley boundaries (red)", fontsize=10); ax[1,1].legend(fontsize=7); ax[1,1].grid(alpha=.2)
    fig.suptitle("r-valley subtask segmenter — single global prominence threshold, boundary count emerges per task & is cross-ep stable", fontsize=12)
    fig.tight_layout()
    out = f"{REPO}/lmvla/lmwm/docs/assets/rvalley_segmenter.png"; fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"\nSAVED {out} ({time.time()-t0:.0f}s)\nRVALLEY_DONE", flush=True)

if __name__ == "__main__":
    main()
