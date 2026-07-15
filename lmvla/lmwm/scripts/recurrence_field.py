#!/usr/bin/env python
"""逐帧跨-episode 复现密度场 r(o) —— 无聚类/无阈值/无K, 验证普适非退化。
r(o_t) = 1/(N_ep-1) * Σ_{j != ep(t)} exp(-dmin(o_t, E_j)^2 / 2σ^2)
  dmin(o_t, E_j) = 到 episode j 最近帧的距离; σ = 所有跨-ep dmin 的中位(尺度无关带宽)。
= "有多少条别的 demo 在 o_t 附近也出现过"(数不同 episode, 免疫本集 dwell)。
出图: task0(清晰) / task6(弥散·聚类塌成1的那个) / kai0(视觉多样) 各 r-vs-time + heatmap。
Run: OMP_NUM_THREADS=8 srpo python recurrence_field.py
Out: lmwm/docs/assets/recurrence_field.png
"""
import os, glob, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = "/vePFS/tim/workspace/deepdive_kai0"
ROOT = f"{REPO}/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
FEAT = f"{REPO}/lmvla/lmwm/data/libero_dinov3base"
KAI  = f"{REPO}/lmvla/crave/data/kai_dinov3base"

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def recurrence(gd):
    """gd: {ep: [Ni,768]}. 返回 {ep: r[Ni]} + 全局 stats。"""
    eps = list(gd); F = l2(np.concatenate([gd[e] for e in eps]).astype(np.float32))
    ep = np.concatenate([np.full(len(gd[e]), i) for i, e in enumerate(eps)])
    ne = len(eps); M = len(F)
    D = cdist(F, F)                                              # [M,M] euclidean on unit-vec
    dmin = np.full((M, ne), 1e9, np.float32)
    for j in range(ne):
        c = np.where(ep == j)[0]; dmin[:, j] = D[:, c].min(1)
    other = ep[:, None] != np.arange(ne)[None]                  # [M,ne] 排除本集
    sig = np.median(dmin[other]); K = np.exp(-dmin**2 / (2*sig*sig)); K[~other] = 0.0
    r = K.sum(1) / (ne - 1)
    out = {}; off = 0
    for e in eps:
        n = len(gd[e]); out[e] = r[off:off+n]; off += n
    return out, r, sig

def heat(gd, rr, nb=40):
    eps = sorted(gd, key=lambda e: -len(gd[e]))
    H = np.zeros((len(eps), nb))
    for i, e in enumerate(eps):
        n = len(gd[e]); t = np.linspace(0, 1, n)
        H[i] = np.interp(np.linspace(0, 1, nb), t, rr[e])
    return H

def load_libero_task(frag):
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    tdf = pd.read_parquet(f"{ROOT}/meta/tasks.parquet")
    name2idx = {n: int(r["task_index"]) for n, r in tdf.iterrows()}
    hit = [v for n, v in name2idx.items() if frag.lower() in n.lower()]
    tid = hit[0]
    meta = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in dpar])
    ep2task = meta.groupby("episode_index")["task_index"].first().to_dict()
    eps = [e for e, t in ep2task.items() if t == tid and os.path.exists(f"{FEAT}/ep{e}.npz")][:40]
    return {e: np.load(f"{FEAT}/ep{e}.npz")["grid"].astype(np.float32).mean(1) for e in eps}

def load_kai0(cap=20, stride=4):
    idx = np.load(f"{KAI}/index.npz"); E = idx["E"]; FR = idx["FR"]
    feat = np.zeros((len(E), 768), np.float16)
    for sh in sorted(glob.glob(f"{KAI}/shard_*.npz")):
        s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s.files else np.ones(len(g), bool); feat[g[v]] = s["feat"][v]
    eps = sorted(np.unique(E).tolist())[:cap]; gd = {}
    for e in eps:
        m = np.where(E == e)[0]; o = m[np.argsort(FR[m])][::stride]; gd[e] = feat[o].astype(np.float32)
    return gd

def main():
    t0 = time.time()
    print("loading...", flush=True)
    tasks = [
        ("LIBERO task0 (clear: soup+sauce in basket)", load_libero_task("alphabet soup and the tomato sauce"), None),
        ("LIBERO task6 (diffuse: mug+chocolate, clustering collapsed M=1)", load_libero_task("white mug on the plate and put the chocolate"), (0.4, 1.0)),
        ("kai0 (high visual change: fold)", load_kai0(), None),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), gridspec_kw={"width_ratios": [1.4, 1]})
    for row, (name, gd, tailshade) in enumerate(tasks):
        rr, rall, sig = recurrence(gd)
        eps = list(gd)
        print(f"[{name}] {len(eps)}ep {len(rall)}f | r mean={rall.mean():.3f} std={rall.std():.3f} "
              f"min={rall.min():.3f} max={rall.max():.3f} | low(r<0.3)={np.mean(rall<0.3)*100:.0f}% ({time.time()-t0:.0f}s)", flush=True)
        ax = axes[row, 0]
        # 各 ep 的 r-vs-time (浅) + 时间对齐中位曲线 (粗)
        nb = 40; stack = []
        for e in eps:
            t = np.linspace(0, 1, len(gd[e])); rb = np.interp(np.linspace(0, 1, nb), t, rr[e]); stack.append(rb)
            if eps.index(e) < 12: ax.plot(np.linspace(0, 1, nb), rb, color="#7c3aed", lw=0.7, alpha=0.25)
        med = np.median(np.array(stack), 0)
        ax.plot(np.linspace(0, 1, nb), med, color="#7c3aed", lw=2.6, label="median r across episodes")
        if tailshade: ax.axvspan(*tailshade, color="#ef4444", alpha=0.08, label="untrained diffuse tail (RESULTS)")
        ax.set_ylim(0, max(0.6, rall.max()*1.05)); ax.set_xlim(0, 1); ax.grid(alpha=0.25)
        ax.set_xlabel("normalized time"); ax.set_ylabel("recurrence r(o)")
        ax.set_title(f"{name}\nr std={rall.std():.3f} (>0 = graded, NOT collapsed)  ·  low r<0.3 = {np.mean(rall<0.3)*100:.0f}%", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper center")
        # heatmap
        H = heat(gd, rr); im = axes[row, 1].imshow(H, aspect="auto", cmap="magma", extent=[0, 1, len(eps), 0], vmin=0)
        axes[row, 1].set_xlabel("normalized time"); axes[row, 1].set_ylabel("episode"); axes[row, 1].set_title("r heatmap (vertical bands = cross-ep recurrence ridges)", fontsize=9)
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)
    fig.suptitle("Per-frame cross-episode RECURRENCE FIELD r(o) — clustering-free, universal (works where BGMM/coverage collapsed)", fontsize=12)
    fig.tight_layout()
    out = f"{REPO}/lmvla/lmwm/docs/assets/recurrence_field.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight"); print(f"SAVED {out} ({time.time()-t0:.0f}s)", flush=True)
    print("RECURRENCE_DONE", flush=True)

if __name__ == "__main__":
    main()
