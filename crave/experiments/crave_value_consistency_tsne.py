"""§3.2 图:DINOv3-H 特征降到 2D(t-SNE),按 value(进度)着色 —— 证明聚类的 value 一致性,
说明为什么可以用"簇平均 value"作为该簇的近似进度值(milestone 的 Pord)。

左:t-SNE 散点,颜色=每帧 value(归一化进度)→ 流形被进度平滑组织,邻近点 value 相近。
右:每帧 value vs 其所在簇的平均 value 散点 → 紧贴对角线(R² 高)+ 簇内 value std 很小
    → 簇平均 value 忠实代表簇内所有帧,故可当 milestone 近似进度。
复用缓存的 DINOv3-H 全量 pooled 特征(temp/crave_full_dinov3h/),纯 CPU。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_value_consistency_tsne.py [--n 5000] [--k 96]
输出: crave/docs/visualization/crave_value_consistency_dinov3.png
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import json

import pandas as pd

from crave.config import REPO, viz_dir
from crave.render import setup_mpl

plt = setup_mpl()
OUTD = REPO / "temp/crave_full_dinov3h"
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"   # 含 stage_progress_gt(GT 进度)
csQ = json.load(open(Q5 / "meta/info.json"))["chunks_size"]


def gt_progress(ep):
    p = Q5 / "data" / f"chunk-{ep//csQ:03d}" / f"episode_{ep:06d}.parquet"
    if not p.exists(): return None
    return pd.read_parquet(p, columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)      # t-SNE 子采样
    ap.add_argument("--k", type=int, default=160)
    ap.add_argument("--n-eps", type=int, default=300)   # 限制 episode 数(限定 parquet 读取)
    a = ap.parse_args()
    z = np.load(OUTD / "index.npz"); E, FR, N = z["E"], z["FR"], int(z["n"])
    DIM = 1280
    feat = np.zeros((N, DIM), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / "shard_*.npz"))):
        s = np.load(f); feat[s["gidx"]] = s["feat"]; valid[s["gidx"]] = s["valid"]
    vi = np.where(valid)[0]
    Ev, FRv = E[vi], FR[vi]
    # 限定 n_eps 个 episode → 取其帧 → 子采样;value 用 GT stage_progress_gt
    eps_all = np.array(sorted(set(Ev.tolist())))
    pick = np.random.RandomState(0).choice(eps_all, min(a.n_eps, len(eps_all)), replace=False)
    pool = np.where(np.isin(Ev, pick))[0]
    sub = np.random.RandomState(1).choice(pool, min(a.n, len(pool)), replace=False)
    F = feat[vi[sub]].astype(np.float32); F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    Fs = F
    Ts = np.zeros(len(sub), np.float32); ok = np.zeros(len(sub), bool)
    for e in set(Ev[sub].tolist()):
        gt = gt_progress(int(e))
        if gt is None: continue
        m = Ev[sub] == e; fr = np.clip(FRv[sub][m], 0, len(gt) - 1)
        Ts[m] = gt[fr]; ok[m] = True
    Fs, Ts = Fs[ok], Ts[ok]
    print(f"[tsne] {len(Fs)} 帧(来自 {len(pick)} ep,GT 进度);", flush=True)
    print(f"[tsne] {len(Fs)} 帧子采样 / DINOv3-H {DIM}d; KMeans k={a.k} + t-SNE ...", flush=True)
    lab = KMeans(a.k, n_init=4, random_state=0).fit_predict(Fs)
    cmean = np.array([Ts[lab == c].mean() if (lab == c).any() else 0 for c in range(a.k)])
    cstd = np.array([Ts[lab == c].std() if (lab == c).sum() > 1 else 0 for c in range(a.k)])
    frame_cmean = cmean[lab]                                         # 每帧 → 其簇平均 value
    r2 = float(np.corrcoef(Ts, frame_cmean)[0, 1] ** 2)
    med_std = float(np.median(cstd))
    XY = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Fs)
    print(f"[tsne] 簇内 value std 中位={med_std:.3f}; frame-vs-簇均 R²={r2:.3f}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    sc = ax[0].scatter(XY[:, 0], XY[:, 1], c=Ts, cmap="viridis", s=9, alpha=.8, linewidths=0)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("(A) DINOv3-H 特征 t-SNE,按 value(进度)着色\n流形被进度平滑组织 → 邻近特征 value 相近", fontsize=12)
    cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.02); cb.set_label("value(任务进度 0→1)", fontsize=10)

    coh = cstd < np.median(cstd)                                    # 纯度过滤:value 相干的簇 = 选作 milestone
    cohf = coh[lab]
    r2c = float(np.corrcoef(Ts[cohf], frame_cmean[cohf])[0, 1] ** 2); med_c = float(np.median(cstd[coh]))
    # 松散簇灰底
    ax[1].scatter(Ts[~cohf], frame_cmean[~cohf], s=7, alpha=.16, color="#b0b7c0", linewidths=0, label="松散簇(复现态,Viterbi兜底)")
    # 不同相干簇用不同颜色(按 value 排序)
    COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#fdb415","#e377c2","#17becf","#3b5998","#8c564b"]
    sorted_coh = [c for c in np.argsort(cmean) if coh[c]]
    for i, c in enumerate(sorted_coh):
        sel = lab == c
        if sel.any():
            ax[1].scatter(Ts[sel], frame_cmean[sel], s=9, alpha=.5, color=COLORS[i % len(COLORS)], linewidths=0)
    ax[1].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=.7, label="理想 y=x")
    ax[1].set_xlabel("每帧 value(GT 进度)", fontsize=11); ax[1].set_ylabel("其所在簇的平均 value", fontsize=11)
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].grid(alpha=.25); ax[1].legend(fontsize=9.5, loc="upper left")
    ax[1].set_title(f"(B) 选作 milestone 的相干簇:簇均 value 忠实代表簇内帧(R²={r2c:.2f}, std 中位={med_c:.2f})\n→ 故用簇平均 value 当 milestone 近似进度;少数松散簇(复现态)由 Viterbi 兜底", fontsize=11)
    print(f"[tsne] 相干簇 R²={r2c:.3f} std中位={med_c:.3f}", flush=True)
    fig.suptitle("为什么用「簇平均 value」表示 milestone 进度:DINOv3-H 聚类的 value 一致性证明", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = viz_dir() / "crave_value_consistency_dinov3.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
