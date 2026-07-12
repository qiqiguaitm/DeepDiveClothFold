#!/usr/bin/env python
"""§2.4 双锚配图(修正版): 同一批 ep 上【带双锚 vs 不带双锚】的 Viterbi 读出对比(一图内).

用户反馈: 旧图 viterbi_complete_vs_crop 讲的是"episode 切半后推理效果",不是双锚 vs 无双锚。
这里在同一条完整 ep 上画三条:
  - gray  : 不带双锚 raw Viterbi(自由端点, 只用 milestones) —— 首≠0、末登顶到 top-milestone(<1) 达不到 1
  - orange: 上面 raw 做 per-ep min-max norm01 —— 强行拉成 0→1, 但按每条自身极值拉伸(尺度不一致、掩盖真实达顶度)
  - green : 带双锚 Viterbi(起点锚→0 / 终点锚→1) —— raw 天然 0→1、端点归位、跨 ep 尺度一致
特征/milestone 复用 render_kai_online_gru 同一套(DINOv3-base img PCA128 ⊕ proprio14 → 贝叶斯GMM + 覆盖率≥0.5)。
Run: PYTHONPATH=src /home/tim/miniconda3/envs/srpo/bin/python experiments/render_anchor_vs_noanchor.py
Out: temp/anchor_vs_noanchor.png
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
rng = np.random.RandomState(0)
CAP = 1000
FPS = 30.0
CSQ = 1000
KAI = REPO / "kai0/data/Task_A/kai0_base"


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def cc(a, b):
    return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


def vstep(F, C, P, lam, anchored):
    """Viterbi milestone step path. anchored=True: 加起点/终点锚并强制首=0/末=1;False: 自由端点。"""
    if anchored:
        sC = l2(F[:3].mean(0)[None])[0]; eC = l2(F[-3:].mean(0)[None])[0]
        C2 = np.vstack([C, sC, eC]); Pp = np.concatenate([P, [0.0], [1.0]])
    else:
        C2 = C; Pp = P.copy()
    bins = np.unique(np.concatenate([[0.0], Pp, [1.0]])); nb = len(bins)
    cb = [int(np.searchsorted(bins, v)) for v in Pp]; pen = lam * np.abs(bins[:, None] - bins[None])
    de = np.linalg.norm(F[:, None] - C2[None], axis=2); em = np.full((len(F), nb), 1e3)
    for ti in range(len(Pp)):
        em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    cost = em[0].copy()
    if anchored:
        cost = np.full(nb, 1e9); cost[0] = em[0, 0]     # 起点锚 → 强制首帧 bin0(=0)
    BP = np.zeros((len(F), nb), int)
    for j in range(1, len(F)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    si = (nb - 1) if anchored else int(cost.argmin())    # 终点锚 → 强制末帧 bin=1.0
    path = np.zeros(len(F), int); path[-1] = si
    for j in range(len(F) - 2, -1, -1):
        si = BP[j + 1][si]; path[j] = si
    return bins[path]


print("加载 kai base bank...", flush=True); t0 = time.time()
d = REPO / "lmvla/crave/data/kai_dinov3base"; idx = np.load(d / "index.npz"); E = idx["E"]; FR = idx["FR"]
feat = np.zeros((len(E), 768), np.float16)
for sh in sorted(d.glob("shard_*.npz")):
    s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s else np.ones(len(g), bool); feat[g[v]] = s["feat"][v]
eps = sorted(np.unique(E).tolist())
if len(eps) > CAP:
    eps = [eps[i] for i in sorted(rng.choice(len(eps), CAP, replace=False))]
keep = np.isin(E, eps); E = E[keep]; FR = FR[keep]; feat = feat[keep]
print(f"  {len(eps)} eps {len(E)} frames; PCA...", flush=True)
pca = PCA(128, random_state=0).fit(l2(feat[rng.choice(len(feat), min(20000, len(feat)), replace=False)].astype(np.float32)))
IMG = l2((l2(feat.astype(np.float32)) - pca.mean_.astype(np.float32)) @ pca.components_.astype(np.float32).T)

print(f"  [{time.time()-t0:.0f}s] proprio...", flush=True)
POS = np.zeros((len(E), 14), np.float32)
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; fr = FR[m][np.argsort(FR[m])]
    st = np.stack(pd.read_parquet(KAI / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                  columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
    POS[o] = st[np.minimum(fr, len(st) - 1)]
SMU = POS.mean(0); SSD = POS.std(0) + 1e-6
JOINT = np.concatenate([IMG, l2((POS - SMU) / SSD)], 1).astype(np.float32)
Dd = JOINT.shape[1]; NC = len(eps)
T = np.zeros(len(E), np.float32)
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; T[o] = np.linspace(0, 1, len(o))

print(f"  [{time.time()-t0:.0f}s] BayesianGMM...", flush=True)
bg = BayesianGaussianMixture(n_components=40, covariance_type="diag", weight_concentration_prior=1e-2,
                             max_iter=120, random_state=0).fit(JOINT[rng.choice(len(JOINT), min(80000, len(JOINT)), replace=False)])
labs = bg.predict(JOINT); C = []; P = []
for k in range(40):
    m = labs == k
    if m.sum() < 20:
        continue
    if len(set(E[m].tolist())) / NC >= 0.5:
        C.append(JOINT[m].mean(0)); P.append(float(np.median(T[m])))
C = l2(np.array(C, np.float32)); P = np.array(P); lam = 16.0 * FPS / 3.0
print(f"  [{time.time()-t0:.0f}s] M={len(C)} milestones (top P={P.max():.2f}); 画 6 ep 对比...", flush=True)

# 选 6 条【双锚行为干净】的较长 ep:与时间相关高、单调、且只在末端登顶(排除早饱和退化如 ep44)
lens = {e: int((E == e).sum()) for e in eps}
cand = [e for e, _ in sorted(lens.items(), key=lambda kv: -kv[1])[:80]]
scored = []
for e in cand:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; F = JOINT[o]; t = T[o]
    anc = vstep(F, C, P, lam, anchored=True)
    hit = (np.argmax(anc >= 0.99) / len(anc)) if (anc >= 0.99).any() else 1.0   # 首次登顶位置(越靠后越好)
    mono = float(np.mean(np.diff(anc) >= -1e-6))
    penalty = 0.6 if hit < 0.82 else 0.0                                          # 早饱和惩罚
    scored.append((cc(anc, t) + 0.3 * mono - penalty, e))
pick = [e for _, e in sorted(scored, reverse=True)[:6]]
print(f"  picked clean eps {pick}", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(14, 7)); axes = axes.flatten()
draw_raw_end = []
for ax, e in zip(axes, pick):
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; F = JOINT[o]; t = T[o]
    raw = vstep(F, C, P, lam, anchored=False)
    nrm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    anc = vstep(F, C, P, lam, anchored=True)
    draw_raw_end.append(raw[-1])
    ax.plot(t, color="#e8830c", lw=1.0, ls=":", alpha=.5, label="norm time")
    ax.plot(raw, color="#888888", lw=1.6, ls="--", label=f"no anchor · raw (end {raw[-1]:.2f})")
    ax.plot(nrm, color="#d98b00", lw=1.4, alpha=.9, label="no anchor · +per-ep norm01")
    ax.plot(anc, color="#2ca02c", lw=2.1, label="double-anchor (0->1 pinned)")
    ax.set_title(f"kai ep{e} · anchor vs no-anchor", fontsize=9); ax.set_ylim(-.05, 1.06); ax.grid(alpha=.25)
axes[0].legend(fontsize=7.2, loc="lower right")
fig.suptitle(f"Double-anchor vs no-anchor Viterbi readout (same eps) · no-anchor raw tops out at ~{np.mean(draw_raw_end):.2f}<1 "
             f"-> needs per-ep norm01 (rescales each ep differently); double-anchor pins 0->1, consistent scale", fontsize=9.5)
fig.tight_layout()
outp = REPO / "lmvla/crave/temp/anchor_vs_noanchor.png"; outp.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outp, dpi=115, bbox_inches="tight")
print(f"SAVED {outp} | raw-end mean {np.mean(draw_raw_end):.2f} ({time.time()-t0:.0f}s)", flush=True)
print("ANCHOR_VS_NOANCHOR_DONE", flush=True)
