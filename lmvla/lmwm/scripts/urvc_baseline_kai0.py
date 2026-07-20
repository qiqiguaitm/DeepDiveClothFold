#!/usr/bin/env python
"""UR-VC(arXiv 2607.12892)复现基线 @ kai0, 对打 CRAVE v1(corr 0.943 vs stage_progress_gt).
方法(论文式4-6): 帧 i → 每条其他 ep 在时间带 |g_j-g_i|<=tau 内取 cos 最相似 1 帧(1-NN/ep),
⚠️ 已弃用(2026-07-20): dino_sub20 特征与本数据错对齐, 数字作废; 用 urvc_aligned_analysis.py + kai0_aligned_urvc 特征。
cos>=rho 才算匹配; ĝ_i = 匹配帧归一化时间标签的平均。免训练。
偏离声明: 编码器用 DINOv3-base pooled(本地现成, 论文用 SigLIP-2)→ 这是机制复现非严格复现;
rho=0.90 是 SigLIP 尺度, DINOv3 余弦偏高 → 同时报 rho∈{0.90,0.95} 与 无阈值 三档。
数据: lmwm/data/dino_sub20/ep*.npz(stride-20 grid[N,256,768]→pooled), GT=kai0_advantage parquet stage_progress_gt。
指标(对齐 v1 收口口径): per-ep Pearson corr(中位), 单调率(相邻diff>=0 比例), 末值均值; 对照=原始时间标签 g。
"""
import glob, os, sys
import numpy as np, pandas as pd

FEAT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/dino_sub20"
GT   = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_advantage/data/chunk-000"
STRIDE = 20
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 150   # 论文用 150 episodes
TAU, RHOS = 0.30, [None, 0.90, 0.95]

rng = np.random.default_rng(0)
files = sorted(glob.glob(f"{FEAT}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
sel = sorted(rng.choice(len(files), min(N_EP, len(files)), replace=False))

pooled, gts, gnorm = [], [], []   # per-ep: [n,768] L2, gt[n], 时间标签[n]
for k in sel:
    epi = int(os.path.basename(files[k])[2:-4])
    pq = f"{GT}/episode_{epi:06d}.parquet"
    if not os.path.exists(pq): continue
    g = np.load(files[k])["grid"].astype(np.float32).mean(1)          # [n,768] pooled
    g /= (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    df = pd.read_parquet(pq, columns=["stage_progress_gt"])
    idx = np.minimum(np.arange(len(g)) * STRIDE, len(df) - 1)          # sub20 帧 → 30Hz 行
    n = len(g)
    pooled.append(g); gts.append(df["stage_progress_gt"].values[idx].astype(np.float32))
    gnorm.append(np.arange(n, dtype=np.float32) / max(n - 1, 1))
ne = len(pooled)
print(f"eps={ne}  frames={sum(len(p) for p in pooled)}", flush=True)

# 逐 ep 对: 余弦 + 时间带掩码 → 每 (帧i, ep e') 的 best match (cos, 时间标签)
best_cos = [np.full((len(p), ne), -1, np.float32) for p in pooled]     # [n_i, ne]
best_g   = [np.zeros((len(p), ne), np.float32) for p in pooled]
for a in range(ne):
    for b in range(ne):
        if a == b: continue
        C = pooled[a] @ pooled[b].T                                    # [na,nb] cos
        M = np.abs(gnorm[a][:, None] - gnorm[b][None, :]) <= TAU       # 时间带
        C = np.where(M, C, -2.0)
        j = C.argmax(1)
        best_cos[a][:, b] = C[np.arange(len(j)), j]
        best_g[a][:, b] = gnorm[b][j]

def metrics(pred, gt):
    cs, mono, ends = [], [], []
    for p, t in zip(pred, gt):
        if np.std(p) > 1e-6 and np.std(t) > 1e-6:
            cs.append(np.corrcoef(p, t)[0, 1])
        mono.append((np.diff(p) >= -1e-6).mean()); ends.append(p[-1])
    return np.median(cs), np.mean(mono), np.mean(ends)

print(f"{'方法':<26} {'corr(med)':>9} {'mono':>6} {'end':>6} {'覆盖率':>7}")
c, m, e = metrics(gnorm, gts)
print(f"{'时间标签 g(论文式1)':<26} {c:>9.3f} {m:>6.3f} {e:>6.3f} {'—':>7}")
for rho in RHOS:
    preds, covs = [], []
    for a in range(ne):
        ok = best_cos[a] >= (rho if rho is not None else -1.5)
        ok[:, a] = False
        cnt = ok.sum(1)
        gh = np.where(cnt > 0, (best_g[a] * ok).sum(1) / np.maximum(cnt, 1), gnorm[a])  # 无匹配回退时间标签
        preds.append(gh.astype(np.float32)); covs.append((cnt > 0).mean())
    c, m, e = metrics(preds, gts)
    tag = f"UR-VC rho={rho if rho is not None else '无'}"
    print(f"{tag:<26} {c:>9.3f} {m:>6.3f} {e:>6.3f} {np.mean(covs):>6.1%}")
print("\n参照: CRAVE v1 收口(BGMM+双锚Viterbi, 3055ep 30Hz) corr 0.943 / mono 0.981 / end 0.999")
print("注意: 本基线 150ep/stride20/DINOv3, 口径不完全同 → 结论只看相对量级与'能否超时间标签'.")
