#!/usr/bin/env python
"""r 场三读法跨编码器空间验证: DINOv3 vs SigLIP-2(后续加 So400m/pi05-tower)。
问题: CRAVE/LMWM 全部结论只在 DINOv3 空间验证过; 若信号要喂 pi05(SigLIP 系),须证 r 场在该空间不退化。
指标(对齐 roadmap V0/V1/V5(a) 协议, kai0 110 ep 对齐特征):
  ① 非退化: r 场 std / range(V0)
  ② r-谷分割: 每 ep 边界数分布 + 跨-ep 稳定 recall 真 vs 随机(V1, 边界归一化位置, 容差0.05)
  ③ r-脊目标质量: persist=cos(g_t, g_下一脊) vs cos(g_t, g_{t+7}), 及脊视界(V5(a): 脊应"更远但可预测")
用法: python rfield_space_check.py dino|siglip2|<dir名>
"""
import glob, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ENC = sys.argv[1]
BASE = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/kai0_aligned_urvc"
THR = 0.03

featL = []
for f in sorted(glob.glob(f"{BASE}/{ENC}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4])):
    p = np.load(f)["pooled"].astype(np.float32)
    p /= (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    featL.append(p)
ne = len(featL)
F = np.concatenate(featL); ep = np.concatenate([np.full(len(p), i) for i, p in enumerate(featL)])
offs = np.cumsum([0] + [len(p) for p in featL])

# r 场(per-ep 1NN + median σ, 同 p1_libero_rvalley_pairs)
D = np.sqrt(np.maximum(2 - 2 * (F @ F.T), 0))          # L2 归一后欧氏
dmin = np.full((len(F), ne), 1e9, np.float32)
for j in range(ne): dmin[:, j] = D[:, ep == j].min(1)
other = ep[:, None] != np.arange(ne)[None]
sig = np.median(dmin[other])
r = (np.exp(-dmin**2 / (2 * sig * sig)) * other).sum(1) / (ne - 1)

# ① 非退化
stds = [r[offs[i]:offs[i+1]].std() for i in range(ne)]
print(f"[{ENC}] ① r 场: σ_kernel={sig:.4f}  per-ep std中位={np.median(stds):.3f}  range=[{r.min():.2f},{r.max():.2f}]  全局std={r.std():.3f}")

# ② r-谷分割 + 跨-ep 稳定
bounds = []
for i in range(ne):
    rr = r[offs[i]:offs[i+1]]; n = len(rr)
    v, _ = find_peaks(-gaussian_filter1d(rr, 1.4), prominence=THR, distance=max(2, n // 12))
    bounds.append(v / max(n - 1, 1))
nb = [len(b) for b in bounds]
rng = np.random.default_rng(0)
def recall(bl, ref_bl, k_ref=5, tol=0.03):
    """每 ep 边界在【5 条随机其他 ep】的边界集内 tol 命中率(防 110ep 全集饱和)"""
    rec = []
    for i in range(ne):
        if len(bl[i]) == 0: continue
        js = rng.choice([j for j in range(ne) if j != i], size=k_ref, replace=False)
        oth = np.concatenate([ref_bl[j] for j in js if len(ref_bl[j])] or [np.array([])])
        if len(oth) == 0: continue
        rec.append(np.mean([np.abs(oth - x).min() < tol for x in bl[i]]))
    return np.median(rec) if rec else float("nan")
rand_b = [rng.uniform(0.05, 0.95, size=len(b)) for b in bounds]
print(f"[{ENC}] ② r-谷: 边界数 med={np.median(nb):.0f} 分布={np.bincount(nb, minlength=6)[:6]}  跨-ep recall(5ep参照,tol0.03) 真={recall(bounds, bounds):.3f} vs 随机={recall(rand_b, bounds):.3f}")

# ③ 脊目标质量
per, per7, hors = [], [], []
for i in range(ne):
    rr = r[offs[i]:offs[i+1]]; n = len(rr); g = featL[i]
    v, _ = find_peaks(-gaussian_filter1d(rr, 1.4), prominence=THR, distance=max(2, n // 12))
    seg = [0] + list(v) + [n]
    ridge = [a + int(np.argmax(rr[a:b])) for a, b in zip(seg[:-1], seg[1:])]
    for t in range(n - 1):
        k = np.searchsorted(np.array(seg[1:-1]), t, side="right")   # 当前段号
        tgt = ridge[k + 1] if k + 1 < len(ridge) else n - 1          # 下一段脊, 末段锚末帧
        if tgt <= t: tgt = n - 1
        if tgt <= t: continue
        per.append(float(g[t] @ g[tgt])); hors.append(tgt - t)
        per7.append(float(g[t] @ g[min(t + 7, n - 1)]))
print(f"[{ENC}] ③ 脊目标: persist(脊)={np.mean(per):.4f} vs persist(t+7)={np.mean(per7):.4f}  Δ={np.mean(per7)-np.mean(per):+.4f}  脊视界={np.mean(hors):.1f}帧(stride20)")
print(f"    判读: ①std>0.05 不退化 ②真>随机 ③脊更远(persist更低)但结构保持 → r 场在 {ENC} 空间成立")
