#!/usr/bin/env python
"""收口标准读出:双锚 Viterbi(无 smooth · 无 norm01)→ temp/crave_ae_labels/final/ep*.npy

milestone spec 来自 gen_final_v3.py 的 temp/crave_final_v3.npz(12 median milestone)。
本脚本只做【读出】:
  - 起点锚 = mean(全ep首3帧 img⊕pos) → value 0
  - 终点锚 = mean(全ep末3帧 img⊕pos) → value 1
  - Viterbi(λ=16, 最近质心): 强制首帧=bin0、末帧=bin1.0
  - 3Hz → 插值 30Hz native
  - ✗ 不 smooth  ✗ 不 norm01(双锚已给真实 0→1)
raw 值本身即 0→1;与监督 stage_progress_gt(kai0_advantage)corr≈0.943。
Run: PYTHONPATH=crave/src:lmwm/src:crave/experiments python crave/experiments/gen_anchored_labels.py
"""
import time, numpy as np, pandas as pd
from pathlib import Path
from crave.config import resolve_dataset
from crave.data import kai0

REPO = Path("/home/tim/workspace/deepdive_kai0")
LAM = 16.0; CSQ = 1000
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ---- milestone spec ----
s = np.load(REPO / "temp/crave_final_v3.npz")
vals = s["vals"]; Ctgt = s["Ctgt"]; pca_m = s["pca_mean"]; pca_c = s["pca_components"]; SMU = s["SMU"]; SSD = s["SSD"]
FEAT = REPO / "temp/crave_d3b_pca128/feats"
eps = sorted(int(p.stem[2:]) for p in FEAT.glob("ep*.npy"))
cfg = resolve_dataset("kai0_base"); DS = Path(cfg.root)
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E_idx, FR_idx = zf["E"], zf["FR"]

def feat3(e):
    f = np.load(FEAT / f"ep{e}.npy").astype(np.float32); fq = l2((l2(f) - pca_m) @ pca_c.T); n = len(fq)
    loc = np.where(E_idx == e)[0]; o = np.argsort(FR_idx[loc]); fr = FR_idx[loc][o][:n]
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
    F = np.concatenate([fq, l2((st[np.minimum(fr, len(st) - 1)] - SMU) / SSD)], 1)
    return F, len(st)

print(f"computing anchors over {len(eps)} eps...", flush=True)
ss = []; ee = []
for e in eps:
    F, _ = feat3(e); ss.append(F[:3].mean(0)); ee.append(F[-3:].mean(0))
sC = l2(np.mean(ss, 0)[None])[0]; eC = l2(np.mean(ee, 0)[None])[0]
Ct2 = np.vstack([Ctgt, sC, eC]).astype(np.float32)          # 12 milestone + start + end
Pord = np.concatenate([vals, [0.0], [1.0]])
bins = np.unique(np.concatenate([[0.0], Pord, [1.0]])); nb = len(bins)
cb = [int(np.searchsorted(bins, v)) for v in Pord]; pen = LAM * np.abs(bins[:, None] - bins[None])

def anchored_viterbi(Fq):
    de = np.linalg.norm(Fq[:, None] - Ct2[None], axis=2); em = np.full((len(Fq), nb), 1e3)
    for ti in range(len(Pord)): em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)   # 强制首帧=bin0
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    ss = nb - 1; path = np.zeros(len(Fq), int); path[-1] = ss                        # 强制末帧=bin(1.0)
    for j in range(len(Fq) - 2, -1, -1): ss = BP[j + 1][ss]; path[j] = ss
    return bins[path]

LAB = REPO / "temp/crave_ae_labels/final"; LAB.mkdir(parents=True, exist_ok=True)
t0 = time.time(); nd = 0; dd = []; ends = []; monos = []
for e in eps:
    F, N = feat3(e); v = anchored_viterbi(F)
    v30 = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(v)), v).astype(np.float32)   # 3Hz→30Hz, 无 smooth/norm01
    np.save(LAB / f"ep{e}.npy", v30); nd += 1
    dd.append(float((np.maximum.accumulate(v30) - v30).max())); ends.append(float(v30[-5:].mean()))
    monos.append(float((np.diff(v30) >= -1e-6).mean()))
    if nd % 1000 == 0: print(f"  [{nd}/{len(eps)}] {(time.time()-t0)/60:.1f}min", flush=True)
print(f"DONE {nd} eps · dd_mean={np.mean(dd):.3f} 末值median={np.median(ends):.3f} "
      f"末值>0.9={np.mean(np.array(ends)>0.9):.0%} 单调={np.mean(monos):.1%} ({(time.time()-t0)/60:.1f}min)", flush=True)
