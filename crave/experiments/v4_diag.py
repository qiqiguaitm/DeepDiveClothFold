"""诊断 v4 ep01 读出为何 value 卡在 0:置信度 / 原始 argmin / viterbi / conf_hold 各层拆解。"""
import sys
from pathlib import Path
import numpy as np, av
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from crave.encoders import load_encoder
from crave.data.kai0 import crop224
from crave.utils import L2
from transition_prior_fix import build_pen, viterbi_pen
from cross_dataset_transition import emit_of, conf_hold, _smooth_block

OUTD = REPO / "temp/crave_full_dinov3h"
z = np.load(OUTD / "milestones_uniform_dinov3h.npz")
C, Pord, sk, pen = z["C"], z["Pord"], z["sk"], z["pen"]; tau_lo, tau_hi = float(z["tau_lo"]), float(z["tau_hi"]); M = len(Pord)
print(f"milestones M={M}; tau_lo={tau_lo:.3f} tau_hi={tau_hi:.3f}", flush=True)
print(f"Pord range {Pord.min():.2f}..{Pord.max():.2f}; #milestones with P<0.1: {(Pord<0.1).sum()}", flush=True)

FQC = REPO / "temp/_v4_ep01_fq.npy"
if FQC.exists():
    Fq = np.load(FQC)
else:
    enc = load_encoder("dinov3-h")
    mp4 = "/transfer-shanghai/KAI0/Task_A/base/v4/2026-06-28-v4/videos/chunk-000/observation.images.top_head/episode_000000.mp4"
    imgs = []; c = av.open(mp4)
    for j, fr in enumerate(c.decode(video=0)):
        if j % 10 == 0: imgs.append(crop224(fr.to_ndarray(format="rgb24")))
    c.close()
    Fq = L2(enc.encode_pooled(imgs)); np.save(FQC, Fq)
Fq = _smooth_block(Fq, 3); nq = len(Fq)

# 置信度:v4 帧到最近 kai0 milestone 的距离 vs kai0 自身 tau
base = np.linalg.norm(Fq[:, None] - C[None], axis=2)  # (nq, M)
dmin = base.min(1)
print(f"\nv4 dmin: min={dmin.min():.3f} p50={np.percentile(dmin,50):.3f} max={dmin.max():.3f}", flush=True)
print(f"  kai0 tau_lo(p50)={tau_lo:.3f} tau_hi(p82)={tau_hi:.3f}", flush=True)
print(f"  v4 帧中 dmin<tau_hi(算 confident)占比: {np.mean(dmin<tau_hi)*100:.0f}%  (<tau_lo: {np.mean(dmin<tau_lo)*100:.0f}%)", flush=True)

# 原始 argmin(无任何先验/门控)
am = base.argmin(1)
print(f"\n[raw argmin] Pord 序列 范围 {Pord[am].min():.2f}..{Pord[am].max():.2f}; 唯一 milestone 数 {len(np.unique(am))}", flush=True)
print(f"  前10帧 P={np.round(Pord[am[:10]],2).tolist()}", flush=True)
print(f"  后10帧 P={np.round(Pord[am[-10:]],2).tolist()}", flush=True)

# emit + viterbi(无 conf_hold)
ms_v = viterbi_pen(emit_of(Fq, C, sk, Pord, 0.8, 0.15, 1.3, 0.25), pen)
print(f"\n[emit+viterbi 无门控] Pord 范围 {Pord[ms_v].min():.2f}..{Pord[ms_v].max():.2f}; end P={Pord[ms_v[-1]]:.2f}", flush=True)

# + conf_hold
ms_h = conf_hold(ms_v, base, tau_lo, tau_hi, 0.5)
print(f"[+conf_hold] Pord 范围 {Pord[ms_h].min():.2f}..{Pord[ms_h].max():.2f}; end P={Pord[ms_h[-1]]:.2f}", flush=True)
print(f"  conf_hold 改变了 {np.mean(ms_h!=ms_v)*100:.0f}% 帧", flush=True)
