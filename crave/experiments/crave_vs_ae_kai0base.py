#!/usr/bin/env python
"""CRAVE vs AWBC pi0-AE 等价对比 —— 同域(kai0)同数据(base+dagger 挖矿)同 episode。
公平性: AE 训练于 kai0(base+dagger), 故 CRAVE 也在 kai0_base+kai0_dagger 上重新挖 milestone;
测试 episode 取自 kai0_base 的一条长 episode(ep2302, 2960f≈99s, 便于观察)。
  - CRAVE 特征: temp/crave_kai0bd/feat_cache (base offset0 + dagger offset100000, lerobot_v2_extract)
  - AE 值: advantage_q5 ep2302 absolute_value (AE 对 kai0_base 的 Stage-2 输出, 1:1 同 episode)
CRAVE 核心复用 hdf5_v24_eval.build_model(配方逐字一致)。
输出: crave/docs/visualization/crave_vs_ae_kai0base.png + 数组 npz

Rewrite onto the `crave` library:
  - hdf5_v24_eval.build_model (KMeans96 fixed-select + time-order + hardbin Viterbi) is exactly
    crave.value.DiscreteValue(select="fixed", order="time") over crave.value.FeatureSpace; its
    loadep is crave.data.cache.loadep. The mining cache temp/crave_kai0bd/feat_cache is a
    one-off (base⊕dagger) cache, not a registered dataset, so FeatureSpace takes its path.
    legacy `value(...,ret_lab=True)` → DiscreteValue.value(...,ret_lab=True) (v3, lab, marg).
  - smooth_monotone/advantage from crave.utils; REPO from crave.config.
  - OUT OF SCOPE (supervised AE): advantage_q5 absolute_value parquet + the kai0_base last-frame
    mp4 grab (plain pandas/av reads, not crave).
"""
import json
import numpy as np, pandas as pd, cv2, av
from pathlib import Path
from scipy.stats import pearsonr, kendalltau

from crave.config import REPO, viz_dir
from crave.data import list_cache_eps, loadep
from crave.render import setup_mpl
from crave.utils import smooth_monotone, advantage
from crave.value import FeatureSpace, DiscreteValue

plt = setup_mpl()

FC = REPO / "temp/crave_kai0bd/feat_cache"
BASE = REPO / "kai0/data/Task_A/kai0_base"
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"  # OUT OF SCOPE: supervised pi0-AE labels
TEST = 2302; W = 50
csB = json.load(open(BASE / "meta/info.json"))["chunks_size"]
csQ = json.load(open(Q5 / "meta/info.json"))["chunks_size"]

eps = list_cache_eps(FC)
nbase = sum(1 for e in eps if e < 100000); ndag = sum(1 for e in eps if e >= 100000)
print(f"挖矿集: {len(eps)} eps (kai0_base {nbase} + kai0_dagger {ndag}); test=kai0_base ep{TEST}", flush=True)
fs = FeatureSpace(FC, eps)
dv = DiscreteValue(fs, eps, select="fixed", order="time")  # == hdf5_v24_eval.build_model 配方
Pord = dv.Pord

aa, rr, st, n = loadep(FC, TEST)
v3, lab, marg = dv.value(aa, rr, st, ret_lab=True)
NF = len(pd.read_parquet(BASE / "data" / f"chunk-{TEST//csB:03d}" / f"episode_{TEST:06d}.parquet", columns=["frame_index"]))
crave = np.repeat(v3, 10)[:NF]
if len(crave) < NF: crave = np.concatenate([crave, np.full(NF - len(crave), crave[-1])])
crave = smooth_monotone(crave, fps=30.0)  # 连续读出(标准 smooth_monotone)

dQ = pd.read_parquet(Q5 / "data" / f"chunk-{TEST//csQ:03d}" / f"episode_{TEST:06d}.parquet")
ae = dQ["absolute_value"].to_numpy().astype(float)
NF = min(NF, len(ae)); crave = crave[:NF]; ae = ae[:NF]


def adv(v, w=W):
    return np.clip(advantage(v, w), -1, 1)


crave_adv = adv(crave); ae_adv = adv(ae)
np.savez(REPO / "temp/_crave_ae_kai0base.npz", crave=crave, ae=ae, crave_adv=crave_adv, ae_adv=ae_adv, fps=30.0)
x = np.arange(NF)
r_shape = pearsonr(crave, ae)[0]; tau_shape = kendalltau(crave, ae)[0]
print(f"corr(CRAVE,AE)={r_shape:.3f} tau={tau_shape:.3f}", flush=True)
print(f"CRAVE end{crave[-1]:.2f} max{crave.max():.2f} 单调{np.mean(np.diff(crave)>=-1e-6):.0%} neg{np.mean(crave_adv<0):.0%}", flush=True)
print(f"AE    end{ae[-1]:.2f} max{ae.max():.2f} 单调{np.mean(np.diff(ae)>=-1e-6):.0%} neg{np.mean(ae_adv<0):.0%}", flush=True)

# 末帧锚
vid = BASE / "videos" / f"chunk-{TEST//csB:03d}" / "observation.images.top_head" / f"episode_{TEST:06d}.mp4"
c = av.open(str(vid)); last = None
for f in c.decode(video=0): last = f.to_ndarray(format="rgb24")
c.close()


def norm01(z): return (z - z.min()) / (z.max() - z.min() + 1e-9)


fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.0, 0.9], hspace=0.42, wspace=0.5)
axv = fig.add_subplot(gs[0, :3])
axv.plot(x, crave, color="#2ca02c", lw=2.2, label=f"CRAVE 零训练 (end {crave[-1]:.2f}, 单调 {np.mean(np.diff(crave)>=-1e-6)*100:.0f}%)")
axv.plot(x, ae, color="#d62728", lw=1.6, alpha=.85, label=f"pi0-AE 监督 absolute_value (end {ae[-1]:.2f}, max {ae.max():.2f})")
axv.axhline(1, color="#2ca02c", ls=":", lw=1, alpha=.5); axv.axhline(0, color="k", lw=.5)
axv.set_ylabel("value"); axv.set_xlim(0, NF); axv.grid(alpha=.25); axv.legend(fontsize=9, loc="upper left")
axv.set_title(f"kai0_base ep{TEST} ({NF}f≈{NF/30:.0f}s): 等价对比(同域 base+dagger 挖矿, AE 同域训练)", fontsize=10.5)
axe = fig.add_subplot(gs[0, 3]); axe.imshow(last); axe.axis("off"); axe.set_title("末帧(任务终态)", fontsize=9)
axa = fig.add_subplot(gs[1, :3], sharex=axv)
axa.plot(x, crave_adv, color="#2ca02c", lw=1.4, label=f"CRAVE adv (neg {np.mean(crave_adv<0)*100:.0f}%)")
axa.plot(x, ae_adv, color="#d62728", lw=1.2, alpha=.8, label=f"AE adv (neg {np.mean(ae_adv<0)*100:.0f}%)")
axa.axhline(0, color="k", lw=.7); axa.fill_between(x, 0, crave_adv, where=crave_adv < 0, color="#2ca02c", alpha=.15)
axa.fill_between(x, 0, ae_adv, where=ae_adv < 0, color="#d62728", alpha=.12)
axa.set_ylabel(f"advantage (n vs n+{W})"); axa.set_xlabel("frame"); axa.set_xlim(0, NF); axa.grid(alpha=.25); axa.legend(fontsize=9, loc="lower left")
axa.set_title("advantage 对比", fontsize=10)
axn = fig.add_subplot(gs[2, :2])
axn.plot(x, norm01(crave), color="#2ca02c", lw=1.8, label="CRAVE (min-max)")
axn.plot(x, norm01(ae), color="#d62728", lw=1.4, alpha=.8, label="AE (min-max)")
axn.plot(x, x / (NF - 1), "k--", lw=1, alpha=.6, label="线性时间参考")
axn.set_title(f"形状归一: corr={r_shape:.2f}, τ={tau_shape:.2f}", fontsize=10)
axn.set_xlabel("frame"); axn.set_ylabel("min-max"); axn.grid(alpha=.25); axn.legend(fontsize=8)
axt = fig.add_subplot(gs[2, 2:]); axt.axis("off")
txt = (f"【等价对比 · kai0_base ep{TEST} · 同域同数据】\n\n"
       f"  CRAVE(零训练, base+dagger 挖矿): end {crave[-1]:.2f}  单调 {np.mean(np.diff(crave)>=-1e-6)*100:.0f}%\n"
       f"  pi0-AE(监督, kai0 训练):        end {ae[-1]:.2f}  max {ae.max():.2f}  单调 {np.mean(np.diff(ae)>=-1e-6)*100:.0f}%\n\n"
       f"  形状相关 Pearson {r_shape:.2f} / τ {tau_shape:.2f}\n"
       f"  退步(adv<0)占比: CRAVE {np.mean(crave_adv<0)*100:.0f}% vs AE {np.mean(ae_adv<0)*100:.0f}%\n\n"
       "公平设定: AE 同域(kai0 base+dagger)训练, CRAVE 同域同数据\n"
       "重新挖矿; 测试取 kai0_base 长 episode。此为 in-distribution\n"
       "样本(AE 主场), 看两者在公平条件下的一致性与差异。")
axt.text(0, 1, txt, fontsize=9.5, va="top", bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#bbb"))
fig.suptitle(f"CRAVE(零训练) vs pi0-AE(监督) — 等价对比(kai0_base+dagger 挖矿, kai0_base ep{TEST})", fontsize=13, y=0.98)
out = viz_dir() / "crave_vs_ae_kai0base.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print("SAVED", out, flush=True)
print("CRAVE_VS_AE_KAI0BASE_DONE")
