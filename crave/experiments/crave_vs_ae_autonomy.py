#!/usr/bin/env python
"""CRAVE(零训练 milestone-value) vs AWBC pi0-AE(监督) 在 autonomy 真机 rollout 的 value 对比.

同一 rollout(temp/autonomy, 3 轮叠衣 7676f, 含 round1 衣物被拿走 / round2 叠完被弄乱两次退步):
  - CRAVE: dagger demo 挖 milestone 模型 → 应用到 autonomy 特征(tcc_autonomy_{raw,armmask}) → value
  - AE:    temp/autonomy_pi0ae.npy(pi0-AE 监督模型对同一 rollout 的 absolute_value, 离线已算)
两者都是 "demo 训/挖 → rollout 应用", 公平对照退步检测能力。
CRAVE 核心与 smooth800_v24_full.py / crave_vs_ae_ep808.py 逐字一致。
输出: crave/docs/visualization/crave_vs_ae_autonomy.png

Rewrite onto the `crave` library:
  - mkp/med/viterbi/smooth_monotone/advantage come from crave.utils; REPO from crave.config.
  - The mining source (vis0526) is NOT in the crave dataset registry, so its lpst/loadep
    (tcc 3-path cache, key "f") + the script-specific 3-path `emb` and KMeans96 mining stay
    inline. TODO(crave-lib): register vis0526 (vis_base/v3/2026-05-26-v3 + tcc_vis0526_*)
    as a DatasetConfig so kai0.loadep_tcc / resolve_dataset can serve it.
  - OUT OF SCOPE (supervised AE): temp/autonomy_pi0ae.npy is the pi0-AE absolute_value, a
    plain npy load.
"""
import json, os
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, kendalltau

from crave.config import REPO, viz_dir
from crave.render import setup_mpl
from crave.utils import mkp, med, viterbi, smooth_monotone, advantage

plt = setup_mpl()

# 与规范 rollout_v24_sync_video.py 完全一致: autonomy 属 vis 域, CRAVE 挖矿源 = vis0526 全集
DS = REPO / "kai0/data/Task_A/vis_base/v3/2026-05-26-v3"
ARM = REPO / "temp/tcc_vis0526_armmask/feat_cache"
RAW = REPO / "temp/tcc_vis0526_raw/feat_cache"
AROLL = REPO / "temp/tcc_autonomy_armmask/feat_cache"
RROLL = REPO / "temp/tcc_autonomy_raw/feat_cache"
ROLLDS = REPO / "temp/autonomy"
AE_NPY = REPO / "temp/autonomy_pi0ae.npy"  # OUT OF SCOPE: supervised pi0-AE absolute_value
W = 50
csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]


def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//csDS:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n


rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
mined = all_eps   # 全集挖矿(同规范 rollout, TEST=-1)
print(f"mining {len(mined)} vis0526 eps (规范源)", flush=True)
Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st):
    an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1)


A, R, S, T, E, SP, EP_ = [], [], [], [], [], [], []
for e in mined:
    aa, rr, st, n = loadep(e); g = emb(aa, rr, st)
    A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
    SP.append(g[:2]); EP_.append(g[-2:])
A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
G = emb(A, R, S)
km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
N = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
Pstart = {}
for e in sorted(set(E.tolist())):
    m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Pstart[e] = float(np.median(tpos[nn]))
cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / N) for c in range(96)])
bk = np.linspace(0, 1, 11); sel = []
for b in range(10):
    inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
    if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:2]
sel = sorted(set(sel), key=lambda c: tpos[c])


def gr(idx):
    o = []; s = None; pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s is not None: o.append((s, pv))
            s = i
        pv = i
    if s is not None: o.append((s, pv))
    return [x for x in o if x[1] - x[0] >= 1]


Pk = {}
for c in sel:
    fe = []
    for e in sorted(set(E.tolist())):
        m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
        if rs: fe.append(T[rs[0][0]])
    Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]
startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP_)).cluster_centers_
NB = 21; bins = np.linspace(0, 1, NB); cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]


def value(aa, rr, st):
    Fq = emb(aa, rr, st); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
    for ci in range(len(order)):
        for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
    ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
    tn = np.arange(nq) / nq
    em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
    em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    return med(viterbi(em, bins, lam=8.0)[0], 9)


# ---- CRAVE on autonomy rollout ----
aR = np.load(AROLL / "ep0.npz")["f"]; rR = np.load(RROLL / "ep0.npz")["f"]; nR = min(len(aR), len(rR))
stR = np.stack(pd.read_parquet(ROLLDS / "data/chunk-000/episode_000000.parquet", columns=["observation.state"])["observation.state"].to_numpy())
stR = stR[np.minimum(np.arange(nR) * 10, len(stR) - 1)]
v3 = value(aR[:nR], rR[:nR], stR)
ae = np.load(AE_NPY).astype(float)
NF = len(ae)
crave = np.repeat(v3, 10)[:NF]
if len(crave) < NF: crave = np.concatenate([crave, np.full(NF - len(crave), crave[-1])])
crave = smooth_monotone(crave, fps=30.0)  # 连续读出(标准 smooth_monotone)


def adv(v, w=W):
    return np.clip(advantage(v, w), -1, 1)


crave_adv = adv(crave); ae_adv = adv(ae)
np.savez(REPO / "temp/_crave_ae_autonomy.npz", crave=crave, ae=ae, crave_adv=crave_adv, ae_adv=ae_adv, fps=30.0)
x = np.arange(NF); FPS = 30.0; t = x / FPS
r_shape = pearsonr(crave, ae)[0]; tau_shape = kendalltau(crave, ae)[0]
print(f"corr(CRAVE,AE)={r_shape:.3f} tau={tau_shape:.3f}", flush=True)
print(f"CRAVE: end{crave[-1]:.2f} max{crave.max():.2f} min{crave.min():.2f} neg-adv{np.mean(crave_adv<0):.0%}", flush=True)
print(f"AE:    end{ae[-1]:.2f} max{ae.max():.2f} min{ae.min():.2f} neg-adv{np.mean(ae_adv<0):.0%}", flush=True)


def norm01(z): return (z - z.min()) / (z.max() - z.min() + 1e-9)


B1, B2 = NF / 3, 2 * NF / 3
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 0.95], hspace=0.5)
axv = fig.add_subplot(gs[0])
axv.plot(t, crave, color="#2ca02c", lw=2.2, label=f"CRAVE 零训练 (end {crave[-1]:.2f}, 两次回落到~0)")
axv.plot(t, ae, color="#d62728", lw=1.6, alpha=.85, label=f"pi0-AE 监督 (end {ae[-1]:.2f}, max {ae.max():.2f})")
for b in (B1, B2): axv.axvline(b / FPS, color="orange", ls=":", lw=1.3)
for s_, nm in [(0, "round1"), (1, "round2"), (2, "round3")]:
    axv.text((s_ * NF / 3 + 60) / FPS, 1.04, nm, fontsize=9, color="gray")
axv.axhline(1, color="#2ca02c", ls=":", lw=1, alpha=.5); axv.axhline(0, color="k", lw=.5)
axv.set_ylabel("value"); axv.set_xlim(0, NF / FPS); axv.set_ylim(-.05, 1.13); axv.grid(alpha=.25)
axv.legend(fontsize=9, loc="center left")
axv.set_title("autonomy 真机 rollout(3 轮, round1 衣物被拿走 / round2 叠完被弄乱): CRAVE 两次退步到 0 + 每轮恢复, round3→1.0", fontsize=10.5)
axa = fig.add_subplot(gs[1], sharex=axv)
axa.plot(t, crave_adv, color="#2ca02c", lw=1.4, label=f"CRAVE adv (neg {np.mean(crave_adv<0)*100:.0f}%)")
axa.plot(t, ae_adv, color="#d62728", lw=1.2, alpha=.8, label=f"AE adv = Δabsolute_value 派生(rollout 无 AE relative 标签; neg {np.mean(ae_adv<0)*100:.0f}%)")
axa.axhline(0, color="k", lw=.7)
axa.fill_between(t, 0, crave_adv, where=crave_adv < 0, color="#2ca02c", alpha=.15)
for b in (B1, B2): axa.axvline(b / FPS, color="orange", ls=":", lw=1.3)
axa.set_ylabel(f"advantage (n vs n+{W})"); axa.set_xlabel("seconds"); axa.grid(alpha=.25); axa.legend(fontsize=9, loc="lower left")
axa.set_title("退步信号: CRAVE 在两个轮次边界给出集中负 advantage(对齐真退步); AE 退步信号弱/弥散\n(注:autonomy 是真机 rollout,AE 只有离线 absolute_value、无 relative_advantage 标签,故 P/N 只能 Δvalue 派生;要直出相对值需重跑 AE 模型推理)", fontsize=9)
axn = fig.add_subplot(gs[2])
axn.plot(t, norm01(crave), color="#2ca02c", lw=1.8, label="CRAVE (min-max)")
axn.plot(t, norm01(ae), color="#d62728", lw=1.4, alpha=.8, label="AE (min-max)")
for b in (B1, B2): axn.axvline(b / FPS, color="orange", ls=":", lw=1.3)
axn.set_title(f"形状归一: corr(CRAVE,AE)={r_shape:.2f}, τ={tau_shape:.2f}  |  CRAVE 退步结构清晰可分轮, AE 整体偏单调爬升(对退步不敏感)", fontsize=10)
axn.set_xlabel("seconds"); axn.set_ylabel("min-max"); axn.grid(alpha=.25); axn.legend(fontsize=8, loc="center left")
fig.suptitle("CRAVE(零训练 milestone-value) vs AWBC pi0-AE(监督) — autonomy 真机 rollout value/advantage 对比", fontsize=13, y=0.97)
out = viz_dir() / "crave_vs_ae_autonomy.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print("SAVED", out, flush=True)
print("CRAVE_VS_AE_AUTONOMY_DONE")
