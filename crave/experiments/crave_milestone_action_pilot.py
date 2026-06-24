#!/usr/bin/env python
"""B1 决定性小实验: milestone 是否比"时间"更能解释机械臂动作?
若 R²(action | milestone) > R²(action | 时间分桶), 说明 milestone 捕捉的是动作相关的真技能结构
(不只是计时器)→ "value↔动作基元对齐"可被利用做基元级 action 信号(零 RL)。
用 kai0bd 3Hz 模型 + kai0_base 的 action 列。同时看转移帧动作是否更"剧烈"(基元完成事件)。

Thin entrypoint over `crave`: triple-cache `loadep` + `mkp` from the package; kai0_base
dataset from crave.config (action column read inline). The inlined milestone mining
(covn-count binning, no Pk re-order) stays verbatim to reproduce the legacy numbers.
"""
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.cluster import KMeans

from crave.config import REPO, resolve_dataset
from crave.data import kai0
from crave.data import loadep as loadep_triple
from crave.utils import mkp

FC = REPO / "temp/crave_kai0bd/feat_cache"
BASE_CFG = resolve_dataset("kai0_base")
BASE = Path(BASE_CFG.root)
cs = kai0.chunks_size(str(BASE))
eps_all = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz") if int(p.stem[2:]) < 100000)
MINE = sorted(np.random.RandomState(0).permutation(eps_all)[:82].tolist())


def loadnpz(e):
    return loadep_triple(FC, e)


Sall = [loadnpz(e)[2] for e in MINE]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st):
    rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True); an = a_ / np.linalg.norm(a_, axis=1, keepdims=True)
    Pn = (mkp(st) - PMU) / PSD; Pn /= np.linalg.norm(Pn, axis=1, keepdims=True); return np.concatenate([rn, an, Pn], 1)


A, R, S, T, E = [], [], [], [], []
for e in MINE:
    aa, rr, st, n = loadnpz(e); A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E); G = emb(A, R, S)
km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
bk = np.linspace(0, 1, 11); sel = []
covn = np.array([len(set(E[lab == c].tolist())) for c in range(96)])
for b in range(10):
    inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
    if inb: sel += sorted(inb, key=lambda c: -covn[c])[:2]
order = sorted(set(sel), key=lambda c: tpos[c]); C = allC[order]; M = len(order)
print(f"milestones {M}", flush=True)


def load_action(e, n):
    pq = BASE / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
    act = np.stack(pd.read_parquet(pq, columns=["action"])["action"].to_numpy())
    return act[np.minimum(np.arange(n) * 10, len(act) - 1)]


# 收集 (最近milestone, 时间分桶, action, 是否转移帧)
NEAR, TBIN, ACT, TRANS = [], [], [], []
for e in MINE:
    aa, rr, st, n = loadnpz(e); Fq = emb(aa, rr, st)
    nm = np.linalg.norm(Fq[:, None] - C[None], axis=2).argmin(1)        # 最近 milestone(序号)
    tb = np.minimum((np.arange(n) / n * M).astype(int), M - 1)          # 时间分桶(同 M 个)
    act = load_action(e, n)
    tr = np.zeros(n, bool); tr[1:] = nm[1:] != nm[:-1]                  # 最近milestone 改变 = 转移帧
    NEAR.append(nm); TBIN.append(tb); ACT.append(act); TRANS.append(tr)
NEAR = np.concatenate(NEAR); TBIN = np.concatenate(TBIN); ACT = np.concatenate(ACT); TRANS = np.concatenate(TRANS)


def r2_by_group(group, y):  # R² = 用组均值预测 y 的解释方差比(多维 action 总和)
    ss_tot = ((y - y.mean(0)) ** 2).sum()
    pred = np.zeros_like(y)
    for g in np.unique(group):
        m = group == g; pred[m] = y[m].mean(0)
    ss_res = ((y - pred) ** 2).sum()
    return 1 - ss_res / ss_tot


r2_ms = r2_by_group(NEAR, ACT)        # milestone 解释 action
r2_t = r2_by_group(TBIN, ACT)         # 时间分桶解释 action(同桶数基线)
# 动作"剧烈度" = 相邻 action 差范数
dact = np.zeros(len(ACT)); dact[1:] = np.linalg.norm(np.diff(ACT, axis=0), axis=1)
print(f"R²(action | milestone) = {r2_ms:.3f}", flush=True)
print(f"R²(action | 时间分桶{M})= {r2_t:.3f}   (基线)", flush=True)
print(f"→ milestone 比时间多解释 {(r2_ms-r2_t):.3f} 的动作方差 ({'milestone 更优 ✓' if r2_ms>r2_t else '时间更优'})", flush=True)
print(f"转移帧动作变化幅度 {dact[TRANS].mean():.4f} vs 非转移 {dact[~TRANS].mean():.4f} "
      f"(比值 {dact[TRANS].mean()/max(1e-9,dact[~TRANS].mean()):.2f}× → 转移帧是否更动作-eventful)", flush=True)
print("MS_ACTION_PILOT_DONE", flush=True)
