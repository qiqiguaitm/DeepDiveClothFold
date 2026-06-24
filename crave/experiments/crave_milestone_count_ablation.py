#!/usr/bin/env python
"""消融: milestone 数量由什么决定 + 固定数 vs 覆盖率阈值 选择哪个更合理。
问题1: KMeans-K(48/96/144) 改不改 milestone 数? (答: 不改, 数由 bins×topN 定)
问题2: 固定"每bin top-2=20" vs "覆盖率≥τ 全采" 哪个更合理?
对每配置报: #milestone / 进度间距中位 / **最大间距(gap)** / 每 milestone 最少帧 / ep808 value 与基线 corr / 单调率。
输出: temp/crave_interp_ep808/milestone_count_ablation.png + 控制台表。

Thin entrypoint over `crave`: mkp/med/smooth_monotone from crave.utils, REPO from
crave.config, Agg+SimHei via crave.render.setup_mpl. The smooth800_dagger dataset +
tcc_smooth800_dagger_{raw,armmask} caches + the FeatureSpace-style emb / milestone build
are kept inlined (this script predates the DiscreteValue class) — see TODOs below.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import med, mkp, smooth_monotone

plt = setup_mpl()

# TODO(crave-lib): the A_smooth800_dagger_all dataset + tcc_smooth800_dagger_{raw,armmask}
# feature caches should move into crave.config.datasets / crave.data (kai0-family caches).
# NOTE: legacy used REPO=/home/tim/workspace/deepdive_kai0 (local mount); crave.config.REPO
# is the canonical /vePFS root (env-overridable via CRAVE_REPO) — same tree, different mount.
DS = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"
ARM = REPO / "temp/tcc_smooth800_dagger_armmask/feat_cache"; RAW = REPO / "temp/tcc_smooth800_dagger_raw/feat_cache"
OUT = REPO / "temp/crave_interp_ep808"; EP = 808; MINE_N = 500
csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]


def lpst(e, n):
    pq = DS / f"data/chunk-{e//csDS:03d}/episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n


def gr(idx):
    o = []; s0 = None; pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s0 is not None: o.append((s0, pv))
            s0 = i
        pv = i
    if s0 is not None: o.append((s0, pv))
    return [x for x in o if x[1] - x[0] >= 1]


rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
all_eps = sorted(e for e in (int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset)
mined = sorted(np.random.RandomState(0).permutation(all_eps)[:min(MINE_N, len(all_eps))].tolist())
if EP not in mined: mined = sorted(mined + [EP])
print(f"载入 {len(mined)} eps ...", flush=True)
EPD = {e: loadep(e) for e in mined}
Sall = [EPD[e][2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st):
    an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1)


A = []; R = []; S = []; T = []; E = []; SP = []; EP_ = []
for e in mined:
    aa, rr, st, n = EPD[e]; g = emb(aa, rr, st)
    A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); SP.append(g[:2]); EP_.append(g[-2:])
A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
G = emb(A, R, S); SPc = np.concatenate(SP); EPc = np.concatenate(EP_)
STARTK = KMeans(8, n_init=2, random_state=0).fit(SPc).cluster_centers_
ENDK = KMeans(8, n_init=2, random_state=0).fit(EPc).cluster_centers_
print(f"挖矿帧 {len(G)}", flush=True)
_KMCACHE = {}


def kmeans_cached(K):
    if K not in _KMCACHE:
        km = KMeans(K, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
        Nset = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(K)])
        Pstart = {}
        for e in sorted(set(E.tolist())):
            m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Pstart[e] = float(np.median(tpos[nn]))
        cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / Nset) for c in range(K)])
        _KMCACHE[K] = (lab, allC, tpos, cov_n)
    return _KMCACHE[K]


def build(K, selector):
    lab, allC, tpos, cov_n = kmeans_cached(K)
    sel = sorted(set(selector(tpos, cov_n, K)), key=lambda c: tpos[c])
    Pk = {}
    for c in sel:
        fe = []
        for e in sorted(set(E.tolist())):
            m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
            if rs: fe.append(T[rs[0][0]])
        Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
    order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]; Pord = np.array([Pk[c] for c in order])
    NB = 21; bins = np.linspace(0, 1, NB); cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]
    nm = np.empty(len(G), int)
    for i0 in range(0, len(G), 20000): nm[i0:i0 + 20000] = np.linalg.norm(G[i0:i0 + 20000, None] - C[None], axis=2).argmin(1)
    return dict(order=order, C=C, Pord=Pord, NB=NB, bins=bins, cb=cb, mcount=np.bincount(nm, minlength=len(order)))


def value808(M):
    aa, rr, st, n = EPD[EP]; Fq = emb(aa, rr, st); nq = len(Fq)
    C, order, cb, bins, NB = M["C"], M["order"], M["cb"], M["bins"], M["NB"]
    d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
    for ci in range(len(order)):
        for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
    ds = np.linalg.norm(Fq[:, None] - STARTK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - ENDK[None], axis=2).min(1)
    tn = np.arange(nq) / nq
    em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
    em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    pen = 8.0 * np.abs(bins[:, None] - bins[None]); NF = len(em)
    cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return smooth_monotone(np.repeat(med(bins[path], 9), 10), fps=30.0)


def sel_bin_topN(nbins, topN):
    def f(tpos, cov_n, K):
        bk = np.linspace(0, 1, nbins + 1); sel = []
        for b in range(nbins):
            inb = [c for c in range(K) if bk[b] <= tpos[c] < bk[b + 1]]
            if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:topN]
        return sel
    return f


def sel_cov_global(tau):
    return lambda tpos, cov_n, K: [c for c in range(K) if cov_n[c] >= tau]


def sel_bin_cov(nbins, tau):
    def f(tpos, cov_n, K):
        bk = np.linspace(0, 1, nbins + 1); sel = []
        for b in range(nbins):
            sel += [c for c in range(K) if bk[b] <= tpos[c] < bk[b + 1] and cov_n[c] >= tau]
        return sel
    return f


# 看一眼 K=96 的覆盖率分布(选阈值参考)
_, _, _, cov96 = kmeans_cached(96)
print(f"K96 覆盖率 cov_n: 中位{np.median(cov96):.2f} 分位[.5/.7/.9]={np.quantile(cov96,[.5,.7,.9]).round(2)} max{cov96.max():.2f}", flush=True)

CONFIGS = [
    ("K96·10bin·top2  (当前=固定≤20)", 96, sel_bin_topN(10, 2)),
    ("K48·10bin·top2  (老/少数据设置)", 48, sel_bin_topN(10, 2)),
    ("K144·10bin·top2", 144, sel_bin_topN(10, 2)),
    ("K96·cov≥0.5  全局阈值", 96, sel_cov_global(0.5)),
    ("K96·cov≥0.4  全局阈值", 96, sel_cov_global(0.4)),
    ("K96·cov≥0.3  全局阈值", 96, sel_cov_global(0.3)),
    ("K96·10bin·cov≥0.4 (每bin阈值)", 96, sel_bin_cov(10, 0.4)),
]
curves = {}
for name, K, selr in CONFIGS: curves[name] = value808(build(K, selr))
base_v = curves["K96·10bin·top2  (当前=固定≤20)"]
print(f"\n{'配置':<30}{'#里程碑':>7}{'间距中位':>9}{'最大gap':>8}{'最少帧':>8}{'corr基线':>9}{'单调率':>8}", flush=True)
rows = []
for name, K, selr in CONFIGS:
    M = build(K, selr); nmile = len(M["order"]); v = curves[name]
    sp = float(np.median(np.diff(M["Pord"]))) if nmile > 1 else 0
    gap = float(np.diff(np.concatenate([[0], M["Pord"], [1]])).max())  # 含 0/1 端点的最大空隙
    mn = int(M["mcount"].min()); corr = pearsonr(v, base_v)[0]; mono = float(np.mean(np.diff(v) >= -1e-6))
    rows.append((name, nmile, sp, gap, mn, corr, mono))
    print(f"{name:<30}{nmile:>7}{sp:>9.3f}{gap:>8.2f}{mn:>8}{corr:>9.3f}{mono:>8.2%}", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.5))
for name in ["K48·10bin·top2  (老/少数据设置)", "K96·10bin·top2  (当前=固定≤20)", "K144·10bin·top2"]:
    ax[0].plot(np.arange(len(curves[name])) / 30, curves[name], lw=1.6, label=name.split("(")[0])
ax[0].set_title("改 KMeans-K(48/96/144),选择规则不变 → value 几乎重合\nK 只改候选粒度,不是 milestone 数的杠杆", fontsize=10)
ax[0].set_xlabel("秒"); ax[0].set_ylabel("ep808 value"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
for name in ["K96·10bin·top2  (当前=固定≤20)", "K96·cov≥0.4  全局阈值", "K96·cov≥0.3  全局阈值", "K96·10bin·cov≥0.4 (每bin阈值)"]:
    ax[1].plot(np.arange(len(curves[name])) / 30, curves[name], lw=1.5, label=name)
ax[1].set_title("固定数(top2) vs 覆盖率阈值(全局/每bin)→ value 形状高度一致\n阈值法=变长 milestone,但需配 bin 防进度空隙", fontsize=10)
ax[1].set_xlabel("秒"); ax[1].set_ylabel("ep808 value"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
fig.suptitle("milestone 数量: KMeans-K vs 选择规则(固定 top-N / 覆盖率阈值)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "milestone_count_ablation.png", dpi=120); plt.close(fig)
print("\nSAVED", OUT / "milestone_count_ablation.png", flush=True); print("ABLATION_DONE", flush=True)
