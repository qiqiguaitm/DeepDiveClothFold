#!/usr/bin/env python
"""评估超参设置法是否"自适应": 不看超参是否固定, 看**输出质量是否随数据稳定/改善**。
做法: 固定规则(10bin·top2), 把挖矿数据从 50→500 ep 扫一遍, 测:
  #milestone(是否随数据爆炸?应饱和≈任务复杂度) / 每 milestone 帧数(应随数据增→更稳) /
  最大进度 gap(覆盖均匀性) / ep808 value 与全量(500ep)基线 corr(输出是否稳) / 单调率。
判据: 自适应好 = 数据↑ 时 输出质量稳/升、#milestone 饱和、frames/milestone 单增。
输出: temp/crave_interp_ep808/adaptivity_datasize.png + 表。

Thin entrypoint over `crave`: paths from crave.config, kai0 state subsample + the
mkp/med/viterbi/smooth_monotone helpers from crave; the dagger raw/armmask npz loading and
the per-config milestone build stay inline (the experiment's own ablation logic).
跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_adaptivity_datasize.py
"""
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

from crave.config import REPO, resolve_dataset
from crave.data import kai0
from crave.render import setup_mpl
from crave.utils import med, mkp, smooth_monotone, viterbi

plt = setup_mpl()

_cfg = resolve_dataset("smooth800_dagger")
DS = Path(_cfg.root)
ARM = Path(_cfg.arm_cache); RAW = Path(_cfg.raw_cache)
OUT = REPO / "temp/crave_interp_ep808"; EP = 808
csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]
KFRAC = 96 / 500.0  # K 随数据等比例缩放(每 KMeans 簇维持~相同帧数), 体现"K 自适应数据"


def loadep(e):
    a = np.load(ARM / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], kai0.state_subsampled(_cfg, e, n), n


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
pool = sorted(np.random.RandomState(0).permutation(all_eps)[:500].tolist())
if EP not in pool: pool = sorted(pool + [EP])
print(f"载入 {len(pool)} eps ...", flush=True)
EPD = {e: loadep(e) for e in pool}
# 全量 PMU/PSD(各子集复用同一 proprio 归一, 隔离变量=数据量)
Pm = mkp(np.concatenate([EPD[e][2] for e in pool])); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st):
    an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1)


Gd = {e: emb(*EPD[e][:3]) for e in pool}
SP = {e: Gd[e][:2] for e in pool}; EPe = {e: Gd[e][-2:] for e in pool}


def build_value(eps, K):
    G = np.concatenate([Gd[e] for e in eps]); E = np.concatenate([np.full(len(Gd[e]), e) for e in eps])
    T = np.concatenate([np.arange(len(Gd[e])) / max(1, len(Gd[e]) - 1) for e in eps])
    km = KMeans(K, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
    Nset = len(eps); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(K)])
    Pstart = {}
    for e in eps:
        m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Pstart[e] = float(np.median(tpos[nn]))
    cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / Nset) for c in range(K)])
    bk = np.linspace(0, 1, 11); sel = []
    for b in range(10):
        inb = [c for c in range(K) if bk[b] <= tpos[c] < bk[b + 1]]
        if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:2]
    sel = sorted(set(sel), key=lambda c: tpos[c])
    Pk = {}
    for c in sel:
        fe = []
        for e in eps:
            m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
            if rs: fe.append(T[rs[0][0]])
        Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
    order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]; Pord = np.array([Pk[c] for c in order])
    startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate([SP[e] for e in eps])).cluster_centers_
    endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate([EPe[e] for e in eps])).cluster_centers_
    NB = 21; bins = np.linspace(0, 1, NB); cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]
    nm = np.empty(len(G), int)
    for i0 in range(0, len(G), 20000): nm[i0:i0 + 20000] = np.linalg.norm(G[i0:i0 + 20000, None] - C[None], axis=2).argmin(1)
    mcount = np.bincount(nm, minlength=len(order))
    # ep808 value
    aa, rr, st, nq = EPD[EP]; Fq = emb(aa, rr, st)
    d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
    for ci in range(len(order)):
        for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
    ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
    tn = np.arange(nq) / nq
    em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
    em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    path = viterbi(em, bins, lam=8.0, end_bonus=2)[1]
    v = smooth_monotone(np.repeat(med(bins[path], 9), 10), fps=30.0)
    return dict(nmile=len(order), mcount=mcount, Pord=Pord, frames=len(G), v=v)


SIZES = [50, 100, 200, 350, 500]
res = {}
for N in SIZES:
    eps = sorted(pool[:N]) if EP in pool[:N] else sorted(pool[:N] + [EP])
    K = max(16, int(round(N * KFRAC)))  # K 随数据缩放
    res[N] = build_value(eps, K); res[N]["K"] = K
    print(f"N={N:3d}ep K={K:3d}: {res[N]['frames']}帧 #milestone={res[N]['nmile']} frames/ms均{res[N]['frames']/res[N]['nmile']:.0f}/min{res[N]['mcount'].min()}", flush=True)

base = res[500]["v"]
print(f"\n{'N(ep)':>6}{'帧':>8}{'K':>5}{'#里程碑':>8}{'每ms最少帧':>11}{'最大gap':>8}{'corr(500)':>10}{'单调率':>8}", flush=True)
for N in SIZES:
    r = res[N]; gap = float(np.diff(np.concatenate([[0], r["Pord"], [1]])).max())
    corr = pearsonr(r["v"], base)[0]; mono = float(np.mean(np.diff(r["v"]) >= -1e-6))
    print(f"{N:>6}{r['frames']:>8}{r['K']:>5}{r['nmile']:>8}{int(r['mcount'].min()):>11}{gap:>8.2f}{corr:>10.3f}{mono:>8.2%}", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].plot(SIZES, [res[N]["nmile"] for N in SIZES], "o-", color="#3b6fb0")
ax[0].set_xlabel("挖矿 ep 数"); ax[0].set_ylabel("#milestone"); ax[0].set_ylim(0, 22)
ax[0].set_title("#milestone vs 数据量\n(饱和≈任务复杂度, 不随数据爆炸 = 好)", fontsize=10); ax[0].grid(alpha=.25)
ax[1].plot(SIZES, [res[N]["frames"] / res[N]["nmile"] for N in SIZES], "o-", color="#1a9641", label="均")
ax[1].plot(SIZES, [int(res[N]["mcount"].min()) for N in SIZES], "s--", color="#e08a1e", label="最少")
ax[1].set_xlabel("挖矿 ep 数"); ax[1].set_ylabel("每 milestone 帧数"); ax[1].legend(fontsize=8)
ax[1].set_title("每 milestone 数据量 vs 数据量\n(单增 = 数据让里程碑更稳 = 好)", fontsize=10); ax[1].grid(alpha=.25)
ax[2].plot(SIZES, [pearsonr(res[N]["v"], base)[0] for N in SIZES], "o-", color="#d7191c")
ax[2].set_xlabel("挖矿 ep 数"); ax[2].set_ylabel("ep808 value corr(vs 500ep)"); ax[2].set_ylim(0, 1.02)
ax[2].set_title("输出(value)随数据收敛到全量\n(早早趋稳 = 自适应稳健 = 好)", fontsize=10); ax[2].grid(alpha=.25)
fig.suptitle("自适应性评估: 固定规则下, 数据 50→500ep 时输出是否稳定/改善(而非超参是否固定)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "adaptivity_datasize.png", dpi=120); plt.close(fig)
print("\nSAVED", OUT / "adaptivity_datasize.png", flush=True); print("ADAPTIVITY_DONE", flush=True)
