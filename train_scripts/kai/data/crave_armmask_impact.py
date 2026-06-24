#!/usr/bin/env python
"""验证"机械臂噪声影响": 三种特征配置各自重新挖矿 + 30Hz value, 比谁更干净/更贴 AE。
配置: ① 三路 raw+arm+prop(现状) ② 去raw arm+prop(弱化臂视觉) ③ 纯cloth arm-only(无臂视觉无proprio)。
臂在 raw(含臂)与 proprio(关节=臂姿)里; armmask 去臂。看丢掉含臂路是否让 value 更干净(臂噪声更小)。
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
FC = REPO / "temp/crave_kai0bd/feat_cache"; FC30 = REPO / "temp/crave_30hz/feat_cache"
BASE = REPO / "kai0/data/Task_A/kai0_base"; Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"; cs = 1000
TEST = [2047, 2238]


def loadnpz(fc, e):
    d = np.load(fc / f"ep{e}.npz"); a, r, s = d["armmask"], d["raw"], d["state"]
    n = min(len(a), len(r), len(s)); return a[:n], r[:n], s[:n], n


def mkp_dt(s, dt):
    d = np.zeros_like(s); d[dt:] = s[dt:] - s[:-dt]; return np.concatenate([s, d], 1)


eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
Sall = [loadnpz(FC, e)[2] for e in eps]; Pm = mkp_dt(np.concatenate(Sall), 1); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
NB = 21; bins = np.linspace(0, 1, NB); pen = 8.0 * np.abs(bins[:, None] - bins[None])


def med_causal(arr, w): return np.array([np.median(arr[max(0, j - w + 1):j + 1]) for j in range(len(arr))])


def dp_fixedlag(em, L):
    NF = len(em); cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((NF, NB), int); es = np.zeros(NF, int); es[0] = int(cost.argmin())
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k; es[j] = int(cost.argmin())
    out = np.zeros(NF, int)
    for j in range(NF):
        t = min(j + L, NF - 1); s = es[t]
        for jj in range(t, j, -1): s = bp[jj][s]
        out[j] = s
    return med_causal(bins[out], max(3, L // 2))


def make_emb(use_raw, use_arm, use_prop):
    def emb(a_, r_, st, dt=1):
        parts = []
        if use_raw: parts.append(r_ / np.linalg.norm(r_, axis=1, keepdims=True))
        if use_arm: parts.append(a_ / np.linalg.norm(a_, axis=1, keepdims=True))
        if use_prop:
            Pn = (mkp_dt(st, dt) - PMU) / PSD; Pn /= np.linalg.norm(Pn, axis=1, keepdims=True); parts.append(Pn)
        return np.concatenate(parts, 1)
    return emb


def mine(emb):
    A, R, S, T, E, SP, EP_ = [], [], [], [], [], [], []
    for e in eps:
        aa, rr, st, n = loadnpz(FC, e); g = emb(aa, rr, st)
        A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); SP.append(g[:2]); EP_.append(g[-2:])
    A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E); G = emb(A, R, S)
    km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
    Nn = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
    Ps = {}
    for e in sorted(set(E.tolist())):
        m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Ps[e] = float(np.median(tpos[nn]))
    cov = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Ps if Ps[e] > tpos[c] + 0.1)) / Nn) for c in range(96)])
    bk = np.linspace(0, 1, 11); sel = []
    for b in range(10):
        inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
        if inb: sel += sorted(inb, key=lambda c: -cov[c])[:2]
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
    cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]

    def emission(aa, rr, st, dt):
        Fq = emb(aa, rr, st, dt); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
        for ci in range(len(order)):
            for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
        ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
        tn = np.arange(nq) / nq
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        return em
    return emission, len(order)


CONFIGS = [("三路 raw+arm+prop(现状)", 1, 1, 1, "#2ca02c"),
           ("去raw arm+prop(弱化臂视觉)", 0, 1, 1, "#1f77ff"),
           ("纯cloth arm-only(无臂)", 0, 1, 0, "#d62728")]
res = {EP: {} for EP in TEST}
for name, ur, ua, up, col in CONFIGS:
    emission, nms = mine(make_emb(ur, ua, up))
    for EP in TEST:
        aa, rr, st, n = loadnpz(FC30, EP); em = emission(aa, rr, st, 10); v = dp_fixedlag(em, 120)
        dQ = pd.read_parquet(Q5 / f"data/chunk-{EP//cs:03d}/episode_{EP:06d}.parquet"); ae = dQ["absolute_value"].to_numpy().astype(float)
        L = min(len(v), len(ae)); cr = pearsonr(v[:L], ae[:L])[0]
        h = len(v) // 2; bstd = float(v[h:].std()); mono = float((np.diff(v) >= -1e-6).mean())
        # 帧间抖动(相邻差绝对值均值, 越小越平滑/噪声越小)
        jit = float(np.abs(np.diff(v)).mean())
        res[EP][name] = (v, cr, bstd, mono, jit, nms, col)

print("\n配置 | milestone数 | ep | corr(value,AE) | 后半std | 单调 | 帧间抖动jitter")
for name, *_ in [(c[0],) for c in CONFIGS]:
    for EP in TEST:
        v, cr, bstd, mono, jit, nms, col = res[EP][name]
        print(f"{name} | {nms} | ep{EP} | {cr:.3f} | {bstd:.3f} | {mono:.0%} | {jit:.4f}")

fig, axs = plt.subplots(len(TEST), 1, figsize=(13, 7))
for r, EP in enumerate(TEST):
    ax = axs[r]
    dQ = pd.read_parquet(Q5 / f"data/chunk-{EP//cs:03d}/episode_{EP:06d}.parquet"); ae = dQ["absolute_value"].to_numpy().astype(float)
    ax.plot(np.arange(len(ae)) / 30.0, ae, color="#999", lw=1.2, ls="--", label="pi0-AE absolute_value")
    for name, *_ in [(c[0],) for c in CONFIGS]:
        v, cr, bstd, mono, jit, nms, col = res[EP][name]
        ax.plot(np.arange(len(v)) / 30.0, v, color=col, lw=1.5, alpha=.9, label=f"{name}: corr{cr:.2f} 抖{jit:.3f}")
    ax.set_title(f"ep{EP} (30Hz): 三种特征配置 value vs 臂噪声", fontsize=10)
    ax.set_xlabel("秒"); ax.set_ylabel("value"); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.25); ax.legend(fontsize=7.5, loc="lower right")
fig.suptitle("机械臂噪声影响验证: 去掉含臂的 raw 路 / 用纯 cloth(armmask)是否更干净", fontsize=12)
fig.tight_layout(); out = REPO / "docs/visualization/cross_episode_recurrence_value/crave_armmask_impact.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True); print("ARMMASK_IMPACT_DONE", flush=True)
