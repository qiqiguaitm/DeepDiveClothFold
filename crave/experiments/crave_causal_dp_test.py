#!/usr/bin/env python
"""验证: CRAVE 的 value 读出能否改成"因果/前向 DP"(在线, 只用 ≤t 的帧, 保零训练)。
全局 Viterbi-DP 的非因果只来自: 末帧 end-bin 奖励 + 反向 backtrace + 未来中值。
前向 cost 累积本身是因果的 → 在线读出 = value[t] = bins[argmin(forward_cost[t])]。
本脚本在 kai0_base 真 episode 上对比 全局DP(v_full) vs 因果前向(v_causal),报相关/单调/滞后/末值差,
并对照 AE absolute_value(advantage_q5)。挖矿核心与其它 crave_* 逐字一致(kai0bd 缓存)。

Thin entrypoint over `crave`: triple-cache `loadep`, `mkp`, `med` from the package; mpl
from crave.render. The inlined V2.4 mining (cov_n binning) + custom causal DP variants
(dp_full / dp_causal / dp_fixedlag) stay verbatim — they predate the library's builders.
"""
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, kendalltau

from crave.config import REPO, viz_dir
from crave.data import loadep as loadep_triple
from crave.data import kai0
from crave.render import setup_mpl
from crave.utils import med, mkp

plt = setup_mpl()

FC = REPO / "temp/crave_kai0bd/feat_cache"
BASE = REPO / "kai0/data/Task_A/kai0_base"; Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
csB = kai0.chunks_size(str(BASE)); csQ = kai0.chunks_size(str(Q5))
TEST_EPS = [2302, 23, 2047, 2238]


def loadep(e):
    return loadep_triple(FC, e)


eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz")); mined = eps
Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st):
    an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1)


A, R, S, T, E, SP, EP_ = [], [], [], [], [], [], []
for e in mined:
    aa, rr, st, n = loadep(e); g = emb(aa, rr, st)
    A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); SP.append(g[:2]); EP_.append(g[-2:])
A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E); G = emb(A, R, S)
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
pen = 8.0 * np.abs(bins[:, None] - bins[None])


def emission(aa, rr, st, causal_tn=False):
    Fq = emb(aa, rr, st); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
    for ci in range(len(order)):
        for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
    ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
    # tn: 全局用 arange/nq; 因果用"已用帧/估计长度"(这里取已用帧, 末端门控改为绝对进度门控的近似)
    tn = np.arange(nq) / nq
    em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
    em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    return em


def med_causal(arr, w=9):  # 只用过去 w 帧的中值(因果)
    return np.array([np.median(arr[max(0, j - w + 1):j + 1]) for j in range(len(arr))])


def dp_full(em):  # 全局 Viterbi(非因果): 前向+末端奖励+反向 backtrace+未来中值
    NF = len(em); cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return med(bins[path], 9)


def dp_causal(em):  # 朴素因果前向: bins[argmin(累积 cost_t)] —— 有"粘滞"缺陷(对变化不敏感/到不了1)
    NF = len(em); cost = np.full(NB, 1e9); cost[0] = em[0, 0]; out = np.zeros(NF)
    out[0] = bins[int(cost.argmin())]
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]
        out[j] = bins[int(cost.argmin())]
    return med_causal(out, 9)


def dp_fixedlag(em, L=12):  # 固定滞后 Viterbi: 从 t+L 帧的最优态局部回溯 L 步解码第 t 帧(在线, 延迟 L 帧)
    NF = len(em); cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((NF, NB), int)
    endstate = np.zeros(NF, int); endstate[0] = int(cost.argmin())
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k
        endstate[j] = int(cost.argmin())
    out = np.zeros(NF, int)
    for j in range(NF):
        t = min(j + L, NF - 1); s = endstate[t]
        for jj in range(t, j, -1): s = bp[jj][s]
        out[j] = s
    return med_causal(bins[out], 9)


def lag(a, b, maxlag=60):  # b 相对 a 的滞后(正=b 落后)
    best, bl = -2, 0
    for L in range(0, maxlag):
        c = np.corrcoef(a[L:], b[:len(b) - L])[0, 1] if len(a) - L > 5 else -2
        if c > best: best, bl = c, L
    return bl


def backhalf_corr(a, b):  # 后半段相关 = 对后段变化的敏感度
    h = len(a) // 2; return pearsonr(a[h:], b[h:])[0] if a[h:].std() > 1e-6 and b[h:].std() > 1e-6 else float("nan")


rows = []; fig, axs = plt.subplots(2, 2, figsize=(14, 7.5)); axs = axs.ravel()
for k, EP in enumerate(TEST_EPS):
    aa, rr, st, n = loadep(EP); em = emission(aa, rr, st)
    vf = dp_full(em); vc = dp_causal(em); vfl = dp_fixedlag(em, 12)
    pr_c = pearsonr(vc, vf)[0]; pr_l = pearsonr(vfl, vf)[0]
    endgap_c = float(abs(vc[-1] - vf[-1])); endgap_l = float(abs(vfl[-1] - vf[-1]))
    bh_c = backhalf_corr(vf, vc); bh_l = backhalf_corr(vf, vfl)       # 后半敏感度
    end_l = float(vfl[-1]); mono_l = float((np.diff(vfl) >= -1e-6).mean())
    rows.append((EP, n, pr_c, pr_l, endgap_c, endgap_l, bh_c, bh_l, end_l, mono_l))
    ax = axs[k]; x = np.arange(n)
    ax.plot(x, vf, color="#2ca02c", lw=2.2, label="全局DP(非因果)")
    ax.plot(x, vc, color="#bbbbbb", lw=1.2, alpha=.8, label="朴素前向argmin(粘滞)")
    ax.plot(x, vfl, color="#1f77ff", lw=1.5, alpha=.9, label="固定滞后L=12(在线·修正)")
    ax.set_title(f"ep{EP} ({n}f): 修正corr={pr_l:.3f} 末值={end_l:.2f}(差{endgap_l:.2f}) 后半敏感={bh_l:.2f} (朴素后半{bh_c:.2f})", fontsize=8.5)
    ax.set_xlabel("frame@3Hz"); ax.set_ylabel("value"); ax.grid(alpha=.25); ax.legend(fontsize=7, loc="lower right"); ax.set_ylim(-.05, 1.05)
fig.suptitle("在线 DP 读出修正: 固定滞后 Viterbi(L=12) vs 朴素前向argmin vs 全局DP — kai0_base", fontsize=12)
fig.tight_layout(); out = viz_dir() / "crave_causal_dp_test.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True)

print("\nep | n | corr朴素 | corr固滞 | 末差朴素 | 末差固滞 | 后半敏感朴素 | 后半敏感固滞 | 固滞末值 | 固滞单调")
for r in rows:
    print(f"{r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} | {r[8]:.2f} | {r[9]:.0%}")
a = np.array([[r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in rows])
print(f"\n均值: corr 朴素{a[:,0].mean():.3f}→固滞{a[:,1].mean():.3f} | 末值差 朴素{a[:,2].mean():.2f}→固滞{a[:,3].mean():.2f} | "
      f"后半敏感 朴素{a[:,4].mean():.2f}→固滞{a[:,5].mean():.2f} | 固滞末值{a[:,6].mean():.2f}")
print("CAUSAL_DP_TEST_DONE")
