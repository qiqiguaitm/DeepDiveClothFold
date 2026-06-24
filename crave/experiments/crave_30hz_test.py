#!/usr/bin/env python
"""30Hz vs 3Hz 推理对比: 同一 3Hz 挖出的 CRAVE 模型, 应用到 30Hz(全帧率)特征, 看是否更"跟过程"。
特别看 ep2047/ep2238 的中段平台是否被 30Hz 解开。
- 3Hz: kai0bd 缓存; 30Hz: crave_30hz 缓存(stride1)。proprio Δ 在 30Hz 用 10 帧差(对齐 3Hz 时间语义)。
- 离散 value(固定滞后 DP, 时间窗对齐)+ 连续 soft-progress(softmax 隶属 → 期望 P_k, 仅监控用)。
挖矿核心与 crave_causal_dp_test 逐字一致。
"""
import numpy as np
from sklearn.cluster import KMeans

from crave.config import REPO, viz_dir
from crave.data import loadep
from crave.render import setup_mpl
from crave.utils import mkp_gap

plt = setup_mpl()

# TODO(crave-lib): crave_kai0bd / crave_30hz feature caches are not in the dataset
# registry (no resolve_dataset entry); kept as explicit temp paths.
FC = REPO / "temp/crave_kai0bd/feat_cache"; FC30 = REPO / "temp/crave_30hz/feat_cache"
TEST = [2047, 2238, 2302]


# state ⊕ Δstate(隔 dt 帧, 对齐时间语义) == crave.utils.mkp_gap
def mkp_dt(s, dt): return mkp_gap(s, dt)


# ---- 挖矿(3Hz kai0bd) ----
eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
Sall = [loadep(FC, e)[2] for e in eps]; Pm = mkp_dt(np.concatenate(Sall), 1); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st, dt=1):
    an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp_dt(st, dt) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1)


A, R, S, T, E, SP, EP_ = [], [], [], [], [], [], []
for e in eps:
    aa, rr, st, n = loadep(FC, e); g = emb(aa, rr, st)
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
order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]; Pord = np.array([Pk[c] for c in order])
startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP_)).cluster_centers_
NB = 21; bins = np.linspace(0, 1, NB); cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]; pen = 8.0 * np.abs(bins[:, None] - bins[None])
print(f"milestones {len(order)}", flush=True)


def emission(aa, rr, st, dt):
    Fq = emb(aa, rr, st, dt); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
    for ci in range(len(order)):
        for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
    ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
    tn = np.arange(nq) / nq
    em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
    em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    return em, d


def med_causal(arr, w): return np.array([np.median(arr[max(0, j - w + 1):j + 1]) for j in range(len(arr))])


def dp_fixedlag(em, L):
    NF = len(em); cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((NF, NB), int); endstate = np.zeros(NF, int); endstate[0] = int(cost.argmin())
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k; endstate[j] = int(cost.argmin())
    out = np.zeros(NF, int)
    for j in range(NF):
        t = min(j + L, NF - 1); s = endstate[t]
        for jj in range(t, j, -1): s = bp[jj][s]
        out[j] = s
    return med_causal(bins[out], max(3, L // 2))


def soft_progress(d, temp=0.04):  # 连续: softmax(-dist) 隶属 → 期望 P_k(仅监控, 非 AWBC 排序)
    w = np.exp(-d / temp); w /= w.sum(1, keepdims=True); return w @ Pord


fig, axs = plt.subplots(3, 1, figsize=(13, 9));
for k, EP in enumerate(TEST):
    aa3, rr3, st3, n3 = loadep(FC, EP); em3, d3 = emission(aa3, rr3, st3, 1)
    v3 = dp_fixedlag(em3, 12); t3 = np.arange(n3) / 3.0                       # 3Hz, 秒
    aa30, rr30, st30, n30 = loadep(FC30, EP); em30, d30 = emission(aa30, rr30, st30, 10)
    v30_time = dp_fixedlag(em30, 120)   # 同时间窗(~4s)
    v30_light = dp_fixedlag(em30, 24)   # 轻平滑(~0.8s, 更敏感)
    soft30 = med_causal(soft_progress(d30), 30)                              # 连续 soft, 过去 1s 中值
    t30 = np.arange(n30) / 30.0
    # 中段平台"活跃度"(后半 std)
    def alive(v): h = len(v) // 2; return float(v[h:].std())
    ax = axs[k]
    ax.plot(t3, v3, color="#2ca02c", lw=2.4, label=f"3Hz 离散 (后半std={alive(v3):.3f})")
    ax.plot(t30, v30_time, color="#1f77ff", lw=1.3, alpha=.85, label=f"30Hz 离散·同时间窗 (std={alive(v30_time):.3f})")
    ax.plot(t30, v30_light, color="#ff7f0e", lw=1.0, alpha=.7, label=f"30Hz 离散·轻平滑 (std={alive(v30_light):.3f})")
    ax.plot(t30, soft30, color="#d62728", lw=1.2, alpha=.8, label=f"30Hz 连续soft (std={alive(soft30):.3f})")
    ax.set_title(f"ep{EP}: 3Hz {n3}帧 / 30Hz {n30}帧 — 中段平台 30Hz 是否更跟过程?", fontsize=9.5)
    ax.set_xlabel("秒"); ax.set_ylabel("value"); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.25); ax.legend(fontsize=7.5, loc="lower right")
fig.suptitle("CRAVE 30Hz vs 3Hz 推理对比(离散 staircase 是设计; 连续 soft 才跟细过程)", fontsize=12)
fig.tight_layout(); out = viz_dir() / "crave_30hz_test.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True); print("HZ30_TEST_DONE", flush=True)
