"""CRAVE 斜坡读出(ramped): 平台→随"下一个 milestone 置信度"上升做斜线→到达后变平。
比硬阶梯信息更丰富。用真 stage_progress_gt(ep2047)验证: ramped 是否比阶梯更贴 GT。
ramp = 相对置信 d_cur/(d_cur+d_next) 在相邻两 milestone 间插值, 再单调化。
注: §4.4.19 已证段内插值伤 AWBC 排序 τ → ramped 仅用于评价/监控, AWBC 仍用阶梯。
"""
import os, glob, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, kendalltau

_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
FC = REPO / "temp/crave_kai0bd/feat_cache"; Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
csQ = json.load(open(Q5 / "meta/info.json"))["chunks_size"]; EP = 2047; NB = 21; bins = np.linspace(0, 1, NB)
eps_all = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(str(FC / "ep*.npz")) if int(os.path.basename(p)[2:-4]) < 100000)
MINE = sorted(np.random.RandomState(0).permutation(eps_all)[:82].tolist())


def loadnpz(e):
    d = np.load(FC / f"ep{e}.npz"); a, r, s = d["armmask"], d["raw"], d["state"]; n = min(len(a), len(r), len(s)); return a[:n], r[:n], s[:n], n


def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


Sall = [loadnpz(e)[2] for e in MINE]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st):
    rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True); an = a_ / np.linalg.norm(a_, axis=1, keepdims=True)
    Pn = (mkp(st) - PMU) / PSD; Pn /= np.linalg.norm(Pn, axis=1, keepdims=True); return np.concatenate([rn, an, Pn], 1)


A, R, S, T, E, SP, EP_ = [], [], [], [], [], [], []
for e in MINE:
    aa, rr, st, n = loadnpz(e); g = emb(aa, rr, st); A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); SP.append(g[:2]); EP_.append(g[-2:])
A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E); G = emb(A, R, S)
km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
covn = np.array([len(set(E[lab == c].tolist())) for c in range(96)]); bk = np.linspace(0, 1, 11); sel = []
for b in range(10):
    inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
    if inb: sel += sorted(inb, key=lambda c: -covn[c])[:2]
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
order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]; Pord = np.array([Pk[c] for c in order]); M = len(order)
startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP_)).cluster_centers_
cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]; pen = 8.0 * np.abs(bins[:, None] - bins[None])
print(f"milestones {M}", flush=True)


def med(a, w): h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


def staircase_and_ramp(aa, rr, st):
    Fq = emb(aa, rr, st); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
    for ci in range(M):
        for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
    ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
    tn = np.arange(nq) / nq; em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
    em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((nq, NB), int)
    for j in range(1, nq):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= 2; path = np.zeros(nq, int); path[-1] = cost.argmin()
    for j in range(nq - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    stair = med(bins[path], 9)
    # 段内斜坡: 当前 milestone k 来自 DP 阶梯(可回退→保留退步); intra = 到下一个 anchor 的相对置信
    raw = np.zeros(nq)
    for t in range(nq):
        k = int(np.searchsorted(Pord, stair[t] + 1e-6) - 1); k = max(0, min(k, M - 1)); kn = min(k + 1, M - 1)
        dc = d[t, k]; dn = d[t, kn]; frac = dc / (dc + dn + 1e-9)   # 0=在cur, 1=在next
        raw[t] = Pord[k] + (Pord[kn] - Pord[k]) * np.clip(frac, 0, 1)
    # §2.2 正解: 段内单调(每个 stage 段内 cummax 去噪), 段间(staircase 变化处)重置 → 保留退步
    seg = np.cumsum(np.abs(np.diff(stair, prepend=stair[0])) > 1e-6)
    raw_seg = raw.copy()
    for s in np.unique(seg):
        m = seg == s; raw_seg[m] = np.maximum.accumulate(raw[m])
    ramp_within = med(raw_seg, 9)                    # 段内单调斜坡(去噪 + 保退步)
    ramp_mono = med(np.maximum.accumulate(raw), 9)   # 全局单调(抹掉退步, 我上一版)
    return stair, ramp_within, ramp_mono


aa, rr, st, n = loadnpz(EP); stair, ramp, ramp_mono = staircase_and_ramp(aa, rr, st)
gt = pd.read_parquet(Q5 / f"data/chunk-{EP//csQ:03d}/episode_{EP:06d}.parquet")["stage_progress_gt"].to_numpy()
tgt = np.arange(len(gt)) / 30.0


def score(v, fps):
    t = np.arange(len(v)) / fps; gi = np.interp(t, tgt, gt); return pearsonr(v, gi)[0], float(np.abs(v - gi).mean()), kendalltau(v, gi)[0]


rs, ms, ts = score(stair, 3); rw, mw, tw = score(ramp, 3); rmo, mmo, tmo = score(ramp_mono, 3)
nreg = lambda v: int((np.diff(v) < -1e-3).sum())   # 退步台阶数
print(f"vs 真 stage_progress_gt:", flush=True)
print(f"  硬阶梯 staircase:     corr {rs:.3f} / MAE {ms:.3f} / τ {ts:.3f} / 退步{nreg(stair)}", flush=True)
print(f"  段内斜坡 within(保退步): corr {rw:.3f} / MAE {mw:.3f} / τ {tw:.3f} / 退步{nreg(ramp)}", flush=True)
print(f"  全局单调 mono(抹退步):  corr {rmo:.3f} / MAE {mmo:.3f} / τ {tmo:.3f} / 退步{nreg(ramp_mono)}", flush=True)
print(f"  → 段内斜坡 τ涨{tw-ts:+.3f} 且保留退步({nreg(ramp)}); 全局单调 退步被抹({nreg(ramp_mono)})——这就是你看到台阶变少的原因", flush=True)

x = np.arange(n) / 3.0; xg = tgt
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(xg, gt, color="#999", lw=1.6, ls="--", label="真 stage_progress_gt(GT)")
ax.plot(x, stair, color="#2ca02c", lw=2.2, drawstyle="steps-post", label=f"硬阶梯 (τ{ts:.2f}, 退步{nreg(stair)})")
ax.plot(x, ramp, color="#1f77ff", lw=1.8, label=f"段内斜坡 within (τ{tw:.2f}, 保退步{nreg(ramp)})")
ax.plot(x, ramp_mono, color="#ff7f0e", lw=1.3, ls=":", label=f"全局单调 mono (τ{tmo:.2f}, 退步{nreg(ramp_mono)}抹掉)")
ax.set_title(f"ep{EP}: 硬阶梯 vs 段内斜坡(保退步) vs 全局单调(抹退步) vs GT —— 正解=段内斜坡(§2.2)", fontsize=11)
ax.set_xlabel("秒"); ax.set_ylabel("value"); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.25); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); out = REPO / "docs/visualization/cross_episode_recurrence_value/crave_ramped_vs_staircase_ep2047.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True); print("RAMPED_DONE", flush=True)
