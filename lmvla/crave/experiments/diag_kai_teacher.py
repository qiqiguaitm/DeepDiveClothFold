#!/usr/bin/env python
"""诊断: kai polyline teacher 为什么接近直线? Viterbi 坏了吗?
打印 M + milestone 值; 对 3 条 ep 画 step(阶梯) vs polyline(去阶梯) vs milestone 水平线 + norm time。
输出 temp/diag_kai_teacher.png。
"""
import numpy as np, time
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

REPO = Path('/vePFS/tim/workspace/deepdive_kai0'); rng = np.random.RandomState(0); CAP = 1000; FPS = 30.
def l2(x): return x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
def cc(a, b): return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


def daw(F, C, P, lam, want_step=False):
    sC = l2(F[:3].mean(0)[None])[0]; eC = l2(F[-3:].mean(0)[None])[0]
    C2 = np.vstack([C, sC, eC]); Pp = np.concatenate([P, [0.], [1.]])
    bins = np.unique(np.concatenate([[0.], Pp, [1.]])); nb = len(bins)
    cb = [int(np.searchsorted(bins, v)) for v in Pp]; pen = lam*np.abs(bins[:, None]-bins[None])
    de = np.linalg.norm(F[:, None]-C2[None], axis=2); em = np.full((len(F), nb), 1e3)
    for ti in range(len(Pp)): em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(F), nb), int)
    for j in range(1, len(F)):
        tr = cost[None, :]+pen; kk = tr.argmin(1); cost = em[j]+tr[np.arange(nb), kk]; BP[j] = kk
    si = nb-1; path = np.zeros(len(F), int); path[-1] = si
    for j in range(len(F)-2, -1, -1): si = BP[j+1][si]; path[j] = si
    step = bins[path]; segs = []; a = 0
    for t in range(1, len(step)):
        if step[t] != step[t-1]: segs.append((a, t-1, step[t-1])); a = t
    segs.append((a, len(step)-1, step[-1])); reps = []
    for i0, i1, val in segs:
        cand = [ti for ti in range(len(Pp)) if abs(Pp[ti]-val) < 1e-9]; fr = np.arange(i0, i1+1); bd = 1e18; bf = i0
        for ti in cand:
            dd = np.linalg.norm(F[fr]-C2[ti], axis=1); k = int(dd.argmin())
            if dd[k] < bd: bd = dd[k]; bf = fr[k]
        reps.append((bf, float(val)))
    if reps[0][0] != 0: reps = [(0, float(step[0]))]+reps
    if reps[-1][0] != len(step)-1: reps = reps+[(len(step)-1, float(step[-1]))]
    rf = np.array([r[0] for r in reps]); rv = np.array([r[1] for r in reps]); keep = np.concatenate([[True], np.diff(rf) > 0])
    poly = np.interp(np.arange(len(step)), rf[keep], rv[keep]).astype(np.float32)
    return (poly, step, rf[keep], rv[keep]) if want_step else poly


t0 = time.time(); d = REPO/'lmvla/crave/data/kai_dinov3base'; idx = np.load(d/'index.npz'); E = idx['E']; FR = idx['FR']
feat = np.zeros((len(E), 768), np.float16)
for sh in sorted(d.glob('shard_*.npz')):
    s = np.load(sh); g = s['gidx']; v = s['valid'] if 'valid' in s else np.ones(len(g), bool); feat[g[v]] = s['feat'][v]
eps = sorted(np.unique(E).tolist())
if len(eps) > CAP: eps = [eps[i] for i in sorted(rng.choice(len(eps), CAP, replace=False))]
keep = np.isin(E, eps); E = E[keep]; FR = FR[keep]; feat = feat[keep]
T = np.zeros(len(E), np.float32)
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; T[o] = np.linspace(0, 1, len(o))
pca = PCA(128, random_state=0).fit(l2(feat[rng.choice(len(feat), 20000, replace=False)].astype(np.float32)))
F128 = l2((l2(feat.astype(np.float32))-pca.mean_.astype(np.float32))@pca.components_.astype(np.float32).T); NC = len(eps)
bg = BayesianGaussianMixture(n_components=40, covariance_type='diag', weight_concentration_prior=1e-2, max_iter=120, random_state=0).fit(F128[rng.choice(len(F128), 80000, replace=False)])
labs = bg.predict(F128); C = []; P = []; COV = []
for k in range(40):
    m = labs == k
    if m.sum() < 20: continue
    cov = len(set(E[m].tolist()))/NC
    if cov >= 0.5: C.append(F128[m].mean(0)); P.append(float(np.median(T[m]))); COV.append(cov)
C = l2(np.array(C, np.float32)); P = np.array(P); lam = 16.*FPS/3.
order = np.argsort(P)
print(f'[{time.time()-t0:.0f}s] M={len(C)} milestones', flush=True)
print('milestone median-T (sorted):', np.round(np.sort(P), 3).tolist(), flush=True)
print('coverage (same order):', np.round(np.array(COV)[order], 2).tolist(), flush=True)

# 诊断图: 3 条 ep, step vs polyline vs milestone 水平线
pickeps = eps[::len(eps)//3][:3]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, e in zip(axes, pickeps):
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; f = F128[o]; t = T[o]
    poly, step, rf, rv = daw(f, C, P, lam, want_step=True)
    x = np.arange(len(f))
    for pv in P: ax.axhline(pv, color='#bbb', lw=.6, ls=':')
    ax.plot(x, t, color='#e8830c', lw=1.2, alpha=.6, label='norm time')
    ax.plot(x, step, color='#888', lw=1.0, alpha=.9, label='step (staircase Viterbi)')
    ax.plot(x, poly, color='#2ca02c', lw=2.0, label='polyline (de-staircased)')
    ax.plot(rf, rv, 'ko', ms=4, label='rep-frame knots')
    ax.set_title(f'kai ep{e}  step-vs-time corr={cc(step,t):.3f}', fontsize=9); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.2)
axes[0].legend(fontsize=7, loc='lower right')
fig.suptitle(f'kai teacher 诊断: M={len(C)} milestones (灰点线=milestone median-T 值) · 灰=step 阶梯 / 绿=polyline', fontsize=11)
fig.tight_layout(); outp = REPO/'lmvla/crave/temp/diag_kai_teacher.png'
fig.savefig(outp, dpi=120, bbox_inches='tight'); print('SAVED', outp, flush=True)
