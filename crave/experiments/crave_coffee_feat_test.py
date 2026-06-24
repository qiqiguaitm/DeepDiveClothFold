"""决定性测试: 旧 small 特征(raw⊕armmask⊕state, 3Hz feat_cache) vs 我的 large 特征, 同一读出, 看 coffee ep0 是否单调。
隔离: 是特征(small 3路)还是配方/全帧 导致 ep0 别名崩。

Thin entrypoint over `crave`: `mkp`/`L2`/`med`/`otsu`/`smooth_monotone` come from
`crave.utils`, `gpu_kmeans` from `crave.clustering`, the coffee 3Hz feature cache path
from `crave.config.resolve_dataset`. `build_clusters` + `readout_viterbi_ms` are the
legacy `crave_align_analyze` helpers (no library equivalent yet) re-inlined below — see
TODOs.

跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_coffee_feat_test.py
"""
from pathlib import Path

import numpy as np

from crave.clustering import gpu_kmeans
from crave.config import resolve_dataset
from crave.render import setup_mpl
from crave.utils import L2, med, mkp, otsu, smooth_monotone

# coffee 3-path feature cache (raw/armmask/state @ ~3Hz) — exposed as the coffee
# DatasetConfig.statecache in crave.config.datasets.
FC = Path(resolve_dataset("coffee").statecache)

BINS = np.linspace(0, 1, 41)


# TODO(crave-lib): build_clusters (full-frame cluster → precedence/isotonic milestone
# selection, returns the `cl` dict consumed by the readout_* variants) should move into
# crave.clustering — it is the crave_align_analyze replica of crave_generalize.main.
def build_clusters(F, E, Tv, ne, seed=0):
    N = len(F)
    K0 = int(np.clip(round(0.55 * np.sqrt(N)), 64, 320))
    cen, lab = gpu_kmeans(F, K0, seed=seed)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(E[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov)
    tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel); eps_sorted = sorted(set(E.tolist()))
    fe = np.full((len(eps_sorted), M), np.nan)
    for ei, e in enumerate(eps_sorted):
        fi = np.where(E == e)[0]; labe = lab[fi]; te = Tv[fi]
        for m in range(M):
            hit = te[labe == sel[m]]
            if len(hit): fe[ei, m] = hit.min()
    Pk = np.array([np.nanmedian(fe[:, m]) for m in range(M)])
    Pbef = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i != j:
                both = np.isfinite(fe[:, i]) & np.isfinite(fe[:, j])
                if both.sum() >= 5: Pbef[i, j] = float(np.mean(fe[both, i] < fe[both, j]))
    soft = np.nansum(np.where(np.isnan(Pbef), 0.0, Pbef), 1); prec = list(np.argsort(-soft))
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec])
    Pord = np.asarray(iso, float)
    order = [sel[p] for p in prec]; C = cen[order]  # milestone centroids in value order
    # start-anchor km (sk) as production
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    cb = [int(np.argmin(abs(BINS - Pord[m]))) for m in range(M)]
    return dict(K0=K0, cen=cen, lab=lab, sel=sel, order=order, M=M, C=C, Pord=Pord,
                cb=cb, sk=sk, tstd=tstd, cov=cov, tpos=tpos, tau_pur=tau_pur, tau_cov=tau_cov)


# TODO(crave-lib): readout_viterbi_ms (forward-aware hard-start Viterbi DP directly over M
# milestone centers, value = Pord[assigned]) should move into crave (e.g.
# crave.value / crave.utils) — it is the crave_align_analyze readout variant.
def readout_viterbi_ms(Fq, cl, lam=8.0, fps=3.0):
    """Fix (ii): Viterbi DP directly over M milestones; emit=dist, transition=lam*|Pord[i]-Pord[j]|.
    value = Pord[assigned] (no bins, no value->bin->ms indirection)."""
    C, sk, Pord, M = cl["C"], cl["sk"], cl["Pord"], cl["M"]
    nn = len(Fq)
    emit = np.linalg.norm(Fq[:, None] - C[None], axis=2)  # (nn,M) dist to each milestone
    dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
    emit[:, 0] = np.minimum(emit[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))  # start anchor -> milestone 0
    pen = lam * np.abs(Pord[:, None] - Pord[None])  # (M,M)
    cost = np.full(M, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((nn, M), int)  # HARD start: frame0=milestone0(value0), prevents start aliasing
    for j in range(1, nn):
        tr = cost[None, :] + pen          # tr[i,k] = cost_prev[k] + pen[i,k]
        k = tr.argmin(1); cost = emit[j] + tr[np.arange(M), k]; bp[j] = k
    ms = np.zeros(nn, int); ms[-1] = int(cost.argmin())
    for j in range(nn - 2, -1, -1): ms[j] = bp[j + 1, ms[j + 1]]
    mw = max(5, int(round(5 * fps / 3))) | 1
    v = smooth_monotone(med(Pord[ms], mw), fps=fps)
    return v.astype(np.float32), ms


def build(eps, mode):
    F, E, T = [], [], []
    Pall = np.concatenate([mkp(np.load(FC / f"ep{e}.npz")["state"]) for e in eps])
    PMU, PSD = Pall.mean(0), Pall.std(0) + 1e-8
    for e in eps:
        d = np.load(FC / f"ep{e}.npz")
        n = min(len(d["raw"]), len(d["armmask"]), len(d["state"]))   # 三路对齐到公共长度
        rn = L2(d["raw"][:n].astype(np.float32)); an = L2(d["armmask"][:n].astype(np.float32))
        pn = L2((mkp(d["state"])[:n] - PMU) / PSD)
        if mode == "small3": f = np.concatenate([rn, an, pn], 1)
        elif mode == "small_raw_proprio": f = np.concatenate([rn, pn], 1)
        F.append(f); E.append(np.full(n, e)); T.append(np.arange(n) / max(1, n - 1))
    return np.concatenate(F), np.concatenate(E), np.concatenate(T)


def main():
    plt = setup_mpl()
    eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for row, mode in enumerate(["small3", "small_raw_proprio"]):
        F, E, T = build(eps, mode); cl = build_clusters(F, E, T, len(eps))
        for col, e in enumerate([0, 1]):
            fi = np.where(E == e)[0]; fi = fi[np.argsort(T[fi])]; Fq = F[fi]
            v, ms = readout_viterbi_ms(Fq, cl, lam=8.0, fps=3.0)
            corr = float(np.corrcoef(v, T[fi])[0, 1]) if v.std() > 0 else 0
            ax = axes[row, col]; ax.plot(v, color="#1a7f37", lw=2); ax.plot(np.linspace(0, 1, len(v)), color="0.6", ls="--", lw=1)
            ax.set_ylim(-.02, 1.02); ax.set_title(f"[{mode}] coffee ep{e}  M={cl['M']}  corr(v,t)={corr:.3f}", fontsize=10); ax.grid(alpha=.3)
    fig.suptitle("旧 small 特征 + 我的读出: coffee ep0/ep1 是否单调(对比我 large 双路 ep0 崩)", fontsize=12)
    from crave.config import REPO
    out = REPO / "temp/crave_align/coffee_feat_test.png"
    fig.tight_layout(); fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig); print("SAVED", out)


if __name__ == "__main__":
    main()
