"""Full-frame milestone builder — cluster → coverage/purity/spacing select →
precedence+isotonic order, returning a `cl` dict consumed by the readout variants.

Ported verbatim from crave_align_analyze.build_clusters. This is the shared milestone
stage behind both the generalize pipeline and the align/diagnostic scripts.
"""
from __future__ import annotations

import numpy as np

from crave.clustering.kmeans import gpu_kmeans
from crave.utils.thresholds import otsu

BINS = np.linspace(0, 1, 41)   # value-bin grid used by the production readout


def build_clusters(F, E, Tv, ne, seed=0) -> dict:
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
    order = [sel[p] for p in prec]; C = cen[order]
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    cb = [int(np.argmin(abs(BINS - Pord[m]))) for m in range(M)]
    return dict(K0=K0, cen=cen, lab=lab, sel=sel, order=order, M=M, C=C, Pord=Pord,
                cb=cb, sk=sk, tstd=tstd, cov=cov, tpos=tpos, tau_pur=tau_pur, tau_cov=tau_cov)
