"""Uniformity-aware milestone selection (enhances generalize.build_milestones WITHOUT editing it).

The stock selector keeps clusters by coverage + temporal-purity + a MINIMUM spacing g0, but
nothing caps the MAXIMUM gap → coffee leaves a 0.38-wide void (v=0.56→0.94) with no milestone,
so the value jumps coarsely there. This adds a gap-fill pass that, while any internal gap in
mean-time(tpos) exceeds `max_gap`, admits the best cluster lying inside that gap, scored jointly
by COVERAGE (high), temporal purity (low std) and UNIFORMITY (near the gap center). Clusters that
narrowly missed the global coverage/purity thresholds are eligible here (relaxed pool), so voids
get filled by the most-recurrent available state instead of being left empty.
"""
from __future__ import annotations
import numpy as np
from crave.clustering import gpu_kmeans
from crave.utils import otsu


def build_milestones_uniform(F, E, Tv, ne, max_gap=0.12, cov_relax=0.5, pur_relax=1.5):
    from sklearn.isotonic import IsotonicRegression
    N = len(F)
    K0 = int(np.clip(round(0.55 * np.sqrt(N)), 64, 320))
    cen, lab = gpu_kmeans(F, K0)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(E[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov, tau_pur = otsu(cov), float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    # ---- NEW: uniformity gap-fill (jointly weigh coverage + spread) ----
    pool = [c for c in range(K0) if (lab == c).sum() > 2 and cov[c] >= tau_cov * cov_relax and tstd[c] <= tau_pur * pur_relax]
    nfill = 0
    for _ in range(K0):
        ss = sorted(sel, key=lambda c: tpos[c])
        diffs = [tpos[ss[i + 1]] - tpos[ss[i]] for i in range(len(ss) - 1)]
        if not diffs: break
        gi = int(np.argmax(diffs))
        if diffs[gi] <= max_gap: break
        lo, hi = tpos[ss[gi]], tpos[ss[gi + 1]]; center = (lo + hi) / 2
        inside = [c for c in pool if lo < tpos[c] < hi and c not in sel]
        if not inside: break
        # coverage-primary, purity bonus, uniformity (proximity to gap center) — all normalized to ~[0,1]
        best = max(inside, key=lambda c: cov[c] - 0.5 * tstd[c] - 0.4 * abs(tpos[c] - center) / max(hi - lo, 1e-6))
        sel.append(best); nfill += 1
    # ---- order + isotonic value (identical to generalize.build_milestones) ----
    M = len(sel); eps_sorted = sorted(set(E.tolist()))
    fe = np.full((len(eps_sorted), M), np.nan)
    for ei, e in enumerate(eps_sorted):
        fi = np.where(E == e)[0]; labe, te = lab[fi], Tv[fi]
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
    Pord = np.asarray(IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec]), float)
    order = [sel[p] for p in prec]
    return cen, lab, np.array(order), Pord, M, nfill
