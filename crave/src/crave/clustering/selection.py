"""Milestone selection & ordering from a clustering.

Given cluster labels over frames tagged with (episode, normalized-time), pick which
clusters are *milestones* (recurrent, time-pure waypoints) and assign each a monotone
progress value. Two ordering modes, both ported from crave_value.DiscreteValue:

  - 'time'       : order by median first-arrival time (legacy default).
  - 'precedence' : Copeland pairwise-precedence order + isotonic value
                   (節奏/normalization-invariant; fixes near-tie ordering).

These are pure functions over (labels, episodes, tpos) so any clustering source — the
DiscreteValue path or the full-frame generalize path — can reuse them.
"""
from __future__ import annotations

import numpy as np


def cluster_stats(labels, episodes, tnorm, K, n_eps=None):
    """Per-cluster time position, coverage (frac of episodes touched) and time-purity (std)."""
    if n_eps is None:
        n_eps = len(set(episodes.tolist()))
    tpos = np.array([tnorm[labels == c].mean() if (labels == c).any() else 0.5 for c in range(K)])
    cov = np.array([len(set(episodes[labels == c].tolist())) / n_eps if (labels == c).any() else 0.0 for c in range(K)])
    tstd = np.array([tnorm[labels == c].std() if (labels == c).sum() > 2 else 9.0 for c in range(K)])
    return tpos, cov, tstd


def runs(idx):
    """Contiguous runs (start,end) of a sorted frame-index list, length>=1."""
    o = []
    s0 = pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s0 is not None:
                o.append((s0, pv))
            s0 = i
        pv = i
    if s0 is not None:
        o.append((s0, pv))
    return [x for x in o if x[1] - x[0] >= 1]


def first_arrival_matrix(sel, labels, episodes, tnorm):
    """(n_eps, n_sel) median first-arrival time per episode×selected-cluster (NaN if absent)."""
    eps_sorted = sorted(set(episodes.tolist()))
    fe = np.full((len(eps_sorted), len(sel)), np.nan)
    for ei, e in enumerate(eps_sorted):
        m = np.where(episodes == e)[0]
        for si, c in enumerate(sel):
            rs = runs(m[labels[m] == c].tolist())
            if rs:
                fe[ei, si] = tnorm[rs[0][0]]
    return fe, eps_sorted


def precedence_order(fe_mat, pk, min_co=5):
    """Copeland soft-precedence order + isotonic value over selected clusters.

    Args:
        fe_mat: (n_eps, n_sel) first-arrival times.
        pk:     (n_sel,) fallback time position per selected cluster.
        min_co: min co-occurring episodes for a pairwise comparison.
    Returns:
        (order_idx, value) — permutation of selected-cluster indices and their isotonic values.
    """
    from sklearn.isotonic import IsotonicRegression
    ns = fe_mat.shape[1]
    Pbef = np.full((ns, ns), np.nan)
    for i in range(ns):
        for j in range(ns):
            if i == j:
                continue
            both = np.isfinite(fe_mat[:, i]) & np.isfinite(fe_mat[:, j])
            if both.sum() >= min_co:
                Pbef[i, j] = float(np.mean(fe_mat[both, i] < fe_mat[both, j]))
    soft = np.nansum(np.where(np.isnan(Pbef), 0.0, Pbef), axis=1)   # higher = earlier
    prec = list(np.argsort(-soft))
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(ns), np.array([pk[si] for si in prec]))
    return prec, np.asarray(iso, float)
