"""Per-episode readout variants over a `cl` dict from crave.clustering.build_clusters.

Ported verbatim from crave_align_analyze:
  - readout_production : Viterbi over 41 value-bins, displayed ms = argmin|Pord - value|.
  - readout_direct     : nearest-milestone assignment; value = Pord[assigned].
  - readout_viterbi_ms : Viterbi DP directly over M milestones (hard-start, start-anchor).
"""
from __future__ import annotations

import numpy as np

from crave.clustering.milestones import BINS
from crave.utils.array import med
from crave.utils.dp import viterbi
from crave.utils.signal import smooth_monotone


def readout_production(Fq, cl, fps=3.0):
    C, sk, M, cb, Pord = cl["C"], cl["sk"], cl["M"], cl["cb"], cl["Pord"]
    nn = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nn, 41), 1e3)
    for m in range(M): em[:, cb[m]] = np.minimum(em[:, cb[m]], d[:, m])
    dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
    em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
    mw = max(5, int(round(5 * fps / 3))) | 1
    v = smooth_monotone(med(viterbi(em, BINS, 8.0)[0], mw), fps=fps)
    ms = np.array([int(np.argmin(np.abs(Pord - v[t]))) for t in range(nn)])
    return v, ms


def readout_direct(Fq, cl, fps=3.0):
    C, Pord = cl["C"], cl["Pord"]
    d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
    ms = d.argmin(1)
    return Pord[ms].astype(np.float32), ms


def readout_viterbi_ms(Fq, cl, lam=8.0, fps=3.0):
    C, sk, Pord, M = cl["C"], cl["sk"], cl["Pord"], cl["M"]
    nn = len(Fq)
    emit = np.linalg.norm(Fq[:, None] - C[None], axis=2)
    dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
    emit[:, 0] = np.minimum(emit[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
    pen = lam * np.abs(Pord[:, None] - Pord[None])
    cost = np.full(M, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((nn, M), int)
    for j in range(1, nn):
        tr = cost[None, :] + pen
        k = tr.argmin(1); cost = emit[j] + tr[np.arange(M), k]; bp[j] = k
    ms = np.zeros(nn, int); ms[-1] = int(cost.argmin())
    for j in range(nn - 2, -1, -1): ms[j] = bp[j + 1, ms[j + 1]]
    mw = max(5, int(round(5 * fps / 3))) | 1
    v = smooth_monotone(med(Pord[ms], mw), fps=fps)
    return v.astype(np.float32), ms
