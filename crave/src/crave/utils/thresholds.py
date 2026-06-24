"""Threshold helpers (Otsu) used for data-driven coverage/purity cutoffs."""
from __future__ import annotations

import numpy as np


def otsu(xs: np.ndarray) -> float:
    """Otsu's method: the threshold maximizing between-class variance over `xs`.

    Used to split cluster coverage / purity into 'milestone' vs 'noise' without a
    hand-tuned cutoff. Returns a value from the sorted uniques of `xs`.
    """
    xs = np.asarray(xs)
    s = np.unique(np.sort(xs))
    bt, bv = s[0], -1.0
    for t in s:
        lo, hi = xs[xs < t], xs[xs >= t]
        if len(lo) and len(hi):
            v = (len(lo) / len(xs)) * (len(hi) / len(xs)) * (lo.mean() - hi.mean()) ** 2
            if v > bv:
                bv, bt = v, t
    return float(bt)
