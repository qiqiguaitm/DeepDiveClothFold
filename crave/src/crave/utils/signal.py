"""Continuous readout smoothing — ported verbatim from crave_readout.py.

`smooth_monotone` is the single shared continuization used by all CRAVE visualizers /
labelers, so a DP step-ladder never gets plotted raw. It preserves structure (incl.
genuine backward steps from recurrent milestones) and only removes the hard-step look.
"""
from __future__ import annotations

import numpy as np


def smooth_monotone(v: np.ndarray, fps: float = 30.0, w: int | None = None) -> np.ndarray:
    """Edge-padded moving average + re-clip to [0, 1].

    Window scales with frame rate (baseline 41 @ 30fps → 4 @ 3Hz) unless `w` is given.
    Despite the name it is NOT forced monotone — backward dips (recurrent milestones)
    survive; only the staircase quantization is smoothed.
    """
    v = np.asarray(v, dtype=np.float64)
    if w is None:
        w = max(2, int(round(41 * fps / 30.0)))
    if len(v) < 3 or w < 2:
        return v.astype(np.float32)
    h = w // 2
    vp = np.concatenate([np.full(h, v[0]), v, np.full(h, v[-1])])
    k = np.ones(w, dtype=np.float64) / w
    vs = np.convolve(vp, k, mode="valid")[: len(v)]
    return np.clip(vs, 0.0, 1.0).astype(np.float32)
