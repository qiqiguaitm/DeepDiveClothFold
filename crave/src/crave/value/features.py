"""FeatureSpace — the 3-path CRAVE embedding (raw-DINO ⊕ armmask-DINO ⊕ proprio).

Ported verbatim from crave_value.FeatureSpace. Proprio normalization stats (PMU/PSD)
come from the mining episodes so the embedding is comparable across episodes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from crave.data.cache import loadep
from crave.utils.array import mkp


class FeatureSpace:
    """raw-DINOv2 ⊕ armmask-DINOv2 ⊕ proprio(state+Δstate, z-score), each L2-normalized."""

    def __init__(self, fc: Path, mine_eps):
        self.fc = Path(fc)
        Pm = mkp(np.concatenate([loadep(self.fc, e)[2] for e in mine_eps]))
        self.PMU, self.PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(self, a_, r_, s_):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True)
        rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = (mkp(s_) - self.PMU) / self.PSD
        Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    def emb_ep(self, e):
        a, r, s, _ = loadep(self.fc, e)
        return self.emb(a, r, s)
