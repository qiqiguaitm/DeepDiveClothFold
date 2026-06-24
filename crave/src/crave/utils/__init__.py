"""Dependency-light numpy helpers shared across all CRAVE layers."""
from crave.utils.array import L2, advantage, adv_density, med, mkp, mkp_gap, mono
from crave.utils.dp import forward_penalty, viterbi, viterbi_forward
from crave.utils.signal import smooth_monotone
from crave.utils.thresholds import otsu

__all__ = [
    "L2", "mkp", "mkp_gap", "med", "advantage", "mono", "adv_density",
    "viterbi", "viterbi_forward", "forward_penalty",
    "smooth_monotone", "otsu",
]
