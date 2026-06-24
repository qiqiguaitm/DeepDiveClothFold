"""Array / signal helpers shared across CRAVE (ported verbatim from the legacy crave_* scripts).

These are deliberately dependency-light (numpy only) so every layer can use them.
"""
from __future__ import annotations

import numpy as np


def L2(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize (eps-safe)."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def mkp(s: np.ndarray) -> np.ndarray:
    """Proprio feature: concat state with its per-step delta (Δ via diff, first row 0).

    Doubles the state dimension so velocity is represented. Used by FeatureSpace.
    """
    return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def mkp_gap(s: np.ndarray, g: int) -> np.ndarray:
    """Like mkp but the delta spans a `g`-frame gap.

    When reading out at full rate (e.g. 30Hz) but clustering was defined at a coarser
    stride, this keeps the velocity Δ on the same scale as the clustered features.
    """
    d = np.zeros_like(s)
    if g >= 1:
        d[g:] = s[g:] - s[:-g]
    return np.concatenate([s, d], 1)


def med(a: np.ndarray, w: int) -> np.ndarray:
    """Centered median filter over window `w` (edge-truncated)."""
    a = np.asarray(a)
    h = w // 2
    return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


def advantage(v: np.ndarray, W: int = 50) -> np.ndarray:
    """Forward advantage: v[t+W] - v[t] (clamped at the tail)."""
    return np.array([v[min(i + W, len(v) - 1)] - v[i] for i in range(len(v))])


def mono(v: np.ndarray) -> float:
    """Fraction of steps that are non-decreasing (monotonicity score)."""
    return float(np.mean(np.diff(v) >= -1e-6))


def adv_density(v: np.ndarray, W: int = 50) -> float:
    """Fraction of frames whose |advantage| is non-trivial — measures how 'live' the signal is."""
    return float(np.mean(np.abs(np.clip(advantage(v, W), -1, 1)) > 1e-3))
