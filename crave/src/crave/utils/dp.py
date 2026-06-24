"""Viterbi dynamic-programming readout over a progress ladder.

Two variants, both ported from the legacy scripts:
  - `viterbi`            : symmetric transition penalty `lam*|Δbin|` + end-格 bonus (crave_value.py).
  - `viterbi_forward`    : asymmetric penalty — cheap to go up, expensive to go back —
                           the validated "forward-biased" readout (crave_generalize.py).
"""
from __future__ import annotations

import numpy as np


def viterbi(emit: np.ndarray, bins: np.ndarray, lam: float, end_bonus: float = 2.0):
    """Generic Viterbi-DP.

    Args:
        emit: (NF, NB) per-frame cost of being at each progress bin.
        bins: (NB,) the progress values (e.g. linspace 0..1).
        lam:  transition penalty weight (|Δbin| * lam).
        end_bonus: subtracted from the last bin's terminal cost (mild pull to完成).
    Returns:
        (values, path) where values = bins[path], path = bin index per frame.
    """
    NB = len(bins)
    pen = lam * np.abs(bins[:, None] - bins[None])
    NF = len(emit)
    cost = np.full(NB, 1e9)
    cost[0] = emit[0, 0]
    bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen
        k = tr.argmin(1)
        cost = emit[j] + tr[np.arange(NB), k]
        bp[j] = k
    cost[NB - 1] -= end_bonus
    path = np.zeros(NF, int)
    path[-1] = int(cost.argmin())
    for j in range(NF - 2, -1, -1):
        path[j] = bp[j + 1, path[j + 1]]
    return bins[path], path


def forward_penalty(values: np.ndarray, up: float = 3.0, down: float = 25.0) -> np.ndarray:
    """Asymmetric transition penalty matrix over states with given `values`.

    Going forward (value increases) costs `up*Δ`; going backward costs `down*Δ`.
    Default (3 vs 25) is the validated forward-biased setting.
    """
    dv = values[:, None] - values[None]          # value_i - value_k
    return np.where(dv >= 0, up * dv, down * (-dv))


def viterbi_forward(emit: np.ndarray, values: np.ndarray, up: float = 3.0, down: float = 25.0,
                    hard_start: bool = True):
    """Forward-biased Viterbi over arbitrary states (e.g. milestone centers).

    Args:
        emit:   (NF, S) per-frame cost to each state (e.g. distance to milestone center).
        values: (S,) progress value of each state.
        up/down: asymmetric penalty (forward cheap, backward expensive).
        hard_start: force frame 0 into the lowest-value state (prevents start aliasing).
    Returns:
        (states, ) — assigned state index per frame. Map through `values` for the curve.
    """
    NF, S = emit.shape
    pen = forward_penalty(values, up, down)
    cost = np.full(S, 1e9)
    s0 = int(values.argmin()) if hard_start else None
    if hard_start:
        cost[s0] = emit[0, s0]
    else:
        cost = emit[0].copy()
    bp = np.zeros((NF, S), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen
        k = tr.argmin(1)
        cost = emit[j] + tr[np.arange(S), k]
        bp[j] = k
    st = np.zeros(NF, int)
    st[-1] = int(cost.argmin())
    for j in range(NF - 2, -1, -1):
        st[j] = bp[j + 1, st[j + 1]]
    return st
