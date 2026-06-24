"""KMeans implementations.

`gpu_kmeans` — chunked Lloyd's on the GPU (torch), for full-frame feature matrices that
are too large for sklearn. Ported verbatim from crave_generalize.py.
`cpu_kmeans` — thin sklearn wrapper for small problems (matches the DiscreteValue path).
"""
from __future__ import annotations

import numpy as np


def gpu_kmeans(F: np.ndarray, K: int, iters: int = 25, chunk: int = 30000, seed: int = 0):
    """GPU KMeans. Returns (centers (K,D) float32, labels (N,) int64) as numpy."""
    import torch
    g = torch.device("cuda")
    Fg = torch.from_numpy(np.ascontiguousarray(F, np.float32)).to(g)
    Nn = Fg.shape[0]
    torch.manual_seed(seed)
    C = Fg[torch.randperm(Nn, device=g)[:K]].clone()
    lab = torch.zeros(Nn, dtype=torch.long, device=g)
    for _ in range(iters):
        Cn = (C * C).sum(1)
        for s in range(0, Nn, chunk):
            Fc = Fg[s:s + chunk]
            lab[s:s + chunk] = ((Fc * Fc).sum(1, keepdim=True) - 2 * Fc @ C.T + Cn[None]).argmin(1)
        nc = torch.zeros_like(C)
        cnt = torch.zeros(K, device=g)
        nc.index_add_(0, lab, Fg)
        cnt.index_add_(0, lab, torch.ones(Nn, device=g))
        msk = cnt > 0
        C[msk] = nc[msk] / cnt[msk, None]
    return C.cpu().numpy(), lab.cpu().numpy()


def cpu_kmeans(F: np.ndarray, K: int, n_init: int = 2, seed: int = 0):
    """sklearn KMeans → (centers, labels)."""
    from sklearn.cluster import KMeans
    km = KMeans(K, n_init=n_init, random_state=seed).fit(F)
    return km.cluster_centers_, km.labels_
