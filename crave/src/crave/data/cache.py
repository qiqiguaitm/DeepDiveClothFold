"""Feature-cache IO. The 3-path cache stores per-episode `ep{e}.npz` with keys
`raw` (DINOv2 raw frame), `armmask` (arm-masked frame), `state` (proprio), all @ ~3Hz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def loadep(fc: Path, e: int):
    """Load one episode's cached features, truncated to the common length.

    Returns (armmask, raw, state, n).
    """
    d = np.load(Path(fc) / f"ep{e}.npz")
    a, r, s = d["armmask"], d["raw"], d["state"]
    n = min(len(a), len(r), len(s))
    return a[:n], r[:n], s[:n], n


def list_cache_eps(fc: Path):
    """Sorted episode ids present in a feature cache dir."""
    return sorted(int(p.stem[2:]) for p in Path(fc).glob("ep*.npz"))


def load_full_shards(shard_dir: Path, enc: str = "dino"):
    """Load a full-scale sharded encode (temp/crave_full).

    Layout: `index_{enc}.npz` with per-frame arrays E/FR/T (+ per-episode `n`), and
    `{enc}/shard_*.npz` shards with `gidx` (global row indices), `feat`, `valid`.
    Features are scattered back into one (N,D) array by `gidx`.

    Returns (feat (N,D), valid (N,) bool, E, FR, T).
    """
    shard_dir = Path(shard_dir)
    idx = dict(np.load(shard_dir / f"index_{enc}.npz"))
    E, FR, T = idx["E"], idx["FR"], idx["T"]
    N = len(E)
    feat = None
    valid = np.zeros(N, bool)
    for s in sorted((shard_dir / enc).glob("shard_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
        d = np.load(s); g = d["gidx"]; f = d["feat"]; v = d["valid"]
        if feat is None:
            feat = np.zeros((N, f.shape[1]), f.dtype)
        feat[g] = f
        valid[g] = v
    return feat, valid, E, FR, T


# Back-compat alias (older name); prefer load_full_shards.
def load_dino_shards(shard_dir: Path):
    return load_full_shards(shard_dir, enc="dino")
