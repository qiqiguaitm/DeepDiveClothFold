"""kai0-family dataset access (kai0_base / smooth800_dagger / kai0_dagger ...).

These datasets use chunked parquet (`chunk-{e//cs:03d}`) + a `top_head` mp4 per episode,
and pair with 3-path feature caches (`tcc_*_{raw,armmask}/feat_cache`, key "f").
Ported from crave_decoder_scale_ablation.{lpst,loadep,camp,crop224,grab_ep,decode_images}.

Everything is parametrized by a `DatasetConfig` (kind="kai0") so the same code serves
every kai0-family dataset; nothing is hardcoded to kai0_base.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from crave.config.datasets import DatasetConfig


@lru_cache(maxsize=8)
def chunks_size(root: str) -> int:
    return json.load(open(Path(root) / "meta/info.json"))["chunks_size"]


def _chunk(e: int, cs: int) -> str:
    return f"chunk-{e // cs:03d}"


def state_subsampled(cfg: DatasetConfig, e: int, n: int) -> np.ndarray:
    """Proprio state, subsampled to n points at stride 10 (legacy `lpst`)."""
    cs = chunks_size(cfg.root)
    pq = Path(cfg.root) / "data" / _chunk(e, cs) / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def video_path(cfg: DatasetConfig, e: int) -> Path:
    cs = chunks_size(cfg.root)
    return Path(cfg.root) / "videos" / _chunk(e, cs) / cfg.cam / f"episode_{e:06d}.mp4"


def crop224(rgb: np.ndarray) -> np.ndarray:
    """Resize so the short side is 224, then center-crop 224×224."""
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])


def grab_ep(cfg: DatasetConfig, e: int, frames30) -> dict:
    """Grab specific native-frame indices from an episode mp4 → {idx: rgb224}."""
    import av
    want = set(int(f) for f in frames30); out = {}
    try:
        c = av.open(str(video_path(cfg, e)))
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = crop224(f.to_ndarray(format="rgb24"))
                if len(out) == len(want): break
        c.close()
    except Exception:
        pass
    return out


def loadep_tcc(cfg: DatasetConfig, e: int):
    """3-path cache reader for kai0-family: (armmask, raw, state, n) from the "f" key."""
    a = np.load(Path(cfg.arm_cache) / f"ep{e}.npz")["f"]
    r = np.load(Path(cfg.raw_cache) / f"ep{e}.npz")["f"]
    n = min(len(a), len(r))
    return a[:n], r[:n], state_subsampled(cfg, e, n), n


def decode_images(cfg: DatasetConfig, pool_idx, E, FR, workers=32, log=True):
    """Parallel (per-episode) pyav decode of 224 crops for a set of (episode, native-frame)
    selections — solves the single-process pyav bottleneck. Returns (imgs224, valid)."""
    t0 = time.time()
    by_ep: dict[int, list] = {}
    for k, i in enumerate(pool_idx):
        by_ep.setdefault(int(E[i]), []).append((k, int(FR[i])))
    imgs224 = np.zeros((len(pool_idx), 224, 224, 3), np.uint8)
    valid = np.zeros(len(pool_idx), bool)

    def work(item):
        e, kfs = item; fm = grab_ep(cfg, e, [f for _, f in kfs])
        return [(k, fm[f]) for k, f in kfs if f in fm]

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(work, list(by_ep.items())):
            for k, im in res:
                imgs224[k] = im; valid[k] = True
            done += 1
            if log and done % 80 == 0:
                print(f"    decoded {done}/{len(by_ep)} eps  ({time.time()-t0:.0f}s)", flush=True)
    return imgs224, valid
