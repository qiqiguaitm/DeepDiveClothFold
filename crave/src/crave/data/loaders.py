"""Per-dataset frame/state loaders for the three formats (lerobot2 / hdf5 / lerobotv3).

Ported from crave_generalize.{load_ep,list_eps,load_ep_native}; takes a DatasetConfig.
Returns RGB uint8 frames + proprio state; the encoder owns any further resize.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from crave.config.datasets import DatasetConfig


def list_eps(cfg: DatasetConfig):
    if cfg.kind == "lerobot2":
        eps = sorted(int(p.stem[8:]) for p in (Path(cfg.root) / "data/chunk-000").glob("episode_*.parquet"))
    elif cfg.kind == "hdf5":
        eps = sorted(int(p.stem.split("_")[1]) for p in Path(cfg.root).glob("episode_*.hdf5"))
    elif cfg.kind == "lerobotv3":
        eps = sorted(int(p.stem[2:]) for p in Path(cfg.statecache).glob("ep*.npz"))
    else:
        raise ValueError(f"unknown kind {cfg.kind}")
    return eps[:cfg.maxep]


def load_ep(cfg: DatasetConfig, e: int, strd: int | None = None):
    """Sub-sampled (stride) load → (frames224_rgb, state(n,D), thumb128, native_idx)."""
    st = strd if strd is not None else cfg.stride
    if cfg.kind == "lerobot2":
        root = Path(cfg.root)
        df = pd.read_parquet(root / "data/chunk-000" / f"episode_{e:06d}.parquet", columns=["observation.state"])
        state_full = np.stack(df["observation.state"].to_numpy())
        cap = cv2.VideoCapture(str(root / "videos/chunk-000" / cfg.cam / f"episode_{e:06d}.mp4"))
        frames = []; i = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % st == 0: frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
    elif cfg.kind == "hdf5":
        import h5py
        with h5py.File(Path(cfg.root) / f"episode_{e}.hdf5", "r") as h:
            state_full = h["observations/qpos"][:]
            jpg = h["observations/images/" + cfg.cam]
            frames = [cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg[i], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                      for i in range(0, len(state_full), st)]
    elif cfg.kind == "lerobotv3":
        import av
        epm = pd.read_parquet(Path(cfg.root) / "meta/episodes/chunk-000/file-000.parquet")
        row = epm[epm["episode_index"] == e].iloc[0]; f0, f1 = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        s3 = np.load(Path(cfg.statecache) / f"ep{e}.npz")["state"]
        cont = av.open(str(Path(cfg.root) / "videos" / cfg.cam / "chunk-000/file-000.mp4")); frames = []
        for gi, fr in enumerate(cont.decode(video=0)):
            if gi >= f1: break
            if gi >= f0 and (gi - f0) % st == 0: frames.append(fr.to_ndarray(format="rgb24"))
        cont.close(); n = len(frames)
        xs = np.linspace(0, 1, len(s3)); xo = np.linspace(0, 1, n)
        state = np.stack([np.interp(xo, xs, s3[:, j]) for j in range(s3.shape[1])], 1)
        f224 = [cv2.resize(f, (224, 224)) for f in frames]; th = [cv2.resize(f, (128, 128)) for f in frames]
        return f224, state, th, np.arange(n)
    else:
        raise ValueError(f"unknown kind {cfg.kind}")
    idx = np.arange(0, len(state_full), st)
    state = state_full[idx]; n = min(len(frames), len(state)); frames = frames[:n]; state = state[:n]
    f224 = [cv2.resize(f, (224, 224)) for f in frames]; th = [cv2.resize(f, (128, 128)) for f in frames]
    return f224, state, th, idx[:n]


def load_ep_native(cfg: DatasetConfig, e: int):
    """Native frame rate (all frames) → (frames224_rgb, state(n,D), fps)."""
    if cfg.kind == "lerobot2":
        root = Path(cfg.root)
        state = np.stack(pd.read_parquet(root / "data/chunk-000" / f"episode_{e:06d}.parquet",
                                         columns=["observation.state"])["observation.state"].to_numpy())
        cap = cv2.VideoCapture(str(root / "videos/chunk-000" / cfg.cam / f"episode_{e:06d}.mp4")); frames = []
        while True:
            ok, fr = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release(); fps = 30
    elif cfg.kind == "hdf5":
        import h5py
        with h5py.File(Path(cfg.root) / f"episode_{e}.hdf5", "r") as h:
            state = h["observations/qpos"][:]; jpg = h["observations/images/" + cfg.cam]
            frames = [cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg[i], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                      for i in range(len(state))]
        fps = 30
    elif cfg.kind == "lerobotv3":
        import av
        epm = pd.read_parquet(Path(cfg.root) / "meta/episodes/chunk-000/file-000.parquet")
        row = epm[epm["episode_index"] == e].iloc[0]; f0, f1 = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        cont = av.open(str(Path(cfg.root) / "videos" / cfg.cam / "chunk-000/file-000.mp4")); frames = []
        for gi, fr in enumerate(cont.decode(video=0)):
            if gi >= f1: break
            if gi >= f0: frames.append(fr.to_ndarray(format="rgb24"))
        cont.close()
        s3 = np.load(Path(cfg.statecache) / f"ep{e}.npz")["state"]
        xs = np.linspace(0, 1, len(s3)); xo = np.linspace(0, 1, len(frames))
        state = np.stack([np.interp(xo, xs, s3[:, j]) for j in range(s3.shape[1])], 1); fps = 50
    else:
        raise ValueError(f"unknown kind {cfg.kind}")
    n = min(len(frames), len(state)); frames = frames[:n]; state = state[:n]
    return [cv2.resize(f, (224, 224)) for f in frames], state, fps
