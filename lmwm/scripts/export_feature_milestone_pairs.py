#!/usr/bin/env python
"""Build multi-episode CRAVE-style milestone pair datasets from cached features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import load_config  # noqa: E402


def make_state_delta(state: np.ndarray) -> np.ndarray:
    d = np.zeros_like(state)
    if len(state) > 1:
        d[1:] = state[1:] - state[:-1]
    return np.concatenate([state, d], axis=1)


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def load_episode(path: Path, pmu: np.ndarray | None = None, psd: np.ndarray | None = None) -> tuple[int, np.ndarray, np.ndarray]:
    ep = int(path.stem[2:])
    z = np.load(path)
    raw = z["raw"].astype(np.float32)
    armmask = z["armmask"].astype(np.float32)
    state = np.clip(np.nan_to_num(z["state"].astype(np.float32)), -10, 10)
    n = min(len(raw), len(armmask), len(state))
    raw, armmask, state = raw[:n], armmask[:n], state[:n]
    prop = make_state_delta(state)
    if pmu is not None and psd is not None:
        prop = (prop - pmu) / psd
    feat = np.concatenate([l2(raw), l2(armmask), l2(prop)], axis=1).astype(np.float32)
    return ep, feat, prop.astype(np.float32)


def next_unique_indices(milestones: np.ndarray) -> np.ndarray:
    n = len(milestones)
    fut = np.full(n, -1, dtype=np.int64)
    next_change = -1
    for i in range(n - 2, -1, -1):
        if milestones[i + 1] != milestones[i]:
            next_change = i + 1
        fut[i] = next_change
    return fut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(int(cfg.get("seed", 2026)))

    cache = Path(cfg["feature_cache"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(cache.glob("ep*.npz"), key=lambda p: int(p.stem[2:]))
    if not paths:
        raise FileNotFoundError(f"no ep*.npz found under {cache}")

    # Proprio normalization over all cached episodes.
    props = []
    for p in paths:
        z = np.load(p)
        props.append(make_state_delta(np.clip(np.nan_to_num(z["state"].astype(np.float32)), -10, 10)))
    all_prop = np.concatenate(props, axis=0)
    pmu = all_prop.mean(axis=0)
    psd = all_prop.std(axis=0) + 1e-8

    eps: list[int] = []
    feats: list[np.ndarray] = []
    lengths: list[int] = []
    for p in paths:
        ep, feat, _ = load_episode(p, pmu, psd)
        eps.append(ep)
        feats.append(feat)
        lengths.append(len(feat))
    all_feat = np.concatenate(feats, axis=0)
    frame_episode = np.concatenate([np.full(n, ep, dtype=np.int64) for ep, n in zip(eps, lengths)])
    frame_tnorm = np.concatenate([np.arange(n, dtype=np.float32) / max(1, n - 1) for n in lengths])

    num_clusters = int(cfg.get("num_clusters", 96))
    km = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=int(cfg.get("seed", 2026)),
        batch_size=int(cfg.get("kmeans_batch_size", 8192)),
        n_init=3,
        max_iter=200,
    )
    labels = km.fit_predict(all_feat)
    centers = km.cluster_centers_.astype(np.float32)

    cov = np.zeros(num_clusters, dtype=np.float32)
    tpos = np.zeros(num_clusters, dtype=np.float32)
    all_eps = set(eps)
    for c in range(num_clusters):
        m = labels == c
        if m.any():
            cov[c] = len(set(frame_episode[m].tolist())) / max(1, len(all_eps))
            tpos[c] = float(frame_tnorm[m].mean())
        else:
            tpos[c] = 0.5

    num_milestones = min(int(cfg.get("num_milestones", 64)), num_clusters)
    selected = sorted(np.argsort(-cov)[:num_milestones].tolist(), key=lambda c: tpos[c])
    proto = centers[selected].astype(np.float32)
    pord = np.array([tpos[c] for c in selected], dtype=np.float32)

    # Assign every frame to selected milestone centers.
    milestone_seqs: dict[int, np.ndarray] = {}
    offset = 0
    for ep, feat, n in zip(eps, feats, lengths):
        d = np.linalg.norm(feat[:, None, :] - proto[None, :, :], axis=2)
        milestone_seqs[ep] = d.argmin(axis=1).astype(np.int64)
        offset += n

    pair_mode = str(cfg.get("pair_mode", "fixed_horizon"))
    horizon = int(cfg.get("horizon", 8))
    cur_rows = []
    fut_rows = []
    cur_m_rows = []
    fut_m_rows = []
    t_rows = []
    ft_rows = []
    ep_rows = []
    for ep in eps:
        ms = milestone_seqs[ep]
        n = len(ms)
        if pair_mode == "fixed_horizon":
            t = np.arange(0, max(0, n - horizon), dtype=np.int64)
            ft = t + horizon
            suffix = f"fixed_h{horizon}"
        elif pair_mode == "next_unique":
            fut = next_unique_indices(ms)
            t = np.where(fut >= 0)[0].astype(np.int64)
            ft = fut[t].astype(np.int64)
            suffix = "next_unique"
        else:
            raise ValueError(f"unsupported pair_mode={pair_mode}")
        if len(t) == 0:
            continue
        cm = ms[t]
        fm = ms[ft]
        cur_rows.append(proto[cm])
        fut_rows.append(proto[fm])
        cur_m_rows.append(cm)
        fut_m_rows.append(fm)
        t_rows.append(t)
        ft_rows.append(ft)
        ep_rows.append(np.full(len(t), ep, dtype=np.int64))

    current = np.concatenate(cur_rows, axis=0)
    future = np.concatenate(fut_rows, axis=0)
    current_m = np.concatenate(cur_m_rows, axis=0)
    future_m = np.concatenate(fut_m_rows, axis=0)
    t = np.concatenate(t_rows, axis=0)
    ft = np.concatenate(ft_rows, axis=0)
    ep_arr = np.concatenate(ep_rows, axis=0)

    perm = rng.permutation(len(current))
    current = current[perm]
    future = future[perm]
    current_m = current_m[perm]
    future_m = future_m[perm]
    t = t[perm]
    ft = ft[perm]
    ep_arr = ep_arr[perm]

    out_npz = out_dir / f"pairs_{suffix}.npz"
    np.savez_compressed(
        out_npz,
        current=current.astype(np.float32),
        future=future.astype(np.float32),
        current_milestone=current_m.astype(np.int64),
        future_milestone=future_m.astype(np.int64),
        t=t.astype(np.int64),
        future_t=ft.astype(np.int64),
        episode_id=ep_arr.astype(np.int64),
        prototype_table=proto.astype(np.float32),
        pord=pord.astype(np.float32),
        selected_clusters=np.array(selected, dtype=np.int64),
        cluster_coverage=cov.astype(np.float32),
        cluster_tpos=tpos.astype(np.float32),
    )

    meta = {
        "name": cfg.get("name", out_dir.name),
        "feature_cache": str(cache),
        "output_npz": str(out_npz),
        "pair_mode": pair_mode,
        "horizon": horizon if pair_mode == "fixed_horizon" else None,
        "num_episodes": len(eps),
        "num_base_like": sum(1 for e in eps if e < 100000),
        "num_dagger_like_offset": sum(1 for e in eps if e >= 100000),
        "num_frames": int(sum(lengths)),
        "num_pairs": int(len(current)),
        "feature_dim": int(current.shape[1]),
        "num_clusters": num_clusters,
        "num_milestones": num_milestones,
        "prototype_source": "kmeans_feature_center",
        "episode_split_required": True,
    }
    (out_dir / f"meta_{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
