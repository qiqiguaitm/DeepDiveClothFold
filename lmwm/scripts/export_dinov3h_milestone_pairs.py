#!/usr/bin/env python
"""Export LaWM-shaped LMWM pairs from cached DINOv3-H CRAVE features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import load_config  # noqa: E402


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def load_full_features(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.load(feature_dir / "index.npz")
    e = idx["E"].astype(np.int64)
    fr = idx["FR"].astype(np.int64)
    tnorm = idx["T"].astype(np.float32)
    n = int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        gidx = z["gidx"].astype(np.int64)
        feat[gidx] = z["feat"]
        valid[gidx] = z["valid"].astype(bool)
    return e[valid], fr[valid], tnorm[valid], l2(feat[valid].astype(np.float32))


def next_unique_indices(ms: np.ndarray) -> np.ndarray:
    n = len(ms)
    fut = np.full(n, -1, dtype=np.int64)
    next_change = -1
    for i in range(n - 2, -1, -1):
        if ms[i + 1] != ms[i]:
            next_change = i + 1
        fut[i] = next_change
    return fut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(int(cfg.get("seed", 2026)))

    feature_dir = Path(cfg["feature_dir"])
    milestone_file = Path(cfg["milestone_file"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    E, FR, T, F = load_full_features(feature_dir)
    mz = np.load(milestone_file)
    proto = l2(mz["C"].astype(np.float32))
    pord = mz["Pord"].astype(np.float32)
    num_milestones = len(proto)

    # Assign every valid frame to nearest DINOv3-H milestone center.
    assign = np.empty(len(F), dtype=np.int64)
    for i in range(0, len(F), 32768):
        d = np.linalg.norm(F[i:i + 32768, None, :] - proto[None, :, :], axis=2)
        assign[i:i + 32768] = d.argmin(axis=1)

    pair_mode = str(cfg.get("pair_mode", "fixed_horizon"))
    horizon = int(cfg.get("horizon", 3))
    rows = []
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]
        order = loc[np.argsort(FR[loc])]
        ms = assign[order]
        if pair_mode == "fixed_horizon":
            t_local = np.arange(0, max(0, len(order) - horizon), dtype=np.int64)
            f_local = t_local + horizon
            suffix = f"fixed_h{horizon}"
        elif pair_mode == "next_unique":
            fut = next_unique_indices(ms)
            t_local = np.where(fut >= 0)[0].astype(np.int64)
            f_local = fut[t_local].astype(np.int64)
            suffix = "next_unique"
        else:
            raise ValueError(f"unsupported pair_mode={pair_mode}")
        if len(t_local) == 0:
            continue
        cur_g = order[t_local]
        fut_g = order[f_local]
        rows.append((cur_g, fut_g))

    cur_idx = np.concatenate([r[0] for r in rows])
    fut_idx = np.concatenate([r[1] for r in rows])
    if int(cfg.get("max_pairs", 0)) > 0 and len(cur_idx) > int(cfg["max_pairs"]):
        keep = rng.choice(len(cur_idx), int(cfg["max_pairs"]), replace=False)
        cur_idx = cur_idx[keep]
        fut_idx = fut_idx[keep]
    else:
        keep = rng.permutation(len(cur_idx))
        cur_idx = cur_idx[keep]
        fut_idx = fut_idx[keep]

    current_m = assign[cur_idx]
    future_m = assign[fut_idx]
    current_representation = str(cfg.get("current_representation", "prototype"))
    future_representation = str(cfg.get("future_representation", "prototype"))
    if current_representation == "prototype":
        current = proto[current_m]
    elif current_representation == "frame_feature":
        current = F[cur_idx]
    else:
        raise ValueError(f"unsupported current_representation={current_representation}")
    if future_representation == "prototype":
        future = proto[future_m]
    elif future_representation == "frame_feature":
        future = F[fut_idx]
    else:
        raise ValueError(f"unsupported future_representation={future_representation}")
    out_npz = out_dir / f"pairs_{suffix}.npz"
    np.savez_compressed(
        out_npz,
        current=current.astype(np.float32),
        future=future.astype(np.float32),
        current_milestone=current_m.astype(np.int64),
        future_milestone=future_m.astype(np.int64),
        t=FR[cur_idx].astype(np.int64),
        future_t=FR[fut_idx].astype(np.int64),
        episode_id=E[cur_idx].astype(np.int64),
        progress_t=pord[current_m].astype(np.float32),
        progress_future=pord[future_m].astype(np.float32),
        prototype_table=proto.astype(np.float32),
        pord=pord.astype(np.float32),
    )
    meta = {
        "name": cfg.get("name", out_dir.name),
        "feature_dir": str(feature_dir),
        "milestone_file": str(milestone_file),
        "output_npz": str(out_npz),
        "pair_mode": pair_mode,
        "horizon": horizon if pair_mode == "fixed_horizon" else None,
        "num_valid_frames": int(len(F)),
        "num_episodes": int(len(np.unique(E))),
        "num_pairs": int(len(cur_idx)),
        "feature_dim": int(proto.shape[1]),
        "num_milestones": int(num_milestones),
        "current_representation": current_representation,
        "future_representation": future_representation,
        "prototype_source": "DINOv3-H 1280D milestone centers",
        "split": "episode-level split expected during training",
    }
    (out_dir / f"meta_{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
