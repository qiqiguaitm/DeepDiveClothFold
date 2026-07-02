#!/usr/bin/env python
"""Export CRAVE milestone pairs for LMWM smoke/prototype training.

This first exporter intentionally supports the existing ep2302 `_cache.npz`
artifact. It produces LaWM-shaped pairs `(r_t, r_future)` while keeping a clear
metadata record that the initial prototype vector is a smoke-test placeholder:
one-hot milestone id plus scalar progress.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _prototype_table(pord: np.ndarray) -> np.ndarray:
    """Smoke representation: [one_hot(milestone), progress]."""
    m = int(len(pord))
    eye = np.eye(m, dtype=np.float32)
    return np.concatenate([eye, pord.astype(np.float32)[:, None]], axis=1)


def _next_unique_indices(milestones: np.ndarray) -> np.ndarray:
    n = len(milestones)
    fut = np.full(n, -1, dtype=np.int64)
    next_change = -1
    for i in range(n - 2, -1, -1):
        if milestones[i + 1] != milestones[i]:
            next_change = i + 1
        fut[i] = next_change
    return fut


def export_pairs(cfg: dict) -> dict:
    source_cache = Path(cfg["source_cache"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(source_cache)
    milestones = z["nm30"].astype(np.int64)
    pord = z["Pord"].astype(np.float32)
    progress = z["cv"].astype(np.float32)
    episode_id = int(z["ep"]) if "ep" in z.files else -1
    horizon = int(cfg.get("horizon", 30))
    pair_mode = str(cfg.get("pair_mode", "fixed_horizon"))

    proto = _prototype_table(pord)
    n = len(milestones)

    if pair_mode == "fixed_horizon":
        t = np.arange(0, max(0, n - horizon), dtype=np.int64)
        future_t = t + horizon
        suffix = f"fixed_h{horizon}"
    elif pair_mode == "next_unique":
        fut = _next_unique_indices(milestones)
        t = np.where(fut >= 0)[0].astype(np.int64)
        future_t = fut[t].astype(np.int64)
        suffix = "next_unique"
    else:
        raise ValueError(f"unsupported pair_mode={pair_mode!r}")

    cur_m = milestones[t]
    fut_m = milestones[future_t]
    current = proto[cur_m]
    future = proto[fut_m]

    out_npz = out_dir / f"pairs_{suffix}.npz"
    np.savez_compressed(
        out_npz,
        current=current.astype(np.float32),
        future=future.astype(np.float32),
        current_milestone=cur_m.astype(np.int64),
        future_milestone=fut_m.astype(np.int64),
        t=t.astype(np.int64),
        future_t=future_t.astype(np.int64),
        progress_t=progress[t].astype(np.float32),
        progress_future=progress[future_t].astype(np.float32),
        prototype_table=proto.astype(np.float32),
        pord=pord.astype(np.float32),
        episode_id=np.full(len(t), episode_id, dtype=np.int64),
    )

    meta = {
        "name": cfg.get("name", out_dir.name),
        "source_cache": str(source_cache),
        "output_npz": str(out_npz),
        "episode_id": episode_id,
        "pair_mode": pair_mode,
        "horizon": horizon if pair_mode == "fixed_horizon" else None,
        "num_frames": int(n),
        "num_pairs": int(len(t)),
        "num_milestones": int(len(pord)),
        "prototype_source": cfg.get("prototype_source", "one_hot_progress_smoke"),
        "warning": (
            "Smoke-test prototype vectors are one-hot milestone ids plus progress. "
            "Replace with CRAVE prototype latents for the first full experiment."
        ),
    }
    (out_dir / f"meta_{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = _load_config(args.config)
    meta = export_pairs(cfg)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
