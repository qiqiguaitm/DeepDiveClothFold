"""Standalone norm stats computation using pandas/numpy only (no giga_datasets).

Usage:
  python scripts/compute_norm_stats_simple.py \
    --data_path <lerobot_dataset_dir> \
    --output_path <out.json>

Reads all parquet files under <data_path>/data/chunk-*/episode_*.parquet,
computes per-dim mean/std/q01/q99 for observation.state and action,
writes the same JSON format as compute_norm_stats.py.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from tqdm import tqdm


def collect_parquets(data_path: pathlib.Path) -> list[pathlib.Path]:
    return sorted(data_path.glob("data/chunk-*/*.parquet"))


def compute_stats(arrays: np.ndarray) -> dict:
    """arrays: (N, D) float64"""
    mean = arrays.mean(axis=0)
    std = arrays.std(axis=0)
    q01 = np.quantile(arrays, 0.01, axis=0)
    q99 = np.quantile(arrays, 0.99, axis=0)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="LeRobot dataset root dir")
    ap.add_argument("--output_path", required=True, help="Output JSON path")
    args = ap.parse_args()

    data_path = pathlib.Path(args.data_path)
    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquets = collect_parquets(data_path)
    print(f"Found {len(parquets)} parquet files in {data_path}")

    all_state, all_action = [], []
    for pq in tqdm(parquets, desc="Reading parquets"):
        df = pd.read_parquet(pq, columns=["observation.state", "action"])
        states = np.stack(df["observation.state"].values).astype(np.float64)
        actions = np.stack(df["action"].values).astype(np.float64)
        all_state.append(states)
        all_action.append(actions)

    all_state = np.concatenate(all_state, axis=0)
    all_action = np.concatenate(all_action, axis=0)
    print(f"Total frames: {len(all_state)}, state dim: {all_state.shape[1]}, action dim: {all_action.shape[1]}")

    norm_stats = {
        "observation.state": compute_stats(all_state),
        "action": compute_stats(all_action),
    }
    dim = all_action.shape[1]
    output = {
        "norm_stats": norm_stats,
        "delta_mask": [False] * dim,
        "action_repr": "abs",
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
