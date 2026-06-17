"""Compute FastWAM dataset_stats.json from a LeRobot dataset (no fastwam stack needed).

Usage:
  python scripts/compute_dataset_stats.py \
    --data_path <lerobot_train_dir> \
    --output_path <out.json> \
    --action_chunk 48

Format mirrors the output of RobotVideoDataset.get_dataset_stats():
  state.default: stepwise_*/global_*  shape [1, 14] / [14]
  action.default: stepwise_*/global_*  shape [48, 14] / [14]
  num_episodes, num_transition
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from tqdm import tqdm


def sliding_window_replication(arr: np.ndarray, window: int) -> np.ndarray:
    """arr: [T, dim] → [T, window, dim], replicating last frame at end."""
    T = arr.shape[0]
    t_idx = np.arange(T)[:, np.newaxis]          # [T, 1]
    k_idx = np.arange(window)[np.newaxis, :]     # [1, window]
    idx = np.minimum(t_idx + k_idx, T - 1)       # [T, window]
    return arr[idx]                               # [T, window, dim]


def process_episode(pq_path: pathlib.Path, action_chunk: int):
    """Returns state [T,1,14] and action [T,action_chunk,14]."""
    df = pd.read_parquet(pq_path, columns=["observation.state", "action"])
    states = np.stack(df["observation.state"].values).astype(np.float32)   # [T,14]
    actions = np.stack(df["action"].values).astype(np.float32)             # [T,14]
    state_win = states[:, np.newaxis, :]  # [T,1,14]
    action_win = sliding_window_replication(actions, action_chunk)         # [T,48,14]
    return state_win, action_win


def agg_stats(per_ep_mins, per_ep_maxs, per_ep_means, per_ep_vars,
              per_ep_q01, per_ep_q99):
    """Aggregate per-episode stats into global stats. Each element is [window, dim]."""
    mins  = np.stack(per_ep_mins,  axis=0)  # [N, W, dim]
    maxs  = np.stack(per_ep_maxs,  axis=0)
    means = np.stack(per_ep_means, axis=0)
    vars_ = np.stack(per_ep_vars,  axis=0)
    q01s  = np.stack(per_ep_q01,   axis=0)
    q99s  = np.stack(per_ep_q99,   axis=0)

    sw_min  = mins.min(axis=0)     # [W, dim]
    sw_max  = maxs.max(axis=0)
    sw_mean = means.mean(axis=0)
    sw_std  = np.sqrt((vars_ + (means - sw_mean[np.newaxis]) ** 2).mean(axis=0))
    sw_q01  = q01s.min(axis=0)
    sw_q99  = q99s.max(axis=0)

    gl_min  = sw_min.min(axis=0)   # [dim]
    gl_max  = sw_max.max(axis=0)
    gl_mean = sw_mean.mean(axis=0)
    gl_std  = np.sqrt((vars_ + (means - gl_mean[np.newaxis, np.newaxis]) ** 2).mean(axis=(0, 1)))
    gl_q01  = sw_q01.min(axis=0)
    gl_q99  = sw_q99.max(axis=0)

    return {
        "stepwise_min":  sw_min.tolist(),
        "stepwise_max":  sw_max.tolist(),
        "stepwise_mean": sw_mean.tolist(),
        "stepwise_std":  sw_std.tolist(),
        "stepwise_q01":  sw_q01.tolist(),
        "stepwise_q99":  sw_q99.tolist(),
        "global_min":    gl_min.tolist(),
        "global_max":    gl_max.tolist(),
        "global_mean":   gl_mean.tolist(),
        "global_std":    gl_std.tolist(),
        "global_q01":    gl_q01.tolist(),
        "global_q99":    gl_q99.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--action_chunk", type=int, default=48)
    args = ap.parse_args()

    data_path = pathlib.Path(args.data_path)
    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquets = sorted(data_path.glob("data/chunk-*/*.parquet"))
    print(f"Found {len(parquets)} parquet files")

    s_mins, s_maxs, s_means, s_vars, s_q01, s_q99 = [], [], [], [], [], []
    a_mins, a_maxs, a_means, a_vars, a_q01, a_q99 = [], [], [], [], [], []
    total_frames = 0

    for pq in tqdm(parquets, desc="Processing episodes"):
        s_win, a_win = process_episode(pq, args.action_chunk)  # [T,1,14], [T,48,14]
        total_frames += s_win.shape[0]

        s_mins.append(s_win.min(axis=0))
        s_maxs.append(s_win.max(axis=0))
        s_means.append(s_win.mean(axis=0))
        s_vars.append(s_win.var(axis=0))
        s_q01.append(np.quantile(s_win, 0.01, axis=0))
        s_q99.append(np.quantile(s_win, 0.99, axis=0))

        a_mins.append(a_win.min(axis=0))
        a_maxs.append(a_win.max(axis=0))
        a_means.append(a_win.mean(axis=0))
        a_vars.append(a_win.var(axis=0))
        a_q01.append(np.quantile(a_win, 0.01, axis=0))
        a_q99.append(np.quantile(a_win, 0.99, axis=0))

    num_eps = len(parquets)
    print(f"Episodes: {num_eps}, total frames: {total_frames}")

    stats = {
        "state":  {"default": agg_stats(s_mins, s_maxs, s_means, s_vars, s_q01, s_q99)},
        "action": {"default": agg_stats(a_mins, a_maxs, a_means, a_vars, a_q01, a_q99)},
        "num_episodes":   num_eps,
        "num_transition": total_frames,
    }
    output_path.write_text(json.dumps(stats, indent=2))
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
