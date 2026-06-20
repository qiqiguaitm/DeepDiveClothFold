#!/usr/bin/env python3
"""
Compute cosmos-policy dataset_statistics.json from converted ALOHA-format train HDF5s.

cosmos-policy normalizes actions/proprio to [-1,1] via min/max with NO clipping
(dataset_utils.rescale_data). Raw min/max is fragile to teleop/DAgger outliers, so —
following the kai0 pipeline — we write q01/q99 into the *_min/*_max fields for robustness
(the ~1% tails just land slightly outside [-1,1], which is harmless and better-behaved than
true min/max). Mean/std/median are stored truthfully (loaded but unused by normalization).

Output: <data_dir>/dataset_statistics.json  (flat {stat_name: list}, schema per
calculate_dataset_statistics()).
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np


def collect(split_dir):
    acts, props = [], []
    for f in sorted(glob.glob(os.path.join(split_dir, "*.hdf5"))):
        with h5py.File(f, "r") as h:
            acts.append(np.asarray(h["action"][:], np.float64))
            props.append(np.asarray(h["observations/qpos"][:], np.float64))
    return np.concatenate(acts, 0), np.concatenate(props, 0)


def stats_for(arr, name):
    return {
        f"{name}_min": np.quantile(arr, 0.01, axis=0).tolist(),   # robust lower (q01)
        f"{name}_max": np.quantile(arr, 0.99, axis=0).tolist(),   # robust upper (q99)
        f"{name}_mean": arr.mean(0).tolist(),
        f"{name}_std": arr.std(0).tolist(),
        f"{name}_median": np.median(arr, 0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="task dir containing train/ (and where json is written one level up)")
    ap.add_argument("--out_dir", required=True, help="ALOHADataset data_dir (where dataset_statistics.json is read)")
    args = ap.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    actions, proprio = collect(train_dir)
    print(f"actions {actions.shape}  proprio {proprio.shape}")

    stats = {}
    stats.update(stats_for(actions, "actions"))
    stats.update(stats_for(proprio, "proprio"))

    # sanity: report fraction of action values outside the robust [q01,q99] range
    amin = np.array(stats["actions_min"]); amax = np.array(stats["actions_max"])
    frac_out = float(((actions < amin) | (actions > amax)).mean())
    print(f"fraction of action entries outside [q01,q99]: {frac_out:.4f} (expect ~0.02)")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "dataset_statistics.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"wrote {out}")
    print("actions_min(q01):", np.round(amin, 3))
    print("actions_max(q99):", np.round(amax, 3))


if __name__ == "__main__":
    main()
