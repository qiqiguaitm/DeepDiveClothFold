"""Compute state_01/state_99 normalization percentiles (14-dim observation.state)
for an Agilex LeRobot dataset and write dataset_meta_info/<name>/stat.json."""
import os
import json
import glob
import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_path", required=True, help="LeRobot dataset dir (has data/, meta/)")
    p.add_argument("--name", required=True, help="dataset_meta_info subdir name to write stat.json")
    p.add_argument("--meta_out", default=None, help="output root for stat.json (default: this script's dir)")
    p.add_argument("--frame_stride", type=int, default=5, help="subsample frames per episode for speed")
    p.add_argument("--max_episodes", type=int, default=None)
    args = p.parse_args()

    pqs = sorted(glob.glob(f"{args.src_path}/data/chunk-*/*.parquet"))
    if args.max_episodes:
        pqs = pqs[: args.max_episodes]
    print(f"{args.name}: {len(pqs)} parquet files")

    chunks = []
    for i, f in enumerate(pqs):
        df = pd.read_parquet(f, columns=["observation.state"])
        s = np.asarray(df["observation.state"].tolist(), dtype=np.float32)[:: args.frame_stride]
        chunks.append(s)
        if i % 500 == 0:
            print(f"  read {i}/{len(pqs)}")
    alls = np.concatenate(chunks, axis=0)
    print("collected states:", alls.shape)

    state_01 = np.percentile(alls, 1, axis=0)
    state_99 = np.percentile(alls, 99, axis=0)
    stat = {"state_01": state_01.tolist(), "state_99": state_99.tolist()}

    meta_root = args.meta_out or os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(meta_root, args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/stat.json", "w") as f:
        json.dump(stat, f)
    print("wrote", f"{out_dir}/stat.json")
    print("state_01:", np.round(state_01, 3))
    print("state_99:", np.round(state_99, 3))


if __name__ == "__main__":
    main()
