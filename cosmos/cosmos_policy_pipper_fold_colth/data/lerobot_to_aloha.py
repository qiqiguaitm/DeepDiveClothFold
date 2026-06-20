#!/usr/bin/env python3
"""
Convert a LeRobot v2.1 wam_fold_v3 split -> cosmos-policy ALOHA per-episode HDF5 format.

Why this exists: cosmos-policy's ALOHADataset expects, under data_dir/<task>/{train,val}/:
    episode_<i>.hdf5  with
        /action                 (T,14) float32   ABSOLUTE joint targets
        /observations/qpos      (T,14) float32   proprio state
        /observations/video_paths/{cam_high,cam_left_wrist,cam_right_wrist}  (str)
        attrs: sim(int)=0, success(bool)=True, task_description(str)
We store ABSOLUTE video paths in video_paths (os.path.join keeps absolute paths), so NO
video transcode is needed — ALOHADataset.load_video_as_images() resizes to final_image_size
(224) at load time. Camera map:  top_head->cam_high, hand_left->cam_left_wrist,
hand_right->cam_right_wrist.

wam_fold_v3 quirk (per data owner): each parquet's INTERNAL data is correct (row order,
per-episode timestamp zeroed, row count) but the global `episode_index`/`index` columns are
WRONG. So we IGNORE those two columns entirely and key episode identity off the parquet
FILE NUMBER; train/val come from the physical split dirs.

Usage:
  python lerobot_to_aloha.py --src <visrobot01_v3_train> --out <task_dir> --split train \
         --task fold_cloth --n 144 --seed 0
"""
import argparse
import json
import os
import random
import glob

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

CAM_MAP = {  # cosmos-policy cam name -> LeRobot video_key
    "cam_high": "observation.images.top_head",
    "cam_left_wrist": "observation.images.hand_left",
    "cam_right_wrist": "observation.images.hand_right",
}


def n_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def episode_num_from_parquet(p):
    # episode_000123.parquet -> 123
    return int(os.path.basename(p).split("_")[-1].split(".")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LeRobot split root (has data/, videos/, meta/)")
    ap.add_argument("--out", required=True, help="task dir; episodes written to <out>/<split>/")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--task", default="fold_cloth")
    ap.add_argument("--n", type=int, default=-1, help="max episodes (-1 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)

    parquets = sorted(glob.glob(os.path.join(args.src, "data", "chunk-*", "episode_*.parquet")),
                      key=episode_num_from_parquet)
    assert parquets, f"no parquet under {args.src}/data/chunk-*/"
    print(f"[{args.split}] found {len(parquets)} parquet episodes in {args.src}")

    if args.n > 0 and args.n < len(parquets):
        rng = random.Random(args.seed)
        parquets = sorted(rng.sample(parquets, args.n), key=episode_num_from_parquet)
        print(f"[{args.split}] sampled {len(parquets)} episodes (seed={args.seed})")

    task_desc = args.task.replace("_", " ")
    written, skipped = 0, 0
    manifest = []
    for new_idx, pq_path in enumerate(parquets):
        src_n = episode_num_from_parquet(pq_path)
        chunk = src_n // 1000  # robust: info.json chunk count is unreliable
        # locate the 3 camera mp4s by FILE NUMBER (not the buggy column)
        vpaths = {}
        ok = True
        for cam, vkey in CAM_MAP.items():
            vp = os.path.join(args.src, "videos", f"chunk-{chunk:03d}", vkey, f"episode_{src_n:06d}.mp4")
            if not os.path.exists(vp):
                print(f"  ! missing video {vp}; skip ep {src_n}")
                ok = False
                break
            vpaths[cam] = os.path.abspath(vp)
        if not ok:
            skipped += 1
            continue

        tbl = pq.read_table(pq_path, columns=["action", "observation.state"])
        action = np.asarray(tbl.column("action").to_pylist(), dtype=np.float32)   # (T,14)
        qpos = np.asarray(tbl.column("observation.state").to_pylist(), dtype=np.float32)  # (T,14)
        rows = action.shape[0]
        assert action.shape[1] == 14 and qpos.shape[1] == 14, f"bad dims {action.shape} {qpos.shape}"

        nf_high = n_frames(vpaths["cam_high"])
        if nf_high <= 0:
            print(f"  ! unreadable video for ep {src_n}; skip")
            skipped += 1
            continue
        # Align lengths: num_steps in the dataset is video-derived; keep action/qpos == nf_high.
        T = nf_high
        if rows != T:
            if abs(rows - T) > 3:
                print(f"  ~ ep {src_n}: rows={rows} vs frames={T} (diff {rows - T}); aligning to {T}")
            if rows >= T:
                action, qpos = action[:T], qpos[:T]
            else:  # pad by repeating last row
                action = np.concatenate([action, np.tile(action[-1:], (T - rows, 1))], 0)
                qpos = np.concatenate([qpos, np.tile(qpos[-1:], (T - rows, 1))], 0)

        h5_path = os.path.join(out_dir, f"episode_{new_idx:06d}.hdf5")
        with h5py.File(h5_path, "w") as f:
            f.attrs["sim"] = 0
            f.attrs["success"] = True
            f.attrs["success_score"] = 1.0
            f.attrs["task_description"] = task_desc
            f.attrs["src_episode_num"] = src_n
            f.create_dataset("action", data=action.astype(np.float32))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=qpos.astype(np.float32))
            vg = obs.create_group("video_paths")
            for cam, vp in vpaths.items():
                vg.create_dataset(cam, data=np.bytes_(vp))
        manifest.append({"new_idx": new_idx, "src_episode_num": src_n, "T": int(T),
                         "parquet": pq_path, "videos": vpaths})
        written += 1
        if written % 25 == 0:
            print(f"  [{args.split}] wrote {written}/{len(parquets)} (last ep {src_n}, T={T})")

    with open(os.path.join(args.out, f"manifest_{args.split}.json"), "w") as f:
        json.dump({"task": args.task, "split": args.split, "src": args.src,
                   "written": written, "skipped": skipped, "episodes": manifest}, f, indent=1)
    print(f"[{args.split}] DONE  written={written}  skipped={skipped}  -> {out_dir}")


if __name__ == "__main__":
    main()
