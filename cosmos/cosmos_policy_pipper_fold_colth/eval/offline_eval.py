#!/usr/bin/env python3
"""
Offline action-MAE eval for cosmos_policy_pipper_fold_colth on the converted val HDF5 set.

Loads the trained policy (get_model) and, for sampled timesteps of each val episode, builds
the ALOHA observation dict {primary_image, left_wrist_image, right_wrist_image, proprio},
queries get_action (10 denoising steps), and compares the predicted 50-step action chunk to
the ground-truth chunk. Reports overall MAE and MAE@{1,10,25,50} in raw joint units.

This mirrors deploy.py's model-loading path but runs fully offline (no robot/server).

Usage (inside the cosmos-policy uv env):
  python offline_eval.py --ckpt <path/to/iter_XXXX> --val_dir <PIPPER_DATA>/fold_cloth/val \
     --stats <PIPPER_DATA>/dataset_statistics.json --t5 <PIPPER_DATA>/t5_embeddings.pkl \
     --n_episodes 20 --stride 50 --out <report.json>
"""
import argparse
import dataclasses
import glob
import json
import os

import cv2
import h5py
import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)


@dataclasses.dataclass
class EvalCfg:
    # mirrors deploy.py PolicyEvalConfig fields read by get_model / get_action
    model_family: str = "cosmos"
    suite: str = "aloha"
    config: str = "cosmos_predict2_2b_480p_pipper_fold_colth__inference_only"
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 2
    use_proprio: bool = True
    flip_images: bool = False
    use_variance_scale: bool = False
    use_jpeg_compression: bool = False
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 10
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 50
    num_open_loop_steps: int = 50
    deterministic: bool = True
    seed: int = 195
    randomize_seed: bool = False


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {idx} of {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--t5", required=True)
    ap.add_argument("--task", default="fold cloth")
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--stride", type=int, default=50, help="sample a query every <stride> frames")
    ap.add_argument("--max_queries_per_ep", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = EvalCfg(ckpt_path=args.ckpt, dataset_stats_path=args.stats, t5_text_embeddings_path=args.t5)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, _ = get_model(cfg)

    eps = sorted(glob.glob(os.path.join(args.val_dir, "*.hdf5")))[: args.n_episodes]
    horizons = [1, 12, 25]   # instant / ~0.5s / ~1s @25Hz (model emits ~25-step chunk)
    agg = {h: [] for h in horizons}
    agg["all"] = []
    n_queries = 0

    for ei, ep in enumerate(eps):
        with h5py.File(ep, "r") as h:
            gt_action = np.asarray(h["action"][:], np.float32)  # (T,14) raw absolute
            qpos = np.asarray(h["observations/qpos"][:], np.float32)
            vp = {c: h[f"observations/video_paths/{c}"][()].decode()
                  for c in ["cam_high", "cam_left_wrist", "cam_right_wrist"]}
        T = gt_action.shape[0]
        starts = list(range(0, max(1, T - cfg.chunk_size), args.stride))[: args.max_queries_per_ep]
        for t in starts:
            obs = {
                "primary_image": read_frame(vp["cam_high"], t),
                "left_wrist_image": read_frame(vp["cam_left_wrist"], t),
                "right_wrist_image": read_frame(vp["cam_right_wrist"], t),
                "proprio": qpos[t],
            }
            ret = get_action(cfg, model, dataset_stats, obs, args.task,
                             num_denoising_steps_action=cfg.num_denoising_steps_action,
                             generate_future_state_and_value_in_parallel=False)
            pred = np.asarray(ret["actions"], np.float32).reshape(-1, 14)[: cfg.chunk_size]
            gt = gt_action[t : t + pred.shape[0]]
            L = min(len(pred), len(gt))
            err = np.abs(pred[:L] - gt[:L])  # (L,14)
            agg["all"].append(err.mean())
            for hh in horizons:
                if L >= 1:
                    agg[hh].append(err[: min(hh, L)].mean())
            n_queries += 1
        print(f"[{ei+1}/{len(eps)}] {os.path.basename(ep)} T={T} queries={len(starts)} "
              f"running MAE={np.mean(agg['all']):.4f}", flush=True)

    report = {
        "ckpt": args.ckpt,
        "n_episodes": len(eps),
        "n_queries": n_queries,
        "chunk_size": cfg.chunk_size,
        "num_denoising_steps_action": cfg.num_denoising_steps_action,
        "mae_overall": float(np.mean(agg["all"])) if agg["all"] else None,
        **{f"mae@{h}": (float(np.mean(agg[h])) if agg[h] else None) for h in horizons},
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("=== REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
