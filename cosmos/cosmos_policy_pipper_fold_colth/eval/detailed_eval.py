#!/usr/bin/env python3
"""Detailed val-set eval for one checkpoint: full 100 val episodes, per-joint (L/R arm + grippers),
per-horizon MAE curve, per-episode distribution. Writes JSON + Markdown + PNG.

14-D layout (Piper dual-arm): [L_j1..L_j6, L_grip, R_j1..R_j6, R_grip] = idx 0-5,6,7-12,13.
"""
import argparse, dataclasses, glob, json, os
import cv2, h5py, numpy as np, torch
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats)

DIM_NAMES = [f"L_j{i}" for i in range(1, 7)] + ["L_grip"] + [f"R_j{i}" for i in range(1, 7)] + ["R_grip"]
ARM_IDX = [0,1,2,3,4,5, 7,8,9,10,11,12]; GRIP_IDX = [6, 13]


@dataclasses.dataclass
class EvalCfg:
    model_family: str = "cosmos"; suite: str = "aloha"
    config: str = "cosmos_predict2_2b_480p_pipper_fold_colth__inference_only"
    ckpt_path: str = ""; config_file: str = "cosmos_policy/config/config.py"
    use_third_person_image: bool = True; num_third_person_images: int = 1
    use_wrist_image: bool = True; num_wrist_images: int = 2
    use_proprio: bool = True; flip_images: bool = False
    use_variance_scale: bool = False; use_jpeg_compression: bool = False
    ar_future_prediction: bool = False; ar_value_prediction: bool = False; ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 10; num_denoising_steps_future_state: int = 1; num_denoising_steps_value: int = 1
    unnormalize_actions: bool = True; normalize_proprio: bool = True
    dataset_stats_path: str = ""; t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True; chunk_size: int = 50; num_open_loop_steps: int = 50
    deterministic: bool = True; seed: int = 195; randomize_seed: bool = False


def read_frame(path, idx):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read(); cap.release()
    if not ok: raise RuntimeError(f"frame {idx} {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--val_dir", required=True)
    ap.add_argument("--stats", required=True); ap.add_argument("--t5", required=True)
    ap.add_argument("--task", default="fold cloth"); ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--stride", type=int, default=50); ap.add_argument("--max_q", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = EvalCfg(ckpt_path=a.ckpt, dataset_stats_path=a.stats, t5_text_embeddings_path=a.t5)
    ds = load_dataset_stats(cfg.dataset_stats_path); init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, _ = get_model(cfg)
    eps = sorted(glob.glob(os.path.join(a.val_dir, "*.hdf5")))[: a.n_episodes]
    C = cfg.chunk_size
    err_hd = np.zeros((C, 14)); cnt = np.zeros((C, 14))   # sum |err| per (horizon, dim)
    ep_mae = []; nq = 0
    for ei, ep in enumerate(eps):
        with h5py.File(ep, "r") as h:
            gt = np.asarray(h["action"][:], np.float32); qpos = np.asarray(h["observations/qpos"][:], np.float32)
            vp = {c: h[f"observations/video_paths/{c}"][()].decode() for c in ["cam_high","cam_left_wrist","cam_right_wrist"]}
        T = gt.shape[0]; starts = list(range(0, max(1, T - C), a.stride))[: a.max_q]; errs = []
        for t in starts:
            obs = {"primary_image": read_frame(vp["cam_high"], t),
                   "left_wrist_image": read_frame(vp["cam_left_wrist"], t),
                   "right_wrist_image": read_frame(vp["cam_right_wrist"], t), "proprio": qpos[t]}
            r = get_action(cfg, model, ds, obs, a.task, num_denoising_steps_action=cfg.num_denoising_steps_action,
                           generate_future_state_and_value_in_parallel=False)
            pred = np.asarray(r["actions"], np.float32).reshape(-1, 14)[:C]
            g = gt[t:t+pred.shape[0]]; L = min(len(pred), len(g))
            e = np.abs(pred[:L] - g[:L])               # (L,14)
            err_hd[:L] += e; cnt[:L] += 1; errs.append(e.mean()); nq += 1
        if errs: ep_mae.append(float(np.mean(errs)))
        print(f"[{ei+1}/{len(eps)}] {os.path.basename(ep)} q={len(starts)} runMAE={np.mean(ep_mae):.4f}", flush=True)
    valid_h = int((cnt.sum(1) > 0).sum())              # number of horizons that actually had predictions
    per_dim = err_hd.sum(0) / np.maximum(cnt.sum(0), 1)   # (14,) over valid horizons only
    hmask = cnt.sum(1) > 0
    per_h = np.where(hmask, err_hd.sum(1) / np.maximum(cnt.sum(1), 1), np.nan)   # (C,)
    overall = float(err_hd.sum() / max(cnt.sum(), 1))
    ha = lambda h: float(per_h[h-1]) if h-1 < C and hmask[h-1] else None
    rep = {
        "ckpt": a.ckpt, "n_episodes": len(eps), "n_queries": nq, "chunk_size": C, "valid_horizon": valid_h,
        "mae_overall": overall,
        "mae_arm_joints": float(per_dim[ARM_IDX].mean()), "mae_grippers": float(per_dim[GRIP_IDX].mean()),
        "mae_left_arm": float(per_dim[0:6].mean()), "mae_right_arm": float(per_dim[7:13].mean()),
        "mae_at": {h: ha(h) for h in [1, 5, 10, valid_h]},
        "per_dim_mae": {DIM_NAMES[i]: float(per_dim[i]) for i in range(14)},
        "per_horizon_mae": [float(x) if not np.isnan(x) else None for x in per_h[:valid_h]],
        "per_episode_mae": {"min": float(np.min(ep_mae)), "median": float(np.median(ep_mae)),
                             "mean": float(np.mean(ep_mae)), "max": float(np.max(ep_mae))},
    }
    json.dump(rep, open(a.out, "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].plot(range(1, valid_h+1), per_h[:valid_h], marker="."); ax[0].set_title("MAE vs prediction horizon"); ax[0].set_xlabel("step"); ax[0].set_ylabel("MAE (rad)"); ax[0].grid(alpha=.3)
        colors = ["#3b7" if i in ARM_IDX else "#e74" for i in range(14)]
        ax[1].bar(DIM_NAMES, per_dim, color=colors); ax[1].set_title("per-joint MAE (green=arm, red=gripper)"); ax[1].tick_params(axis="x", rotation=60); ax[1].grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(a.out.replace(".json", ".png"), dpi=90); plt.close(fig)
    except Exception as e: print("plot skip", e)
    print("=== DETAILED REPORT ==="); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
