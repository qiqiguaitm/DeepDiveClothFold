#!/usr/bin/env python
"""Strict apples-to-apples: pi05 (14D joint) → FK → 20D EE6D MAE, on the SAME val
windows X3.C XVLA used.

Why: pi05 smooth_800 reports joint-space MAE 0.0089@1; X3.C XVLA reports EE6D-space
MAE 0.0146@1. Not comparable (different spaces). This script puts pi05 in EE6D space:

  1. Replicate X3.C window selection EXACTLY: read the EE6D dataset meta, take the
     held-out region (episode_index >= len(ds.episodes)-50), deterministic strided
     subset of n_windows. Each window = (episode_index, f_idx).
  2. For each window, find the matching base (14D joint) episode via (orig_source,
     orig_ep) and run pi05 inference at frame f_idx -> 14D joint chunk (horizon 50,
     take first 30).
  3. FK every predicted joint frame -> 20D EE6D (SAME joint_to_ee6d_row as the X3.C
     training data, so GT and pred live in identical EE6D space).
  4. GT = base joint chunk [f_idx : f_idx+30] -> FK -> 20D EE6D.
  5. MAE @{1,10,25,30} averaged over action dims, identical metric to eval_xvla_ee6d.py.

Run on uc03 (pi05 JAX ckpt local). Outputs JSON for offline comparison.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

# --- paths (uc03) ---
# Images: read from the EE6D dataset (its video symlinks point at ubuntu/.../vis_base/,
#   reachable on uc03). Joint GT + state: read from base parquet (ubuntu_old, readable).
# Both indexed by the SAME EE6D episode_index (orig_source/orig_ep identical), so frames align.
EE6D_ROOT = "/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/self_built/A_new_smooth_800"
EE6D_META = EE6D_ROOT + "/meta/episodes.jsonl"
BASE_ROOT = "/data/shared/ubuntu_old/data/Task_A/A_new_smooth_800/base"
CKPT = "/data/shared/ubuntu_old/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_smooth_800_new_norm/task_a_new_smooth_800_new_norm/49999"
NORM_STATS_DIR = "/data/shared/ubuntu_old/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_smooth_800_new_norm/task_a_new_smooth_800_new_norm"
FK_CONVERTER = "/data/shared/ubuntu/workspace/deepdive_kai0/train_scripts/xvla/data"

PROMPT = "Flatten and fold the cloth."
N_HELDOUT_EP = 50
ACTION_CHUNK = 30      # X3.C chunk_size (compare on common horizon)
PI05_HORIZON = 50
HORIZONS = [1, 10, 25, 30]
CAMS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")


def select_windows_like_x3c(ee6d_eps, n_windows):
    """Replicate LeRobotEE6DDataset.samples + select_windows EXACTLY.
    samples = [(episode_index, f_idx) for ep in episodes for f_idx in 0..length-ACTION_CHUNK].
    held-out region = samples whose episode_index >= (len(episodes) - N_HELDOUT_EP).
    NOTE: X3.C uses len(ds.episodes) (=806) - 50 = 756 as the episode_index cutoff."""
    total_ep = len(ee6d_eps)
    cutoff = total_ep - N_HELDOUT_EP  # 756, compared against episode_index field
    samples = []
    for ep in ee6d_eps:
        epi = ep["episode_index"]
        length = ep["length"]
        for f in range(max(0, length - ACTION_CHUNK + 1)):
            samples.append((epi, f, ep["orig_source"], ep["orig_ep"]))
    region = [s for s in samples if s[0] >= cutoff]
    stride = max(1, len(region) // n_windows)
    chosen = region[::stride][:n_windows]
    meta = {"total_ee6d_ep": total_ep, "cutoff_index": cutoff,
            "region_n": len(region), "stride": stride, "n_selected": len(chosen)}
    return chosen, meta


def load_base_index(base_root):
    """Map (orig_source, orig_ep) -> base parquet path, using base meta."""
    eps = [json.loads(l) for l in (Path(base_root) / "meta" / "episodes.jsonl").read_text().splitlines()]
    by_orig = {}
    for ep in eps:
        by_orig[(ep["orig_source"], ep["orig_ep"])] = ep["episode_index"]
    return by_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=1000)
    ap.add_argument("--out", default="/tmp/pi05_fk_ee6d_eval.json")
    ap.add_argument("--flow-samples", type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, FK_CONVERTER)
    from joint_to_ee6d import joint_to_ee6d_row  # same FK as X3.C training data
    import pyarrow.parquet as pq
    import av
    from openpi.training import config as _train_config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import normalize as _normalize  # noqa
    import dataclasses
    from openpi.models import pi0_config
    from openpi.training import config as cfgmod

    t0 = time.time()
    # --- build pi05 config inline (smooth800 not registered in this tree) ---
    # Use A_0423_0527 as a structural template, override repo_id/assets to smooth800.
    base_cfg = _train_config.get_config("pi05_flatten_fold_A_0423_0527")
    # repo_id drives norm_stats asset_id; point at the base dataset + norm_stats dir.
    data = dataclasses.replace(base_cfg.data, repo_id=BASE_ROOT)
    cfg = dataclasses.replace(base_cfg, data=data,
                              assets_base_dir=NORM_STATS_DIR if hasattr(base_cfg, "assets_base_dir") else base_cfg.assets_dirs)
    print(f"[load] create_trained_policy({CKPT})", flush=True)
    policy = _policy_config.create_trained_policy(cfg, CKPT)
    print(f"[load] policy ready in {time.time()-t0:.1f}s", flush=True)

    ee6d_eps = [json.loads(l) for l in Path(EE6D_META).read_text().splitlines()]
    chosen, sel_meta = select_windows_like_x3c(ee6d_eps, args.n_windows)
    print(f"[data] selected {len(chosen)} windows: {json.dumps(sel_meta)}", flush=True)

    base_by_orig = load_base_index(BASE_ROOT)

    # cache per-episode base data (parquet joint + video frames decoded lazily)
    eff_h = ACTION_CHUNK
    per_step_abs = np.zeros(eff_h, dtype=np.float64)
    n_seen = 0
    _vid_cache = {}

    def get_base_ep(orig_source, orig_ep):
        base_idx = base_by_orig.get((orig_source, orig_ep))
        if base_idx is None:
            return None
        pqpath = Path(BASE_ROOT) / "data" / "chunk-000" / f"episode_{base_idx:06d}.parquet"
        df = pq.read_table(pqpath).to_pandas()
        actions = np.stack([np.array(a, dtype=np.float32) for a in df["action"]])
        states = np.stack([np.array(s, dtype=np.float32) for s in df["observation.state"]])
        return base_idx, actions, states

    def decode_frame(ee_idx, cam, f_idx):
        # images come from the EE6D dataset (symlinks reachable on uc03), indexed by EE6D episode_index
        key = (ee_idx, cam)
        if key not in _vid_cache:
            vp = Path(EE6D_ROOT) / "videos" / "chunk-000" / cam / f"episode_{ee_idx:06d}.mp4"
            c = av.open(str(vp)); c.streams.video[0].thread_type = "AUTO"
            frames = [fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)]
            c.close()
            _vid_cache[key] = frames
        frames = _vid_cache[key]
        return frames[min(f_idx, len(frames) - 1)]

    last_ep = None
    for wi, (epi, f_idx, orig_source, orig_ep) in enumerate(chosen):
        be = get_base_ep(orig_source, orig_ep)
        if be is None:
            continue
        base_idx, actions, states = be
        if epi != last_ep:
            _vid_cache.clear()  # free previous ep frames
            last_ep = epi
        # observation at f_idx, in AgilexInputs repack format (images dict + state).
        # images from EE6D (by ee idx epi), state (14D joint) from base parquet.
        obs = {
            "state": states[f_idx],
            "images": {
                "top_head": decode_frame(epi, "observation.images.top_head", f_idx),
                "hand_left": decode_frame(epi, "observation.images.hand_left", f_idx),
                "hand_right": decode_frame(epi, "observation.images.hand_right", f_idx),
            },
            "prompt": PROMPT,
        }
        # pi05 inference -> (horizon, 14) joint
        if args.flow_samples > 1:
            preds = np.stack([policy.infer(obs)["actions"] for _ in range(args.flow_samples)])
            pred_joint = np.median(preds, axis=0)
        else:
            pred_joint = policy.infer(obs)["actions"]  # (50,14)
        pred_joint = np.asarray(pred_joint)[:eff_h]   # (30,14)

        # GT joint chunk from base
        gt_joint = actions[f_idx:f_idx + eff_h]
        h = min(len(pred_joint), len(gt_joint))
        if h == 0:
            continue
        # FK both to EE6D 20D
        pred_ee6d = np.stack([joint_to_ee6d_row(pred_joint[t]) for t in range(h)])
        gt_ee6d = np.stack([joint_to_ee6d_row(gt_joint[t]) for t in range(h)])
        ae = np.abs(pred_ee6d - gt_ee6d).mean(axis=1)  # (h,) mean over 20 dims
        per_step_abs[:h] += ae
        n_seen += 1
        if wi % 50 == 0:
            print(f"[pred] {wi+1}/{len(chosen)} (eval {n_seen}) elapsed={time.time()-t0:.0f}s", flush=True)

    per_step_mae = per_step_abs / max(1, n_seen)
    results = {f"MAE@{hh}": float(per_step_mae[:min(hh, eff_h)].mean()) for hh in HORIZONS}
    out = {"ckpt": CKPT, "n_windows_eval": n_seen, "horizons": HORIZONS,
           "window_selection": sel_meta, "per_step_mae": per_step_mae.tolist(),
           "results": results, "flow_samples": args.flow_samples, "elapsed_s": time.time() - t0}
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n===== pi05 FK->EE6D EVAL =====", flush=True)
    for hh in HORIZONS:
        print(f"MAE@{hh}: {results[f'MAE@{hh}']:.4f}", flush=True)
    print(f"windows={n_seen}  json->{args.out}", flush=True)


if __name__ == "__main__":
    main()
