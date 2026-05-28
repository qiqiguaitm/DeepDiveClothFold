"""
Compare merged model vs official model vs normal training.
Fair evaluation: each model uses its own norm_stats for normalization.
Runs 3 models in parallel on separate GPUs.

Usage:
  cd kai0/
  uv run python model_arithmetic/evaluate_models.py \
    --data-path model_arithmetic/flatten_fold_val_small.pkl \
    --config pi05_flatten_fold_split_0
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


CHECKPOINTS = {
    "official_mixed": {
        "path": "checkpoints/Task_A/mixed_1",
        "is_mixed": True,
        "gpu": 0,
    },
    "our_mixed": {
        "path": "checkpoints/pi05_flatten_fold_mixed/mixed_inverse_loss",
        "is_mixed": True,
        "gpu": 2,
    },
    "normal_99999": {
        "path": "checkpoints/pi05_flatten_fold_normal/normal_v1",
        "is_mixed": False,
        "step": "99999",
        "gpu": 4,
    },
}

# The dump data was normalized with this norm_stats
DUMP_NORM_STATS_PATH = "data/Task_A/kai0_base"


def _quantile_unnormalize(x, q01, q99):
    """Reverse quantile normalization: raw = x * (q99 - q01) + q01"""
    return x * (q99 - q01) + q01


def _quantile_normalize(x, q01, q99):
    """Quantile normalization: norm = (raw - q01) / (q99 - q01)"""
    scale = q99 - q01
    scale = np.where(scale == 0, 1.0, scale)
    return (x - q01) / scale


def _eval_single_model(name, info, config_name, data_path, dump_norm_path, num_action_batches):
    """Evaluate one model on its assigned GPU with correct norm_stats."""
    gpu_id = info["gpu"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    import jax.numpy as jnp
    from tqdm import tqdm
    import openpi.training.config as _config
    import openpi.shared.normalize as _normalize
    import openpi.policies.policy_config as _policy_config
    import flax.nnx as nnx
    import flax.traverse_util as traverse_util

    config = _config.get_config(config_name)

    with open(data_path, "rb") as f:
        data_samples_list = pickle.load(f)

    # Load dump norm_stats (used to un-normalize the dump data)
    dump_ns = _normalize.load(dump_norm_path)

    # Load this model's own norm_stats
    model_ns = _normalize.load(info["path"])

    # Extract quantile arrays for state and actions
    dump_state_q01 = np.array(dump_ns["state"].q01)
    dump_state_q99 = np.array(dump_ns["state"].q99)
    dump_act_q01 = np.array(dump_ns["actions"].q01)
    dump_act_q99 = np.array(dump_ns["actions"].q99)

    model_state_q01 = np.array(model_ns["state"].q01)
    model_state_q99 = np.array(model_ns["state"].q99)
    model_act_q01 = np.array(model_ns["actions"].q01)
    model_act_q99 = np.array(model_ns["actions"].q99)

    # Check if norm_stats are the same (skip re-normalization if so)
    same_ns = np.allclose(dump_state_q01, model_state_q01, atol=1e-6) and \
              np.allclose(dump_act_q01, model_act_q01, atol=1e-6)
    print(f"[GPU {gpu_id}] {name}: norm_stats {'SAME' if same_ns else 'DIFFERENT'} as dump", flush=True)

    # Load checkpoint
    norm_stats_for_policy = model_ns
    if info.get("is_mixed"):
        ckpt_dir = os.path.join(info["path"], "0")
        if not os.path.exists(ckpt_dir):
            ckpt_dir = info["path"]
    else:
        ckpt_dir = os.path.join(info["path"], info["step"])

    print(f"[GPU {gpu_id}] Loading {name} from {ckpt_dir}", flush=True)
    policy = _policy_config.create_trained_policy(config, ckpt_dir, norm_stats=norm_stats_for_policy)

    def renormalize_obs(obs):
        """Un-normalize obs with dump stats, re-normalize with model stats."""
        if same_ns:
            return obs
        import dataclasses
        # State: un-normalize then re-normalize
        raw_state = _quantile_unnormalize(np.array(obs.state), dump_state_q01, dump_state_q99)
        new_state = _quantile_normalize(raw_state, model_state_q01, model_state_q99)
        return dataclasses.replace(obs, state=jnp.array(new_state, dtype=obs.state.dtype))

    def renormalize_actions(actions):
        """Un-normalize actions with dump stats, re-normalize with model stats."""
        if same_ns:
            return actions
        raw = _quantile_unnormalize(np.array(actions), dump_act_q01, dump_act_q99)
        return jnp.array(_quantile_normalize(raw, model_act_q01, model_act_q99), dtype=actions.dtype)

    def unnormalize_actions_to_raw(actions_norm, use_model_ns=True):
        """Convert normalized actions back to raw space."""
        if use_model_ns:
            return _quantile_unnormalize(np.array(actions_norm), model_act_q01, model_act_q99)
        else:
            return _quantile_unnormalize(np.array(actions_norm), dump_act_q01, dump_act_q99)

    # ── Eval 1: Loss (with correctly normalized data) ──
    losses = []
    for data_samples in tqdm(data_samples_list, desc=f"[GPU {gpu_id}] {name} loss"):
        obs, actions = data_samples
        obs_renorm = renormalize_obs(obs)
        actions_renorm = renormalize_actions(actions)
        loss = policy._model.compute_loss(jax.random.key(0), obs_renorm, actions_renorm)
        losses.append(float(jnp.mean(loss)))
    avg_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))
    print(f"[GPU {gpu_id}] {name}: loss={avg_loss:.6f} ± {std_loss:.6f}", flush=True)

    # ── Eval 3: Action predictions (compare in RAW space) ──
    num_batches = min(num_action_batches, len(data_samples_list))
    raw_pred_list = []
    raw_gt_list = []
    for i in tqdm(range(num_batches), desc=f"[GPU {gpu_id}] {name} actions"):
        obs, actions = data_samples_list[i]
        obs_renorm = renormalize_obs(obs)
        pred_norm = policy._model.sample_actions(jax.random.key(42), obs_renorm, num_steps=10)
        # Convert to raw space
        raw_pred = unnormalize_actions_to_raw(pred_norm, use_model_ns=True)
        raw_gt = unnormalize_actions_to_raw(actions, use_model_ns=False)  # GT uses dump norm_stats
        raw_pred_list.append(raw_pred)
        raw_gt_list.append(raw_gt)

    raw_preds = np.concatenate(raw_pred_list, axis=0)
    raw_gts = np.concatenate(raw_gt_list, axis=0)
    print(f"[GPU {gpu_id}] {name}: actions shape={raw_preds.shape}", flush=True)

    # ── Eval 2: Flat params ──
    params = nnx.state(policy._model)
    flat_params = {}
    for k, v in traverse_util.flatten_dict(params.to_pure_dict()).items():
        flat_params["/".join(k)] = np.array(v).astype(np.float32)

    return {
        "name": name,
        "loss_mean": avg_loss,
        "loss_std": std_loss,
        "actions_raw": raw_preds,
        "gt_actions_raw": raw_gts,
        "flat_params": flat_params,
    }


def compute_distances(a1, a2):
    f1 = a1.reshape(a1.shape[0], -1).astype(np.float64)
    f2 = a2.reshape(a2.shape[0], -1).astype(np.float64)
    mse = float(np.mean((f1 - f2) ** 2))
    l1 = float(np.mean(np.abs(f1 - f2)))
    cos_dists = []
    for s in range(f1.shape[0]):
        n1, n2 = np.linalg.norm(f1[s]), np.linalg.norm(f2[s])
        if n1 > 0 and n2 > 0:
            cos_dists.append(1.0 - float(np.dot(f1[s], f2[s]) / (n1 * n2)))
    return {"mse": mse, "l1": l1, "cosine_dist": float(np.mean(cos_dists)) if cos_dists else 0.0}


def compute_weight_distance(p1, p2):
    common_keys = set(p1.keys()) & set(p2.keys())
    l2_dists, cos_sims = [], []
    for k in common_keys:
        v1, v2 = p1[k].flatten(), p2[k].flatten()
        l2_dists.append(float(np.sqrt(np.sum((v1 - v2) ** 2))))
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_sims.append(float(np.dot(v1, v2) / (n1 * n2)))
    return {"l2": float(np.sqrt(np.sum(np.array(l2_dists) ** 2))), "cosine": float(np.mean(cos_sims))}


def main():
    parser = argparse.ArgumentParser(description="Compare models (fair norm_stats, 3-GPU parallel)")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--config", default="pi05_flatten_fold_split_0")
    parser.add_argument("--action-batches", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Launching 3 models on GPUs 0/2/4 in parallel...")
    print(f"Each model uses its OWN norm_stats. Actions compared in RAW space.\n")

    futures = {}
    with ProcessPoolExecutor(max_workers=3) as executor:
        for name, info in CHECKPOINTS.items():
            f = executor.submit(
                _eval_single_model, name, info, args.config,
                args.data_path, DUMP_NORM_STATS_PATH, args.action_batches
            )
            futures[f] = name

        model_results = {}
        for f in as_completed(futures):
            name = futures[f]
            try:
                model_results[name] = f.result()
                print(f"\n✓ {name} completed")
            except Exception as e:
                print(f"\n✗ {name} failed: {e}")
                import traceback
                traceback.print_exc()

    names = list(model_results.keys())

    print("\n" + "=" * 70)
    print("EVALUATION 1: Validation Loss (each model with its own norm_stats)")
    print("=" * 70)
    print(f"  {'Model':<20} {'Loss':>10} {'Std':>10}")
    print(f"  {'-'*40}")
    for n in names:
        r = model_results[n]
        print(f"  {n:<20} {r['loss_mean']:>10.6f} {r['loss_std']:>10.6f}")

    print("\n" + "=" * 70)
    print("EVALUATION 2: Weight-Space Distance")
    print("=" * 70)
    print(f"  {'Pair':<40} {'L2 Dist':>12} {'Cosine Sim':>12}")
    print(f"  {'-'*64}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            wd = compute_weight_distance(model_results[n1]["flat_params"], model_results[n2]["flat_params"])
            print(f"  {n1+' vs '+n2:<40} {wd['l2']:>12.4f} {wd['cosine']:>12.6f}")

    print("\n" + "=" * 70)
    print("EVALUATION 3: Action Output Distance (RAW action space)")
    print("=" * 70)
    print(f"  {'Pair':<40} {'MSE':>12} {'L1':>12} {'Cosine Dist':>12}")
    print(f"  {'-'*76}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            d = compute_distances(model_results[n1]["actions_raw"], model_results[n2]["actions_raw"])
            print(f"  {n1+' vs '+n2:<40} {d['mse']:>12.6f} {d['l1']:>12.6f} {d['cosine_dist']:>12.6f}")

    print(f"\n  Action distance to Ground Truth (RAW space):")
    print(f"  {'Model':<40} {'MSE':>12} {'L1':>12} {'Cosine Dist':>12}")
    print(f"  {'-'*76}")
    gt = model_results[names[0]]["gt_actions_raw"]
    for n in names:
        d = compute_distances(model_results[n]["actions_raw"], gt)
        print(f"  {n:<40} {d['mse']:>12.6f} {d['l1']:>12.6f} {d['cosine_dist']:>12.6f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total evaluation time: {elapsed / 60:.1f} min")
    print(f"{'=' * 70}")

    out_path = "model_arithmetic/eval_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model_results, f)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
