"""
Evaluate kai0_mixed_1 candidates on held-out DAgger data.
Compares:
  - kai0_mixed_1_inverse_loss: inverse_loss weighted merge (our reproduction)
  - kai0_mixed_1_greedy:       greedy = split_2 only (collapsed)
  - kai0_mixed_1_grad:         gradient_descent optimized (if exists)
  - official_mixed:            Task_A/mixed_1 (official kai0 release)
  - normal_99999:              full-data baseline (pi05_flatten_fold_normal)
  - best_single_split:         split_2 alone (lowest individual loss)

Uses 8 GPUs: each model's 50 batches split across all 8 GPUs for fast parallel eval.
Reuses existing DAgger dumps in model_arithmetic/heldout_dumps/.

Usage: uv run python model_arithmetic/eval_kai0_mixed_1.py
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


MODELS = {
    "normal_99999": {
        "ckpt": "checkpoints/pi05_flatten_fold_normal/normal_v1/99999",
        "norm_stats": "data/Task_A/kai0_base",
    },
    "official_mixed": {
        "ckpt": "checkpoints/Task_A/mixed_1",
        "norm_stats": "checkpoints/Task_A/mixed_1",
    },
    "kai0_mixed_1_inverse_loss": {
        "ckpt": "checkpoints/kai0_mixed_1_inverse_loss/0",
        "norm_stats": "checkpoints/kai0_mixed_1_inverse_loss",
    },
    "kai0_mixed_1_greedy": {
        "ckpt": "checkpoints/kai0_mixed_1_greedy_only_split2/0",
        "norm_stats": "checkpoints/kai0_mixed_1_greedy_only_split2",
    },
    "kai0_mixed_1_grad": {
        "ckpt": "checkpoints/kai0_mixed_1_grad/0",
        "norm_stats": "checkpoints/kai0_mixed_1_grad",
    },
    "best_single_split_2": {
        "ckpt": "checkpoints/kai0_mixed_1_split_2/split_2_v1/24999",
        "norm_stats": "data/Task_A/kai0_mixed_1_data",
    },
}

HELDOUT_DATA = "model_arithmetic/heldout_dumps/dagger_normal_99999.pkl"
N_BATCHES = 50
N_ACTION_BATCHES = 10
N_GPUS = 8


def _eval_shard(model_name, ckpt_path, norm_stats_path, data_path, gpu_id,
                loss_batch_indices, action_batch_indices):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    import jax.numpy as jnp
    from tqdm import tqdm
    import openpi.training.config as _config
    import openpi.shared.normalize as _normalize
    import openpi.policies.policy_config as _policy_config

    config = _config.get_config("pi05_flatten_fold_split_0")

    with open(data_path, "rb") as f:
        all_batches = pickle.load(f)

    norm_stats = _normalize.load(norm_stats_path)
    policy = _policy_config.create_trained_policy(config, ckpt_path, norm_stats=norm_stats)

    losses = []
    for i in tqdm(loss_batch_indices, desc=f"[GPU {gpu_id}] {model_name} loss"):
        batch = all_batches[i]
        loss = policy._model.compute_loss(jax.random.key(0), batch[0], batch[1])
        losses.append(float(jnp.mean(loss)))

    q01 = np.array(norm_stats["actions"].q01)
    q99 = np.array(norm_stats["actions"].q99)
    preds, gts = [], []
    for i in tqdm(action_batch_indices, desc=f"[GPU {gpu_id}] {model_name} actions"):
        obs, gt = all_batches[i]
        pred = policy._model.sample_actions(jax.random.key(42), obs, num_steps=10)
        preds.append(np.array(pred) * (q99 - q01) + q01)
        gts.append(np.array(gt) * (q99 - q01) + q01)

    return {
        "gpu_id": gpu_id,
        "losses": losses,
        "raw_preds": np.concatenate(preds, axis=0) if preds else np.zeros((0, 50, 32)),
        "raw_gts": np.concatenate(gts, axis=0) if gts else np.zeros((0, 50, 32)),
    }


def split_indices(total, k):
    base, extra = divmod(total, k)
    result, s = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        result.append(list(range(s, s + size)))
        s += size
    return result


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
    return mse, l1, float(np.mean(cos_dists)) if cos_dists else 0.0


def evaluate_one_model(model_name, info, data_path, n_gpus, n_batches, n_action_batches):
    loss_splits = split_indices(n_batches, n_gpus)
    action_splits = split_indices(n_action_batches, n_gpus)

    print(f"\n  Launching {n_gpus} workers for {model_name}")
    with ProcessPoolExecutor(max_workers=n_gpus) as ex:
        futures = []
        for gpu_id in range(n_gpus):
            f = ex.submit(
                _eval_shard, model_name, info["ckpt"], info["norm_stats"], data_path,
                gpu_id, loss_splits[gpu_id], action_splits[gpu_id]
            )
            futures.append((gpu_id, f))

        shards = [None] * n_gpus
        for gpu_id, f in futures:
            try:
                shards[gpu_id] = f.result()
            except Exception as e:
                print(f"  GPU {gpu_id} failed: {e}")
                import traceback; traceback.print_exc()
                return None

    all_losses, all_preds, all_gts = [], [], []
    for s in shards:
        all_losses.extend(s["losses"])
        all_preds.append(s["raw_preds"])
        all_gts.append(s["raw_gts"])

    return {
        "loss_mean": float(np.mean(all_losses)),
        "loss_std": float(np.std(all_losses)),
        "losses": all_losses,
        "raw_preds": np.concatenate(all_preds, axis=0),
        "raw_gts": np.concatenate(all_gts, axis=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-grad", action="store_true", help="Skip gradient_descent variant")
    parser.add_argument("--only", nargs="+", help="Only evaluate specific models")
    args = parser.parse_args()

    t0 = time.time()

    models_to_eval = {n: i for n, i in MODELS.items()}
    if args.skip_grad:
        models_to_eval.pop("kai0_mixed_1_grad", None)
    if args.only:
        models_to_eval = {n: i for n, i in models_to_eval.items() if n in args.only}

    # Skip models whose checkpoint doesn't exist
    for n in list(models_to_eval.keys()):
        ckpt_path = models_to_eval[n]["ckpt"]
        if not os.path.exists(ckpt_path):
            print(f"SKIP {n}: checkpoint not found at {ckpt_path}")
            models_to_eval.pop(n)

    print(f"\n=== Evaluating {len(models_to_eval)} models on held-out DAgger data ===")
    print(f"Data: {HELDOUT_DATA}")
    print(f"Models: {list(models_to_eval.keys())}")

    model_results = {}
    for name, info in models_to_eval.items():
        print(f"\n--- {name} ---")
        ts = time.time()
        result = evaluate_one_model(
            name, info, HELDOUT_DATA,
            N_GPUS, N_BATCHES, N_ACTION_BATCHES
        )
        if result is not None:
            model_results[name] = result
            print(f"  ✓ {name}: loss={result['loss_mean']:.6f} (took {(time.time()-ts)/60:.1f} min)")

    names = list(model_results.keys())

    print("\n" + "=" * 80)
    print("KAI0_MIXED_1 CANDIDATES EVAL (DAgger held-out)")
    print("=" * 80)
    print(f"  {'Model':<30} {'Loss':>12} {'Std':>12}")
    print(f"  {'-'*54}")
    for n in names:
        r = model_results[n]
        print(f"  {n:<30} {r['loss_mean']:>12.6f} {r['loss_std']:>12.6f}")

    print("\n  Action Distance to Ground Truth (RAW joint space):")
    print(f"  {'Model':<30} {'MSE':>12} {'L1':>12} {'Cosine Dist':>12}")
    print(f"  {'-'*66}")
    for n in names:
        r = model_results[n]
        mse, l1, cos = compute_distances(r["raw_preds"], r["raw_gts"])
        print(f"  {n:<30} {mse:>12.6f} {l1:>12.6f} {cos:>12.6f}")

    print(f"\n{'=' * 80}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
    print(f"{'=' * 80}")

    with open("model_arithmetic/kai0_mixed_1_eval_results.pkl", "wb") as f:
        pickle.dump(model_results, f)
    print("Saved: model_arithmetic/kai0_mixed_1_eval_results.pkl")


if __name__ == "__main__":
    main()
