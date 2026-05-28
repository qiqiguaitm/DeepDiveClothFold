"""
Evaluate AWBC model only on held-out DAgger data (8 GPUs).
Merges result into existing heldout_results.pkl.
"""

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


AWBC = {
    "ckpt": "checkpoints/pi05_flatten_fold_awbc/awbc_v1/99999",
    "norm_stats": "data/Task_A/kai0_base",  # same as normal
}

# Reuse normal's dump (same norm_stats)
DATA_PATH = "model_arithmetic/heldout_dumps/dagger_normal_99999.pkl"
N_BATCHES = 50
N_ACTION_BATCHES = 10
N_GPUS = 8


def _eval_shard(ckpt_path, norm_stats_path, data_path, gpu_id,
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
    for i in tqdm(loss_batch_indices, desc=f"[GPU {gpu_id}] awbc loss"):
        batch = all_batches[i]
        loss = policy._model.compute_loss(jax.random.key(0), batch[0], batch[1])
        losses.append(float(jnp.mean(loss)))

    q01 = np.array(norm_stats["actions"].q01)
    q99 = np.array(norm_stats["actions"].q99)
    preds, gts = [], []
    for i in tqdm(action_batch_indices, desc=f"[GPU {gpu_id}] awbc actions"):
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
    result = []
    s = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        result.append(list(range(s, s + size)))
        s += size
    return result


def main():
    t0 = time.time()
    loss_splits = split_indices(N_BATCHES, N_GPUS)
    action_splits = split_indices(N_ACTION_BATCHES, N_GPUS)

    print(f"Evaluating AWBC on {N_GPUS} GPUs")
    print(f"Reusing dump: {DATA_PATH}")
    print(f"Batch distribution: {[len(s) for s in loss_splits]} loss, {[len(s) for s in action_splits]} action")

    with ProcessPoolExecutor(max_workers=N_GPUS) as ex:
        futures = []
        for gpu_id in range(N_GPUS):
            f = ex.submit(
                _eval_shard, AWBC["ckpt"], AWBC["norm_stats"], DATA_PATH,
                gpu_id, loss_splits[gpu_id], action_splits[gpu_id]
            )
            futures.append((gpu_id, f))

        shards = [None] * N_GPUS
        for gpu_id, f in futures:
            try:
                shards[gpu_id] = f.result()
            except Exception as e:
                print(f"GPU {gpu_id} failed: {e}")
                import traceback; traceback.print_exc()
                return

    # Merge
    all_losses, all_preds, all_gts = [], [], []
    for s in shards:
        all_losses.extend(s["losses"])
        all_preds.append(s["raw_preds"])
        all_gts.append(s["raw_gts"])

    awbc_result = {
        "loss_mean": float(np.mean(all_losses)),
        "loss_std": float(np.std(all_losses)),
        "losses": all_losses,
        "raw_preds": np.concatenate(all_preds, axis=0),
        "raw_gts": np.concatenate(all_gts, axis=0),
    }

    # Merge into existing results
    results_path = "model_arithmetic/heldout_results.pkl"
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    results["awbc_99999"] = awbc_result
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✓ awbc_99999: loss={awbc_result['loss_mean']:.6f}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")

    # Print full comparison
    print("\n" + "=" * 72)
    print("HELD-OUT EVAL (DAgger data) - ALL 4 MODELS")
    print("=" * 72)
    print(f"  {'Model':<20} {'Loss':>12} {'Std':>12}")
    print(f"  {'-'*44}")
    for n, r in results.items():
        print(f"  {n:<20} {r['loss_mean']:>12.6f} {r['loss_std']:>12.6f}")

    print("\n  Action Distance to Ground Truth (RAW joint space):")
    print(f"  {'Model':<20} {'MSE':>12} {'L1':>12} {'Cosine Dist':>12}")
    print(f"  {'-'*56}")
    for n, r in results.items():
        p = r['raw_preds'].reshape(r['raw_preds'].shape[0], -1).astype(np.float64)
        g = r['raw_gts'].reshape(r['raw_gts'].shape[0], -1).astype(np.float64)
        mse = float(np.mean((p - g) ** 2))
        l1 = float(np.mean(np.abs(p - g)))
        cos = [1 - np.dot(p[s], g[s])/(np.linalg.norm(p[s])*np.linalg.norm(g[s]))
               for s in range(p.shape[0]) if np.linalg.norm(p[s])>0 and np.linalg.norm(g[s])>0]
        print(f"  {n:<20} {mse:>12.6f} {l1:>12.6f} {float(np.mean(cos)):>12.6f}")


if __name__ == "__main__":
    main()
