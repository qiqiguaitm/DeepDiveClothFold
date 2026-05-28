"""
Evaluate AWBC model with its expected "Advantage: positive" prompt on DAgger data.
Dumps DAgger data with advantage-conditioned prompt, then evaluates AWBC (8 GPUs).
"""

import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


AWBC_CKPT = "checkpoints/pi05_flatten_fold_awbc/awbc_v1/99999"
NORM_STATS = "data/Task_A/kai0_base"
HELDOUT_REPO = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_dagger"
ADVANTAGE_PROMPT = "Flatten and fold the cloth. Advantage: positive"

DUMP_PATH = "model_arithmetic/heldout_dumps/dagger_awbc_adv_prompt_v2.pkl"
N_BATCHES = 50
N_ACTION_BATCHES = 10
N_GPUS = 8


def _dump_with_advantage_prompt(gpu_id, n_batches):
    """Dump DAgger data with 'Advantage: positive' prompt."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    import dataclasses
    from tqdm import tqdm
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.training.sharding as sharding

    config = _config.get_config("pi05_flatten_fold_split_0")
    # Override default_prompt to inject advantage label
    new_data = dataclasses.replace(
        config.data,
        repo_id=HELDOUT_REPO,
        default_prompt=ADVANTAGE_PROMPT,
    )
    config = dataclasses.replace(config, data=new_data, batch_size=16)

    # Swap norm_stats to AWBC's (which is base)
    dagger_ns = os.path.join(HELDOUT_REPO, "norm_stats.json")
    backup = os.path.join(HELDOUT_REPO, "norm_stats_backup_awbc_adv.json")
    src_ns = os.path.join(NORM_STATS, "norm_stats.json")
    if not os.path.exists(backup) and os.path.exists(dagger_ns):
        shutil.copy(dagger_ns, backup)
    shutil.copy(src_ns, dagger_ns)

    try:
        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        batches = []
        it = iter(loader)
        for i in tqdm(range(n_batches), desc=f"[GPU {gpu_id}] dump awbc"):
            batches.append(next(it))
        with open(DUMP_PATH, "wb") as f:
            pickle.dump(batches, f)
        print(f"[GPU {gpu_id}] ✓ Saved {n_batches} batches with prompt='{ADVANTAGE_PROMPT}'", flush=True)
    finally:
        if os.path.exists(backup):
            shutil.copy(backup, dagger_ns)
            os.remove(backup)


def _eval_shard(gpu_id, loss_batch_indices, action_batch_indices):
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

    with open(DUMP_PATH, "rb") as f:
        all_batches = pickle.load(f)

    norm_stats = _normalize.load(NORM_STATS)
    policy = _policy_config.create_trained_policy(config, AWBC_CKPT, norm_stats=norm_stats)

    losses = []
    for i in tqdm(loss_batch_indices, desc=f"[GPU {gpu_id}] awbc+ loss"):
        batch = all_batches[i]
        loss = policy._model.compute_loss(jax.random.key(0), batch[0], batch[1])
        losses.append(float(jnp.mean(loss)))

    q01 = np.array(norm_stats["actions"].q01)
    q99 = np.array(norm_stats["actions"].q99)
    preds, gts = [], []
    for i in tqdm(action_batch_indices, desc=f"[GPU {gpu_id}] awbc+ actions"):
        obs, gt = all_batches[i]
        pred = policy._model.sample_actions(jax.random.key(42), obs, num_steps=10)
        preds.append(np.array(pred) * (q99 - q01) + q01)
        gts.append(np.array(gt) * (q99 - q01) + q01)

    return {
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


def main():
    t0 = time.time()

    # Step 1: dump (single GPU, I/O bound anyway)
    if not os.path.exists(DUMP_PATH):
        print("=== Step 1: Dumping DAgger with advantage prompt ===")
        with ProcessPoolExecutor(max_workers=1) as ex:
            f = ex.submit(_dump_with_advantage_prompt, 0, N_BATCHES)
            f.result()
    else:
        print(f"✓ Dump exists: {DUMP_PATH}")
    print(f"Dump phase: {(time.time()-t0)/60:.1f} min\n")

    # Step 2: parallel eval on 8 GPUs
    t1 = time.time()
    loss_splits = split_indices(N_BATCHES, N_GPUS)
    action_splits = split_indices(N_ACTION_BATCHES, N_GPUS)
    print(f"=== Step 2: Eval AWBC on {N_GPUS} GPUs (prompt='{ADVANTAGE_PROMPT}') ===")

    with ProcessPoolExecutor(max_workers=N_GPUS) as ex:
        futures = []
        for gpu_id in range(N_GPUS):
            f = ex.submit(_eval_shard, gpu_id, loss_splits[gpu_id], action_splits[gpu_id])
            futures.append((gpu_id, f))
        shards = [None] * N_GPUS
        for gpu_id, f in futures:
            shards[gpu_id] = f.result()

    all_losses, all_preds, all_gts = [], [], []
    for s in shards:
        all_losses.extend(s["losses"])
        all_preds.append(s["raw_preds"])
        all_gts.append(s["raw_gts"])

    awbc_adv = {
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
    results["awbc_correct_prompt"] = awbc_adv
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✓ awbc_correct_prompt: loss={awbc_adv['loss_mean']:.6f} (eval {(time.time()-t1)/60:.1f} min)")

    # Print full comparison
    print("\n" + "=" * 72)
    print("HELD-OUT EVAL - ALL MODELS (DAgger data)")
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
