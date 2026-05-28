"""
Held-out eval on DAgger data. Processes 3 models sequentially.
For EACH model, all 8 GPUs are used via data parallelism (batches split across GPUs).

Flow:
  Step 1: Dump 3 DAgger variants in parallel (3 GPUs)
  Step 2: For each model (sequentially):
    - Spawn 8 subprocess, each loads full model on one GPU
    - Each GPU processes ~6-7 batches of the 50 total
    - Merge results

Usage:
  cd kai0/
  uv run python model_arithmetic/evaluate_heldout.py
"""

import argparse
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


MODELS = {
    "normal_99999": {
        "ckpt": "checkpoints/pi05_flatten_fold_normal/normal_v1/99999",
        "norm_stats": "data/Task_A/kai0_base",
    },
    "our_mixed": {
        "ckpt": "checkpoints/pi05_flatten_fold_mixed/mixed_inverse_loss/0",
        "norm_stats": "checkpoints/pi05_flatten_fold_mixed/mixed_inverse_loss",
    },
    "official_mixed": {
        "ckpt": "checkpoints/Task_A/mixed_1",
        "norm_stats": "checkpoints/Task_A/mixed_1",
    },
    "awbc_99999": {
        "ckpt": "checkpoints/pi05_flatten_fold_awbc/awbc_v1/99999",
        "norm_stats": "data/Task_A/kai0_base",  # same as normal
    },
}

HELDOUT_REPO = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_dagger"
N_BATCHES = 50
N_ACTION_BATCHES = 10
N_GPUS = 8


def _dump_worker(model_name, norm_stats_src, repo_id, out_path, gpu_id, n_batches):
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
    new_data = dataclasses.replace(config.data, repo_id=repo_id)
    config = dataclasses.replace(config, data=new_data, batch_size=16)

    dagger_ns = os.path.join(repo_id, "norm_stats.json")
    backup = os.path.join(repo_id, f"norm_stats_backup_{model_name}.json")
    src_ns = os.path.join(norm_stats_src, "norm_stats.json")

    if not os.path.exists(backup) and os.path.exists(dagger_ns):
        shutil.copy(dagger_ns, backup)
    shutil.copy(src_ns, dagger_ns)

    try:
        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        batches = []
        it = iter(loader)
        for i in tqdm(range(n_batches), desc=f"[GPU {gpu_id}] dump {model_name}"):
            batches.append(next(it))
        with open(out_path, "wb") as f:
            pickle.dump(batches, f)
        print(f"[GPU {gpu_id}] ✓ {model_name}: {n_batches} batches → {out_path}", flush=True)
    finally:
        if os.path.exists(backup):
            shutil.copy(backup, dagger_ns)
            os.remove(backup)
    return out_path


def _eval_shard(model_name, ckpt_path, norm_stats_path, data_path, gpu_id,
                loss_batch_indices, action_batch_indices):
    """Evaluate a subset of batches on one GPU."""
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

    # Loss
    losses = []
    for i in tqdm(loss_batch_indices, desc=f"[GPU {gpu_id}] {model_name} loss"):
        batch = all_batches[i]
        loss = policy._model.compute_loss(jax.random.key(0), batch[0], batch[1])
        losses.append(float(jnp.mean(loss)))

    # Actions
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
    """Split [0, total) into k roughly equal index lists (contiguous)."""
    base, extra = divmod(total, k)
    result = []
    s = 0
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
    """Run one model across all n_gpus by splitting batches."""
    loss_splits = split_indices(n_batches, n_gpus)
    action_splits = split_indices(n_action_batches, n_gpus)

    print(f"\n  Launching {n_gpus} workers for {model_name}")
    print(f"  Batch distribution: {[len(s) for s in loss_splits]} loss, {[len(s) for s in action_splits]} action")

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

    # Merge in GPU order (which matches batch order because splits are contiguous)
    all_losses = []
    all_preds = []
    all_gts = []
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
    parser.add_argument("--n-batches", type=int, default=N_BATCHES)
    parser.add_argument("--action-batches", type=int, default=N_ACTION_BATCHES)
    parser.add_argument("--n-gpus", type=int, default=N_GPUS)
    parser.add_argument("--skip-dump", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    out_dir = "model_arithmetic/heldout_dumps"
    os.makedirs(out_dir, exist_ok=True)

    # AWBC shares norm_stats with normal → reuse its dump
    dump_paths = {}
    for name in MODELS:
        if name == "awbc_99999":
            dump_paths[name] = f"{out_dir}/dagger_normal_99999.pkl"
        else:
            dump_paths[name] = f"{out_dir}/dagger_{name}.pkl"

    # ── Step 1: Dump (3 GPUs parallel) ──
    # Only dump models that don't already have a dump file
    models_to_dump = {n: info for n, info in MODELS.items()
                      if not os.path.exists(dump_paths[n])}
    if not models_to_dump:
        print("✓ All dumps exist, skipping dump phase")
    else:
        print(f"\n=== Step 1: Dumping DAgger data for {list(models_to_dump.keys())} ===")
        dump_gpu_map = {n: i for i, n in enumerate(models_to_dump.keys())}
        with ProcessPoolExecutor(max_workers=len(models_to_dump)) as ex:
            futures = {}
            for name, info in models_to_dump.items():
                f = ex.submit(
                    _dump_worker, name, info["norm_stats"],
                    HELDOUT_REPO, dump_paths[name], dump_gpu_map[name], args.n_batches
                )
                futures[f] = name
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"dump {futures[f]} failed: {e}")
                    import traceback; traceback.print_exc()
                    return

    dump_time = time.time() - t0
    print(f"\n✓ Dump phase: {dump_time/60:.1f} min")

    # ── Step 2: Sequential per model, each uses all 8 GPUs ──
    print(f"\n=== Step 2: Evaluating 3 models sequentially, {args.n_gpus} GPUs each ===")
    t1 = time.time()

    model_results = {}
    for name, info in MODELS.items():
        print(f"\n--- Evaluating {name} ---")
        ts = time.time()
        result = evaluate_one_model(
            name, info, dump_paths[name],
            args.n_gpus, args.n_batches, args.action_batches
        )
        if result is not None:
            model_results[name] = result
            print(f"  ✓ {name}: loss={result['loss_mean']:.6f} (took {(time.time()-ts)/60:.1f} min)")

    eval_time = time.time() - t1

    # ── Print ──
    names = list(model_results.keys())
    print("\n" + "=" * 70)
    print("HELD-OUT EVAL (DAgger data, each model with own norm_stats, 8 GPUs)")
    print("=" * 70)
    print(f"  {'Model':<20} {'Loss':>12} {'Std':>12}")
    print(f"  {'-'*44}")
    for n in names:
        r = model_results[n]
        print(f"  {n:<20} {r['loss_mean']:>12.6f} {r['loss_std']:>12.6f}")

    print("\n  Action Distance to Ground Truth (RAW joint space):")
    print(f"  {'Model':<20} {'MSE':>12} {'L1':>12} {'Cosine Dist':>12}")
    print(f"  {'-'*56}")
    for n in names:
        r = model_results[n]
        mse, l1, cos = compute_distances(r["raw_preds"], r["raw_gts"])
        print(f"  {n:<20} {mse:>12.6f} {l1:>12.6f} {cos:>12.6f}")

    print(f"\n{'=' * 70}")
    print(f"Dump: {dump_time/60:.1f} min | Eval: {eval_time/60:.1f} min | Total: {(time.time()-t0)/60:.1f} min")
    print(f"{'=' * 70}")

    with open("model_arithmetic/heldout_results.pkl", "wb") as f:
        pickle.dump(model_results, f)
    print("Saved: model_arithmetic/heldout_results.pkl")


if __name__ == "__main__":
    main()
