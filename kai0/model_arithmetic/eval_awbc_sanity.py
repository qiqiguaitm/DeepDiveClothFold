"""
Sanity check: evaluate AWBC on its OWN training data (advantage dataset).
Should give very low loss if model works correctly.
"""
import os, pickle, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

AWBC_CKPT = "checkpoints/pi05_flatten_fold_awbc/awbc_v1/99999"
NORM_STATS = "data/Task_A/kai0_base"
ADVANTAGE_REPO = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_advantage"
DUMP_PATH = "model_arithmetic/heldout_dumps/advantage_awbc_sanity.pkl"
N_BATCHES = 20
N_GPUS = 8


def _dump(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    import jax, dataclasses
    from tqdm import tqdm
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.training.sharding as sharding
    import openpi.training.config as _cfg_mod

    # Use AWBC config directly to get prompt_from_task=True
    config = _config.get_config("pi05_flatten_fold_awbc")
    config = dataclasses.replace(config, batch_size=16)
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    batches = []
    it = iter(loader)
    for i in tqdm(range(N_BATCHES), desc=f"[GPU {gpu_id}] dump awbc sanity"):
        batches.append(next(it))
    with open(DUMP_PATH, "wb") as f:
        pickle.dump(batches, f)
    print(f"[GPU {gpu_id}] ✓ Saved {N_BATCHES} advantage batches", flush=True)


def _eval(gpu_id, start, end):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    import jax, jax.numpy as jnp
    from tqdm import tqdm
    import openpi.training.config as _config
    import openpi.shared.normalize as _normalize
    import openpi.policies.policy_config as _policy_config

    config = _config.get_config("pi05_flatten_fold_awbc")
    with open(DUMP_PATH, "rb") as f:
        batches = pickle.load(f)
    norm_stats = _normalize.load(NORM_STATS)
    policy = _policy_config.create_trained_policy(config, AWBC_CKPT, norm_stats=norm_stats)

    losses = []
    for i in tqdm(range(start, end), desc=f"[GPU {gpu_id}] sanity"):
        loss = policy._model.compute_loss(jax.random.key(0), batches[i][0], batches[i][1])
        losses.append(float(jnp.mean(loss)))

    # Compute action MSE
    q01 = np.array(norm_stats["actions"].q01)
    q99 = np.array(norm_stats["actions"].q99)
    preds, gts = [], []
    for i in range(start, min(end, start + 2)):  # only 2 action batches per shard
        obs, gt = batches[i]
        pred = policy._model.sample_actions(jax.random.key(42), obs, num_steps=10)
        preds.append(np.array(pred) * (q99 - q01) + q01)
        gts.append(np.array(gt) * (q99 - q01) + q01)

    return losses, np.concatenate(preds, axis=0) if preds else np.zeros((0,50,32)), np.concatenate(gts, axis=0) if gts else np.zeros((0,50,32))


def main():
    t0 = time.time()
    if not os.path.exists(DUMP_PATH):
        print("Dumping advantage data...")
        with ProcessPoolExecutor(max_workers=1) as ex:
            ex.submit(_dump, 0).result()
    print(f"Dump: {(time.time()-t0)/60:.1f} min")

    # Split across 8 GPUs
    base, extra = divmod(N_BATCHES, N_GPUS)
    ranges = []
    s = 0
    for i in range(N_GPUS):
        size = base + (1 if i < extra else 0)
        ranges.append((s, s+size))
        s += size

    with ProcessPoolExecutor(max_workers=N_GPUS) as ex:
        futures = [ex.submit(_eval, i, r[0], r[1]) for i, r in enumerate(ranges)]
        all_losses, all_preds, all_gts = [], [], []
        for f in futures:
            losses, preds, gts = f.result()
            all_losses.extend(losses)
            all_preds.append(preds)
            all_gts.append(gts)

    loss_mean = np.mean(all_losses)
    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)
    p = preds.reshape(preds.shape[0], -1).astype(np.float64)
    g = gts.reshape(gts.shape[0], -1).astype(np.float64)
    mse = np.mean((p - g) ** 2)
    l1 = np.mean(np.abs(p - g))

    print(f"\n{'='*60}")
    print(f"AWBC SANITY CHECK on advantage dataset (its own training data):")
    print(f"{'='*60}")
    print(f"  Loss:  {loss_mean:.6f}")
    print(f"  MSE:   {mse:.6f}")
    print(f"  L1:    {l1:.6f}")
    print(f"  Action preds shape: {preds.shape}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
