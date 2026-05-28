"""
Test if official_mixed uses SA (Stage Advantage) by comparing outputs
with different prompts. If SA is baked in, advantage prompts should
produce meaningfully different actions.
"""
import os, pickle, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def test_prompt_response(gpu_id, prompt_text):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    import jax, dataclasses
    from tqdm import tqdm
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.training.sharding as sharding
    import openpi.shared.normalize as _normalize
    import openpi.policies.policy_config as _policy_config

    # Dump 5 batches on base data with the given prompt
    config = _config.get_config("pi05_flatten_fold_split_0")
    new_data = dataclasses.replace(
        config.data,
        repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base",
        default_prompt=prompt_text,
    )
    config = dataclasses.replace(config, data=new_data, batch_size=4)

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False)
    batches = []
    it = iter(loader)
    for i in range(5):
        batches.append(next(it))

    # Load official_mixed
    norm_stats = _normalize.load("checkpoints/Task_A/mixed_1")
    policy = _policy_config.create_trained_policy(
        config, "checkpoints/Task_A/mixed_1", norm_stats=norm_stats
    )

    # Sample actions
    preds = []
    for batch in batches:
        obs, _ = batch
        pred = policy._model.sample_actions(jax.random.key(42), obs, num_steps=10)
        preds.append(np.array(pred))

    q01 = np.array(norm_stats["actions"].q01)
    q99 = np.array(norm_stats["actions"].q99)
    raw = np.concatenate(preds, axis=0) * (q99 - q01) + q01
    return raw, prompt_text


def main():
    t0 = time.time()
    prompts = [
        ("default", "Flatten and fold the cloth."),
        ("task_name", "flat the cloth"),
        ("adv_positive", "Flatten and fold the cloth. Advantage: positive"),
        ("adv_negative", "Flatten and fold the cloth. Advantage: negative"),
    ]

    print(f"Running {len(prompts)} prompt variants in parallel on GPUs 0-{len(prompts)-1}")
    with ProcessPoolExecutor(max_workers=len(prompts)) as ex:
        futures = {ex.submit(test_prompt_response, i, p[1]): p[0] for i, p in enumerate(prompts)}
        results = {}
        for f in futures:
            name = futures[f]
            results[name] = f.result()[0]
            print(f"  ✓ {name} done")

    # Compare pairwise distances between prompt variants
    names = list(results.keys())
    print("\n" + "=" * 70)
    print("OFFICIAL_MIXED: prompt-to-prompt action difference")
    print("=" * 70)
    print(f"(If SA baked in, adv_positive vs adv_negative should have LARGE diff)")
    print(f"(If SA NOT used, all prompts should give similar outputs)\n")
    print(f"  {'Pair':<45} {'MSE':>12} {'L1':>12}")
    print(f"  {'-'*69}")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = results[names[i]], results[names[j]]
            a_f = a.reshape(a.shape[0], -1).astype(np.float64)
            b_f = b.reshape(b.shape[0], -1).astype(np.float64)
            mse = float(np.mean((a_f - b_f)**2))
            l1 = float(np.mean(np.abs(a_f - b_f)))
            print(f"  {names[i]+' vs '+names[j]:<45} {mse:>12.6f} {l1:>12.6f}")

    print(f"\n  Reference: typical within-model per-frame MSE ≈ 0.07")
    print(f"  Reference: if prompts don't matter → MSE should be < 0.001")
    print(f"  Reference: if SA is baked in → MSE between adv+/- should be > 0.01")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
