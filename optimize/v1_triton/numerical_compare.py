"""
Numerical comparison: V1 Triton vs deepdive_kai0 JAX original.

Full input/output pipeline alignment per V1 test.py PiModelEvaluator:
  - state: pad 14→32, normalize via norm_stats q01/q99, digitize for prompt
  - prompt: "Task: {task}, State: {digitized_state_tokens};\nAction: "
  - image: normalize uint8→[-1,1], resize_with_pad to 224x224
  - Triton output: unnormalize via q01/q99, [:, :14], + ori_state, gripper untweak

Two-phase script (each in its own venv).

Usage:
    # Phase JAX
    OPENPI_EXTRA_CONFIG=<ckpt>/train_config.json \\
    kai0/.venv/bin/python optimize/v1_triton/numerical_compare.py jax \\
        --ckpt <ckpt> --base-config-name <name> --out /tmp/jax_out.npz

    # Phase Triton
    kai0/.venv_5090_trt/bin/python optimize/v1_triton/numerical_compare.py triton \\
        --pkl <pkl> --inputs /tmp/jax_out.npz \\
        --norm-stats <ckpt>/assets/<asset_id>/norm_stats.json \\
        --tokenizer-model /data1/tim/workspace/deepdive_kai0/openpi_cache/big_vision/paligemma_tokenizer.model \\
        --out /tmp/triton_out.npz

    # Phase Compare
    kai0/.venv_5090_trt/bin/python optimize/v1_triton/numerical_compare.py compare \\
        --jax /tmp/jax_out.npz --triton /tmp/triton_out.npz
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared transforms (mirror V1 test.py PiModelEvaluator)
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    current = x.shape[axis]
    if current >= target_dim:
        return x
    pad = [(0, 0)] * len(x.shape)
    pad[axis] = (0, target_dim - current)
    return np.pad(x, pad)


def normalize_state_q01q99(state: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def digitize_state(state_normed: np.ndarray) -> np.ndarray:
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    return np.digitize(state_normed, bins=bins) - 1


def unnormalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """V1 test.py uses (actions + 1) / 2 * (q99-q01) + q01"""
    return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


# ─────────────────────────────────────────────────────────────────────────────
# Phase JAX
# ─────────────────────────────────────────────────────────────────────────────

def phase_jax(args):
    import jax
    sys.path.insert(0, "/home/tim/workspace/deepdive_kai0/kai0/src")
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    print(f"[JAX] devices: {jax.devices()}")
    print(f"[JAX] loading config {args.base_config_name} ...")
    cfg = _config.get_config(args.base_config_name)

    print(f"[JAX] loading policy from {args.ckpt} ...")
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)

    rng = np.random.RandomState(42)
    image_hwc_u8 = (rng.rand(224, 224, 3) * 255).astype(np.uint8)
    state_14 = (rng.randn(14) * 0.3).astype(np.float32)
    # Fixed noise (50, 32) shared with Triton
    noise = rng.randn(50, 32).astype(np.float32)

    # deepdive_kai0 / agilex_policy expects: top_head, hand_left, hand_right
    # Image format: uint8 HWC (agilex_policy will normalize internally)
    obs = {
        "images": {
            "top_head": image_hwc_u8,
            "hand_left": image_hwc_u8.copy(),
            "hand_right": image_hwc_u8.copy(),
        },
        "state": state_14,
        "prompt": args.prompt,
    }
    print(f"[JAX] running policy.infer(obs, noise=noise) with fixed noise ...")
    out = policy.infer(obs, noise=noise)
    action = np.array(out["actions"])  # (50, 14) unnormalized + state-added
    print(f"[JAX] action shape: {action.shape}, dtype={action.dtype}")
    print(f"[JAX] sample[0]: {action[0]}")
    print(f"[JAX] range: [{action.min():.4f}, {action.max():.4f}]")

    np.savez(args.out,
             image=image_hwc_u8, state=state_14, noise=noise,
             action=action, prompt=args.prompt)
    print(f"[JAX] saved to {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase Triton — full transform mirroring V1 test.py
# ─────────────────────────────────────────────────────────────────────────────

def phase_triton(args):
    import torch
    import sentencepiece
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pi05_infer import Pi05Inference

    print(f"[Triton] loading inputs from {args.inputs}")
    data = np.load(args.inputs, allow_pickle=True)
    image_hwc_u8 = data["image"]  # (224, 224, 3) uint8
    ori_state = data["state"].astype(np.float32)  # (14,) raw
    noise_np = data["noise"]  # (50, 32) shared
    prompt = str(data["prompt"]) if "prompt" in data.files else args.prompt
    print(f"[Triton] prompt='{prompt}', state_dim={ori_state.shape}")

    # Load norm_stats
    with open(args.norm_stats) as f:
        norm_stats = json.load(f).get("norm_stats", {})
    q01_state = pad_to_dim(np.array(norm_stats["state"]["q01"]), 32)
    q99_state = pad_to_dim(np.array(norm_stats["state"]["q99"]), 32)
    q01_act = pad_to_dim(np.array(norm_stats["actions"]["q01"]), 32)
    q99_act = pad_to_dim(np.array(norm_stats["actions"]["q99"]), 32)

    # Normalize + digitize state for prompt (matches V1 PaliGemma path)
    state_padded = pad_to_dim(ori_state, 32)
    state_normed = normalize_state_q01q99(state_padded, q01_state, q99_state)
    state_digitized = digitize_state(state_normed)  # int array (32,) in [0,255]

    # Load sentencepiece tokenizer (deepdive_kai0 path)
    sp = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer_model)
    state_str = " ".join(map(str, state_digitized.tolist()))
    full_prompt = f"Task: {prompt.strip().replace('_', ' ')}, State: {state_str};\nAction: "
    print(f"[Triton] full_prompt (first 150 chars): {full_prompt[:150]}")
    token_ids = sp.encode(full_prompt, add_bos=True)
    print(f"[Triton] tokenized prompt length: {len(token_ids)}")

    # Load ckpt
    with open(args.pkl, "rb") as f:
        ckpt = pickle.load(f)

    # Build Pi05Inference with discrete_state_input=False, but we'll override language_embeds
    # to include the digitized state encoding
    # Re-encode lang embeds using ckpt's embedding_weight
    embedding_weight_torch = ckpt["embedding_weight"].cuda()  # (vocab, 2048)
    import torch.nn as nn
    emb = nn.Embedding(num_embeddings=embedding_weight_torch.shape[0], embedding_dim=2048).bfloat16().cuda()
    with torch.no_grad():
        emb.weight.copy_(embedding_weight_torch)
    lang_tokens = torch.tensor(token_ids, dtype=torch.long, device="cuda")
    lang_embeds = emb(lang_tokens) * (2048 ** 0.5)
    lang_embeds_cpu = lang_embeds.to("cpu")
    ckpt["language_embeds"] = lang_embeds_cpu  # replace with state-conditioned embeds
    print(f"[Triton] language_embeds shape: {lang_embeds_cpu.shape}")

    print(f"[Triton] building Pi05Inference ...")
    infer = Pi05Inference(ckpt, num_views=3, chunk_size=50, discrete_state_input=False)

    # Image: uint8 HWC → float [-1, 1] bf16, 3 copies
    img_f = image_hwc_u8.astype(np.float32) / 255.0 * 2.0 - 1.0  # (224, 224, 3) [-1,1]
    images_3 = np.stack([img_f, img_f, img_f], axis=0)  # (3, 224, 224, 3)
    input_image = torch.tensor(images_3, dtype=torch.bfloat16, device="cuda")
    input_noise = torch.tensor(noise_np, dtype=torch.bfloat16, device="cuda")

    print(f"[Triton] running Pi05Inference.forward() with fixed noise ...")
    raw = infer.forward(input_image, input_noise)
    action_padded_normed = raw.cpu().float().numpy()  # (50, 32) in [-1, 1] roughly
    print(f"[Triton] raw range: [{action_padded_normed.min():.4f}, {action_padded_normed.max():.4f}]")

    # Unnormalize via q01/q99 (only valid 14 dims; padding is junk)
    action_padded_unnorm = unnormalize_actions(action_padded_normed, q01_act, q99_act)
    action_14 = action_padded_unnorm[:, :14]
    # deepdive_kai0 trains on absolute action (NOT delta-from-state like V1's reference).
    # So we skip "+ ori_state" — matches what openpi.policies.agilex_policy outputs from JAX.
    # (V1 test.py adds ori_state because their target Pi05 was trained on delta; ours isn't.)

    print(f"[Triton] post-process shape: {action_14.shape}")
    print(f"[Triton] sample[0]: {action_14[0]}")
    print(f"[Triton] range: [{action_14.min():.4f}, {action_14.max():.4f}]")

    np.savez(args.out, action=action_14, action_raw=action_padded_normed)
    print(f"[Triton] saved to {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase Compare
# ─────────────────────────────────────────────────────────────────────────────

def phase_compare(args):
    jax_data = np.load(args.jax, allow_pickle=True)
    trt_data = np.load(args.triton, allow_pickle=True)
    a_jax = jax_data["action"]
    a_trt = trt_data["action"]
    if a_jax.shape != a_trt.shape:
        min_d = min(a_jax.shape[-1], a_trt.shape[-1])
        a_jax = a_jax[..., :min_d]
        a_trt = a_trt[..., :min_d]

    diff = np.abs(a_jax - a_trt)
    print()
    print("=" * 60)
    print("Numerical comparison: V1 Triton vs deepdive_kai0 JAX")
    print("=" * 60)
    print(f"  JAX shape:    {a_jax.shape}, dtype={a_jax.dtype}")
    print(f"  Triton shape: {a_trt.shape}, dtype={a_trt.dtype}")
    print(f"  JAX    range: [{a_jax.min():.4f}, {a_jax.max():.4f}]  mean abs = {np.abs(a_jax).mean():.4f}")
    print(f"  Triton range: [{a_trt.min():.4f}, {a_trt.max():.4f}]  mean abs = {np.abs(a_trt).mean():.4f}")
    print(f"  maxabs diff: {diff.max():.4e}")
    print(f"  mean abs diff: {diff.mean():.4e}")
    print(f"  median abs diff: {np.median(diff):.4e}")
    print(f"  rel (mean_abs / mean_abs_jax): {diff.mean() / max(1e-9, np.abs(a_jax).mean()):.4f}")
    per_dim = diff.mean(axis=0)
    print(f"  per-dim MAE: {per_dim}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)

    p_jax = sub.add_parser("jax")
    p_jax.add_argument("--ckpt", required=True)
    p_jax.add_argument("--base-config-name", required=True)
    p_jax.add_argument("--prompt", default="Flatten and fold the cloth")
    p_jax.add_argument("--out", required=True)

    p_trt = sub.add_parser("triton")
    p_trt.add_argument("--pkl", required=True)
    p_trt.add_argument("--inputs", required=True)
    p_trt.add_argument("--norm-stats", required=True, help="norm_stats.json path")
    p_trt.add_argument("--tokenizer-model", required=True, help="sentencepiece .model path")
    p_trt.add_argument("--prompt", default="Flatten and fold the cloth")
    p_trt.add_argument("--out", required=True)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--jax", required=True)
    p_cmp.add_argument("--triton", required=True)

    args = parser.parse_args()
    {"jax": phase_jax, "triton": phase_triton, "compare": phase_compare}[args.phase](args)


if __name__ == "__main__":
    main()
