"""
5090 BLOCK_SIZE autotune for V1 hot kernels.

Strategy: monkey-patch transformer_decoder with variants that pass different
BLOCK_SIZE to the hottest GEMMs (matmul_small_gate FFN gate+up, called 180×
per forward). For each variant, rebuild Pi05Inference (re-captures CUDA Graph
with new kernel specializations) and run 100-iter benchmark. Report best.

Baseline (4090-tuned defaults): 35.4 ms P50 on 5090.

Usage:
    cd /home/tim/workspace/deepdive_kai0
    CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090_trt/bin/python \\
        optimize/v1_triton/tune_5090.py \\
        --pkl optimize/results/task_a_mix_b6000_p1200_v1.pkl \\
        --n-test 50
"""
import argparse
import os
import pickle
import sys
import time
import importlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────────────
# Candidate BLOCK_SIZE configs for matmul_small_gate (FFN gate+up 1024→4096)
# ──────────────────────────────────────────────────────────────────────
# Tuple: (BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_K)
# V1 default: (128, 64, 32). Try larger M/K to exploit 5090 L2 (96MB vs 4090 64MB)
GATE_FFN_CANDIDATES = [
    (128, 64, 32),    # V1 default (baseline)
    (64,  64, 64),    # smaller N, bigger K
    (128, 128, 32),   # bigger M
    (128, 128, 64),   # bigger M+K (5090 L2 friendly)
    (64,  128, 64),
    (256, 64, 32),    # bigger N (5090 has more SMs)
    (128, 64, 64),    # bigger K only
    (32,  64, 128),   # small NM, huge K
    (64,  64, 32),    # smaller everything
    (256, 128, 64),   # massive (test if fits)
]


def make_patched_decoder(gate_cfg):
    """Create a transformer_decoder variant with custom matmul_small_gate BLOCK_SIZE.

    Only matmul_small_gate (FFN gate+up, hottest) is varied. Other kernels keep
    V1 defaults to isolate the experiment.
    """
    import pi05_infer
    # Local imports of all original symbols (mirror transformer_decoder body)
    from pi05_infer import (
        matmul_k_32_1024_bias,
        adarms_norm_style_proj,
        matmul_k_1024_2560_qkv_rope,
        matmul_abT_scale,
        softmax_kernel_prefix_suffix,
        matmul_k8_n_256,
        matmul_k_2048_1024_gate,
        matmul_k_4096_1024_gate,
        adarms_matmul_k_1024_32_bias_res,
    )
    from pi0_infer import matmul_small_gate

    bN, bM, bK = gate_cfg
    # grid: (num_blocks_along_seq, num_blocks_along_hidden)
    # With persistent kernel pattern in matmul_small_gate, grid = (ceil(seq_len/bN), ceil(hidden/bM))
    # But seq_len=50 < bN often, so cdiv(50, 128)=1, cdiv(50, 64)=1, etc.
    # The persistent pattern walks p over (grid_i * grid_j) anyway, so grid size is fine.

    def transformer_decoder_5090(weights, buffers, encoder_seq_len, num_steps=10):
        for step in range(num_steps):
            matmul_k_32_1024_bias(
                buffers['diffusion_noise'],
                weights['decoder_action_in_proj_w'],
                weights['decoder_action_in_proj_b'],
                buffers['decoder_x']
            )
            seq_len = buffers['decoder_x'].shape[0]
            for i in range(18):
                adarms_norm_style_proj(
                    buffers['decoder_x'],
                    buffers['decoder_time_emb'][step],
                    weights['decoder_pre_attn_norm_mod_w'][i],
                    weights['decoder_pre_attn_norm_mod_b'][i],
                    buffers['x_normed_buf'],
                    buffers['gate_buf'],
                    buffers['decoder_style_attn'][step, i]
                )
                matmul_k_1024_2560_qkv_rope(
                    buffers['x_normed_buf'],
                    weights['decoder_attn_qkv_w'][i],
                    buffers['decoder_rope_weights'],
                    buffers['decoder_q_buf'],
                    buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len + seq_len],
                    buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len + seq_len],
                )
                total_queries = buffers['decoder_q_buf'].shape[0]
                prefix_keys = encoder_seq_len
                suffix_keys = seq_len
                total_keys = prefix_keys + suffix_keys

                matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                    buffers['decoder_q_buf'],
                    buffers['encoder_K'][i, :encoder_seq_len + seq_len],
                    buffers['decoder_logits_buf'],
                    total_queries, total_keys, 256, 256 ** -0.5,
                    BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64,
                )
                softmax_kernel_prefix_suffix[((total_queries + 3) // 4,)](
                    buffers['decoder_logits_buf'],
                    total_queries, prefix_keys, suffix_keys,
                    buffers['valid_encoder_len'],
                    buffers['decoder_attn_buf'],
                    BLOCK_SIZE_M=4, BLOCK_SIZE=1024,
                )
                matmul_k8_n_256(
                    buffers['decoder_attn_buf'],
                    buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                    buffers['decoder_q_buf'],
                )
                matmul_k_2048_1024_gate(
                    buffers['decoder_q_buf'].view(-1, 2048),
                    weights['decoder_attn_o_w'][i],
                    buffers['decoder_x'],
                    buffers['gate_buf']
                )
                adarms_norm_style_proj(
                    buffers['decoder_x'],
                    buffers['decoder_time_emb'][step],
                    weights['decoder_pre_ffn_norm_mod_w'][i],
                    weights['decoder_pre_ffn_norm_mod_b'][i],
                    buffers['x_normed_buf'],
                    buffers['gate_buf'],
                    buffers['decoder_style_ffn'][step, i]
                )
                seq_len = buffers['decoder_x'].shape[0]
                # === CUSTOM BLOCK_SIZE for matmul_small_gate (FFN gate+up) ===
                grid_n = (seq_len + bN - 1) // bN
                grid_m = (4096 + bM - 1) // bM
                matmul_small_gate[(grid_n, grid_m)](
                    buffers['x_normed_buf'],
                    weights['decoder_ffn_gate_w'][i],
                    weights['decoder_ffn_up_w'][i],
                    buffers['decoder_hidden'],
                    seq_len, 1024, 4096,
                    BLOCK_SIZE_N=bN, BLOCK_SIZE_M=bM, BLOCK_SIZE_K=bK,
                )
                # ============================================================
                matmul_k_4096_1024_gate(
                    buffers['decoder_hidden'],
                    weights['decoder_ffn_down_w'][i],
                    buffers['decoder_x'],
                    buffers['gate_buf']
                )

            adarms_matmul_k_1024_32_bias_res(
                buffers['decoder_x'],
                buffers['decoder_time_emb'][step],
                weights['decoder_final_norm_mod_w'],
                weights['decoder_final_norm_mod_b'],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
                buffers['decoder_style_final'][step],
                weights['decoder_action_out_proj_w'],
                weights['decoder_action_out_proj_b'],
                buffers['diffusion_noise'],
                buffers['diffusion_noise'],
            )

    return transformer_decoder_5090


def benchmark_one_variant(ckpt, num_views, chunk_size, n_warmup, n_test, gate_cfg):
    """Patch transformer_decoder + pi05_model with given config, build Pi05Inference, benchmark."""
    import pi05_infer

    # Patch
    original_decoder = pi05_infer.transformer_decoder
    original_model = pi05_infer.pi05_model
    new_decoder = make_patched_decoder(gate_cfg)
    pi05_infer.transformer_decoder = new_decoder

    # pi05_model imports transformer_decoder via local scope; need to re-define it too
    def patched_pi05_model(weights, buffers, num_views_, encoder_seq_len_, num_steps_=10):
        pi05_infer.vision_encoder(weights, buffers, num_views_)
        pi05_infer.transformer_encoder(weights, buffers, encoder_seq_len_)
        new_decoder(weights, buffers, encoder_seq_len_, num_steps_)
    pi05_infer.pi05_model = patched_pi05_model

    try:
        # Build inference (this captures CUDA Graph with new kernels)
        infer = pi05_infer.Pi05Inference(
            ckpt, num_views=num_views, chunk_size=chunk_size,
            discrete_state_input=False,
        )
        input_image = torch.randn(num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
        input_noise = torch.randn(chunk_size, 32, dtype=torch.bfloat16, device="cuda")

        # Warm-up
        for _ in range(n_warmup):
            _ = infer.forward(input_image, input_noise)
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_test):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = infer.forward(input_image, input_noise)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return np.array(times)
    finally:
        # Restore
        pi05_infer.transformer_decoder = original_decoder
        pi05_infer.pi05_model = original_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()

    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Loading checkpoint from {args.pkl} ...")
    with open(args.pkl, "rb") as f:
        ckpt = pickle.load(f)

    print(f"\nTuning matmul_small_gate (decoder FFN gate+up, 180× per forward)")
    print(f"Candidates: {len(GATE_FFN_CANDIDATES)}")
    print(f"Iters per variant: {args.n_test}")
    print()

    results = []
    for idx, cfg in enumerate(GATE_FFN_CANDIDATES):
        print(f"[{idx+1}/{len(GATE_FFN_CANDIDATES)}] Trying gate_cfg={cfg} ...", flush=True)
        try:
            arr = benchmark_one_variant(
                ckpt, args.num_views, args.chunk_size,
                args.n_warmup, args.n_test, cfg
            )
            stats = {
                "cfg": cfg,
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p99": float(np.percentile(arr, 99)),
                "std": float(arr.std()),
            }
            results.append(stats)
            print(f"      mean={stats['mean']:.2f}ms p50={stats['p50']:.2f}ms p99={stats['p99']:.2f}ms std={stats['std']:.3f}")
        except Exception as e:
            print(f"      FAILED: {type(e).__name__}: {str(e)[:200]}")
            results.append({"cfg": cfg, "mean": float("inf"), "error": str(e)[:100]})
        finally:
            torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("FINAL RANKING")
    print("=" * 70)
    print(f"{'rank':<5}{'cfg (N,M,K)':<18}{'mean':>10}{'p50':>10}{'p99':>10}{'speedup':>10}")
    valid = [r for r in results if r["mean"] < float("inf")]
    baseline = next((r["mean"] for r in valid if r["cfg"] == GATE_FFN_CANDIDATES[0]), valid[0]["mean"])
    valid_sorted = sorted(valid, key=lambda r: r["mean"])
    for rank, r in enumerate(valid_sorted, 1):
        speedup = baseline / r["mean"]
        marker = " ← V1 default" if r["cfg"] == GATE_FFN_CANDIDATES[0] else ""
        print(f"{rank:<5}{str(r['cfg']):<18}{r['mean']:>9.2f}ms{r['p50']:>9.2f}ms{r['p99']:>9.2f}ms{speedup:>9.2f}x{marker}")

    if results[0].get("mean", float("inf")) < float("inf"):
        best = valid_sorted[0]
        if best["cfg"] != GATE_FFN_CANDIDATES[0]:
            improve = (baseline - best["mean"]) / baseline * 100
            print(f"\nBEST: cfg={best['cfg']}  mean={best['mean']:.2f}ms  ({improve:+.1f}% vs V1 default)")
        else:
            print(f"\nV1 default already optimal on this hardware.")


if __name__ == "__main__":
    main()
