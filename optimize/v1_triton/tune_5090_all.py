"""
5090 BLOCK_SIZE autotune — 扩展版, 同时 tune 4 个 hot kernels.

Hot kernels in decoder (180× per forward each):
  - matmul_small_gate     (FFN gate+up 1024→4096)  ← 已 tune: (32,64,128) = -7.8%
  - matmul_small_res_gate via matmul_k_4096_1024_gate (FFN down 4096→1024)
  - matmul_small_res_gate via matmul_k_2048_1024_gate (Attn O 2048→1024)
  - matmul_rope_qkv via matmul_k_1024_2560_qkv_rope (QKV+RoPE 1024→2560)

Strategy: greedy — fix the best config from previous step, sweep the next kernel.

Usage:
    CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090_trt/bin/python \\
        optimize/v1_triton/tune_5090_all.py \\
        --pkl optimize/results/task_a_mix_b6000_p1200_v1.pkl --n-test 50
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Best from previous sweep
BEST_GATE_CFG = (32, 64, 128)  # matmul_small_gate (FFN gate+up)


# Candidate BLOCK_SIZE for matmul_small_res_gate (used by FFN down & Attn O)
# V1 defaults:
#   FFN down (4096→1024):  N=16, M=32, K=256
#   Attn O   (2048→1024):  N=32, M=32, K=128
# Try "small N, big K" pattern that won for matmul_small_gate
RES_GATE_CANDIDATES = [
    # (N, M, K)
    None,           # placeholder for V1 default per-call-site
    (16, 32, 256),  # match FFN down default
    (32, 32, 128),  # match Attn O default
    (16, 32, 128),
    (32, 64, 128),
    (16, 64, 256),
    (32, 32, 256),
    (16, 16, 256),
    (32, 16, 128),
    (8,  32, 256),  # tiny N
    (16, 32, 512),  # huge K
]


def make_decoder_with_configs(gate_cfg, ffn_down_cfg, attn_o_cfg):
    """Build a transformer_decoder with custom BLOCK_SIZE for 3 hot kernels."""
    import pi05_infer
    from pi05_infer import (
        matmul_k_32_1024_bias, adarms_norm_style_proj,
        matmul_k_1024_2560_qkv_rope, matmul_abT_scale,
        softmax_kernel_prefix_suffix, matmul_k8_n_256,
        adarms_matmul_k_1024_32_bias_res,
    )
    from pi0_infer import matmul_small_gate
    from pi05_infer import matmul_small_res_gate

    gN, gM, gK = gate_cfg
    fN, fM, fK = ffn_down_cfg  # FFN down: features=4096, hidden=1024
    aN, aM, aK = attn_o_cfg   # Attn O: features=2048, hidden=1024

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
                    buffers['decoder_x'], buffers['decoder_time_emb'][step],
                    weights['decoder_pre_attn_norm_mod_w'][i],
                    weights['decoder_pre_attn_norm_mod_b'][i],
                    buffers['x_normed_buf'], buffers['gate_buf'],
                    buffers['decoder_style_attn'][step, i]
                )
                matmul_k_1024_2560_qkv_rope(
                    buffers['x_normed_buf'], weights['decoder_attn_qkv_w'][i],
                    buffers['decoder_rope_weights'], buffers['decoder_q_buf'],
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
                    buffers['valid_encoder_len'], buffers['decoder_attn_buf'],
                    BLOCK_SIZE_M=4, BLOCK_SIZE=1024,
                )
                matmul_k8_n_256(
                    buffers['decoder_attn_buf'],
                    buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                    buffers['decoder_q_buf'],
                )
                # === Attn O (matmul_k_2048_1024_gate -> matmul_small_res_gate) ===
                # features=2048, hidden=1024
                grid_n = (seq_len + aN - 1) // aN
                grid_m = (1024 + aM - 1) // aM
                matmul_small_res_gate[(grid_n * grid_m,)](
                    buffers['decoder_q_buf'].view(-1, 2048),
                    weights['decoder_attn_o_w'][i],
                    buffers['decoder_x'],  # out
                    buffers['decoder_x'],  # res (in-place)
                    buffers['gate_buf'],
                    seq_len=seq_len, features=2048, hidden=1024,
                    BLOCK_SIZE_N=aN, BLOCK_SIZE_M=aM, BLOCK_SIZE_K=aK,
                )
                adarms_norm_style_proj(
                    buffers['decoder_x'], buffers['decoder_time_emb'][step],
                    weights['decoder_pre_ffn_norm_mod_w'][i],
                    weights['decoder_pre_ffn_norm_mod_b'][i],
                    buffers['x_normed_buf'], buffers['gate_buf'],
                    buffers['decoder_style_ffn'][step, i]
                )
                seq_len = buffers['decoder_x'].shape[0]
                # === FFN gate+up (matmul_small_gate) ===
                grid_n_gate = (seq_len + gN - 1) // gN
                grid_m_gate = (4096 + gM - 1) // gM
                matmul_small_gate[(grid_n_gate, grid_m_gate)](
                    buffers['x_normed_buf'],
                    weights['decoder_ffn_gate_w'][i],
                    weights['decoder_ffn_up_w'][i],
                    buffers['decoder_hidden'],
                    seq_len, 1024, 4096,
                    BLOCK_SIZE_N=gN, BLOCK_SIZE_M=gM, BLOCK_SIZE_K=gK,
                )
                # === FFN down (matmul_k_4096_1024_gate -> matmul_small_res_gate) ===
                # features=4096, hidden=1024
                grid_n_f = (seq_len + fN - 1) // fN
                grid_m_f = (1024 + fM - 1) // fM
                matmul_small_res_gate[(grid_n_f * grid_m_f,)](
                    buffers['decoder_hidden'],
                    weights['decoder_ffn_down_w'][i],
                    buffers['decoder_x'],  # out
                    buffers['decoder_x'],  # res
                    buffers['gate_buf'],
                    seq_len=seq_len, features=4096, hidden=1024,
                    BLOCK_SIZE_N=fN, BLOCK_SIZE_M=fM, BLOCK_SIZE_K=fK,
                )

            adarms_matmul_k_1024_32_bias_res(
                buffers['decoder_x'], buffers['decoder_time_emb'][step],
                weights['decoder_final_norm_mod_w'], weights['decoder_final_norm_mod_b'],
                buffers['x_normed_buf'], buffers['gate_buf'],
                buffers['decoder_style_final'][step],
                weights['decoder_action_out_proj_w'], weights['decoder_action_out_proj_b'],
                buffers['diffusion_noise'], buffers['diffusion_noise'],
            )

    return transformer_decoder_5090


def benchmark_variant(ckpt, num_views, chunk_size, n_warmup, n_test,
                      gate_cfg, ffn_down_cfg, attn_o_cfg):
    import pi05_infer
    new_decoder = make_decoder_with_configs(gate_cfg, ffn_down_cfg, attn_o_cfg)
    orig_decoder = pi05_infer.transformer_decoder
    orig_model = pi05_infer.pi05_model
    pi05_infer.transformer_decoder = new_decoder

    def patched_model(weights, buffers, num_views_, encoder_seq_len_, num_steps_=10):
        pi05_infer.vision_encoder(weights, buffers, num_views_)
        pi05_infer.transformer_encoder(weights, buffers, encoder_seq_len_)
        new_decoder(weights, buffers, encoder_seq_len_, num_steps_)
    pi05_infer.pi05_model = patched_model

    try:
        infer = pi05_infer.Pi05Inference(
            ckpt, num_views=num_views, chunk_size=chunk_size, discrete_state_input=False,
        )
        input_image = torch.randn(num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
        input_noise = torch.randn(chunk_size, 32, dtype=torch.bfloat16, device="cuda")

        for _ in range(n_warmup):
            _ = infer.forward(input_image, input_noise)
            torch.cuda.synchronize()

        times = []
        for _ in range(n_test):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = infer.forward(input_image, input_noise)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return np.array(times)
    finally:
        pi05_infer.transformer_decoder = orig_decoder
        pi05_infer.pi05_model = orig_model
        torch.cuda.empty_cache()


def sweep_kernel(name, ckpt, args, fixed_others, candidates, current_default):
    """Sweep candidates for one kernel, return best cfg + stats."""
    print()
    print("=" * 70)
    print(f"Sweeping {name}")
    print(f"  Fixed: {fixed_others}")
    print("=" * 70)

    results = []
    for idx, cfg in enumerate(candidates):
        effective_cfg = current_default if cfg is None else cfg
        label = "V1 default" if cfg is None else ""
        cfg_dict = {**fixed_others, name: effective_cfg}
        print(f"[{idx+1}/{len(candidates)}] {name}={effective_cfg} {label} ...", flush=True)
        try:
            arr = benchmark_variant(ckpt, args.num_views, args.chunk_size,
                                     args.n_warmup, args.n_test, **cfg_dict)
            stats = {"cfg": effective_cfg, "mean": float(arr.mean()),
                     "p50": float(np.percentile(arr, 50)), "p99": float(np.percentile(arr, 99))}
            results.append(stats)
            print(f"      mean={stats['mean']:.2f}ms p50={stats['p50']:.2f}ms p99={stats['p99']:.2f}ms")
        except Exception as e:
            print(f"      FAILED: {type(e).__name__}: {str(e)[:150]}")
            results.append({"cfg": effective_cfg, "mean": float("inf"), "error": str(e)[:100]})

    valid = [r for r in results if r["mean"] < float("inf")]
    if not valid:
        return current_default, None
    best = min(valid, key=lambda r: r["mean"])
    return best["cfg"], best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()

    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    with open(args.pkl, "rb") as f:
        ckpt = pickle.load(f)

    # Step 1: Confirm baseline + previously-found best gate cfg
    print("\n[Step 1] Re-confirm BEST_GATE_CFG = (32, 64, 128) for matmul_small_gate")
    arr_baseline = benchmark_variant(ckpt, args.num_views, args.chunk_size,
                                      args.n_warmup, args.n_test,
                                      gate_cfg=(128, 64, 32),
                                      ffn_down_cfg=(16, 32, 256),
                                      attn_o_cfg=(32, 32, 128))
    print(f"  V1 default all: mean={arr_baseline.mean():.2f}ms p50={np.percentile(arr_baseline, 50):.2f}ms")

    arr_g = benchmark_variant(ckpt, args.num_views, args.chunk_size,
                               args.n_warmup, args.n_test,
                               gate_cfg=BEST_GATE_CFG,
                               ffn_down_cfg=(16, 32, 256),
                               attn_o_cfg=(32, 32, 128))
    print(f"  + best gate cfg: mean={arr_g.mean():.2f}ms p50={np.percentile(arr_g, 50):.2f}ms ({(arr_baseline.mean() - arr_g.mean())/arr_baseline.mean()*100:+.2f}%)")

    # Step 2: Sweep FFN down (matmul_small_res_gate with features=4096 hidden=1024)
    best_ffn_cfg, _ = sweep_kernel(
        "ffn_down_cfg", ckpt, args,
        fixed_others={"gate_cfg": BEST_GATE_CFG, "attn_o_cfg": (32, 32, 128)},
        candidates=RES_GATE_CANDIDATES,
        current_default=(16, 32, 256),
    )
    print(f"\n  >>> Best FFN down cfg: {best_ffn_cfg}")

    # Step 3: Sweep Attn O (matmul_small_res_gate with features=2048 hidden=1024)
    best_attn_o_cfg, _ = sweep_kernel(
        "attn_o_cfg", ckpt, args,
        fixed_others={"gate_cfg": BEST_GATE_CFG, "ffn_down_cfg": best_ffn_cfg},
        candidates=RES_GATE_CANDIDATES,
        current_default=(32, 32, 128),
    )
    print(f"\n  >>> Best Attn O cfg: {best_attn_o_cfg}")

    # Final benchmark with all best configs
    print()
    print("=" * 70)
    print("FINAL CONFIG (all 3 kernels tuned)")
    print("=" * 70)
    arr_final = benchmark_variant(
        ckpt, args.num_views, args.chunk_size, args.n_warmup, args.n_test,
        gate_cfg=BEST_GATE_CFG,
        ffn_down_cfg=best_ffn_cfg,
        attn_o_cfg=best_attn_o_cfg,
    )
    print(f"  matmul_small_gate  (FFN gate+up): {BEST_GATE_CFG}")
    print(f"  matmul_small_res_gate (FFN down): {best_ffn_cfg}")
    print(f"  matmul_small_res_gate (Attn O ): {best_attn_o_cfg}")
    print(f"  Mean: {arr_final.mean():.2f} ms")
    print(f"  P50:  {np.percentile(arr_final, 50):.2f} ms")
    print(f"  P99:  {np.percentile(arr_final, 99):.2f} ms")
    print(f"  Std:  {arr_final.std():.3f} ms")
    print()
    print(f"vs V1 default all (4090-tuned): {arr_baseline.mean():.2f} ms")
    print(f"Improvement: {(arr_baseline.mean() - arr_final.mean())/arr_baseline.mean()*100:+.2f}%")


if __name__ == "__main__":
    main()
