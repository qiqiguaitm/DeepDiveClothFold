"""
5090 Triton pipelining tune — sweep num_warps + num_stages for hot kernels.

Background: Step 6 tuned BLOCK_SIZE; this tunes the orthogonal dimension —
how Triton pipelines memory loads + matmul (controls async copy + register
reuse). For sm_120 (Blackwell), num_warps=8 + num_stages=3-4 often enables
wgmma + multi-buffered async load (vs default num_warps=4, num_stages=2).

Kernels swept:
  - matmul_small_gate (FFN gate+up, current best BLOCK=(32,64,128))
  - matmul_small_res_gate FFN down (current best (16,32,512))
  - matmul_small_res_gate Attn O (current best (16,32,256))

Usage:
    CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090_trt/bin/python \\
        optimize/v1_triton/tune_5090_pipelining.py \\
        --pkl optimize/results/task_a_mix_b6000_p1200_v1.pkl
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Current best BLOCK_SIZE (from Step 6)
BEST_GATE_BLOCK = (32, 64, 128)
BEST_FFN_DOWN_BLOCK = (16, 32, 512)
BEST_ATTN_O_BLOCK = (16, 32, 256)


# Pipelining configs to sweep (num_warps, num_stages)
PIPE_CONFIGS = [
    (4, 2),   # default
    (4, 3),
    (4, 4),
    (8, 2),
    (8, 3),   # commonly best for wgmma
    (8, 4),
    (16, 3),  # might be too many for small block
    (16, 4),
]


def make_decoder(pipe_gate, pipe_ffn, pipe_attno):
    """Build a transformer_decoder where 3 hot kernels are called with given num_warps/num_stages."""
    import pi05_infer
    from pi05_infer import (
        matmul_k_32_1024_bias, adarms_norm_style_proj,
        matmul_k_1024_2560_qkv_rope, matmul_abT_scale,
        softmax_kernel_prefix_suffix, matmul_k8_n_256,
        adarms_matmul_k_1024_32_bias_res, matmul_small_res_gate,
    )
    from pi0_infer import matmul_small_gate

    gN, gM, gK = BEST_GATE_BLOCK
    fN, fM, fK = BEST_FFN_DOWN_BLOCK
    aN, aM, aK = BEST_ATTN_O_BLOCK
    g_warps, g_stages = pipe_gate
    f_warps, f_stages = pipe_ffn
    a_warps, a_stages = pipe_attno

    def transformer_decoder_pipe(weights, buffers, encoder_seq_len, num_steps=10):
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
                total_keys = encoder_seq_len + seq_len
                matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                    buffers['decoder_q_buf'],
                    buffers['encoder_K'][i, :encoder_seq_len + seq_len],
                    buffers['decoder_logits_buf'],
                    total_queries, total_keys, 256, 256 ** -0.5,
                    BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64,
                )
                softmax_kernel_prefix_suffix[((total_queries + 3) // 4,)](
                    buffers['decoder_logits_buf'],
                    total_queries, encoder_seq_len, seq_len,
                    buffers['valid_encoder_len'], buffers['decoder_attn_buf'],
                    BLOCK_SIZE_M=4, BLOCK_SIZE=1024,
                )
                matmul_k8_n_256(
                    buffers['decoder_attn_buf'],
                    buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                    buffers['decoder_q_buf'],
                )
                # ▼ Attn O (5090 BLOCK + pipelining)
                grid = ((seq_len + aN - 1) // aN) * ((1024 + aM - 1) // aM)
                matmul_small_res_gate[(grid,)](
                    buffers['decoder_q_buf'].view(-1, 2048),
                    weights['decoder_attn_o_w'][i],
                    buffers['decoder_x'], buffers['decoder_x'],
                    buffers['gate_buf'],
                    seq_len=seq_len, features=2048, hidden=1024,
                    BLOCK_SIZE_N=aN, BLOCK_SIZE_M=aM, BLOCK_SIZE_K=aK,
                    num_warps=a_warps, num_stages=a_stages,
                )
                adarms_norm_style_proj(
                    buffers['decoder_x'], buffers['decoder_time_emb'][step],
                    weights['decoder_pre_ffn_norm_mod_w'][i],
                    weights['decoder_pre_ffn_norm_mod_b'][i],
                    buffers['x_normed_buf'], buffers['gate_buf'],
                    buffers['decoder_style_ffn'][step, i]
                )
                seq_len = buffers['decoder_x'].shape[0]
                # ▼ FFN gate+up (5090 BLOCK + pipelining)
                grid_n = (seq_len + gN - 1) // gN
                grid_m = (4096 + gM - 1) // gM
                matmul_small_gate[(grid_n, grid_m)](
                    buffers['x_normed_buf'],
                    weights['decoder_ffn_gate_w'][i],
                    weights['decoder_ffn_up_w'][i],
                    buffers['decoder_hidden'],
                    seq_len, 1024, 4096,
                    BLOCK_SIZE_N=gN, BLOCK_SIZE_M=gM, BLOCK_SIZE_K=gK,
                    num_warps=g_warps, num_stages=g_stages,
                )
                # ▼ FFN down (5090 BLOCK + pipelining)
                grid = ((seq_len + fN - 1) // fN) * ((1024 + fM - 1) // fM)
                matmul_small_res_gate[(grid,)](
                    buffers['decoder_hidden'],
                    weights['decoder_ffn_down_w'][i],
                    buffers['decoder_x'], buffers['decoder_x'],
                    buffers['gate_buf'],
                    seq_len=seq_len, features=4096, hidden=1024,
                    BLOCK_SIZE_N=fN, BLOCK_SIZE_M=fM, BLOCK_SIZE_K=fK,
                    num_warps=f_warps, num_stages=f_stages,
                )

            adarms_matmul_k_1024_32_bias_res(
                buffers['decoder_x'], buffers['decoder_time_emb'][step],
                weights['decoder_final_norm_mod_w'], weights['decoder_final_norm_mod_b'],
                buffers['x_normed_buf'], buffers['gate_buf'],
                buffers['decoder_style_final'][step],
                weights['decoder_action_out_proj_w'], weights['decoder_action_out_proj_b'],
                buffers['diffusion_noise'], buffers['diffusion_noise'],
            )

    return transformer_decoder_pipe


def benchmark(ckpt, num_views, chunk_size, n_warmup, n_test, pipe_gate, pipe_ffn, pipe_attno):
    import pi05_infer
    new_dec = make_decoder(pipe_gate, pipe_ffn, pipe_attno)
    orig_dec = pi05_infer.transformer_decoder
    orig_model = pi05_infer.pi05_model
    pi05_infer.transformer_decoder = new_dec

    def patched(weights, buffers, num_views_, encoder_seq_len_, num_steps_=10):
        pi05_infer.vision_encoder(weights, buffers, num_views_)
        pi05_infer.transformer_encoder(weights, buffers, encoder_seq_len_)
        new_dec(weights, buffers, encoder_seq_len_, num_steps_)
    pi05_infer.pi05_model = patched

    try:
        infer = pi05_infer.Pi05Inference(
            ckpt, num_views=num_views, chunk_size=chunk_size, discrete_state_input=False,
        )
        img = torch.randn(num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
        noise = torch.randn(chunk_size, 32, dtype=torch.bfloat16, device="cuda")
        for _ in range(n_warmup):
            _ = infer.forward(img, noise); torch.cuda.synchronize()
        times = []
        for _ in range(n_test):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = infer.forward(img, noise); torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return np.array(times)
    finally:
        pi05_infer.transformer_decoder = orig_dec
        pi05_infer.pi05_model = orig_model
        torch.cuda.empty_cache()


def sweep(name, ckpt, args, fixed_others):
    print(f"\n=== Sweeping pipelining for {name} ===")
    results = []
    for warps, stages in PIPE_CONFIGS:
        cfg = (warps, stages)
        kwargs = {**fixed_others, name: cfg}
        label = "default" if cfg == (4, 2) else ""
        print(f"  num_warps={warps}, num_stages={stages} {label} ...", flush=True)
        try:
            arr = benchmark(ckpt, args.num_views, args.chunk_size,
                            args.n_warmup, args.n_test, **kwargs)
            stats = {"cfg": cfg, "mean": float(arr.mean()), "p50": float(np.percentile(arr, 50))}
            results.append(stats)
            print(f"    mean={stats['mean']:.2f}ms p50={stats['p50']:.2f}ms")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {str(e)[:120]}")
            results.append({"cfg": cfg, "mean": float("inf"), "error": str(e)[:80]})

    valid = [r for r in results if r["mean"] < float("inf")]
    if not valid:
        return (4, 2), None
    best = min(valid, key=lambda r: r["mean"])
    print(f"  >>> best for {name}: warps={best['cfg'][0]} stages={best['cfg'][1]} → {best['mean']:.2f}ms")
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

    # Re-confirm baseline (all default pipelining (4, 2))
    arr0 = benchmark(ckpt, args.num_views, args.chunk_size, args.n_warmup, args.n_test,
                     pipe_gate=(4, 2), pipe_ffn=(4, 2), pipe_attno=(4, 2))
    print(f"\nBaseline (all 4w/2s): mean={arr0.mean():.2f}ms p50={np.percentile(arr0, 50):.2f}ms")

    # Sweep gate (most important kernel)
    best_gate, _ = sweep(
        "pipe_gate", ckpt, args,
        fixed_others={"pipe_ffn": (4, 2), "pipe_attno": (4, 2)},
    )

    # Sweep ffn_down
    best_ffn, _ = sweep(
        "pipe_ffn", ckpt, args,
        fixed_others={"pipe_gate": best_gate, "pipe_attno": (4, 2)},
    )

    # Sweep attn_o
    best_attno, _ = sweep(
        "pipe_attno", ckpt, args,
        fixed_others={"pipe_gate": best_gate, "pipe_ffn": best_ffn},
    )

    print()
    print("=" * 60)
    print("FINAL")
    print("=" * 60)
    arr_final = benchmark(ckpt, args.num_views, args.chunk_size, args.n_warmup, args.n_test,
                           pipe_gate=best_gate, pipe_ffn=best_ffn, pipe_attno=best_attno)
    print(f"  gate:  num_warps={best_gate[0]} num_stages={best_gate[1]}")
    print(f"  ffn:   num_warps={best_ffn[0]} num_stages={best_ffn[1]}")
    print(f"  attno: num_warps={best_attno[0]} num_stages={best_attno[1]}")
    print(f"  Mean: {arr_final.mean():.2f} ms, P50: {np.percentile(arr_final, 50):.2f} ms")
    print(f"  vs baseline: {(arr0.mean() - arr_final.mean()) / arr0.mean() * 100:+.2f}%")


if __name__ == "__main__":
    main()
