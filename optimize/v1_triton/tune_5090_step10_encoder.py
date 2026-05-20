"""Step 10: Encoder FFN gate+up (rms_matmul_n_2048_16384_gate, 18×) BLOCK_SIZE sweep.

Encoder seq_len = num_views×256 + prompt_len ≈ 775 (3 cam + 7 prompt).
FFN gate+up: 2048 → 16384, biggest single GEMM in the entire forward pass.
Per layer FLOP ≈ 26 GFLOP; 18 layers = 468 GFLOP total.

Fixed: Step 6 best decoder BLOCK_SIZE.
"""
import argparse, os, pickle, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 6 best (decoder, fixed)
BEST_GATE = (32, 64, 128)
BEST_FFN_DOWN = (16, 32, 512)
BEST_ATTNO = (16, 32, 256)

# Encoder FFN gate+up candidates — seq=775, hidden=16384 (very wide)
ENCODER_GATE_CANDIDATES = [
    None,                  # V1 default (128, 64, 32)
    (64, 64, 32),
    (64, 64, 64),
    (128, 64, 64),
    (128, 128, 32),
    (128, 128, 64),
    (256, 64, 32),         # bigger N (seq=775 long)
    (256, 128, 64),        # may OOM
    (64, 128, 64),
    (32, 64, 128),         # Step 6 best (try same)
    (64, 128, 32),
]


def make_encoder_patched(enc_gate_cfg):
    """Replace rms_matmul_n_2048_16384_gate with custom BLOCK_SIZE version."""
    import pi0_infer
    from pi0_infer import rms_norm_kernel, matmul_small_gate

    eN, eM, eK = enc_gate_cfg if enc_gate_cfg else (128, 64, 32)

    def rms_matmul_2048_16384_tuned(x, weight1, weight2, out, x_norm):
        seq_len = x.shape[0]
        rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2048)
        grid_n = (seq_len + eN - 1) // eN
        grid_m = (16384 + eM - 1) // eM
        matmul_small_gate[(grid_n, grid_m)](
            x_norm, weight1, weight2, out,
            seq_len, 2048, 16384,
            BLOCK_SIZE_N=eN, BLOCK_SIZE_M=eM, BLOCK_SIZE_K=eK,
        )
    return rms_matmul_2048_16384_tuned


def bench(ckpt, n_warmup, n_test, enc_gate_cfg):
    import pi05_infer, pi0_infer
    from pi05_infer import (
        matmul_k_32_1024_bias, adarms_norm_style_proj, matmul_k_1024_2560_qkv_rope,
        matmul_abT_scale, softmax_kernel_prefix_suffix, matmul_k8_n_256,
        matmul_small_res_gate, adarms_matmul_k_1024_32_bias_res,
    )
    from pi0_infer import matmul_small_gate

    # Patch the encoder FFN gate+up function in pi0_infer
    orig_enc_ffn = pi0_infer.rms_matmul_n_2048_16384_gate
    pi0_infer.rms_matmul_n_2048_16384_gate = make_encoder_patched(enc_gate_cfg)

    # Note: transformer_encoder in pi0_infer imports rms_matmul_n_2048_16384_gate at top.
    # Need to update its global ref too.
    # Find where it's used and patch the module-level reference.
    # The simplest: patch via the function name in the module — pi0_infer.transformer_encoder
    # accesses pi0_infer.rms_matmul_n_2048_16384_gate at call time, OR by direct reference.
    # Let me check.

    # Also patch decoder to use Step 6 best configs
    gN, gM, gK = BEST_GATE
    fN, fM, fK = BEST_FFN_DOWN
    aN, aM, aK = BEST_ATTNO

    def decoder_tuned(weights, buffers, encoder_seq_len, num_steps=10):
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
                grid = ((seq_len + aN - 1) // aN) * ((1024 + aM - 1) // aM)
                matmul_small_res_gate[(grid,)](
                    buffers['decoder_q_buf'].view(-1, 2048),
                    weights['decoder_attn_o_w'][i],
                    buffers['decoder_x'], buffers['decoder_x'],
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
                grid_n = (seq_len + gN - 1) // gN
                grid_m = (4096 + gM - 1) // gM
                matmul_small_gate[(grid_n, grid_m)](
                    buffers['x_normed_buf'],
                    weights['decoder_ffn_gate_w'][i],
                    weights['decoder_ffn_up_w'][i],
                    buffers['decoder_hidden'],
                    seq_len, 1024, 4096,
                    BLOCK_SIZE_N=gN, BLOCK_SIZE_M=gM, BLOCK_SIZE_K=gK,
                )
                grid = ((seq_len + fN - 1) // fN) * ((1024 + fM - 1) // fM)
                matmul_small_res_gate[(grid,)](
                    buffers['decoder_hidden'],
                    weights['decoder_ffn_down_w'][i],
                    buffers['decoder_x'], buffers['decoder_x'],
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

    orig_dec = pi05_infer.transformer_decoder
    orig_mod = pi05_infer.pi05_model
    pi05_infer.transformer_decoder = decoder_tuned

    def patched_model(w, b, nv, esl, ns_=10):
        pi05_infer.vision_encoder(w, b, nv)
        pi05_infer.transformer_encoder(w, b, esl)
        decoder_tuned(w, b, esl, ns_)
    pi05_infer.pi05_model = patched_model

    try:
        infer = pi05_infer.Pi05Inference(ckpt, num_views=3, chunk_size=50, discrete_state_input=False)
        img = torch.randn(3, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
        noise = torch.randn(50, 32, dtype=torch.bfloat16, device="cuda")
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
        pi05_infer.pi05_model = orig_mod
        pi0_infer.rms_matmul_n_2048_16384_gate = orig_enc_ffn
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()

    with open(args.pkl, "rb") as f: ckpt = pickle.load(f)
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("Step 10: Encoder FFN gate+up (2048→16384) BLOCK_SIZE sweep\n")

    results = []
    baseline_cfg = (128, 64, 32)
    for idx, cfg in enumerate(ENCODER_GATE_CANDIDATES):
        eff = cfg if cfg else baseline_cfg
        label = " V1 default" if cfg is None else ""
        print(f"[{idx+1}/{len(ENCODER_GATE_CANDIDATES)}] enc_gate={eff}{label} ...", flush=True)
        try:
            arr = bench(ckpt, args.n_warmup, args.n_test, eff)
            r = {"cfg": eff, "mean": float(arr.mean()), "p50": float(np.percentile(arr, 50))}
            results.append(r)
            print(f"      mean={r['mean']:.2f}ms p50={r['p50']:.2f}ms")
        except Exception as e:
            print(f"      FAILED: {type(e).__name__}: {str(e)[:120]}")
            results.append({"cfg": eff, "mean": float("inf")})

    print("\n=== RANKING ===")
    valid = [r for r in results if r["mean"] < float("inf")]
    valid.sort(key=lambda r: r["mean"])
    baseline_mean = next((r["mean"] for r in results if r["cfg"] == baseline_cfg), valid[0]["mean"])
    for rank, r in enumerate(valid, 1):
        sp = baseline_mean / r["mean"]
        mark = " ← V1 default" if r["cfg"] == baseline_cfg else ""
        print(f"{rank:<3} {str(r['cfg']):<18} mean={r['mean']:.2f}ms  {sp:.3f}x{mark}")
    best = valid[0]
    if best["cfg"] != baseline_cfg:
        improve = (baseline_mean - best["mean"]) / baseline_mean * 100
        print(f"\nBEST encoder gate: {best['cfg']} → {best['mean']:.2f} ms ({improve:+.2f}% vs V1 default)")


if __name__ == "__main__":
    main()
