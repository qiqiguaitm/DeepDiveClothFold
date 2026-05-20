"""Step 8: matmul_rope_qkv (QKV+RoPE, decoder 180×) BLOCK_SIZE sweep."""
import argparse, os, pickle, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 6 best (kept fixed)
BEST_GATE = (32, 64, 128)
BEST_FFN = (16, 32, 512)
BEST_ATTNO = (16, 32, 256)

# head_dim=256 must be divisible by BLOCK_SIZE_N
QKV_CANDIDATES = [
    None,             # V1 default (64, 32, 64)
    (32, 32, 64),     # smaller M
    (32, 32, 128),    # smaller M, bigger K
    (16, 32, 64),     # tiny M
    (16, 32, 128),    # tiny M, bigger K
    (64, 32, 128),    # default M, bigger K
    (32, 64, 64),     # bigger N (256/64=4 grids)
    (64, 64, 64),
    (16, 32, 256),    # tiny M, huge K
    (8,  32, 128),    # micro M
]


def make_decoder(qkv_cfg):
    import pi05_infer
    from pi05_infer import (
        matmul_k_32_1024_bias, adarms_norm_style_proj, matmul_abT_scale,
        softmax_kernel_prefix_suffix, matmul_k8_n_256, matmul_small_res_gate,
        adarms_matmul_k_1024_32_bias_res,
    )
    from pi0_infer import matmul_small_gate

    qM, qN, qK = qkv_cfg if qkv_cfg else (64, 32, 64)
    gN, gM, gK = BEST_GATE
    fN, fM, fK = BEST_FFN
    aN, aM, aK = BEST_ATTNO

    # Replicate matmul_k_1024_2560_qkv_rope with custom BLOCK
    from pi05_infer import matmul_rope_qkv

    def qkv_rope_tuned(x_normed, weight_qkv, rope_weight, Q, K, V):
        seq_len = x_normed.shape[0]
        matmul_rope_qkv[(128,)](
            x_normed, seq_len, 1024, 256, 8,
            weight_qkv, rope_weight, Q, K, V,
            BLOCK_SIZE_M=qM, BLOCK_SIZE_N=qN, BLOCK_SIZE_K=qK,
        )

    def transformer_decoder_step8(weights, buffers, encoder_seq_len, num_steps=10):
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
                # ▼ TUNED Step 8: QKV+RoPE
                qkv_rope_tuned(
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
                # Step 6 Attn O
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
                # Step 6 FFN gate
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
                # Step 6 FFN down
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

    return transformer_decoder_step8


def bench(ckpt, num_views, chunk_size, n_warmup, n_test, qkv_cfg):
    import pi05_infer
    new_dec = make_decoder(qkv_cfg)
    orig_dec = pi05_infer.transformer_decoder
    orig_model = pi05_infer.pi05_model
    pi05_infer.transformer_decoder = new_dec

    def patched(w, b, nv, esl, ns_=10):
        pi05_infer.vision_encoder(w, b, nv)
        pi05_infer.transformer_encoder(w, b, esl)
        new_dec(w, b, esl, ns_)
    pi05_infer.pi05_model = patched

    try:
        infer = pi05_infer.Pi05Inference(
            ckpt, num_views=num_views, chunk_size=chunk_size, discrete_state_input=False)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()

    with open(args.pkl, "rb") as f: ckpt = pickle.load(f)
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("Step 8: matmul_rope_qkv BLOCK_SIZE sweep (decoder 180× QKV+RoPE)\n")

    results = []
    for idx, cfg in enumerate(QKV_CANDIDATES):
        eff = cfg if cfg else (64, 32, 64)
        label = " V1 default" if cfg is None else ""
        print(f"[{idx+1}/{len(QKV_CANDIDATES)}] BLOCK_SIZE={eff}{label} ...", flush=True)
        try:
            arr = bench(ckpt, 3, 50, args.n_warmup, args.n_test, eff)
            r = {"cfg": eff, "mean": float(arr.mean()), "p50": float(np.percentile(arr, 50))}
            results.append(r)
            print(f"      mean={r['mean']:.2f}ms p50={r['p50']:.2f}ms")
        except Exception as e:
            print(f"      FAILED: {type(e).__name__}: {str(e)[:100]}")
            results.append({"cfg": eff, "mean": float("inf")})

    print("\n=== RANKING ===")
    valid = [r for r in results if r["mean"] < float("inf")]
    valid.sort(key=lambda r: r["mean"])
    baseline = next(r["mean"] for r in results if r["cfg"] == (64, 32, 64))
    for rank, r in enumerate(valid, 1):
        speed = baseline / r["mean"]
        mark = " ← V1 default" if r["cfg"] == (64, 32, 64) else ""
        print(f"{rank:<3} {str(r['cfg']):<18} mean={r['mean']:.2f}ms  {speed:.3f}x{mark}")
    best = valid[0]
    if best["cfg"] != (64, 32, 64):
        improve = (baseline - best["mean"]) / baseline * 100
        print(f"\nBEST: {best['cfg']} → {best['mean']:.2f} ms ({improve:+.2f}% vs V1 default)")


if __name__ == "__main__":
    main()
