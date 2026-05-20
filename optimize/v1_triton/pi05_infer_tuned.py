"""
pi0.5 Triton inference — 5090-tuned production version.

Differs from upstream pi05_infer.py in BLOCK_SIZE config of 5 hot decoder kernels:

| Kernel                      | V1 default (4090)      | 5090 tuned       | Δ      |
|-----------------------------|------------------------|------------------|--------|
| matmul_small_gate (FFN g+u) | (128, 64, 32)          | (32, 64, 128)    | -7.8%  |
| matmul_small_res_gate FFN d | (16, 32, 256)          | (16, 32, 512)    | -0.1%  |
| matmul_small_res_gate AttnO | (32, 32, 128)          | (16, 32, 256)    | -0.9%  |
| matmul_rope_qkv (QKV+RoPE)  | (64, 32, 64)           | (64, 32, 128)    | -0.4%  |
| matmul_abT_scale (Attn QK)  | (32, 32, 64)           | (32, 64, 64)     | -0.4%  |

Combined: 35.4 ms → 32.05 ms (-9.5%).

Why "small N, big K" wins on 5090:
  - 5090 SM count = 170 (vs 4090 128); small BLOCK_SIZE_N → more grids → fill SMs
  - 5090 L2 = 96MB (vs 4090 64MB); big BLOCK_SIZE_K exploits L2 reuse
  - Large BLOCK (256+) hits shared memory limit (101KB/SM on sm_120)

Usage:
  from pi05_infer_tuned import Pi05InferenceTuned
  infer = Pi05InferenceTuned(checkpoint, num_views=3, chunk_size=50, ...)
  out = infer.forward(image, noise)
"""
import math
import sys
import os

# Inherit all symbols from upstream
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from pi05_infer import *  # noqa: F401, F403
from pi05_infer import (
    Pi05Inference, matmul_k_32_1024_bias, adarms_norm_style_proj,
    matmul_abT_scale, softmax_kernel_prefix_suffix,
    matmul_k8_n_256, adarms_matmul_k_1024_32_bias_res,
    matmul_small_res_gate, vision_encoder, transformer_encoder,
    matmul_rope_qkv,
)
from pi0_infer import matmul_small_gate


# 5090 tune configs (from optimize/v1_triton/tune_5090_*.py)
TUNED_GATE_FFN = (32, 64, 128)     # Step 6: matmul_small_gate (FFN gate+up 1024→4096)
TUNED_FFN_DOWN = (16, 32, 512)     # Step 6: matmul_small_res_gate (FFN down 4096→1024)
TUNED_ATTN_O   = (16, 32, 256)     # Step 6: matmul_small_res_gate (Attn O  2048→1024)
TUNED_QKV_RoPE = (64, 32, 128)     # Step 8: matmul_rope_qkv (QKV+RoPE 1024→2560)
TUNED_ATTN_QK  = (32, 64, 64)      # Step 9: matmul_abT_scale (Attn QK matmul)


def matmul_k_2048_1024_gate_tuned(x, weight, out, gate):
    """Attn O (decoder): 2048 → 1024, 5090-tuned BLOCK_SIZE."""
    seq_len = x.shape[0]
    aN, aM, aK = TUNED_ATTN_O
    grid_n = (seq_len + aN - 1) // aN
    grid_m = (1024 + aM - 1) // aM
    matmul_small_res_gate[(grid_n * grid_m,)](
        x, weight, out, out, gate,
        seq_len=seq_len, features=2048, hidden=1024,
        BLOCK_SIZE_N=aN, BLOCK_SIZE_M=aM, BLOCK_SIZE_K=aK,
    )


def matmul_k_4096_1024_gate_tuned(x, weight, out, gate):
    """FFN down (decoder): 4096 → 1024, 5090-tuned BLOCK_SIZE."""
    seq_len = x.shape[0]
    fN, fM, fK = TUNED_FFN_DOWN
    grid_n = (seq_len + fN - 1) // fN
    grid_m = (1024 + fM - 1) // fM
    matmul_small_res_gate[(grid_n * grid_m,)](
        x, weight, out, out, gate,
        seq_len=seq_len, features=4096, hidden=1024,
        BLOCK_SIZE_N=fN, BLOCK_SIZE_M=fM, BLOCK_SIZE_K=fK,
    )


def transformer_decoder_tuned(weights, buffers, encoder_seq_len, num_steps=10):
    """Decoder with 5090-tuned BLOCK_SIZE on 5 hot kernels (gate+up / FFN-down / Attn-O / QKV+RoPE / Attn-QK)."""
    gN, gM, gK = TUNED_GATE_FFN
    qM, qN, qK = TUNED_QKV_RoPE
    sM, sN, sK = TUNED_ATTN_QK
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
            # ▼ TUNED Step 8: QKV+RoPE BLOCK_SIZE
            qkv_seq = buffers['x_normed_buf'].shape[0]
            matmul_rope_qkv[(128,)](
                buffers['x_normed_buf'], qkv_seq, 1024, 256, 8,
                weights['decoder_attn_qkv_w'][i],
                buffers['decoder_rope_weights'], buffers['decoder_q_buf'],
                buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len + seq_len],
                buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len + seq_len],
                BLOCK_SIZE_M=qM, BLOCK_SIZE_N=qN, BLOCK_SIZE_K=qK,
            )
            total_queries = buffers['decoder_q_buf'].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            # ▼ TUNED Step 9: Attention QK matmul BLOCK_SIZE
            matmul_abT_scale[(((total_queries + sM - 1) // sM) * ((total_keys + sN - 1) // sN),)](
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, :encoder_seq_len + seq_len],
                buffers['decoder_logits_buf'],
                total_queries, total_keys, 256, 256 ** -0.5,
                BLOCK_SIZE_M=sM, BLOCK_SIZE_N=sN, BLOCK_SIZE_K=sK,
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
            # ▼ TUNED: Attn O (5090)
            matmul_k_2048_1024_gate_tuned(
                buffers['decoder_q_buf'].view(-1, 2048),
                weights['decoder_attn_o_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf']
            )
            adarms_norm_style_proj(
                buffers['decoder_x'], buffers['decoder_time_emb'][step],
                weights['decoder_pre_ffn_norm_mod_w'][i],
                weights['decoder_pre_ffn_norm_mod_b'][i],
                buffers['x_normed_buf'], buffers['gate_buf'],
                buffers['decoder_style_ffn'][step, i]
            )
            seq_len = buffers['decoder_x'].shape[0]
            # ▼ TUNED: FFN gate+up (5090)
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
            # ▼ TUNED: FFN down (5090)
            matmul_k_4096_1024_gate_tuned(
                buffers['decoder_hidden'],
                weights['decoder_ffn_down_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf']
            )

        adarms_matmul_k_1024_32_bias_res(
            buffers['decoder_x'], buffers['decoder_time_emb'][step],
            weights['decoder_final_norm_mod_w'], weights['decoder_final_norm_mod_b'],
            buffers['x_normed_buf'], buffers['gate_buf'],
            buffers['decoder_style_final'][step],
            weights['decoder_action_out_proj_w'], weights['decoder_action_out_proj_b'],
            buffers['diffusion_noise'], buffers['diffusion_noise'],
        )


def pi05_model_tuned(weights, buffers, num_views, encoder_seq_len, num_steps=10):
    vision_encoder(weights, buffers, num_views)
    transformer_encoder(weights, buffers, encoder_seq_len)
    transformer_decoder_tuned(weights, buffers, encoder_seq_len, num_steps)


class Pi05InferenceTuned(Pi05Inference):
    """5090-tuned variant of Pi05Inference.

    Inherits __init__ + forward from Pi05Inference but record_run uses
    the tuned pi05_model that calls transformer_decoder_tuned.
    """

    def record_run(self):
        pi05_model_tuned(self.weights, self.buffers, self.num_views, self.encoder_seq_len)
