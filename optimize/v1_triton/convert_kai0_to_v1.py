"""
Convert deepdive_kai0 JAX orbax ckpt → V1 pi05_infer weight dict (pickle).

Adapter for V1's convert_from_jax_pi05.py:
- deepdive_kai0 uses sentencepiece tokenizer (not HF AutoTokenizer)
- ckpt path: <ckpt_dir>/params/ (standard orbax format)
- output pickle is loadable directly by Pi05Inference(..., discrete_state_input=False)

Usage:
    python convert_kai0_to_v1.py \
        --jax_path /data1/DATA_IMP/checkpoints/task_a_mix_b6000_p1200_mixed_1_step49999 \
        --output /data1/DATA_IMP/checkpoints/task_a_mix_b6000_p1200_mixed_1_step49999_v1.pkl \
        --prompt "Flatten and fold the cloth" \
        --tokenizer_model /data1/tim/workspace/deepdive_kai0/openpi_cache/big_vision/paligemma_tokenizer.model
"""
import argparse
import os
import pickle
import sys

import numpy as np
import sentencepiece
import torch
import torch.nn as nn

# Import V1 conversion logic (same dir)
sys.path.insert(0, os.path.dirname(__file__))
from convert_from_jax_pi05 import convert_weights_pi05, load_jax_weights, prepare_adarms_cond


def prepare_prompt_sentencepiece(prompt: str, embedding_weight, tokenizer_model_path: str, max_len: int = 48):
    """Generate language_embeds from sentencepiece tokenizer (deepdive_kai0 path).

    Mirrors prepare_prompt() in V1 convert_from_jax_pi05.py but using
    sentencepiece directly (.model file) instead of HF AutoTokenizer.
    """
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_model_path)
    cleaned = prompt.strip().replace("_", " ") + "\n"
    token_ids = tokenizer.encode(cleaned, add_bos=True)
    # Truncate/pad to max_len
    token_ids = token_ids[:max_len]
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device="cuda")

    # Embed via nn.Embedding using embedding_weight
    embedding_weight_torch = torch.tensor(embedding_weight, dtype=torch.bfloat16, device="cuda")
    num_embeddings, embedding_dim = embedding_weight_torch.shape
    language_embedding = nn.Embedding(
        num_embeddings=num_embeddings, embedding_dim=embedding_dim
    ).bfloat16().cuda()
    with torch.no_grad():
        language_embedding.weight.copy_(embedding_weight_torch)
    language_embeds = language_embedding(token_ids_tensor)
    language_embeds = language_embeds * (language_embeds.shape[-1] ** 0.5)
    return language_embeds.to(device="cpu"), int(language_embeds.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax_path", required=True, help="deepdive_kai0 ckpt dir (contains params/)")
    parser.add_argument("--output", required=True, help="output .pkl path")
    parser.add_argument("--prompt", required=True, help="task prompt string")
    parser.add_argument("--tokenizer_model", required=True,
                        help="sentencepiece .model path (paligemma_tokenizer.model)")
    args = parser.parse_args()

    print(f"Loading JAX weights from {args.jax_path}/params ...")
    dump_weights = load_jax_weights(args.jax_path)

    print(f"Preparing prompt embeds with sentencepiece tokenizer ...")
    embedding_weight = dump_weights["PaliGemma"]["llm"]["embedder"]["input_embedding"]["value"]
    language_embeds, prompt_len = prepare_prompt_sentencepiece(
        args.prompt, embedding_weight, args.tokenizer_model
    )
    print(f"  prompt='{args.prompt}', tokens={prompt_len}")

    embedding_weight_torch = torch.tensor(embedding_weight, dtype=torch.bfloat16, device="cuda")

    print(f"Preparing adaRMS time embeds (10 flow steps) ...")
    time_embeds = prepare_adarms_cond(num_steps=10)

    print("Initializing V1 weight dict ...")
    weights = {
        "embedding_weight":                  torch.zeros(257152, 2048, dtype=torch.bfloat16, device="cpu"),

        "vision_patch_embedding_w":          torch.zeros(14, 14, 3, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_patch_embedding_b":          torch.zeros(1152, dtype=torch.bfloat16, device="cpu"),
        "vision_position_embedding":         torch.zeros(256, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_w":                 torch.zeros(27, 1152, 3 * 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_b":                 torch.zeros(27, 3 * 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_w":                   torch.zeros(27, 1152, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_b":                   torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_w":                   torch.zeros(27, 1152, 4304, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_b":                   torch.zeros(27, 4304, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_w":                 torch.zeros(27, 4304, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_b":                 torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_w":            torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_b":            torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_w":             torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_b":             torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_w":               torch.zeros(1152, dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_b":               torch.zeros(1152, dtype=torch.bfloat16, device="cpu"),

        "encoder_multi_modal_projector_w":   torch.zeros(1152, 2048, dtype=torch.bfloat16, device="cpu"),
        "encoder_multi_modal_projector_b":   torch.zeros(2048, dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_qkv_w":                torch.zeros(18, 2048, 2560, dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_o_w":                  torch.zeros(18, 2048, 2048, dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_gate_w":                torch.zeros(18, 2048, 16384, dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_up_w":                  torch.zeros(18, 2048, 16384, dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_down_w":                torch.zeros(18, 16384, 2048, dtype=torch.bfloat16, device="cpu"),

        "decoder_time_embeds":               torch.zeros(10, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_in_w":             torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_in_b":             torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_out_w":            torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_out_b":            torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_attn_norm_mod_w":       torch.zeros(18, 1024, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_attn_norm_mod_b":       torch.zeros(18, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_ffn_norm_mod_w":        torch.zeros(18, 1024, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_ffn_norm_mod_b":        torch.zeros(18, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_final_norm_mod_w":          torch.zeros(1024, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_final_norm_mod_b":          torch.zeros(3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_attn_qkv_w":                torch.zeros(18, 1024, 2560, dtype=torch.bfloat16, device="cpu"),
        "decoder_attn_o_w":                  torch.zeros(18, 2048, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_gate_w":                torch.zeros(18, 1024, 4096, dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_up_w":                  torch.zeros(18, 1024, 4096, dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_down_w":                torch.zeros(18, 4096, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_w":          torch.zeros(32, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_b":          torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_out_proj_w":         torch.zeros(1024, 32, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_out_proj_b":         torch.zeros(32, dtype=torch.bfloat16, device="cpu"),
        "language_embeds":                   torch.zeros(prompt_len, 2048, dtype=torch.bfloat16, device="cpu"),
    }

    print("Converting weights via V1 convert_weights_pi05 ...")
    convert_weights_pi05(weights, dump_weights)
    weights["embedding_weight"].copy_(embedding_weight_torch)
    weights["language_embeds"].copy_(language_embeds)
    weights["decoder_time_embeds"].copy_(time_embeds.cpu())

    print(f"Saving to {args.output} ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(weights, f)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"OK: {args.output} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
