"""Pure inference-speed benchmark for GigaWorld-Policy's core transformer.

Measures the per-rollout latency of the CasualWorldActionTransformer denoising
loop with the real Wan2.2-TI2V-5B architecture config but RANDOM weights
(no trained checkpoint / VAE / T5 required). This isolates the dominant
compute cost of the policy: the diffusion-transformer forward passes.

Shapes mirror the inference_server.py defaults (the canonical serving config):
  dst 768x192, num_frames=5, action_chunk=48, action_dim=14, 10 denoise steps,
  guidance_scale=0.0 (no CFG -> 1 forward / step).

VAE encode/decode and T5 are intentionally excluded: action-only serving uses
random ref latents here, so this is a lower bound that reflects the transformer
loop (which is where the "9x faster than Motus" action-decoding claim lives).
"""
import argparse
import json
import time

import torch

from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer


def build_model(cfg_path, dtype, device):
    cfg = json.load(open(cfg_path))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    cfg["patch_size"] = tuple(cfg["patch_size"])
    model = CasualWorldActionTransformer(**cfg)
    model = model.to(device=device, dtype=dtype).eval()
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="/tmp/wan_cfg/transformer/config.json")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--action_chunk", type=int, default=48)
    p.add_argument("--action_dim", type=int, default=14)
    p.add_argument("--state_dim", type=int, default=14)
    p.add_argument("--t5_len", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--vae_spatial", type=int, default=16)
    p.add_argument("--vae_temporal", type=int, default=4)
    p.add_argument("--fuse", action="store_true", help="fuse QKV projections")
    p.add_argument("--compile", action="store_true", help="torch.compile reduce-overhead (CUDA graphs)")
    p.add_argument("--compile_mode", default="reduce-overhead")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg = build_model(args.cfg, dtype, device)
    if args.fuse:
        for m in model.modules():
            if hasattr(m, "fuse_projections") and hasattr(m, "set_processor"):
                try:
                    m.fuse_projections()
                except Exception:
                    pass
        print("[opt] fused QKV projections")
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
        print(f"[opt] torch.compile(mode={args.compile_mode})")
    base_model = getattr(model, "_orig_mod", model)
    z_dim = cfg["in_channels"]
    p_t, p_h, p_w = cfg["patch_size"]
    text_dim = cfg["text_dim"]

    # latent geometry
    lat_h = args.height // args.vae_spatial
    lat_w = args.width // args.vae_spatial
    num_lat_frames = (args.num_frames - 1) // args.vae_temporal + 1  # = 2

    ref = torch.randn(1, z_dim, 1, lat_h, lat_w, device=device, dtype=dtype)
    noisy = torch.randn(1, z_dim, num_lat_frames - 1, lat_h, lat_w, device=device, dtype=dtype)
    state = torch.randn(1, 1, args.state_dim, device=device, dtype=dtype)
    action = torch.randn(1, args.action_chunk, args.action_dim, device=device, dtype=dtype)
    prompt_embeds = torch.randn(1, args.t5_len, text_dim, device=device, dtype=dtype)

    # timestep_full exactly as the pipeline builds it
    num_state = state.shape[1]
    num_action = action.shape[1]
    frame_per_tokens = (lat_h // p_h) * (lat_w // p_w)
    num_lat_tokens = frame_per_tokens * num_lat_frames
    num_clean = frame_per_tokens
    total_tokens_full = num_state + num_action + num_lat_tokens

    def make_timestep(t_val):
        ts = torch.zeros(1, total_tokens_full, device=device, dtype=dtype)
        ts[:, num_state + num_clean:] = t_val
        return ts

    # fake descending timesteps (values don't affect compute cost)
    t_values = torch.linspace(1000, 0, args.num_inference_steps, device=device)

    nheads = cfg["num_attention_heads"]
    head_dim = cfg["attention_head_dim"]
    print(f"[model] CasualWorldActionTransformer  layers={cfg['num_layers']} heads={nheads} "
          f"head_dim={head_dim} inner={nheads*head_dim} ffn={cfg['ffn_dim']} in_ch={z_dim}")
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparams/1e9:.3f}B  dtype={args.dtype}  device={torch.cuda.get_device_name(0)}")
    print(f"[shape] latent {z_dim}x{num_lat_frames}x{lat_h}x{lat_w}  "
          f"tokens: state={num_state} ref={frame_per_tokens} action={num_action} "
          f"noisy={frame_per_tokens*(num_lat_frames-1)}  "
          f"L_full={num_state+frame_per_tokens+num_action+frame_per_tokens*(num_lat_frames-1)} "
          f"L_action_only={num_state+frame_per_tokens+num_action}")
    print(f"[cfg]   steps={args.num_inference_steps} action_chunk={args.action_chunk} "
          f"h={args.height} w={args.width} frames={args.num_frames} t5_len={args.t5_len}\n")

    @torch.no_grad()
    def one_rollout(action_only):
        act = action.clone()
        lat = noisy.clone()
        for t_val in t_values:
            ts = make_timestep(t_val)
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            out = model(
                ref_latents=ref,
                noisy_latents=lat,
                timestep=ts,
                encoder_hidden_states=prompt_embeds,
                action=act,
                state=state,
                action_only=action_only,
                return_dict=False,
            )
            if action_only:
                act = out.clone()  # action_pred fed back into next step
            else:
                _, act = out
                act = act.clone()

    def bench(action_only, label):
        for _ in range(args.warmup):
            one_rollout(action_only)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.rollouts):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_rollout(action_only)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times = torch.tensor(times)
        mean = times.mean().item()
        std = times.std().item()
        peak = torch.cuda.max_memory_allocated() / 1e9
        per_step = mean / args.num_inference_steps
        print(f"=== {label} ===")
        print(f"  per-rollout ({args.num_inference_steps} steps): "
              f"{mean*1e3:7.2f} ms  (+/- {std*1e3:.2f})   min {times.min()*1e3:.2f} ms")
        print(f"  per-step                       : {per_step*1e3:7.2f} ms")
        print(f"  rollout throughput             : {1.0/mean:7.2f} rollouts/s  "
              f"({args.action_chunk/mean:7.1f} actions/s)")
        print(f"  peak GPU mem                   : {peak:7.2f} GB\n")

    bench(True, "ACTION-ONLY (serving fast path)")
    bench(False, "FULL (action + video latents)")


if __name__ == "__main__":
    main()
