"""Verify the prefix KV cache is numerically exact vs the baseline forward, and time it."""
import argparse
import json
import time

import torch

from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
from scripts.prefix_cache import PrefixCachedRunner


def build(cfg_path, dtype, device):
    cfg = json.load(open(cfg_path))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    cfg["patch_size"] = tuple(cfg["patch_size"])
    m = CasualWorldActionTransformer(**cfg).to(device=device, dtype=dtype).eval()
    return m, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="/tmp/wan_cfg/transformer/config.json")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--action_chunk", type=int, default=48)
    p.add_argument("--action_dim", type=int, default=14)
    p.add_argument("--state_dim", type=int, default=14)
    p.add_argument("--t5_len", type=int, default=64)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--vae_spatial", type=int, default=16)
    p.add_argument("--vae_temporal", type=int, default=4)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="max-autotune-no-cudagraphs")
    p.add_argument("--fuse", action="store_true")
    p.add_argument("--fp8", action="store_true", help="FP8 weight-only quant on transformer blocks")
    p.add_argument("--fp8_native", action="store_true", help="native FP8 _scaled_mm linears")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(0)
    m, cfg = build(args.cfg, dtype, device)
    z = cfg["in_channels"]
    p_t, p_h, p_w = cfg["patch_size"]
    td = cfg["text_dim"]

    lh, lw = args.height // args.vae_spatial, args.width // args.vae_spatial
    nlf = (args.num_frames - 1) // args.vae_temporal + 1
    ref = torch.randn(1, z, 1, lh, lw, device=device, dtype=dtype)
    noisy = torch.randn(1, z, nlf - 1, lh, lw, device=device, dtype=dtype)
    state = torch.randn(1, 1, args.state_dim, device=device, dtype=dtype)
    action = torch.randn(1, args.action_chunk, args.action_dim, device=device, dtype=dtype)
    enc = torch.randn(1, args.t5_len, td, device=device, dtype=dtype)

    num_state = 1
    num_action = args.action_chunk
    fpt = (lh // p_h) * (lw // p_w)
    num_lat = fpt * nlf
    num_clean = fpt
    total = num_state + num_action + num_lat
    noise_t = torch.tensor(500.0, device=device)

    def timestep_full(tv):
        ts = torch.zeros(1, total, device=device, dtype=dtype)
        ts[:, num_state + num_clean:] = tv
        return ts

    # ---- baseline single forward (action-only) ----
    with torch.no_grad():
        base = m(ref_latents=ref, noisy_latents=noisy, timestep=timestep_full(noise_t),
                 encoder_hidden_states=enc, action=action, state=state,
                 action_only=True, return_dict=False)

    if args.fuse:
        for mod in m.modules():
            if hasattr(mod, "fuse_projections") and hasattr(mod, "set_processor"):
                try:
                    mod.fuse_projections()
                except Exception:
                    pass

    if args.fp8:
        from torchao.quantization import quantize_, Float8WeightOnlyConfig
        quantize_(m.blocks, Float8WeightOnlyConfig())
        print("[opt] FP8 weight-only quant on transformer blocks (torchao)")
    if args.fp8_native:
        from scripts.fp8_linear import swap_linears_to_fp8
        n = swap_linears_to_fp8(m.blocks)
        print(f"[opt] native FP8 (_scaled_mm) on {n} block linears")

    # ---- prefix-cached single forward (parity, uncompiled) ----
    runner = PrefixCachedRunner(m)
    runner.prepare(ref, noisy, enc, state)
    runner.set_action_rope(args.action_chunk)
    cached = runner.step(action, noise_t)

    diff = (base.float() - cached.float()).abs()
    rel = diff / (base.float().abs() + 1e-6)
    print(f"[parity] baseline vs prefix-cache action_pred:")
    print(f"         max|abs diff| = {diff.max().item():.3e}   mean = {diff.mean().item():.3e}")
    print(f"         max|rel diff| = {rel.max().item():.3e}   (bf16 round-off ~1e-2 expected)")
    print(f"         shapes base={tuple(base.shape)} cached={tuple(cached.shape)}\n")

    if args.compile:
        runner.compile_step(mode=args.compile_mode)
        runner.compile_prepare()
        print(f"[opt] torch.compile step (mode={args.compile_mode}) + prepare fuse={args.fuse}")

    # ---- speed: full rollout = prepare once + N action steps ----
    def rollout():
        runner.prepare(ref, noisy, enc, state)
        runner.set_action_rope(args.action_chunk)
        act = action.clone()
        for _ in range(args.steps):
            act = runner.step(act, noise_t)
        return act

    for _ in range(5):
        rollout()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # full rollout timing
    ts = []
    for _ in range(args.rollouts):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rollout()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts = torch.tensor(ts)

    # breakdown: prepare-only vs single step (steady state)
    tp = []
    for _ in range(args.rollouts):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        runner.prepare(ref, noisy, enc, state); runner.set_action_rope(args.action_chunk)
        torch.cuda.synchronize(); tp.append(time.perf_counter() - t0)
    tp = torch.tensor(tp)
    act = action.clone()
    tstep = []
    for _ in range(args.rollouts * args.steps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        act = runner.step(act, noise_t)
        torch.cuda.synchronize(); tstep.append(time.perf_counter() - t0)
    tstep = torch.tensor(tstep)

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[speed] prefix-cache action-only, {args.steps} steps  (compile={args.compile} {args.compile_mode}):")
    print(f"        full rollout   {ts.mean()*1e3:7.2f} ms (+/- {ts.std()*1e3:.2f})  min {ts.min()*1e3:.2f}")
    print(f"        prepare (1x)   {tp.mean()*1e3:7.2f} ms")
    print(f"        per-step       {tstep.mean()*1e3:7.2f} ms  x{args.steps} = {tstep.mean()*1e3*args.steps:.1f} ms")
    print(f"        peak mem       {peak:7.2f} GB")


if __name__ == "__main__":
    main()
