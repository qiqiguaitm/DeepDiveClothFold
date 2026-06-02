"""BAC (Block-wise Adaptive Caching) on top of prefix-cache (+ optional FP8) + compile.

Mechanism: per-block residual delta cache. Step 0 refreshes all blocks; later steps
reuse cached deltas for the redundant middle blocks (static schedule -> CUDA-graph safe).
We sweep the number of skipped middle blocks S and report latency.

NOTE on accuracy: which/how-many blocks are redundant is a property of the TRAINED
weights. With random weights the *speed* dividend of skipping is faithful, but the
*quality* (safe skip set / threshold) must be calibrated on the real checkpoint.
"""
import argparse
import json
import time

import torch

from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
from scripts.prefix_cache import PrefixCachedRunner


def build(cfg_path, dtype, device):
    cfg = json.load(open(cfg_path))
    cfg.pop("_class_name", None); cfg.pop("_diffusers_version", None)
    cfg["patch_size"] = tuple(cfg["patch_size"])
    return CasualWorldActionTransformer(**cfg).to(device=device, dtype=dtype).eval(), cfg


def middle_skip_mask(nb, s):
    mask = [True] * nb
    if s > 0:
        start = (nb - s) // 2
        for i in range(start, start + s):
            mask[i] = False
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="/tmp/wan_cfg/transformer/config.json")
    p.add_argument("--height", type=int, default=192); p.add_argument("--width", type=int, default=768)
    p.add_argument("--num_frames", type=int, default=5); p.add_argument("--action_chunk", type=int, default=48)
    p.add_argument("--action_dim", type=int, default=14); p.add_argument("--state_dim", type=int, default=14)
    p.add_argument("--t5_len", type=int, default=64); p.add_argument("--steps", type=int, default=10)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--vae_spatial", type=int, default=16); p.add_argument("--vae_temporal", type=int, default=4)
    p.add_argument("--skip_middle", type=int, default=12, help="# middle blocks to skip on steps>=1")
    p.add_argument("--fp8", action="store_true"); p.add_argument("--fuse", action="store_true")
    p.add_argument("--compile_mode", default="max-autotune")
    p.add_argument("--parity", action="store_true", help="measure action error vs bf16 full-compute reference")
    args = p.parse_args()

    dtype = torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(0)
    m, cfg = build(args.cfg, dtype, device)
    z = cfg["in_channels"]; p_t, p_h, p_w = cfg["patch_size"]; td = cfg["text_dim"]
    nb = cfg["num_layers"]

    lh, lw = args.height // args.vae_spatial, args.width // args.vae_spatial
    nlf = (args.num_frames - 1) // args.vae_temporal + 1
    ref = torch.randn(1, z, 1, lh, lw, device=device, dtype=dtype)
    noisy = torch.randn(1, z, nlf - 1, lh, lw, device=device, dtype=dtype)
    state = torch.randn(1, 1, args.state_dim, device=device, dtype=dtype)
    action = torch.randn(1, args.action_chunk, args.action_dim, device=device, dtype=dtype)
    enc = torch.randn(1, args.t5_len, td, device=device, dtype=dtype)
    noise_t = torch.tensor(500.0, device=device)

    if args.fuse:
        for mod in m.modules():
            if hasattr(mod, "fuse_projections") and hasattr(mod, "set_processor"):
                try: mod.fuse_projections()
                except Exception: pass

    runner = PrefixCachedRunner(m)
    runner.init_bac(nb)

    # ---- bf16 reference rollout (exact prefix-cache, all 30 blocks every step, eager) ----
    ref_action = None
    if args.parity:
        with torch.no_grad():
            runner.prepare(ref, noisy, enc, state); runner.set_action_rope(args.action_chunk)
            a = action.clone()
            for _ in range(args.steps):
                a = runner.step_refresh(a, noise_t)   # bf16, full compute
            ref_action = a.float().clone()

    if args.fp8:
        from scripts.fp8_linear import swap_linears_to_fp8
        n = swap_linears_to_fp8(m.blocks)
        print(f"[opt] native FP8 on {n} block linears")
    runner.compile_bac(cached_mode=args.compile_mode)
    runner.compile_prepare()

    skip_mask = middle_skip_mask(nb, args.skip_middle)
    n_comp = sum(skip_mask)
    print(f"[bac] blocks={nb}  skip_middle={args.skip_middle}  -> compute {n_comp}/{nb} blocks on steps>=1")

    def rollout():
        runner.prepare(ref, noisy, enc, state)
        runner.set_action_rope(args.action_chunk)
        act = action.clone()
        act = runner.step_refresh(act, noise_t)               # step 0: refresh all deltas
        for _ in range(args.steps - 1):
            act = runner.step_cached(act, noise_t, skip_mask)  # steps 1..N-1: skip middle
        return act

    if args.parity and ref_action is not None:
        with torch.no_grad():
            test_action = rollout().float()
        d = (test_action - ref_action).abs()
        rms_ref = ref_action.pow(2).mean().sqrt()
        print(f"[parity] full-stack (fp8={args.fp8}, skip_middle={args.skip_middle}) vs bf16 full-compute reference:")
        print(f"         max|abs|={d.max():.3e}  mean|abs|={d.mean():.3e}  "
              f"rms_err={d.pow(2).mean().sqrt():.3e}  ref_rms={rms_ref:.3e}  "
              f"rel_rms={(d.pow(2).mean().sqrt()/rms_ref):.3%}")

    for _ in range(6):
        rollout()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(args.rollouts):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rollout()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts = torch.tensor(ts)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[speed] fp8={args.fp8} skip_middle={args.skip_middle}: full rollout "
          f"{ts.mean()*1e3:7.2f} ms (+/- {ts.std()*1e3:.2f})  min {ts.min()*1e3:.2f}   peak {peak:.2f} GB")


if __name__ == "__main__":
    main()
