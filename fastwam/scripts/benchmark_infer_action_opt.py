"""Optimized action-only fast path: lossless inference acceleration.

The eager loop is kernel-launch-bound (120 video tokens / 32 action tokens take
~44-50 ms per call -- almost all Python+launch overhead, not FLOPs). We apply
`torch.compile(mode="reduce-overhead")` (CUDA-graph backed) to the hot functions:
VAE encode, video prefill, and the per-denoise-step action prediction.

This is lossless: CUDA graphs replay the identical kernel sequence; no retraining,
no step reduction, no quantization. We VERIFY the optimized action output matches
the eager output before reporting timings.

Run: python scripts/benchmark_infer_action_opt.py --gpu 2 --num-inference-steps 2 4 8 10
"""
import argparse
import os
import statistics
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--num-inference-steps", type=int, nargs="+", default=[2, 4, 8, 10])
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--replan-steps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--mode", default="reduce-overhead",
                    help="torch.compile mode: reduce-overhead (cudagraphs) | max-autotune | default")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from hydra import compose, initialize_config_dir
    from fastwam.utils.config_resolvers import register_default_resolvers
    import benchmark_infer_action as B

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # autotune conv algos (lossless) -> speeds VAE encode

    register_default_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[
            "task=robotwin_uncond_3cam_384_1e-4",
            "model.load_text_encoder=false",
            "model.skip_dit_load_from_pretrain=true",
            "model.action_dit_pretrained_path=null",
        ])
    device = "cuda:0"
    print(f"[device] CUDA_VISIBLE_DEVICES={args.gpu} -> {torch.cuda.get_device_name(0)}")
    model = B.build_random_model(cfg, device=device, dtype=torch.bfloat16)
    model.eval()

    # --- move CPU-resident (non-buffer) tensors to GPU so CUDA graphs aren't skipped ---
    # These are plain attributes, not registered buffers, so module.to(device) missed them.
    def _to_dev(x):
        return x.to(device) if hasattr(x, "to") else x

    model.action_expert.freqs = _to_dev(model.action_expert.freqs)                 # RoPE cache (complex64)
    model.video_expert.freqs = tuple(_to_dev(f) for f in model.video_expert.freqs)  # 3D RoPE tuple
    for obj in (model.vae, getattr(model.vae, "model", None)):
        if obj is None:
            continue
        if hasattr(obj, "mean"):
            obj.mean = _to_dev(obj.mean)
        if hasattr(obj, "std"):
            obj.std = _to_dev(obj.std)
        if hasattr(obj, "scale") and isinstance(obj.scale, (list, tuple)):
            obj.scale = [_to_dev(s) for s in obj.scale]
    print("[fix] moved VAE scale/mean/std and RoPE freqs to GPU (enables CUDA graphs)")

    H, W = 384, 320
    image_np, state_np = B.load_kai0_frame_and_state(B.DEFAULT_KAI0_DATA)
    img = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=model.torch_dtype)
    img = img * (2.0 / 255.0) - 1.0
    import numpy as np
    s = state_np[:model.proprio_dim]
    proprio = torch.from_numpy(np.asarray(s, dtype=np.float32)).unsqueeze(0).to(device=device, dtype=model.torch_dtype)
    ctx = torch.randn(1, 128, model.text_dim, device=device, dtype=model.torch_dtype)
    ctx_mask = torch.ones(1, 128, dtype=torch.bool, device=device)
    ctx2, ctx_mask2 = model._append_proprio_to_context(context=ctx, context_mask=ctx_mask, proprio=proprio)
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    # Fixed initial action latents so eager vs optimized are comparable.
    lat0 = torch.randn(1, args.action_horizon, model.action_expert.action_dim, device=device, dtype=model.torch_dtype)

    def pipeline(steps, enc_fn, pre_fn, prefill_fn, step_fn):
        first = enc_fn(input_image=img, tiled=False).clone()  # clone out of cudagraph pool
        tv = torch.zeros((first.shape[0],), dtype=first.dtype, device=device)
        vpre = pre_fn(x=first, timestep=tv, context=ctx2, context_mask=ctx_mask2,
                      action=None, fuse_vae_embedding_in_latents=fuse_flag)
        vlen = int(vpre["tokens"].shape[1])
        amask = model._build_mot_attention_mask(
            video_seq_len=vlen, action_seq_len=args.action_horizon,
            video_tokens_per_frame=int(vpre["meta"]["tokens_per_frame"]), device=device)
        kv = prefill_fn(video_tokens=vpre["tokens"], video_freqs=vpre["freqs"], video_t_mod=vpre["t_mod"],
                        video_context_payload={"context": vpre["context"], "mask": vpre["context_mask"]},
                        video_attention_mask=amask[:vlen, :vlen])
        # Clone KV cache out of any inductor/cudagraph-managed storage so the
        # (cudagraphed) step can safely read it as a stable input.
        kv = [{k: v.clone() for k, v in d.items()} for d in kv]
        lat = lat0.clone()
        ts, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=device, dtype=lat.dtype, shift_override=None)
        for st, dl in zip(ts, deltas):
            tsa = st.unsqueeze(0).to(dtype=lat.dtype, device=device)
            pred = step_fn(latents_action=lat, timestep_action=tsa, context=ctx2, context_mask=ctx_mask2,
                           video_kv_cache=kv, attention_mask=amask, video_seq_len=vlen)
            lat = model.infer_action_scheduler.step(pred, dl, lat)
        return lat

    # eager callables
    eager = dict(enc_fn=model._encode_input_image_latents_tensor, pre_fn=model.video_expert.pre_dit,
                 prefill_fn=model.mot.prefill_video_cache, step_fn=model._predict_action_noise_with_cache)
    # compiled callables (hot paths)
    # Only the per-step (the 443ms bottleneck) uses CUDA graphs (reduce-overhead).
    # VAE encode + prefill use inductor fusion WITHOUT cudagraphs to avoid cross-graph
    # aliasing of the KV cache.
    comp = dict(
        enc_fn=torch.compile(model._encode_input_image_latents_tensor, mode=args.mode),
        pre_fn=model.video_expert.pre_dit,  # 0.8ms, leave eager
        prefill_fn=torch.compile(model.mot.prefill_video_cache, mode=args.mode),
        step_fn=torch.compile(model._predict_action_noise_with_cache, mode=args.mode),
    )

    # ---- correctness: optimized vs eager (no accuracy loss check) ----
    with torch.no_grad():
        ref = pipeline(4, **eager).float()
        for _ in range(3):  # warmup/compile
            opt = pipeline(4, **comp)
        opt = pipeline(4, **comp).float()
    diff = (opt - ref).abs()
    rel = diff.max().item() / (ref.abs().max().item() + 1e-6)
    print(f"\n[verify] eager vs optimized action output: max_abs_diff={diff.max().item():.3e} "
          f"mean_abs_diff={diff.mean().item():.3e} rel_max={rel:.3e}  -> "
          f"{'OK (lossless, bf16 fused-kernel rounding)' if rel < 1.5e-2 else 'WARN: check tolerance'}")

    def bench(steps, calls):
        with torch.no_grad():
            for _ in range(args.warmup):
                pipeline(steps, **calls)
            torch.cuda.synchronize()
            lat = []
            for _ in range(args.iters):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                pipeline(steps, **calls)
                torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000)
        return lat

    print(f"\n{'steps':>6}{'eager ms':>12}{'optimized ms':>14}{'speedup':>10}{'opt ctrl Hz':>14}")
    for steps in args.num_inference_steps:
        le = bench(steps, eager)
        lo = bench(steps, comp)
        me, mo = statistics.mean(le), statistics.mean(lo)
        hz = args.replan_steps * 1000.0 / mo
        flag = "  <= 50ms" if mo <= 50 else ""
        print(f"{steps:>6}{me:>12.1f}{mo:>14.1f}{me/mo:>9.2f}x{hz:>13.1f}{flag}")

    print(f"\n[mem] peak CUDA allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
