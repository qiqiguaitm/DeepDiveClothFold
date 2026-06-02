"""Fully-fused optimized action-only fast path.

Captures the ENTIRE per-chunk computation -- VAE encode + video KV prefill + all N
denoise steps -- into a single `torch.compile(mode="reduce-overhead")` CUDA graph.
The attention mask and flow-matching schedule are constant (depend only on shapes /
step count) and are precomputed once, so no Python orchestration runs between kernels.

Lossless: same math, same step count, no quantization, no retraining. Verified
against the eager output. The denoise loop is unrolled at the fixed step count, and
the video token length is a compile-time constant.

Run: python scripts/benchmark_infer_action_fused.py --gpu 2 --num-inference-steps 2 4 8 10
"""
import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--num-inference-steps", type=int, nargs="+", default=[2, 4, 8, 10])
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--replan-steps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    return ap.parse_args()


def move_cpu_buffers_to_gpu(model, device):
    def _to(x):
        return x.to(device) if hasattr(x, "to") else x
    model.action_expert.freqs = _to(model.action_expert.freqs)
    model.video_expert.freqs = tuple(_to(f) for f in model.video_expert.freqs)
    for obj in (model.vae, getattr(model.vae, "model", None)):
        if obj is None:
            continue
        if hasattr(obj, "mean"):
            obj.mean = _to(obj.mean)
        if hasattr(obj, "std"):
            obj.std = _to(obj.std)
        if hasattr(obj, "scale") and isinstance(obj.scale, (list, tuple)):
            obj.scale = [_to(s) for s in obj.scale]


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from hydra import compose, initialize_config_dir
    from fastwam.utils.config_resolvers import register_default_resolvers
    import benchmark_infer_action as B

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    move_cpu_buffers_to_gpu(model, device)
    print("[fix] moved VAE scale/mean/std and RoPE freqs to GPU")

    H, W = 384, 320
    image_np, state_np = B.load_kai0_frame_and_state(B.DEFAULT_KAI0_DATA)
    img = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=model.torch_dtype)
    img = img * (2.0 / 255.0) - 1.0
    proprio = torch.from_numpy(np.asarray(state_np[:model.proprio_dim], dtype=np.float32)).unsqueeze(0).to(
        device=device, dtype=model.torch_dtype)
    ctx = torch.randn(1, 128, model.text_dim, device=device, dtype=model.torch_dtype)
    ctx_mask = torch.ones(1, 128, dtype=torch.bool, device=device)
    ctx2, ctx_mask2 = model._append_proprio_to_context(context=ctx, context_mask=ctx_mask, proprio=proprio)
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    ah = args.action_horizon
    lat0 = torch.randn(1, ah, model.action_expert.action_dim, device=device, dtype=model.torch_dtype)

    # --- one eager dry run to discover constant shapes (video token length) + build mask ---
    with torch.no_grad():
        first0 = model._encode_input_image_latents_tensor(input_image=img, tiled=False)
        tv = torch.zeros((first0.shape[0],), dtype=first0.dtype, device=device)
        vpre0 = model.video_expert.pre_dit(x=first0, timestep=tv, context=ctx2, context_mask=ctx_mask2,
                                           action=None, fuse_vae_embedding_in_latents=fuse_flag)
    VLEN = int(vpre0["tokens"].shape[1])
    TPF = int(vpre0["meta"]["tokens_per_frame"])
    amask = model._build_mot_attention_mask(video_seq_len=VLEN, action_seq_len=ah,
                                            video_tokens_per_frame=TPF, device=device)
    amask_video = amask[:VLEN, :VLEN].contiguous()
    print(f"[shapes] video_seq_len={VLEN} tokens_per_frame={TPF} action_tokens={ah}")

    def eager_denoise(steps):
        with torch.no_grad():
            first = model._encode_input_image_latents_tensor(input_image=img, tiled=False)
            tvv = torch.zeros((first.shape[0],), dtype=first.dtype, device=device)
            vpre = model.video_expert.pre_dit(x=first, timestep=tvv, context=ctx2, context_mask=ctx_mask2,
                                              action=None, fuse_vae_embedding_in_latents=fuse_flag)
            kv = model.mot.prefill_video_cache(
                video_tokens=vpre["tokens"], video_freqs=vpre["freqs"], video_t_mod=vpre["t_mod"],
                video_context_payload={"context": vpre["context"], "mask": vpre["context_mask"]},
                video_attention_mask=amask_video)
            lat = lat0.clone()
            ts, deltas = model.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=steps, device=device, dtype=lat.dtype, shift_override=None)
            for i in range(steps):
                tsa = ts[i].unsqueeze(0)
                pred = model._predict_action_noise_with_cache(
                    latents_action=lat, timestep_action=tsa, context=ctx2, context_mask=ctx_mask2,
                    video_kv_cache=kv, attention_mask=amask, video_seq_len=VLEN)
                lat = lat + pred * deltas[i]
            return lat

    def make_fused(steps, ts, deltas):
        def fused(img_in, c, cm, lat_in):
            first = model._encode_input_image_latents_tensor(input_image=img_in, tiled=False)
            tvv = torch.zeros((first.shape[0],), dtype=first.dtype, device=device)
            vpre = model.video_expert.pre_dit(x=first, timestep=tvv, context=c, context_mask=cm,
                                              action=None, fuse_vae_embedding_in_latents=fuse_flag)
            kv = model.mot.prefill_video_cache(
                video_tokens=vpre["tokens"], video_freqs=vpre["freqs"], video_t_mod=vpre["t_mod"],
                video_context_payload={"context": vpre["context"], "mask": vpre["context_mask"]},
                video_attention_mask=amask_video)
            lat = lat_in
            for i in range(steps):  # unrolled at constant `steps`
                tsa = ts[i].unsqueeze(0)
                pred = model._predict_action_noise_with_cache(
                    latents_action=lat, timestep_action=tsa, context=c, context_mask=cm,
                    video_kv_cache=kv, attention_mask=amask, video_seq_len=VLEN)
                lat = lat + pred * deltas[i]
            return lat
        return torch.compile(fused, mode="reduce-overhead", dynamic=False)

    print(f"\n{'steps':>6}{'eager ms':>12}{'fused ms':>12}{'speedup':>10}{'ctrl Hz':>10}{'max|diff|':>12}{'':>10}")
    for steps in args.num_inference_steps:
        ts, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=device, dtype=lat0.dtype, shift_override=None)
        fused = make_fused(steps, ts, deltas)

        with torch.no_grad():
            ref = eager_denoise(steps).float()
            for _ in range(args.warmup):
                out = fused(img, ctx2, ctx_mask2, lat0.clone())
            torch.cuda.synchronize()
            opt = out.float()
            diff = (opt - ref).abs().max().item()

            # time eager
            le = []
            for _ in range(args.iters):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                eager_denoise(steps)
                torch.cuda.synchronize(); le.append((time.perf_counter() - t0) * 1000)
            # time fused
            lo = []
            for _ in range(args.iters):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                fused(img, ctx2, ctx_mask2, lat0.clone())
                torch.cuda.synchronize(); lo.append((time.perf_counter() - t0) * 1000)

        me, mo = statistics.mean(le), statistics.mean(lo)
        hz = args.replan_steps * 1000.0 / mo
        flag = "  <=50ms" if mo <= 50 else ""
        print(f"{steps:>6}{me:>12.1f}{mo:>12.1f}{me/mo:>9.2f}x{hz:>10.1f}{diff:>12.3e}{flag}")

    print(f"\n[mem] peak CUDA allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
