"""Profile the OPTIMIZED (cudagraphed) stages to find remaining headroom.

Times each compiled stage individually: VAE encode | prefill | per-step.
Also compares torch.compile modes: reduce-overhead vs max-autotune.
"""
import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--modes", nargs="+", default=["reduce-overhead", "max-autotune"])
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from hydra import compose, initialize_config_dir
    from fastwam.utils.config_resolvers import register_default_resolvers
    import benchmark_infer_action as B
    from benchmark_infer_action_fused import move_cpu_buffers_to_gpu

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    register_default_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[
            "task=robotwin_uncond_3cam_384_1e-4", "model.load_text_encoder=false",
            "model.skip_dit_load_from_pretrain=true", "model.action_dit_pretrained_path=null"])
    device = "cuda:0"
    print(f"[device] {torch.cuda.get_device_name(0)}")
    model = B.build_random_model(cfg, device=device, dtype=torch.bfloat16)
    model.eval()
    move_cpu_buffers_to_gpu(model, device)

    img = (torch.rand(1, 3, 384, 320, device=device, dtype=model.torch_dtype) * 2 - 1)
    ctx = torch.randn(1, 128, model.text_dim, device=device, dtype=model.torch_dtype)
    cmask = torch.ones(1, 128, dtype=torch.bool, device=device)
    proprio = torch.randn(1, model.proprio_dim, device=device, dtype=model.torch_dtype)
    ctx2, cmask2 = model._append_proprio_to_context(context=ctx, context_mask=cmask, proprio=proprio)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    ah = args.action_horizon

    with torch.no_grad():
        first0 = model._encode_input_image_latents_tensor(input_image=img, tiled=False)
        tv = torch.zeros((first0.shape[0],), dtype=first0.dtype, device=device)
        vpre0 = model.video_expert.pre_dit(x=first0, timestep=tv, context=ctx2, context_mask=cmask2,
                                           action=None, fuse_vae_embedding_in_latents=fuse)
    VLEN = int(vpre0["tokens"].shape[1]); TPF = int(vpre0["meta"]["tokens_per_frame"])
    amask = model._build_mot_attention_mask(video_seq_len=VLEN, action_seq_len=ah,
                                            video_tokens_per_frame=TPF, device=device)
    amask_v = amask[:VLEN, :VLEN].contiguous()
    lat = torch.randn(1, ah, model.action_expert.action_dim, device=device, dtype=model.torch_dtype)

    def timeit(fn, n):
        with torch.no_grad():
            for _ in range(8):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                fn(); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
        return statistics.mean(ts), min(ts)

    for mode in args.modes:
        print(f"\n===== mode = {mode} =====")
        enc = torch.compile(model._encode_input_image_latents_tensor, mode=mode)
        prefill = torch.compile(model.mot.prefill_video_cache, mode=mode)
        step = torch.compile(model._predict_action_noise_with_cache, mode=mode)

        def run_enc():
            return enc(input_image=img, tiled=False).clone()

        with torch.no_grad():
            for _ in range(3):
                first = run_enc()
        def run_prefill():
            return prefill(video_tokens=vpre0["tokens"], video_freqs=vpre0["freqs"], video_t_mod=vpre0["t_mod"],
                           video_context_payload={"context": vpre0["context"], "mask": vpre0["context_mask"]},
                           video_attention_mask=amask_v)
        with torch.no_grad():
            for _ in range(3):
                kv = [{k: v.clone() for k, v in d.items()} for d in run_prefill()]
        def run_step():
            return step(latents_action=lat, timestep_action=torch.zeros(1, device=device, dtype=lat.dtype),
                        context=ctx2, context_mask=cmask2, video_kv_cache=kv, attention_mask=amask, video_seq_len=VLEN)

        e_mean, e_min = timeit(run_enc, args.iters)
        p_mean, p_min = timeit(run_prefill, args.iters)
        s_mean, s_min = timeit(run_step, args.iters)
        print(f"  vae_encode : mean {e_mean:6.2f} ms  min {e_min:6.2f}")
        print(f"  prefill    : mean {p_mean:6.2f} ms  min {p_min:6.2f}")
        print(f"  per_step   : mean {s_mean:6.2f} ms  min {s_min:6.2f}")
        for N in (4, 10):
            print(f"  => est total {N:2d} steps: {e_mean + p_mean + N*s_mean:6.2f} ms")


if __name__ == "__main__":
    main()
