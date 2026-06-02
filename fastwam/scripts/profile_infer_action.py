"""Phase-by-phase profile of the action-only fast path, to locate the bottleneck.

Times: VAE encode | video pre_dit | prefill video KV cache | each denoising step
(action pre_dit + MoT action-with-cache + post_dit) | scheduler step.

Run: python scripts/profile_infer_action.py --gpu 2 --num-inference-steps 10
"""
import argparse
import os
import statistics
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from hydra import compose, initialize_config_dir
    from fastwam.utils.config_resolvers import register_default_resolvers
    import benchmark_infer_action as B  # reuse build_random_model

    register_default_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[
            "task=robotwin_uncond_3cam_384_1e-4",
            "model.load_text_encoder=false",
            "model.skip_dit_load_from_pretrain=true",
            "model.action_dit_pretrained_path=null",
        ])
    device = "cuda:0"
    model = B.build_random_model(cfg, device=device, dtype=torch.bfloat16)
    model.eval()

    H, W = 384, 320
    img = (torch.rand(1, 3, H, W, device=device, dtype=model.torch_dtype) * 2 - 1)
    ctx = torch.randn(1, 128, model.text_dim, device=device, dtype=model.torch_dtype)
    ctx_mask = torch.ones(1, 128, dtype=torch.bool, device=device)
    proprio = torch.randn(1, model.proprio_dim, device=device, dtype=model.torch_dtype)
    ctx2, ctx_mask2 = model._append_proprio_to_context(context=ctx, context_mask=ctx_mask, proprio=proprio)

    def sync(): torch.cuda.synchronize()

    class T:
        def __init__(s): s.d = {}
        def __call__(s, name): s.name = name; return s
        def __enter__(s): sync(); s.t0 = time.perf_counter()
        def __exit__(s, *a): sync(); s.d.setdefault(s.name, []).append((time.perf_counter() - s.t0) * 1000)

    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    def one(t):
        with t("1_vae_encode"):
            first = model._encode_input_image_latents_tensor(input_image=img, tiled=False)
        with t("2_video_pre_dit"):
            tv = torch.zeros((first.shape[0],), dtype=first.dtype, device=device)
            vpre = model.video_expert.pre_dit(x=first, timestep=tv, context=ctx2, context_mask=ctx_mask2,
                                              action=None, fuse_vae_embedding_in_latents=fuse_flag)
        vlen = int(vpre["tokens"].shape[1])
        amask = model._build_mot_attention_mask(video_seq_len=vlen, action_seq_len=args.action_horizon,
                                                video_tokens_per_frame=int(vpre["meta"]["tokens_per_frame"]),
                                                device=device)
        with t("3_prefill_cache"):
            kv = model.mot.prefill_video_cache(
                video_tokens=vpre["tokens"], video_freqs=vpre["freqs"], video_t_mod=vpre["t_mod"],
                video_context_payload={"context": vpre["context"], "mask": vpre["context_mask"]},
                video_attention_mask=amask[:vlen, :vlen])
        lat = torch.randn(1, args.action_horizon, model.action_expert.action_dim, device=device, dtype=model.torch_dtype)
        ts, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=args.num_inference_steps, device=device, dtype=lat.dtype, shift_override=None)
        for st, dl in zip(ts, deltas):
            tsa = st.unsqueeze(0).to(dtype=lat.dtype, device=device)
            with t("4_per_step"):
                pred = model._predict_action_noise_with_cache(
                    latents_action=lat, timestep_action=tsa, context=ctx2, context_mask=ctx_mask2,
                    video_kv_cache=kv, attention_mask=amask, video_seq_len=vlen)
            with t("5_sched_step"):
                lat = model.infer_action_scheduler.step(pred, dl, lat)
        return vlen

    t = T()
    for _ in range(3):
        vlen = one(t)  # warmup
    t.d.clear()
    for _ in range(args.iters):
        vlen = one(t)

    print(f"\n[shapes] video_seq_len={vlen}  action_tokens={args.action_horizon}  steps={args.num_inference_steps}")
    print(f"{'phase':<18}{'mean ms':>10}{'min':>8}{'calls/iter':>12}{'total/iter ms':>16}")
    order = ["1_vae_encode", "2_video_pre_dit", "3_prefill_cache", "4_per_step", "5_sched_step"]
    grand = 0.0
    for k in order:
        v = t.d.get(k, [])
        if not v:
            continue
        cpi = len(v) // args.iters
        per_iter = statistics.mean(v) * cpi
        grand += per_iter
        print(f"{k:<18}{statistics.mean(v):>10.2f}{min(v):>8.2f}{cpi:>12}{per_iter:>16.2f}")
    print(f"{'TOTAL (sum)':<18}{'':>10}{'':>8}{'':>12}{grand:>16.2f}")


if __name__ == "__main__":
    main()
