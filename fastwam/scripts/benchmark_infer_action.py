"""Benchmark the FastWAM *action-only fast path* (`model.infer_action`).

This times the deployed per-chunk action inference latency (no test-time video
imagination), mirroring `experiments/robotwin/fastwam_policy/deploy_policy.py`.

Inference latency is weight-independent (it depends on tensor shapes, not values),
so a trained checkpoint is optional -- pass `--ckpt` if you have one, otherwise the
model is constructed from config with random/base weights and timing is identical.

The input frame + proprio state are pulled from the deepdive_kai0 robot data
(`../kai0/data/Task_A/...`, LeRobot format, 3 cams + 14-dim state) to honor
"use the data in deepdive_kai0"; values do not affect timing, only tensor shapes do.

Example (GPU2, default robotwin-profile config, 4 denoising steps):

    python scripts/benchmark_infer_action.py --gpu 2 --num-inference-steps 4

    # sweep denoising steps, with a checkpoint:
    python scripts/benchmark_infer_action.py --gpu 2 \
        --ckpt ./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
        --num-inference-steps 2 4 8 --action-horizon 32 --iters 30
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
# Default frame/state source: deepdive_kai0 robot data (3 cams + 14-dim state).
DEFAULT_KAI0_DATA = REPO_ROOT.parent / "kai0" / "data" / "Task_A" / "self_built" / "A_new_pure_200_val"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=str, default="2", help="Physical GPU index (sets CUDA_VISIBLE_DEVICES). Default: 2")
    p.add_argument("--task", type=str, default="robotwin_uncond_3cam_384_1e-4",
                   help="Hydra task config (decides model dims). kai0 data is 3-cam/14-dim -> robotwin profile.")
    p.add_argument("--ckpt", type=str, default=None, help="Optional trained checkpoint .pt (timing is weight-independent).")
    p.add_argument("--num-inference-steps", type=int, nargs="+", default=[4],
                   help="Denoising steps to sweep, e.g. --num-inference-steps 2 4 8")
    p.add_argument("--action-horizon", type=int, default=32, help="Predicted action chunk length.")
    p.add_argument("--replan-steps", type=int, default=4, help="Actions executed per chunk (for effective-control-rate report).")
    p.add_argument("--text-cfg-scale", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=30, help="Timed iterations per setting.")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations (excluded from timing).")
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_KAI0_DATA),
                   help="kai0 LeRobot dataset dir to pull one input frame + state from.")
    p.add_argument("--no-text-encoder", action="store_true",
                   help="Skip loading the T5 text encoder; feed dummy cached context (value-independent for timing).")
    p.add_argument("--download-base", action="store_true",
                   help="Construct via from_wan22_pretrained (downloads the Wan2.2 VAE/text encoder). "
                        "Default OFF: build a fully RANDOM-INITIALIZED model with ZERO downloads.")
    return p.parse_args()


def load_kai0_frame_and_state(data_dir: Path):
    """Return (image_uint8 [384,320,3], state [14]) built robotwin-style from kai0 data.

    Falls back to synthetic tensors (shapes preserved) if the data can't be read.
    """
    import numpy as np

    def _resize(img, size_wh):
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)

    def _synthetic():
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, size=(384, 320, 3), dtype=np.uint8), rng.standard_normal(14).astype(np.float32)

    try:
        import imageio.v3 as iio

        vid_root = data_dir / "videos" / "chunk-000"
        cams = {
            "head": "observation.images.top_head",
            "left": "observation.images.hand_left",
            "right": "observation.images.hand_right",
        }
        frames = {}
        for name, key in cams.items():
            cam_dir = vid_root / key
            mp4s = sorted(cam_dir.glob("episode_*.mp4"))
            if not mp4s:
                raise FileNotFoundError(f"No mp4 under {cam_dir}")
            frames[name] = iio.imread(mp4s[0], index=0)  # [H,W,3] uint8

        head = _resize(frames["head"], (320, 256))
        left = _resize(frames["left"], (160, 128))
        right = _resize(frames["right"], (160, 128))
        image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)  # [384,320,3]

        state = None
        try:
            import pandas as pd
            pq = sorted((data_dir / "data" / "chunk-000").glob("episode_*.parquet"))
            if pq:
                df = pd.read_parquet(pq[0], columns=["observation.state"])
                state = np.asarray(df["observation.state"].iloc[0], dtype=np.float32).reshape(-1)[:14]
        except Exception as e:
            print(f"[warn] could not read state parquet ({e}); using synthetic state.", file=sys.stderr)
        if state is None or state.shape[0] != 14:
            state = np.random.default_rng(0).standard_normal(14).astype(np.float32)
        print(f"[data] using real kai0 frame from {vid_root} (input image {image.shape}, state dim {state.shape[0]})")
        return image, state
    except Exception as e:
        print(f"[warn] could not load kai0 data ({e}); falling back to synthetic input.", file=sys.stderr)
        return _synthetic()


def build_random_model(cfg, device, dtype):
    """Construct FastWAM with fully RANDOM-INITIALIZED weights and ZERO downloads.

    Bypasses `from_wan22_pretrained` (which always downloads the Wan2.2 VAE). Every
    component -- video DiT, action DiT, VAE -- is built from config with random weights.
    No text encoder is loaded (dummy cached context is fed at inference). Inference
    latency is weight-independent, so this gives the same timing as a trained model.
    """
    from omegaconf import OmegaConf

    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.fastwam_idm import FastWAMIDM
    from fastwam.models.wan22.fastwam_joint import FastWAMJoint
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE38

    cls = {
        "fastwam.runtime.create_fastwam": FastWAM,
        "fastwam.runtime.create_fastwam_joint": FastWAMJoint,
        "fastwam.runtime.create_fastwam_idm": FastWAMIDM,
    }[str(cfg.model._target_)]

    video_dit_config = OmegaConf.to_container(cfg.model.video_dit_config, resolve=True)
    action_dit_config = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
    vsched = OmegaConf.to_container(cfg.model.video_scheduler, resolve=True)
    asched = OmegaConf.to_container(cfg.model.action_scheduler, resolve=True)
    loss = OmegaConf.to_container(cfg.model.get("loss", {}) or {}, resolve=True)
    text_dim = int(video_dit_config["text_dim"])
    proprio_dim = cfg.model.get("proprio_dim", None)
    proprio_dim = None if proprio_dim is None else int(proprio_dim)

    print("[build] fully random init, zero downloads (video DiT + action DiT + VAE all random)")
    video_expert = WanVideoDiT(**video_dit_config).to(device=device, dtype=dtype)
    action_expert = ActionDiT(**action_dit_config).to(device=device, dtype=dtype)
    vae = WanVideoVAE38().to(device=device, dtype=dtype)
    mot = MoT(mixtures={"video": video_expert, "action": action_expert},
              mot_checkpoint_mixed_attn=bool(cfg.model.get("mot_checkpoint_mixed_attn", False)))

    model = cls(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_dim=text_dim,
        proprio_dim=proprio_dim,
        device=device,
        torch_dtype=dtype,
        video_train_shift=float(vsched.get("train_shift", 5.0)),
        video_infer_shift=float(vsched.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(vsched.get("num_train_timesteps", 1000)),
        action_train_shift=float(asched["train_shift"]),
        action_infer_shift=float(asched["infer_shift"]),
        action_num_train_timesteps=int(asched["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
    return model


def summarize(latencies_s, label, replan_steps):
    ms = sorted(t * 1000.0 for t in latencies_s)
    n = len(ms)
    mean = statistics.mean(ms)
    p = lambda q: ms[min(n - 1, int(q * n))]
    print(f"\n=== {label} ===")
    print(f"  iters={n}  mean={mean:.1f} ms  std={statistics.pstdev(ms):.1f}  "
          f"min={ms[0]:.1f}  p50={p(0.5):.1f}  p90={p(0.9):.1f}  max={ms[-1]:.1f}")
    print(f"  throughput: {1000.0 / mean:.2f} chunks/s  |  effective control rate "
          f"(replan={replan_steps}): {replan_steps * 1000.0 / mean:.1f} Hz")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # pin to GPU2 BEFORE importing torch

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()

    overrides = [
        f"task={args.task}",
        f"model.load_text_encoder={'false' if args.no_text_encoder else 'true'}",
        "model.skip_dit_load_from_pretrain=true",   # init video DiT randomly; eval-style
        "model.action_dit_pretrained_path=null",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="train", overrides=overrides)

    device = "cuda:0"  # physical GPU set via CUDA_VISIBLE_DEVICES above
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; this benchmark requires a GPU.")
    print(f"[device] CUDA_VISIBLE_DEVICES={args.gpu} -> {torch.cuda.get_device_name(0)}")

    if args.download_base or args.ckpt:
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
        if args.ckpt:
            ckpt = Path(args.ckpt)
            if ckpt.exists():
                print(f"[ckpt] loading {ckpt}")
                model.load_checkpoint(str(ckpt))
            else:
                print(f"[warn] ckpt not found, using config-initialized weights: {ckpt}", file=sys.stderr)
    else:
        model = build_random_model(cfg, device=device, dtype=torch.bfloat16)
    model.eval()

    # ---- build a single observation (input image + proprio) from kai0 data ----
    import numpy as np

    image_np, state_np = load_kai0_frame_and_state(Path(args.data_dir))
    image = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    image = image * (2.0 / 255.0) - 1.0  # [1,3,384,320] in [-1,1]

    proprio = None
    if model.proprio_dim is not None:
        pd_ = model.proprio_dim
        s = state_np[:pd_] if state_np.shape[0] >= pd_ else np.pad(state_np, (0, pd_ - state_np.shape[0]))
        proprio = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)

    infer_kwargs = dict(
        input_image=image,
        action_horizon=args.action_horizon,
        proprio=proprio,
        text_cfg_scale=args.text_cfg_scale,
        sigma_shift=None,
        seed=0,
        rand_device="cpu",
        tiled=False,
    )
    if args.no_text_encoder or model.text_encoder is None:
        # dummy cached context (umt5-xxl dim); value-independent for timing.
        ctx_len = 128
        infer_kwargs["prompt"] = None
        infer_kwargs["context"] = torch.randn(1, ctx_len, model.text_dim, device=model.device, dtype=model.torch_dtype)
        infer_kwargs["context_mask"] = torch.ones(1, ctx_len, dtype=torch.bool, device=model.device)
    else:
        infer_kwargs["prompt"] = "fold the t-shirt"
        infer_kwargs["negative_prompt"] = ""

    print(f"[setup] action_horizon={args.action_horizon} replan={args.replan_steps} "
          f"action_dim={model.action_expert.action_dim} proprio_dim={model.proprio_dim} "
          f"input_image={tuple(image.shape)} dtype={model.torch_dtype}")

    for steps in args.num_inference_steps:
        kw = dict(infer_kwargs, num_inference_steps=steps)
        # warmup
        for _ in range(args.warmup):
            with torch.no_grad():
                model.infer_action(**kw)
        torch.cuda.synchronize()
        # timed
        lat = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.infer_action(**kw)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
        summarize(lat, f"infer_action  num_inference_steps={steps}", args.replan_steps)

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\n[mem] peak CUDA allocated: {peak:.2f} GiB")


if __name__ == "__main__":
    main()
