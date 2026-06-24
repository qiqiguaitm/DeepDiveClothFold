"""Video generation + comparison for the tau0 report (world-model rollout vs GT).

For N held-out windows: condition on the observed frame (ref latent), generate the
future video latent via the video-diffusion path (pretrained tau0 backbone, frozen in P1),
decode predicted + GT latents through the Wan VAE, compute PSNR/SSIM, and save a
side-by-side (GT | pred) frame grid PNG for embedding in report.html.

Single GPU. The 3 camera views are concatenated along width in the cached latent, so a
decoded frame is a 192x768 panorama (top_head | left_wrist | right_wrist).
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finetune.model_joint import build_joint_wanmodel  # noqa: E402
from finetune.train_tau0 import compute_seq_len, video_latent_shape, PATCH, CHUNK  # noqa: E402
from finetune.data_joint import LatentJointDataset  # noqa: E402

VAL = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_val"
ASSETS = os.path.join(ROOT, "finetune", "assets")
CKPT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/checkpoints/tau-0-wm"
VAE_DIR = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/checkpoints/Wan2.2-TI2V-5B-Diffusers"


def load_vae(device, dtype):
    from diffusers.models import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(VAE_DIR, subfolder="vae", torch_dtype=dtype).to(device).eval()
    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    ls = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    return vae, lm, ls


@torch.no_grad()
def decode(vae, lm, ls, latent_norm):
    # latent_norm: [C,T,h,W] normalized -> un-normalize -> [B,C,T,h,W] -> decode -> video [-1,1]
    z = latent_norm.unsqueeze(0)
    z = z / (1.0 / ls) + lm   # stored = (raw-mean)*(1/std) -> raw = stored*std + mean
    vid = vae.decode(z, return_dict=False)[0]   # [1,3,T,H,W] in [-1,1]
    return vid[0].clamp(-1, 1)


@torch.no_grad()
def gen_video(model, z_cond_full, ctx, steps, device, seq_len):
    """Denoise the future video latent conditioned on frame-0 = observed (z_cond_full[:, :1])."""
    from models.wan_2_2_models.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
    C, t_lat, h, Wv, _, _ = video_latent_shape()
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=5.0, use_dynamic_shifting=False)
    sched.set_timesteps(steps, device=device, shift=5.0)
    mask = torch.ones(C, t_lat, h, Wv, device=device, dtype=z_cond_full.dtype); mask[:, 0] = 0
    latent = torch.randn(C, t_lat, h, Wv, device=device, dtype=torch.float32)
    latent = ((1 - mask) * z_cond_full + mask * latent).to(z_cond_full.dtype)
    for t in sched.timesteps:
        temp = (mask[0][:, ::PATCH[1], ::PATCH[2]] * float(t.item())).flatten()
        if temp.numel() < seq_len:
            temp = torch.cat([temp, temp.new_full((seq_len - temp.numel(),), float(t.item()))])
        v_ts = temp[:seq_len].unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x=[latent], t=v_ts, context=[ctx], seq_len=seq_len,
                        return_video=True, return_action=False, store_buffer=False)
        npred = out["video"][0].float()
        latent = sched.step(npred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)
        latent = ((1 - mask) * z_cond_full + mask * latent).to(z_cond_full.dtype)
    return latent


@torch.no_grad()
def gen_action(model, z_cond_full, ctx, state, steps, device, seq_len):
    """Full iterative action inference (deployment path): video features computed once
    (store_buffer at step 0), then denoise the action chunk over its own schedule.
    Returns predicted NORMALIZED action chunk [1,ACTION_CHUNK,14]."""
    from models.wan_2_2_models.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from finetune.train_tau0 import ACTION_CHUNK, ACTION_DIM
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
    sched.set_timesteps(steps, device=device, shift=1.0)
    action = torch.randn(1, ACTION_CHUNK, ACTION_DIM, device=device, dtype=z_cond_full.dtype)
    vbuf = None
    # video held at t=1000 EXCEPT the conditioning frame-0 (t=0), so the buffer captures real
    # observed-frame features (mirrors pipeline.infer's mask2/temp_ts).
    vmask = torch.ones_like(z_cond_full); vmask[:, 0] = 0
    temp = (vmask[0][:, ::PATCH[1], ::PATCH[2]] * 1000.0).flatten()
    if temp.numel() < seq_len:
        temp = torch.cat([temp, temp.new_full((seq_len - temp.numel(),), 1000.0)])
    v_ts = temp[:seq_len].unsqueeze(0).to(device)
    for i, t in enumerate(sched.timesteps):
        a_ts = t.view(1, 1).repeat(1, ACTION_CHUNK).float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x=[z_cond_full], t=v_ts, context=[ctx], seq_len=seq_len,
                        action_states=action, action_timestep=a_ts,
                        return_video=(i == 0), return_action=True, store_buffer=(i == 0),
                        video_states_buffer=vbuf, history_action_state=state)
        if i == 0:
            vbuf = out["video_states_buffer"]
        action = sched.step(out["action"], t, action, return_dict=False)[0]
    return action.float()


@torch.no_grad()
def rollout_video(model, ref0, ctx, K, steps, device, seq_len):
    """GigaWorld-style closed-loop rollout: generate a 2-frame chunk conditioned on the current
    frame, take the predicted next frame as the new conditioning, repeat K times.
    ref0: [C,1,h,Wv] (observed). Returns [C,K+1,h,Wv] latent (frame0=observed + K predicted)."""
    C, t_lat, h, Wv, _, _ = video_latent_shape()  # t_lat=2
    frames = [ref0]
    cur = ref0
    for _ in range(K):
        z_cond = torch.cat([cur, torch.zeros_like(cur).repeat(1, t_lat - 1, 1, 1)], dim=1)  # [C,2,..]
        gen = gen_video(model, z_cond, ctx, steps, device, seq_len)  # [C,2,h,Wv]
        nxt = gen[:, t_lat - 1:t_lat]
        frames.append(nxt)
        cur = nxt
    return torch.cat(frames, dim=1)


def to_uint8_panorama(vid_cthw):
    # vid: [3,T,H,W] in [-1,1] -> stack frames horizontally per timestep into a [T] list of HxWx3 uint8
    v = ((vid_cthw.float() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()  # [3,T,H,W]
    v = np.transpose(v, (1, 2, 3, 0))  # [T,H,W,3]
    return v


# GigaWorld episode_report-style 14-dim action trajectory (pred raw vs GT), deployment layout.
DIM = [f"L_j{i}" for i in range(6)] + ["L_grip"] + [f"R_j{i}" for i in range(6)] + ["R_grip"]


def save_traj_png(pred_abs, gt_abs, path, title=""):
    """pred_abs/gt_abs: (T,14) absolute. 14 subplots (2 arms x 7), pred(raw,red) vs GT(black--)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = min(len(pred_abs), len(gt_abs))
    x = np.arange(T)
    fig, axes = plt.subplots(2, 7, figsize=(20, 5))
    axes = axes.flatten()
    for d in range(14):
        axes[d].plot(x, gt_abs[:T, d], "k--", lw=1.5, label="GT")
        axes[d].plot(x, pred_abs[:T, d], "r-", lw=1.2, label="pred(raw)")
        axes[d].set_title(DIM[d], fontsize=9)
        axes[d].tick_params(labelsize=7)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/runs/tau0_fold_p1/final.pt")
    ap.add_argument("--n_windows", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rollout_k", type=int, default=8, help="closed-loop rollout chunks (video length)")
    ap.add_argument("--fps", type=int, default=4, help="GIF playback fps")
    ap.add_argument("--out_dir", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/runs/report_assets")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device); dt = torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)

    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from PIL import Image

    vae, lm, ls = load_vae(dev, dt)
    model, _ = build_joint_wanmodel(action_in_dim=14, ckpt_dir=CKPT, load_pretrained=True,
                                    dtype=dt, device=str(dev), verbose=True)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    model.eval()
    seq_len = compute_seq_len()

    from finetune.data_joint import proprio_sample, undo_proprio_action
    from finetune.train_tau0 import ACTION_CHUNK
    HORIZONS = [1, 10, 25, ACTION_CHUNK]
    ds = LatentJointDataset(VAL, f"{ASSETS}/statistics_visrobot01.json", action_chunk=ACTION_CHUNK, embed_id=0)
    rng = np.random.RandomState(7)
    results = []
    mae_acc = {h: [] for h in HORIZONS}
    for k in range(args.n_windows):
        ep = ds._load_episode(rng.randint(len(ds)))
        w = rng.randint(0, ep["visual"].shape[0]); t = ep["starts"][w]; nL = len(ep["s"])
        idx = np.clip(np.arange(t, t + ACTION_CHUNK), 0, nL - 1)
        s_abs = ep["s"][t]; a_abs_gt = ep["a"][idx]             # raw absolute (14,), (33,14)
        ns, na = proprio_sample(s_abs, a_abs_gt, ds.stats)
        gt_lat = ep["visual"][w].float().to(dev, dt)
        z_cond = gt_lat.clone(); z_cond[:, 0:1] = ep["ref"][w].float().to(dev, dt)
        ctx = ep["t5"].to(dev, dt)
        state = torch.from_numpy(ns).unsqueeze(0).to(dev, dt)
        # ---- action: full inference -> abs -> MAE@horizon (vs pi0.5) ----
        pred_norm = gen_action(model, z_cond, ctx, state, args.steps, dev, seq_len)[0].cpu().numpy()
        pred_abs = undo_proprio_action(pred_norm, s_abs, ds.stats)   # (33,14) abs
        win_mae = {}
        for h in HORIZONS:
            m = float(np.abs(pred_abs[:h] - a_abs_gt[:h]).mean())
            mae_acc[h].append(m); win_mae[f"mae@{h}"] = round(m, 5)
        # GigaWorld-style action trajectory plot (14-dim pred raw vs GT)
        traj_fn = os.path.join(args.out_dir, f"traj_{k}.png")
        save_traj_png(pred_abs, a_abs_gt, traj_fn,
                      title=f"sample {k} · action chunk (33 steps) · pred(raw) vs GT · mae@1={win_mae['mae@1']} mae@33={win_mae.get('mae@'+str(ACTION_CHUNK),'')}")
        # ---- video: CLEAN single-chunk full-clip generation vs contiguous GT ----
        # gt_lat = ep["visual"][w] is the CONTIGUOUS future clip; z_cond = it with frame0=observed ref.
        # One gen_video() call denoises the whole t_lat-frame chunk (chunk=33 -> 33-frame one-shot video),
        # so there is NO autoregressive recursion and NO window-ref temporal misalignment.
        gen_lat = gen_video(model, z_cond, ctx, args.steps, dev, seq_len)   # [C,t_lat,h,Wv]
        gt_vid = to_uint8_panorama(decode(vae, lm, ls, gt_lat))             # contiguous GT clip [T,H,W,3]
        pred_vid = to_uint8_panorama(decode(vae, lm, ls, gen_lat))
        T = min(gt_vid.shape[0], pred_vid.shape[0])
        ev = list(range(1, T)) if T > 1 else [0]                           # exclude conditioning frame-0
        ps = float(np.mean([psnr(gt_vid[i], pred_vid[i], data_range=255) for i in ev]))
        ss = float(np.mean([ssim(gt_vid[i], pred_vid[i], data_range=255, channel_axis=2) for i in ev]))
        sep = np.full((4, gt_vid.shape[2], 3), 255, np.uint8)
        gif = [Image.fromarray(np.concatenate([gt_vid[ti], sep, pred_vid[ti]], axis=0)) for ti in range(T)]
        fn = os.path.join(args.out_dir, f"vidcmp_{k}.gif")
        gif[0].save(fn, save_all=True, append_images=gif[1:], duration=int(1000 / args.fps), loop=0)
        results.append({"window": k, "psnr": round(ps, 2), "ssim": round(ss, 4), "frames": T,
                        "img": os.path.basename(fn), "traj": os.path.basename(traj_fn), "mae": win_mae})
        print(f"[vid] window {k}: single-chunk ({T} frames, chunk={CHUNK}) PSNR={ps:.2f} SSIM={ss:.4f} -> {fn}", flush=True)

    import json
    mae_at = {f"mae@{h}": round(float(np.mean(mae_acc[h])), 5) for h in HORIZONS}
    print(f"[action] full-inference MAE@horizon (abs, phys): {mae_at}", flush=True)
    out = {"ckpt": args.ckpt, "steps": args.steps, "n_windows": args.n_windows,
           "mean_psnr": round(float(np.mean([r["psnr"] for r in results])), 2),
           "mean_ssim": round(float(np.mean([r["ssim"] for r in results])), 4),
           "action_mae_at_horizon": mae_at,
           "windows": results}
    json.dump(out, open(os.path.join(args.out_dir, "video_metrics.json"), "w"), indent=2)
    print(f"[vid] mean PSNR={out['mean_psnr']} SSIM={out['mean_ssim']}; wrote {args.out_dir}/video_metrics.json")


if __name__ == "__main__":
    main()
