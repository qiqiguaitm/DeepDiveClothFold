"""GigaWorld-aligned distributed eval for tau0 (video + action metrics), 16 GPU.

Mirrors giga_world_policy/scripts/wam_pipeline/eval_watch.py:
  closed-loop generation (ref frame + state -> generate future video + action chunk), then
  VIDEO  : PSNR, SSIM (Gaussian 11x11), temporal_absdiff_ratio, LPIPS   (video_metrics_gpu, ported)
  ACTION : physical MAE/MSE, mae@{1,10,ck/2,ck}, mae_move, beat_stay_move, shape_corr_move
           (denormalize + add_state -> abs units, move-dim threshold 0.05 rad)
Windows sharded across ranks; per-metric sum/count all-reduced (horizons/move metrics may be
absent on some windows). Writes runs/eval_gigaworld.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finetune.model_joint import build_joint_wanmodel  # noqa: E402
from finetune.train_tau0 import compute_seq_len, ACTION_CHUNK, ACTION_DIM  # noqa: E402
from finetune.data_joint import LatentJointDataset, proprio_sample, undo_proprio_action  # noqa: E402
from finetune.gen_video_compare import load_vae, decode, gen_video, gen_action, to_uint8_panorama  # noqa: E402

VAL = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_val"
ASSETS = os.path.join(ROOT, "finetune", "assets")
CKPT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/checkpoints/tau-0-wm"
HORIZONS = sorted({h for h in (1, 10, ACTION_CHUNK // 2, ACTION_CHUNK)})
KEYS = ["psnr", "ssim", "temporal_absdiff_ratio", "lpips", "action_mae", "action_mse",
        *[f"mae@{h}" for h in HORIZONS], "mae_move", "beat_stay_move", "shape_corr_move"]


def _gauss_win(ws, sigma, device, dtype):
    c = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return g[:, None] * g[None, :]


def video_metrics_gpu(pred_thwc, gt_thwc, lpips_fn, device):
    """Ported verbatim from GigaWorld eval_watch.video_metrics_gpu. (T,H,W,C) float[0,255]."""
    T = min(pred_thwc.shape[0], gt_thwc.shape[0])
    p = pred_thwc[:T].permute(0, 3, 1, 2).contiguous().float()
    g = gt_thwc[:T].permute(0, 3, 1, 2).contiguous().float()
    C = p.shape[1]
    mse = ((p - g) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    out = {"psnr": float((10.0 * torch.log10((255.0 ** 2) / mse)).mean().item())}
    ws = 11; w = _gauss_win(ws, 1.5, device, p.dtype).expand(C, 1, ws, ws); pad = ws // 2
    cv = lambda x: F.conv2d(x, w, padding=pad, groups=C)
    mu1, mu2 = cv(p), cv(g); mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1, s2, s12 = cv(p * p) - mu1s, cv(g * g) - mu2s, cv(p * g) - mu12
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    out["ssim"] = float((((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1s + mu2s + C1) * (s1 + s2 + C2))).mean().item())
    if T >= 2:
        out["temporal_absdiff_ratio"] = float(((p[1:] - p[:-1]).abs().mean() / ((g[1:] - g[:-1]).abs().mean() + 1e-6)).item())
    if lpips_fn is not None:
        with torch.no_grad():
            out["lpips"] = float(lpips_fn(p / 127.5 - 1.0, g / 127.5 - 1.0).mean().item())
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/runs/tau0_fold_p2_32g/final.pt")
    ap.add_argument("--tag", default="p2_final")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--windows_per_ep", type=int, default=1)
    ap.add_argument("--out", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/runs/eval_gigaworld.json")
    ap.add_argument("--lpips", action="store_true", help="enable LPIPS (needs 233MB AlexNet download)")
    args = ap.parse_args()

    from accelerate import Accelerator
    accel = Accelerator(mixed_precision="bf16")
    dev = accel.device; dt = torch.bfloat16
    R, W = accel.process_index, accel.num_processes
    is_main = accel.is_main_process

    def log(*a):
        if is_main: print("[gw-eval]", *a, flush=True)

    vae, lm, ls = load_vae(dev, dt)
    # LPIPS needs a 233MB AlexNet download (offline/slow here -> hangs); optional in GigaWorld too.
    lpips_fn = None
    if args.lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").to(dev).eval()
        except Exception as e:
            lpips_fn = None; log(f"lpips unavailable ({e}); skipping")
    model, _ = build_joint_wanmodel(action_in_dim=14, ckpt_dir=CKPT, load_pretrained=True,
                                    dtype=dt, device=str(dev), verbose=is_main)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    model.eval()
    seq_len = compute_seq_len()
    ds = LatentJointDataset(VAL, f"{ASSETS}/statistics_visrobot01.json", ACTION_CHUNK, embed_id=0)
    my_eps = list(range(R, len(ds), W))
    log(f"world={W} ckpt={args.ckpt} tag={args.tag} eps/rank={len(my_eps)} horizons={HORIZONS}")

    sums = {k: 0.0 for k in KEYS}; cnts = {k: 0 for k in KEYS}
    rng = np.random.RandomState(100 + R)
    for ei in my_eps:
        ep = ds._load_episode(ei)
        for _ in range(args.windows_per_ep):
            w = rng.randint(0, ep["visual"].shape[0]); t = ep["starts"][w]; nL = len(ep["s"])
            idx = np.clip(np.arange(t, t + ACTION_CHUNK), 0, nL - 1)
            s_abs = ep["s"][t]; gt_act = ep["a"][idx]
            ns, _na = proprio_sample(s_abs, gt_act, ds.stats)
            gt_lat = ep["visual"][w].float().to(dev, dt)
            z_cond = gt_lat.clone(); z_cond[:, 0:1] = ep["ref"][w].float().to(dev, dt)
            ctx = ep["t5"].to(dev, dt); state = torch.from_numpy(ns).unsqueeze(0).to(dev, dt)
            # ---- video ----
            pred_lat = gen_video(model, z_cond, ctx, args.steps, dev, seq_len)
            gt_v = torch.from_numpy(to_uint8_panorama(decode(vae, lm, ls, gt_lat))).to(dev).float()
            pr_v = torch.from_numpy(to_uint8_panorama(decode(vae, lm, ls, pred_lat))).to(dev).float()
            vm = video_metrics_gpu(pr_v, gt_v, lpips_fn, dev)
            # ---- action ----
            pred_norm = gen_action(model, z_cond, ctx, state, args.steps, dev, seq_len)[0].cpu().numpy()
            pred_abs = undo_proprio_action(pred_norm, s_abs, ds.stats)
            L = min(len(pred_abs), len(gt_act)); ae = np.abs(pred_abs[:L] - gt_act[:L])
            am = {"action_mae": float(ae.mean()), "action_mse": float(((pred_abs[:L] - gt_act[:L]) ** 2).mean())}
            for h in HORIZONS:
                if h <= L: am[f"mae@{h}"] = float(ae[h - 1].mean())
            mv = np.abs(gt_act[:L] - gt_act[:1]).max(axis=0); move = mv > 0.05      # 0.05 rad
            if move.any():
                stay_ae = np.abs(s_abs[None, :] - gt_act[:L])
                am["mae_move"] = float(ae[:, move].mean())
                am["beat_stay_move"] = float((ae[:, move].mean(0) < stay_ae[:, move].mean(0)).mean())
                cs = [float(np.corrcoef(pred_abs[:L, dd], gt_act[:L, dd])[0, 1]) for dd in np.where(move)[0]
                      if pred_abs[:L, dd].std() > 1e-4 and gt_act[:L, dd].std() > 1e-4]
                if cs: am["shape_corr_move"] = float(np.mean(cs))
            for k, v in {**vm, **am}.items():
                if k in sums and np.isfinite(v):
                    sums[k] += v; cnts[k] += 1

    # all-reduce per-metric sum + count
    sv = torch.tensor([sums[k] for k in KEYS], device=dev, dtype=torch.float64)
    cv = torch.tensor([cnts[k] for k in KEYS], device=dev, dtype=torch.float64)
    accel.wait_for_everyone()
    torch.distributed.all_reduce(sv); torch.distributed.all_reduce(cv)
    if is_main:
        res = {"tag": args.tag, "ckpt": args.ckpt, "world_size": W, "steps": args.steps}
        for i, k in enumerate(KEYS):
            res[k] = round(float(sv[i] / cv[i]), 5) if cv[i] > 0 else None
        res["n_windows"] = int(cv[KEYS.index("action_mae")].item())
        allr = []
        if os.path.exists(args.out):
            try: allr = json.load(open(args.out))
            except Exception: allr = []
        allr.append(res); json.dump(allr, open(args.out, "w"), indent=2)
        print(f"[gw-eval] === {args.tag} (n={res['n_windows']}, {W} GPU) ===", flush=True)
        print(f"[gw-eval] VIDEO  PSNR={res['psnr']} SSIM={res['ssim']} temporal={res['temporal_absdiff_ratio']} LPIPS={res['lpips']}", flush=True)
        print(f"[gw-eval] ACTION MAE={res['action_mae']} MSE={res['action_mse']} "
              f"mae@1={res.get('mae@1')} mae@{ACTION_CHUNK}={res.get('mae@'+str(ACTION_CHUNK))} "
              f"mae_move={res['mae_move']} beat_stay={res['beat_stay_move']} shape_corr={res['shape_corr_move']}", flush=True)


if __name__ == "__main__":
    main()
