r"""预计算 VAE latent 缓存——消除训练期在线视频解码+VAE(吞吐 3-5×)。

策略(episode 级批量,避免 per-window 随机 seek——PFS 上会卡死):
  每个 episode 顺序解码 3 条 mp4 一次 → 内存切出全部 stride=4 窗口 → 像素链严格复刻
  训练路径(processor Resize 240x320 → robotwin 拼图 384x320 → normalize)→ 批量 VAE →
  存 1 个 .pt:{"starts": [全局帧号...], "latents": [W,48,4,24,20] bf16}。

像素链对齐依据(必须逐步一致,否则缓存 latent ≠ 在线 VAE):
  FastWAMProcessor.preprocess: ToTensor(/255) + Resize([240,320]) per-cam
  RobotVideoDataset._get robotwin 分支: top→[256,320], wrist→[128,160], concat→[384,320]
  resize/crop(video_size=[384,320] → identity) + Normalize((x-0.5)/0.5)
  VAE: WanVideoVAE38.encode(确定性,取 mu)
验证:scripts/check_latent_parity.py(与原路径逐位对拍,放量前必须通过)。

用法(8 卡,经 launcher 脚本):bash scripts/launch_compute_latents.sh
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
os.chdir(str(REPO))
sys.path.insert(0, "src")

DATA = str((REPO / ".." / "kai0" / "data" / "wam_fold_v1" / "visrobot01_train").resolve())
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
VID_IDX = np.arange(0, 49, 4)  # 13 帧/窗(action_video_freq_ratio=4)


def load_vae(device="cuda"):
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(REPO / "checkpoints")
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    _, _, vae_cfg, _ = _resolve_configs("Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.1-T2V-1.3B", redirect_common_files=False)
    vae_cfg.download_if_necessary()
    return _load_registered_model(str(vae_cfg.path), "wan_video_vae", torch_dtype=torch.bfloat16, device=device).eval()


def dec_episode(ep):
    """顺序解码一个 episode 的 3 条 mp4(无 seek)→ {cam: [T,H,W,3] uint8}。"""
    import av
    out = {}
    for cam in CAMS:
        c = av.open(f"{DATA}/videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4")
        frames = [f.to_ndarray(format="rgb24") for packet in c.demux(c.streams.video[0]) for f in packet.decode()]
        c.close()
        out[cam] = np.stack(frames, axis=0)
    return out


def window_pixels(frames, lf):
    """从单集帧数组切窗(lf=集内局部起始帧)→ 严格复刻训练像素链 → [C,13,384,320] in [-1,1]。"""
    per_cam = []
    for cam in CAMS:
        x = torch.from_numpy(frames[cam][lf + VID_IDX].copy()).permute(0, 3, 1, 2).float() / 255.0  # ToTensor
        x = TF.resize(x, [240, 320], interpolation=InterpolationMode.BILINEAR, antialias=True)       # processor Resize
        per_cam.append(x)
    top = TF.resize(per_cam[0], [256, 320], interpolation=InterpolationMode.BILINEAR, antialias=True)
    lw = TF.resize(per_cam[1], [128, 160], interpolation=InterpolationMode.BILINEAR, antialias=True)
    rw = TF.resize(per_cam[2], [128, 160], interpolation=InterpolationMode.BILINEAR, antialias=True)
    v = torch.cat([top, torch.cat([lw, rw], dim=-1)], dim=-2)  # [13,C,384,320](video_size 384x320 → identity)
    v = (v - 0.5) / 0.5
    return v.permute(1, 0, 2, 3)  # [C,13,384,320]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--total", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0, help=">0: 只跑 N 个 episode")
    args = ap.parse_args()

    vae = load_vae(args.device)
    print("[latent] VAE loaded", flush=True)

    # episode 边界:直接读 meta(不构造 lerobot dataset,避免 HF split 并发问题)
    import json
    meta = [json.loads(l) for l in open(f"{DATA}/meta/episodes.jsonl")]
    ep_len = {int(m["episode_index"]): int(m["length"]) for m in meta}
    eps_sorted = sorted(ep_len)
    # 全局帧偏移(拼接序 = episode_index 升序,与 BaseLerobotDataset episode_data_index 一致)
    ep_global = {}
    off = 0
    for e in eps_sorted:
        ep_global[e] = off
        off += ep_len[e]
    print(f"[latent] eps={len(eps_sorted)} total_frames={off} stride={args.stride}", flush=True)

    out_dir = Path(DATA) / "vae_latent"; out_dir.mkdir(parents=True, exist_ok=True)
    my = eps_sorted[args.shard :: args.total]
    if args.smoke:
        my = my[: args.smoke]

    for ep in tqdm(my, desc=f"shard{args.shard}"):
        out_f = out_dir / f"episode_{ep:06d}.pt"
        if out_f.exists():
            continue  # 断点续跑
        L = ep_len[ep]
        locs = list(range(0, L - 49, args.stride))  # 集内局部起点(完整 49 帧窗)
        if not locs:
            continue
        frames = dec_episode(ep)
        T_vid = min(f.shape[0] for f in frames.values())
        locs = [l for l in locs if l + 48 < T_vid]  # 视频长度兜底
        videos = [window_pixels(frames, l) for l in locs]
        lat = []
        for b0 in range(0, len(videos), args.batch):
            batch = torch.stack(videos[b0:b0 + args.batch]).to(args.device, dtype=torch.bfloat16)
            with torch.no_grad():
                z = vae.encode(batch, device=args.device)
            lat.append((z[0] if isinstance(z, list) else z).cpu())
        latents = torch.cat(lat, dim=0)
        starts = [ep_global[ep] + l for l in locs]  # 存全局帧号(训练侧 idx 即全局帧号)
        tmp = out_f.with_suffix(".tmp")
        torch.save({"starts": starts, "latents": latents, "stride": args.stride}, tmp)
        os.replace(tmp, out_f)
    print(f"[latent] shard{args.shard} done ({len(my)} eps)", flush=True)


if __name__ == "__main__":
    main()
