"""FastWAM 离线 200-ep 同协议评测适配器(对齐 GWP episode_report:exec 窗口 stride16、
chunk48、per-episode 均值再全局均值、raw mae@{1,10,24,48}+延迟;输出 summary.json 兼容
giga_world_policy/scripts/wam_pipeline/cmp_step_by_step.py)。

用法(本机,fastwam venv):
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_offline_fold.py --shard_id 0 --num_shards 8 \
    --weights runs/visrobot01_fold_uncond_1e-4/aihc_5n8g/checkpoints/weights/step_002500.pt \
    --out_dir runs/visrobot01_fold_uncond_1e-4/aihc_5n8g/report_step2500 [--nfe 20] [--joint]
  汇总: --aggregate --num_shards 8 --out_dir <同上>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
VAL = "../kai0/data/wam_fold_v1/visrobot01_val"
VK = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
HOR = [1, 10, 24, 48]


def build_model(weights, device="cuda"):
    from hydra import compose, initialize
    from hydra.utils import instantiate
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train",
                      overrides=["data=visrobot01_fold", "model=fastwam", "task=visrobot01_fold_uncond_1e-4"])
    model = instantiate(cfg.model)  # create_fastwam(已含 ActionDiT backbone 初始化)
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "dit"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    tgt = model.mot if all(k.startswith(("mixtures", "video", "action")) for k in list(sd)[:3]) else model
    missing, unexpected = tgt.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)} (前3: {missing[:3]} / {unexpected[:3]})", flush=True)
    return model


def prep_image(frames_by_cam):
    """复刻 RobotVideoDataset 'robotwin' 拼图:top 256x320 + 左右 128x160 → 384x320 → resize 240x320。
    输入: {cam: HxWx3 uint8};输出 [3,240,320] float [0,1]。"""
    t = {k: torch.from_numpy(v).permute(2, 0, 1).float() / 255.0 for k, v in frames_by_cam.items()}
    top = TF.resize(t["cam_high"].unsqueeze(0), [256, 320], antialias=True)[0]
    lf = TF.resize(t["cam_left_wrist"].unsqueeze(0), [128, 160], antialias=True)[0]
    rt = TF.resize(t["cam_right_wrist"].unsqueeze(0), [128, 160], antialias=True)[0]
    img = torch.cat([top, torch.cat([lf, rt], dim=-1)], dim=-2)  # [3,384,320]
    return TF.resize(img.unsqueeze(0), [240, 320], antialias=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights"); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_id", type=int, default=0); ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--n_metric_eps", type=int, default=200)
    ap.add_argument("--nfe", type=int, default=20)
    ap.add_argument("--joint", action="store_true", help="用 infer_joint(带视频想象)替代 infer_action")
    ap.add_argument("--stats", default="data/visrobot01_fold/dataset_stats.json")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    os.makedirs(f"{args.out_dir}/shards", exist_ok=True)

    if args.aggregate:
        eps = {}
        for i in range(args.num_shards):
            d = json.load(open(f"{args.out_dir}/shards/shard_{i}.json"))
            eps.update({int(k): v for k, v in d["metric"].items()})
            lat = d.get("latency", {})
        agg = {f"mae@{h}": float(np.mean([v[f"mae@{h}"] for v in eps.values() if v.get(f"mae@{h}") is not None])) for h in HOR}
        out = {"n_metric_eps": len(eps), "latency": lat,
               "raw_mae": {str(h): agg[f"mae@{h}"] for h in HOR},
               "pi05": {"1": 0.0219, "10": 0.0425, "24": 0.0743, "48": 0.1155}}
        json.dump(out, open(f"{args.out_dir}/summary.json", "w"), indent=1)
        print("[aggregate] raw mae@: " + " ".join(f"@{h} {agg[f'mae@{h}']:.4f}" for h in HOR)
              + f" | act-lat {lat.get('action_ms',0):.0f}ms", flush=True)
        return

    stats = json.load(open(args.stats))
    a_mean = np.array(stats["action"]["default"]["global_mean"]); a_std = np.array(stats["action"]["default"]["global_std"])
    s_mean = np.array(stats["state"]["default"]["global_mean"]); s_std = np.array(stats["state"]["default"]["global_std"])

    # t5 缓存(单 prompt)
    cache = list(Path("data/text_embeds_cache/visrobot01_fold").glob("*.pt"))[0]
    t5 = torch.load(cache, map_location="cpu", weights_only=False)
    ctx = t5.get("context") if isinstance(t5, dict) else t5
    cmask = t5.get("context_mask") if isinstance(t5, dict) else None
    if ctx.ndim == 2: ctx = ctx.unsqueeze(0)
    if cmask is not None and cmask.ndim == 1: cmask = cmask.unsqueeze(0)

    model = build_model(args.weights)
    from torchcodec.decoders import VideoDecoder

    meta = [json.loads(l) for l in open(f"{VAL}/meta/episodes.jsonl")]
    eps = sorted(int(m["episode_index"]) for m in meta)[: args.n_metric_eps]
    my = eps[args.shard_id :: args.num_shards]
    print(f"[shard {args.shard_id}/{args.num_shards}] eps={len(my)} nfe={args.nfe} joint={args.joint}", flush=True)

    metric = {}; lat_ms = []
    for ep in my:
        pq = f"{VAL}/data/chunk-000/episode_{ep:06d}.parquet"
        df = pd.read_parquet(pq)
        gt_all = np.stack(df["action"].to_numpy())[:, :14]
        st_all = np.stack(df["observation.state"].to_numpy())[:, :14]
        decs = {k: VideoDecoder(f"{VAL}/videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4") for k in VK}
        L = len(df)
        em = {f"mae@{h}": [] for h in HOR}
        for f in range(0, max(1, L - 1), 16):
            frames = {k: decs[k].get_frames_at([min(f, decs[k].metadata.num_frames - 1)]).data[0].permute(1, 2, 0).numpy() for k in VK}
            img = prep_image(frames)
            prop = torch.from_numpy((st_all[f] - s_mean) / (s_std + 1e-8)).float()
            t0 = time.time()
            with torch.no_grad():
                fn = model.infer_joint if args.joint else model.infer_action
                out = fn(prompt=None, input_image=img, action_horizon=48, proprio=prop,
                         context=ctx.to(model.device, model.torch_dtype),
                         context_mask=None if cmask is None else cmask.to(model.device),
                         num_inference_steps=args.nfe, seed=0)
            lat_ms.append((time.time() - t0) * 1000)
            pa = out["action"][0].float().cpu().numpy() * (a_std + 1e-8) + a_mean  # z-score 逆 + 全 abs(无 delta)
            gt = gt_all[f : f + 48]
            n = min(len(pa), len(gt)); ae = np.abs(pa[:n] - gt[:n])
            for h in HOR:
                if h <= n: em[f"mae@{h}"].append(float(ae[h - 1].mean()))
        metric[ep] = {f"mae@{h}": (float(np.mean(em[f"mae@{h}"])) if em[f"mae@{h}"] else None) for h in HOR}
        print(f"[shard {args.shard_id}] ep{ep} done mae@48={metric[ep]['mae@48']}", flush=True)
    json.dump({"metric": metric, "latency": {"action_ms": float(np.mean(lat_ms[3:]) if len(lat_ms) > 3 else np.mean(lat_ms))}},
              open(f"{args.out_dir}/shards/shard_{args.shard_id}.json", "w"))
    print(f"[shard {args.shard_id}] saved", flush=True)


if __name__ == "__main__":
    main()
