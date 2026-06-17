"""可复用的 FastWAM fold 评测核心:全 val 集 sharded MAE@{1,10,chunk/2,chunk}。

被两处共用,保证数值口径一致:
  1. scripts/eval_offline_fold.py —— CLI / 外部 per-ckpt watcher(从 .pt 重建模型)。
  2. trainer 内联 eval —— 直接用 live model + 现有训练 rank 分片(无需重载权重/不抢卡)。

评测口径(对齐 GWP episode_report):exec 窗口 stride16、动作 chunk=48、per-episode 均值再全局均值、
raw mae(z-score 逆 + 全 abs)。相机按 view_keys 位置拼图(view_keys[0]=top/overhead→256x320 主图)。
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

HOR_DEFAULT = [1, 10, 24, 48]


def prep_image(frames_by_cam, view_keys):
    """复刻训练像素链:per-cam ToTensor(/255)+Resize(240,320) → top 256x320 / 腕 128x160 → 拼 [3,384,320] in [-1,1]。
    view_keys 按角色顺序 [top, left_wrist, right_wrist];按【位置】取,不依赖具体相机名。"""
    t = {}
    for k, v in frames_by_cam.items():
        x = torch.from_numpy(v).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t[k] = TF.resize(x, [240, 320], antialias=True)[0]
    top = TF.resize(t[view_keys[0]].unsqueeze(0), [256, 320], antialias=True)[0]
    lf = TF.resize(t[view_keys[1]].unsqueeze(0), [128, 160], antialias=True)[0]
    rt = TF.resize(t[view_keys[2]].unsqueeze(0), [128, 160], antialias=True)[0]
    img = torch.cat([top, torch.cat([lf, rt], dim=-1)], dim=-2)  # [3,384,320]
    return img * 2.0 - 1.0


def load_eval_assets(stats_path, text_emb_dir):
    """读 norm stats(action/state 的 global_mean/std)+ 单 prompt T5 缓存(context/mask)。"""
    stats = json.load(open(stats_path))
    a_mean = np.array(stats["action"]["default"]["global_mean"]); a_std = np.array(stats["action"]["default"]["global_std"])
    s_mean = np.array(stats["state"]["default"]["global_mean"]); s_std = np.array(stats["state"]["default"]["global_std"])
    cache = list(Path(text_emb_dir).glob("*.pt"))[0]
    t5 = torch.load(cache, map_location="cpu", weights_only=False)
    ctx = t5["context"]; cmask = t5["mask"].bool()
    ctx = ctx.clone(); ctx[~cmask] = 0.0  # padding 清零后 mask 置全 1(对齐训练约定)
    cmask = torch.ones_like(cmask)
    if ctx.ndim == 2: ctx = ctx.unsqueeze(0)
    if cmask.ndim == 1: cmask = cmask.unsqueeze(0)
    return a_mean, a_std, s_mean, s_std, ctx, cmask


def eval_fold(model, val_root, view_keys, a_mean, a_std, s_mean, s_std, ctx, cmask,
              shard_id=0, num_shards=1, nfe=20, n_eps=100, action_chunk=48, hor=HOR_DEFAULT,
              joint=False, opt_runner=None, opt_infer=None, log=print):
    """评本 shard 负责的 episode 子集 → {ep: {mae@h}};另返回 per-window 延迟列表。
    用 model.infer_action(默认)或 model.infer_joint(joint=True)或 opt 引擎(opt_runner/opt_infer)。"""
    import pandas as pd
    from torchcodec.decoders import VideoDecoder
    meta = [json.loads(l) for l in open(f"{val_root}/meta/episodes.jsonl")]
    eps = sorted(int(m["episode_index"]) for m in meta)[:n_eps]
    my = eps[shard_id::num_shards]
    metric = {}; lat_ms = []
    for ep in my:
        df = pd.read_parquet(f"{val_root}/data/chunk-000/episode_{ep:06d}.parquet")
        gt_all = np.stack(df["action"].to_numpy())[:, :14]
        st_all = np.stack(df["observation.state"].to_numpy())[:, :14]
        decs = {k: VideoDecoder(f"{val_root}/videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4") for k in view_keys}
        L = len(df); em = {f"mae@{h}": [] for h in hor}
        for f in range(0, max(1, L - 1), 16):
            frames = {k: decs[k].get_frames_at([min(f, decs[k].metadata.num_frames - 1)]).data[0].permute(1, 2, 0).numpy() for k in view_keys}
            img = prep_image(frames, view_keys)
            prop = torch.from_numpy((st_all[f] - s_mean) / (s_std + 1e-8)).float()
            t0 = time.time()
            if opt_runner is not None:
                out = opt_infer(model, opt_runner, context=ctx.to(model.device, model.torch_dtype),
                                context_mask=cmask.to(model.device), image=img, proprio=prop,
                                action_horizon=action_chunk, num_inference_steps=nfe, seed=0)
            else:
                with torch.no_grad():
                    fn = model.infer_joint if joint else model.infer_action
                    out = fn(prompt=None, input_image=img, action_horizon=action_chunk, proprio=prop,
                             context=ctx.to(model.device, model.torch_dtype),
                             context_mask=None if cmask is None else cmask.to(model.device),
                             num_inference_steps=nfe, seed=0)
            lat_ms.append((time.time() - t0) * 1000)
            pa = out["action"].float().cpu().numpy() * (a_std + 1e-8) + a_mean  # z-score 逆 + 全 abs
            gt = gt_all[f:f + action_chunk]
            n = min(len(pa), len(gt)); ae = np.abs(pa[:n] - gt[:n])
            for h in hor:
                if h <= n: em[f"mae@{h}"].append(float(ae[h - 1].mean()))
        metric[ep] = {f"mae@{h}": (float(np.mean(em[f"mae@{h}"])) if em[f"mae@{h}"] else None) for h in hor}
        log(f"[shard {shard_id}] ep{ep} mae@{hor[-1]}={metric[ep][f'mae@{hor[-1]}']}")
    return metric, lat_ms


def aggregate(metrics_list, hor=HOR_DEFAULT):
    """合并多 shard 的 {ep:{mae@h}} → 全局 {h: 全集均值}(per-ep 已是窗口均值)。"""
    eps = {}
    for m in metrics_list:
        eps.update(m)
    out = {}
    for h in hor:
        vals = [v[f"mae@{h}"] for v in eps.values() if v.get(f"mae@{h}") is not None]
        out[h] = float(np.mean(vals)) if vals else None
    return out, len(eps)
