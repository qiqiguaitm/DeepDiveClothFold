"""解耦式 eval-agent —— 与训练完全分离,跑在独立 GPU 上,只读 PFS 上的 checkpoint。

设计(与训练解耦,触发即 cadence):
  loop:
    new = 轮询 OUTPUT_DIR 里新出现的 checkpoint_*_step_*/(训练每 checkpoint_interval 落一个)
    load transformer(优先 transformer_ema,回退 transformer) + VAE -> WAPipeline
    for clip in 固定 hold-out(visrobot01 末尾 N 集,永不进训练):
        闭环逐块生成: 一块(ref帧+state)进 -> 生成 num_frames 未来帧 + action_chunk 出,
                      预测末帧/末state 反馈作下一块输入(--closed-loop),重复若干块
        指标 = PSNR/SSIM/LPIPS + 时序一致性(帧间差) vs GT;并算 action MSE
    append {step, psnr, ssim, lpips, temporal, action_mse} -> eval_log.jsonl + tensorboard
    存几条 预测|GT 并排 MP4 供肉眼看
要点:零训练开销(只读权重,另一批 GPU);出指标-step 曲线替代 flat loss 作"在不在变好"的信号。

用法(在评估机 / 空闲 GPU 上):
  CUDA_VISIBLE_DEVICES=7 python -m scripts.wam_pipeline.eval_watch \
    --output_dir runs/visrobot01_fold --model_id ../checkpoints/Wan2.2-TI2V-5B-Diffusers \
    --stats_path assets_visrobot01/norm_stats_vis.json --val_root <val_dataset> \
    --t5_pkl <val>/t5_embedding/episode_000000.pt --n_clips 50 --blocks 4 --closed-loop \
    [--poll 120] [--once] [--smoke]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch


# ----------------------------- metrics -----------------------------
def _to_uint8_thwc(vid: torch.Tensor) -> np.ndarray:
    """pipeline 视频输出 -> (T,H,W,C) uint8。输入约定 (C,T,H,W) in [-1,1] 或 (T,C,H,W)。"""
    v = vid.detach().float().cpu()
    if v.ndim == 4 and v.shape[0] in (1, 3):      # (C,T,H,W)
        v = v.permute(1, 2, 3, 0)
    elif v.ndim == 4 and v.shape[1] in (1, 3):    # (T,C,H,W)
        v = v.permute(0, 2, 3, 1)
    if v.min() < -0.01:                            # [-1,1] -> [0,1]
        v = (v + 1.0) / 2.0
    v = (v.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return v.numpy()


def video_metrics(pred_thwc: np.ndarray, gt_thwc: np.ndarray, lpips_fn=None, device="cuda") -> dict:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    T = min(len(pred_thwc), len(gt_thwc))
    pred_thwc, gt_thwc = pred_thwc[:T], gt_thwc[:T]
    ps, ss = [], []
    for t in range(T):
        ps.append(psnr(gt_thwc[t], pred_thwc[t], data_range=255))
        ss.append(ssim(gt_thwc[t], pred_thwc[t], channel_axis=2, data_range=255))
    out = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}
    # temporal consistency: 帧间差的接近程度(pred 的运动是否贴合 GT 的运动)
    if T >= 2:
        pd = np.abs(np.diff(pred_thwc.astype(np.float32), axis=0)).mean()
        gd = np.abs(np.diff(gt_thwc.astype(np.float32), axis=0)).mean()
        out["temporal_absdiff_ratio"] = float(pd / (gd + 1e-6))
    if lpips_fn is not None:
        def to_lp(x):
            t = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 127.5 - 1.0
            return t.to(device)
        with torch.no_grad():
            out["lpips"] = float(lpips_fn(to_lp(pred_thwc), to_lp(gt_thwc)).mean().item())
    return out


def _to_thwc_gpu(vid: torch.Tensor, device) -> torch.Tensor:
    """pipeline 视频输出 -> (T,H,W,C) float[0,255],**留在 GPU**(不落 CPU)。(C,T,H,W)/(T,C,H,W) 皆可。"""
    v = vid.detach().to(device=device, dtype=torch.float32)
    if v.ndim == 4 and v.shape[0] in (1, 3):      # (C,T,H,W)
        v = v.permute(1, 2, 3, 0)
    elif v.ndim == 4 and v.shape[1] in (1, 3):    # (T,C,H,W)
        v = v.permute(0, 2, 3, 1)
    if float(v.min()) < -0.01:                     # [-1,1] -> [0,1]
        v = (v + 1.0) / 2.0
    return (v.clamp(0, 1) * 255.0)


def _gauss_win(ws, sigma, device, dtype):
    c = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return g[:, None] * g[None, :]


def video_metrics_gpu(pred_thwc: torch.Tensor, gt_thwc: torch.Tensor, lpips_fn=None, device="cuda") -> dict:
    """全 GPU 视频指标(替代 skimage CPU 路径,消除 SSIM 的 CPU 瓶颈)。
    pred_thwc/gt_thwc: (T,H,W,C) float[0,255] GPU 张量。PSNR=逐帧 MSE→dB;
    SSIM=高斯 11x11 窗(Wang et al,与 skimage gaussian_weights=True 同式,略异于其默认 uniform-7);
    temporal=帧间绝对差比。全部 conv2d/逐元素在 GPU 上,GPU 不再因 CPU 度量而空转。"""
    import torch.nn.functional as F
    T = min(pred_thwc.shape[0], gt_thwc.shape[0])
    p = pred_thwc[:T].permute(0, 3, 1, 2).contiguous()   # (T,C,H,W)
    g = gt_thwc[:T].permute(0, 3, 1, 2).contiguous()
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
        pd = (p[1:] - p[:-1]).abs().mean(); gd = (g[1:] - g[:-1]).abs().mean()
        out["temporal_absdiff_ratio"] = float((pd / (gd + 1e-6)).item())
    if lpips_fn is not None:
        with torch.no_grad():
            out["lpips"] = float(lpips_fn(p / 127.5 - 1.0, g / 127.5 - 1.0).mean().item())
    return out


# ----------------------------- window-index enumeration -----------------------------
def build_window_indices(val_root: str, coverage: str, stride: int, action_chunk: int, exec_horizon: int = 16):
    """枚举 held-out 全部 episode 的窗口全局索引(与 LatentEpisodeSampler/dataset 同一映射:
    global_idx = 累计长度 gs + 集内起始 s)。集内起始按 stride 取,stride 由 coverage 决定:
      coverage='episode'(A): stride=action_chunk(48)  → 非重叠块,全集覆盖(~6k 窗口,~5min/ckpt)。
      coverage='exec'(C,默认): stride=exec_horizon(默认8) → **部署实际执行步长**:机器人每执行
                               ~8 步就重规划/接新块,按此步长采样 = 复现部署 query 节奏,每窗口误差≈每次
                               推理误差(默认 8→~3.7万窗口~33min/ckpt;RTC 16/V1 12 更省)。
      coverage='frames'(B):  stride=1                 → 每帧一窗口,全量(~29 万,仅终评 ~4h)。
    起始上限取 L-action_chunk(保证有完整 action_chunk 步未来 GT,不触发 padding)。
    返回 (idxs, gs_total, info):info[gi]=(episode_index, frame_in_episode) —— 用于按 episode 排序
    (解码局部性)+ 从 episode 帧缓存取帧。gs_total 应等于 len(val_ds)。
    """
    eps = [json.loads(l) for l in open(os.path.join(val_root, "meta", "episodes.jsonl")) if l.strip()]
    eps = sorted(eps, key=lambda e: int(e["episode_index"]))  # jsonl 顺序即 dataset 帧顺序
    _auto = {"episode": action_chunk, "exec": exec_horizon, "frames": 1}.get(coverage, exec_horizon)
    st = stride if stride and stride > 0 else _auto
    idxs, gs, info = [], 0, {}
    for e in eps:
        ei = int(e["episode_index"]); L = int(e["length"])
        last = max(1, L - action_chunk)            # 需要 action_chunk 步未来
        for s in range(0, last, st):
            gi = gs + s; idxs.append(gi); info[gi] = (ei, s)
        gs += L
    return idxs, gs, info


# ----------------------------- per-episode 帧缓存(消除随机seek解码瓶颈) -----------------------------
class EpisodeFrameCache:
    """每个 episode 的 3 路 mp4 **顺序解码一次**(h264 顺序解码快、无 seek),缓存全部帧(LRU),
    供该集所有窗口按帧索引。把 ~5s/窗口的随机seek解码摊销成 ~0.1s/窗口 → eval 转为 GPU-bound。
    需配合「按 (episode,frame) 排序窗口」让每集只解一次。"""
    def __init__(self, val_root, view_keys, cache_size=2):
        self.root = val_root; self.vk = view_keys; self.cap = max(1, int(cache_size))
        self.cache = {}; self.order = []
        self._vdir = os.path.join(val_root, "videos")

    def _vpath(self, cam, ep):
        for chunk in sorted(glob.glob(os.path.join(self._vdir, "*"))):
            p = os.path.join(chunk, cam, f"episode_{ep:06d}.mp4")
            if os.path.isfile(p):
                return p
        return None

    def get(self, ep):
        if ep in self.cache:
            return self.cache[ep]
        import av
        frames = {}
        for cam in self.vk:
            p = self._vpath(cam, ep)
            c = av.open(p)
            frames[cam] = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])  # [L,H,W,C] uint8
            c.close()
        self.cache[ep] = frames; self.order.append(ep)
        if len(self.order) > self.cap:
            self.cache.pop(self.order.pop(0), None)
        return frames


def _hwc_to_chw01(fr: np.ndarray) -> torch.Tensor:
    """缓存帧 [H,W,C] uint8 -> CHW float[0,1](build_ref_image 期望)。"""
    return torch.from_numpy(fr).permute(2, 0, 1).float() / 255.0


# ----------------------------- checkpoint discovery -----------------------------
def list_checkpoints(output_dir: str):
    found = []
    for d in glob.glob(os.path.join(output_dir, "**", "checkpoint_*_step_*"), recursive=True):
        m = re.search(r"step_(\d+)", os.path.basename(d))
        if not m:
            continue
        sub = None
        for cand in ("transformer_ema", "transformer"):
            if os.path.isdir(os.path.join(d, cand)) and glob.glob(os.path.join(d, cand, "*.safetensors")) + glob.glob(os.path.join(d, cand, "*.bin")):
                sub = os.path.join(d, cand); break
        if sub:
            found.append((int(m.group(1)), d, sub))
    return sorted(found)


# ----------------------------- one checkpoint eval -----------------------------
def eval_checkpoint(transformer_dir, args, val_ds, clip_idxs, info, framecache, norm, t5, lpips_fn, delta_mask, device, dtype):
    """评 clip_idxs(全局窗口索引列表)上的全部指标,返回 (sums, counts, n_done):
    sums[metric]=Σ value、counts[metric]=#samples(逐指标,horizon 因 L 不同可能缺项)。
    上层(单机或聚合器)用 Σ/Σcount 得均值 —— 这样分片可无损合并。
    图像帧走 framecache(每集顺序解码一次),val_ds 只取 state/action(skip_video_decoding,快)。
    info[gi]=(episode, frame_in_episode)。clip_idxs 已按 (episode,frame) 排序 → 帧缓存每集只解一次。
    """
    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
    from world_action_model.pipeline.wa_pipeline import WAPipeline
    from world_action_model.pipeline.utils import add_state_to_action, build_ref_image, denormalize_action, normalize_state
    from diffusers.models import AutoencoderKLWan

    # action 横向 horizon:第 1 / 10 / chunk的一半 / chunk末步(1-indexed),"max"=chunk 长度
    horizons = sorted({h for h in (1, 10, args.action_chunk // 2, args.action_chunk) if 1 <= h <= args.action_chunk})

    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    transformer = CasualWorldActionTransformer.from_pretrained(transformer_dir).to(dtype)
    pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=transformer, torch_dtype=dtype).to(device)

    view_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    sums, counts = {}, {}
    import numpy as _np
    ck = args.action_chunk
    gt_offsets = [0, ck // 4, ck // 2, 3 * ck // 4, ck]      # GT 帧相对窗口起点(=训练 delta_frames)
    for i, gi in enumerate(clip_idxs):
        d = val_ds[int(gi)]                                 # state + action chunk(skip_video_decoding,无解码)
        ep, f = info[int(gi)]
        ep_frames = framecache.get(ep)                      # {cam:[L,H,W,C] uint8},每集顺序解码一次
        Lf = ep_frames[view_keys[0]].shape[0]
        # 首帧(窗口起点 f)3 视角拼成 ref(PIL,768x192)喂给 pipe
        ref = build_ref_image(images={k: _hwc_to_chw01(ep_frames[k][f]) for k in view_keys},
                              dst_size=(args.width, args.height), crop_mode="center")
        state = d["observation.state"].float().unsqueeze(0).to(device)   # norm 张量在 cuda,state 须同设备
        nstate = normalize_state(state, norm, mode="zscore").to(device=device, dtype=dtype)
        with torch.no_grad():
            imgs, action = pipe(height=args.height, width=args.width, action_chunk=args.action_chunk,
                                state=nstate, num_frames=args.num_frames, guidance_scale=0.0,
                                num_inference_steps=args.steps, image=ref, action_only=False,
                                return_dict=False, prompt_embeds=t5.unsqueeze(0).to(device=device, dtype=torch.float32))
        pred_t = _to_thwc_gpu(imgs[0], device)          # (T,H,W,C) float[0,255] 留 GPU
        # GT: 在 f+gt_offsets 处把 3 视角拼成 768x192(与 pred 同空间),堆 (T,H,W,C) uint8 → 推 GPU
        gt = _np.stack([_np.array(build_ref_image(
                            images={k: _hwc_to_chw01(ep_frames[k][min(f + off, Lf - 1)]) for k in view_keys},
                            dst_size=(args.width, args.height), crop_mode="center"))
                        for off in gt_offsets], axis=0)
        gt_t = torch.from_numpy(gt).to(device=device, dtype=torch.float32)
        m = video_metrics_gpu(pred_t, gt_t, lpips_fn=lpips_fn, device=device)   # PSNR/SSIM/temporal 全 GPU
        # action:反归一化 + add_state(与 inference_server 一致)-> 真实单位,再算逐 horizon MAE
        pred_act = denormalize_action(action[0].float(), norm, mode="zscore")
        pred_act = add_state_to_action(pred_act, state[0].float().to(pred_act.device),
                                       action_chunk=args.action_chunk, mask=delta_mask).cpu().numpy()
        gt_act = d["action"].float().numpy()
        L = min(len(pred_act), len(gt_act))
        ae = np.abs(pred_act[:L] - gt_act[:L])             # [L,14] 真实单位绝对误差
        m["action_mae"] = float(ae.mean())                 # 全 chunk 均值
        m["action_mse"] = float(((pred_act[:L] - gt_act[:L]) ** 2).mean())
        for h in horizons:                                  # mae@h:第 h 步(1-indexed)所有维均值
            if h <= L:
                m[f"mae@{h}"] = float(ae[h - 1].mean())
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
        if args.save_mp4 and i < args.n_mp4:           # 仅存图时才把 pred 落 CPU(np uint8)
            pred_np = pred_t.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            _save_side_by_side(pred_np, gt, os.path.join(args.vis_dir, f"clip{i:03d}.mp4"), fps=args.fps)
    del pipe, transformer, vae
    torch.cuda.empty_cache()
    return sums, counts, len(clip_idxs)


def _save_side_by_side(pred_thwc, gt_thwc, path, fps=5):
    try:
        import torchvision
        T = min(len(pred_thwc), len(gt_thwc))
        cat = np.concatenate([gt_thwc[:T], pred_thwc[:T]], axis=2)  # 左GT|右Pred
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchvision.io.write_video(path, torch.from_numpy(cat), fps=fps)
    except Exception as e:
        print(f"[eval] mp4 save failed: {e}")


# ----------------------------- sharding / aggregation -----------------------------
def _means(sums: dict, counts: dict) -> dict:
    return {k: sums[k] / counts[k] for k in sums if counts.get(k, 0) > 0}


def _shard_path(output_dir, step, shard_id):
    return os.path.join(output_dir, "eval_shards", f"step_{step}", f"shard_{shard_id:02d}.json")


def write_record(output_dir, step, ckdir, metrics, extra, tb=None):
    rec = {"step": step, "ckpt": ckdir, **extra, **metrics}
    with open(os.path.join(output_dir, "eval_log.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    if tb is not None:
        for k, v in metrics.items():
            tb.add_scalar(f"eval/{k}", v, step)
    return rec


def aggregate_shards(output_dir, step, expect=None, tb=None):
    """合并 eval_shards/step_<step>/shard_*.json(各含 sums/counts/n)→ 均值记录写 eval_log.jsonl。
    expect=期望 shard 数(不足则报警但仍合并已有)。返回 (rec, n_shards)。"""
    sdir = os.path.join(output_dir, "eval_shards", f"step_{step}")
    files = sorted(glob.glob(os.path.join(sdir, "shard_*.json")))
    if not files:
        return None, 0
    sums, counts, n_tot, cov, ckdir = {}, {}, 0, None, None
    for fp in files:
        s = json.load(open(fp))
        cov = cov or s.get("coverage"); ckdir = ckdir or s.get("ckpt")
        n_tot += int(s.get("n", 0))
        for k, v in s.get("sums", {}).items():
            sums[k] = sums.get(k, 0.0) + float(v)
        for k, v in s.get("counts", {}).items():
            counts[k] = counts.get(k, 0) + int(v)
    if expect and len(files) < expect:
        print(f"[agg] WARNING step {step}: only {len(files)}/{expect} shards present")
    rec = write_record(output_dir, step, ckdir, _means(sums, counts),
                        {"coverage": cov, "n_windows": n_tot, "n_shards": len(files)}, tb=tb)
    return rec, len(files)


# ----------------------------- main loop -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="训练 project_dir(轮询 checkpoint)")
    ap.add_argument("--model_id", required=True, help="Wan2.2-TI2V-5B-Diffusers(取 vae/config)")
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--val_root", required=True,
                    help="held-out 验证集根目录(visrobot01_val,split_heldout.py 切出的自洽 200 集)")
    ap.add_argument("--t5_pkl", required=True)
    ap.add_argument("--coverage", choices=["episode", "exec", "frames", "sample"], default="exec",
                    help="C=exec(默认,部署RTC执行步长16采样~1.8万窗口~16min); A=episode(非重叠块~6k~5min); "
                         "B=frames(每帧~29万,仅终评~4h); sample=跨全集均匀取 n_clips 个(便宜单卡监控)")
    ap.add_argument("--exec_horizon", type=int, default=16,
                    help="coverage=exec 的采样步长=部署 RTC 实际执行步长(默认 16=session_launch rtc_execute_horizon;"
                         "→~1.8万窗口,约为 8 的一半评估量)")
    ap.add_argument("--stride", type=int, default=0,
                    help="集内窗口步长(0=按 coverage 自动:episode→48, exec→exec_horizon, frames→1)")
    ap.add_argument("--n_clips", type=int, default=200,
                    help="仅 coverage=sample 用:跨全部 held-out 集均匀取样的窗口数(≈1/episode)")
    # ---- 分布式分片(b1+b2 16 卡):worker 评一个 shard 写 partial;aggregate 合并 ----
    ap.add_argument("--ckpt_subdir", default=None, help="worker 模式:只评这个 transformer 目录(配 --step)")
    ap.add_argument("--step", type=int, default=-1, help="worker/aggregate 模式的 ckpt step")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1, help=">1 时为分片 worker;窗口按 shard_id::num_shards 取")
    ap.add_argument("--aggregate", action="store_true", help="聚合模式:合并 eval_shards/step_<step>/shard_*.json")
    ap.add_argument("--list", action="store_true",
                    help="列出未评估的 ckpt(每行 'step<TAB>transformer_subdir'),供 orchestrator 分发;无需 GPU")
    ap.add_argument("--n_mp4", type=int, default=3)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--action_chunk", type=int, default=48)
    ap.add_argument("--num_frames", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--t5_len", type=int, default=64)
    ap.add_argument("--state_dim", type=int, default=14)
    ap.add_argument("--action_dim", type=int, default=14)
    ap.add_argument("--delta_mask", type=str, default="1,1,1,1,1,1,0,1,1,1,1,1,1,0",
                    help="14维 piper:关节 delta、夹爪(idx6/13)绝对;与 inference_server 一致")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--once", action="store_true", help="只评当前最新 ckpt 一次后退出")
    ap.add_argument("--smoke", action="store_true", help="不评 ckpt,只在 GT 上自测指标函数")
    ap.add_argument("--no_lpips", action="store_true", help="跳过 LPIPS(离线无 alex 权重时)")
    ap.add_argument("--save_mp4", action="store_true", default=True)
    ap.add_argument("--vis_dir", default=None)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()
    args.vis_dir = args.vis_dir or os.path.join(args.output_dir, "eval_vis")

    # ---- 聚合模式:无需 GPU/模型,合并分片 partial → eval_log.jsonl(由 orchestrator 在 16 worker 完成后调) ----
    if args.aggregate:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(os.path.join(args.output_dir, "eval_tb"))
        except Exception:
            tb = None
        rec, n = aggregate_shards(args.output_dir, args.step, expect=args.num_shards, tb=tb)
        print(f"[agg] step {args.step}: merged {n} shards -> {rec}" if rec else f"[agg] step {args.step}: no shards")
        return

    # ---- 列表模式:打印未评估 ckpt(step<TAB>subdir),供 orchestrator 分发;无需 GPU ----
    if args.list:
        log_path = os.path.join(args.output_dir, "eval_log.jsonl")
        done = set()
        if os.path.exists(log_path):
            for l in open(log_path):
                try: done.add(int(json.loads(l)["step"]))
                except Exception: pass
        for step, ckdir, subdir in list_checkpoints(args.output_dir):
            if step not in done:
                print(f"{step}\t{subdir}")
        return

    device = "cuda"; dtype = torch.bfloat16
    from world_action_model.pipeline.utils import extract_normalization_tensors, load_stats, load_t5_embedding_from_pkl
    lpips_fn = None
    if not args.no_lpips:
        try:                       # LPIPS 需下载 alex 权重;离线/无网时优雅跳过(PSNR/SSIM/temporal 仍算)
            import lpips as lpips_mod
            lpips_fn = lpips_mod.LPIPS(net="alex").to(device).eval()
        except Exception as e:
            print(f"[eval] LPIPS unavailable ({type(e).__name__}); skipping lpips, keep PSNR/SSIM/temporal")

    if args.smoke:
        gt = (np.random.rand(5, 192, 768, 3) * 255).astype(np.uint8)
        print("[smoke] GT vs GT:", video_metrics(gt, gt, lpips_fn, device))
        noisy = np.clip(gt.astype(int) + np.random.randint(-40, 40, gt.shape), 0, 255).astype(np.uint8)
        print("[smoke] GT vs noisy:", video_metrics(noisy, gt, lpips_fn, device))
        print("SMOKE_OK"); return

    from giga_datasets import load_dataset
    view_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    # val_ds 只取 state/action(skip_video_decoding=True,不解 mp4 → 快);图像帧改走 EpisodeFrameCache
    # (每集顺序解码一次)。消除原 ~5s/窗口的随机seek解码瓶颈,eval 转为 GPU-bound。
    val_entry = dict(_class_name="LeRobotDataset", data_path=args.val_root,
                     delta_info={"action": args.action_chunk}, skip_video_decoding=True,
                     embodiment="visrobot01", tolerance_s=1e-3)
    val_ds = load_dataset([val_entry])
    framecache = EpisodeFrameCache(args.val_root, view_keys, cache_size=2)
    stats = load_stats(args.stats_path)
    norm = extract_normalization_tensors(stats, device=device, state_dim=args.state_dim, action_dim=args.action_dim)
    t5 = load_t5_embedding_from_pkl(args.t5_pkl, target_len=args.t5_len).to(device=device, dtype=torch.float32)
    delta_mask = torch.tensor([p.strip() in ("1", "true", "True") for p in args.delta_mask.split(",") if p.strip()],
                              device=device, dtype=torch.bool)
    assert delta_mask.numel() == args.action_dim, f"delta_mask {delta_mask.numel()} != action_dim {args.action_dim}"

    # ---- 构建窗口索引(coverage 决定 stride),按需取本 shard ----
    all_idxs, gs_total, info = build_window_indices(args.val_root, args.coverage, args.stride,
                                                    args.action_chunk, args.exec_horizon)
    if gs_total != len(val_ds):
        print(f"[eval] WARNING: cum-length {gs_total} != len(val_ds) {len(val_ds)} (索引映射可能错位)")
    if args.coverage == "sample":     # 便宜单卡监控:跨全集均匀取 n_clips 个
        n = min(args.n_clips, len(all_idxs))
        all_idxs = [all_idxs[i] for i in np.unique(np.linspace(0, len(all_idxs) - 1, n).astype(int))]
    # 分片按 **episode** 分配(非按窗口轮询):每片拿若干完整 episode → 帧缓存每集只解一次,
    # 解码摊销到该集全部窗口(~5s/集 / ~56窗口 ≈ 0.09s/窗口)。轮询窗口会打散 episode、毁掉缓存。
    if args.num_shards > 1:
        eps_all = sorted({info[gi][0] for gi in all_idxs})
        my_eps = set(eps_all[args.shard_id::args.num_shards])
        my_idxs = [gi for gi in all_idxs if info[gi][0] in my_eps]
    else:
        my_idxs = list(all_idxs)
    my_idxs = sorted(my_idxs, key=lambda gi: info[gi])     # 按 (episode,frame) 排序 → 帧缓存每集只解一次
    print(f"[eval] coverage={args.coverage} total_windows={len(all_idxs)} "
          f"shard {args.shard_id}/{args.num_shards} -> {len(my_idxs)} windows "
          f"({len({info[gi][0] for gi in my_idxs})} episodes)")

    # ---- worker 模式:只评一个 ckpt 的本 shard,写 partial(sums/counts/n),由聚合器合并 ----
    if args.ckpt_subdir is not None:
        t0 = time.time()
        sums, counts, n = eval_checkpoint(args.ckpt_subdir, args, val_ds, my_idxs, info, framecache,
                                          norm, t5, lpips_fn, delta_mask, device, dtype)
        sp = _shard_path(args.output_dir, args.step, args.shard_id)
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        tmp = sp + ".tmp"
        json.dump({"step": args.step, "shard_id": args.shard_id, "num_shards": args.num_shards,
                   "coverage": args.coverage, "ckpt": os.path.dirname(args.ckpt_subdir),
                   "n": n, "secs": round(time.time() - t0, 1), "sums": sums, "counts": counts}, open(tmp, "w"))
        os.replace(tmp, sp)
        print(f"[eval] shard {args.shard_id} step {args.step} done: {n} windows, {round(time.time()-t0,1)}s -> {sp}")
        return

    # ---- 单卡 watch 模式(num_shards=1):轮询新 ckpt,本机评全部 my_idxs,本地聚合写记录 ----
    log_path = os.path.join(args.output_dir, "eval_log.jsonl")
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(args.output_dir, "eval_tb"))
    except Exception:
        tb = None
    done = set()
    if os.path.exists(log_path):
        for l in open(log_path):
            try: done.add(int(json.loads(l)["step"]))
            except Exception: pass

    print(f"[eval] watching {args.output_dir} (poll={args.poll}s, coverage={args.coverage})")
    while True:
        cks = [c for c in list_checkpoints(args.output_dir) if c[0] not in done]
        if not cks and args.once:
            print("[eval] no new checkpoint"); break
        for step, ckdir, subdir in cks:
            t0 = time.time()
            print(f"[eval] step {step}: {subdir}")
            try:
                sums, counts, n = eval_checkpoint(subdir, args, val_ds, my_idxs, info, framecache, norm, t5, lpips_fn, delta_mask, device, dtype)
            except Exception as e:
                print(f"[eval] step {step} FAILED: {e}"); done.add(step); continue
            write_record(args.output_dir, step, ckdir, _means(sums, counts),
                         {"coverage": args.coverage, "n_windows": n, "secs": round(time.time() - t0, 1)}, tb=tb)
            print(f"[eval] step {step} -> {_means(sums, counts)}")
            done.add(step)
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
