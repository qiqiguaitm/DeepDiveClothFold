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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent          # fastwam/
_WS_DIR = _REPO_DIR.parent             # deepdive_kai0/
sys.path.insert(0, str(_REPO_DIR / "src"))
# v1 默认;v3 评测用 env 覆盖(EVAL_VAL_ROOT / EVAL_VIEW_KEYS / EVAL_DATA / EVAL_TASK / EVAL_TEXT_EMB)。
# VK 按角色顺序 [top, left_wrist, right_wrist] —— prep_image 按位置取,不依赖具体相机名。
VAL = os.environ.get("EVAL_VAL_ROOT") or str(_WS_DIR / "kai0" / "data" / "wam_fold_v1" / "visrobot01_val")
VK = (os.environ.get("EVAL_VIEW_KEYS") or "cam_high,cam_left_wrist,cam_right_wrist").split(",")
HOR = [1, 10, 24, 48]


def build_model(weights, device="cuda"):
    from hydra import compose, initialize
    from hydra.utils import instantiate
    _data = os.environ.get("EVAL_DATA", "visrobot01_fold")
    _task = os.environ.get("EVAL_TASK", "visrobot01_fold_uncond_1e-4")
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train",
                      overrides=[f"data={_data}", "model=fastwam", f"task={_task}",
                                 "model.skip_dit_load_from_pretrain=true"])  # skip Wan2.2 backbone weights (overwritten by ckpt anyway)
    model = instantiate(cfg.model)  # create_fastwam — architecture only, no pretrained weights
    sd = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    # trainer 落盘结构:{"mot": mixtures.video/action.* 全量, "proprio_encoder": ..., "step", "torch_dtype"}
    model.mot.load_state_dict(sd["mot"], strict=True)
    if "proprio_encoder" in sd and getattr(model, "proprio_encoder", None) is not None:
        model.proprio_encoder.load_state_dict(sd["proprio_encoder"], strict=True)
    print(f"[load] mot keys={len(sd['mot'])} step={sd.get('step')} strict OK", flush=True)
    return model


def prep_image(frames_by_cam):
    """严格复刻训练像素链(与 compute_latents.window_pixels 一致):
    per-cam ToTensor(/255)+Resize(240,320) → top 256x320 / 腕 128x160 → 拼 [3,384,320]。
    输入: {cam: HxWx3 uint8};输出 [3,384,320] float [0,1](infer_action 内部转 [-1,1]?
    —— 不,VAE 期望 [-1,1],infer_action 直接 encode → 调用方给 [-1,1])。"""
    t = {}
    for k, v in frames_by_cam.items():
        x = torch.from_numpy(v).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t[k] = TF.resize(x, [240, 320], antialias=True)[0]
    top = TF.resize(t[VK[0]].unsqueeze(0), [256, 320], antialias=True)[0]   # VK[0]=top (cam_high / top_head)
    lf = TF.resize(t[VK[1]].unsqueeze(0), [128, 160], antialias=True)[0]    # VK[1]=left wrist
    rt = TF.resize(t[VK[2]].unsqueeze(0), [128, 160], antialias=True)[0]    # VK[2]=right wrist
    img = torch.cat([top, torch.cat([lf, rt], dim=-1)], dim=-2)  # [3,384,320] in [0,1]
    return img * 2.0 - 1.0  # [-1,1],对齐训练 Normalize((x-0.5)/0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights"); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_id", type=int, default=0); ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--n_metric_eps", type=int, default=200)
    ap.add_argument("--nfe", type=int, default=20)
    ap.add_argument("--joint", action="store_true", help="用 infer_joint(带视频想象)替代 infer_action")
    ap.add_argument("--engine", default="stock", choices=["stock", "opt"],
                    help="stock=model.infer_action  opt=opt_infer_action(compile+可选FP8)")
    ap.add_argument("--opt_tier", default="exact", choices=["eager", "exact", "fp8"],
                    help="opt 引擎的加速档位(仅 --engine opt 生效)")
    ap.add_argument("--stats", default=str(_REPO_DIR / "data" / "visrobot01_fold" / "dataset_stats.json"))
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
        aggc = {f"cmae@{h}": float(np.mean([v[f"cmae@{h}"] for v in eps.values() if v.get(f"cmae@{h}") is not None])) for h in HOR}
        out = {"n_metric_eps": len(eps), "latency": lat,
               "raw_mae": {str(h): agg[f"mae@{h}"] for h in HOR},
               "cum_mae": {str(h): aggc[f"cmae@{h}"] for h in HOR},
               "pi05": {"1": 0.0219, "10": 0.0425, "24": 0.0743, "48": 0.1155}}
        json.dump(out, open(f"{args.out_dir}/summary.json", "w"), indent=1)
        print("[aggregate] single-step mae@: " + " ".join(f"@{h} {agg[f'mae@{h}']:.4f}" for h in HOR)
              + " | cumulative mae@: " + " ".join(f"@{h} {aggc[f'cmae@{h}']:.4f}" for h in HOR)
              + f" | act-lat {lat.get('action_ms',0):.0f}ms", flush=True)
        return

    stats = json.load(open(args.stats))
    a_mean = np.array(stats["action"]["default"]["global_mean"]); a_std = np.array(stats["action"]["default"]["global_std"])
    s_mean = np.array(stats["state"]["default"]["global_mean"]); s_std = np.array(stats["state"]["default"]["global_std"])

    # t5 缓存(单 prompt);v3 用 EVAL_TEXT_EMB=visrobot01_v3_fold
    _te = os.environ.get("EVAL_TEXT_EMB", "visrobot01_fold")
    cache = list((_REPO_DIR / "data" / "text_embeds_cache" / _te).glob("*.pt"))[0]
    t5 = torch.load(cache, map_location="cpu", weights_only=False)
    ctx = t5["context"]                  # [L,D];缓存键 = context/mask(对齐 _get_cached_text_context)
    cmask = t5["mask"].bool()            # [L]
    ctx = ctx.clone(); ctx[~cmask] = 0.0  # 复刻训练约定:padding 清零后 mask 置全 1
    cmask = torch.ones_like(cmask)
    if ctx.ndim == 2: ctx = ctx.unsqueeze(0)
    if cmask.ndim == 1: cmask = cmask.unsqueeze(0)

    model = build_model(args.weights)

    # opt engine setup
    opt_runner = None
    if args.engine == "opt":
        from opt_infer_action import ActionStepRunner, opt_infer_action as _opt_infer
        import sys; gwp_scripts = Path(__file__).resolve().parents[2] / "giga_world_policy" / "scripts"
        sys.path.insert(0, str(gwp_scripts))
        if args.opt_tier == "fp8":
            from opt_infer_action import _swap_fp8
            n, fp8_mode = _swap_fp8(model.action_expert.blocks)
            n2, _ = _swap_fp8(model.action_expert.text_embedding)
            n3, _ = _swap_fp8(model.action_expert.time_embedding)
            n4, _ = _swap_fp8(model.action_expert.time_projection)
            print(f"[fp8/{fp8_mode}] blocks={n} text_emb={n2} time_emb={n3} time_proj={n4}", flush=True)
        opt_runner = ActionStepRunner(model)
        if args.opt_tier in ("exact", "fp8"):
            opt_runner.compile_step("reduce-overhead")
        print(f"[engine] opt tier={args.opt_tier} nfe={args.nfe}", flush=True)

    from torchcodec.decoders import VideoDecoder

    meta = [json.loads(l) for l in open(f"{VAL}/meta/episodes.jsonl")]
    eps = sorted(int(m["episode_index"]) for m in meta)[: args.n_metric_eps]
    my = eps[args.shard_id :: args.num_shards]
    print(f"[shard {args.shard_id}/{args.num_shards}] eps={len(my)} nfe={args.nfe} engine={args.engine} joint={args.joint}", flush=True)

    metric = {}; lat_ms = []
    for ep in my:
        pq = f"{VAL}/data/chunk-000/episode_{ep:06d}.parquet"
        df = pd.read_parquet(pq)
        gt_all = np.stack(df["action"].to_numpy())[:, :14]
        st_all = np.stack(df["observation.state"].to_numpy())[:, :14]
        decs = {k: VideoDecoder(f"{VAL}/videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4") for k in VK}
        L = len(df)
        em = {f"mae@{h}": [] for h in HOR}; emc = {f"mae@{h}": [] for h in HOR}
        for f in range(0, max(1, L - 1), 16):
            frames = {k: decs[k].get_frames_at([min(f, decs[k].metadata.num_frames - 1)]).data[0].permute(1, 2, 0).numpy() for k in VK}
            img = prep_image(frames)
            prop = torch.from_numpy((st_all[f] - s_mean) / (s_std + 1e-8)).float()
            t0 = time.time()
            if opt_runner is not None:
                out = _opt_infer(model, opt_runner,
                                 context=ctx.to(model.device, model.torch_dtype),
                                 context_mask=cmask.to(model.device),
                                 image=img, proprio=prop,
                                 action_horizon=48, num_inference_steps=args.nfe, seed=0)
            else:
                with torch.no_grad():
                    fn = model.infer_joint if args.joint else model.infer_action
                    out = fn(prompt=None, input_image=img, action_horizon=48, proprio=prop,
                             context=ctx.to(model.device, model.torch_dtype),
                             context_mask=None if cmask is None else cmask.to(model.device),
                             num_inference_steps=args.nfe, seed=0)
            lat_ms.append((time.time() - t0) * 1000)
            pa = out["action"].float().cpu().numpy() * (a_std + 1e-8) + a_mean  # infer_action 返回已是 [48,14];z-score 逆 + 全 abs
            gt = gt_all[f : f + 48]
            n = min(len(pa), len(gt)); ae = np.abs(pa[:n] - gt[:n])
            for h in HOR:
                if h <= n:
                    em[f"mae@{h}"].append(float(ae[h - 1].mean()))   # single-step
                    emc[f"mae@{h}"].append(float(ae[:h].mean()))     # cumulative
        metric[ep] = {f"mae@{h}": (float(np.mean(em[f"mae@{h}"])) if em[f"mae@{h}"] else None) for h in HOR}
        metric[ep].update({f"cmae@{h}": (float(np.mean(emc[f"mae@{h}"])) if emc[f"mae@{h}"] else None) for h in HOR})
        print(f"[shard {args.shard_id}] ep{ep} done mae@48={metric[ep]['mae@48']}", flush=True)
    json.dump({"metric": metric, "latency": {"action_ms": float(np.mean(lat_ms[3:]) if len(lat_ms) > 3 else np.mean(lat_ms))}},
              open(f"{args.out_dir}/shards/shard_{args.shard_id}.json", "w"))
    print(f"[shard {args.shard_id}] saved", flush=True)


if __name__ == "__main__":
    main()
