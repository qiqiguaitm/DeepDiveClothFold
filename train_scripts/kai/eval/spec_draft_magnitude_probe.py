#!/usr/bin/env python3
"""真机现象诊断: 带 spec 动得慢/欠完成但**轨迹趋势对**, --no-spec 正常 → 量化 draft 是否**幅值衰减**.

假设: 1 层 draft 用回归 (huber/mse) 蒸馏 → 预测**条件均值** → 方向对、**幅值被压缩** (mean
regression)。radius 接受查不出, 因为 verify-from-draft **锚在 draft 上** (x_t=t·noise+(1-t)·x0_draft,
一步去噪 → x0_hat≈x0_draft), 衰减的 draft 自洽地被接受 (50/50) —— 又一个自洽盲点 (§8/§9/§12)。

每帧同噪声比两条输出 (都在模型 normalized 空间, arm 非夹爪维):
  • spec  = sampler.sample(obs)["actions"]            (50/50 接受时即 draft chunk; 机器人 spec 路径所见)
  • full  = sampler.full_denoise_from_observation(obs) (eager 10 步; --no-spec 路径所见)
指标 (逐 chunk, 在 H 步轨迹上):
  • motion = sum_arm (max-min over H)  —— 轨迹动态范围 (越大动得越多)
  • ratio  = motion_spec / motion_full  —— <1 即 spec 幅值衰减 (动得慢的量化)
  • cos    = cosine(spec 去均值, full 去均值)  —— 接近 1 即"趋势对"
  • 逐维 ratio —— 看哪些关节被压得最狠

纯离线、加性, 复用 R1-d sampler + draft, 不改旧码。

Run:
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090/bin/python train_scripts/kai/eval/spec_draft_magnitude_probe.py \
    --config pi05_pytorch_a_new_pure_200 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_others/pytorch_pure200_step50000 \
    --asset-id a_new_pure_200 --draft /tmp/draft_r1d_pure200.pt \
    --val /data1/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val \
    --holdout-eps 4 --frames-per-ep 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from spec_draft_r1d import read_video


def _arm_dims(action_dim, gripper_dims):
    grip = {int(g) for g in gripper_dims}
    return [i for i in range(min(14, action_dim)) if i not in grip]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--holdout-eps", type=int, default=4)
    ap.add_argument("--frames-per-ep", type=int, default=40)
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--tau", type=float, default=0.3)
    args = ap.parse_args()

    import jax

    from openpi.models import model as _model
    from openpi.models_pytorch.draft import DraftChunkHead
    from openpi.models_pytorch.spec_pi0_pytorch import SpecArgs
    from openpi.models_pytorch.spec_pi0_pytorch import SpeculativeSampler
    from openpi.policies import policy_config as pc
    from openpi.training import checkpoints as ck
    from openpi.training import config as tc

    ckpt = Path(args.ckpt).resolve()
    train_cfg = tc.get_config(args.config)
    norm_stats = ck.load_norm_stats(ckpt / "assets", args.asset_id)
    policy = pc.create_trained_policy(train_cfg, ckpt, norm_stats=norm_stats)
    model = policy._model  # noqa: SLF001
    device = next(model.parameters()).device
    mdtype = next(model.parameters()).dtype
    ah, ad = int(model.config.action_horizon), int(model.config.action_dim)

    spec_args = SpecArgs(chunk_m=ah, tau_radius=args.tau, max_exec_steps=ah, full_num_steps=10)
    sampler = SpeculativeSampler(model, None, spec_args)
    vlm = model.paligemma_with_expert.paligemma.language_model
    blob = torch.load(args.draft, map_location=device)
    draft = DraftChunkHead(img_dim=int(blob["img_dim"]), chunk_m=int(blob["chunk_m"]),
                           out_dim=int(blob["out_dim"]), use_state_token=False,
                           num_layers=int(blob.get("num_layers", 1)), gemma_config=vlm.config)
    draft.load_state_dict(blob["state_dict"])
    sampler.draft = draft.to(device, mdtype).eval()

    arm = _arm_dims(ad, spec_args.gripper_dims)
    cams = ("top_head", "hand_left", "hand_right")
    val = Path(args.val).resolve()
    eps = [json.loads(line) for line in (val / "meta" / "episodes.jsonl").read_text().strip().split("\n")]
    hold = eps[: args.holdout_eps]

    n_mc = 4  # MC 噪声数: full_mean = 模型自身均值轨迹 (区分"draft 衰减" vs "执行均值≠采样")
    g = torch.Generator(device=device).manual_seed(0)
    noises = [torch.randn((1, ah, ad), generator=g, device=device, dtype=torch.float32) for _ in range(n_mc)]
    n1 = noises[0]

    def build_obs(imgs, state_row):
        inputs = policy._input_transform({"images": imgs, "state": state_row, "prompt": args.prompt})  # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def _motion(chunk):  # (H,|arm|) → 逐维 max-min 之和 (轨迹动态范围)
        return float((chunk.max(0) - chunk.min(0)).sum())

    def _cos_demean(x, y):
        a = (x - x.mean(0)).ravel()
        b = (y - y.mean(0)).ravel()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 1e-9 else np.nan

    import pyarrow.parquet as pq
    accepts = []
    r_vs_sample, r_vs_mean, r_sample_vs_mean = [], [], []  # spec/full(n1), spec/full_mean, full(n1)/full_mean
    cos_vs_sample, cos_vs_mean = [], []
    pd_spec = np.zeros(len(arm))
    pd_sample = np.zeros(len(arm))
    pd_mean = np.zeros(len(arm))
    with torch.no_grad():
        for ep in hold:
            ei, nf = ep["episode_index"], ep["length"]
            tbl = pq.read_table(val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet").to_pandas()
            state = np.stack([np.asarray(x) for x in tbl["observation.state"]]).astype(np.float32)
            vid = {c: read_video(val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", nf)
                   for c in cams}
            for k in np.linspace(0, nf - 1, args.frames_per_ep).astype(int):
                obs = build_obs({c: vid[c][int(k)] for c in cams}, state[int(k)])
                out = sampler.sample(obs, noise=n1, last_gripper=None)
                spec = out["actions"][0, :, arm].float().cpu().numpy()
                accepts.append(int(out["accepted_prefix_len"].item()))
                fulls = [sampler.full_denoise_from_observation(obs, noise=nz)[0, :, arm].float().cpu().numpy()
                         for nz in noises]
                fsample = fulls[0]                       # 单噪声样本 (= --no-spec 路径所见之一)
                fmean = np.mean(fulls, axis=0)           # 模型自身均值轨迹
                pd_spec += spec.max(0) - spec.min(0)
                pd_sample += fsample.max(0) - fsample.min(0)
                pd_mean += fmean.max(0) - fmean.min(0)
                ms = _motion(spec)
                r_vs_sample.append(ms / max(_motion(fsample), 1e-9))
                r_vs_mean.append(ms / max(_motion(fmean), 1e-9))
                r_sample_vs_mean.append(_motion(fsample) / max(_motion(fmean), 1e-9))
                cos_vs_sample.append(_cos_demean(spec, fsample))
                cos_vs_mean.append(_cos_demean(spec, fmean))

    accepts = np.array(accepts)
    r_vs_sample = np.array(r_vs_sample)
    r_vs_mean = np.array(r_vs_mean)
    r_sample_vs_mean = np.array(r_sample_vs_mean)
    agg_vs_sample = pd_spec.sum() / max(pd_sample.sum(), 1e-9)   # 聚合比 (抗 per-frame 离群)
    agg_vs_mean = pd_spec.sum() / max(pd_mean.sum(), 1e-9)
    print("\n========== draft/spec 幅值诊断 (pure200, MC=%d) ==========" % n_mc)
    print(f"  frames={len(accepts)}  arm_dims={arm}  accept(spec)={accepts.mean():.1f}/{ah}")
    print("  motion ratio (聚合 sum_dim, 抗离群) | (per-frame median):")
    print(f"    spec / full(单样本)  = {agg_vs_sample:.3f}  | median {np.median(r_vs_sample):.3f}  "
          f"← 机器人 spec 路径 vs --no-spec 路径")
    print(f"    spec / full(均值轨迹)= {agg_vs_mean:.3f}  | median {np.median(r_vs_mean):.3f}  "
          f"← draft 是否忠实于模型**自身中心趋势**")
    print(f"    full(单样本)/均值轨迹= {pd_sample.sum()/max(pd_mean.sum(),1e-9):.3f} | median "
          f"{np.median(r_sample_vs_mean):.3f}  ← 采样相对均值的固有摆幅")
    print(f"  趋势 cosine: spec vs 样本 median={np.nanmedian(cos_vs_sample):.3f}  "
          f"spec vs 均值 median={np.nanmedian(cos_vs_mean):.3f}")
    print("  逐 arm 维聚合 motion (spec / sample / mean):")
    for j, d in enumerate(arm):
        print(f"    dim {d:2d}: spec={pd_spec[j]:7.2f}  sample={pd_sample[j]:7.2f}  mean={pd_mean[j]:7.2f}  "
              f"(spec/sample={pd_spec[j]/max(pd_sample[j],1e-9):.2f}, spec/mean={pd_spec[j]/max(pd_mean[j],1e-9):.2f})")
    print("  -- verdict --")
    if agg_vs_mean < 0.8:
        print(f"     ✅ draft **比模型自身均值轨迹还小** (spec/mean={agg_vs_mean:.2f}) → draft 本身欠拟合/过平滑, "
              "不只是'均值 vs 采样'。蒸馏问题: 加幅值/速度损失、分位匹配、或更大 draft。")
    elif agg_vs_sample < 0.8 <= agg_vs_mean:
        print(f"     draft 忠实于模型均值 (spec/mean={agg_vs_mean:.2f}) 但**均值轨迹本就比采样小** "
              f"(spec/sample={agg_vs_sample:.2f}) → 真机慢的根因是**执行确定性均值 (draft/spec) 欠了采样摆幅**;")
        print("     --no-spec 采样把摆幅带回来 → 能完成。修向: draft 输出乘速度增益, 或 spec 路径保留少量采样性 (verify 不锚 draft)。")
    else:
        print(f"     spec/sample={agg_vs_sample:.2f} spec/mean={agg_vs_mean:.2f} 均不低 → 幅值非主因; "
              "看 cosine/逐维或 RTC/发布率/IK。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
