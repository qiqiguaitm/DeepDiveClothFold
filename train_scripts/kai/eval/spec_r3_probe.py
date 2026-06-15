#!/usr/bin/env python3
"""R3: is FLASH radius (||draft-full||) a within-distribution UNCERTAINTY / OOD signal?

R5 (`spec_r3_probe` 的姊妹脚本 spec_r5_probe.py) 已**证伪**"radius 是输入退化探针": 全相机
置黑时 draft 与 full **一起**平移 → radius 几乎不动 (radius 是*自洽*量, 对输入退化结构性失明)。

R3 问一个**互补、不矛盾**的问题: 在**正常 (in-distribution) 帧**之间, radius 会不会随**模型
自身的去噪不确定度**而起伏? 即——
  • 模型自身不确定度地板 floor = mean|teacher(n1) - teacher(n2)|  (同一帧两条噪声的 full 去噪分歧;
    flow-matching 多模态/难帧上 full 自己就不稳 → floor 高);
  • radius = ||draft - full|| 是 draft 追不上 full 的程度。
若 corr(radius, floor) 显著 → **radius 追踪模型自身的内在不确定度** (R3 成立, 与 R5 不冲突:
对*输入退化*盲, 但对*模型内在难度*敏感) → 可作"模型在这帧没把握"的免费旗标 → 触发 DAgger/RLT
(见 dagger/rlt plan)。若 radius 方差≈0 (R1-d 离线饱和) → R3 离线也被压平, 需 off-manifold/闭环帧。

另外两个无需模型的帧描述子做佐证:
  • motion  = mean_arm |state[k+1] - state[k]|        (本帧运动速度: 快帧是否 radius 更大)
  • grip_prox = 到最近夹爪开合切换的步距          (夹爪事件附近 radius 是否尖峰; FLASH 本就在
                                                    夹爪切换处强制全量 → 这里验证该先验的数据依据)

**纯离线、加性**: 复用 R1-d 的 sampler + 训练好的 draft + R5 的 teacher/obs 构造, 不改任何旧代码。

Run:
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090/bin/python train_scripts/kai/eval/spec_r3_probe.py \
    --config pi05_pytorch_a_new_pure_200 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_others/pytorch_pure200_step50000 \
    --asset-id a_new_pure_200 \
    --draft /tmp/draft_r1d_pure200.pt \
    --val /data1/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val \
    --holdout-eps 4 --frames-per-ep 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from spec_draft_r1d import read_video  # 复用 R1-d 的视频解码 (同目录)
import torch


def _arm_dims(action_dim: int, gripper_dims) -> list[int]:
    grip = {int(g) for g in gripper_dims}
    # pi05: 真实双臂 0-13, 夹爪 (6,13); 14-31 是 padding → 只取 0-13 去夹爪 = 12 臂关节
    return [i for i in range(min(14, action_dim)) if i not in grip]


def _gripper_switch_indices(state: np.ndarray, gripper_dims) -> np.ndarray:
    """整段轨迹上夹爪通道穿过自身中位线的帧索引 (开↔合事件)。"""
    sw = []
    for g in gripper_dims:
        col = state[:, int(g)]
        thr = float(np.median(col))
        s = np.sign(col - thr)
        # 相邻符号变化处即一次穿越
        sw.extend(np.nonzero(np.diff(s) != 0)[0].tolist())
    return np.unique(np.asarray(sw, dtype=np.int64)) if sw else np.asarray([], dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--val", required=True, help="LeRobot val root (data/ + videos/ + meta/)")
    ap.add_argument("--holdout-eps", type=int, default=4)
    ap.add_argument("--frames-per-ep", type=int, default=60)
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

    g = torch.Generator(device=device).manual_seed(0)
    n1 = torch.randn((1, ah, ad), generator=g, device=device, dtype=torch.float32)
    n2 = torch.randn((1, ah, ad), generator=g, device=device, dtype=torch.float32)

    def build_obs(imgs, state_row):
        obs = {"images": imgs, "state": state_row, "prompt": args.prompt}
        inputs = policy._input_transform(obs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def teacher(obs, noise):
        pe, ppad, patt, st = sampler._embed_prefix(obs)  # noqa: SLF001
        pkv = sampler._prefill_kv(pe, ppad, patt)  # noqa: SLF001
        return sampler._full_denoise(st, ppad, pkv, noise).float()  # noqa: SLF001

    # per-frame: floor, accept, rad_mean, rad_max, motion, grip_prox
    rows = []
    import pyarrow.parquet as pq

    with torch.no_grad():
        for ep in hold:
            ei, nf = ep["episode_index"], ep["length"]
            tbl = pq.read_table(val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet").to_pandas()
            state = np.stack([np.asarray(x) for x in tbl["observation.state"]]).astype(np.float32)
            vid = {c: read_video(val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", nf)
                   for c in cams}
            sw = _gripper_switch_indices(state, spec_args.gripper_dims)
            # 逐帧运动速度 (arm 维 |Δstate|), 末帧复用前一帧
            dstate = np.abs(np.diff(state[:, arm], axis=0)).mean(axis=1)
            dstate = np.concatenate([dstate, dstate[-1:]]) if len(dstate) else np.zeros(nf)
            for k in np.linspace(0, nf - 1, args.frames_per_ep).astype(int):
                ki = int(k)
                obs_r = build_obs({c: vid[c][ki] for c in cams}, state[ki])
                # 模型自身不确定度地板
                tr1 = teacher(obs_r, n1)
                tr2 = teacher(obs_r, n2)
                floor = float(np.mean(np.abs((tr1 - tr2)[0, :, arm].cpu().numpy())))
                # FLASH 信号
                o = sampler.sample(obs_r, noise=n1, last_gripper=None)
                accept = int(o["accepted_prefix_len"].item())
                radv = o["radius_dist"].min(dim=1).values[0].float().cpu().numpy()  # (eval_h,)
                rad_mean, rad_max = float(radv.mean()), float(radv.max())
                # 帧描述子
                motion = float(dstate[ki])
                grip_prox = float(np.min(np.abs(sw - ki))) if sw.size else float(nf)
                rows.append((floor, accept, rad_mean, rad_max, motion, grip_prox))

    a = np.asarray(rows, dtype=np.float64)
    floor, accept, rad_mean, rad_max, motion, grip_prox = (a[:, i] for i in range(6))
    n = len(a)

    def corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    print("\n========== R3 PROBE (pure200, in-distribution holdout) ==========")
    print(f"  frames={n}  tau={args.tau}  eval_h={ah}  arm_dims={arm}")
    print("  -- 动态范围 (有无信号可言) --")
    print(f"     floor    : mean={floor.mean():.4f}  std={floor.std():.4f}  "
          f"min={floor.min():.4f}  max={floor.max():.4f}")
    print(f"     radius   : mean={rad_mean.mean():.4f}  std={rad_mean.std():.4f}  "
          f"min={rad_mean.min():.4f}  max={rad_mean.max():.4f}  (max-over-pos: mean={rad_max.mean():.4f})")
    print(f"     accept   : mean={accept.mean():.1f}/{ah}  std={accept.std():.2f}  "
          f"min={accept.min():.0f}  max={accept.max():.0f}")
    print("  -- 相关 (radius/accept ↔ 模型自身不确定度 & 帧难度) --")
    print(f"     corr(radius_mean, floor)      = {corr(rad_mean, floor):+.3f}   <- 核心: radius 追踪内在不确定度?")
    print(f"     corr(radius_max,  floor)      = {corr(rad_max, floor):+.3f}")
    print(f"     corr(accept,      floor)      = {corr(accept, floor):+.3f}   (期望负: 不确定→少接受)")
    print(f"     corr(radius_mean, motion)     = {corr(rad_mean, motion):+.3f}   (快帧 radius 更大?)")
    print(f"     corr(radius_max,  -grip_prox) = {corr(rad_max, -grip_prox):+.3f}   (近夹爪事件 radius 尖峰?)")

    print("  -- verdict --")
    rad_dyn = rad_mean.std() / max(rad_mean.mean(), 1e-9)  # 变异系数
    c_floor = corr(rad_mean, floor)
    if rad_dyn < 0.05:
        print(f"     radius 在离线 holdout 上几乎恒定 (CV={rad_dyn:.3f}, R1-d 饱和延续) → radius 无离线方差,")
        print("     R3 '不确定度信号' 在 in-distribution 离线**测不出**; 须 off-manifold/扰动/闭环帧才有判别力。")
    elif not np.isnan(c_floor) and abs(c_floor) > 0.3:
        print(f"     radius 与模型自身去噪地板显著相关 (corr={c_floor:+.3f}) → radius **追踪模型内在不确定度**;")
        print("     与 R5 不冲突 (R5: 对*输入退化*盲; R3: 对*模型内在难度*敏感)。可作免费'此帧没把握'旗标 → 触发 DAgger/RLT。")
    else:
        print(f"     radius 有方差但与 floor/motion/夹爪邻近度均无强相关 (corr_floor={c_floor:+.3f}) → 离线上 radius")
        print("     不是清晰的不确定度探针; 其波动更可能是 draft 拟合噪声而非语义难度。需闭环帧再判。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
