#!/usr/bin/env python3
"""R5: is FLASH acceptance (or radius) a FREE deploy-time open-loop / vision-sensitivity probe?

R1-d 证明: 在 pure200 同任务 holdout 上 FLASH 接受率**饱和** (50/50, 0 fallback) → 单看接受率
分不出"用视觉的好模型"和"开环坏模型"。R5 问: 把同一帧的输入做**视觉消融** (全相机置黑),
模型输出会变 (vision-sensitivity, = eval_vision_ablation 的 SNR); 那么 FLASH 的**免费副产物**
(accepted_prefix_len / radius) 会不会也随之变? 若变 → radius 可当 0 成本在线探针; 若不变 →
坐实"接受率单独不够, 必须接受率 x SNR 联合"。

每帧两条件: real (全相机) vs black (全相机=0)。两路指标解耦:
  • 模型 vision-SNR  (从 sampler 的 eager `_full_denoise` teacher 算, 与 eval_vision_ablation 同义):
        floor  = mean|teacher_real(n1) - teacher_real(n2)|   (去噪随机地板)
        Δblack = mean|teacher_real(n1) - teacher_black(n1)|   (置黑引起的输出位移)
        SNR    = Δblack / floor      (~1 忽略视觉, ≫1 用视觉)   —— 在 ARM 非夹爪维上
  • FLASH 免费信号 (`sample_from_prefix`): real/black 各自的 accepted_prefix_len + min-over-K radius。

输出: real vs black 的 accept/radius 聚合 + 逐帧 SNR↔Δradius / SNR↔Δaccept 的相关。
**纯离线、加性**: 复用 R1-d 的 sampler + 训练好的 draft, 不改任何旧代码。

Run:
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090/bin/python train_scripts/kai/eval/spec_r5_probe.py \
    --config pi05_pytorch_a_new_pure_200 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_others/pytorch_pure200_step50000 \
    --asset-id a_new_pure_200 \
    --draft /tmp/draft_r1d_pure200.pt \
    --val /data1/DATA_IMP/KAI0/Task_A/A_new_pure_200_val \
    --holdout-eps 4 --frames-per-ep 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spec_draft_r1d import read_video  # 复用 R1-d 的视频解码 (同目录)


def _arm_dims(action_dim: int, gripper_dims) -> list[int]:
    grip = {int(g) for g in gripper_dims}
    # pi05: 真实双臂 0-13, 夹爪 (6,13); 14-31 是 padding → 只取 0-13 去夹爪 = 12 臂关节
    return [i for i in range(min(14, action_dim)) if i not in grip]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--val", required=True, help="LeRobot val root (data/ + videos/ + meta/)")
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
                           out_dim=int(blob["out_dim"]), use_state_token=False, gemma_config=vlm.config)
    draft.load_state_dict(blob["state_dict"])
    sampler.draft = draft.to(device, mdtype).eval()

    arm = _arm_dims(ad, spec_args.gripper_dims)
    cams = ("top_head", "hand_left", "hand_right")
    val = Path(args.val).resolve()
    eps = [json.loads(line) for line in (val / "meta" / "episodes.jsonl").read_text().strip().split("\n")]
    hold = eps[: args.holdout_eps]

    # 两条固定噪声 (floor 用 n1 vs n2; Δblack/FLASH 用 n1) — 跨帧恒定, 保证可比
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

    rows = []  # per-frame: (snr, accept_r, accept_b, rad_r, rad_b)
    import pyarrow.parquet as pq

    with torch.no_grad():
        for ep in hold:
            ei, nf = ep["episode_index"], ep["length"]
            tbl = pq.read_table(val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet").to_pandas()
            state = np.stack([np.asarray(x) for x in tbl["observation.state"]]).astype(np.float32)
            vid = {c: read_video(val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", nf)
                   for c in cams}
            for k in np.linspace(0, nf - 1, args.frames_per_ep).astype(int):
                real_imgs = {c: vid[c][int(k)] for c in cams}
                black_imgs = {c: np.zeros_like(vid[c][int(k)]) for c in cams}
                obs_r = build_obs(real_imgs, state[int(k)])
                obs_b = build_obs(black_imgs, state[int(k)])

                # --- 模型 vision-SNR (teacher denoise) ---
                tr1 = teacher(obs_r, n1)
                tr2 = teacher(obs_r, n2)
                tb1 = teacher(obs_b, n1)
                floor = float(np.mean(np.abs((tr1 - tr2)[0, :, arm].cpu().numpy())))
                dblack = float(np.mean(np.abs((tr1 - tb1)[0, :, arm].cpu().numpy())))
                snr = dblack / max(floor, 1e-9)

                # --- FLASH 免费信号 (real / black) ---
                def flash(obs):
                    o = sampler.sample(obs, noise=n1, last_gripper=None)
                    return int(o["accepted_prefix_len"].item()), float(o["radius_dist"].min(dim=1).values.mean().item())
                ar, rr = flash(obs_r)
                ab, rb = flash(obs_b)
                rows.append((snr, ar, ab, rr, rb))

    a = np.asarray(rows, dtype=np.float64)  # (N,5)
    snr, ar, ab, rr, rb = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    n = len(a)

    def corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    print("\n========== R5 PROBE (pure200, real vs all-cams-black) ==========")
    print(f"  frames={n}  tau={args.tau}  eval_h={ah}  arm_dims={arm}")
    print(f"  model vision-SNR (Δblack/floor, arm): mean={snr.mean():.2f}x  median={np.median(snr):.2f}x  "
          f"p10={np.percentile(snr,10):.2f}x  p90={np.percentile(snr,90):.2f}x")
    print(f"  FLASH accept  real: mean={ar.mean():.1f}/{ah}  black: mean={ab.mean():.1f}/{ah}  "
          f"Δ(real-black)={ (ar-ab).mean():.2f}")
    print(f"  FLASH radius  real: mean={rr.mean():.4f}  black: mean={rb.mean():.4f}  "
          f"Δ(black-real)={ (rb-rr).mean():.4f}")
    print("  -- per-frame correlation vs model vision-SNR --")
    print(f"     corr(SNR, radius_black-radius_real) = {corr(snr, rb-rr):+.3f}")
    print(f"     corr(SNR, accept_real-accept_black) = {corr(snr, ar-ab):+.3f}")
    print(f"     corr(SNR, radius_black)             = {corr(snr, rb):+.3f}")
    # verdict
    acc_blind = abs((ar - ab).mean()) < 1.0  # 接受率对置黑几乎不动
    rad_moves = (rb - rr).mean() > 0.02 or abs(corr(snr, rb - rr)) > 0.2
    print("  -- verdict --")
    if acc_blind and rad_moves:
        print("     接受率对视觉消融近乎'盲' (Δ<1 步) 但 radius 有响应 → 单看接受率不够,")
        print("     radius 可作免费视觉敏感度信号; 坐实 R5 '接受率 x (radius/SNR)' 联合。")
    elif acc_blind and not rad_moves:
        print("     接受率与 radius 对置黑都几乎不动 → 此 ckpt 上 FLASH 信号整体对视觉不敏感")
        print("     (可能 ckpt 本身偏开环, vision-SNR 低)。需外接 SNR, FLASH 信号单独不构成门禁。")
    else:
        print("     接受率本身对置黑有响应 → 在此 ckpt 上接受率即可见视觉依赖 (与 R1-d 饱和结论对照)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
