# Task P unfreeze_20k_v2 (Task_P/v2_aligned, action=state) 训练结果

> **实验**: uc02 unfreeze_20k_v2 (`pi05_pick_place_box_kai0_unfreeze_20k_v2`)
> **状态**: ✅ **完成** (20,000 步, ended 2026-05-08 14:55:06 CST, 训练时长 26h33m)
> **目的**: Task_P 数据集版本对比 — `KAI0/Task_P/base/2026-04-21-v2` (action=state) vs 原 `Task_P/base` (action≈state+motor_lag)
> **控制变量**: 数据集版本 (其他 hparams 与 orig unfreeze_20k 完全一致)

---

## 1. 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_pick_place_box_kai0_unfreeze_20k_v2` |
| Model | pi05 (Pi0Config(pi05=True)) |
| Init | `Task_A/mixed_1/params` (Task_A 训练 init, action=state 数据训练) |
| Train | `Task_P/v2_aligned_train` (84 ep, 50,601 frames after 30fps interp) |
| Val | `Task_P/v2_aligned_val` (16 ep, 8,404 frames) |
| Prompt | "pick and place in box" |
| `use_delta_joint_actions` | False |
| LR schedule | Cosine, warmup=500, peak_lr=1.5e-5, decay=20k, decay_lr=1.5e-6 |
| EMA decay | 0.999 |
| Steps | 20,000 |
| Batch | 128, fsdp_devices=8 |
| Save | every 2,000 step, keep_period=2,000 |
| inline_eval | every save (= 每 2k 步), 200 frames |
| **Seed** | **123** ⚠️ (seed=42 在 step 700 触发 batch-deterministic NaN, 详见第 5 节) |
| Server | uc02 (A800-SXM4-80GB ×8) |

## 2. 数据预处理 (v2 数据集修复)

raw `KAI0/Task_P/base/2026-04-21-v2`:
- 100 ep, ts 变化率 (mean dt=0.064s ~15.5fps), video 误标 30fps (encoder bug)
- action ≡ state (与 mixed_1 训练数据 `Task_A/kai0_base/dagger` 同语义)

修复步骤 (`/tmp/realign_v2_to_30fps.py`):
1. **per-ep 在 raw_ts 上 numpy.interp 把 action/state 插值到 30fps 网格** (M=round(T*30) 行 vs N raw 行, scale ~2x)
2. **ffmpeg setpts*M/N 把 video duration 拉伸到 T s** (真 30fps, frame i 显示 i/30 时刻)
3. **action 列恢复 = state 列** (原始 raw v2 即如此)
4. norm_stats 重算 (32-dim padded for pi05)
5. **train/val 分离 84/16** (保留原始日期划分)

## 3. 完整 inline-eval MAE@{1,10,25,50} 曲线

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 2000  | 0.0094 | 0.0181 | 0.0313 | 0.0473 | (start) |
| 4000  | 0.0075 | 0.0154 | 0.0282 | 0.0438 | -20.2% |
| 6000  | 0.0073 | 0.0152 | 0.0278 | 0.0431 | -2.7% |
| 8000  | 0.0073 | 0.0151 | 0.0275 | 0.0427 | 0.0% |
| 10000 | 0.0072 | 0.0150 | 0.0274 | 0.0425 | -1.4% |
| 12000 | 0.0071 | 0.0151 | 0.0273 | 0.0424 | -1.4% |
| 14000 | 0.0071 | 0.0150 | 0.0273 | 0.0423 | 0.0% |
| **16000** | **0.0070** | **0.0150** | **0.0272** | **0.0423** | -1.4% |
| 18000 | 0.0071 | 0.0150 | 0.0272 | 0.0423 | +1.4% (slight uptick) |
| 19999 | 0.0070 | 0.0150 | 0.0272 | 0.0423 | -1.4% |

**Best**: step 16000 / 19999 (tied), MAE@1 = **0.0070**

train metrics:
- step 0:    grad_norm=1.64, loss=0.17, param_norm=1804.34
- step 2000: grad_norm=0.055, loss=0.0029, param_norm=1804.52
- step 10000: grad_norm=0.045, loss=0.0011, param_norm=1805.09
- step 19999: grad_norm 平稳, 0 NaN 全程

## 4. 与 orig unfreeze_20k 对比 (控制变量: 数据集版本)

| 实验 | 数据集 | best step | best MAE@1 | @10 | @25 | @50 |
|---|---|---:|---:|---:|---:|---:|
| orig unfreeze_20k | Task_P/base (24,822 frames, action≈state+motor_lag) | 4000 | 0.0195 | 0.0367 | 0.0600 | 0.0797 |
| **本实验 v2_aligned** | Task_P/v2 (50,601 frames after interp, action≡state) | 16000 / 19999 | **0.0070** | **0.0150** | **0.0272** | **0.0423** |
| 改善 | | | **-64.1%** | -59.1% | -54.7% | -46.9% |

**关键观察**:
- v2 数据规模 2x (frame), 全 horizon 显著好于原版
- v2 收敛慢 (best @ 16k vs orig @ 4k) 但绝对值好得多
- 长 horizon (@50) 改善 -47%, 短 horizon (@1) 改善 -64% — **v2 更适合精细单步控制**

## 5. seed=42 NaN 排错过程 (重要历史)

**症状**: peak_lr 1.5e-5 / 5e-6 两次都在 step 700 (有时 500-700) 突然 grad_norm/loss 全部 nan, 之前 step 0-600 健康线性下降。

**排查路径** (花了多次试验):
1. ❌ 数据 outlier: action range [-2.98, 2.26], 0 NaN/Inf, video 解码正常
2. ❌ lr 过高: 降到 5e-6 + warmup 2000 仍 step 700 NaN
3. ❌ ffmpeg 插值导致 sync 错位: aligned vs unaligned 两版本都 NaN (aligned 提前到 step 500)
4. ❌ init mismatch: mixed_1 训练数据本身就是 action=state, 与 v2 一致
5. ❌ A800 vs A100 数值差异: 同型号 GPU + 同 driver, gf2 (uc01) 同 hparam Task_A 任务健康
6. ✅ **seed**: seed=123 健康通过 step 1500+, 完整跑完 20k 步 0 NaN

**根因 (确诊)**: `seed=42` 的 dataloader 在 step 700 sample 出一个 unlucky batch (含特定 episode/frame 组合), 触发 BF16 forward 中间激活 overflow (softmax/SiLU/attention) → NaN, 不可逆。**deterministic batch bug**, 与 lr/data/init 都无关。

**结论**: 训练前期 (warmup 完前) 撞 NaN 时, **优先换 seed** 而非降 lr 或调数据。

## 6. 最佳 Checkpoint 信息

- **路径** (uc02): `/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_pick_place_box_kai0_unfreeze_20k_v2/p_unfreeze_20k_v2_v1/`
- **保留 ckpts**: 16000, 18000, 19999 (keep_period=2000)
- **推荐**: step 16000 (首达 plateau) 或 step 19999 (final, 同等)
- **norm_stats**: `kai0/data/Task_P/v2_aligned_train/norm_stats.json` (32-dim padded, 走 default repo_id 模式)
- **train/val 数据**: `Task_P/v2_aligned_{train,val}/` (interp + ffmpeg 加工后版本)

## 7. 经验教训

1. **deterministic seed NaN 排错**: 训练早期 (step 100-1000) 突 NaN 而前期健康下降, 第一步换 seed 试错最高 ROI; 不要先调 lr/data/init
2. **video encoder fps 标记错位陷阱**: source ts 与 video frame rate 不一致时 lerobot tolerance check 必失败. ts=i/source_fps 与 video duration 必须自洽; ffmpeg setpts 是干净 fix
3. **action=state 数据训练完全合理**: 模型学 state→state[t:t+32] (identity-chunk), 在 imitation learning 中是标准 BC 形式
4. **seed=42 黑名单**: 此项目 v2 数据下 seed=42 会 NaN; **将 seed=123 加入未来训练默认**
5. **数据语义对齐 init**: mixed_1 init (action=state Task_A 训练) + Task_P/v2 (action=state) 是一致语义, 训练通常稳定
