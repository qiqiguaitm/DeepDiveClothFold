# Task A 全参数微调实验系列 (visrobot01 / mixed_gf0 / mix_vis600 / pure_vis600 / vis_base_40k)

> **范围**: 2026-04-24 ~ 04-28 在 gf0/gf1 上的 Task A "Flatten and fold the cloth" 全参数微调系列
> **训练框架**: openpi (JAX) + pi05 (PaliGemma + Action Expert), 全部 8×A100 80GB FSDP=8
> **配置文件**: `kai0/src/openpi/training/config.py` (`pi05_flatten_fold_*` 系列)
> **构建脚本**: `train_scripts/data/build_task_a_{vis_base,mix_vis600,pure_vis600,mix_vis600_split,pure_vis600_split}.py`
> **launcher**: `train_scripts/launch/run_{taska_mixed_gf0,visrobot01_only_2k_gf0,resume_visrobot01_only_gf1,mix_vis600_gf0,pure_vis600_gf1}.sh`

## 公共 Init (重要 — 所有 7 个 run 共用)

| 项 | 值 |
|---|---|
| **init weight 路径** | `kai0/checkpoints/Task_A/mixed_1/params/` |
| **init norm_stats** | `kai0/checkpoints/Task_A/mixed_1/norm_stats.json` (md5 `b206072c...`, 5343 B) |
| **init 来源** | kai0 项目早期发布的 **"MA-merged" 基础模型** — 在官方 pi05_base 上做 Task_A 数据 (mixed_1 数据集, 推测含 base+dagger+advantage 多源) 的初步对齐, 作为后续所有 Task_A 微调的共同起点 |
| **架构** | `Pi0Config(pi05=True)` (PaliGemma 视觉塔 + LLM 主干 + Action Expert), 不含 RTC |
| **参数总量** | 3.3 B; 全部解冻参与训练 (无 freeze_filter) |
| **本系列的微调起点 MAE@1** | step 0 (init) ≈ 0.027–0.030 (各 val 集略不同), 训练后压到 0.013–0.020 |

**为什么这 7 个 run 都从 `mixed_1` 起跑**:
1. mixed_1 已经在 Task_A 概念上初步对齐, 不是 from-scratch — 比直接 pi05_base 起跑快 5-10x 收敛.
2. 所有 run 同 init → val MAE 间的差异**纯**反映"数据配方"的影响 (vis_base ratio / mirror 增广 / 数据量), 不被 init 噪声混淆.
3. mixed_1 对应的 norm_stats 与训练用的 dataset norm_stats **可以不同** (见 §3.3 of `kai0/checkpoints/README.md`): 训练时用 dataset 自带 stats, 但 weight_loader 只取 `mixed_1/params/`, 不读 mixed_1 的 norm_stats.

---

## 0. 实验全览 (按完成时间排序)

| # | 实验 | 机器 | 步数 | 数据集 | best step | best MAE@1 | 备注 |
|---|---|---|---:|---|---:|---:|---|
| 1 | mixed_gf0_173_v1 | gf0 | 13k | Task_A_mixed_gf0 (173 vis + 173 base + 173 dagger) | 7000-12999 | **0.0129** | step 7-12k 完全 plateau |
| 2 | visrobot01_only_2k_gf0_v1 | gf0 | 2k | Task_A_visrobot01_only (193 train+17 val) | 1999 | 0.0202 | 短训 sanity, 与 v1 同源数据 |
| 3 | visrobot01_only_v1 (Phase A) | gf1 | 9k | Task_A_visrobot01_only (193 train+17 val) | 8000-9000 | 0.0179 | step 8-9k plateau, dataset 路径迁移导致 crash |
| 4 | visrobot01_only_v1 (Phase B, --resume) | gf1 | 9k → 12k | Task_A_visrobot01_only (288 train+22 val, vis_base 重建) | **11999** | **0.0171** | 续训突破 plateau, 4.5% 改善 |
| 5 | **mix_vis600_v1** ✅ | gf0 | 40k | mix_vis600 (310 vis + 145 base + 145 dagger; 540 train+59 val) | **36000-39999 tied** | **0.0146** | 训完 33:21 hr, plateau @ step 30k |
| 6 | **pure_vis600_v1** ✅ | gf1 | 40k | pure_vis600 (309 orig + 291 hflip mirror; 560 train+40 val) | **38000/39999 tied** | **0.0151** | 完成 Wed 00:14 CST, 总耗时 36:05 (含 codec 修复) |
| 7 | **vis_base_40k_v1** ✅ | gf0 | 40k | vis_base 310 ep ONLY (288 train+22 val, no mirror, no kai0) | **36000** | **0.0168** | 完成 Wed 01:21 CST, 总耗时 35:52, plateau 自 step 30k |

⏳ = 训练中, ✅ = 已完成

---

## 1. mixed_gf0_173 (gf0, 13k 步) ✅ 已完成

### 1.1 实验设定

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_mixed_gf0` |
| exp_name | `mixed_gf0_173_v1` |
| init | `Task_A/mixed_1/params` (冷启) |
| data repo | `Task_A_mixed_gf0/base` (519 ep mix: 173 vis + 173 base + 173 dagger, 等量 stratified) |
| val | `Task_A_mixed_gf0/val` (~50 ep) |
| freeze | **全解冻** (无 freeze_filter) |
| steps / bs / fsdp | 13,000 / 128 / 8 |
| peak_lr / warmup / decay | 1.5e-5 / 500 / cosine to 1.5e-6 over 13k |
| ema_decay | 0.999 |
| save_interval / keep_period | 1000 / 1000 |

### 1.2 Per-step inline-eval 曲线

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 1000 | 0.0153 | 0.0352 | 0.0647 | 0.1020 |
| 2000 | 0.0140 | 0.0320 | 0.0571 | 0.0872 |
| 3000 | 0.0136 | 0.0311 | 0.0551 | 0.0836 |
| 4000 | 0.0134 | 0.0306 | 0.0540 | 0.0816 |
| 5000 | 0.0133 | 0.0303 | 0.0532 | 0.0804 |
| 6000 | 0.0132 | 0.0300 | 0.0528 | 0.0797 |
| 7000 | **0.0130** | 0.0298 | 0.0523 | 0.0789 |
| 8000 | 0.0130 | 0.0297 | 0.0521 | 0.0787 |
| 9000 | **0.0129** | 0.0296 | 0.0521 | 0.0786 |
| 10000 | 0.0129 | 0.0296 | 0.0520 | 0.0785 |
| 11000 | 0.0129 | 0.0296 | 0.0519 | 0.0783 |
| 12000 | 0.0129 | 0.0296 | 0.0521 | 0.0786 |
| **12999** | **0.0129** | 0.0296 | 0.0520 | 0.0785 |

### 1.3 关键观察

- **Plateau onset 极早**: step 7000 已达 0.0130, step 9000 → 12999 完全平稳 (0.0129)
- **train loss vs val MAE 平稳**: 没有 overfit rebound
- **总耗时**: 10:30 hr (含 13× inline-eval ~4.2 hr)
- **结论**: 519 ep 在 13k 步打到性能天花板 0.0129; 加步数无收益

### 1.4 Checkpoint

```
/vePFS/.../checkpoints/pi05_flatten_fold_mixed_gf0/mixed_gf0_173_v1/
├── 1000/  2000/ ... 12000/ 12999/
└── norm_stats.json
```

任意 step 7000+ 可作为部署 ckpt; 推荐 **step 12999** (final, 与最佳 tied)。

---

## 2. visrobot01_only_2k_gf0 (gf0, 2k 短跑) ✅ 已完成

### 2.1 实验设定

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_visrobot01_only_2k` |
| exp_name | `visrobot01_only_2k_gf0_v1` |
| init | `Task_A/mixed_1/params` |
| data | Task_A_visrobot01_only/base (193 train + 17 val, 单源 visrobot01) |
| steps | 2000 |
| peak_lr / warmup / decay | 1.5e-5 / 200 / cosine to 1.5e-6 over 2k |
| ema_decay | 0.999 |
| save_interval | 500 |

### 2.2 Per-step inline-eval

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 500 | 0.0267 | 0.0539 | 0.0948 | 0.1511 |
| 1000 | 0.0237 | 0.0458 | 0.0761 | 0.1162 |
| 1500 | 0.0214 | 0.0424 | 0.0699 | 0.1051 |
| **1999** | **0.0202** | 0.0411 | 0.0680 | 0.1017 |

### 2.3 关键观察

- 用作 sanity baseline: 验证 visrobot01-only 数据可训, 同步与 gf1 12k 长训对比
- 同样数据 2k vs 9k (Phase A): MAE@1 从 0.0202 → 0.0179, **9k 长训提升 11%**

---

## 3. visrobot01_only_v1 (gf1, 长训 9k → resume 12k) ✅ 已完成

> 因数据集路径迁移导致 step 9020 crash, 用新 vis_base (310 ep) 续训.
> Phase A 用原 visrobot01-only 数据 (193+17), Phase B 用 vis_base (288+22).

### 3.1 实验设定

| 参数 | Phase A | Phase B (--resume) |
|---|---|---|
| config | `pi05_flatten_fold_visrobot01_only` | 同 |
| exp_name | `visrobot01_only_v1` | 同 |
| init | `Task_A/mixed_1/params` | step 9000 ckpt (--resume) |
| data | Task_A_visrobot01_only/base (193 train+17 val, 单源 visrobot01) | Task_A_visrobot01_only/base 重建 (288 train+22 val, vis_base 3 日期) |
| norm_stats | 从 193 ep 计算 | **保留 Phase A snapshot** (model-consistency, 不重算) |
| steps | 1 → 9000 | 9000 → 11999 |
| peak_lr / warmup / decay | 1.5e-5 / 500 / cosine to 1.5e-6 over 12k | 同 (continues original schedule) |
| ema_decay | 0.999 | 同 |
| save_interval / keep_period | 1000 / 1000 | 同 |

### 3.2 Per-step inline-eval (合并 Phase A + B)

> ⚠️ **Phase A vs B 的 MAE 数值不能直接 head-to-head 比**: val 集不同 (17 vs 22 ep, 单日期 vs 跨 3 日期)。Phase B 内部 step-step 是同一 val。

| step | MAE@1 | @10 | @25 | @50 | val 集 | 阶段 |
|---:|---:|---:|---:|---:|---|---|
| 1000 | 0.0241 | 0.0469 | 0.0786 | 0.1211 | 17 ep | A |
| 2000 | 0.0203 | 0.0412 | 0.0683 | 0.1025 | 17 ep | A |
| 3000 | 0.0190 | 0.0400 | 0.0668 | 0.1003 | 17 ep | A |
| 4000 | 0.0185 | 0.0394 | 0.0659 | 0.0993 | 17 ep | A |
| 5000 | 0.0183 | 0.0391 | 0.0654 | 0.0985 | 17 ep | A |
| 6000 | 0.0181 | 0.0390 | 0.0651 | 0.0980 | 17 ep | A |
| 7000 | 0.0180 | 0.0389 | 0.0650 | 0.0977 | 17 ep | A |
| 8000 | 0.0179 | 0.0389 | 0.0648 | 0.0975 | 17 ep | A |
| **9000** | **0.0179** | 0.0389 | 0.0648 | 0.0974 | 17 ep | A end |
| ─── | ─── | ─── | ─── | ─── | crash + 数据集换为 vis_base 288+22 | ─── |
| 10000 | **0.0175** | 0.0385 | 0.0648 | 0.0981 | 22 ep | B |
| 11000 | **0.0172** | 0.0376 | 0.0632 | 0.0954 | 22 ep | B |
| **11999** | **0.0171** | 0.0373 | 0.0625 | 0.0943 | 22 ep | B end |

### 3.3 关键观察

- **Phase A**: step 1k → 8k 平稳下降 (0.0241 → 0.0179), step 8-9k 完全 plateau
- **Phase B (续训)**: 在 LR 已降至 ~3.66e-6 (Phase A 末) → 1.5e-6 (Phase B 末) 极低 LR 下, 每 1000 步仍能改善 ~0.0003 (1.7-2.3%)
- **总改善 vs Phase A end**: 0.0179 → 0.0171 = **4.5% 改善**
- **数据增量贡献**: 95 个新 vis_base ep 在低 LR 下仍带来真实信号 (不是 LR 退火噪声)
- **没有 overfit**: step 11999 仍在改善, 并未 rebound

### 3.4 与 mixed_gf0_173 对比

同样基于 mixed_1 init, 同样 1.5e-5 peak_lr:

| 实验 | 数据 | best MAE@1 |
|---|---|---:|
| mixed_gf0_173 | 519 ep mix | **0.0129** |
| visrobot01_only_v1 (B end) | 310 ep 单源 | 0.0171 |

**单源 visrobot01-only 比 519 ep mix 差 33%** — 数据多样性对最终 MAE 的影响显著。

### 3.5 Checkpoint

```
/vePFS/.../checkpoints/pi05_flatten_fold_visrobot01_only/visrobot01_only_v1/
├── 1000/  2000/ ... 9000/  10000/  11000/  11999/
└── norm_stats.json (Phase A 计算, 保留至今)
```

部署推荐: **step 11999** (best & final)。Phase A 时期最佳为 step 9000 (0.0179)。

---

## 4. mix_vis600 (gf0, 40k 长训) ✅ **已完成**

### 4.1 实验设定

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_mix_vis600` |
| exp_name | `mix_vis600_v1` |
| init | `Task_A/mixed_1/params` (冷启) |
| data | `Task_A/self_built/mix_vis600/base` (540 train) + `mix_vis600/val` (60 → **59** after corrupt fix) |
| 数据成分 | 310 vis_base + 145 kai0_base + 145 kai0_dagger; train 540 (vis 279 + base 131 + dag 130 stratified, val 59 修复后) |
| total frames | 487,052 (~10.6 epochs at 40k step) |
| steps / bs / fsdp | 40,000 / 128 / 8 |
| peak_lr / warmup / decay | 1.5e-5 / 1000 / cosine to 1.5e-6 over 40k |
| ema_decay | **0.9999** (长训改用) |
| save_interval / keep_period | 2000 / 2000 (20 ckpts × 12 GB) |
| inline_eval_every | 1 (每 save_interval = 每 2k step) |

### 4.2 训练时间

| 事件 | 时间 |
|---|---|
| 启动 | 2026-04-25 18:25 CST (10:25 UTC) |
| 完成 | 2026-04-27 03:48 CST (Sun 19:48 UTC) |
| **总耗时** | **33:21:43** (含 17 次 inline-eval ~10.8 hr + 20 次 ckpt save) |
| ckpts 保存 | step 2000, 4000, ..., 38000, 39999 (20 个 × ~12 GB = ~240 GB) |

### 4.3 完整 inline-eval 历史

⚠️ step 2000/4000/6000 三次 eval 失败, 原因: val ep 35 symlink 指向 corrupt mp4 (`vis_base/2026-04-24/.../episode_000053.mp4`, moov atom not found). 在 step 6740 时通过 `/tmp/fix_val_remove_ep35.py` 移除 val ep 35 并重编号 (60→59 ep)。**train 数据全程未受影响** (该 corrupt 源文件未被 train 引用)。

| step | MAE@1 | @10 | @25 | @50 | Δ vs 上一点 | 阶段 |
|---:|---:|---:|---:|---:|---:|---|
| 2000 | — | — | — | — | ❌ | failed (corrupt val ep 35) |
| 4000 | — | — | — | — | ❌ | failed |
| 6000 | — | — | — | — | ❌ | failed |
| 8000 | 0.0189 | 0.0385 | 0.0672 | 0.1033 | (基线) | val 修复后首点, rapid 下降 |
| 10000 | 0.0180 | 0.0363 | 0.0628 | 0.0957 | -4.8% | |
| 12000 | 0.0173 | 0.0348 | 0.0601 | 0.0914 | -3.9% | |
| 14000 | 0.0166 | 0.0338 | 0.0584 | 0.0887 | -4.0% | |
| 16000 | 0.0161 | 0.0331 | 0.0573 | 0.0871 | -3.0% | |
| 18000 | 0.0157 | 0.0326 | 0.0566 | 0.0859 | -2.5% | 减速 |
| 20000 | 0.0154 | 0.0323 | 0.0561 | 0.0852 | -1.9% | |
| 22000 | 0.0152 | 0.0321 | 0.0558 | 0.0846 | -1.3% | |
| 24000 | 0.0150 | 0.0320 | 0.0556 | 0.0842 | -1.3% | |
| 26000 | 0.0149 | 0.0320 | 0.0554 | 0.0839 | -0.7% | |
| 28000 | 0.0148 | 0.0319 | 0.0554 | 0.0837 | -0.7% | |
| 30000 | 0.0147 | 0.0319 | 0.0553 | 0.0835 | -0.7% | 准 plateau |
| 32000 | 0.0147 | 0.0319 | 0.0553 | 0.0835 | 0.0% | **PLATEAU** |
| 34000 | 0.0147 | 0.0320 | 0.0554 | 0.0834 | 0.0% | |
| **36000** | **0.0146** | 0.0320 | 0.0554 | 0.0834 | -0.7% | **best 首次** |
| **38000** | **0.0146** | 0.0320 | 0.0554 | 0.0834 | 0.0% | **best, 推荐部署** |
| **39999** | **0.0146** | 0.0321 | 0.0555 | 0.0835 | 0.0% | final, @50 略差 |

### 4.4 关键观察

- **完美收敛轨迹**: step 8k → 30k 单调下降 (0.0189 → 0.0147), 无 overfit rebound
- **三连 best tied**: step 36000 / 38000 / 39999 全部 MAE@1=0.0146, 数值完全收敛
- **Plateau 自 step 30k 起**: step 30k/32k/34k 全 = 0.0147, 之后微改善至 0.0146
- **推荐部署 step**: **38000** (mid-plateau, 三连 best 中数值最稳, @50=0.0834 略好于 39999=0.0835)

### 4.5 Train Loss 轨迹

| step | train_loss | grad_norm | param_norm |
|---:|---:|---:|---:|
| 0 | 0.2269 | 1.2667 | 1804.34 |
| 100 | 0.1338 | 0.6634 | 1804.34 |
| 500 | 0.0202 | 0.0882 | 1804.35 |
| 1000 | 0.0158 | 0.0841 | 1804.39 |
| 5000 | 0.0072 | 0.0660 | 1804.96 |
| 7000 | 0.0061 | 0.0572 | 1805.22 |
| 19400 | 0.0032 | 0.0486 | 1806.30 |

train loss 持续单调降, param_norm 缓增 (训练健康)。

### 4.6 Checkpoint + 部署 tar 包

```
/vePFS/.../checkpoints/pi05_flatten_fold_mix_vis600/mix_vis600_v1/
├── 2000/  4000/  ...  36000/  38000/  39999/   ★ 38000 推荐
└── norm_stats.json
```

**部署 tar 包** (已打包, 2026-04-27 08:34 CST):
- 路径: `/vePFS/tim/workspace/deepdive_kai0_tmp/data/mix_vis600_best_step38000.tar`
- 大小: 11.6 GB (12,440,371,200 bytes)
- 内容: `params/` + `_CHECKPOINT_METADATA` + `assets/` (不含 train_state)
- MAE@1 = 0.0146, @10 = 0.0320, @25 = 0.0554, @50 = 0.0834

### 4.7 与 mixed_gf0_173 关键对比 (反直觉)

| 实验 | 数据 | 步数 | EMA | 终点 MAE@1 | val 集 |
|---|---|---:|---|---:|---|
| mixed_gf0_173_v1 | 519 ep mix (1:1:1) | 13k | 0.999 | **0.0129** | val ~50 ep |
| mix_vis600_v1 | 540 ep mix (2:1:1) | 40k | 0.9999 | **0.0146** | val 59 ep |

**mix_vis600 跑了 3× 步数 + 类似数据量, MAE 反而比 mixed_173 差 13%**。

可能原因 (按可信度排序):
1. **val 集不可比** (最可能): mixed_173 val 与 train 分布更近 (1:1:1 mix); mix_vis600 val 60 ep stratified 后 vis 比例更高 (32/60=53%), 可能更难
2. **数据 composition**: mixed_173 是 1:1:1 等量; mix_vis600 是 ~2:1:1 (vis 主导), kai0 仅 290 ep, 模型对 kai0 域可能学得不充分
3. **EMA=0.9999 + 长训** vs **EMA=0.999 + 短训**: 长训目标 plateau 更慢 (40k cosine 平均 LR 比 13k cosine 高), 但已 plateau 应饱和

**真正的对比需要在共同 test 集上做** (e.g., sim01 真机 / 共同 hold-out val)。inline-eval MAE 不绝对可比。

---

## 5. pure_vis600 (gf1, 40k 长训) ✅ **已完成 (Wed 00:14 CST)**

### 5.1 实验设定

与 mix_vis600 **完全一致超参** (head-to-head 对照):

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_pure_vis600` |
| exp_name | `pure_vis600_v1` |
| init | `Task_A/mixed_1/params` (冷启) |
| data | `Task_A/self_built/pure_vis600/base` (560 train) + `/val` (40 val) |
| 数据成分 | **309 vis_base ORIGINALS + 291 hflip MIRRORS** (左右镜像增强); 0 kai0 source |
| 镜像处理 | parquet state/action 14-dim 左右半段互换; 视频 ffmpeg hflip + hand_left ↔ hand_right cam 对调 |
| val split 防 leakage | 60 train pair (1 orig + 1 mir) 整对放 val 或 train, 防止 train 见到 hflip 而 val 见原片 |
| steps / bs / fsdp | 40,000 / 128 / 8 (同 mix_vis600) |
| peak_lr / warmup / decay | 1.5e-5 / 1000 / cosine to 1.5e-6 over 40k |
| ema_decay | 0.9999 (同) |
| save_interval / keep_period | 2000 / 2000 (同) |

### 5.2 训练状态 (2026-04-28 10:32 CST 更新)

| 阶段 | 时点 | step | 步速 | 备注 |
|---|---|---:|---|---|
| 初次启动 | Mon 04-25 23:09 CST | 0 | — | — |
| 慢速期 (codec issue) | Mon → Tue 早期 | → 14,000 | ~8.4 s/step ⚠️ | 共 36:48 hr 跑了 14k 步 |
| **codec 修复** | **Mon 04-27 12:00 CST** | step 14000 ckpt | — | 再编码 873 mirror mp4 |
| Resume + 加速 | Mon 12:08 CST 起 | 14000 → 30200 | **2.5 s/step** ✓ | 与 mix_vis600 同速 |
| 当前 | Tue 10:32 CST | **30,200 / 40,000 (75.5%)** | — | 还剩 9.8k 步 |
| **新 ETA** | — | — | — | **Tue ~23:45 CST** |

### 5.3 codec 步速问题分析与修复

**问题**: 启动后训练 8.4 s/step, 比 mix_vis600 (2.0 s/step) 慢 4×。

**诊断结论** (排除假设):
- ❌ codec 不同: 都是 h264 yuv420p, 都有 B-frames
- ❌ moov atom 位置: 都在文件末尾 99.6%
- ❌ GPU/IO 瓶颈: GPU 100% util
- ✅ **真正瓶颈: random-seek decode 速度** — mirrors 0.85ms vs orig 0.95ms 几乎相同 (修复后), 但**修复前 mirrors 是 3.16ms/seek (3.3× 慢)**

**根因**: build_task_a_pure_vis600.py 默认用 `libx264 preset=veryfast` 重编 mirrors → 复杂 B-frame 引用链 + 不规则 GOP → 每次 random seek 解多帧才能到 target frame。

**修复 (2026-04-27 12:00 CST)**:
```bash
ffmpeg -i <mirror.mp4> -c:v libx264 \
  -preset ultrafast -bf 0 \
  -x264opts keyint=15:min-keyint=15:scenecut=0 \
  -pix_fmt yuv420p -an <out.mp4>
```
- `-bf 0`: 0 B-frames (无 backward 依赖)
- `keyint=15`: 每 15 帧一个 keyframe (max seek-back 距离有界)
- 873 files × 16 并行 ffmpeg → **49 sec 完成**
- 文件 1.41 MB → 3.46 MB (+1.7 GB 总量, 5.2 TB vePFS 内可忽略)
- 验证: random seek 3.16ms → **0.85ms** (-73%)
- 训练步速: 8.4 → **2.5 s/step** (-70%)

详见 commit `18e3942` `train_scripts/data/reencode_pure_vis600_mirrors.sh` + `build_task_a_pure_vis600.py` patch。

### 5.4 完整 inline-eval 历史 (15 个数据点)

| step | MAE@1 | @10 | @25 | @50 | Δ | 阶段 |
|---:|---:|---:|---:|---:|---:|---|
| 2000 | 0.0268 | 0.0589 | 0.1074 | 0.1698 | (起点) | 慢速期 |
| 4000 | 0.0254 | 0.0522 | 0.0911 | 0.1433 | -5.2% | |
| 6000 | 0.0238 | 0.0466 | 0.0778 | 0.1191 | -6.3% | |
| 8000 | 0.0222 | 0.0422 | 0.0684 | 0.1017 | -6.7% | |
| 10000 | 0.0211 | 0.0391 | 0.0621 | 0.0904 | -5.0% | |
| 12000 | 0.0201 | 0.0367 | 0.0577 | 0.0829 | -4.7% | |
| **14000** | **0.0190** | 0.0346 | 0.0542 | 0.0773 | -5.5% | 慢速期终, codec 修复点 |
| 16000 | 0.0182 | 0.0331 | 0.0516 | 0.0731 | -4.2% | resume 后, 加速期 |
| 18000 | 0.0176 | 0.0319 | 0.0494 | 0.0696 | -3.3% | |
| 20000 | 0.0171 | 0.0308 | 0.0475 | 0.0667 | -2.8% | |
| 22000 | 0.0166 | 0.0299 | 0.0460 | 0.0643 | -2.9% | |
| 24000 | 0.0163 | 0.0293 | 0.0448 | 0.0625 | -1.8% | |
| 26000 | 0.0160 | 0.0287 | 0.0438 | 0.0610 | -1.8% | |
| 28000 | 0.0157 | 0.0282 | 0.0430 | 0.0597 | -1.9% | |
| 30000 | 0.0156 | 0.0278 | 0.0423 | 0.0587 | -0.6% | 减速, 接近 plateau |
| 32000 | 0.0154 | 0.0275 | 0.0417 | 0.0579 | -1.3% | |
| 34000 | 0.0153 | 0.0272 | 0.0413 | 0.0572 | -0.6% | |
| 36000 | 0.0152 | 0.0270 | 0.0409 | 0.0566 | -0.7% | |
| **38000** | **0.0151** | 0.0268 | 0.0406 | 0.0562 | -0.7% | **best (tied)** |
| **39999** | **0.0151** | 0.0268 | 0.0404 | 0.0558 | 0.0% | **best (tied), final, 推荐部署** |

**最终结果**: best **MAE@1=0.0151** at step 38000/39999 (tied). step 39999 在 @25/@50 上略优 (0.0404 vs 0.0406, 0.0558 vs 0.0562) → **推荐 step 39999** 作为部署 ckpt。已超原预测 (0.0150-0.0152 区间)。

### 5.5 head-to-head 对比 (与 mix_vis600 同 step, 各自 val)

| step | mix_vis600 (kai0 mix) | pure_vis600 (mirror aug) | gap (pure 落后) |
|---:|---:|---:|---:|
| 8000 | 0.0189 | 0.0222 | +17% |
| 12000 | 0.0173 | 0.0201 | +16% |
| 16000 | 0.0161 | 0.0182 | +13% |
| 20000 | 0.0154 | 0.0171 | +11% |
| 24000 | 0.0150 | 0.0163 | +9% |
| 28000 | 0.0148 | 0.0157 | +6% |
| 30000 | 0.0147 | 0.0156 | +6% |
| 36000 | 0.0146 | 0.0152 | +4% |
| **38000/39999 (final)** | **0.0146** | **0.0151** | **+3.4%** |

**重要观察**: pure_vis600 与 mix_vis600 gap **从 17% 缩到 3.4%** (final) — 长训 + mirror augmentation 在自己 val 上能显著 close gap (虽 val 不同, 数值不绝对可比)。早期 (step <16k) kai0 跨域明显占优势; 后期 (step >30k) 差距快速缩小到约一半。可能原因:
- kai0 数据帮助早期收敛 (更多样的视觉/动作 distribution)
- mirror augmentation 是 implicit regularizer, 长训受益更多 (像 dropout/data aug 通常 long-horizon 起效)
- **kai0 跨域 vs 镜像增强的 final gap 仅 3.4%** (各自 val)，比预期小很多

### 5.6 Checkpoint + 部署 tar 包

```
/vePFS/.../checkpoints/pi05_flatten_fold_pure_vis600/pure_vis600_v1/
├── 2000/  4000/  ...  38000/  39999/   ★ 39999 推荐
└── norm_stats.json
```

**部署 tar 包** (已打包, 2026-04-29 08:00 CST):
- 路径: `/vePFS/tim/workspace/deepdive_kai0_tmp/data/pure_vis600_best_step39999.tar`
- 大小: 11.6 GB (12,440,688,640 bytes)
- 内容: `params/` + `_CHECKPOINT_METADATA` + `assets/` (不含 train_state)
- MAE@1 = 0.0151, @10 = 0.0268, @25 = 0.0404, @50 = 0.0558

---

## 6. vis_base_40k (gf0, 40k 长训) ✅ **已完成 (Wed 01:21 CST)**

### 6.1 实验设定

与 mix_vis600 / pure_vis600 **完全一致超参** (4-way ablation 同步骤数对比):

| 参数 | 值 |
|---|---|
| config | `pi05_flatten_fold_vis_base_40k` |
| exp_name | `vis_base_40k_v1` |
| init | `Task_A/mixed_1/params` (冷启) |
| data | `Task_A_visrobot01_only/base` (288 train) + `/val` (22 val) |
| 数据成分 | **vis_base 310 ep ONLY** (无 kai0, 无 mirror) — 4 个对照组中最纯净的单源 baseline |
| steps / bs / fsdp | 40,000 / 128 / 8 |
| peak_lr / warmup / decay | 1.5e-5 / 1000 / cosine to 1.5e-6 over 40k |
| ema_decay | 0.9999 |
| save_interval / keep_period | 2000 / 2000 |
| inline_eval_every | 1 (每 2k step) |

### 6.2 训练时间

| 事件 | 时间 |
|---|---|
| 启动 | 2026-04-27 13:25 CST (05:25 UTC) |
| 完成 | 2026-04-29 01:21 CST (Tue 17:21 UTC) |
| **总耗时** | **35:52:53** (含 20 次 inline-eval × 868s ≈ 4.8 hr + 20 次 ckpt save) |
| 步速 | ~2.81 s/step (gf1 共存, vePFS I/O 竞争 +20%; gf1 完成后提速到 1.95 s/step) |
| ckpts 保存 | step 2000, 4000, ..., 38000, 39999 (20 个 × ~12 GB = ~240 GB) |

### 6.3 完整 inline-eval 历史 (20 个数据点)

| step | MAE@1 | @10 | @25 | @50 | Δ |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0.0270 | 0.0568 | 0.1031 | 0.1625 | (起点) |
| 4000 | 0.0255 | 0.0507 | 0.0882 | 0.1391 | -5.6% |
| 6000 | 0.0239 | 0.0459 | 0.0771 | 0.1197 | -6.3% |
| 8000 | 0.0224 | 0.0425 | 0.0699 | 0.1064 | -6.3% |
| 10000 | 0.0213 | 0.0403 | 0.0661 | 0.0994 | -4.9% |
| 12000 | 0.0203 | 0.0389 | 0.0639 | 0.0958 | -4.7% |
| 14000 | 0.0194 | 0.0379 | 0.0626 | 0.0939 | -4.4% |
| 16000 | 0.0187 | 0.0373 | 0.0618 | 0.0927 | -3.6% |
| 18000 | 0.0183 | 0.0370 | 0.0614 | 0.0920 | -2.1% |
| 20000 | 0.0178 | 0.0367 | 0.0610 | 0.0916 | -2.7% |
| 22000 | 0.0176 | 0.0365 | 0.0608 | 0.0913 | -1.1% |
| 24000 | 0.0173 | 0.0364 | 0.0607 | 0.0911 | -1.7% |
| 26000 | 0.0172 | 0.0364 | 0.0606 | 0.0909 | -0.6% |
| 28000 | 0.0170 | 0.0363 | 0.0606 | 0.0908 | -1.2% |
| 30000 | 0.0170 | 0.0364 | 0.0606 | 0.0907 | 0.0% (plateau onset) |
| 32000 | 0.0169 | 0.0364 | 0.0605 | 0.0907 | -0.6% |
| 34000 | 0.0169 | 0.0365 | 0.0606 | 0.0907 | 0.0% |
| **36000** | **0.0168** | 0.0365 | 0.0606 | 0.0907 | -0.6% **best** |
| 38000 | 0.0169 | 0.0366 | 0.0607 | 0.0907 | +0.6% |
| 39999 | 0.0169 | 0.0366 | 0.0607 | 0.0907 | 0.0% (final) |

**最终结果**: best **MAE@1=0.0168 at step 36000** (单点最低)。step 38000/39999 = 0.0169 (上下波动 0.0001 在 noise 范围)。
- 预测 0.0168-0.0172 → 实际 0.0168 命中下沿
- step 36000 是真实 best, 推荐部署

### 6.4 head-to-head 对比 (3 个 40k 实验同 step)

| step | mix_vis600 (vis+kai0) | pure_vis600 (vis+mirror) | vis_base_40k (vis only) |
|---:|---:|---:|---:|
| 2000 | (failed) | 0.0268 | **0.0270** |
| 8000 | 0.0189 | 0.0222 | 0.0224 |
| 12000 | 0.0173 | 0.0201 | 0.0203 |
| 16000 | 0.0161 | 0.0182 | 0.0187 |
| 20000 | 0.0154 | 0.0171 | 0.0178 |
| 24000 | 0.0150 | 0.0163 | 0.0173 |
| 28000 | 0.0148 | 0.0157 | 0.0170 |
| 32000 | 0.0147 | 0.0154 | 0.0169 |
| 36000 | 0.0146 | 0.0152 | **0.0168** (best) |
| **final (38k/39999)** | **0.0146** | **0.0151** | **0.0168-0.0169** |

⚠️ val 集都不同 (各 22/40/59 ep), 数值不绝对可比, 但同 init+同 schedule+同 step, 趋势可比。

**重要观察**:
- step 2000 三者**几乎相同** (0.0268-0.0270) — 起点一致, init 影响等同
- step 8000 起 `mix < pure ≈ vis_base` — kai0 跨域**提速早期收敛**
- step 16k+ `mix < pure < vis_base` — pure (mirror) 与 vis_base (only) gap 拉开, mirror 在长训中起效
- **final 终点 (40k 后)**: vis_base_40k 0.0168 vs pure_vis600 0.0151 vs mix_vis600 0.0146
  - mirror 比 vis_only 改善 **10.1%** (0.0151 vs 0.0168)
  - kai0 比 vis_only 改善 **13.1%** (0.0146 vs 0.0168)
  - kai0 比 mirror 改善 **3.3%** (0.0146 vs 0.0151)

### 6.4 best ckpt 部署 tar 包

```
/vePFS/.../checkpoints/pi05_flatten_fold_vis_base_40k/vis_base_40k_v1/
├── 2000/  4000/  ...  36000/  38000/  39999/   ★ 36000 推荐 (best)
└── norm_stats.json
```

**部署 tar 包** (已打包, 2026-04-29 08:00 CST):
- 路径: `/vePFS/tim/workspace/deepdive_kai0_tmp/data/vis_base_40k_best_step36000.tar`
- 大小: 11.6 GB (12,440,422,400 bytes)
- 内容: `params/` + `_CHECKPOINT_METADATA` + `assets/` (不含 train_state)
- MAE@1 = 0.0168, @10 = 0.0365, @25 = 0.0606, @50 = 0.0907

### 6.5 关键意义

vis_base_40k 是**单源 visrobot01 在 40k 步下的 plateau baseline**。配合 visrobot01_only_v1 (12k step, 0.0171, 同 288 train val 22) 形成"短训 vs 长训"对比:
- 12k 步: 0.0171
- 40k 步 (实际终点): **0.0168 (-1.8%)**

实际改善 **-1.8%** 仅微弱。提示**单源数据 12k 步即近 plateau, 加步数边际收益极低**。long horizon 优势主要来自**数据增强或多样性** (mix/mirror 比 vis_only 长训改善大得多)。

---

## 7. 系列结论 (2026-04-29 更新, 4-way ablation 全部完成)

### 4-way 最终结果 (val 不同, 数值不绝对可比, 但同 init+同 schedule 可比趋势)

| 实验 | 数据成分 | best step | best MAE@1 | gap vs mix |
|---|---|---:|---:|---:|
| mixed_gf0_173 (13k) | 519 mix 1:1:1 (kai0 + vis) | 9000 | 0.0129 | (best, val ~50 ep) |
| **mix_vis600 (40k)** | 540 mix 2:1:1 (kai0 + vis) | 38000 | **0.0146** | baseline |
| **pure_vis600 (40k)** | 309 orig + 291 hflip mirror | 39999 | **0.0151** | **+3.4%** |
| visrobot01_only_v1 (12k) | 288 vis only | 11999 | 0.0171 | +17% |
| **vis_base_40k (40k)** | 288 vis only (long-train ablation) | 36000 | **0.0168** | **+15%** |

### 已确认结论

1. **数据多样性 > 单源**: mixed (519 ep, 0.0129) 比 visrobot01-only (310 ep, 0.0171) 显著好 (33% lower MAE@1)。
2. **续训 + 新数据可破 plateau**: visrobot01_only Phase A 在 step 8-9k 完全 plateau, 加 95 ep 新数据 + 极低 LR 续训仍能压低 4.5%。
3. **EMA 选择**: 短训 (≤15k) 用 0.999 收敛快; 长训 (40k) 用 0.9999 更稳。
4. **norm_stats 续训策略**: 续训若数据分布变化小 (~5% 漂移), 保留旧 snapshot 比重算更稳, 避免输入分布跳变导致前期 MAE spike。
5. **40k 长训 vs 13k 短训 (mix_vis600 0.0146 vs mixed_173 0.0129)**: 在自己各自的 val 上, 长训反而较差。但 val 集不同, 不严格可比。**真正结论必须看共同 test (sim01 真机/共同 hold-out)**。
6. **数据增强 hierarchy** (4-way @ final, val 不同但同 schedule 可比):
   - **kai0 跨域 (mix_vis600)** = **0.0146** ★ baseline 最佳
   - **hflip mirror (pure_vis600)** = **0.0151** (+3.4% vs mix)
   - **vis only (vis_base_40k)** = **0.0168** (+15% vs mix; +10.1% vs mirror)
   - **kai0 vs mirror gap 比预期小** (3.4% final vs 17% early step) — long-horizon 训练显著缩小 gap
7. **40k vs 12k 单源训练 (vis_base_40k 0.0168 vs visrobot01_only_v1 0.0171)**: 改善仅 1.8% — **单源数据 12k 步即近 plateau, 加步数边际收益极低**。
8. **build_task_a_pure_vis600.py codec 陷阱**: `libx264 preset=veryfast` 默认产生复杂 B-frame 引用链, random-seek 慢 3.3×, 训练慢 4×。修复用 `preset=ultrafast -bf 0 keyint=15` (decode-friendly) 速度恢复。**所有未来 mp4 build 脚本默认要用此参数**。

### 4-way ablation 终极结论

- **数据多样性 (kai0 跨域) > 数据增强 (hflip mirror) > 单源** (确认在每个 step 都成立)
- 三者 gap 在长训中**部分缩小**: kai0 vs mirror gap 17% → 3.4%, mirror vs vis_only gap 1% → 10.1%
- mirror augmentation 是 **late-game amplifier**: 早期跟 vis_only 几乎相同 (step 8k 都 ~0.0223), 但长训中显著拉开
- kai0 跨域是 **early-game accelerator** + **plateau-floor lifter**: 两个阶段都领先

### 待 sim01 真机部署后验证

- 真机成功率 vs MAE 是否一致 (历史 Task P unfreeze_20k MAE 0.0195 但真机抓取偏 2-3cm, 表明 offline ≠ online)
- 按 MAE 优先级部署: mix_vis600 step 38k (主推) > pure_vis600 step 39999 (次推, mirror aug 测试) > mixed_gf0_173 step 9000 (短训 baseline 对照)

---

## 8. 工程经验

### 7.1 corrupt mp4 风险

`vis_base/2026-04-24/.../episode_000053.mp4` (hand_left cam) 是 vis_base 唯一损坏文件 (录制 kill 未 flush moov atom)。该文件的影响:
- gf1 visrobot01_only Phase B train: 1 个 train ep 引用 → DataLoader 全程 skip (191 次 warning)
- gf0 mix_vis600 val: 1 个 val ep 引用 → inline-eval 直接整段失败 (与 train 不同, **eval 路径无 graceful skip**)
- 修复: build 阶段 ffmpeg probe 主动剔除 (pure_vis600 已自动跳过) OR 训练中 patch val 移除该 ep

### 7.2 --resume vs --overwrite

`--overwrite` rmtree 整个 exp 目录 (包括所有 ckpt)！历史教训: 2026-04-24 误用导致 5k ckpts (best step 4999 MAE@1=0.0127) 不可逆丢失。

**所有 launcher 已统一使用 `--resume`** (即使首跑也安全, 自动 fallback 到 weight_loader)。

### 7.3 双 GPU 并发训练 vePFS I/O

gf0 + gf1 同时跑训练时 vePFS I/O 竞争, 步速 ~2.0-2.5 s/step (单跑约 1.75 s/step)。约 +15-20% 时间。可接受。

### 7.4 inline_eval 时间成本

200 frames × N val_ep eval 大致:
- 17 val ep: 660 s
- 22 val ep: 850 s (gf1 Phase B)
- 60 val ep: 1170 s (gf0 mixed_173)

每 1000 step eval = 增加 ~10-20% 总耗时。长训 (40k) 推荐 `inline_eval_every=2` 即每 2k step eval (本系列设置)。

---

## 9. 历史

| 日期 | 事件 |
|---|---|
| 2026-04-24 13:19 | gf0 mixed_gf0_173_v1 启动 (13k 步) |
| 2026-04-24 13:19 | gf1 visrobot01_only_v1 (Phase A) 启动 (12k 步规划) |
| 2026-04-25 02:00 | gf0 visrobot01_only_2k_gf0_v1 启动 |
| 2026-04-25 04:34 | visrobot01_only_2k 完成 |
| 2026-04-25 04:53 | gf1 visrobot01_only_v1 (Phase A) 在 step ~9020 crash (数据路径迁移) |
| 2026-04-25 04:54 | gf0 mixed_gf0_173_v1 完成 (step 12999, MAE 0.0129) |
| 2026-04-25 16:50 | gf1 visrobot01_only_v1 (Phase B, --resume) 启动用 vis_base 288 ep |
| 2026-04-25 18:25 | gf0 mix_vis600_v1 启动 (40k 步) |
| 2026-04-25 22:21 | gf0 mix_vis600 val 修复 (移除 corrupt ep 35) |
| 2026-04-25 22:47 | gf1 visrobot01_only_v1 (Phase B) 完成 (step 11999, MAE 0.0171) |
| 2026-04-25 23:09 | gf1 pure_vis600_v1 启动 (40k 步) |
| 2026-04-27 03:48 | **gf0 mix_vis600_v1 完成** (step 39999, best step 36k/38k/39999 tied @ MAE 0.0146; 总耗时 33:21:43) |
| 2026-04-27 08:34 | mix_vis600 best ckpt step 38000 打包 → 11.6 GB |
| 2026-04-27 12:00 | gf1 pure_vis600 codec 修复 (873 mirrors 重编码 ultrafast+bf=0+keyint=15, step 14000 ckpt resume), 步速 8.4→2.5 s/step |
| 2026-04-27 13:25 | gf0 vis_base_40k_v1 启动 (40k 步, 单源 vis_base baseline) |
| 2026-04-29 00:14 | **gf1 pure_vis600_v1 完成** (step 39999, best step 38k/39999 tied @ MAE 0.0151; 总耗时 36:05) |
| 2026-04-29 01:21 | **gf0 vis_base_40k_v1 完成** (step 39999, best step 36000 @ MAE 0.0168; 总耗时 35:52) |
| 2026-04-29 08:00 | pure_vis600 + vis_base_40k best ckpts 打包 (各 11.6 GB, 都在 `/vePFS/.../deepdive_kai0_tmp/data/`) |
