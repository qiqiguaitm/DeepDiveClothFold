# Task A mix_b6000_p1200 (mixed_1 init) 训练结果

> **实验**: uc01 实验1 — `pi05_flatten_fold_mix_b6000_p1200_init_mixed_1`
> **状态**: ✅ **完成** (50,000 步, ended 2026-05-07 12:24:37 CST, 训练时长 26h17m)
> **目的**: 用 base6000 + pure1200 大规模混合数据集 + mixed_1 init, 评估 Task A 大数据下的微调上限
> **对比实验**: 实验3 (gf0 同 dataset, init from pi05_base, 100k 步) — 控制变量为 init weights

---

## 1. 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_mix_b6000_p1200_init_mixed_1` |
| Model | pi05 (Pi0Config(pi05=True)) |
| Init | `Task_A/mixed_1/params` (MA-merged + previous Task_A 微调) |
| Dataset | `Task_A/self_built/mix_b6000_p1200/base` (6000 base + 1200 pure mix, 共 7200+ episodes) |
| Val | `Task_A/self_built/mix_b6000_p1200/val_self_built` (30 ep paired orig+mirror) |
| Prompt | "Flatten and fold the cloth." |
| `use_delta_joint_actions` | False |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50k, decay_lr=1.5e-6 |
| EMA decay | 0.9999 |
| Steps | 50,000 |
| Batch | 128, fsdp_devices=8 |
| Save | every 2,000 step, keep_period=10,000 |
| inline_eval | every 2 saves (= 每 4k 步), 200 frames |
| Seed | 42 |
| Server | uc01 (A800-SXM4-80GB ×8, driver 550.144.03, CUDA 12.4) |

---

## 2. 完整 inline-eval MAE@{1,10,25,50} 曲线

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 4000  | 0.0161 | 0.0393 | 0.0779 | 0.1345 | (start) |
| 8000  | 0.0141 | 0.0315 | 0.0571 | 0.0948 | -12.4% |
| 12000 | 0.0127 | 0.0283 | 0.0506 | 0.0815 | -9.9% |
| 16000 | 0.0123 | 0.0275 | 0.0492 | 0.0783 | -3.1% |
| 20000 | 0.0120 | 0.0269 | 0.0483 | 0.0766 | -2.4% |
| 24000 | 0.0116 | 0.0263 | 0.0474 | 0.0753 | -3.3% |
| 28000 | 0.0114 | 0.0260 | 0.0468 | 0.0743 | -1.7% |
| 32000 | 0.0111 | 0.0256 | 0.0462 | 0.0735 | -2.6% |
| 36000 | 0.0110 | 0.0254 | 0.0460 | 0.0732 | -0.9% |
| 40000 | 0.0109 | 0.0253 | 0.0458 | 0.0729 | -0.9% |
| 44000 | **0.0108** | 0.0252 | 0.0457 | 0.0728 | -0.9% |
| 48000 | **0.0108** | 0.0252 | 0.0457 | 0.0728 | 0.0% |
| 49999 | **0.0108** | 0.0252 | 0.0457 | 0.0728 | 0.0% |

**Best**: step 44000 (首次达 plateau), MAE@1 = **0.0108**

---

## 3. 训练动力学分析

### 3.1 Loss/grad 健康性

| step | loss | grad_norm | param_norm |
|---:|---:|---:|---:|
| 18100 | 0.0095 | 0.0615 | 1806.5115 |
| 30000 | ~0.0050 | ~0.05 | ~1807.0 |
| 49900 | 0.0039 | 0.0508 | 1807.3181 |

- 0 NaN / 0 grad explosion 全程
- param_norm 从 1804.34 (init) → 1807.32 (final) 稳步增长 (+2.97), 模型参数移动幅度合理
- 收敛速度: 每 4k 步 MAE@1 -3% ~ -10% 早期, 28k 后 < -2%, **40k 后 plateau**

### 3.2 数据规模优势

| 指标 | 实验1 (mix_b6000_p1200) | task_a_pure_1200_new_norm (#24) | task_a_new_pure_1200 (#25) |
|---|---:|---:|---:|
| 数据规模 | ~7200 ep | 1142 ep | 1143 ep |
| best MAE@1 | **0.0108** | 0.0145 | **0.0104** (gf1, step 38k crash) |
| best step | 44k | 49999 | 38000 |

- mix_b6000_p1200 数据规模是 pure_1200 的 ~6x, 但 MAE 只比 pure_1200 (1142 ep) 小 25.5%, 比 new_pure_1200 (#25, 真正高质量 -new 数据) 反而高 3.8%
- **结论**: 单纯堆数据不如使用高质量 -new 限定数据 — 数据质量 > 数据量

### 3.3 Plateau 检测

```
step 40k: 0.0109
step 44k: 0.0108 (-0.9%)
step 48k: 0.0108 (0.0%)
step 49999: 0.0108 (0.0%)
```

**自 step 44k 后训练完全 plateau**, 后 6k 步无改进。建议未来同类实验:
- 30k 或 40k 步即可 (节省 ~30% 训练时间)
- 或加 cosine restart 跳出 local minimum

---

## 4. 与 Task P unfreeze_20k 对比 (mixed_1 init)

| 实验 | 任务 | data ep | best step | best MAE@1 |
|---|---|---:|---:|---:|
| mix_b6000_p1200 (本实验) | Task A (fold) | ~7200 | 44k | **0.0108** |
| task_a_pure_1200_new_norm | Task A (fold) | 1142 | 49999 | 0.0145 |
| task_a_new_pure_1200_new_norm | Task A (fold) | 1143 | 38k | 0.0104 |
| pi05_pick_place_box_kai0_unfreeze_20k | Task P (pick-place) | 84 | 20k | TBD |

---

## 5. 最佳 Checkpoint 信息

- **路径** (uc01): `/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_mix_b6000_p1200_init_mixed_1/task_a_mix_base6000_pure1200_new_norm_base_mixed_1/{20000,30000,40000,49999}/`
- **保留 ckpts**: 20k, 30k, 40k, 49999 (keep_period=10000)
- **推荐**: step 44k (首次 plateau) — 但仅 20k/30k/40k/49999 保留, 实际可用最佳 = step 40k 或 49999 (同等 MAE=0.0108-0.0109)
- **norm_stats**: 与 ckpt 同目录 `norm_stats.json`

---

## 6. 部署 Checklist

- [ ] Pack ckpt 49999 per `kai0/checkpoints/README.md` Type A flat 格式
- [ ] norm_stats: dataset path = `kai0/data/Task_A/self_built/mix_b6000_p1200/base/norm_stats.json` (走 default repo_id 模式)
- [ ] sim01 部署测试

---

## 7. 经验教训

1. **数据质量 > 数据量**: mix_b6000_p1200 (~7200 ep) 的 best MAE 比同 hparams 的 new_pure_1200 (1143 高质量 -new ep) 还差 3.8%
2. **40k 后 plateau**: 未来同类实验可缩短到 40k 步, 节省 30% 训练时间
3. **mixed_1 init 健康**: 大数据 + mixed_1 全程 0 NaN, 与小数据集 (84 ep) 形成鲜明对比 (Task P/v2 因 seed=42 数据采样在 step 700 NaN)
