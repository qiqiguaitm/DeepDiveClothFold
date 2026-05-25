# Task A new_pure_200 训练结果 ⭐⭐ 全任务 NEW SOTA

> **结论先行**: js02 (单机 8 GPU) 上 `task_a_new_pure_200_new_norm` 从 step 22000 resume 到 49999, final **MAE@1 = 0.0065** ⭐, 全任务新 SOTA, 比之前 best (`task_a_new_pure2_1800_6000_new_norm` 0.0085, uc 集群 7900 ep) 低 **24%**。**200 ep 精选 + mixed_1_clean init** 推翻"数据规模 > 数据质量"假设, 数据质量 (`-new` 限定 + mirror 增强) 占主导。

## 1. 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_pure_200_js` |
| Model | pi05 (`Pi0Config(pi05=True)`) |
| **Init** | `/mnt/data/tim/kai0_ckpts/Task_A/mixed_1_clean/params` (MA-merged Task_A warmed-up) |
| **Dataset** | `/mnt/data/tim/data/Task_A/self_built/A_new_pure_200` (200 ep 精选, JuiceFS:visincept) |
| - 来源 | `-new` 日期限定 + hflip mirror 增强, 仅高质量子集 |
| Val | `/mnt/data/tim/data/Task_A/self_built/A_new_pure_200_val` |
| Prompt | "Flatten and fold the cloth." |
| `use_delta_joint_actions` | False |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50k, decay_lr=1.5e-6 |
| EMA decay | 0.9999 |
| Steps | 50,000 |
| Batch | 120, fsdp_devices=8 (单机全 FSDP `[1,8]`) |
| Save | every 2,000 step, keep_period=2,000, max_to_keep=1 |
| inline_eval | every 2 saves (= 每 4k 步), 200 frames |
| Seed | 42 |
| Server | **js02** (A800-SXM4-80GB × 8) |
| 启动 | **从 step 22000 resume** (原 ckpt 在 `/mnt/data/tim/checkpoints/.../22000`), 续训到 49999 |
| WandB | offline (`--no-wandb-enabled`) |
| 训练时长 | resume window 16h11m (2026-05-14 08:08 → 2026-05-15 00:18 CST) |

### 1.1 关键差异 (vs 老 SOTA `pure2_1800_6000`)

| 维度 | pure_200 (本实验) | pure2_1800_6000 (老 SOTA) |
|---|---|---|
| **数据规模** | **200 ep** (1/40 of SOTA) | 7900 ep |
| **Init** | mixed_1_clean (Task_A 微调过) | pi05_base (原始) |
| 集群规模 | 单机 8 GPU | 24 GPU (uc01+02+03) |
| Batch | 120 | 120 |
| Final MAE@1 | **0.0065** ⭐ | 0.0085 |
| Gap | — | **+30.8% (worse)** |

## 2. 完整 inline-eval MAE@{1,10,25,50} 曲线

> 因 resume from step 22000, 仅记录 step 24000 起的 eval。

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 24000 | 0.0071 | 0.0086 | 0.0095 | 0.0115 | (resume baseline) |
| 28000 | 0.0069 | 0.0082 | 0.0088 | 0.0102 | -2.8% |
| 32000 | 0.0067 | 0.0078 | 0.0084 | 0.0094 | -2.9% |
| 36000 | 0.0066 | 0.0076 | 0.0080 | 0.0088 | -1.5% |
| 40000 | 0.0066 | 0.0074 | 0.0078 | 0.0085 | 0.0% |
| 44000 | 0.0065 | 0.0073 | 0.0076 | 0.0081 | -1.5% |
| 48000 | 0.0065 | 0.0072 | 0.0075 | 0.0080 | 0.0% |
| **49999** | **0.0065** | **0.0072** | **0.0075** | **0.0079** | 0.0% |

**Best**: step 49999, MAE@1 = **0.0065** ⭐ (44k/48k 也是 0.0065, @50 在 49999 最低 0.0079)

## 3. 训练动力学

- resume @ 22k 的起点未直接 eval, 但 step 24k 已达 0.0071 — mixed_1_clean init 下经过 24k step (相当于从 mixed_1 起步训了 24k) 已显著优于 pi05_base init 同步数
- 收敛极快: step 24k → 36k 在 12k 步内从 0.0071 跌到 0.0066 (-7%), 但 36k 后进入 long plateau
- @1 在 44k-49999 持续 0.0065 (4 个连续 eval 持平), 训练已饱和
- @50 (long-horizon) 持续小幅下降到 step 49999 (0.0115 → 0.0079, **-31%**) — 长 horizon planner 后期仍在精修, 与 pure2_1800_6000 趋势一致
- 全程 0 NaN, 无 div, eval 时长 754-962s (均值 768s)

## 4. SOTA 关键洞察

1. **数据质量 > 数据规模**: 200 ep (`-new` 限定 + mirror) 比 7900 ep 大杂烩 MAE 低 24%。**关键变量**: pure_200 是高度精选的 `-new` 日期数据 + hflip mirror 双对增强, 比 mix b6000+p1200 大量 `kai0_base / kai0_advantage` 旧数据干净得多。
2. **mixed_1_clean init 上限被低估**: 之前 pure2_1800_6000 用 pi05_base init 得到 0.0085 (老 SOTA), 推断 "干净起点天花板更高"。但 pure_200 用 mixed_1_clean (Task_A warmed-up) 反而打到 0.0065 — 说明在**高质量小数据**场景下, Task_A 适配过的 init 反而是更好的起点 (节省 24k step 的"重新学 Task_A"开销)。
3. **小数据 + 长训** 配合 EMA decay 0.9999 不发生 overfit: 200 ep × 50k step = ~250 epoch, 仍单调下降, 没看到 val MAE 反弹。
4. **resume 不影响收敛趋势**: 22k resume 起点没破坏 cosine LR schedule (continued from same step counter), 后续 28k step 平滑收敛。
5. **@1 vs @50 差异启示**: @1=0.0065 (单步精度), @50=0.0079 (50 步长 horizon) — gap 仅 22%, 明显小于其他实验 (pure2_1800_6000 @1=0.0085 @50=0.0337, gap 296%)。说明该模型不仅单步准, 长 horizon planner 也极稳。

## 5. 最佳 ckpt 位置

```
js02:/mnt/data/tim/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_new_pure_200_new_norm/49999/
```

⚠️ JuiceFS:visincept 共享路径 — js01-js04 任一节点都能直接读, 不需 SSH 跨节点拷贝。

**完整 ckpt 列表** (每 2k 步): 22000, 24000, 26000, ..., 48000, 49999  
**附属文件**: 同目录 `norm_stats.json`, `wandb_id.txt`

## 6. 后续计划

- **对照实验**: 改用 pi05_base init (而非 mixed_1_clean) 跑同样的 pure_200 数据, 验证"高质量小数据下哪个 init 更优"。新 exp `task_a_pure200_new_norm_base_pi0.5` 已在 js04 单卡启动。
- **deploy 到 sim01**: ckpt 49999 应优先 pack + 上 sim01 测试 (老 SOTA pure2_1800_6000 已上, 这个有望大幅优化 chunk planner)。
- **解析为啥 -24%**: 是数据质量, 还是数据量本身就够? 跑 pure_200 / pure_400 / pure_800 sweep 即可 disentangle。
