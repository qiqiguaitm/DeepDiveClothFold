# 结果对比：LMWAM (Arm M) vs LaWAM baseline (Arm B) · LIBERO libero_10

> 2026-07-15 本机 eval。原始数据：[`data/lmwm_vs_lawam_libero10_20k.json`](data/lmwm_vs_lawam_libero10_20k.json)（机器可读，per-task + 失败模式 + 未训练尾巴）。
> 报告正文见 [`LMWM_report_full_draft.md`](LMWM_report_full_draft.md) §3；本文件为 §3 的原始数据源与失败模式分析。

## 实验设置（公平 A/B）

| 项 | 值 |
|---|---|
| 基准 | LIBERO **libero_10**（10 长周期任务 × 50 trials = 500 episodes） |
| step | **20000**（两臂同 step；M 原训 25000、B 至 23856，取共同 @20000 保证公平） |
| seed | 0（**单 seed**，见保留意见） |
| 硬件 | 本机 A100 · lawam env policy server + libero env client |
| **Arm M** (LMWAM) | LMWM decoder + **milestone+1** 目标 + swap_teacher · `armM_milestone_20k/steps_20000` · run `v5/20260715_021132` |
| **Arm B** (baseline) | released LaWM decoder + **t+7** 目标 + no-swap 自训 · `armB_baseline_20k/steps_20000` · run `v4/20260715_021734` |

## 总体成功率

| 方法 | SR (libero_10) | succ/total |
|---|---|---|
| **Arm B** baseline (t+7) | **96.40%** | 482/500 |
| **Arm M** LMWAM (milestone+1) | **92.20%** | 461/500 |
| Δ (M − B) | **−4.20 pt** | −21 ep |

> ~~参考：LaWAM released ckpt 本机复现 libero_10 = 98.0%（不同 recipe/step，仅作锚点，不入公平对比）。~~
>
> **[2026-07-19 更正]** 上面这个锚点**不可用**,且曾被错误地与论文 98.6% 相比：
> - 论文 **98.6% 是四 suite 平均**;libero_10 = Long suite 的论文值是 **97.0%**。
> - 该 98.0% 只有 **10 trials/task、100 个 episode、seed=0**,聚合二项 SE ≈ ±1.4pt
>   → **98.0±1.4 与论文 97.0 一致**,不构成复现差距。本表各臂是 50 trials/task × 500 ep,口径不同。
> - 正在用 50 trials/task × 4 变 seed 重测该 ckpt,得到同口径锚点后再更新。

**当前结论：Arm M（milestone）低于 Arm B（baseline）4.2 点。** 这是一个诚实的负/中性结果，不宣称 milestone 有效。

## 逐任务对比

| task | 描述 | M_SR | B_SR | Δ | M失败 | B失败 | #ms | 未训练尾巴 |
|---|---|---|---|---|---|---|---|---|
| 0 | put both alphabet soup and tomato sauce in basket | 0.98 | 0.98 | +0.00 | 1 | 1 | 11 | 0.13 |
| 1 | put both cream cheese box and butter | 0.96 | 1.00 | −0.04 | 2 | 0 | 9 | 0.33 |
| 2 | turn on the stove and put the moka pot on it | 1.00 | 1.00 | +0.00 | 0 | 0 | 6 | 0.20 |
| 3 | put the black bowl in the bottom drawer | 0.94 | 0.96 | −0.02 | 3 | 2 | 7 | 0.08 |
| 4 | put white mug on left plate and yellow-white mug right | 1.00 | 0.98 | +0.02 | 0 | 1 | 8 | 0.15 |
| 5 | pick up the book and place it in the back caddy | 0.98 | 1.00 | −0.02 | 1 | 0 | 4 | 0.42 |
| **6** | **put white mug on plate + chocolate pudding to right** | **0.68** | **0.84** | **−0.16** | 16 | 8 | 4 | **0.60** |
| 7 | put both alphabet soup and cream cheese in cart | 1.00 | 1.00 | +0.00 | 0 | 0 | 10 | 0.59 |
| **8** | **put both moka pots on the stove** | **0.86** | **1.00** | **−0.14** | 7 | 0 | 8 | 0.05 |
| **9** | **put yellow-white mug in microwave and close it** | **0.82** | **0.88** | **−0.06** | 9 | 6 | 8 | 0.52 |

**7/10 任务基本打平（Δ∈[−0.04,+0.02]）；4.2 点差距的 71% 来自 task 6 + task 8。**

## 失败模式分析（已验证）

1. **"后期骤降"是假象**：eval 按 task 0→9 固定顺序，日志 `success_rate` 是累计均值；难任务(6,8,9)恰在后半段，造成累计曲线在 7/10 处跳水。逐任务无时序退化。
2. **失败=超时**：M 与 B 的**所有**失败 episode 的 `num_actions` 均为 **550±0 = max horizon**。无一发散/早期失手——M 不是"崩"，是最长多阶段任务上"跑不完"。
3. **未训练最终 milestone 段假设**（用户提出）：训练对构造 `p1_libero_milestone_pairs.py` 对最终 milestone(`m=M-1`, 无 `m+1`)的帧全部丢弃 → 世界模型对"任务最后一段"零监督。验证：
   - **task 6/9 成立**：末 milestone 占 episode **60% / 52%**（CRAVE 欠分割/疑似 cummax 棘轮，末 milestone 起于 ~40%，第二子任务整段落入未训练区）。
   - **task 8 不成立**：末段仅 5%，分割良好；失因是**两个相同 moka 壶别名**（milestone 3→6 跳变、4/5 塌缩）+ 单 seed 波动。
   - 跨 10 任务 Spearman(Δ vs 尾巴占比)= **−0.21**（弱，条件性效应）；反例 task 7 尾巴 59% 却持平 → 大未训练尾巴是**必要不充分**（尾部子任务也难时才致命）。

## 诚实保留 / 待补

- **单 seed**：task 8 的 B 50/50 vs M 43/50、task 6 的 8 vs 16 需 **多 seed** 确认显著性（50 trials，SE≈5%）。
- **未直接观测卡点**：尾巴分析证明"未训练区大且完成态在其中"，但需**回放 task 6 失败 episode 轨迹**直接确认卡在第二子任务。
- **M 训练步数**：两臂 @20000 公平，但不能排除 milestone 信号需更多步。

## 可落地修复（针对已验证缺口）

1. **最终 milestone 建对**：`p1_...pairs.py` 不再丢弃 `m=M-1` 帧，改为目标=episode 末帧(达标态)或该 milestone 代表帧，给最后一段"向目标收敛"监督。
2. **改善末段分割**：查 cummax 棘轮是否仍在 task 6/5 生效（#ms 仅 4），必要时上 Viterbi readout（BUG_AUDIT MODERATE-4）。

## volc 任务提交式 eval（交叉验证）

北京 Robot-North-H20 队列当前 0 任务；提交式 eval 路径(v10 PYTHONPATH 方案)未取回终态。本机结果为当前权威值，volc 侧留作后续交叉验证。
