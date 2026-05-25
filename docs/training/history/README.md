# Training History — 训练历史记录汇总

> **范围**: 列出所有已完成/进行中实验记录与分析报告。**深度结果排行榜在 [`experiments/00_training_history.md`](experiments/00_training_history.md)。**
> **更新**: 2026-05-25。

---

## 1. ⭐ Master 历史汇总 (必看入口)

| 文档 | 用途 |
|---|---|
| [00_training_history.md](experiments/00_training_history.md) | **全实验 best MAE 排行榜 + per-step 曲线 + freeze 策略约定**。每次实验完成都要更新这里。 |

---

## 2. 单实验记录 (按主题分组)

### Task A — 7000 ep 系列 (kai0_official base + dagger 上 finetune)

| 实验 | 状态 | 文件 |
|---|---|---|
| new_mixed_pure2_1800_6000 (老 SOTA) | ✅ done | [task_a_new_mixed_pure2_1800_6000_results.md](experiments/task_a_new_mixed_pure2_1800_6000_results.md) |
| new_pure_1800_new_norm_base_mixed1 (两阶段) | ✅ done | [task_a_new_pure_1800_new_norm_base_mixed1_results.md](experiments/task_a_new_pure_1800_new_norm_base_mixed1_results.md) |
| new_pure_200_new_norm | ✅ done | [task_a_new_pure_200_new_norm_results.md](experiments/task_a_new_pure_200_new_norm_results.md) |
| new_pure2_1800_new_norm_js | ✅ done | [task_a_new_pure2_1800_new_norm_js_results.md](experiments/task_a_new_pure2_1800_new_norm_js_results.md) |
| new_smooth_800_new_norm | ✅ done | [task_a_new_smooth_800_new_norm_results.md](experiments/task_a_new_smooth_800_new_norm_results.md) |
| pure_1200_new_norm | ✅ done | [task_a_pure_1200_new_norm_results.md](experiments/task_a_pure_1200_new_norm_results.md) |
| visrobot01 mixed 600 (jiu vis) | ✅ done | [task_a_visrobot01_mixed_600.md](experiments/task_a_visrobot01_mixed_600.md) |
| mixed visrobot01 1500 (扩展 vis) | ✅ done | [mixed_visrobot01_1500_experiment.md](experiments/mixed_visrobot01_1500_experiment.md) |

### Task P — 全解冻 vs action-only 对照

| 实验 | 状态 | 文件 |
|---|---|---|
| Unfreeze 20k v2 | ✅ done | [task_p_unfreeze_20k_v2_results.md](experiments/task_p_unfreeze_20k_v2_results.md) |
| Unfreeze 8k vs 20k analysis | ✅ done | [task_p_unfreeze_8k_20k_analysis.md](experiments/task_p_unfreeze_8k_20k_analysis.md) |

### KAI0 复现 (mixed_1 / Task A 开源分析)

| 文档 | 状态 | 文件 |
|---|---|---|
| kai0_mixed_1 复现结果 | ✅ done | [kai0_mixed_1_results.md](experiments/kai0_mixed_1_results.md) |
| KAI0 Task A 开源分析 | reference | [kai0_task_a_opensource_analysis.md](experiments/kai0_task_a_opensource_analysis.md) |
| kai0 Task A 训练复现日志 | reference | [training_reproduction_log.md](experiments/training_reproduction_log.md) |
| Task A 训练范式对比 (single-stage vs two-stage) | reference | [training_paradigm_comparison.md](experiments/training_paradigm_comparison.md) |

### AWBC

| 实验 | 状态 | 文件 |
|---|---|---|
| gf0 AWBC baseline v2 | ✅ done | [gf0_awbc_baseline_v2_results.md](experiments/gf0_awbc_baseline_v2_results.md) |

### Conditioning × Action Representation Ablation

| 实验 | 状态 | 文件 |
|---|---|---|
| XVLA conditioning 三方法 (hard / soft / action-cond) | 🟢 tracking | [xvla_conditioning_methods_results.md](experiments/xvla_conditioning_methods_results.md) |
| Conditioning × Action Representation 2×2 ablation | 🟢 tracking | [conditioning_vs_action_representation_ablation.md](experiments/conditioning_vs_action_representation_ablation.md) |
| Norm stats ablation (apr28 450) | ✅ done | [norm_stats_ablation_apr28_450.md](experiments/norm_stats_ablation_apr28_450.md) |

### 数据集诊断

| 文档 | 状态 | 文件 |
|---|---|---|
| 数据集诊断报告 | reference | [dataset_diagnostic_report.md](experiments/dataset_diagnostic_report.md) |

---

## 3. 工具与 Reference

| 用途 | 文件 |
|---|---|
| 训练 CLI 用法笔记 | [training_cli_notes.md](experiments/training_cli_notes.md) |
| W&B 监控指南 | [wandb_monitoring.md](experiments/wandb_monitoring.md) |
| 数据集动态 workflow | [dynamic_dataset_workflow.md](experiments/dynamic_dataset_workflow.md) |

---

## 4. 如何阅读这些文档?

1. **找 best MAE**: 直接 [00_training_history.md](experiments/00_training_history.md) §1 TL;DR 排行榜。
2. **复盘某个实验**: 先看本 README 找文件名 → 打开对应 `experiments/<name>_results.md`。
3. **看某个 ablation 主题**: §2 子表已按主题分组 (Task A / Task P / Conditioning / AWBC / KAI0)。
4. **看 CLI/W&B 用法**: §3 reference 区。

## 5. 新加 history 文档

实验跑完出 eval 后:
1. 把 `future_plans/plans/<name>.md` 移到 `history/experiments/<name>_results.md` (加 `_results` 后缀)
2. 更新本 README 表格添一行
3. 更新 [00_training_history.md](experiments/00_training_history.md) 排行榜
