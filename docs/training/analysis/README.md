# docs/training/analysis — 实验结果分析与归因

> **作用**: 汇总跨实验对比、归因分析、反直觉结果的诊断。与 `history/experiments/` 单实验结果记录不同, 本目录聚焦**多实验联合分析** + **训练动态归因**。
> **建立**: 2026-05-26
> **添加新分析的原则**: 每个文件聚焦一个具体问题, 标题以问题或反直觉发现命名。

---

## 索引

| 分析主题 | 文件 | 创建 | 状态 |
|---|---|---|---|
| **数据量增大反而 MAE 变差** (vis_v2_full 0.0131 vs pure_200 0.0065) — 排除 init 后的训练动态归因 + **2026-05-27 §11 chunk/noise 诊断 + 真机 oscillation 修复路线** | [data_scale_vs_quality_vis_v2_full_vs_pure_200.md](data_scale_vs_quality_vis_v2_full_vs_pure_200.md) | 2026-05-26 | 主线 |
| **base 数据集是否值得预处理?** — 全 19 日期质量扫描 (高频噪声/teleport/end-snap/gripper). 结论: 不值得去噪平滑 (数据已干净); 4-29 gripper "异常" 经分桶证实是正常夹布非半夹 | [base_dataset_preprocess_assessment.md](base_dataset_preprocess_assessment.md) | 2026-05-28 | 结论 |

---

## 与其他目录的边界

| 目录 | 内容 |
|---|---|
| `history/experiments/` | **单实验 results** (一实验一文件, 含训练参数 + 完整 MAE 表 + 决策点) |
| `analysis/` (本目录) | **跨实验对比 + 反直觉归因** (诊断 "为什么 X 比 Y 好/差", 不是单实验记录) |
| `future_plans/plans/` | 待执行的实验计划 |

如果一个实验**单独存在**, 写在 `history/experiments/`. 如果是**两个或多个实验对比+反直觉发现**, 写在本目录.
