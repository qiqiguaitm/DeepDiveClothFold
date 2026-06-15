# 跨 Episode 重复度挖掘 → 自动 Milestone / Value(零训练 milestone-value 工作集)

> **Method name: CRAVE** — **C**ross-episode **R**ecurrence **a**s **V**alue **E**stimation
> *(Training-free dense value from what demonstrations repeat. 实现代号 = milestone-value / V2.4。)*

> 一条独立工作线的归档目录。核心命题:**同任务多条 demo 中反复出现的状态 = 任务必经 milestone**;
> 把跨 episode 的统计重复性 → 自动浮现的 milestone → 经 Viterbi-DP 读出为稠密单调 progress value,
> **全程零训练**(frozen DINOv2-small + KMeans + Viterbi-DP,零梯度更新),替代 AWBC 逐帧监督回归(pi0-AE)。
>
> **三支柱**:① 跨 episode 统计重复性揭示任务结构 → ② 重复态 = 自动 milestone → ③ 零训练产稠密 value。
> 已验证:demo 域干净 0→1、撞色衣物兜底、跨天 16/16、rollout 退步+恢复、kai0 GT MAE 0.105;
> **跨数据集强泛化**(新本体 XVLA corr 0.956 / 真实 ALOHA coffee corr 0.988);
> 单 episode 对比监督 pi0-AE(METHOD §3.3b,等价设定):**in-distribution**(kai0_base,同域 base+dagger 挖矿)与 AE 平手且更平滑(corr 0.82,单调 100% vs 52%);**out-of-distribution**(真机 rollout)明显更稳(AE 欠读 end 0.33 + 退步噪声)。

## 文档导航

| 文档 | 作用 |
|---|---|
| [cross_episode_recurrence_value_METHOD.md](cross_episode_recurrence_value_METHOD.md) | **最终方法 V2.4**(9步配方 + 四场景验证 + 否决死路 + 结论)。干净版,先读这个。 |
| [cross_episode_recurrence_value_GENERALIZATION.md](cross_episode_recurrence_value_GENERALIZATION.md) | **跨数据集泛化实证**:XVLA soft_fold(新本体)corr 0.956/100%≥0.7;真实 ALOHA coffee(新任务)corr 0.988/单调100%。配方逐字不改。 |
| [cross_episode_recurrence_value_plan.md](cross_episode_recurrence_value_plan.md) | **完整探索记录**:18 次迭代 + 56 图 + 文献调研 + 所有否决死路的诊断过程。 |
| [awbc_milestone_value_AB_plan.md](awbc_milestone_value_AB_plan.md) | **下游落地 A/B 对照 plan**:A=V2.4 直接当 value 源 / B=蒸馏训 AE,对照已跑的 C=pi0-AE。 |

## 关联资源(目录外)
- 可视化图/预览帧:[`docs/visualization/cross_episode_recurrence_value/`](../../../../visualization/cross_episode_recurrence_value/)(含 `generalization/` 子目录)
- 评估脚本(任意 HDF5 / LeRobot-v3 数据集即插即用):`train_scripts/kai/data/{hdf5,lerobot_v3}_extract_features.py` + `{hdf5,lerobot}_v24_eval.py`
- 上游 AWBC 总纲:[`awbc_implementation_plan.md`](../../../../deployment/strategy/awbc_implementation_plan.md)

## 状态
✅ 方法收口(V2.4)· ✅ 四场景验证 · ✅ 跨数据集强泛化 · ⏳ AWBC A/B 真机对照(plan 已就绪,待执行)
