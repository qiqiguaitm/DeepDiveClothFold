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
| [cross_episode_recurrence_value_plan.md](cross_episode_recurrence_value_plan.md) | **完整探索记录**:迭代 + 图 1-55 + 文献调研 + 所有否决死路的诊断过程;§4.6 = **段间连续化(TCC+DP 连续 value)**子线。 |
| [awbc_milestone_value_AB_plan.md](awbc_milestone_value_AB_plan.md) | **下游落地 A/B 对照 plan**:A=V2.4 直接当 value 源 / B=蒸馏训 AE,对照已跑的 C=pi0-AE。 |
| [value_advantage_methods_comparison.md](value_advantage_methods_comparison.md) | **方法论证对比**:kai0-AE(监督进度差)vs π*0.6-RECAP(RL 分布式回报优势)vs CRAVE(零训练离散),含逐维表 + 分场景结论 + 文献定位。 |
| [CRAVE_roadmap_and_positioning.md](CRAVE_roadmap_and_positioning.md) | **定位与可行方案**:重定位为"零标签结构/技能引擎"(value 是副产品)+ 四组工作流(能做什么/怎么做/对标 SOTA 优势)+ 分阶段安排 + B1 决定性实验。 |
| [frequency_window_params.md](frequency_window_params.md) | **频率相关参数**:不同图像频率需不同窗(lam ∝ fps、时间窗按秒);3Hz/30Hz 落表 + 标定规则 + 离线/在线读出 + 实测(窗标定让抖动降 8×)。 |

## 两条 value 形态
- **离散 CRAVE V2.4**(主交付,METHOD 文档):milestone 阶梯,零训练,跨数据集强泛化。
- **连续 TCC+DP**(plan §4.6,图47-55):端到端/frozen TCC 进度感知特征 + 相似度场 Viterbi-DP + 子bin软期望 → 逐帧连续 value;消 fold 凹口、保留真回退;跨数据集泛化(xvla/vis_base/coffee corr 0.94-1.00)。

## 关联资源(目录外)
- **统一计算库**:`train_scripts/kai/data/crave_value.py`(`FeatureSpace` + `DiscreteValue` + `ContinuousValue`,高内聚低耦合;`verify_refactor.py` 验证与旧实现 bit 级一致)
- 可视化图/预览帧:[`docs/visualization/cross_episode_recurrence_value/`](../../../../visualization/cross_episode_recurrence_value/)(含 `generalization/` 子目录)
- 评估脚本(任意 HDF5 / LeRobot-v3 数据集即插即用):`train_scripts/kai/data/{hdf5,lerobot_v3}_extract_features.py` + `{hdf5,lerobot}_v24_eval.py`
- 上游 AWBC 总纲:[`awbc_implementation_plan.md`](../../../../deployment/strategy/awbc_implementation_plan.md)

## 状态
✅ 离散 V2.4 收口/四场景/跨数据集 · ✅ 连续 TCC+DP 收口(§4.6,跨数据集泛化) · ✅ value 计算重构为统一库(crave_value.py) · ⏳ AWBC A/B 真机对照(plan 已就绪,待执行)
