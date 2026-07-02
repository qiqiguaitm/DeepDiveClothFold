# CRAVE 文档索引

> **CRAVE** = **C**ross-episode **R**ecurrence **a**s **V**alue **E**stimation
> *(零训练 · 跨 episode 重复度 → milestone → 稠密 value)*

## 文档结构

### 总览

| 文档 | 作用 |
|---|---|
| **[CRAVE_overview](CRAVE_overview.md)** | **总体介绍(入口)** — 方法、架构、效果、代码结构、未来方向。先读这个。 |
| **[CRAVE_positioning_and_roadmap](CRAVE_positioning_and_roadmap.md)** | 定位 / 场景 / roadmap:前沿地图 + vs SOTA + 工作项 A/B/C/D + 分阶段安排 |

### 核心方法

| 文档 | 作用 |
|---|---|
| **[cross_episode_recurrence_value_METHOD](cross_episode_recurrence_value_METHOD.md)** | **离散主线 V2.4**:9 步配方 + 四场景验证 + 否决死路 |
| **[viterbi_computation](viterbi_computation.md)** | Viterbi-DP 计算流程详解 (含频率参数标定附录) |
| **[sym_adaptive_vote](sym_adaptive_vote.md)** | **🆕 Viterbi-free 替代**: 对称自适应投票 (在线,3 参数,corr=0.974 vs Viterbi) |
| **[milestone_centroid_decoding](milestone_centroid_decoding.md)** | 簇中心解码 + 解码器对比 + 标准配置 (含 centroid config 附录) |
| **[encoders](encoders.md)** | 编码器作用 / 选型 / 对比:DINOv3 vs Wan-VAE |

### 验证

| 文档 | 作用 |
|---|---|
| **[cross_dataset_validation](cross_dataset_validation.md)** | 跨数据集验证 (XVLA 新本体 / coffee 真 ALOHA),含早期三路版泛化附录 |
| **[value_advantage_methods_comparison](value_advantage_methods_comparison.md)** | 机理对比:kai0-AE vs π*0.6-RECAP vs CRAVE |

### 落地与实验

| 文档 | 作用 |
|---|---|
| **[awbc_milestone_value_AB_plan](awbc_milestone_value_AB_plan.md)** | AWBC A/B 对照执行 plan |
| **[crave_rpo_minimal_validation_plan](crave_rpo_minimal_validation_plan.md)** | CRAVE-RPO 最小验证 |

### 元文档

| 文档 | 作用 |
|---|---|
| **[STATUS](STATUS.md)** | 单页 TODO:已收口 / 已否决 / 未做可做 |
| **[em_hmm_negative_result](em_hmm_negative_result.md)** | ❌ 已否决:EM-HMM 统一概率框架(hmm-cluster collapsed in 768D) |
| **[archive/](archive/)** | 历史归档:探索记录索引 · 网站大纲 · 连续 value 路线 |

## 状态

✅ 离散 V2.4 收口 · ✅ 跨数据集泛化 · ✅ 可视化整理 (77 files) · 🔄 AWBC A/B 真机对照 · 📋 CRAVE-RPO plan 已立
