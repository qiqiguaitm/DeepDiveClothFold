# CRAVE 文档索引

> **CRAVE** = **C**ross-episode **R**ecurrence **a**s **V**alue **E**stimation
> *(零训练 · 跨 episode 重复度 → milestone → 稠密 value)*

> 🧭 **agent 先读这个**:[**HISTORY.md**](HISTORY.md) = 唯一的「**当前权威 vs 已淘汰方案**」总索引。
> CRAVE 迭代多、`docs/` 与 `experiments/` 混着最终与淘汰方案;**动手/复用代码前先查 HISTORY**,避免误用旧脚本/旧图/旧结论(如阶梯-teacher 蒸馏、img-only 聚类、DINOv3-H、KMeans-K10、固定滞后 等均已被取代)。
>
> **当前权威文档**(其余为历史/过渡):[final_architecture](final_architecture.md)(离线收口,scope=kai0 任务A value)· [multitask_value](multitask_value.md)(在线 multitask)· [online_readout_route](online_readout_route.md)(在线因果)· [decoder_benchmark](decoder_benchmark.md)(解码)· [cross_dataset_validation](cross_dataset_validation.md)(泛化)。
> ⭐ **普适方向(阶段代表帧/边界/WM目标/OOD)当前权威 = v2 复现密度场 r(o)**:[recurrence_field_architecture_v2](recurrence_field_architecture_v2.md)(r-谷=边界、r-脊=代表帧,无聚类;详源在 `lmvla/lmwm/docs/RECURRENCE_UNIVERSAL_goals_and_roadmap.md`)。

## 文档结构

### 总览

| 文档 | 作用 |
|---|---|
| **[CRAVE_overview](CRAVE_overview.md)** | **总体介绍(入口)** — 方法、架构、效果、代码结构、未来方向。先读这个。 |
| **[CRAVE_positioning_and_roadmap](CRAVE_positioning_and_roadmap.md)** | 定位 / 场景 / roadmap:前沿地图 + vs SOTA + 工作项 A/B/C/D + 分阶段安排 |
| **[related_work_map](related_work_map.md)** | **四轴谱系图(价值 / 世界模型 / 中间表示 / 交付算子)**:~20 篇归位(VIP/LIV/RECAP/χ0/SARM2/SVM/STEAM/SRPO · Fast-WAM/LaWAM/Ctrl-World · RT-Traj/RT-Aff · CFGRL/REACH/DiffusionNFT…)+ CRAVE 护城河 + 跨轴空位 + ①→④ 闭环接口。投稿 positioning 一页图。 |

### 核心方法

| 文档 | 作用 |
|---|---|
| **[cross_episode_recurrence_value_METHOD](cross_episode_recurrence_value_METHOD.md)** | **离散主线 V2.4**:9 步配方 + 四场景验证 + 否决死路 |
| **[viterbi_computation](viterbi_computation.md)** | Viterbi-DP 计算流程详解 (含频率参数标定附录) |
| **[online_readout_route](online_readout_route.md)** | **🆕 在线读出路线(完全因果)**: 三档(对称/非对称 forward-DP + 因果 GRU 蒸馏)追平离线 Viterbi;GRU 留出 corr vs GT 0.955=离线水平, 0.735M 参数, 零未来。含分布式双头(progress+advantage)追平/反超 KAI0-AE |
| **[multitask_value](multitask_value.md)** | **🆕 Multitask 在线 value**: 一个共享 0.72M 模型跨 kai/vis/coffee/xvla 4 任务 corr 0.95-0.98;0.75M 绰绰有余、大模型反而过拟合、task-embed 冗余、共享对小任务正迁移 |
| **[sym_adaptive_vote](sym_adaptive_vote.md)** | Viterbi-free 替代: 对称自适应投票 (在线,3 参数,corr=0.974 vs Viterbi;粗milestone+3Hz下退化, 见 online_readout_route) |
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
| **[HISTORY](HISTORY.md)** | 🧭 **迭代史 & 淘汰方案总索引**:当前权威 canonical 脚本/文档清单 + 已淘汰方案 ledger(方法→为何淘汰→取代→旧脚本路径)。复用旧代码前必看。 |
| **[STATUS](STATUS.md)** | 单页 TODO:已收口 / 已否决 / 未做可做 |
| **[decoder_benchmark](decoder_benchmark.md)** | 🏆 解码器统一基准(最优解码方案):检索(最近真实帧)语义保真+锐度双赢 cos0.84/940;合成保真封顶 ~0.47 |
| **[milestone_concept_and_universal_next](milestone_concept_and_universal_next.md)** | 🧭 概念再审视:recurrence 是频率代理非本质;更合理 milestone=瓶颈+变点+不可逆;普适 milestone+1=离散功能化 latent world model |
| **[greedy_vs_maxprod_aliasing](greedy_vs_maxprod_aliasing.md)** | 🔬 诊断:成功数据下 greedy≠max-product(43% 一致)= milestone 混叠指纹;rollout 的 "milestone +1" 贴合 greedy |
| **[em_hmm_negative_result](em_hmm_negative_result.md)** | ❌ 已否决:EM-HMM 统一概率框架(hmm-cluster collapsed in 768D) |
| **[clustering_method_comparison](clustering_method_comparison.md)** | 🔬 KMeans vs GMM vs HDBSCAN vs Overcluster+Otsu:自适应K+时间纯度对比(KMeans不是缺陷是feature);HDBSCAN高维崩塌 |
| **[archive/](archive/)** | 历史归档:探索记录索引 · 网站大纲 · 连续 value 路线 |

## 状态

✅ 离散 V2.4 收口 · ✅ 跨数据集泛化 · ✅ 可视化整理 (77 files) · 🔄 AWBC A/B 真机对照 · 📋 CRAVE-RPO plan 已立
