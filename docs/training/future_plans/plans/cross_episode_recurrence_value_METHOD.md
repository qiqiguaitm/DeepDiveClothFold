# 跨 Episode 重复度挖掘 → 自动 Milestone / Value(最终方法 V2.4)

> **本文档 = 最终可靠方法的方法 + 效果 + 结论(干净版)。** 完整探索过程(所有迭代、否决的死路、诊断图、文献调研)见 [cross_episode_recurrence_value_plan.md](cross_episode_recurrence_value_plan.md)(探索记录,§4.4.6-4.4.18 共 56 图)。
> **状态**:✅ V2.4 验证完成,四场景全绿(2026-06-13)。可直接产 AWBC milestone-value 标签。
> **上游**:AWBC pipeline([awbc_implementation_plan.md](../../../deployment/strategy/awbc_implementation_plan.md))。**用途**:零训练 milestone-value 替代/对照现有 pi0-AE 监督 value(§4.3 决策点)。

---

## 1. 核心方法

**假说**(已验证):同任务多条 demo 中**反复出现的状态 = 任务必经 milestone**;从跨 episode 结构挖 value,零训练替代 AWBC 的逐帧监督回归。

**V2.4 = 三路特征 + 增分子 coverage 修正 + 进度均匀分桶 + GMM 多模式别名 + 端点锚 + 连续性 Viterbi DP**。每个组件都是被实验逼出来的(对应一个失败/发现,见探索文档)。

---

## 2. 实现配方(9 步,照此实现)

**输入**:demo episodes(lerobot:top_head 视频 + observation.state 14 维)。特征 GPU 提,挖掘+value CPU。

| # | 步骤 | 方法 | 关键参数 | 解决的问题 |
|---|---|---|---|---|
| 1 | 特征 | **三路** raw-DINOv2 patch-mean ⊕ armmask-DINOv2 ⊕ proprio(state+Δstate) | DINOv2-small,3Hz,各 L2 归一拼接(384+384+28) | 单路撞色失效→三路冗余兜底 |
| 2 | 聚类 | KMeans | k=96(N≈500),seed=0 | — |
| 3 | **coverage 修正** | 增分子 `(hits+miss)/N`(保分母一致) | miss=起点晚于该簇的 ep 数;P_start=ep 前3帧最近簇 tpos 中位 | partial-start 数据 left-truncation 偏差 |
| 4 | **milestone 选择** | 进度均匀分桶(非 top-K!) | tpos 每 0.1 区间选 cov_n 最高 2 簇 → ~20 | 前段动作多样致 coverage 天然低→空洞 |
| 5 | P_k 标定 | 门控首入时刻中位数 | 命中=驻留≥2帧 ∨ margin≤0.8 | 单帧误配/边界混淆 |
| 6 | 多模式别名 | GMM(1-2,BIC)检测双峰簇→多 value | 两峰间距>0.35 | 首尾别名(初态≈终态) |
| 7 | 端点锚 | start 原型(首帧 KMeans8)→P=0 / end→P=1 | 时间门控:start 仅前 30%,end 仅后 40% | 初始/终止语义锚定,value 标度 |
| 8 | **value 读出** | 连续性 Viterbi DP | λ=8,硬边界 V[首帧]=0,末帧奖 bin20,中值 W9 | 别名消歧+空桌平滑+退步,全局单调连续 |
| 9 | 退步(rollout) | DP 连续性双向自然实现;含失败/重试数据时 value 可升可降 | (demo 域 GT 单调,退步段≡0) | 多轮/失败 rollout 的退步信号 |

---

## 3. 四场景效果验证(全绿)

### 3.1 demo 域:干净 0→1 阶梯

vis 5-26 ep7(抓取→摊开→叠好):value 0-29s 在 0-0.15(抓取/调整)→ 阶梯爬升 → 70s 后到 1.0(叠好),前段无误判。milestone 进度均匀(前段 10/20)。

![demo milestone](../../../visualization/cross_episode_recurrence_value/vis0526_v24_milestones.png)
![demo ep7 value](../../../visualization/cross_episode_recurrence_value/vis0526_ep7_v24_value.png)

### 3.2 撞色稀有衣物:三路兜底

vis 5-20 ep37(**橙色**衣物,armmask 单路会误吃→误判 0.99):V2.4 三路下 **开头 0.15**(不误判),全程 0→1,前段 10/20 均匀。raw-DINOv2 路保留撞色衣物判别力。

![撞色 milestone](../../../visualization/cross_episode_recurrence_value/vis0520_ep37_v24_milestones.png)
![撞色 value](../../../visualization/cross_episode_recurrence_value/vis0520_ep37_v24_value.png)

### 3.3 kai0 GT 量化(有 stage_progress_gt)

| 方法 | MAE↓ | Pearson↑ | τ↑ |
|---|---|---|---|
| 旧 DP(两路 top-K) | 0.113 | 0.906 | 0.862 |
| **V2.4(三路+增分子+分桶)** | **0.105** | **0.928** | 0.841 |
| pi0-AE(监督,循环论证*) | 0.054 | 0.971 | 0.881 |

V2.4 MAE/Pearson 优于旧 DP;τ 略低(前段加 milestone,动作多样使排序稍噪声)。*pi0-AE 的低 MAE 是拟合自己训练目标 stage_progress_gt,非更懂状态(探索文档 §2.12/4.4.7);真优劣在鲁棒性+标注成本。

### 3.4 rollout:退步 + 恢复

autonomy 真机 3 轮叠衣(轮1中途衣物被拿走、轮2叠完被弄乱):V2.4 value **两次回落到 0**(~100s/~175s,正好两个轮次边界,衣物摊开=退步)+ 每轮爬升,round3 到 **1.0**。退步+恢复结构清晰。

![rollout V2.4](../../../visualization/cross_episode_recurrence_value/rollout_v24_value.png)

### 3.4b 跨数据集泛化(配方逐字不改,零训练)

同一冻结配方套两个全新数据集:**XVLA soft_fold**(新本体/相机/布料,168ep)corr(value,时间) mean **0.956**、≥0.7 占比 **100%**;**互联网真实 ALOHA `lerobot/aloha_static_coffee`**(全新长程"做咖啡"任务,50ep)corr mean **0.988**、单调 100%。证明学的是「跨 demo 重复结构=进度」通用规律而非 kai0 过拟合。详见 [cross_episode_recurrence_value_GENERALIZATION.md](cross_episode_recurrence_value_GENERALIZATION.md)。

![coffee 泛化](../../../visualization/cross_episode_recurrence_value/generalization/coffee_sync_ep30_preview.png)

### 3.5 跨天鲁棒(V2.3 已验证,V2.4 一致)

milestone 单天挖掘应用到 8 个日期(跨月):16/16 正常 0→1(探索文档 §4.4.14 图47)。V2.4 在 5-18/5-20/5-26 多数据集表现一致。

---

## 4. 否决的死路(实证排除,勿重试;详见探索文档)

| 死路 | 失败原因 | 探索章节 |
|---|---|---|
| 段内 value 细化(双锚欧氏/测地/簇内2-NN/**cosine-softmax**) | 实测 τ 0.841→0.805(MAE 持平 0.105→0.103,平滑不变);frozen 局部信号噪声≈增益,势函数不变性下不改最优策略,AWBC 关心 advantage 排序故净伤 τ。软加权(脱 DP)更垮 MAE0.179/τ0.665 | §4.4.15/19 |
| 因果/时序硬约束 | milestone 顺序是统计概率非强因果,错杀合法变体 | §4.4.16 |
| task-specific(夹爪规则/布料占比门槛) | 不泛化,只适叠衣 | §4.4.14 |
| min-max 归一化 value | 损单调(0.632→0.542) | §4.4.13 |
| K==M(全簇皆 milestone) | 退化为计时器,失败段标正 | §2.12 |
| top-K coverage 选 milestone | 前段空洞(partial+动作多样压低) | §4.4.16/17 |
| coverage 减分母修正 | 破坏对比一致性(分母不一);改增分子 | §4.4.17 |
| TCC 冻结特征 / 多路分歧消歧 | 塌缩 / 多模态一致别名躲过分歧 | §2.4/4.4.16 |

---

## 5. 结论 + 下一步

**方法已收口、验证充分、配方可复现**:demo 域干净 0→1、撞色衣物兜底、跨天 16/16、rollout 退步+恢复、kai0 GT MAE 0.105(≥旧 DP)。

**核心贡献(可发表点)**:① recurrence→自动 milestone→AWBC 标签全链;② partial-start 数据的 left-truncation coverage 修正(增分子保一致性);③ 进度均匀分桶替代 top-K 频率。

**下一步**:① **全量打标 + AWBC 对照训练**(§4.3,最终用途,对照 pi0-AE 标签,真机为终判;A/B 对照执行 plan → [awbc_milestone_value_AB_plan.md](awbc_milestone_value_AB_plan.md):A=直接当 value 源/B=蒸馏训 AE,对照已跑的 C=pi0-AE);② 可选增强:外观冗余合并(后段去重)、**soft-DP(soft-DTW/Drop-DTW/GTCC)内建连续 progress**(段内细化的正解——把连续性内建进对齐而非事后插值,原生处理 idle 帧;但 GTCC 需训练→违背零训练,故备选,见 §4.4.19)、TCC 端到端学习 progress-aware 度量(根治别名)。

**TCC 互补线(非竞品,探索文档 §2.4.3)**:TCC 不比 τ,而提供聚类做不到的"逐帧连续学习对齐"。已验证 **App① 锚位消歧**——TCC 对齐-进度把聚类 rollout 唯一残留(高位误吸:f400 团布 0.92→0.26、f3100 空桌 0.94→0.36)压掉,可即接 V2.2 rollout 标注复核高位帧;近期可做 **App② 失败定位**(rollout→demo 对齐,比退步阈值更原生)、**App④ OOD 门控**(对齐残差);**端到端微调 backbone 已验证(§2.4.4,本地 A100)**:只解冻末 4 块,TCC held-out τ 0.718→**0.842**、MAE 0.137→**0.107**,**追平聚类主线**(τ 0.841/MAE 0.105,Pearson 反超)——TCC 升级为与聚类并列的第二条可交付 value 路线(连续 vs 离散)。frozen 上限假设成立并已捅破;App③(连续亚阶段)随之从远景转近期。

> **段内细化已实证否决(§4.4.19)**:事后用前后 milestone 相似度/距离插值细化 value,实测 τ 0.841→0.805 而 MAE 几乎不变——frozen 局部信号噪声≈增益,势函数不变性下不改最优策略,AWBC 关心 advantage 排序故净伤。文献(CRR/势函数/LLE)与实测一致:局部插值理论合法但无净增益。staircase 是更优基线。

---

## 附录 — 关键脚本与产物

**脚本**(`train_scripts/kai/data/`):`v24_complete_milestone.py`(完整 V2.4 挖掘+ep value)· `bucketed_milestone.py`(进度分桶)· `coverage_correction_compare.py`(增分子对比)· `extract_masked_features.py`(armmask 特征)· generic raw 提取。

**特征缓存**(`temp/`):`tcc_{vis0526,vis0520,kai0}_armmask/feat_cache`(armmask)· `tcc_{vis0526,vis0520,kai0,autonomy}_raw/feat_cache`(raw-DINOv2)。

**效果图**(`docs/visualization/cross_episode_recurrence_value/`):`vis0526_v24_milestones.png` · `vis0526_ep7_v24_value.png` · `vis0520_ep37_v24_*.png` · `rollout_v24_value.png`。

**探索记录**:完整 56 图 + 18 次迭代 + 文献调研在 [cross_episode_recurrence_value_plan.md](cross_episode_recurrence_value_plan.md)。
