# 研究方向探讨:普适 milestone 定义 + LMWM↔VLA 深融合(2026-07)

跳出 CRAVE 特例,调研"milestone+1 预测"的本质,以及 LMWM 与 π0.5 更深融合的架构路径。
基于两份文献 survey(普适 subgoal / VLA-fusion)+ 本仓库代码(`lewm_vision_encoder.py`、
`lam_model.py`、`pi0_pytorch.py`)。附核心文献 arXiv id 可查。

---

## 0. 一句话结论
- **目标定义**:k-means+Viterbi 的簇中心不是正典;正典是 **temporal-distance/progress**(milestone = 距离 level-set,milestone+1 = 近一档的可达状态)。CRAVE 有个几乎同款的去 hack 版:**UVD**(单 episode、无聚类、零训练)。
- **多峰性**:我们 best-of-8 只回收 +0.02 **不是"多峰天花板低",而是目标被 medoid 抹平成单峰**——测的是塌缩后的目标。真多峰要 per-episode 目标 + 生成式。
- **grid vs pool**:文献几乎一致 **预测 grid,不要 pool**。pool 丢"哪里"(物体/夹爪位、遮挡、几何),LaWAM 自己的评测**依赖 per-patch 对应**,pool 根本没法验证。pool 只配当"辅助 reasoning token / reward",不能当唯一空间条件。
- **共享编码器/KV**:方向对、是趋势(WorldVLA/Being-H0.7/π0.5 自己就复用 prefix-KV)。但**要把"目标空间"和"预测器来源"拆开**:目标仍用冻结 DINOv3-H grid;预测器**从 VLA 已算好的 prefix hidden states 取源**,不再跑第二个编码器。这正好落实"DINOv3-H+CRAVE 当离线数据处理"的想法。

---

## 1. milestone+1 的普适定义(Point A / B1 / B2a / B2b)

### 1.1 正典 backbone = temporal-distance / goal-conditioned value
跨 RL/IL/VLA 三条线,"普适"都落在**学一个时序距离/进度标量**,而非视觉聚类:
- contrastive RL = GCRL(2206.07568):对比表示的内积**就是**目标条件 value = 时序邻近度。
- Quasimetric RL(2304.01203)、TLDR(2407.08464)、GAS(2506.07744):把"下一个 subgoal"变成时序距离空间里的**最短路搜索**。
- **milestone = 该距离的一个 level-set;milestone+1 = 朝目标近一个距离档的可达状态**。连续、自监督、跨本体、无需固定 37、无需聚类。

→ 直接回应 **B2a(时间不稳定)**:把条件从"绝对时间(1.6s)"换成"剩余进度/距离档",时长长短的方差天然被吸收——**不稳定时长是 milestone 打赢 fixed-horizon 的理由,不是缺点**。

### 1.2 CRAVE 的去 hack 直系:UVD(2310.08581)
同前提(冻结视觉特征编码进度),但**单 episode 内**找"embedding→goal 距离不再单调下降的相变点"当 subgoal——**无 k-means、无跨集 Viterbi、零训练**。我们的聚类+单调分配 = 它的**跨 episode 平均版**。TCC(1904.07846)、GVL(2411.04549 VLM 零样本读进度)同族。
→ **可直接 benchmark 对标 UVD**,证明我们的聚类步是否真带来增益。

### 1.3 离散化:VQ latent-action ≫ k-means(直击 B2b)
SuSIE(2310.10639 子目标图)、VPP(2412.14803 预测 latent)、LAPA(2410.11758)、Genie(2402.15391)、LAPO(2312.10812):
- 它们的目标是**生成未来观测/latent**,不是簇中心;
- 离散化(若有)是**和 dynamics 联合学的 VQ code**(代表"下一步真正变了什么"= 转移),不是对冻结特征 post-hoc k-means。
→ 簇中心是"可辩护但非正典"的量化;VQ latent-action 是更普适的"milestone 身份"。

### 1.4 多峰性真相(修正 FINAL_REPORT 的"天花板")
**B2b 洞察正确且尖锐**:milestone+1 里 current 帧对**精确未来像素**说得少(远、多峰),对**"下一个是哪个 milestone"说得多**(类别/语义)。所以 milestone 预测器 ≈ **分类器(下一个 milestone 是谁)+ 生成器(它长什么样)**,而非我们现在的 extrapolator(warp 当前 grid)。
- "簇中心是有价值提示" = 预测出 milestone-index 后的**检索结果**。
- **为何 best-of-8 只 +0.02**:我们目标 = Viterbi medoid = **已经把多峰抹成单峰**,VAE 无峰可采。→ FINAL_REPORT §4"多模态天花板 +0.02"测的是**塌缩后的目标**,不是真多峰上限。要测真多峰须换 per-episode 目标。

### 1.5 → coarse-to-fine 分解(A/B1/B2b 的共同落点)
把 milestone+1 拆成:
1. **离散身份**(下一个 milestone 是谁)——便宜、单峰、可 pool/CLS/VQ-code 表达;
2. **空间残差**(这个 milestone 在**本 episode** 长什么样)——难、多峰、**必须 grid**、生成式(VAE/flow)。

---

## 2. grid vs pool(Point C)

**结论:预测 grid,不要退成 pool。**
- 主流全预测 grid 且是刻意的:DINO-WM(2411.04983,保 patch 才可 plan)、Genie、**LaWAM(2606.15768,我们的孪生)**、UniVLA(2505.06111)、DeltaWorld(2604.04913)。
- **LaWAM 自己的验证依赖 grid**:取机械臂 patch 的 DINO 特征,和预测 subgoal map 的**每个 patch** 算 cos 对应——pool 向量根本没法这么验,更别说驱动精细落点。
- Survey 2510.16732 给了正好的分类学:Global Latent Vector vs Spatial Latent Grid;pool"省算力但丢 occlusion / object permanence / geometry-aware planning"。
- pool 成单向量还有个陷阱:相邻帧 pool 几乎不变 → persistence 逼近 1.0,grid-cos 看着高其实**没信息量**(和我们 CRAVE 簇可视化"pooled 是连续流形"一致)。
- **pool 什么时候可以**:只当**辅助 reasoning token / reward**、且下游策略**自己还在看空间**时(VIP/R3M 当 reward;GRIF/CLIP-goal;Being-H0.7 的 K=16 query 是 reasoning 接口非空间子目标)。
→ **可选增强**(非替换):在 grid 之上加**几个** pooled reasoning query(Being-H0.7 式)做便宜的高层融合;若 256 token 太重,用 delta-token(DeltaWorld)压 token 但保空间。你想要的"好融合"应由 §3 解决,不是靠 pool 降维。

---

## 3. LMWM↔VLA 深融合(Point D / E)

### 3.1 方向对,是趋势
共享 backbone / KV / prefix 是活跃且收敛的趋势,你的直觉有充分先例:
- **π0.5 本身就复用 prefix-KV**:PaliGemma VLM 把 image+language 编成 **prefix KV-cache**,可训练 Gemma action expert 在 suffix 上 attend 进去;**Knowledge Insulation(2505.23705)**把 expert 梯度**隔离**出 VLM backbone(否则退化语义)——我们已用 KI,是正确护栏。
- **WorldVLA(2506.21539)**:单 AR transformer,**共享词表 + 共享 KV-cache**,world-frame 和 action 同序列。
- **Being-H0.7(2605.00078)**:单序列单次前向,**部署无第二编码器、无视觉 rollout**——正是你要的"复用前向而非跑第二个编码器"的最干净存在性证明。
- WLA(2606.05979,meta-queries)、RynnVLA-002(2511.17502)、τ0-WM/GigaWorld(共享 video-diffusion backbone)、**Privileged Foresight Distillation(2604.25859,训练时共享注意力、推理时蒸馏掉未来分支)**——和"训练共享/部署省掉"极接近。

### 3.2 但有真实的特征张力 → 拆"目标空间" vs "预测器来源"
- **VLA-JEPA(2602.10098)**最锋利:**重建/外观丰富特征 vs 动作相关特征冲突**。像素/重建目标诱发"appearance bias"(纹理/光照/背景,高方差低控制相关),故它**在冻结 V-JEPA2 latent 里预测、和 VLM 分开**,避免信息泄漏与外观捷径。
- KI(2505.23705)同理:动作梯度进语义 backbone 会退化。
- 共识:**align/condition,别 merge 编码器**。

**→ 干净设计(线程过张力):**
- **目标空间 = 保留冻结 DINOv3-H grid**(预测进这个空间)。别 retarget 到 VLM 特征——否则踩 VLA-JEPA 的外观偏置/泄漏。
- **预测器来源 = VLA 已算好的 prefix hidden states / KV**,而非第二次编码器前向。把"部署再跑一遍 DINOv3-H"换成一个**轻量 dynamics head 读 π0.5 prefix → 预测 DINOv3 grid**。= LaWAM 的 Alternate-DiT,但**source 自共享 prefix**。省下部署第二次编码,又不污染目标。
- **这正好落实你的原话**:"DINOv3-H+CRAVE 当一种数据处理方式"(**离线 label 工厂**,可重),"LMWM 架构上直接用 VLA 相同编码器"(**在线预测器活在 VLA prefix 上**)。张力被"离线目标 / 在线来源"这条缝解决。

### 3.3 三条硬约束(否则静默 bug)
1. **必须有 projector/adapter**:不能把 VLM 的 KV 直接当 DINOv3 的 KV 用(embedding 不同、attention 权重不同;WorldVLA 需要统一词表)。
2. **注意力 mask 要重设计**:WorldVLA 核心教训——subgoal/action token 进序列后,action⊥prior-action、frame 用 causal;复用 cache ≠ 复用 mask。
3. **RoPE / 位置一致**:future patch 和 current patch 空间位重叠,须按时间 index 偏移,否则模型混淆"现在/未来"patch——**和我们 v3 PTS 那种静默错位同类的坑**。

### 3.4 仓库里已有的半成品
`kai0/src/openpi/models_pytorch/lewm_vision_encoder.py`:π0.5 已有 `vision_encoder="lewm"` 旁路,用 **DINOv3-L + OctCompactor** 把 SigLIP 768 dense token 换成 **15 个 object-centric token**(3 view ×(1 CLS+4 obj)),在 `embed_prefix` 零侵入替换;`pi0_pytorch.forward` 本就带 `past_key_values`。→ "换 VLA 视觉前端 + 压 token 进 PaliGemma"的管线**已跑通**,是 §3.2 的现成脚手架(但它换的是 observation 编码器,不是 WM 预测器)。

---

## 4. 建议的下一版架构(把 1–3 缝起来)
1. **目标空间**:冻结 DINOv3-H grid(不 pool、不 retarget)。
2. **目标分解(coarse-to-fine)**:milestone+1 = 离散身份(VQ code / 分类,便宜单峰)+ episode 特定空间残差(grid,生成式 VAE/flow,还多峰)。
3. **目标定义**:用 temporal-distance/progress(UVD / contrastive-GCRL)替代 k-means+Viterbi;自监督、无固定 37、天然吸收时长方差。CRAVE 保留为离线 label 工厂 + 对照。
4. **融合**:预测器 source 自 π0.5 prefix hidden states(共享前向、adapter 桥接、重设 mask、RoPE 偏移),预测 DINOv3-H grid 目标;保留 KI;部署单次编码。
5. **验收闸**:per-patch 对应 cos + 下游 SR(仍是最大缺口)。

## 5. 待定 / 需拍板
- 先做**便宜验证**(UVD 对标 + per-episode 目标重测多峰性,不改架构),还是直接上 §4 的融合改造(重)?
- temporal-distance 目标是否值得替换 CRAVE,还是 CRAVE 作离线 label、在线换 VLA 空间即可?
- 部署单次编码的融合改造依赖 π0.5 训练侧改动(embed_prefix + mask + RoPE),要不要先在 `lewm` 旁路上做最小 PoC?

### 核心必读
LaWAM 2606.15768(孪生)· WorldVLA 2506.21539(KV/mask)· Being-H0.7 2605.00078(单前向无二编码器)· VLA-JEPA 2602.10098(特征张力)· UVD 2310.08581(去 hack milestone)· KI 2505.23705(π0.5 prefix-KV 机制)· survey 2510.16732(grid-vs-pool 分类学)
