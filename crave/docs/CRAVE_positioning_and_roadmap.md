# CRAVE 定位、场景与 roadmap

> CRAVE 在 VLA / 世界模型 / 价值学习前沿中的定位 + 能做什么 × 怎么做 × 对标 SOTA 优劣 + 可做项验证 + 分阶段 roadmap。
> 合并自原 `CRAVE_roadmap_and_positioning` / `CRAVE_frontier_positioning_and_scenarios` / `CRAVE_doable_items_verification`(2026-06-16)。
> 方法见 [METHOD](cross_episode_recurrence_value_METHOD.md);与监督/RL value 的机理对比见 [方法对比](value_advantage_methods_comparison.md);A/B 落地见 [AB_plan](awbc_milestone_value_AB_plan.md)。

---

## 0. 重新定位(一切方案的前提)

**CRAVE 的不可替代资产 = milestone 图 + 零标签技能切分;value 是副产品。**
- 离散 value 阶梯**紧对齐机械臂动作基元**(milestone = 基元交界/子目标)→ CRAVE 实际在做**无监督跨 episode 技能切分**。实测 milestone 比时间多解释 **2×** 的动作方差(R² 0.43 vs 0.22),是动作相关的技能相位,不只是计时器。
- 因此**不要**把 CRAVE 当 RECAP 的标量 value 平替(那条路 RECAP 严格更强);把它当**结构/技能引擎 + RL 冷启 + 零标签数据工具**。
- 对标:VIP/LIV/TOPReward/GVL 给的是**连续标量** progress/reward,**都没有**可检视的离散 milestone/技能结构;RECAP 给真回报但**黑箱 + 贵**。CRAVE 独占"零训练 + 离散结构 + 跨 episode 重复 grounding"。

---

## 1. 前沿地图与 CRAVE 的结构性优势

### 1.1 前沿地图(与"价值/奖励"相关的部分)

| 方向 | 代表工作 | 与 value/reward 的关系 |
|---|---|---|
| **VLA 策略 + 自我改进** | π0/π0.5/**π\*0.6-RECAP** ([2511.14759](https://arxiv.org/abs/2511.14759))· OpenVLA · GR00T · [openpi](https://github.com/Physical-Intelligence/openpi) | RECAP 用**分布式 MC 回报 value** 算 advantage 做优势条件 BC,闭合"从经验改进"环 |
| **零训练 value(VLM)** | **GVL** ([2411.04549](https://arxiv.org/abs/2411.04549))· TOPReward | GVL = VLM 对**打乱帧做时序排序** → 逐帧完成度;零样本仅需任务文本 |
| **零训练 value(对比/生成)** | VIP ([2210.00030](https://arxiv.org/abs/2210.00030))· LIV ([2306.00958](https://arxiv.org/abs/2306.00958))· ViVa | 预训练一次→零样本逐帧标量进度 |
| **世界模型(RL-in-imagination)** | Genie Envisioner · Cosmos Policy · Ctrl-World ([2510.10125](https://arxiv.org/abs/2510.10125)) | 学到的模拟器里做 RL **必须有 reward/value 信号** |
| **真机 RL(样本高效)** | **HIL-SERL** / SERL ([2401.16013](https://arxiv.org/abs/2401.16013)) | 人标 reward classifier + SAC;**瓶颈是 reward 来源** |
| **离线 RL / 优势条件 BC** | CRR · AWAC · RECAP | 先有 advantage 再筛/加权数据 |

**共性洞察**:几乎所有"自我改进 / RL / 世界模型"路线的**真正瓶颈都是 value/reward 信号的获取成本**——RECAP 要数千次真机 rollout + 失败标注;HIL-SERL 要人标 reward classifier;world-model RL 要 reward 头。**CRAVE 的全部价值就压在"零标签零训练地提供这个信号"上**。

### 1.2 CRAVE 的结构性优势(别人都没同时具备的组合)

1. **零训练 + 零标签 + 零 VLM/API**:frozen DINO(默认 DINOv3-H)+ KMeans + Viterbi-DP,纯 CPU 可跑,可扩百万级离线数据。
2. **离散技能结构(milestone 图),非仅标量**:milestone 比时间多解释 2× 动作方差;VIP/LIV/TOPReward/GVL 都只给连续标量。
3. **重复性是信号而非噪声**:GVL 明确在"重复/次优轨迹"上失效;CRAVE 恰恰靠跨 episode 重复挖结构 → 大批同任务 demo 上是优势区。
4. **确定性、可复现、可检视**:无 VLM 随机性;milestone 簇可视化/可审计(簇间流转 2D/3D)。
5. **在线可因果化**:固定滞后 Viterbi,零训练拿到 corr 0.94 的在线 value(频率窗按 fps 标定,见 [frequency_window_params](viterbi_computation.md))。
6. **质量已逼近监督**:kai0_base ep2047 对真 stage_progress_gt corr **0.865**(监督 pi0-AE 0.897),零标注。

**结构性短板(诚实)**:① 需**同任务 demo 集**(不能像 GVL 单视频/纯文本零样本);② 无结果信号(不能区分自信但错的动作,不能超越示教);③ 跨任务/跨本体零样本弱(GVL 强);④ value 是"demo 流形进度"代理,非真回报。

### 1.3 最直接对标:CRAVE vs GVL(两个零训练 value,反向取舍)

| 维度 | **GVL**(VLM 在上下文) | **CRAVE**(重复几何) |
|---|---|---|
| 信号来源 | VLM 世界知识 + 帧时序排序 | 跨 episode 统计重复 |
| 输入需求 | **仅任务文本**(可单视频/零样本) | **需同任务 demo 集** |
| 跨任务/跨本体零样本 | **强**(300+ 任务) | 弱(需各域挖矿) |
| 成本 | 大 VLM 逐视频 API,贵 | frozen 小模型 + 聚类,**极廉,可扩百万级** |
| 输出结构 | 连续标量完成度 | **离散 milestone 技能结构 + 标量** |
| 重复/次优轨迹 | **失效**(官方局限) | **优势区**(重复=信号) |
| 失败/退步 | 概率式成功检测 | 结构性退步(value 回落) |

→ **不是竞争是互补**:少样本/零样本/跨任务/语义用 GVL;大批同任务 demo + 要便宜 + 要离散结构 + 要确定可复现用 CRAVE。二者可串联(GVL 跨任务冷启 → CRAVE 在目标任务大数据上精修结构)。

---

## 2. 工作项:能做什么 × 怎么做 × 对标优势(含验证状态)

> 验证状态图例:✅已验 · ❌已验·否决 · ⏳待跑 · ⏸搁置(高 GPU)。快速核验详情见本节各项。

### A 组 · 立即可做(低成本,与现有 AWBC 闭环)

**A1 自动子任务切分 → AWBC `prompt_from_task`** ⏳待跑
- 怎么做:milestone 边界切每条 demo 成基元段 → VLM 描述边界帧命名("grasp corner"/"fold left")→ 写 task 串 → 接现成 `prompt_from_task`。切分零成本(B1 已证 milestone=技能相位);命名需 VLM(~20 次/任务,廉价)。
- 对标优势:vs 人工子任务标注/LLM 凭空分解——CRAVE 切分 **grounded 在本机器人数据的真实重复结构**,零人工、与该本体动作一致。

**A2 零标签数据工具(keyframe / 失败定位 / dedup)** ✅keyframe 平凡可做 / ⚠️失败定位部分可做
- 怎么做:milestone 边界帧 = keyframe 导出(零成本);帧到最近簇残差 = OOD/异常;value 卡在 milestone k = 卡在基元 k 的失败定位。
- 边界:同 B2——**粗异常**(value 掉)可,**细微失败**(on-manifold)不可。
- 对标优势:VIP/LIV 标量 reward 定位不了 WHERE;监督切分要标签。CRAVE 免费给可归因结构。

### B 组 · 核心研究(补最大软肋:无 action/无结果信号)

**B1 基元→milestone 转移挖掘 → RL-free 基元级 advantage** ✅**已验**【决定性实验】
- 怎么做:用现成 milestone 模型 + parquet `action` 列。每条 episode 在每个 milestone 转移 k→k+1 处取动作段 → 聚类/刻画 → 算"该动作类推进 milestone 的可靠度"。脚本 `crave_milestone_action_pilot.py`。
- **核验结果**:R²(action|milestone)=**0.43** vs R²(action|时间分桶)=**0.22**(2×)→ milestone 捕捉动作相关技能结构而非计时器;但转移帧动作变化仅 1.10× 非转移 → milestone 是**视觉状态(技能相位)边界,非动作不连续点**。
- 含义:**支持**"按 milestone 条件化动作"(分层/基元信号);**不支持** keyframe-硬动作切分。
- 对标优势:RECAP 要数千次真机 rollout + 失败标注才拿到 action-aware 优势;此法零 RL 零标签拿到(相关非因果的)action 级信号,且在可解释的基元粒度。⚠️ 边界:相关非因果(milestone 在动作 A 后推进 ≠ A 好)。

**B2 终点可达性 + OOD 残差 → 弱成败信号** ❌**已验·否决**
- 怎么做:没到终止 milestone 簇的 episode = 弱失败;`CRAVE_value × 到达终点指示` → 粗 advantage。脚本 `crave_b2_failure_signal.py`。
- **核验结果(否决)**:① 末值<0.7 仅 **2%**,corr(未完成度, AE-neg)=**0.13**(弱);② AE-neg 帧残差 **0.981** vs AE-pos **0.979**(几乎相同,不可分),高残差帧里 AE-neg 仅 6%。
- 含义:**细微 dagger 失败是 on-manifold**(像 demo 的合理布料态),残差/终点都抓不到。CRAVE 只能抓**粗失败**(布料被拿走/全脱轨 → value 自己掉,无需 B2)。→ "廉价补 neg 洞"不成立,**确认 CRAVE 无失败信号的根本局限**;真 neg 仍需 RL/结果信号(C2)或人标。降级为"仅粗失败/OOD 场景成功检测"。

### C 组 · 模型化 / 规模化(高 GPU,搁置)

**C1 蒸馏成在线分布式离散 value 头** ⏸搁置
- 怎么做:小 frozen-feature MLP + **分布式 201-bin CE 头(RECAP 式,非 scalar+MSE)**,用 CRAVE 标签训。
- 对标优势:vs kai0-AE(scalar+MSE,OOD 欠读压到 0.27)——分布式更校准、零人工标;vs RECAP——无需 RL。去 cache 依赖、对未见状态平滑泛化。
- 状态:因 causal-DP 已 corr 0.94 而非必要;留待资源到位。

**C2 CRAVE 冷启 V + 少量真机 rollout RL 微调(exceed-demonstrator 路径)** ⏸搁置
- 怎么做:CRAVE value 当 RECAP pipeline 的冷启 baseline V(替代昂贵 MC value 预训练)→ 少量自主 rollout 成败做 advantage 修正 → 优势条件 BC。
- 对标优势:RECAP value 预训练贵;CRAVE 零成本替代 → 低成本走通"从经验改进/超越示教",这是 CRAVE 唯一能碰到"超越示教"的途径。需真机 RL。

### D 组 · 持续性

**D1 增量挖矿 + 漂移监控 + 域自适应** ⏳待设计
- 怎么做:增量 KMeans 吸收新 episode、milestone split/merge;覆盖率/漂移指标告警"该重挖";自动按目标域选/加权挖矿集。
- 对标优势:把 vis0526-vs-dagger 那种**手动挖矿域错误制度化消灭**;自我维护 vs 一次性挖矿。

---

## 3. 场景 × SOTA × CRAVE 优缺点

### 场景 A · 大规模离线数据集 value/advantage 打标(AWBC / 离线 RL)— **CRAVE 主场**
- **SOTA**:GVL(VLM-VOC)、RECAP-value(需 RL)、监督 pi0-AE(需人标)。
- **CRAVE 优**:零标注零训练、可扩百万级(GVL 的 VLM API 成本在大数据上爆炸)、离散结构可直接喂 AWBC `prompt_from_task`、in-dist 质量已逼近监督(0.865 vs 0.897)、对重复 demo 是优势区。
- **CRAVE 劣**:需同任务 demo 集;无失败结果信号(neg advantage 弱)。
- **判定**:大批同任务离线打标 = CRAVE 主场([AB_plan](awbc_milestone_value_AB_plan.md) 正在验)。

### 场景 B · RL 冷启 baseline V / 世界模型 imagination 的 reward
- **SOTA**:RECAP 分布式 MC value(贵)、world-model 内置 reward 头、HIL-SERL 人标 reward classifier。
- **CRAVE 优**:零成本给 dense 进度 value 当冷启 baseline V(替代昂贵 MC 预训练/省人标 classifier),再用少量真机成败做 advantage 修正 = "低成本可超越示教"路径;在 world-model imagined rollout 里当零标签 dense reward。
- **CRAVE 劣**:本身不含结果信号,需叠加稀疏成败(B2 已证只抓粗失败)。
- **判定**:当 RL/世界模型的廉价 reward 前端——高杠杆,但需配弱成败信号。

### 场景 C · 子任务/技能分割 — **CRAVE 独占**
- **SOTA**:GVL 进度(连续,无边界)、UVD/AWE 类切分(多需训练或逐任务调)。
- **CRAVE 优**:milestone = 跨 episode 一致的离散技能相位(R² 证明动作相关),帧精确 + 跨 episode 一致;零训练免费切分 → 喂分层 / world-model 子目标 / AWBC prompt。
- **CRAVE 劣**:边界是视觉状态边界,非尖锐动作不连续(转移仅 1.10× 更 eventful)→ "keyframe/硬基元"用途弱,"技能相位条件化"用途强。
- **判定**:零标签一致技能相位划分 = CRAVE 独占(GVL 给不了离散一致边界)。

### 场景 D · 数据 curation / 质量过滤 / 去重
- **SOTA**:GVL 的 VOC(Value-Order Correlation)分数。
- **CRAVE 优**:milestone 覆盖/到达终点率 = 免费质量分,极廉可全量扫(GVL VOC 要逐集 VLM);milestone 表示天然支持去重/检索。
- **CRAVE 劣**:语义判断不如 VLM。
- **判定**:大数据廉价初筛用 CRAVE,语义精筛用 GVL(互补)。

### 场景 E · 成功检测 / OOD / 失败定位
- **SOTA**:GVL-SD、专用 success classifier、HIL-SERL reward classifier。
- **CRAVE 优**:到终止 milestone 簇 = 成功代理;到最近簇残差大 = OOD/异常;value 卡在 milestone k = 卡在第 k 个技能相位的失败定位(可归因)。
- **CRAVE 劣**:无语义,细粒度成功判定不如 VLM;细微失败 on-manifold 抓不到(B2)。
- **判定**:可归因的相位级**粗**失败定位 = CRAVE 强;细微/语义成功判定用 GVL。

### 场景 F · 真机 RL 的实时进度监控 / reward shaping(HIL-SERL 类)
- **SOTA**:HIL-SERL 的人标 reward classifier。
- **CRAVE 优**:在线固定滞后-DP 零训练给 dense 进度 reward(替代人标),corr 0.94、可因果。
- **CRAVE 劣**:dense 进度 reward 是"接近示教流形"代理,纯 sparse 成败仍需另给。
- **判定**:省掉 reward classifier 的人标——直接接 HIL-SERL/SERL 当 dense shaping。

---

## 4. 可行方案安排(按 杠杆×成本×依赖 排序)

| 阶段 | 工作 | 周期 | 成本 | 产出/判据 |
|---|---|---|---|---|
| **Phase 0(决定性)** | B1 pilot | ~3-5 天 | 极低 | ✅已验:milestone=技能相位(R² 2×);转移非动作切分。整个重定位地基成立 |
| **Phase 1(快赢,可并行)** | A1(子任务→prompt)+ A2(数据工具) | ~1-2 周 | 低 | A1 喂现有 AWBC;A2 出 keyframe/粗失败定位 |
| **Phase 2** | C1(分布式离散头蒸馏)+ B1 全量 | ~2-4 周 | 中(单机 GPU) | action-aware 在线 advantage labeler;喂 [AB_plan](awbc_milestone_value_AB_plan.md) B 臂升级版 |
| **Phase 3(高天花板)** | C2(CRAVE 冷启 + RL 微调) | ~1+ 月 | 高(真机 rollout) | 验证"低成本超越示教";仅在 B1 验证 action-aware 后启动 |
| **持续** | D1(增量挖矿/漂移/域自适应) | 长期 | 低-中 | 自我维护,治挖矿脆弱 |

**关键路径**:Phase 0 的 B1 是整套重定位的最小决定性实验——已验证 milestone=动作相关技能相位,A1/C1/C2 全部有地基。但 **B2 否决了"廉价补 neg 洞"** → CRAVE 落地重心是**进度/结构/相位条件化 + 粗成功检测**,不要指望它替代 AWBC/RL 的细粒度失败判别。

**与现有工作的接口**:A1/C1 直接接 AWBC `prompt_from_task` 与 [AB_plan](awbc_milestone_value_AB_plan.md);C2 接 RECAP 式 pipeline;全部复用现成 milestone 模型 + kai0 数据,无新数据采集(Phase 3 除外)。

---

## 5. SOTA 一览与一句话定位

| 子领域 | SOTA 代表 | 它们给什么 | CRAVE 差异化优势 |
|---|---|---|---|
| 经验型 RL value/优势 | **RECAP / π\*0.6** | 真回报、action-aware、能超示教 | 零训练零标签、可解释结构、可当其**冷启 V**(C2) |
| 视频 progress/reward | **VIP · LIV · TOPReward** | 连续标量 progress | **离散 milestone 结构 + 技能切分**;零训练 |
| 零训练 VLM value | **GVL** | 跨任务零样本标量完成度 | 极廉可扩百万级、离散结构、重复=优势区、确定可复现 |
| 监督进度回归 | **kai0-AE** | scalar 进度(需 ~1 周标注) | 零标注、in-dist 打平且更平滑、OOD 更稳 |
| 技能/动作切分 | 一般需训练/逐任务调 | 切分 | 跨 episode 重复一次挖出、零训练、与 progress value 耦合 |

**一句话定位**:在"价值/奖励信号获取成本"这个全行业瓶颈上,CRAVE 占住「大批同任务 demo + 要极廉 + 要离散技能结构 + 要确定可复现」这一格——GVL 占「少样本/零样本/跨任务/语义」,RECAP 占「能超越示教但最贵」。CRAVE 不与它们正面竞争标量精度(in-dist 已 0.865 逼近监督),而做**别人贵到做不起的大规模零标签结构化前端**,并可作 RL/世界模型的廉价冷启 reward。其最该补的洞是**结果信号**(已验证 CRAVE 自身廉价手段补不上,需 C2 引入真机成败)。
