# Next-milestone 提示 → VLA:方案调研 + 决定性验证 plan

> 目标:**狠毒地**论证"给 VLA 加 next-milestone 提示(子目标)到底有没有正增量"。先定 kill criteria,再花算力。
> 前置:LMWM 已 VLA-ready(`VLALMWMPredictor`:next-milestone 分布 + top-k + `subgoal_latent` + confidence/entropy,[vla_integration](vla_integration_20260702.md));本 plan = LMWM Phase E。日期:2026-07-03。

## 0. 命题与零假设(先立判据,后动手)

- **H1**:next-milestone 子目标提示能提升**长程/多阶段**任务(叠衣、LIBERO-Long)成功率 & 子阶段通过率,尤其 OOD。
- **H0(必须认真对待)**:对已有 **language-conditioned pi0** + 我们已有的 **dense-value(advantage-weighted BC)**,next-milestone 提示**无正增量** → 定位为"补充"或直接否决。
- **决定性 kill criteria**:**GT 子目标**(A1/A2/A3)在 LIBERO-Long 上 ≤ A0 baseline(噪声内)→ **停,不投预测器**。子目标条件本身没用,预测得再准也没用。

## 1. 文献态势(2025–2026,**更新**旧结论)

我们 2026-06 的 deep-research 曾判"subgoal-image > language 被否决(0-3)"。**新证据把这个结论收窄了**:

- **视觉子目标在长程/OOD 上有效**:CoT-VLA(子目标图作视觉 CoT,条件动作)长程 **+10.4%**;GT goal-image 在 OOD 上 **+40% 绝对成功率**;趋势是 **视觉子目标 ⊕ 语言 并用**(VISTA 把离散目标图序列作"视觉里程碑"与文本子任务交织),不是二选一。
- **主失败模式(GHIL-Glue)**:**生成**的子目标"逼真但物理不可行 / 带伪影"→ 拖垮低层策略;需**子目标过滤器**(分类器选"最能推进指令"的子目标)+ 增强去同步 → **+25%,CALVIN SOTA**。
- **✅ CRAVE 的结构性优势**:我们的子目标是**真实 demo 帧(检索 medoid)或真实 milestone latent**,**不是生成的** → **天生规避 GHIL-Glue 的核心失败**(无幻觉伪影)。外加自带**离散 milestone 结构 + 转移图先验 + confidence/entropy 门控**。这是相对 SuSIE/BAGEL/扩散子目标的真卖点。
- **⚠️ CRAVE 的软肋**:next-milestone 预测 **top1 仅 0.42**(~13 分支歧义)→ **预测误差是主风险** → 必须靠 **top-k(top5 0.86)+ 置信门控 + 先用 GT 隔离**。

**结论**:命题在**长程叠衣**这个 regime 上有文献支撑(正是子目标增益最大处),且我们有"真实帧子目标"的独特优势。风险集中在"是否胜过已有 dense-value"+"预测误差"。

## 2. 三种注入方式(方案,来自 §7.2 精炼)

| 方案 | 做法 | 优 | 险 |
|---|---|---|---|
| **D1 · goal-image** | 检索 milestone+1 的 **medoid 真实帧** → 目标图喂 pi0(CoT-VLA/SuSIE 式,走图像通道) | 真实帧无幻觉;pi0 图像通道直接 | 需目标图对齐当前视角 |
| **D2 · latent subgoal** | milestone+1 **prototype latent**(`subgoal_latent`)作隐变量子目标注入(LaWAM 式) | 省(无需解码) | **DINOv3-H ≠ pi0 视觉编码器 → 需对齐投影 或 用 pi0 同款编码器重挖 milestone** |
| **D3 · 离散 milestone 文本** | milestone+1 的 **VLM 语义标签**作文本子指令(VISTA 式 interleave) | 最省;pi0 语言通道天然兼容 | 需先给 milestone 命名(§STATUS A1-VLM) |

## 3. 决定性验证设计(便宜优先 + 每步 kill criteria)

**环境:LIBERO-Long**(pi0 原生闭环,本仓 `fastwam/experiments/libero/` 已接好;长程多阶段 = milestone 天然有意义)。**判据:成功率 + 子阶段通过率。**

**阶段 A — "GT 子目标有没有用"(唯一决定性,先做)**
- **A0** baseline:原版 pi0(语言条件,无子目标)。
- **A1** = D1(GT):pi0 + **GT** milestone+1 检索真实帧目标图。← 上界,排除预测误差。
- **A2** = D2(GT):pi0 + **GT** milestone+1 latent(**用 pi0 同款编码器抽 milestone,保证同空间**)。
- **A3** = D3(GT):pi0 + **GT** milestone+1 文本子指令。
- **判据**:A1/A2/A3 任一 **显著 > A0** → 有头部空间,进阶段 B;**全 ≈ A0 → 对 pi0+语言无正增量 → 定位补充/否决(诚实收口)**。

**阶段 B — 换真预测器(仅 A* 有正增量才做)**
- 用 `VLALMWMPredictor`(top-k + 置信 + 转移图先验)**换掉 GT**;因子目标是真实帧,GHIL-Glue 式过滤主要针对**预测错误**(而非幻觉)→ 用 confidence/entropy 门控 + 分类器过滤 top-k。
- **判据**:量化预测误差(0.42)吃掉多少 A* 增量;剩余增量 > 0 才继续。

**阶段 C — 域迁移(真机)**
- 回真机 kai0 **叠衣**(长程,最能显子目标价值)验域迁移。sim 验方法、真机验域。

## 4. 关键设计决策(狠毒)

1. **先 GT 后预测**:必须先用 GT milestone+1 隔离"子目标条件本身有没有用"。否则 top1 0.42 的预测误差会污染结论、误杀命题。
2. **真实帧 ≠ 生成**:用检索真实帧/真实 latent,**不生成** → 规避 GHIL-Glue 核心失败模式(这是卖点,要在对照里体现:D1-真实帧 vs 生成子目标 baseline)。
3. **top-k + 门控,不用 top1 硬目标**:~13 分支,top1 常错、top5 覆盖 86%;高熵/低置信时把决策权交回策略。
4. **只在长程任务上测**:子目标增益最大在长程/OOD;**别在短单阶段任务上测**(那里大概率无增量,会误杀命题)。
5. **必须对照 dense-value**:终极问题是"next-milestone 提示 **vs / ⊕** 我们已有的 dense-value(advantage-weighted BC)"——是替代、叠加、还是无用。加一臂 A0+dense-value 作强 baseline。

## 5. 里程碑 plan(大致)

| P | 内容 | 算力 | 判据 |
|---|---|---|---|
| **P0**(1–2 周)| CRAVE 套 LIBERO demo → 每帧 milestone + **GT** milestone+1(检索帧 / latent / VLM 标签)导出;pi0 子目标注入接口(图像/latent/文本三通道) | 本地+集群特征 | 导出与注入通 |
| **P1**(决定性)| 训 A0 / A1 / A2 / A3 / A0+dense-value,LIBERO-Long 闭环比成功率+子阶段通过率 | 集群(长训不走本地 2 卡)| **A*>A0?** 是→P2,否→收口 |
| **P2**(仅正增量)| 接 `VLALMWMPredictor` + top-k 过滤/门控,换掉 GT | 集群 | 预测误差后仍 >0? |
| **P3**(真机)| kai0 叠衣域迁移 | 真机 | 域迁移是否保持 |

## 6. 诚实边界 / 主风险

- **最大风险**:对 pi0+语言(+dense-value)**无正增量**——短任务上大概率,长程上文献支持有 → 故**先长程、先 GT**。
- **预测误差**:top1 0.42 → top-k + 门控 + GT-first 隔离。
- **D2 编码器空间不匹配**:DINOv3-H ≠ pi0 编码器 → 需对齐投影或用 pi0 编码器重挖 milestone(D1/D3 无此问题,故 **D1/D3 先行**)。
- **sim→真机 gap**:LIBERO 验方法有效性,真机 kai0 验域;两者解耦。

## 参考(2025–2026)

- CoT-VLA(子目标图作视觉 CoT):https://arxiv.org/html/2503.22020
- VISTA / 分层世界模型(离散目标图=视觉里程碑 ⊕ 文本子任务):https://arxiv.org/html/2602.10983v2
- Diffusion Trajectory-guided Policy(长程轨迹引导):https://arxiv.org/html/2502.10040
- GHIL-Glue(子目标过滤,治"逼真但不可行"的生成子目标):https://arxiv.org/abs/2410.20018
- 本仓前置:[vla_integration](vla_integration_20260702.md)、[lmwm_stage_overview](lmwm_stage_overview.md)、CRAVE [§7.2/7.3 milestone→VLA](../../crave/docs/milestone_centroid_decoding.md)
