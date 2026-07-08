# LMWM-2:milestone 预测为 VLA 赋能 —— 全学习式重设计方案

> 日期 2026-07-08(v2:应用户定案,**弃检索、全学习式**)。定位:从基础思想 + milestone 预测任务的本质特性 + 三路深度文献调研(subgoal/milestone 预测、WM-for-VLA、value/进度表征,2022→2026)出发的重设计。
> **设计红线(用户定案)**:最终架构不用检索——检索把模型锁死在历史记忆里,跨数据/跨任务能力必须由**学习**获得。检索仅两处降级使用:① 训练期离线挖矿(CRAVE,本来就是);② 实验诊断上界(不随模型交付)。
> 与现 LMWM 的对比与定案见 §5,实验计划见 §6。

---

## 1. 回归基础思想:我们在赌什么,2026 文献怎么裁决

**赌注(2026-07-01 原始规划)**:给 VLA 的提示应是**价值层的 milestone 计划**,而非通用未来帧 —— milestone 是低熵瓶颈态,比固定时距未来帧对行动更有指导性。

### 1a. 文献对赌注本身的裁决:✅ 方向被独立验证,且生态位仍空白

| 证据 | 结论 |
|---|---|
| **VISTA** (2602.10983) | 世界模型只出 **keyframe 级 milestone 序列**(目标图⊕文本子任务),OOD SR 14%→69%;理由与我们同源:"瓶颈态低熵、状态不变性抑幻觉" |
| **DreamVLA** (2507.04447) | 预测**压缩的任务相关抽象**(动态区/深度/语义)> 预测整帧 |
| **LBP** (2505.06861) | 在**冻结 SigLIP 空间**做隐子目标;粗任务对齐子目标 > 密集短时预测;**反向**(从终局倒推)预测误差不随 horizon 累积 |
| **LaWAM** (2606.15768) | 冻结 VFM 特征上的 latent subgoal 条件 VLA = LIBERO 98.6%、24× 快于像素 WM —— 我们的大方向已是 SOTA 配方 |
| **生态位空白** | 所有 latent-WM 竞品(LaWAM τ=1.2s、FLARE、V-JEPA-2-AC、VPP)全是**固定时距**;"以语义 milestone 定义 horizon 的 WM"**无任何已发表系统** —— 这是我们真正的差异化 |

### 1b. 文献对我们**实现方式**的裁决:⚠️ 三张黄牌

1. **无 grounding 的 latent 预测会塌缩**(OpenHelix 2505.03912:辅助 grounding 损失是最大单杠杆 3.45→4.01;HIQL 2307.11949、LCB 2405.04798 同向):隐子目标必须被辅助任务锚住,否则退化成静态指令码。
2. **层级系统的瓶颈常在"接口"而非"生成质量"**(GHIL-Glue 2410.20018):采 K 候选 + 进度分类器**过滤**(负例含时间反序对)+ 当前帧/目标帧**去同步增广**,+25~53% SR。我们把全部力气花在预测侧,接口/验证侧为零。
3. **唯一裁决是下游 SR,我们从未测**。所有 intrinsic 定案(center_w=0.1、teacher=proto、MDN K=4)都可能被 SR 重排。

### 1c. 考古发现的两处内伤(重设计的直接动机)

- **milestone 特有结构被丢掉**:价值单调只当训练标签、milestone 推进无门控、多模态不确定性无出口 —— 现架构退化成"另一个固定形状的 latent WM",与 LaWM 的差异只剩目标标签来源。
- **搁置的裁决**:`archive/next_milestone_vla_validation_plan.md`(2026-07-03)早已写好 GT-first + kill criteria 的决定性验证 plan,**与今天三路文献调研的结论高度吻合,但被搁置从未执行**。

**重设计 = 把 milestone 的价值/事件/不确定性结构以"学习的方式"放回架构 + 把搁置的 SR 裁决放回第一优先级。**

---

## 2. milestone 预测任务的本质特性 → 设计原则(全学习式)

架构必须由任务特性决定,而不是沿用"世界模型=预测未来特征图"的默认形状。

| # | 任务特性 | → 设计原则 | 依据(文献 + 自家实验) |
|---|---|---|---|
| **T1** | **监督空间有界**:milestone 是挖出来的有限集(M≈15–50/任务),但**部署时要外推到没见过的状态与任务** | 有界性放在**训练监督**(簇中心码 teacher)+ **测试时验证器**(学习型 verifier),而不是查表/检索——让网络学"什么样的下一步合法",不是背"哪些下一步出现过" | proto teacher 自家已定案(开放词表);GHIL-Glue 过滤器是**学习出的**关系函数,天然跨任务;检索被否决:锁死历史记忆(用户定案) |
| **T2** | **未来多模态但模式数少**:单帧条件下真实分支 ~2 峰,top1 天花板 ~0.28–0.42(kNN 探针=信息论上限,非容量问题) | **校准的多模态头(MDN K=4 保留)+ 熵门**:不确定时不硬塞提示,把决策权交还策略的语言通道 | 自家容量消融全平(260M deploy +0.00);MDN top3 随 K 单调;GHIL-Glue sample-K-then-verify |
| **T3** | **价值单调是 milestone 的定义性质**(CRAVE 挖矿的构造) | value 必须**上到测试时**:学习型价值头对候选做 value-forward 门 + 进度停滞才重发 —— 而不只是训练标签 | GHIL-Glue(时间反序负例 ≡ 我们的单调性);TaKSIE 2410.11013 / Anticipation-VLA 2605.01772(进度门控重发);V-GPS 2410.13816(value 测试时重排) |
| **T4** | **消费者是 VLA,不是人眼** | payload 必须**近流形 + 空间结构化 + 与策略同塔**;评估只认 **action-MAE/SR**,grid-cos 降级 | LaWAM 消融(空间结构化子目标>pooled);2605.06388(语义空间>重建);OpenHelix grounding;grid-cos 已被自家判"不敏感" |
| **T5** | **身份与外观解耦**:milestone 身份跨 episode 共享,外观属于当前 episode | 身份=预测**连续码**(proto 谱系);外观=**生成器以当前画布渲染**——这恰是"学习"胜过"检索"的自家最硬证据 | forward-from-current oracle 0.935 > absolute 0.82(外观由当前观测带入,学习渲染而非搬历史帧);proto 与 inv 打平但开放词表 |
| **T6** | **horizon 是事件不是时距**:milestone 达成才该推进 | **事件驱动调度**:价值越过锚点→推进 m+2;进度停滞 τ 秒→重发/重选 | 生态位空白(§1a);LBP 反向一致性 |
| **T7** | **跨任务的共享结构在语言与价值,不在像素**:不同任务的 milestone 内容不同,但"指令→阶段推进"的规律共享 | **语言条件化**预测器与价值头(现架构是纯状态条件,语言是最大的未用杠杆);进度归一化到 [0,1] 跨任务共享价值刻度 | π0.5 2504.16054(语言子任务是开放世界泛化来源);VISTA(文本⊕视觉交织);自家 LOO:固定词表在 unseen 任务全弱→出路只能是语言+连续价值流形,不是更大词表 |

一句话:**milestone 预测是"带价值序、语言索引、低模式数的下一瓶颈态预测"。跨任务能力来自语言条件 + 连续码空间 + 归一化价值 + 关系型验证器这四个可学习的共享结构,而不是查历史库。**

---

## 3. LMWM-2 架构(全学习式,部署零索引、零查表)

### 3a. 数据流

```
在线(全部在 π0.5 SigLIP 同塔空间,冻结 ~400M;部署无 bank、无索引):
  帧 224² → SigLIP grid G_t (16×16×1152) + gist g_t (1152)
  语言 l → π0.5 原生语言嵌入(取 prefix 语言槽 pooled 向量,冻结)

  ① 预测器(~3.5M,MDN K=4,继承 predm 谱系):
       p(z_next | g_t ⊕ lang ⊕ proprio ⊕ prev_z)  → top-K 候选码 {ẑ_k}
       — 训练 teacher = proto(下一 milestone SigLIP 簇中心固定投影,已定案):连续码=身份=开放词表
       — 新增条件:lang(T7,跨任务主杠杆)、prev_z / proprio / t_bin(自家探针 +11/+5.2/+3.5pt 从未落地)
  ② 价值/进度头(~0.2M):v(g_t | lang) ∈ [0,1] 跨任务归一化单调进度(SARM 式 stage-classify-then-regress)
  ③ 生成器(30M AdaLN,保留):Ĝ_sub = AdaLN(G_t 画布, ẑ)
       — 学习渲染,外观由当前观测带入(T5;0.935>0.82)——这是相对检索的本质优势:对没见过的场景外观照样成立
  ④ grounding 辅助损失(训练时,OpenHelix 处方,防塌缩):
       从 ẑ / Ĝ_sub 回归:milestone 价值 v(m*)、Δprogress、(可选)milestone 时刻 proprio
  ⑤ 学习型 verifier(~0.5M,GHIL-Glue 处方,替代一切检索式过滤):
       f(G_t, Ĝ_sub, lang) → "该候选是否推进任务"
       负例 = 时间反序对 + 错任务对 + 扰动码生成的 off-manifold 样本
       — 关系型函数,多任务联训后对新任务泛化(它学的是"推进"这个关系,不是任务内容)
  ⑥ 注入 π0.5(复用 archive/lmwam_v2 接线):
       Ĝ_sub(可 4× 下采样 → 64 token)进 prefix 新视觉槽
       + type-embedding + 零初始化投影 + KI stop-grad + 去同步增广 + subgoal-dropout
       对照臂 = FLARE 式(2505.15659):无显式 token,只加"未来 token 对齐 milestone 嵌入"辅助损失

调度(事件驱动,T6):
  ② v 越过 m* 锚点 → 推进预测 m+2;进度停滞 > τ → 经 ⑤ verifier 重选候选重发
  ①③ 高熵/⑤ 全否 → 熵门:不注入,交还 π0.5 语言通道

离线(label 工厂,只在训练期存在,不随模型交付):
  CRAVE 挖矿(per-task)+ UVD(2310.08581)反向单调性交叉验收
  + bank 质量验收三关卡(簇内纯度 / value 跨度 / 身份可分度;治 vis id3=0.11 无验收上桌的教训)
```

### 3b. 部署形态

| | LMWM(现) | LMWM-2 |
|---|---|---|
| 在线参数 | 34M(predm 3.3 + 生成器 30.3) | **~34M**(① 3.5 + ② 0.2 + ③ 30 + ⑤ 0.5)——形态不变,结构补齐 |
| 部署依赖 | 无索引 ✓ | 无索引 ✓(bank 只活在训练期) |
| 语言条件 | ✗(纯状态) | ✓(①②⑤ 全条件化)|
| 测试时结构 | 单次前向 | top-K → verifier/value 门 → 事件调度 |
| 注入 π0.5 | 未接 | 设计核心(双臂)|

### 3c. 跨任务能力从哪里来(学习式回答)

1. **语言条件**(T7):任务身份由指令携带,预测器学"指令+状态→下一阶段"的映射,新任务=新指令,不需要新词表;
2. **连续码空间**(proto 谱系):身份活在 SigLIP 连续流形上,自家 LOO 已实测连续锚在 unseen 身份小胜离散 CE;
3. **归一化价值**(②):[0,1] 进度是跨任务共享刻度,"快到下一个瓶颈了"这个信号与任务内容无关;
4. **关系型 verifier**(⑤):学"什么叫推进",不是学"推进到哪张历史帧";
5. **生成器画布机制**(③):外观从当前观测来,对 unseen 场景零历史依赖。

### 3d. 明确不采纳的文献建议(自家证据否决,防重复踩坑)

- **"预测器换 ViT"**:自家 H7 已证空间 token 对身份预测无增益(grid vs pooled A≈B)、gist vs grid 打平(Δ0.0003)。ViT 只在"网络必须输出 grid"时有意义——③ 已由 AdaLN CNN 胜任且有消融定档(concat 0.52 / nolift 0.57 vs 定档 0.99)。
- **"上 diffusion/flow 子目标头"**:自家实测 flow best-of-16 0.834 < 回归 0.872,MHP/MCL deploy 反降。多模态上限来自单帧条件的信息论天花板;top-K+verifier 是正确出口。
- **"换 V-JEPA/DINO 目标空间"**(2605.06388):同塔 SigLIP 的 FLARE 逻辑 + 自家 E2 实测(0.716>0.694)优先;留 LaDi-WM 式双空间辅助损失作远期消融,不换塔。
- **"检索真实帧做 payload"**(GHIL-Glue/UVD/B2FF 一系):用户定案否决——锁死历史记忆,跨任务要靠学习。其 on-manifold 优点由 ③画布渲染 + ④grounding + ⑤verifier 学习式补偿;其"过滤"精髓被 ⑤ 以学习形式继承。

---

## 4. 三路调研浓缩(20 篇核心证据索引)

| 主题 | 论文(arXiv) | 对 LMWM-2 的贡献 |
|---|---|---|
| 接口是瓶颈/学习型过滤 | GHIL-Glue 2410.20018 | ⑤ verifier(时间反序负例)、去同步增广 |
| milestone 序列 WM | VISTA 2602.10983 | keyframe 级预测抑幻觉、文本⊕视觉交织 |
| 冻结空间隐子目标 | LBP 2505.06861 | SigLIP 空间验证、粗>密、反向一致 |
| 零成本对照臂 | FLARE 2505.15659 | 辅助对齐损失,无推理开销,+26% |
| 隐子目标要 grounding | OpenHelix 2505.03912 / HIQL 2307.11949 / LCB 2405.04798 | ④ 辅助损失防塌缩(最大单杠杆) |
| 进度门控重发 | TaKSIE 2410.11013 / Anticipation-VLA 2605.01772 | 事件驱动调度 |
| value 测试时用 | V-GPS 2410.13816 / GVL 2411.04549 | ②③门控;GVL=CRAVE 的零训 baseline |
| 语言=跨任务索引 | π0.5 2504.16054 / Hi Robot 2502.19417 | ①②⑤ 语言条件化(T7) |
| 任务相关抽象>整帧 | DreamVLA 2507.04447 | 赌注验证;分块注意力保护 action token |
| 注入方式 | DUST 2510.27607 / LaWAM 2606.15768 | 空间化>pooled;KI/零初始化;latent-action 蒸馏可选 |
| 挖矿对标 | UVD 2310.08581 / InfoCon 2404.10606 / SARM 2509.25358 | 边界交叉验收 / 信息量筛选 / ② 阶段-回归结构 |
| 视觉 CoT / 子目标图 | CoT-VLA 2503.22020 / SuSIE 2310.10639 / Seer 2412.15109 | 长程 +10~17%;GT goal-image OOD +40% |

---

## 5. LMWM-2 vs LMWM 现架构:逐维对比与定案

### 5a. 逐维对比

| 维度 | LMWM(现) | LMWM-2 | 谁优 / 证据 |
|---|---|---|---|
| 子目标身份 | MDN K=4 连续码(proto),**纯状态条件** | 同谱系 + **语言/prev_z/proprio 条件** | **LMWM-2**:语言是 T7 主杠杆;探针 +11/+5.2/+3.5pt 从未落地 |
| 子目标外观 payload | 生成(AdaLN 画布渲染) | **保留同款** | 平;这是现架构最有价值的资产(0.935>0.82),且是学习式跨场景的根 |
| off-manifold 风险 | 有,靠"仅可视化"豁免 | ④grounding + ⑤verifier 学习式压制 + 熵门兜底 | **LMWM-2**(诚实:仍非零——这是坚持学习式路线要用 ⑤ 顶住的代价) |
| value 的用法 | 仅训练标签 | 训练 + **测试时门控/重发**(学习型价值头) | **LMWM-2**;兑现"价值层 WM"承诺 |
| horizon | milestone(名义),推进无门控 | 事件驱动 + 进度门 | **LMWM-2**(论文生态位主张) |
| 多模态出口 | MDN 有分布但部署只取均值 | top-K → verifier 选择 → 熵门放弃权 | **LMWM-2**:不确定性有了行为学出口 |
| VLA 注入 | 未接(半成品) | 设计核心,双臂 + kill criteria | **LMWM-2** |
| 跨任务机制 | 词表并集 / proto 码 | 语言条件 + 连续码 + 归一化价值 + 关系 verifier(四重学习式共享) | **LMWM-2**;LOO 已证固定词表在 unseen 全弱 |
| 部署形态 | 34M 单前向 | ~34M + 测试时结构(K 次生成器前向,K≤4) | 现架构略简;开销可控(30M×4 仍 ≪ 一次 π0.5 前向) |
| 已验证程度 | intrinsic 充分(reach 1.67>LaWM 1.48) | ①③继承已验证;②④⑤⑥新增未跑 | **LMWM**;故 P0/P1 顺序设计成先裁决后建设 |

### 5b. 继承关系(LMWM-2 = 现架构的结构补齐,不是推倒)

- ①预测器 = predm + proto teacher **原样继承**,加条件化;
- ③生成器 = **原样继承**(30M AdaLN,消融已定档);
- ②价值头 = CRAVE 价值蒸馏(AE 塌缩修法教训:锚点+单调投影,SARM 同款);
- ⑤verifier = GHIL-Glue 过滤器的学习式移植(新增,~0.5M);
- ⑥注入 = `archive/lmwam_v2_plan_20260704.md` 接线方案直接复用(KI/零初始化/token 预算已想清楚);
- Phase 0 = `archive/next_milestone_vla_validation_plan.md` 的 GT-first kill criteria 复活升级。

### 5c. 定案

**采纳 LMWM-2(全学习式)为最终方案。现架构的短板不是"预测器/生成器本身",而是五个缺失的学习结构:语言条件、测试时价值、事件调度、不确定性出口、VLA 注入。裁决权交给 P0 的 oracle-SR:**
- 若 A1(oracle 码→生成 payload)>A0:主链路成立,进 P2 换真预测器;
- 若 A1≈A0 但 A_diag(oracle 检索真实帧,**仅诊断**)>A0:说明生成保真是短板 → 火力集中 ③④⑤(生成质量与验证),而不是放弃命题;
- 若全部 ≈A0:子目标条件对 π0.5+语言无正增量 → 诚实收口,转 FLARE-aux 路线(不要求"注入有用",只要求"预测未来是有用的训练信号");
- 无论哪支,**milestone-horizon vs 固定 1.2s** 消融保留——这是对基础思想的直接检验,也是相对 LaWAM 的论文主张。

---

## 6. 实验计划(P0 决定性优先,便宜优先,每步带 kill criteria)

### P0 —— oracle 裁决(最高信息量,不训任何新 WM)
| 臂 | 内容 | 角色 |
|---|---|---|
| A0 | π0.5 baseline(语言条件,无子目标) | 基线 |
| A1 | + **oracle 码 → 生成器 payload**(GT next-milestone 码,现 ③ 渲染) | **主臂**:学习式 payload 上界 |
| A2 | + oracle milestone **文本标签** | 最便宜通道 |
| A_diag | + oracle **检索真实帧** grid | **仅诊断**(度量生成保真损失多少增量;不进架构、不交付) |

- 训法:LoRA(gemma_300m_lora r32)+ KI 全冻结 provider + 零初始化投影 + 去同步增广 + subgoal-dropout;单 top-head 视图先行。
- 评估阶梯:**kai0 离线 action-MAE**(天级先出信号;记 v3-PTS 教训:离线 MAE 只作排序不作绝对判断)→ **LIBERO-Long 闭环 SR**(sim 决定性;`fastwam/experiments/libero/` 已有 π0 闭环,需 CRAVE 套 LIBERO demo)→ 真机叠衣(域)。
- **kill criteria**:全臂 ≈A0 → 停注入路线转 FLARE-aux;A1>A0 → P2;A1≈A0<A_diag → 火力集中生成质量(③④⑤)。
- 算力:LoRA 微调走集群(H20 队列/gf3 恢复后);本地 2 卡做接线 smoke + 离线 eval。

### P1 —— LMWM-2 组件落地(与 P0 并行,本地可做)
① 条件化改造(lang/prev_z/proprio 进 predm,train_multitask.py 骨架)、② 价值头、④ grounding 辅助损失、⑤ verifier(负例生成器)、bank 验收三关卡 + UVD 交叉验收脚本。产出:`LMWM2Provider`(纯前向、全冻结、零索引)。

### P2 —— 换真预测器(仅 A*>A0 才做)
oracle→预测,量化预测误差吃掉多少增量;消融:top-K+verifier vs top1 硬目标、熵门开/关、**语言条件开/关**(T7 主杠杆的直接检验)。判据:剩余增量>0。

### P3 —— 论文主张与 scaling
milestone-horizon vs 固定 1.2s 消融(生态位);GVL 零训 value baseline;UVD 对标;跨任务(3 任务联合)+ LOO 开放词表复测——这次以 action-MAE 为准,并验证语言条件是否兑现 unseen 任务泛化。

### 指标纪律
主指标 = action-MAE → 闭环 SR;grid-cos 降级 sanity;新增:milestone top-K acc、value-forward 率、verifier AUC(分负例类型)、熵门触发率、bank 验收三关卡。

---

## 7. 风险与止损

| 风险 | 概率 | 止损 |
|---|---|---|
| 子目标条件对 π0.5+语言无正增量(H0) | 中(短任务大概率,长程文献支持有) | 只在长程测;kill criteria 明确;FLARE-aux 是有尊严的退路 |
| 生成 payload off-manifold 伤 VLA(弃检索的代价) | 中 | ④grounding + ⑤verifier + 熵门三层压制;A_diag 量化代价;若 A_diag≫A1 火力集中生成质量 |
| 语言条件在单指令数据集上学不出区分度 | 中(kai0/coffee 每任务指令单一) | 跨任务联训天然提供语言方差;P3 LOO 是真检验;不行则退 prev_z/proprio 条件仍有探针背书 |
| verifier 学成"任务分类器"而非"推进判别器" | 中 | 负例设计以**同任务时间反序**为主力(逼它看推进不看内容);verifier AUC 分负例类型报告 |
| bank 质量不过关(vis 教训) | 中 | 验收三关卡前置,不过关任务不进联训 |
| LIBERO 需重挖 bank,周期长 | 高 | 离线 action-MAE 先行;LIBERO 并行准备 |
| 集群/gf3 不可用 | 现状 | P1 本地先行;P0 微调排 H20 队列 |

---

## 8. 参考

三路调研原文(subgoal 预测 / WM-for-VLA / value-进度表征)由 2026-07-08 深度调研产出,核心 20 篇见 §4 表。自家证据全部来自 `PITFALLS_AND_HISTORY.md` 与 `ABLATION_CONVERGENCE_2026-07.md` 的实测编号。
前置文档:`ARCHITECTURE_AND_BASELINE.md`(现架构)、`archive/lmwam_v2_plan_20260704.md`(注入接线)、`archive/next_milestone_vla_validation_plan.md`(GT-first 验证原案)。
版本注:v1(RAMP,检索锚定)因用户定案"最终架构不用检索、跨任务能力必须学习获得"而废弃;其 verifier/门控/事件调度精髓以学习式保留于本方案。
