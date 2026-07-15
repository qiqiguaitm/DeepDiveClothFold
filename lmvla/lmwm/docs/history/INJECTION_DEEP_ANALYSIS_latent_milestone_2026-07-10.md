# latent milestone 注入 VLA — 深度分析(2026-07-10 外部论文对照)

> 承接 [`INJECTION_DESIGN_2026-07.md`](INJECTION_DESIGN_2026-07.md)(主推 P=SigLIP 虚拟图像 token 进 prefix + KI)与 [`LMWM2_FINAL_ARCHITECTURE.md`](LMWM2_FINAL_ARCHITECTURE.md)(定档注入=`Ĝ_sub→prefix 新视觉槽 + 零初始化投影 + KI stop-grad + 去同步增广 + 弃权`)。
> 本轮读 6 篇"视觉/几何/物体信息注入 VLA"的论文(用户点名),尤其 **3D-Mix 的九向注入消融**,回答一个此前只靠推演、现在有实测数据的问题:
> **我们把"预测未来 milestone latent grid"放进 π0.5 prefix,到底安不安全?还是应该改注入位置?**
>
> **一句话结论(不急于定论,先排除+给可测 shortlist):** 我们注入的不是 3D/mask/trace,而是 **SigLIP 同空间的预测未来图像 latent grid**——**图像信息是 VLM 的看家本领,3D 几何才是 VLM 的软肋**。3D-Mix 九向消融的 Early(44.5)/Visual(4.69)崩盘是**模态混淆**的:崩的很可能是"VLM 自注意力处理不了 3D 几何 token",**不能直接推断"图像 latent 进自注意力也会崩"**。反证:TraceVLA 把一整张额外图像塞进视觉流(同位置②)却是**胜出**方案。→ **因此 prefix(③)不该被判死刑,拉回可测候选**;3D-Mix 真正**模态无关、可迁移**的结论只有三条(AE-expert-内部 3.13 该排除、门控软融合稳、OOD 才见真章)。本文改为:**先排除一定不可行的,再给 3 个值得测的方案(T0/T1/T2)头对头,不预设赢家。**
>
> **[2026-07-10 修订]** 上一版曾据 3D-Mix 直接把主推从 ③ 改到 ④——**过度迁移了 3D 模态的结论**。本版按"我们是图像模态"重新平衡:③ 与 ④ 均为 live 候选,由实验裁决。

---

## 1. 六篇论文的注入机制 + 对我们的可迁移性

| 论文 | arXiv | 注入信息 | 注入机制(进 VLA 的位置) | 对我们可迁移性 |
|---|---|---|---|---|
| **3D-Mix for VLA** | 2603.24393 ⚠️ | VGGT(冻结)几何 patch token(=**冻结基础模型 grid**) | 九向对比;**胜=MLLM↔expert 之间门控软融合,不改预训练权重** | ★★★★★ 最相关:同构(冻结 grid)+ 有注入位置的直接实测排序 |
| **TraceVLA** | 2412.10345 | 2D 视觉轨迹(可画) | 轨迹**像素 overlay 到 RGB** + 双图共享编码器 + 全量微调 | ★★ 反例:overlay>token(+6.4 vs +2.4)只在"可画+编码器会读像素 prompt"时成立;latent 不可画 |
| **Oat-VLA**(≈"ObjectCentric-VLA") | 2509.23655 | 物体 slot token(FT-DINOSAUR) | **替换**原 patch token(256→16)+ MLP projector | ★★★ 教训:slot 特征与原 SigLIP patch 统计不合→必重训;→ 我们同源 SigLIP 是优势,但别"替换" |
| **SAM2Act(+)** | 2501.18564 | SAM2 稠密特征 + memory | 多分辨率 feature-map 拼接 + cascaded upsample;memory=SAM2 式 FIFO + memory-attention | ★★ 正参照(稠密特征注入操作策略),但落在离散 heatmap/2.5D 策略、注入的是**当前帧**非未来 |
| **IA-VLA** | 2509.24768 | 大 VLM 选出的目标物体高亮(半透明 mask) | **纯输入像素增强** + OpenVLA LoRA;不改结构 | ★ 正交(改像素非注 latent);唯一有用启示=评测要设**需要外推的困难/OOD 子集** |
| **Manipulate-Anything** | 2406.18915 | — | VLM 只当**高层规划器 + 数据生成器**,策略网络零注入 | ✗ 反面对照(是我们要避开的"退化成纯数据路线") |

> ⚠️ **待核验**:3D-Mix `2603.24393` 由调研 agent 从 arXiv 确证但 HF papers 未收录、我未独立复核;其九向表的具体数值以 agent 转述为准,落地前应亲自取原文核对表 2。其余 5 篇 id 可信度高(TraceVLA=ICLR'25 known-real)。

---

## 2. 统一"注入位置"分类学(把 6 篇 + 我们的 P/A/F 全摆到一条谱上)

按"离预训练自注意力流的距离"从**危险**到**安全**排序(π0.5 = PaliGemma VLM prefix-KV + Gemma action expert 逐层 cross-attn 读 prefix):

```
[最危险 ← 扰动预训练自注意力 → 最安全]

① 改视觉 token 流 / 替换 patch      Oat-VLA(替换256→16)   · 3D-Mix Visual Fusion 4.69💥
② 输入 embedding 直接 concat        3D-Mix Early Fusion 44.5📉 · TraceVLA(双图,但同空间可画+全量FT救回)
③ ★进 VLM prefix(参与自注意力)     我们现行定档 P / LMWM2_FINAL   ← 本文要动的就是这一格
④ MLLM↔action-expert 之间(条件KV)  3D-Mix Gated Fusion 68.2🏆 / Concat 60.4  ← 建议新主推
⑤ action expert 内部新增 attn 头     3D-Mix AE-Fusion 3.13💥 / (我们旧备选A"expert内cross-attn"的危险变体)
⑥ 训练时对齐、推理丢弃(辅助)        3D-Mix Spatial Forcing 58.9 / 3D-Tokens 56.3 · FLARE(我们备选F)
⑦ 纯输入像素增强 / VLM 当规划器      IA-VLA · Manipulate-Anything(不注策略)
```

**读这张谱的两个反直觉点:**
- **③ 和 ④ 只差"要不要进 VLM 自注意力",但风险差一个数量级**。我们现行定档在 ③;3D-Mix 冠军在 ④。
- **⑤ 不是"备选 A"**:我们旧备选 A 写的是"action expert 每(隔)块加 cross-attn 子层, K/V=milestone"——若实现成**expert 内部新增注意力机构**,正是 3D-Mix 崩到 3.13 的 AE-Fusion。**真正安全的备选是 ④**(复用 expert 既有的 conditioning cross-attn、只多喂一段门控 KV),不是 ⑤。这个区分此前被我们混为一谈。

---

## 3. 3D-Mix 九向消融 = 目前最强的"注入位置"经验先验

基线 Qwen3-VL-4B + GR00T DiT expert;SIMPLER(OOD real→sim)/ LIBERO(in-domain)。

| 方案 | 机制 | SIMPLER | LIBERO | 落谱 |
|---|---|---|---|---|
| **Gated Fusion** 🏆 | MLLM↔expert 间;position-specific 门控用 MLLM mean-pool 全局语义自适应加权 2D 语义 vs 外来 grid;**不改 MLLM/expert** | **68.23** | **98.05** | ④ |
| Concat Fusion | 同位置,GateMixer 预处理后直接 concat | 60.42 | 97.75 | ④ |
| Spatial Forcing | 训练中间层 cosine 对齐,推理丢弃上游 | 58.85 | 97.72 | ⑥ |
| 3D-Tokens | 可学习特殊 token + cosine 对齐,推理丢弃 | 56.25 | 97.64 | ⑥ |
| Middle-Layer Inject | MLLM 第 k 层 adapter pre-norm cross-attn + 可学 α | 51.82 | 97.82 | ③′ |
| Early Fusion | MLLM 输入序列 concat 外来 token | 44.53 | 86.45 | ② |
| Visual Fusion | 视觉 token 级 cross-attn(2D 作 Q,外来作 K/V) | **4.69** | 73.40 | ① |
| AE-Fusion | action expert 内双 cross-attn 头晚融合 | **3.13** | 97.40 | ⑤ |

**两条可直接搬的铁律:**
1. **位置压倒一切**:同一批 VGGT 特征,只因注入位置不同,OOD 从 3.13 到 68.23。→ 我们该花在"选位置/门控"上的力气,远大于花在"milestone 质量再抠几个点"上。
2. **门控软融合 + 冻结上游 + 不碰预训练权重 = 稳态**;**硬塞进视觉/输入/expert-内部注意力 = 灾难**。注意 LIBERO(in-domain)对差异远不敏感(83–98 都有),**只有 OOD 才把注入方式的好坏拉开**——见 §5 评测。

---

## 4. 核心张力:"SigLIP 同空间"能否豁免我们于 3D-Mix 的警告?

这是全文关键。我们此前(INJECTION_DESIGN §0)的核心论证是:*"通用文献警告 prefix=最高污染,那是针对任意新空间 token;SigLIP 同分布 token 经原图像投影进 backbone 近零 distribution-shift → prefix 从最危险变最原生。"* 3D-Mix 让我们能把这句话拆成两个可分辨的风险分量:

**风险 A · 特征分布位移**(外来空间 ≠ backbone 空间)。
- 3D-Mix 的 Early/Visual Fusion 崩,主因就是 VGGT 空间 ≠ MLLM 空间。
- **我们确实豁免这一半**:milestone ∈ SigLIP-So400m = PaliGemma 视觉空间(LMWM2_FINAL 赌注 6/9 已用数据锁死走 SigLIP-native)。Oat-VLA 反向印证——它的 slot 特征与原 patch 统计不合就必须重训;我们同源就没这问题。**这一步我们的原论证成立。**

**风险 B · 模态 vs VLM 先验**(注入的信息类型 backbone 会不会处理)。**← 这是上一版漏掉、用户点出的关键轴。**
- 3D-Mix 注入的是 **3D 几何**(VGGT)。**VLM 天生不擅长空间/几何**——预训练全是 2D 图像-文本,自注意力里从没学过怎么 attend 一堆 3D 几何 token。所以 Early(44.5)/Visual(4.69)崩,**很大一部分是"把 VLM 处理不了的模态硬塞进它的注意力"**,不是"任何东西进自注意力都崩"。
- **决定性反证 = TraceVLA**:它把一整张**额外图像**(overlay)塞进视觉流(同位置②的 concat),却是**胜出**方案(+6.4)——因为**图像是 VLM 的母语**,视觉编码器本就会读。同位置②:图像模态赢、3D 模态输 → **模态和位置一样是自变量**,3D-Mix 的位置结论**不能脱离它的 3D 模态外推到我们**。
- **我们的注入 = 图像 latent**(16²×1152 SigLIP grid = 一张图像特征图),模态上站 TraceVLA 这边,不是 3D-Mix 那边。→ **风险 B 我们大概率也豁免大半**——prefix/自注意力注入对我们没有 3D-Mix 那种模态灾难。

**风险 C · 时态 novelty**(预测未来 ≠ 当前观测)。**← 真正剩下的、"同空间"和"同模态"都救不了的残余风险。**
- 我们的 token 是**预测的未来 milestone**,而 VLM 预训练只见过**当前真实图像**。一张"未来图像 token"混进现时 prefix,仍可能:① 和真实当前图像在自注意力里竞争、污染 π0.5 的现时场景理解 / 子任务 ℓ̂ 预测;② 触发 [`RESEARCH_DIRECTION`](RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md) §3.3 警告的 **RoPE/时间位错**(现在 patch vs 未来 patch 空间位重叠)。
- 但这是**弱风险且可缓解**(type/segment embedding 标"未来"、时间 RoPE 偏移、或干脆位置④块掩码),**不是** 3D-Mix 式的模态硬伤。
- 与位置无关的真伤只剩一条:**AE-Fusion(expert 内部新增注意力头)3.13**——特征已对齐、模态已投影,它崩纯粹因"往精调注意力里硬插新机构",**这条模态无关,对我们成立,该排除**。

**结论(修正上一版)**:我们同时豁免风险 A(同空间)和大半风险 B(同为图像模态);只剩弱的风险 C(时态)和一条可排除的机制(AE-内部)。→ **不能因 3D-Mix 就判 prefix(③)死刑**;③ 的真实代价是"时态 novelty",而非 3D-Mix 那种模态崩。**③ 与 ④ 都进可测 shortlist,让实验裁决,不预设。**

---

## 5. 排除清单 + 值得测试的 shortlist

### 5.0 先排除(一定不可行 / 被支配,不必浪费实验)
| 方案 | 排除理由 | 依据 |
|---|---|---|
| **替换当前观测 patch token**(Oat-VLA-replace 谱) | 拿"预测未来"替掉"当前观测"= 让策略瞎掉现时场景,逻辑上就错(Oat-VLA 替的是当前物体 token,没人替成未来) | 原理 |
| **action expert 内部新增 cross-attn 头**(旧备选 A 的危险实现) | 特征已对齐、模态已投影仍崩 3.13 → **模态无关的真伤**,对我们成立 | 3D-Mix AE-Fusion 3.13 |
| **像素域 overlay / 输入像素增强** | 我们是 latent,不可画成像素,机制上不适用 | TraceVLA-overlay / IA-VLA(仅像素) |
| **VLM 当规划器 / 纯数据生成** | 放弃"注入策略网络"这个目标本身 | Manipulate-Anything(反面对照) |
| **MLLM 中间层 adapter cross-attn** | 侵入 backbone 且实测更差,被④支配 | 3D-Mix Middle-Layer 51.8 |
| **pooled 向量走 adaRMS 作唯一空间条件** | 单向量带宽太低喂不下 grid,丢空间;仅配当辅助 | INJECTION_DESIGN §2 |

### 5.1 值得测的 3 个方案(T0 → {T1, T2},头对头,不预设赢家)

**T0 · 辅助对齐 only(位置⑥,先做 go/no-go 闸)** — 最便宜、最保预训练、推理可丢。
```
加 K≈4–8 个可学习 milestone query token → 只用辅助损失(cosine/smooth-L1)对齐 pooled milestone
推理不注入。kill criterion: 困难子集 offline 无增益 → milestone 对策略无信息, 直接止损, 不做 T1/T2。
```
依据:3D-Mix Spatial Forcing/3D-Tokens(⑥,56–59 稳)+ FLARE。**先花一天跑这个,再决定要不要上重的。**

**T1 · 门控条件 KV(位置④,3D-Mix 冠军形态,最安全)** — 不依赖 VLM 会不会处理未来 token。
```
Ĝ_sub(16²×1152, SigLIP 同空间)
  → 复用 π0.5 SigLIP 图像投影(同分布,风险 A 已消)
  → 下采样到 K 个 milestone token(K≈16–64,Oat-VLA 证 16 够,先取小)
  → 每个 Gemma action-expert 层:milestone token 作**额外一段 conditioning KV**
      追加进该层"读 prefix"的既有 cross-attn(H_cond=[prefix-KV ; ★gated milestone-KV★])
  → position-specific 零初始化门 g^(i)=σ(W·meanpool(H_MLLM^(i))+b), b≪0 使初始≈0(3D-Mix π-style 逐层门控)
  → 块掩码:milestone 不进 VLM 自注意力(绕开风险 C)
  → KI stop-grad + CFG dropout + 弃权 全保留;冻 provider,只训投影+门+(可选)LoRA
```

**T2 · prefix 未来图像 token(位置③,我们原定档,模态-native 赌注)** — 让 VLM 自己"看见目标"。
```
同 T1 前三步(同空间投影 + K token) 但:
  → 摆进 prefix,**参与 VLM 自注意力**(赌 TraceVLA 式"图像模态进视觉流可行")
  → 显式 type/segment embedding 标"未来" + 时间 RoPE 偏移(缓解风险 C 时态混淆)
  → KI stop-grad + 零初始化投影 + CFG dropout + 弃权
```
T2 相比 T1 多一个赌注:VLM 能否有益地语义推理这张"未来图像"。**赢面来自模态-native(TraceVLA),风险来自时态 novelty(风险 C)。T1 vs T2 正是要用实验分出来的。**

### 5.2 逐条更新裁决
| 项 | 旧(INJECTION_DESIGN / LMWM2_FINAL) | 新(本文) |
|---|---|---|
| 主推位置 | ③ prefix 虚拟图像 token(单一定档) | **③(T2)与 ④(T1)并列可测,实验裁决**;先 T0 闸 go/no-go |
| 旧备选 A | "expert 内加 cross-attn 子层" | ⚠️ 若实现成 ⑤(expert 内部新注意力)= AE-Fusion 3.13,**排除**;安全形态=T1 的④(复用既有 cross-attn) |
| 备选 F(FLARE 辅助) | 首实验/正则 | 升级为 **T0 必跑 go/no-go 闸**——3D-Mix Spatial Forcing/3D-Tokens(⑥,56–59)证明"训练对齐+推理丢弃"稳且零开销 |
| 门控 | 零初始化投影 | 零初始化投影 **+ 逐层 position-specific 门控**(3D-Mix 冠军关键:不只 init=0,是可学习自适应权重) |
| 同空间论证 | "prefix 从最危险变最原生" | **基本成立且更强**:同空间消风险 A、同为图像模态消大半风险 B;prefix 仅剩弱的风险 C(时态),可缓解——故 prefix 不判死刑 |

### 5.3 保持不变(经受住外部对照)
- **走 SigLIP 同空间**(LMWM2_FINAL 赌注 6):Oat-VLA 反证同源对齐是硬优势,不变。
- **grid 不 pool 作空间条件**:SAM2Act 稠密特征注入、3D-Mix 用 grid token 都印证;pool 仅作可选 reasoning query。
- **冻结上游只训融合 + KI + CFG dropout + 弃权**:3D-Mix "冻结 VGGT 只训 fusion" 完全同构,不变。

---

## 6. 实验计划修订(最重要的可执行改动:评测口径)

3D-Mix(LIBERO in-domain +1.5 vs SIMPLER OOD +10.4)与 IA-VLA(仅在"重复物体需外推"的困难任务上收益大,简单场景反降)给出同一条铁律:**注入信号的价值只在需要它的困难/OOD 子集上才显现,in-domain 平均指标会把它稀释成噪声**。我们旧计划 I0–I3 用"通用 val 的 action-MAE"作判据 —— **这会假阴性**。

| 阶段 | 旧判据 | **修订判据** |
|---|---|---|
| **T0** 辅助对齐 go/no-go(⑥,GT milestone) | offline action-MAE 有无辅助 | 加 **cosine 对齐 loss 收敛性** sanity;**困难子集无增益 → 止损不做 T1/T2** |
| **T1/T2** 接线(GT milestone) | action-MAE base vs +milestone | **分层报**:总体 + **困难/OOD 子集**(多物体外推 / 长程 / 未见摆位);只看总体会漏掉信号 |
| **位置头对头**(T1 ④ vs T2 ③) | — | 同数据同预算比 ④门控KV vs ③prefix,复刻 3D-Mix 方法论,**在我们的图像模态上自证位置排序**(⑤已排除,不必再测) |
| 漂移检查 | 语言跟随/原任务退化 | 同;**重点测 T2(③)是否真掉 VLM 语言 grounding**(风险 C 的直接证据;若掉 → ④胜) |
| 真预测器 + 集群 | SR vs LaWM 98.6% | 同;OOD 子集 SR 为主指标 |

**GT-first 铁律不变**:先 GT milestone 隔离注入机制,再换真预测器。

---

## 7. 一页决策摘要
1. **不急于定论**:3D-Mix 的位置灾难是**3D 模态混淆**的,我们是**图像模态**(TraceVLA 反证图像进视觉流可行)→ prefix(③)不判死刑,与门控 KV(④)并列可测。
2. **风险三分**:A 特征分布位移(同空间→豁免)、B 模态-VLM 先验(同为图像→豁免大半)、C 时态 novelty(预测未来≠当前观测→**唯一残余,弱且可缓解**:future-tag / RoPE 偏移 / 块掩码)。
3. **排除清单(§5.0)**:替换当前观测 token、expert 内新增 attn 头(AE-Fusion 3.13)、像素 overlay/输入增强、VLM 当规划器、MLLM 中间层注入、pooled-adaRMS 作唯一空间条件。
4. **可测 shortlist(§5.1)**:**T0** 辅助对齐 go/no-go 闸(最便宜先跑)→ **T1** 门控条件 KV(④,最安全)vs **T2** prefix 未来图像 token(③,模态-native 赌注),头对头。
5. **评测铁律**:必分**困难/OOD 子集**(3D-Mix in-domain +1.5 / OOD +10.4;IA-VLA 仅困难任务有益),否则假阴性。
6. **不变**:SigLIP 同空间 / grid 不 pool / 冻结上游 + KI + CFG dropout + 弃权 / GT-first / 逐层零初始化门控。

## 8. 引用(本轮)
- 3D-Mix for VLA `2603.24393`(⚠️待核验) · IA-VLA `2509.24768` · TraceVLA `2412.10345`(ICLR'25) · Oat-VLA `2509.23655` · SAM2Act `2501.18564` · Manipulate-Anything `2406.18915`
- 内部:[`INJECTION_DESIGN_2026-07.md`](INJECTION_DESIGN_2026-07.md) · [`LMWM2_FINAL_ARCHITECTURE.md`](LMWM2_FINAL_ARCHITECTURE.md) · [`RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md`](RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md) · π*0.6 `2511.14759` · KI `2505.23705`
