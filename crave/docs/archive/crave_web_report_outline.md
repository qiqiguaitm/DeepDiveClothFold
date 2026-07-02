# CRAVE 方法汇报网页 · 大纲(v1,关键决策已定)

> 目标:在 `web/showcase/reports/` 下新增一个**面向观看者**的 CRAVE 方法汇报页(区别于已有的详细技术报告 `crave_interp/`)。
> 核心诉求:**让人很快明白「CRAVE 是什么方法、好在哪、缺点在哪」**——观看者视角、视觉为主、文字精炼、诚实给优缺点。
> 落点:`web/showcase/reports/crave_report/index.html`(单页,自上而下叙事 + 左侧 TOC)。

## ✅ 已定决策(2026-06-26 沟通)
1. **受众 = 给领导 / 跨团队汇报** → 强结论、强对比、强优缺点;方法机制尽量**直觉化、可折叠**,不堆术语。顶部加 **10 秒 TL;DR 关键结论带**。
2. **深度 = 中等** → 含机制可视化 §3,但文字精炼、图为主、可 skim/折叠。
3. **与旧报告 = 新建汇报版 + 结尾链接 `crave_interp/` 详细版**(两页并存)。
4. **PI 系素材 = 用户提供** → §1.3 PI value 效果先留**占位**,拿到素材再填;KAI0-AE 用现成对比图。

## ⏳ 待你提供 / 待办
- **PI(π*0.6-RECAP 等)的 value 效果图 / 数据**(§1.3 占位);其余可先实现。

---

## 设计原则(观看者视角)

1. **痛点驱动**:先讲"现有 value 模型的现状与痛点",让观看者有代入感,再引出 CRAVE。
2. **一图胜千言**:每个关键点配一张图/一段视频;大量复用 `crave/docs/visualization/` 已有素材,少文字。
3. **渐进式**:每节先一句话结论(粗体大字),想深入再看下面的图/细节;不强迫读全。
4. **诚实**:单独一节讲**优缺点 / 适用边界**(这是观看者最想要、也最容易被藏起来的部分)。
5. **深入有出口**:需要逐行机制/复现的,链接到已有的 `crave_interp/` 详细技术报告,不在本页堆细节。

---

## 页面结构(草案)

### 0. Hero · 一句话讲清 CRAVE
- 大标题 + 一句话:**"零训练,从一堆 demo 里反复出现的状态,自动算出稠密的任务进度 value。"**
- 一张主视觉:一条干净的 value 曲线(0→1)叠在相机帧上(复用对齐视频首帧 / `rollout` 图)。
- 三个关键词徽章:**零训练 · 零标注 · 跨数据集泛化**。

### 0.5 TL;DR · 10 秒关键结论带(为领导/跨团队)
- 一行 3–4 个**大数字卡片**,busy 的人扫一眼就懂价值:
  - **零训练 / 零标注**(省每任务重训与标注成本);
  - **in-dist 与监督 AE 打平**,但**单调 100% vs 52%**(更干净);
  - **OOD 真机更稳**(退步信号 7% vs 45%,对齐真事件);
  - **跨数据集泛化 corr 0.956 / 0.988**(配方逐字不改)。
- 一句话定调:**"用监督模型一半的代价,拿到同等或更稳、还能跨任务直接复用的 value。"**

### 1. 背景:value 模型现状与**三大痛点**(全报告的主线,后续每节都回扣)
- **1.0 value 模型是干嘛的**(1 句给不熟的观看者):AWBC/RL 需要每帧"任务完成到几成"的稠密信号,据此判正/负 advantage(POSITIVE=该学、NEGATIVE=该避)。
- **1.1 一般流程**:KAI0-AE = **监督训练**回归 `stage_progress_gt`(pi0-AE);PI 系 = 监督进度差 / RL 回报。流程图(数据→标注/回报→训练 value 头)。**【PI value 效果占位:待用户素材】**
- **▶ 痛点①:AE 输出抖动严重 → AWBC 的 POSITIVE/NEGATIVE 评价极不稳定、互相交错。**
  - 后果:同一段动作里正负标签来回横跳,AWBC 学到的是噪声而非真信号。
  - **示意图(已有)**:`crave/docs/visualization/crave_3level_vs_ae3_labels_ep808.png` —— 下栏 AE value 剧烈抖动、NEG 35% / POS 38% 红绿点满屏交错;上栏 CRAVE 干净阶梯(NEG 7% / NORMAL 72% / POS 20%)。**对照一眼看穿痛点①**。
- **▶ 痛点②:传统 value 的 positive/negative 不够明确——很多地方主观上看不出 value 为什么升/降。**
  - **示意素材(待渲染→用户裁剪)**:渲染几个**较长 kai0_base episode** 的 **value mp4(相机帧 + value 游标 + POS/NEG/NORMAL 分类条)**;用户主观挑出"看不出凭什么升降"的片段做短视频示意。(脚本基础:`crave_3level_classify_video.py` / `crave_vs_ae_sync_video.py`。)
- **▶ 痛点③:叠衣任务末端大概率"突然下跌→恢复"。**
  - 根因:用**相似度度量**,折叠动作让画面外观骤变 → 与"已折叠"原型的相似度骤降 → value 假跌,折完才恢复(对折叠过程的相似度不鲁棒)。
  - **示意图**:从痛点②渲染的长 episode 视频里裁末端段最直观;静态备选用 raw 最近-milestone / 相似度 value 曲线(`rollout_nn_value.png` 等)显示中后段的相似度凹陷。
- **1.4 小结**:这三点(抖动→标签不稳 / 升降不可解释 / 末端假跌)+ 还要**人工标注、循环论证、每任务重训** → 引出 CRAVE 要解决什么。

> **全报告主线 = 用这三个痛点串起来**:§2/§3 讲 CRAVE 机制时点明各对应解掉哪个痛点,§4 用对比数据证明解掉了,§5 诚实说哪些仍是边界。

### 2. CRAVE 核心思想 + **痛点→解法对照**(对应"CRAVE 概述")
- **2.0 一句话对照表(全报告的钩子)**:
  | 痛点 | CRAVE 怎么解 |
  |---|---|
  | ① AE 抖动 → POS/NEG 交错 | milestone 阶梯 + Viterbi-DP 全局最优 → value 平滑单调,正负**段化稀疏**而非逐帧横跳 |
  | ② 升降不可解释 | value = "在哪个反复出现的 milestone 簇上" → 每次升/降都能指着具体 milestone 说清楚(可解释) |
  | ③ 末端相似度假跌 | 不只看瞬时相似度:Viterbi 转移惩罚 + **簇间转移概率**(知道"末段不会跳回起始态")→ 压住末端假跌 |
- **2.1 灵感来源**:同一任务的众多 demo 里,**关键 milestone 会在不同 episode 反复出现** → "反复出现 = 任务必经结构"。配一张多 episode 对齐 / 簇复现示意图。
- **2.2 三步直觉**(pipeline 图):frozen 视觉特征 → 聚类找"反复出现的态"(自动 milestone) → Viterbi-DP 读成稠密单调 value。**全程零训练**(DINOv2 冻结 + KMeans + DP,无梯度更新)。
- **2.3 充分利用「簇间转移概率」**(直接解痛点③):从 demo 统计"当前 milestone 的下一个最可能是谁"的概率,折进 DP 修正 value 假跌(承接已写好的 no-Viterbi → +Viterbi → 转移概率 叙事 + 转移矩阵图)。
- **2.4 一句话定位**:CRAVE = **C**ross-episode **R**ecurrence **a**s **V**alue **E**stimation。

### 3. 它怎么工作(用图讲清楚,**整节默认折叠/可展开**——领导看完 §2 直觉即可跳到 §4)
- **3.1 milestone 自动浮现**:簇词表 / 质心画廊(`crave_gallery_*` / 质心解码图)。
- **3.2 单 episode 沿簇链前进**:2D/3D 簇间流转图(`crave_cluster_flow_*`)——value 就是这条链上的进度位置。
- **3.3 value 读出三步**:`viterbi_mechanism.png`(无 Viterbi 乱抖 → Viterbi 平滑)+ 转移概率消假跌(`milestone_transition_*` / coffee 假跌修复)。

### 4. 效果对比(本页重头,每条**回扣痛点**)
- **4.1 in-distribution(自家主场)→ 解痛点①②**:CRAVE vs KAI0-AE——**打平且更平滑**(kai0_base ep2302:单调 100% vs 52%,噪声负 advantage 8% vs 32%)。直接对比痛点① 那张 `crave_3level_vs_ae3_labels_ep808.png`:AE NEG/POS 交错 35%/38% → CRAVE 7%/20% 段化。素材 `crave_vs_ae_kai0base.png` + 同步视频。
- **4.2 OOD 真机 rollout → 解痛点①**:CRAVE 退步信号**稀疏且对齐真事件**(两次回落对齐两个轮次边界,仅 7% 负),AE 弥散(45% 负)。`crave_vs_ae_autonomy.png`。
- **4.3 末端假跌修复 → 解痛点③**:转移概率前/后对比(几何读出末端塌到 0 → 转移概率紧贴对角线),`milestone_transition_*` / coffee 假跌修复图;再放一段叠衣末端 before/after 短视频。
- **4.4 跨数据集泛化**(配方逐字不改):**XVLA soft_fold corr 0.956 / 真实 ALOHA coffee corr 0.988**。泛化预览图/视频。
- **4.5 kai0 GT 量化表**:MAE 0.105 / Pearson 0.928 / τ 0.841(vs 旧 DP vs 监督 AE),诚实标注 AE 低 MAE 的循环论证。

### 5. 优缺点 / 适用边界(诚实,观看者必看)
- **优点**:零训练零标注 · 可解释(每帧能说出在哪个 milestone)· 跨数据集泛化 · 退步信号干净稀疏 · 平滑单调。
- **缺点 / 边界**:
  - 依赖**挖矿域与目标域对齐**(挖矿数据要和评估任务同分布);
  - **转移概率折进 DP 对正常走势是中性的**——只在"同态/循环态假跌"这种特定问题上有用,且必须轻权重(重了会冻结真实重复动作);**跳步检测不行**;
  - milestone-space 形态下**完成态读数偏低**,需保留一个软 completion bias;
  - τ 略低于监督 AE(前段动作多样、排序略噪);
  - 需要**足够多的同任务 demo**才能让"重复"显现。
- (可做成"优点绿 / 缺点橙"两栏对照,一眼看完。)

### 6. 一句话总结 + 决策小表 + 深入入口
- 总结句。
- **决策小表(为领导/选型)**:何时用 CRAVE / 何时用监督 AE。
  | 场景 | 选 CRAVE | 选监督 AE |
  |---|---|---|
  | 没有 GT/标注、想零成本起步 | ✅ | — |
  | 新任务/新本体要快速复用 | ✅(配方不改) | 需重训 |
  | 真机 rollout 要干净退步信号 | ✅(稀疏对齐) | 弥散 |
  | 已有大量 GT、只求单数据集极致 MAE | 可选 | ✅(主场) |
  | 同任务 demo 很少(重复不显现) | ⚠️ 慎用 | ✅ |
- 链接:**详细技术报告**(`crave_interp/`)· **代码**(`crave/` 包)· **复现命令**。

---

## 关键问题处置(已定见顶部「✅ 已定决策」)

- 受众=领导/跨团队 · 深度=中等 · 新建汇报版+链接旧版 · PI 素材待提供。
- **素材策略**:默认**直接复用** `crave/docs/visualization/` 与 `crave_interp/assets/` 已有 figures/videos(快);仅"多 episode 复现示意图"和数字卡片可能需**新画 1–2 张轻量示意**(SVG/matplotlib),实现时确认。

---

## 三痛点素材采集计划(沟通用)

| 痛点 | 示意素材 | 现状 |
|---|---|---|
| ① AE 抖动→POS/NEG 交错 | `crave/docs/visualization/crave_3level_vs_ae3_labels_ep808.png`(AE 下栏红绿点满屏交错 vs CRAVE 上栏阶梯) | ✅ 现成,直接用 |
| ② 升降不可解释 | 5 个长 kai0_base episode 的 CRAVE-vs-AE value+分类条视频 | ✅ **已渲染**(`temp/crave_kai0base_videos/`,无需 GPU 推理:AE 值来自 `advantage_q5` 预算输出) |
| ③ 末端假跌 | ②视频的末端段 + 静态备选 `rollout_nn_value.png` / 转移前后对比图 | 视频就绪;若末端假跌不够明显可另渲一段 before/after |

**渲染数据约束(②/③ 视频)**:
- **kai0_base**:原始 parquet **无 AE value**(需 pi0-AE 推理=GPU job)。现成只有 **ep2302**(99s,`_crave_ae_kai0base.npz`,已渲染 `crave_kai0ae_FULL_ep2302.mp4`)。
- **smooth800_dagger**:CRAVE 值(`temp/mv_value_full/`,1117ep)+ AE 值(`A_smooth800_dagger_all_awbc/absolute_value`,1117ep)**全现成**,可立即渲染任意长 episode(脚本 `crave_3level_classify_video.py` 已支持,仅写死了 ep808)。
- **渲染选项(待用户拍板)**:
  - **A(最快,即刻出)**:渲染几个**长 smooth800_dagger** episode 的 CRAVE-vs-AE value+分类条视频(数据全现成,无需推理;但属 dagger 域,ep808 已在旧报告出现过)。
  - **B(严格 kai0_base)**:先出 **ep2302**(现成);要"几个"更多 → 跑 **pi0-AE 推理**(2–3 个长 kai0_base ep,GPU/集群)再渲染。
  - **C(kai0_base 但只 CRAVE)**:为几个长 kai0_base ep 现算 **CRAVE 值**(可即刻)+ 分类条渲染——能讲痛点③末端假跌 before/after,但不展示 AE 痛点②。
  - 我的建议:**B 先出 ep2302 + A 补几个长 dagger** 兼顾"严格 kai0_base 样例 + 立刻有多个可裁素材";是否要为 kai0_base 跑 AE 推理由你定。

## 实现备忘(沟通定稿后再做,非本步)
- 复用 `crave_interp/index.html` 的样式骨架(同一 showcase 风格);新建 `crave_report/index.html` + `assets/`。
- 在 showcase 的报告注册处加一个 tab(具体注册位置实现时再查 `web/showcase/content|templates`)。
- 素材软链/拷贝自 `crave/docs/visualization/`。
