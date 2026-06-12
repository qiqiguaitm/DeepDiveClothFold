# 跨 Episode 重复度挖掘 → 自动 Milestone / Value(AWBC 标签升级)

> **核心假说(用户,2026-06-11)**:同任务多条 episode 中**反复出现的图像/状态 = 任务必经过程(milestone)**;**低重复度的图像 = 非必要操作甚至 error/negative 样本**。据此从跨 episode 结构挖 value,替代/增强现有 AWBC 的逐帧进度回归。
> **状态**:🔬 自循环研究两轮完成(2026-06-11~12)。假说前半成立并落地(零训练 value τ=**0.865**,逼平监督 0.896,图21);后半实证否定(低重复 = 稀有 item,非错误,图6/16);特征层定版 = DINOv2 patch + 臂掩膜(§2.6)。下一决策点:milestone-value 重打标签 → AWBC 对照训练(§4.3)。
> **上游**:AWBC pipeline([awbc_implementation_plan.md](../../../deployment/strategy/awbc_implementation_plan.md));ViVa 对比([awbc_viva_value_comparison_plan.md](awbc_viva_value_comparison_plan.md),其 DSM-r30 变体**手标** milestone——本方案已证明可自动挖出,§2.3)。
> **动机(现有 pipeline 病根)**:pi0-AE 是单帧视觉回归器,`absolute_advantage = V(t+50)−V(t)` 二阶差分放大噪声(corr 0.896→0.3-0.4);完全不利用跨 episode 结构;且 AE 训练数据在完成瞬间截止 → vis episode 尾段 value 系统性下坠(end-drop,已实证)。

图像目录:`docs/visualization/cross_episode_recurrence_value/`(本文图 1-22 均相对引用,GitHub 直接渲染);视频不入 git,路径见附录 A。

---

## 0. 结论速览

**图21 — 一图总览:value 质量递进**(同 kai0_advantage 50 ep / 同协议)

![图21](../../../visualization/cross_episode_recurrence_value/summary_tau_progression.png)

**表1 — 核心结论与证据索引**

| # | 结论 | 关键数据 | 证据 |
|---|---|---|---|
| 1 | ✅ 假说前半成立:重复状态 = 必经 milestone | 覆盖率峰 82-92%,dom≈0,跨 3055 ep 稳定 | 图1/2,图10/11 |
| 2 | ✅ 自动 milestone 可替代 DSM 手标 | 同 30 ep:median \|Δt\| = 3.7% 时长,80% ≤0.10 | 图12,表8 |
| 3 | ✅ 零训练 V_milestone 逼平监督 value | τ 0.812 → **0.865(臂掩膜)** vs 监督 0.896 | 图21,表6/9,图5/20 |
| 4 | ✅ value 是状态触发且泛化零衰减 | held-out 50 ep τ=0.868(=挖掘集内);拼接测试段内τ≫全局τ;真机 3 轮 rollout 旁证 | §2.1(d),图22,图7/8 |
| 5 | ❌ 假说后半("稀有=negative")否定 | 低覆盖段 = 稀有衣物类型,三数据集一致 | 图6/16 |
| 6 | ❌ TCC 冻结特征版不如聚类路线 | TCC τ=−0.31 vs milestone 0.81 | 图13/14 |
| 7 | ⭐ 顶簇偏置 = 机械臂占画面(用户质疑发现)→ 臂掩膜修复 τ +0.05 | 92% 臂伪簇 → 82-90% 真布料状态簇 | 图17/18/19,表9 |

---

## 1. 文献调研与失败模式预注册

### 1.1 调研结论(deep-research,104 agents / 22 源 / 25 claims 三票核验,21 confirmed / 4 killed)

**表2 — 假说两半的文献判定**

| 假说 | 判定 | 依据 |
|---|---|---|
| 前半:重复 = 必经 | ✅ 25 年直接先例 | McGovern & Barto(ICML 2001,[PDF](https://mcgovern-fagg.org/amy_html/old/pubs/mcgovern_barto_isairs2001.pdf)):bottleneck = "在每条成功轨迹上都出现的区域";[L-Cut 2004](http://all.cs.umass.edu/pubs/2004/simsek_wb_TECH04.pdf) → [betweenness 2008](https://proceedings.neurips.cc/paper/2008/file/934815ad542a4a7c5e8a2dfa04fea9f5-Paper.pdf) → [2025 HRL survey](https://arxiv.org/abs/2506.14045) 确认为公认子目标发现准则 |
| 后半:稀有 = negative | ⚠️ 文献明示脆弱 | 2001 原文即警告"有用子目标也出现在稀有路径 → 负证据须软化";经典 diverse density 需要失败轨迹作负包(我们全成功,无形式化对应);现代侧唯一先例 = TCC Fig.7 异常检测(1 个定性例子,从未当 negative 标签用于 BC) |

**表3 — 四条方法学教训**

| # | 教训 | 出处 | 实现含义 |
|---|---|---|---|
| 1 | 裸帧频不行(every-visit 被停留时长主导) | McGovern 2001 | **first-visit:每 episode 每簇只计一次** |
| 2 | recurrence 要双阈值统计(Binomial) | L-Cut 2004 | 出现 episode 数 + hit 比例双门限 |
| 3 | milestone = 局部峰值,非全局阈值 | Betweenness 2008 | 按邻域选峰(后经 §2.5 修订:**宽 basin/top-覆盖率优先**) |
| 4 | 离散图方法在连续 RGB 不 scale | HRL survey 2025 | embedding 层是前提 |

**表4 — 现代采用线**

| 工作 | 提供什么 | 核验 |
|---|---|---|
| **TCC**(CVPR'19,[1904.07846](https://arxiv.org/abs/1904.07846)) | cycle-consistency = 逐帧"共性分数";进度信号 τ 0.75 vs TCN 0.66(from scratch) | 3-0 |
| **XIRL**(CoRL'21,[2106.03911](https://arxiv.org/abs/2106.03911)) | 跨 episode 对齐(零标注)→ value = 到 goal 嵌入负距离;[代码开源](https://github.com/google-research/google-research/tree/master/xirl) | 3-0 |
| **GraphIRL**(CoRL'22,[2207.14299](https://arxiv.org/abs/2207.14299)) | **先抽象掉外观再对齐**(治 nuisance)——本方案臂掩膜(§2.6)即此原则落地 | 3-0 |

被 3 票否决的捷径(勿按此预算):"标 1 条参考 episode 经 TCC 传播 ≈ 50 条全标"(0-3);VIP/LIV/AWE/ILEED 无幸存核验 claim。完整阅读指引见 §6。

### 1.2 失败模式预注册(实验前写下;后续实测命中情况)

**表5**

| 失败模式 | 预注册缓解 | 实测 |
|---|---|---|
| 多策略/多 item 劈裂 recurrence | 按 item/策略分组再挖 | ✅ 命中:稀有 item 全程零 value(图16 右) |
| 稀有恢复动作误杀 | 软负 + 人工审计 | 审计显示低覆盖主因是 item 而非恢复(图6) |
| 外观多样性主导聚类 | 先抽象(GraphIRL 原则) | ✅ 命中变体:**机器人本体**才是最大 nuisance(图17,§2.6) |
| TCC 单调相位假设失真 | 先小规模验证 | ✅ 命中:冻结特征版塌缩/反向(§2.4) |

---

## 2. 实验记录

### 2.1 V0 探针 — 假说首次验证(2026-06-11)

**协议**:50 episode × top_head 相机 × 3Hz → 冻结 DINOv2-small 特征 → KMeans(k=48)→ **first-visit episode 覆盖率**;milestone = top-10 覆盖率簇按时序;V_milestone(t) = 已首入 milestone 数 / 10。脚本 `train_scripts/kai/data/recurrence_v0_probe.py`。

#### (a) 覆盖率结构真实存在(smooth800,50 ep / 5968 帧)

峰值 **c4 = 92%**@t=0.78、c36 = 78%@t=0.78、c23 = 72%@t=0.44(图1);**dom(单 episode 占比)全部 7-15%** → 真跨 episode,非个体伪影。代表帧网格(图2)肉眼可辨任务阶段,且同簇混不同布色 → DINOv2 抓状态而非颜色。
> ⚠️ c4 后被证明是"臂占画面"伪簇(§2.6,图17),中段峰为真。

**图1 — smooth800 覆盖率 vs 簇时间位置**(峰 = 候选 milestone)

![图1](../../../visualization/cross_episode_recurrence_value/v0_smooth800_coverage_curve.png)

**图2 — 高覆盖簇代表帧网格**(每行一簇 × 4 个不同 episode;同簇混布色)

![图2](../../../visualization/cross_episode_recurrence_value/v0_smooth800_milestone_clusters.png)

**图3 — 每 episode 时间线按帧所属簇覆盖率着色**(红 = 低覆盖候选段,审计定位用)

![图3](../../../visualization/cross_episode_recurrence_value/v0_smooth800_episode_timeline.png)

#### (b) GT 验证:零训练 value vs `stage_progress_gt`(kai0_advantage,50 ep)

kai0 覆盖率结构与 smooth800 同构(峰 88%@t=0.39、86%@t=0.46,图4)。

**图4 — kai0 覆盖率曲线**

![图4](../../../visualization/cross_episode_recurrence_value/v0_kai0_coverage_curve.png)

**表6 — 零训练 value 与 GT 的相关性**

| value(纯重复度挖掘,零标注) | Kendall τ(mean/median) | Pearson r |
|---|---|---|
| **V_milestone**(top-10 覆盖簇首入) | **0.812 / 0.854** | 0.805 |
| V_tpos(簇平均时间位置) | 0.553 / 0.593 | 0.641 |
| [对照] pi0-AE absolute_value(**监督**) | — | ≈0.896 |
| [对照] 线性时间(trivial 上界,GT 分段线性) | 1.000 | — |

**图5 — V_milestone(红阶梯)单调跟随 GT(黑)**;蓝点 = V_tpos。已见局限:后段 milestone 稀疏 → 饱和 ~0.7(臂掩膜版改善见图20)

![图5](../../../visualization/cross_episode_recurrence_value/v0_kai0_gt_validation.png)

> 诚实标注:episode 内高 τ 含"时间单调"成分;**鉴别性证据在 (d)**——held-out 泛化零衰减 + 拼接边界重置,时间驱动信号做不到。

#### (c) 低覆盖段审计 — 假说后半的判决

bottom-decile 簇(覆盖率 4-10%)段落缩略图人工抽看,**三个数据集一致:低覆盖段由稀有衣物类型主导**——既非 error 也非 recovery(图6)。
→ **"低重复 = negative" 否定**:硬标 negative 会系统性惩罚稀有品类;只能软降权 + **按 item/策略分组后再挖**(表5 预注册缓解的实证确认;value 层面的后果见图16)。

**图6 — 低覆盖段审计样例(三数据集)**

| smooth800:深色 T 恤 | smooth800:白长袖 | dagger:红长裤 |
|---|---|---|
| ![](../../../visualization/cross_episode_recurrence_value/audit_smooth800_lowcov_rare_tshirt.jpg) | ![](../../../visualization/cross_episode_recurrence_value/audit_smooth800_lowcov_rare_whitesleeve.jpg) | ![](../../../visualization/cross_episode_recurrence_value/audit_dagger_lowcov_rare_redpants.jpg) |

#### (d) 鉴别性实验:value 是状态触发而非时间驱动

**(d1) held-out 泛化验证(主证据,kai0 demo 数据)**:milestone 在 50 ep 上挖掘,在**另外 50 条从未参与挖掘的 episode** 上验证(armmask 特征):

| | 挖掘集内 50 ep | **held-out 50 ep** |
|---|---|---|
| V_milestone τ vs GT | 0.865 / 0.886 | **0.868 / 0.879** |

→ **泛化零衰减**:milestone 是任务的真结构,不是对挖掘集的过拟合。

**(d2) 人工双轮拼接测试(vis demo 数据)**:把两条 held-out episode 首尾拼接 = 可控构造"两轮叠衣",milestone 命中等级在拼接点后**从低位重新爬升**(图22);量化(20 对拼接):段内 τ(等级 vs 段内时间)= 0.33 ≫ 全局 τ = 0.18 —— 时间驱动的信号两者应相等,**状态触发才会段内重置**。诚实标注:逐帧原始等级噪声大(τ 量级温和);累计式 V_milestone(实际使用形态)的强证据在 (d1)。

**图22 — 拼接测试:命中等级在人工轮次边界后重置**

![图22](../../../visualization/cross_episode_recurrence_value/discriminative_concat_heldout.png)

**(d3) 真机 rollout 旁证(解读已修正)**:autonomy rollout(7676 帧)实为 **3 次连续叠衣**(⚠️ 初版误读为"失败重试",经用户指正修正)。demo 挖掘的 milestone 跨数据集 assign 到该 rollout,V_milestone 在每轮边界(~3000 / ~5000 帧,布重新摊开处)回落再爬升,与监督 pi0-AE 的三段结构一致(图7/8;corr(V_ms, pi0-AE)=0.275)——是 (d2) 结论在真机数据上的自然复现,但因 rollout 含多轮、变量不可控,**仅作旁证,主证据为 (d1)/(d2)**。

> 方法注记:曾设计"倒放测试"后撤销——value 是逐帧图像的确定函数(无时序模型),"倒放镜像"在构造上恒成立,无鉴别力;causal 窗口下镜像比较亦不成立。

#### (e) vis_dagger 探针(2026-06-09-v2,50 ep)

覆盖率比 smooth800 更高(median 53% vs 25%,峰 88%,图9)→ dagger 数据更同质;其低覆盖段同样是稀有 item(图6 右)。

**图9 — dagger 覆盖率曲线**

![图9](../../../visualization/cross_episode_recurrence_value/v0_dagger_coverage_curve.png)

### 2.2 全量挖掘 — milestone 跨规模稳定(集群 8×A100 提特征 + 本地挖掘)

**表7 — 全量挖掘**(MiniBatchKMeans k=64 + 局部峰;此选法后经 §2.5 修订)

| 数据集 | eps / 帧 | 覆盖率 | 关键观察 |
|---|---|---|---|
| kai0_advantage | **3055** / 337k | med 36%,max 79% | **中段峰 t≈0.44-0.46(71-79%)与 50-ep 探针一致**;dom≈0%(图10) |
| smooth800 | 806 / 94k | med 27%,max 63% | 峰 t=0.37-0.46 + 0.85;覆盖率更低 = 跨 10 日期/多衣物多样性(图11) |

**图10 — kai0 全量(3055 ep)** | **图11 — smooth800 全量(806 ep)**(红★ = milestone 局部峰)

![图10](../../../visualization/cross_episode_recurrence_value/full_kai0_3055ep_coverage.png)
![图11](../../../visualization/cross_episode_recurrence_value/full_smooth800_806ep_coverage.png)

### 2.3 自动 milestone vs ViVa-DSM 手标 — 同 episode 直接对比

在 DSM 手标过的 **task_a_0509v2 同 30 episodes** 上跑 V0 协议(0509 单日数据极同质:两个簇 **100% 覆盖**,top-10 全 ≥83%)。

**表8 — 对齐精度**(120 个手标边界)

| 指标 | 结果 |
|---|---|
| 到最近自动 milestone 的 \|Δt\| | **median = 3.7% episode 长度(≈2.5s)**,mean 5.5% |
| ≤0.05 / ≤0.10 命中率 | 59% / **80%** |

**图12 — 自动(紫点)vs 手标(黑竖线),30 episodes**

![图12](../../../visualization/cross_episode_recurrence_value/milestones_auto_vs_dsm_hand_0509.png)

→ **自动挖掘以 ~4% 时长精度复现手标边界**——"替代 ViVa-DSM 手标"获得同 episode 直接证据。

### 2.4 TCC 复现(XIRL 官方代码)— 冻结特征版裁决:不如聚类路线

代码 sparse-checkout 至 `/vePFS/tim/workspace/recurrence_research/google-research/{xirl,tcc}`;适配器 `tcc_train_features.py`(冻结 DINOv2 + MLP head + 官方 `compute_tcc_loss`;XIRL value = −‖emb−goal‖)。

- **v1(默认参)塌缩**:loss 恒 0.0815 ≈ Var(U(0,1)) = 1/12(soft-NN 均匀的局部最优,图13);
- **v2**(normalize_embeddings + lr↑;并修 XIRL `one_hot` 的 CPU/CUDA device bug):val τ = **−0.31**(弱且反向;raw DINO 到 goal 距离 τ=−0.11),value 非单调(图14);
- **verdict**:冻结特征上"到 goal 距离"不是好进度信号(与表6 V_milestone ≫ V_tpos 一致);XIRL 原方为端到端训 backbone → 列为后续可选,**聚类-milestone 为已验证主线**。

**图13 — TCC v1 塌缩 loss** | **图14 — TCC value 非单调**

![图13](../../../visualization/cross_episode_recurrence_value/tcc_kai0_loss_curve.png)
![图14](../../../visualization/cross_episode_recurrence_value/tcc_kai0_value_curves.png)

### 2.5 逐 episode 视频与 value 鲁棒性配方(4 版迭代)

为 5 个代表 episode 出同步视频(附录 A):kai0 ep1949/2923(带 GT)、smooth800 ep594 / ep137(稀有 T 恤)、dagger ep73(红长裤)。

- v1(全量挖掘质心 + 局部峰):单帧误配把 max 弹飞(ep2923 开局跳 0.91);
- v2 去抖(≥2 连续帧):持续性误配仍在;v3 顺序门(仅 +1/+2 推进):矫枉过正,前级未命中则卡 0;
- **v4 = 回到 V0 配方**(full KMeans k=48 + **top-覆盖率** milestone + 朴素首入)✅ 干净阶梯(图15)。

**教训**:鲁棒性来自"**宽 basin 簇 + top-覆盖率选 milestone**",不是事后过滤;MiniBatch 细质心 + 弱峰是误配根源(→ 表7 的局部峰选法弃用)。

**图15 — v4 配方:零训练阶梯紧贴 GT(kai0 ep1949)**

![图15](../../../visualization/cross_episode_recurrence_value/epvideo_kai0_1949_staircase_vs_gt.png)

**图16 — 稀有 item 的 value 退化(诚实展示;呼应图6 与表5)**:左 ep137(深色 T 恤)阶梯封顶 0.58;右 ep73(红长裤)**全程 value=0** —— 未分组挖掘下稀有 item 被判零进度,"按 item 分组"非做不可

| ep137:封顶 0.58 | ep73:全程 0 |
|---|---|
| ![](../../../visualization/cross_episode_recurrence_value/epvideo_s800_137_rareitem_cap058.png) | ![](../../../visualization/cross_episode_recurrence_value/epvideo_dagger_73_rareitem_zero.png) |

### 2.6 覆盖率偏置(用户质疑驱动)→ 臂掩膜修复:τ +0.05(2026-06-12)

**用户质疑**:① 高覆盖会不会是"衣物近头部相机、占画面大"导致视觉同质?② 手腕相机夹取时高覆盖是否干扰?

**定量检查**(簇覆盖率 vs 簇内离散度):总体**正相关**(smooth800 r=0.42 / kai0 r=0.47,top8 覆盖簇离散度反而更高)→ 多数高覆盖簇是宽 basin 真语义簇;**但**"高覆盖+低离散"规则恰好命中最高峰 c4(92%@t=0.78)。

**视觉证实(图17)**:c4 代表帧 = **机械臂横穿头部相机前景** —— **真凶不是衣物而是机器人本体**(臂外观跨 episode 恒定,占画面即嵌入塌缩)。质疑② 成立:手腕相机接触段是布料纹理特写、跨阶段同质 → 阶段挖掘**纯干扰**(挖掘只用 top_head 正因于此;手腕"进特写"onset 可作接触事件标记)。

**图17 — c4 臂伪簇代表帧**(两个不同 episode,画面几乎相同)

| ep660 | ep585 |
|---|---|
| ![](../../../visualization/cross_episode_recurrence_value/bias_c4_armcrossing_ep660.png) | ![](../../../visualization/cross_episode_recurrence_value/bias_c4_armcrossing_ep585.png) |

**修复 = 臂掩膜特征**(GraphIRL 原则落地,表4):c4 帧 patch 聚类 + 颜色启发 → 6 个臂原型;帧嵌入 = DINOv2 patch tokens 剔除(原型相似 >0.6 ∪ 橙色线缆)后均值。**纯 DINO 相似度即可分开金属臂 vs 深色布**(早版"暗色"规则会误吃深色 T 恤,已去,图18);8×A100 全量重提(806 + 3055 ep)。

**图18 — 掩膜可视化验证**(左 3 列:深色 T 恤不再被吃;右 2 列:臂帧覆盖完整;上/下行 = 阈值 0.6/0.7)

![图18](../../../visualization/cross_episode_recurrence_value/armmask_overlay_check.png)

**表9 — 臂掩膜前后对比(同 50 ep、同协议)**

| 指标 | 未掩膜 | **臂掩膜** |
|---|---|---|
| kai0 V_milestone τ vs GT(mean/median) | 0.812 / 0.854 | **0.865 / 0.886**(逼平监督 0.896,见图21) |
| smooth800 最高覆盖簇 | c4 92%@t0.78(臂伪簇,图17) | c18 82%@t0.41(**真布料摊开状态**,跨布色,图19 左) |
| kai0 最高覆盖簇 | — | c13 90%@t0.51(中段) |
| 中段 milestone 覆盖率 | 56-72% | **68-82%**(臂噪声剔除后布状态簇被洗干净) |
| 可疑(高覆盖+低离散)簇 | c4 92%(顶) | 各剩 1 个晚段残留 ~75%(非顶,图19 右) |

**图19 — 掩膜后:新顶簇 = 真布料状态(左);晚段残留(右,"空桌+角落布"残签名仍似,处理列入表10 步 a)**

| 新顶簇 c18(82%@t0.41) | 晚段残留 |
|---|---|
| ![](../../../visualization/cross_episode_recurrence_value/armmask_newtop_clothstate_ep566.png) | ![](../../../visualization/cross_episode_recurrence_value/armmask_residual_arm_ep590.png) |

**图20 — 臂掩膜后的零训练阶梯 vs GT**(与图5 同 episodes 对照;ep2923 ——此前误配重灾户——现紧贴 GT)

![图20](../../../visualization/cross_episode_recurrence_value/armmask_gt_validation.png)

---

## 3. 阶段结论

1. **假说前半(重复=必经)**:✅ 三数据集 + 全量 3055 ep 成立,milestone 跨规模稳定(图1/4/10);
2. **自动 milestone 可替代 DSM 手标**:✅ median 3.7% 时长精度(表8,图12);
3. **零训练 V_milestone**:✅ τ=0.865(臂掩膜)逼平监督 0.896(图21,表9),状态触发可迁移真机(图7/8);
4. **假说后半(稀有=negative)**:❌ 低覆盖 = 稀有 item(图6/16),只能软处理 + 按 item 分组;
5. **TCC 冻结特征版**:❌ 弱于聚类路线(§2.4),端到端列后续;
6. **配方定版**:特征 = DINOv2 patch + 臂掩膜均值;milestone = top-覆盖率宽 basin;value = 首入阶梯。

---

## 4. 方案(V1 修订版)

### 4.1 修订依据
原"TCC 对齐为主线"→ "**聚类-milestone 为主线**"(§2.4);特征层加臂掩膜(§2.6);milestone 选法 top-覆盖率(§2.5);低重复段只做软处理(§2.1c)。

### 4.2 V1 pipeline

**表10**

| 步 | 内容 | 依据 |
|---|---|---|
| a | 全量 3Hz 特征:**DINOv2 patch + 臂掩膜均值**(残留簇细化:桌面 patch 同掩 / 布 patch 数下限) | §2.6,表9,图19 |
| b | **按 item/策略分组**(整体轨迹嵌入聚类),组内分别挖掘 | 图6/16 实锤 |
| c | 组内 full KMeans(k≈48)→ **top-覆盖率 milestone**(Binomial 双阈值,表3) | §2.5 教训 |
| d | V(t) = milestone-index(单调)→ 喂回现有 `discretize_advantage.py`(AWBC 训练侧零改动) | 表6/9 |
| e | 低 recurrence 段 → 软降权 / "uncertain" 第三 prompt 标签 + 审计,**绝不自动硬负** | §2.1c |
| f | (可选)TCC 端到端 backbone 增强 | §2.4 |

### 4.3 下一决策点:AWBC 对照训练(待拍板)

milestone-value 重打 smooth800 advantage 标签 → AWBC 训练,对照现 pi0-AE 标签版(**同 init / 同配方,单变量 = 标签来源**,沿用 awbc_viva 对比框架);offline 看差分 advantage corr 是否优于 0.3-0.4 基线,**真机为终判**。

---

## 5. 基础设施与执行记录

**表11 — 集群任务**(均 cnsh;pod venv = `xvla/X-VLA-env/.venv`)

| 任务 | 内容 | 结果 |
|---|---|---|
| `t-20260611215738-pvc4d` | 开发机队列 2×A100 部署测试 | 分卡成功;暴露 `kai0/.venv` python symlink 指 `/home/tim`(pod 无)→ 空跑 |
| `t-20260611230152-x5k2d` | 8×A100 全量特征提取(smooth800+kai0+dagger) | **14 分钟**;坑:缺失视频弄死 shard(已加 skip) |
| `t-20260611232906-qlbsp` | round2:smooth800 补齐 + 0509 探针 | 完成(806/806) |
| `t-20260611234053-jzr7z` | round3:TCC v2 训练 | 完成(τ=−0.31 verdict) |
| `t-20260612100427-kzz5l` | 臂掩膜全量重提(806 + 3055) | ~50 分钟 |

**经验定论**:pod venv 一律用 `xvla/X-VLA-env/.venv`(vePFS 自包含,已补 matplotlib/sklearn);开发机队列禁 Flexible 资源 → Preset `ml.pni2.7xlarge`;DINOv2 权重缓存 `HF_HUB_CACHE=/vePFS/tim/workspce/hf_cache/hub_default`(pod 离线可载);⚠️ gf0 本地 GPU 驱动 2026-06-11 ~15:00 消失(nvidia-smi 0 字节)→ 本地仅 CPU。

---

## 6. 参考文献 + 阅读指引(经 3 票核验)

### 第一梯队(必读)

| # | 文献 | 链接 | 重点读什么 |
|---|---|---|---|
| 1 | Dwibedi et al., **TCC**, CVPR 2019 | [arXiv:1904.07846](https://arxiv.org/abs/1904.07846) | §3 cycle-consistency(= 逐帧共性分数);Table 6(τ 0.75 vs TCN 0.66 from-scratch;finetune 下 TCN 反超、组合最佳 0.878);Fig.7 异常检测("稀有=negative"唯一定性先例) |
| 2 | Zakka et al., **XIRL**, CoRL 2021 | [arXiv:2106.03911](https://arxiv.org/abs/2106.03911) | TCC 跨 episode 对齐 → value = 到 goal 负距离;如何消除"对单条参考轨迹对齐";[代码](https://github.com/google-research/google-research/tree/master/xirl) |
| 3 | McGovern & Barto, ICML 2001 | [PDF](https://mcgovern-fagg.org/amy_html/old/pubs/mcgovern_barto_isairs2001.pdf) | 假说原始形式化;first-visit vs every-visit;§6 软负警告 |

### 第二梯队(实现前)

| # | 文献 | 链接 | 重点 |
|---|---|---|---|
| 4 | **GraphIRL**, CoRL 2022 | [arXiv:2207.14299](https://arxiv.org/abs/2207.14299) | 先抽象外观再对齐(臂掩膜的理论依据);借原则不借实现 |
| 5 | Şimşek et al., **L-Cut**, 2004 | [PDF](http://all.cs.umass.edu/pubs/2004/simsek_wb_TECH04.pdf) | Binomial 双阈值 recurrence 判定 |
| 6 | Şimşek & Barto, NeurIPS 2008 | [PDF](https://proceedings.neurips.cc/paper/2008/file/934815ad542a4a7c5e8a2dfa04fea9f5-Paper.pdf) | milestone = 局部极大(峰在门口"两侧"的启发) |

### 第三梯队(背景)

| # | 文献 | 链接 |
|---|---|---|
| 7 | HRL Survey 2025 | [arXiv:2506.14045](https://arxiv.org/abs/2506.14045) |
| 8 | VIP / LIV(视频 value 对照线,未经本轮核验) | [2210.00030](https://arxiv.org/abs/2210.00030) / [2306.00958](https://arxiv.org/abs/2306.00958) |
| 9 | AWE / Keyframe-Focused IL | [2307.14326](https://arxiv.org/abs/2307.14326) / [2106.06452](https://arxiv.org/abs/2106.06452) |
| 10 | LAV / GTCC(TCC 单调假设局限) | [2103.17260](https://arxiv.org/abs/2103.17260) |

**开放问题(也是可发表贡献点)**:低重复/对齐误差当 negative 标签用于 weighted BC 无已发表先例;"recurrence → 自动 milestone → AWBC 标签"完整链无人发表。

---

## 附录 A — 工件清单

**图像**(图1-21):`docs/visualization/cross_episode_recurrence_value/`(24+ 张,命名规范 `<阶段>_<数据集>_<内容>`)。

**视频**(不入 git,在 `temp/`):

| 文件 | 内容 | 对应图 |
|---|---|---|
| `autonomy_ep0_4model_value_sync.mp4` | 真机 rollout × 四 value 同步(2560×840×7676f) | 图8 为抽帧 |
| `autonomy_ep0_recurrence_value_sync.mp4` | rollout × recurrence(win+mono)/pi0-AE/ViVa 三面板同步 | 图7 为静态版 |
| `recur_ep_kai0_{1949,2923}_sync.mp4` | 带 GT 的零训练阶梯 | 图15 |
| `recur_ep_s800_594_sync.mp4` | smooth800 正常 episode | — |
| `recur_ep_s800_137_rareitem_sync.mp4` / `recur_ep_dagger_73_rareitem_sync.mp4` | 稀有 item 退化展示 | 图16 |

**脚本**(`train_scripts/kai/data/`):`recurrence_v0_probe.py`(探针)· `recurrence_v0_gt_validation.py`(GT 验证)· `recurrence_value_on_rollout.py`(rollout 迁移)· `recurrence_vs_dsm_milestones.py`(手标对比)· `recurrence_full_mining.py`(全量挖掘)· `tcc_train_features.py`(TCC 适配 + 集群分片)· `build_arm_prototypes.py` / `extract_masked_features.py` / `armmask_compare.py`(臂掩膜三件套)。集群 YAML:`train_scripts/kai/volc/recurrence_*.yaml`。

**特征/挖掘缓存**(`temp/`):`tcc_{smooth800,kai0,dagger_*}/feat_cache`(原始)· `tcc_{smooth800,kai0}_armmask/feat_cache`(臂掩膜)· `full_mining_*/mining.npz` · `armmask/arm_prototypes.npz`。

**外部代码**:`/vePFS/tim/workspace/recurrence_research/google-research/{xirl,tcc}`(XIRL `one_hot` device bug 已 patch)。
