# 跨 Episode 重复度挖掘 → 自动 Milestone / Value(AWBC 标签升级)— 调研 + 方案

> **核心假说(用户,2026-06-11)**: 同任务多条 episode 中**反复出现的图像/状态 = 任务必经过程(milestone/bottleneck)**;**低重复度的图像 = 非必要操作甚至 error/negative 样本**。据此从跨 episode 结构挖 value,替代/增强现有 AWBC 的逐帧进度回归。
> **状态**: 🔬 **V0 实验完成(2026-06-11,gf0 本地)→ 假说前半在我们数据上初步成立**:覆盖率峰真实存在(92%@t=0.78)、零训练 V_milestone 与 GT τ=0.81、跨数据集到真机 rollout 复现重试结构;低覆盖段审计 = **稀有衣物类型为主(非错误非恢复)** → 硬负标签证伪。TCC 复现训练进行中。详见 §5。
> **上游**: AWBC pipeline([`../../../deployment/strategy/awbc_implementation_plan.md`](../../../deployment/strategy/awbc_implementation_plan.md));ViVa 对比([`awbc_viva_value_comparison_plan.md`](awbc_viva_value_comparison_plan.md),其 DSM-r30 变体**手标** milestone——本方案目标之一是自动挖出来)。
> **动机(现有 pipeline 的病根)**: pi0-AE 是单帧视觉回归器,`absolute_advantage = V(t+50)−V(t)` 二阶差分把噪声放大(corr 0.896→0.3-0.4);且完全**不利用跨 episode 结构**。另:AE 训练数据(kai0_advantage)在完成瞬间截止、无收尾段 → vis episode 尾段 value 系统性下坠(已实证,见 end-drop 分析)。

---

## 1. 调研结论(全部三票核验,引用见 §6)

### 1.1 假说前半("重复 = 必经")— ✅ 有 25 年直接先例

- **McGovern & Barto (ICML 2001)**([PDF](https://mcgovern-fagg.org/amy_html/old/pubs/mcgovern_barto_isairs2001.pdf))字面形式化了这个假说:bottleneck = "在成功路径上频繁经过、失败路径上不经过的观测区域",目标概念"在**每条**成功轨迹上都出现"。diverse density(多示例学习)挖掘,gridworld(找到门口)+ 连续状态机器人验证有效。
- 后续脉络:[L-Cut (2004)](http://all.cs.umass.edu/pubs/2004/simsek_wb_TECH04.pdf)(统计化 recurrence 判定)→ [betweenness centrality (NeurIPS 2008)](https://proceedings.neurips.cc/paper/2008/file/934815ad542a4a7c5e8a2dfa04fea9f5-Paper.pdf) → [2025 HRL survey](https://arxiv.org/abs/2506.14045) 确认为公认 subgoal 发现准则。

### 1.2 假说后半("稀有 = negative")— ⚠️ 文献明示的脆弱半边

- **McGovern & Barto 2001 原文就警告**:有用的子目标也会出现在稀有/失败路径上 → 负证据必须**软化**(Gaussian 宽度 / 按 bag 分级),不能硬性"出现在负包即排除"。
- 经典 diverse density **需要失败轨迹作负包**;我们 800-3000 条全成功 → "稀有=negative"在最强先例里**没有形式化对应**。
- 现代侧唯一先例 = TCC 论文([1904.07846](https://arxiv.org/abs/1904.07846) Fig.7)的异常检测提议("嵌入轨迹偏离典型轨迹的帧标为异常")——**仅 1 个定性例子(卧推视频),无定量基准,从未当 negative 标签用于 BC**。
- **自家数据里的反例**:抓角失败后的 **regrasp 恢复动作**是低频的,但正是我们最想要的能力(XVLA 抓不到角问题)。硬标 negative = 删掉恢复能力。
- → **结论:低 recurrence 段只能软降权 / 第三档 "uncertain" prompt 标签 + 人工审计,绝不自动硬标 "Advantage: negative"。**

### 1.3 四条方法学教训(决定实现细节)

| # | 教训 | 出处 | 对我们的含义 |
|---|---|---|---|
| 1 | **裸帧频计数不行**:every-visit 被停留时长主导("agent 大部分时间在房间里,极少在门口") | McGovern 2001 | 30Hz 视频慢段会霸占帧数,抓角/对折等关键瞬间反而帧少 → **first-visit:每 episode 每状态簇只计一次** |
| 2 | recurrence 判定要**双阈值统计**(Binomial:出现 episode 数 > t_o 且 hit 比例 > t_p) | L-Cut 2004 | 抗噪的 milestone 接受准则 |
| 3 | milestone = recurrence 的**局部峰值**(相对时间邻域),非全局阈值 | Betweenness 2008 | Rooms 域峰值在门口"两侧"而非门口本身 — 按邻域比较选峰 |
| 4 | 离散图方法在连续 RGB 上**不 scale** | HRL survey 2025 | 学习的 embedding 层是前提,裸像素/哈希不可行 |

### 1.4 现代机器:TCC → XIRL → GraphIRL(推荐采用线)

| 工作 | 提供什么 | 关键证据 |
|---|---|---|
| **TCC** (CVPR'19, [1904.07846](https://arxiv.org/abs/1904.07846)) | 逐帧"共性分数"现成机制:cycle-consistency(帧的软最近邻映射回自己 = 公共路径;误差大 = 稀有/绕路候选)。进度信号 Kendall τ **0.75 vs TCN 0.66**(from scratch) | 3-0 |
| **XIRL** (CoRL'21, [2106.03911](https://arxiv.org/abs/2106.03911)) | **端到端配方**:TCC 跨 episode 对齐(零标注)→ **value = 嵌入空间到 goal 帧的负距离**。明确消除"对单条参考轨迹对齐"(= 我们逐帧回归器的病)。代码开源 | 3-0 |
| **GraphIRL** (CoRL'22, [2207.14299](https://arxiv.org/abs/2207.14299)) | **治布颜色 nuisance**:先抽象掉外观(纹理)再在抽象空间对齐 → 对"同任务、外观多样视频"鲁棒。借**原则**(先抽象再对齐)不借实现(它是刚体物体图;布用分割 mask 形态/DINO 特征) | 3-0 |

**对现有 AWBC 的核心收益**:对齐相位 value **天然单调** → advantage = 相位推进速率,**结构上消除二阶差分崩塌**(0.896→0.3 那个),比省标注更有价值。

### 1.5 被否的捷径与覆盖缺口(诚实标注)

- ❌ 0-3 否决:"标 1 条参考 episode 经 TCC 传播 ≈ 50 条全标视频" — **别按此预算**。
- ❌ 1-2:betweenness 加速效果的"随机子目标对照归因" — 加速是真的,归因到"共性子目标"未坐实。
- **覆盖缺口**:VIP/LIV/R3M 视频 value 预训练、AWE waypoint、ILEED 示范加权三块**无幸存核验 claim**,本结论不依赖它们(自读时留意)。
- 所有实证来自刚体/仿真/gridworld,**无可变形双臂布操作先例**;"recurrence→自动milestone→AWBC标签"完整链**没人发表过** = 风险 + 可发表贡献点。

---

## 2. 失败模式与缓解(预注册)

| 失败模式 | 机理 | 缓解 |
|---|---|---|
| **多策略叠法** | 两种合法折法把 recurrence 劈成两半,各自都"不常见" | 先按整体轨迹嵌入聚类成"策略模式",**按模式分别对齐/挖掘** |
| **稀有恢复动作误杀** | regrasp/纠错低频但宝贵 | 软负 + 人工审计 bottom-decile 段(V0 必做项) |
| **布外观多样性**(白/蓝/米) | 视觉聚类被颜色/纹理主导 | GraphIRL 原则:先抽象(布分割 mask 形态描述子 / DINOv2 语义特征);固定相机视角是优势 |
| **TCC 单调相位假设** | 重复子动作/非单调顺序使对齐失真(LAV/GTCC 已证) | 先 V0 验证;必要时换 soft-DTW 类对齐或分段对齐 |

---

## 3. 方案

### 3.1 V0 探针(1-2 天,先证伪/证实再投入)— 脚本已建

`train_scripts/kai/data/recurrence_v0_probe.py`:

1. 抽 ~50 episode(默认 `A_new_smooth_800/base`,top_head 相机)× 3Hz 降采样;
2. 冻结 **DINOv2-small** 抽帧特征(L2 归一);
3. 全库 KMeans(k≈48)→ 每簇 **episode 覆盖率(first-visit)**;
4. 输出:
   - `coverage_curve.png`:簇覆盖率 vs 簇平均时间位置 → **看峰值是否对上直觉 milestone**(抓角/第一折/第二折/完成);
   - `milestone_clusters.png`:高覆盖簇的代表帧网格(肉眼判 milestone 语义);
   - `low_coverage_segments.md` + 缩略图:**bottom-decile 低覆盖段清单 → 人工审计:真错误还是 regrasp 恢复?**(决定假说后半生死);
   - `per_episode_timeline.png`:每 episode 时间线按所属簇覆盖率着色(低覆盖段一眼可见)。

**V0 判据**:
- 覆盖率峰对上直觉 milestone → 前半成立,进 V1;
- 低覆盖段多为恢复动作 → 后半只能做软降权(预期如此);多为真错误 → 可更激进;
- 簇被布颜色主导(同色聚一起而非同阶段聚一起)→ 先解决抽象层再谈对齐。

**V0 提交记录(2026-06-11)**: cnsh **Robot-GPU开发机队列**(`q-20251205141747-xlxlh`)2×A100,YAML `train_scripts/kai/volc/recurrence_v0_cnsh_2gpu.yaml`,**task `t-20260611215738-pvc4d`**。双 GPU 并行两份探针:GPU0=`A_new_smooth_800/base`(vis 部署域)→ `temp/recurrence_v0`;GPU1=`kai0_advantage`(**带 `stage_progress_gt` GT,可直接验证覆盖率峰 vs 真进度**)→ `temp/recurrence_v0_kai0`。日志 `logs/recurrence_v0_*.log`。
> 提交坑(记录):① 该队列**禁 Flexible 自定义资源**(API 报 "Customized resource spec is not allowed")→ 须用 Preset 小规格 `ml.pni2.7xlarge`(2×A100);② DINOv2 权重预缓存在 vePFS(`HF_HUB_CACHE=/vePFS/tim/workspce/hf_cache/hub_default`,注意 workspce 是历史 typo 路径)+ `HF_HUB_OFFLINE=1`,pod 无外网也能加载;③ 队列实时余量查法:`get_resource_queue` → QuotaCapability(12 A100)− QuotaAllocated(10)= 2 空闲。

### 3.2 V1 正式 pipeline(~2-3 周,V0 通过后)

| 步 | 内容 | 依据 |
|---|---|---|
| a | 全量 2-4Hz 降采样(1000ep×3cam ≈ 0.2-0.4M 帧,数 GPU 时) | — |
| b | 外观鲁棒特征:DINOv2 + 布分割 mask 形态 | GraphIRL 原则 |
| b' | **策略模式预聚类**(整体轨迹嵌入 → k-means),按模式分别处理 | §2 失败模式 1 |
| c | 改 XIRL 开源码训 TCC head(按相机) | XIRL |
| d | recurrence(帧)= 对 K≈20 条随机参考 ep 的软对齐一致性;first-visit 计数;Binomial 双阈值 | 教训 1/2 |
| e | **自动 milestone = recurrence 局部峰值** → 替代 ViVa-DSM 手标 | 教训 3 |
| f | **V(t) = milestone-index / 连续对齐相位**(单调)→ 喂回现有 `discretize_advantage.py`(AWBC 训练侧零改动) | XIRL |
| g | 低 recurrence 段 → 软降权 / "uncertain" 第三 prompt 标签 + 审计 | §1.2 |

### 3.3 评估(沿用项目铁律)

- offline:新 value 与 `stage_progress_gt` 的 corr(kai0_advantage 上可直接验,有 GT);差分后的 advantage corr 是否优于 0.3-0.4 基线;
- **真机为终判**:新标签训出的 AWBC vs 现 pi0-AE 标签版(同 init/同配方,单变量=标签来源,沿用 awbc_viva 对比框架 §1)。

---

## 5. 实验进展(自循环研究,2026-06-11 起)

### 5.1 V0 探针结果 — 假说前半在我们数据上初步成立 ✅

跑法:gf0 本地 A100(开发机队列 2×A100 部署测试通过,但暴露 `kai0/.venv` 的 python symlink 指向 `/home/tim/...` pod 内不存在 → **集群任务须自包含 venv**,task `t-20260611215738-pvc4d` 因此空跑;V0 改本地,后续集群训练前先修 venv)。

**(a) smooth800(vis 部署域,50 ep / 5968 帧 / k=48)— `temp/recurrence_v0/`**

- **覆盖率峰真实存在且时间局域化**:c4 **92%**@t=0.78、c36 78%@t=0.78、c23 72%@t=0.44(`coverage_curve.png`);
- **dom(单 episode 占比)全部 7-15%** → 高覆盖簇真跨 episode,非个体伪影;
- 代表帧网格(`milestone_clusters.png`)肉眼可辨任务阶段(摊开/伸臂/半折/已折),且**同簇混不同布色**(绿/橙/白同行)→ DINOv2 主要抓状态而非颜色(颜色主导担忧初步缓解,但低覆盖簇仍有 item 分型,见 (c))。

**(b) GT 验证(kai0_advantage 有 `stage_progress_gt`,50 ep)— `temp/recurrence_v0_kai0/gt_validation.png`**

| 零训练 value(纯重复度挖掘) | Kendall τ vs GT | Pearson r |
|---|---|---|
| **V_milestone**(top-10 覆盖簇按时序,首入即过) | **0.812(median 0.854)** | 0.805 |
| V_tpos(簇平均时间位置) | 0.553 | 0.641 |
| [对照] pi0-AE absolute_value(监督) | — | ≈0.896 |
| [对照] 线性时间(trivial 上界,GT 本身分段线性) | 1.000 | — |

→ **零标注逼近监督回归器**。诚实标注:episode 内与 GT 的高 τ 有"时间单调"成分;鉴别性在 (d)。已见局限:后段 milestone 稀疏 → V 饱和于 ~0.7(可加密后段 milestone 修)。

**(c) ⭐ 低覆盖段审计 — 假说后半的"第三类"占主导**

bottom-decile 簇(cov 4-10%)的段落缩略图人工抽看:**主要是稀有衣物类型(深色 T 恤、白长袖;smooth800 主流为方巾)**——既非 error 也非 recovery,而是 **item 级多样性**。
→ **"低重复=negative"被进一步证伪**:硬标 negative 会系统性惩罚稀有衣物品类。处理只能:软降权 + **按衣物/策略分组后再挖 recurrence**(= §2 预注册缓解的实证确认)。

**(d) ⭐ 鉴别性实验:demo 挖的 milestone → 真机 rollout(跨数据集,零训练)— `temp/recurrence_v0/rollout_value_compare.png`**

smooth800(demo)挖的 10 个 milestone,KMeans 质心直接 assign 到 autonomy rollout(7676 帧真机执行,含多次重试):
- **V_milestone(windowed)复现了与监督 pi0-AE 一致的多段重试结构**(~3000/~5000 帧处回落重爬),corr(V_ms, pi0-AE)=0.275、corr(pi0-AE, ViVa)=0.418;
- 说明 recurrence value 是**状态触发**的(时间线性信号不可能在重试处回落)→ 假说的核心机制(重复状态=任务锚点)在真机数据上可迁移。

**V0 判据结论**:前半 ✅ 进 V1;后半 → 软处理 + 分组(已实证);颜色主导 → 高覆盖簇 OK、低覆盖簇受 item 分型影响(分组可解)。

### 5.2 TCC 复现/适配(进行中)

- XIRL 官方 PyTorch 代码已拉到 `/vePFS/tim/workspace/recurrence_research/google-research/{xirl,tcc}`(sparse checkout);
- 适配器 `train_scripts/kai/data/tcc_train_features.py`:冻结 DINOv2 + MLP head,直接 import XIRL `compute_tcc_loss`(deterministic regression-MSE cycle-back),XIRL 协议 value=−‖emb−goal‖;
- 首训:kai0_advantage 120 train/20 val,2000 步,gf0 本地;评估 = val 上 TCC-value vs GT 的 τ(对照:原始 DINOv2 距离、pi0-AE)。结果待回填。

## 4. 参考文献 + 阅读指引(按优先级;1-7 经 3 票核验引用)

### 第一梯队(必读,~3 篇决定方案)

| # | 文献 | 链接 | 重点读什么 |
|---|---|---|---|
| 1 | Dwibedi et al., **TCC** — *Temporal Cycle-Consistency Learning*, CVPR 2019 | [arXiv:1904.07846](https://arxiv.org/abs/1904.07846) | §3 cycle-consistency 定义(帧的软最近邻能映射回自己 = 公共路径 → **就是"逐帧共性分数"**);Table 6(进度信号 Kendall τ 0.75 vs TCN 0.66, from-scratch;注意 ImageNet-finetune 下 TCN 反超、TCC+TCN 组合最佳 0.878 → **组合损失最稳**);**Fig.7 异常检测**(偏离典型轨迹=异常 —— "稀有=negative"的唯一定性先例,仅 1 例) |
| 2 | Zakka et al., **XIRL** — *Cross-embodiment Inverse RL*, CoRL 2021 | [arXiv:2106.03911](https://arxiv.org/abs/2106.03911) | 端到端配方:TCC 跨 episode 对齐(零标注)→ **value = 嵌入到 goal 帧的负距离**;重点读它如何"消除对单条参考轨迹的对齐"(= 我们逐帧回归器的病);goal 帧需指定(我们 episode 以完成结束,trivial);[代码开源 google-research/xirl](https://github.com/google-research/google-research/tree/master/xirl) |
| 3 | McGovern & Barto, *Automatic Discovery of Subgoals using Diverse Density*, ICML 2001 | [PDF](https://mcgovern-fagg.org/amy_html/old/pubs/mcgovern_barto_isairs2001.pdf) | 假说的原始形式化(bottleneck = 成功路径频繁经过的区域);**为什么裸频率不行**(every-visit 被停留时长主导 → first-visit);§6 作者自己的警告:有用子目标也出现在稀有路径 → **负证据要软化**(假说后半的脆弱性 25 年前就写明) |

### 第二梯队(实现前读)

| # | 文献 | 链接 | 重点读什么 |
|---|---|---|---|
| 4 | Kumar et al., **GraphIRL**, CoRL 2022 | [arXiv:2207.14299](https://arxiv.org/abs/2207.14299) | 治布颜色 nuisance:**先抽象掉外观(纹理)再在抽象空间做 TCC** → 对"同任务外观多样视频"鲁棒;实现是刚体物体图(布不可直接用)→ **借原则不借实现**(布分割 mask 形态 / DINO 特征);reward `r(o)=-1/c·‖ψ(o)−g‖²` |
| 5 | Şimşek, Wolfe & Barto, **L-Cut**, UMass TR 2004 | [PDF](http://all.cs.umass.edu/pubs/2004/simsek_wb_TECH04.pdf) | recurrence 的统计判定:重复采样下 hit 数服从 **Binomial** → 双阈值接受准则(出现数 > t_o 且 hit 比例 > t_p);"对付噪声的工具就是重复采样" |
| 6 | Şimşek & Barto, *Skill Characterization Based on Betweenness*, NeurIPS 2008 | [PDF](https://proceedings.neurips.cc/paper/2008/file/934815ad542a4a7c5e8a2dfa04fea9f5-Paper.pdf) | milestone = betweenness 的**局部极大**(相对一跳邻域),非全局阈值;注意 Rooms 域峰值在门口**两侧**而非门口本身 → 选峰按邻域比较 |

### 第三梯队(背景/对照,可跳读)

| # | 文献 | 链接 | 价值 |
|---|---|---|---|
| 7 | Klissarov, Bagaria, Konidaris, Precup, Machado et al., *HRL Survey*, 2025 | [arXiv:2506.14045](https://arxiv.org/abs/2506.14045) | subgoal 发现全景;明示"离散图方法在连续 MDP 不 scale" → embedding 层是前提 |
| 8 | **VIP** / **LIV** | [arXiv:2210.00030](https://arxiv.org/abs/2210.00030) / [arXiv:2306.00958](https://arxiv.org/abs/2306.00958) | 视频 value 预训练对照线(⚠️ 本轮核验无幸存 claim,自读时留意) |
| 9 | **AWE** / Keyframe-Focused IL | [arXiv:2307.14326](https://arxiv.org/abs/2307.14326) / [arXiv:2106.06452](https://arxiv.org/abs/2106.06452) | waypoint 自动抽取(几何误差驱动,非跨 episode)/ 升权关键帧而非删帧(与 idle 调研结论同源) |
| 10 | TCC 后续:LAV / GTCC | [arXiv:2103.17260](https://arxiv.org/abs/2103.17260) / GTCC CVPR 2024 | TCC 单调相位假设的已知局限(重复子动作/非单调/多策略)→ §2 失败模式 4 的依据与备选 |

> ⚠️ **读时记住两条被 3 票否决的捷径**:① "标 1 条参考 episode 经 TCC 传播 ≈ 50 条全标视频"(0-3,**别按此预算**);② TCC 异常检测从未被当 negative 标签用于 weighted BC —— 是空白也是贡献点。

**开放问题(也是贡献点)**
- 有没有任何已发表工作把对齐误差当 negative/降权标签用于 weighted BC?(本轮调研:没有 → 空白)
- 我们数据里有几个策略模式?低覆盖段错误 vs 恢复的真实占比?(V0 回答)
