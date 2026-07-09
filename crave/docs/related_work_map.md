# VLA 后训练三轴谱系图 + CRAVE 定位(related work map)

> **建立**: 2026-07-04
> **目的**: 把 CRAVE 相关的 ~15 篇按**三条正交轴**归位,一图看清各方法给的是"多好 / 未来长啥样 / 该怎么动",并标出 **CRAVE 的护城河 + 未被占的空位**。CRAVE 投稿 positioning 直接引这页。
> **配套**: [positioning_and_roadmap](CRAVE_positioning_and_roadmap.md) · [方法对比](value_advantage_methods_comparison.md) · [CRAVE-RPO](crave_rpo_minimal_validation_plan.md)

---

## 0. 三条主轴 + 一条交付轴(先分清"问什么")

```
① 价值/进度轴 (VALUE)         问:"这一步/这条轨迹有多好?到哪了?"  → 打分,用于加权/RL
② 世界模型轴 (WORLD MODEL)    问:"下一步场景会变成什么样?"        → 预测未来观测,用于 condition 动作
③ 中间表示/子目标轴 (SUBGOAL) 问:"该往哪去/怎么动?"               → 指定目标/路径,用于 condition 动作 + 跨任务泛化
④ 交付/改进算子轴 (OPERATOR)  问:"拿到 value 后怎么喂给策略?"      → CFG引导/RL微调/加权BC(①的下游)

②③ 都是"conditioning"(喂给策略),①是"scoring"(评判策略),④是"①→策略"的用法。四者正交、可叠。
```

---

## ① 价值 / 进度 / advantage 轴

| 方法 | 来源 | value 怎么来 | 训练? | 需要什么 | 输出 |
|---|---|---|---|---|---|
| **VIP** (2210.00030) | Meta/UPenn ICLR23 | 嵌入距离=value(对偶RL) | 训练(预训练) | 无动作人类视频 | 逐帧标量,图目标 |
| **LIV** (2306.00958) | UPenn/Meta ICML23 | VIP+CLIP,图/文目标 | 训练 | 视频+caption | 逐帧标量 |
| **RECAP / π\*0.6** | Physical Intelligence 2025 | **学 value 函数**(从经验 outcome,全局 credit assignment) | 训练+**在线RL练习** | demo+纠错+自主rollout成败 | 连续adv→**advantage-conditioned** diffusion policy |
| **χ0 / KAI0** (2602.09021) | HKU MMLab | **stage 条件直接预测 advantage**→二值 | 训练 | **人工 stage 标签** | 逐帧→二值→**AWBC 加权** |
| **SARM2** (2606.10305) | Stanford/Berkeley | stage估计(22基元)+**MoE value** | 训练 | **基元/阶段标签(200h标)** | 逐步标量+SPIRAL |
| **SVM** (2606.23640) | Berkeley (Levine) | **判别器** 成功vs失败(s,a),logit奖励 | 训练 | **需成功+失败** | 逐(s,a),+RLPD,有策略不变性证明 |
| **STEAM** (2606.29834) | Tsinghua 等 | 自监督**时间offset ensemble**,min保守,反向对露倒退 | 训练(label-free) | **仅专家demo(无需失败)** | 逐帧adv+CFGRL |
| **SRPO** (2511.15605) | — | V-JEPA 到**成功簇距离** | **免训**(冻结编码器+聚类) | 需成功rollout(自参考) | 轨迹级+GRPO/AWR |
| pi0-AE(你) | deepdive_kai0 | 监督AE(χ0 谱系) | 训练 | stage_progress_gt | 逐帧标量→AWBC |
| **★ CRAVE(你)** | deepdive_kai0 | **跨episode重复→milestone→DP value** | **零训练(零梯度)** | **仅demo,零标签,无需失败** | **逐帧value + 离散milestone结构** |

**谱系速记**:RECAP=学value(能超示教,方差大)→ χ0/SARM2=stage监督(稳,要stage标签)→ SVM=判别器(要失败)→ STEAM=自监督时间offset(label-free但要训)→ **CRAVE=零训练+跨episode重复+离散milestone**。

---

## ② 世界模型轴(预测未来观测)

| 方法 | 来源 | 表示空间 | WM 输入 → 输出 | 推理时 WM |
|---|---|---|---|---|
| 普通 WM(imagine-then-execute) | LingBot-VA/Cosmos/Motus | **像素** | obs+lang → 未来RGB帧(迭代) | 真生成未来(慢,数秒) |
| **Fast-WAM** (2603.16666) | Tsinghua IIIS | 像素 **VAE latent** | (训)未来视频latent;(推)只编码当前 | **砍掉未来**,WM=encoder;190ms |
| **LaWAM / LaWM** (2606.15768) | Tsinghua 系 | **冻结 DINOv3 特征** | u + **潜在动作z** → 未来DINOv3特征子目标 û_T | **真预测**(一次前向),230M,û_T condition动作 |
| **WorldVLA** (2506.21539) | 阿里 | 离散 token | 图+动作共享词表,AR 同时生成 | 统一AR |
| **UVA** (2503.00200) | Shuran Song | 联合latent | video+action 联合,双扩散头 | 推理bypass video |
| **★ Ctrl-World** (2510.10125) | Stanford+清华 ICLR26 | **像素·多视角(含腕)视频** | 动作块 → 多视角未来帧(policy-in-loop,自回归) | **真生成**;pose记忆检索>20s一致;DROID 95k |
| 你的 **LMWM / LMWAM v2** | deepdive_kai0 | DINOv3 latent | freeze LMWM 当 WM provider 进 pi0.5 | (见下 A 路线) |

**世界模型的两种用法(关键,别混)**:
- **(A) 当表示 / 子目标提供者** — 预测未来特征喂进策略,改**策略输入/feature**,便宜。Fast-WAM(只在训练塑表示,推理丢未来)· LaWM(便宜 DINOv3 latent 里真预测 + 潜在动作瓶颈,吃无动作人类视频)· **你的 LMWM 在此**。
- **(B) 当仿真器 / 评测器 / 数据引擎** — 让真策略在想象里 rollout,做**评测 + 造 SFT 数据**,贵。**Ctrl-World**:像素多视角视频 WM + pose 记忆检索 → **免真机排序策略** + **想象合成成功轨迹拿去 SFT(成功率 +44.7%)**;代价是视频生成重。
- **②轴无人做 value/progress** → **CRAVE 可当 (B) 路线想象 rollout 的评判器**(筛哪些想象轨迹算成功、该拿去 SFT),= 新空位。

---

## ③ 中间表示 / 子目标轴(该怎么动)

```
语言        RT-Affordance         RT-Trajectory          goal image
(太少)  →  关键阶段夹爪位姿(稀疏) → 完整2D运动路径(较密) →  (太多/含外观,不鲁棒)
under-spec        轻                    中                    over-spec
```
| 方法 | 来源 | 表示 |
|---|---|---|
| **RT-Trajectory** (2311.01977) | GDM ICLR24 Spotlight | **2D 轨迹草图**(hindsight 提夹爪路径;画/视频/自动指定)→ 跨任务泛化 |
| **RT-Affordance** (2411.02704) | GDM | **关键阶段夹爪位姿**(pixel xy叠图);cheap affordance图学新任务免采轨迹 |
| **Genima** (2407.07875) | — | 动作画成 RGB 图,微调 Stable Diffusion |
| LaWAM latent subgoal û_T | (见②) | DINOv3 特征子目标(跨 ②③ 轴) |

**③轴也无人做 value/progress**;都是"该往哪去"的 conditioning,靠它做跨任务泛化,且**可从 demo 免费 hindsight 提取**。

---

## ④ 交付 / 改进算子轴(拿到 value 后,怎么喂给策略)

> ①**算出** value 只是一半;这条轴管**另一半:怎么用 value 去改进策略**。三小类,底层多是 **CFG(classifier-free guidance)** 的"正减负"方向 `ε(x,c⁺)−ε(x,c⁻)`——推理期外推,或训练期烧进权重。**三类都要"正/负 + reward",而 CRAVE 天然能供。**

| 小类 | 方法 | 机制 | value 怎么进 | 重训? |
|---|---|---|---|---|
| **A. 推理期引导** | **CFGRL** (2505.23458) | CFG **正向**,`w`=advantage 逆温度 → `π∝π_BC·exp(w·A)` | c⁺ = "最优"(高 advantage) | **否**(纯采样) |
| | **REACH** (CVPR26) | CFG **负向** prompt;错误检测器产 OOD 历史 | c⁻ = OOD/失败 → **recovery** | **否** |
| **B. 训练期 RL 微调** | **FlowGRPO** | 离散化**反向**采样 → GRPO 策略梯度 | reward 排序 | 是(需似然+CFG+特定 solver) |
| | **★ DiffusionNFT** (NVIDIA Cosmos) | **前向** flow-matching,正负对比→隐式改进方向烧进权重 | reward 分正/负样本 | 是(**免似然/免CFG/black-box solver,25× 快**) |
| | DDPO | 把去噪当 MDP,策略梯度 | reward | 是 |
| **C. 加权 BC / adv-conditioned** | **AWBC**(你在用) | advantage 加权 BC 损失 | 逐样本 advantage | 是 |
| | **RECAP / π\*0.6** | advantage-conditioned diffusion policy | 连续 advantage 当条件 | 是 |
| | pi0-AE(你) | 监督 AE advantage → AWBC | stage_progress_gt | 是 |

**CFG 背景(30 秒)**:同一网络训出有条件 `ε(x,c)` 与无条件 `ε(x,∅)`,采样外推 `ε̃ = ε(x,∅) + w·[ε(x,c)−ε(x,∅)]`;`w>1` 超额服从 c,等价从 `p(x)·p(c|x)^w` 采样。负向 prompt 把 `∅` 换成 c⁻ → 推离 c⁻。**A 在推理期用它;B 的 DiffusionNFT 把同一"正减负"当训练梯度,所以训完 CFG-free。**

**🎯 CRAVE 接口(把 ① 连到 ④)**:CRAVE 天然产三样——高 value 帧 = **c⁺**、脱离 milestone 流形的低 value/OOD 帧 = **c⁻**、milestone value = **reward**。于是:
- **最省**:CRAVE → **CFGRL 正向 + REACH 负向**(推理期,零策略重训);
- **更强**:CRAVE → **DiffusionNFT**(前向 flow-matching 微调 pi0.5,免似然/免 CFG,与你 flow 骨架天然兼容);
- 你缺的 **负数据**(遥操 neg 仅 5.1%)由 **CRAVE 低 value/OOD 帧充当**,喂 REACH / DiffusionNFT 的负样本。→ ④ 与 ① 闭环。

---

## 附:其他(action 表示 / VLA 骨干,非三轴)

| 方法 | 是什么 |
|---|---|
| **FAST / π0-FAST** (2501.09747) | 动作 DCT+量化+BPE → 离散**频域 token**(自回归VLA) |
| **FreqPolicy** (2506.01583) | 连续**频域 token** + coarse-to-fine AR + 扩散头(NeurIPS25) |
| **VLANeXt** (2602.18532) | VLA 12 条设计配方(soft connection / DCT loss / proprio 等,ICML26) |

---

## ★ CRAVE 的护城河 + 空位(投稿核心)

### 护城河(价值轴上 CRAVE 独占的一格)
1. **零训练/零梯度**:SARM2/χ0/RECAP/SVM/STEAM 全都要训一个东西;CRAVE frozen DINOv2+KMeans+DP。
2. **零标签、无需失败数据**:χ0/SARM2 要 stage/基元标签;SVM 要失败;CRAVE 只要 demo(治你遥操 neg 仅 5.1% 的洞)。
3. **离散 milestone / 技能结构**:别人多给标量;CRAVE 给可检视的离散阶段图(positioning §0 独占资产)。
4. **跨 episode 重复**当信号:STEAM=轨迹内时间offset、SRPO=成功簇、RECAP=学value —— 信号源都不同。

### 该借的(不丢护城河的前提下)
- **STEAM**:ensemble-min 保守打分 + 反向对露倒退 → 进 CRAVE-RPO 治 OOD 假高 advantage。
- **SVM**:策略不变性证明 → 论证 CRAVE 当 shaping 合法。
- **SARM2**:按 stage 路由 MoE value → CRAVE milestone 天然可路由。
- **χ0/RECAP**:AWBC/advantage-conditioned 交付(你已在用)。

### 🎯 跨轴空位(最大的没人占的位)
> **没有一篇在同一个(DINOv3)latent 空间里同时做**:①价值/milestone结构(CRAVE)× ②前向预测未来(LaWM)× ③子目标 conditioning(RT-Traj/Aff)。
> - LaWM 在 DINOv3 做②但没①;CRAVE 在 DINOv3 做①但没②;RT-Traj/Aff 的子目标可由 **CRAVE milestone 自动生成**(免标注)。
> - **"CRAVE 定关键阶段(①)+ LaWM 预测该阶段视觉(②)+ 从 milestone 自动出轨迹草图/affordance 子目标(③)"** = 一个三轴统一的潜在世界-价值-子目标模型,且三块的基础你都已有(DINOv3 环境 + CRAVE 包)。

---

## 一句话
**CRAVE 稳占价值轴的"零训练 + 离散 milestone + 跨 episode + 无标签无失败"这一格;真正的蓝海是把三轴在 DINOv3 latent 里缝起来(CRAVE×LaWM×RT-Traj/Aff),那是目前文献的空位。**
