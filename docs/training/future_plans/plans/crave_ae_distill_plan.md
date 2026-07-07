# 从数据学 Advantage:CRAVE 监督 → 蒸馏 → ranking-learning(分阶段 + 决策点)

> **目标**:从操作数据(demo + rollout + DAgger)学一个 advantage 信号,卡阈值二值化喂 advantage-conditioned VLA。**不用人工 stage 标注**(不 scale)。CRAVE 提供**零人工的语义进度监督**。
> **本轮范围**:只到"训出 value/advantage 模型 + 离线看效果",**暂不做 VLA/真机**。
> 关联:[awbc_milestone_value_AB_plan](../../../../crave/docs/awbc_milestone_value_AB_plan.md) 的 B 臂。日期:2026-07-04。

## 0. 定位:两篇参考 + 我们的路线

| | value 来源 | advantage |
|---|---|---|
| **π*₀.₆ (RECAP)** | 学分布式 value(MC 回报,201-bin CE) | `A=Σr+V(s')−V(s)`(做差,叠两噪声) |
| **χ₀ (Stage Adv)** | 直接回归 `A=f(s,s')`(成对帧),+ 人工 stage 消歧 | 单次前向,不做差 |
| **我们** | **CRAVE 零训练逐帧 value 作监督** → 蒸/学一个 g(s) | `A=g(s')−g(s)`,g 全局排序更干净 |

两篇下游一致:连续 advantage → 卡阈值 → 二元 indicator `I=1[A>ε]`。差别只在 advantage 怎么来。**我们的关键差异化 = 用 CRAVE 顶掉 χ₀ 的人工 stage / RECAP 的 MC 回报**,零人工。

---

## Phase 0 ✅ 已完成:两套 CRAVE 标签 + 两数据集

两者都在 **DINOv3-H milestones(全 3055 ep)** 上,3Hz 生成 → 插值 30Hz → 端点归一 0→1(kai0_base 全成功折)。

| | **A · anchor-linear** | **B · viterbi + 时间先验** |
|---|---|---|
| 构造 | 段内**离簇心最相似帧=锚**,value=Pord;+start0/end1;isotonic;线性连 | milestone bin 上 Viterbi-DP,emit=(1−sim)+**α·\|tn−Pord\|**;value=Pord[path];中值平滑 |
| 形状 | 少而大的台阶 | 细阶梯(处理 dwell/loop) |

**教训(sanity 已验证)**:naive Viterbi 崩(起末别名 ep763/1527 corr 0.07/−0.80)→ 加时间先验 α=0.6 修到 0.95/0.93,全量鲁棒。A/B 均 mono 1.00 / corr 0.86–0.96 vs 人工;A vs B mean|Δ|≈0.20、corr 0.65(有区分度)。

![两种标签 vs 人工](../../../../crave/docs/visualization/ae_distill/stage_label_compare.png)

**数据集(训练就绪,统一以 `kai0_base` 为底)**:`kai0/data/Task_A/self_built/crave_stage_{A,B}/`(kai0_base 原列 + `stage_progress_gt`=CRAVE,0→1)。脚本 `crave/experiments/gen_ae_stage_labels.py --full` + `write_crave_stage_datasets.py`。

**鲁棒性已验(全 3055 ep)**:`crave_value_robustness.py` —— corr(CRAVE-viterbi, 人工) 均值 **0.957**、**100% ep ≥0.7**(viterbi 100% ≥0.8);最差 6 个 ep 仍 0.89–0.90、单调干净、无塌缩/别名/非单调。24 随机 ep 画廊 + 最差 6 见 `visualization/ae_distill/crave_value_robustness_{gallery,summary}.png`。→ 两套标签作 AE 监督在全量上稳。

---

## Phase 1(= **CRAVE_a 任务**,数据就绪待集群):回归蒸馏 AE-A/B vs AE-C

- **配置** `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`(pi0-AE `value_head`),Step-1 `train_pytorch.py`,50k;唯一变量=标签。
- **三臂**:AE-A(crave_stage_A)· AE-B(crave_stage_B)· **AE-C(现成 baseline,人工标签 `adv_est_v1/100000`)**。
- 走集群(`submit-training-job`)。

**离线判据(不用 circular MAE)** —— held-out 成功 ep 上比 AE 打出的 `absolute_value`/`relative_advantage`:

| 指标 | 期望 |
|---|---|
| P/N 翻转次数 / NEG 帧占比(痛点①抖动) | 越低越好(AE-C 有 256 次翻转) |
| 单调率 mono | 越高越好 |
| relative_advantage 噪声(std / 过零率) | 越低越好 |
| 完成态 value | 接近 1 |

---

## ⭐ 决策点 D1(等 **CRAVE_a** 出结果 → 决定 Phase 2 怎么走)

| 结果 | 判断 | 下一步 |
|---|---|---|
| **AE-A 或 AE-B 明显 > AE-C**(P/N 更干净 / advantage 更稳) | CRAVE 是**好监督** | ✅ 进 Phase 2;**用胜出的标签(A 或 B)作 ranking 目标** |
| **A ≈ B ≈ C** | CRAVE 标签没帮上 AE | 诊断:标签质量 or AE 容量?先修再谈 Phase 2 |
| **A/B 谁更好** | 决定 Phase 2 的 ranking 目标形状 | 少台阶(A) vs 细阶梯(B)哪个更利于学 |

> D1 是"回归蒸馏够不够 / 哪套标签更好"的 gate。**Phase 2 的 ranking 升级正是为了在 D1 若发现"回归有短板"时快速接上**(见下"为什么还要 Phase 2")。

---

## Phase 2(ranking-learning 升级,**D1 后可快速推进**)

**为什么还要 Phase 2(回归蒸馏的三个短板)**:① 回归 MSE 会把 CRAVE 的**幅度错误**(完成态偏弱/平台)也学进去;② CRAVE 标签有 **milestone 内平台 → 段内 A≡0**(整段动作无梯度);③ 特征锁死。ranking 用**序**、连续可微、可放开 backbone,逐一解。

**核心洞察(为何这样设计)**:
- **监督 = CRAVE-value 的序,不是 raw-time,也不是 MSE 值**。raw-time 是弱监督(进度≠线性时间);且冻结档 rank-on-time 只是"头分不开同特征帧→平均"侥幸复现 cluster,一放开就学成**时钟 shortcut**。改用 **CRAVE 语义 value 的序**做监督 → 放开 backbone 学的是**任务进度**不是时钟。**ranking(序)对 CRAVE 幅度错误免疫**(只要序对)。

**网络 / 损失**:
```
g_φ(s) = sigmoid(head(DINOv3-H(s)))                       # 0–1 标量进度, head=小 MLP
L_rank = E_{i,j} max(0, m_ij − (g_φ(s_i)−g_φ(s_j))·sign(y_i−y_j))
   y   = 胜出的 CRAVE 标签(A 或 B, D1 定);  m_ij ∝ |y_i − y_j|
```

**四个要素**:
1. **CRAVE-value 监督**:`y` = Phase-1 胜出标签的 value(替代 raw-time)。
2. **ranking 损失**:margin ∝ |Δy|;采对**先 within-ep**(时序序可靠)→ 再 cross-ep(靠 CRAVE 语义 value 对齐,非时间)。
3. **冻结/放开双档**:
   - **冻结档**:frozen DINOv3-H + head → **应 match Phase-1 CRAVE 曲线**(match 不上=损失/采样 bug,是调试锚)。
   - **放开档**:渐进解冻末几 block + 小 lr + **DINO-anchor 正则** `μ‖φ(s)−φ_DINO(s)‖²`(μ 大→小),防时钟/塌缩 shortcut;因监督是语义 value,放开学任务特化进度。
4. **变点加权**:margin 在 **milestone 转换 / 进度突变处**加大(advantage 信息集中处;接"milestone=瓶颈+变点")→ 决策点信号更强、平台段不浪费。
5. **读 advantage**:`A=g_φ(s_{t+N})−g_φ(s_t)`(N=50,RECAP)→ 卡全局分位阈值二值化。

**⚠️ 天花板提醒(洞察 C)**:per-frame g_φ(冻结/放开都)是**帧的确定性函数** → 视觉相同、相位不同的帧**消不了歧**(和 cluster 一样)。真消歧需时序上下文。**可选路径项**:rank g_φ 拟合 **Viterbi 校正值**(Method B,带路径消歧的序)而非原始进度 → 既拿连续可微、又拿路径消歧。留作 Phase 2b。

### 🔧 冻结档骨架已搭好(D1 一过即可训)
`crave/experiments/train_gphi_ranking.py`(冻结档:直接吃缓存 DINOv3-H 特征 = frozen backbone,只训 head;~1 分钟/40ep)。已实跑验证:
- **sanity 锚成立**:g_φ match CRAVE-viterbi 曲线,held-out **per-ep corr 均值 0.94**、val 逐对 ranking acc **0.91** → 损失/采样正确(图 `visualization/ae_distill/gphi_viterbi_sanity.png`)。
- **发现(修正你 doc §5 的"ranking 隐式平滑"说法)**:frozen per-frame head **只跟特征一样抖**(mono ~0.55),**ranking 只约束序、不约束平滑**。→ 已加 **TV 时序平滑项**(同 ep 相邻帧 `(Δg)²`,不回归 Δy 免重现平台 A≡0,`--smooth` 旋钮);**usable advantage 还需**:调大 λ_s 或**读出时 median/EMA 平滑 g_φ**(= CRAVE 做法)。这条直接进 Phase-2 待办。
- 待接:D1 选出胜出标签(`--label anchor|viterbi`)、放开档(解冻+DINO-anchor)、变点加权、读 advantage。

---

## ⭐ 决策点 D2(Phase 2 后)

| 检查 | go | no-go |
|---|---|---|
| 冻结档 g_φ 是否 match Phase-1 CRAVE 曲线 | 是→损失/采样对 | 否→回查 bug,别往下 |
| 放开档 是否比冻结档**更 smooth + advantage 更可分** | 是→保留放开(突破 DINO 天花板) | 否→**退回冻结**(别默认放开更好) |
| ranking-advantage vs Phase-1 回归-AE-advantage | ranking 信噪比高 + 段内不丢信号→采 ranking | 否→回归够用则不折腾 |
| 变点加权 是否提升 advantage 判别力 | 是→保留 | 否→去掉 |

---

## Sanity 三件套(贯穿 Phase 1/2,每次训完必看)

1. **成功/失败曲线**(类比 RECAP Fig 4):成功 ep 的 g/value **单调升**,失败 ep **平/塌**。(kai0_base 全成功 → 需另找/造失败 ep,用 §truncate/倒放法。)
2. **同 score 段跨 ep 方差**:跨多条 ep 采同 score 段的帧,看 g_φ 方差 → **小 = 跨 ep 对齐成功**(下游全局阈值 ε 可用的前提)。
3. **盯 advantage 不只盯 value**:最终看 **advantage(g 做差)的判别力 + 信噪比**,不是只看 value 曲线平滑度("value 漂亮 ≠ advantage 可用")。

---

## 里程碑 + 状态

| Phase | 内容 | 状态 | gate |
|---|---|---|---|
| **0** | 两套 CRAVE 标签 + 两数据集 | ✅ 完成 | — |
| **1 (CRAVE_a)** | 集群训 AE-A/B vs AE-C(回归蒸馏)+ 离线判据 | 📋 数据就绪, 待集群 | → **D1** |
| **2** | ranking g_φ(CRAVE-value 序 + 冻结/放开 + 变点加权)+ 读 advantage | 🔜 **D1 后快速推进**(架构/损失已定,见上,可直接实现) | → **D2** |
| 2b | 可选:rank 拟合 Viterbi 校正值(路径消歧) | 视 D2 | — |
| 3 | 下游 advantage-conditioned VLA / AWBC + rollout | 后续(本轮不做) | AB_plan Tier3 |

**推进逻辑**:CRAVE_a(Phase 1)一出结果 → D1 选出胜出标签 + 判断回归是否够 → **立即启动 Phase 2**(g_φ+ranking,监督换成该标签的序)。Phase 2 冻结档先 match、再受控放开,D2 决定是否采 ranking-advantage 上下游。

---

## 诚实边界 / 风险
- **天花板 = CRAVE 监督质量**:g_φ 继承 CRAVE 的"完成态偏弱 + 只抓粗失败"。ranking 缓解幅度错误,但**多值歧义(per-frame 天花板)消不掉**——需时序上下文(Phase 2b 路径项 / proprio / 历史)。
- **放开 shortcut**:rank-on-CRAVE-value 比 rank-on-time 安全,但仍要**盯视觉依赖**(借 XVLA proprio-捷径/vision-blind 教训),放开不达标就退冻结。
- **circular 判据**:AE/g 各拟合自己目标,别用"对谁的 MAE"评;用 P/N 干净度 / 单调 / advantage 信噪比 / 跨 ep 方差。
- **端点归一 0→1** 假设 ep 完整完成(kai0_base 成立);partial/failed 需 `de_end` 门控,不盲目拉 1.0。
