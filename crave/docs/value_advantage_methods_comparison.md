# Value/Advantage 计算方法论证与对比:kai0-AE vs π*0.6-RECAP vs CRAVE

> 三种"给 episode 每帧打 value/advantage"的方法,基于**实际代码 + 原论文**逐条论证,再分场景比较优劣。
> 日期 2026-06-16。来源:kai0 代码(本仓库)· π*0.6/RECAP 论文 [arXiv:2511.14759](https://arxiv.org/abs/2511.14759)· 离线 RL 文献(CRR/AWAC/RCBC)。

---

## 1. kai0 AWBC AdvantageEstimator(本仓库实现)

**架构**(`kai0/src/openpi/models_pytorch/pi0_pytorch.py:464-481`):π0.5 主干(`pi05=True`,从 `pi05_base` 初始化)+ 一个 **value head** = 3 层 MLP(width→width→width→1, SiLU)+ **Tanh**,读 state token 末层表示 `suffix_out[:,0,:]`。标量输出 ∈ [-1,1]。

**训练目标**(`advantage_dataset.py:131-135`):**配对 stage-progress 差**。每个样本随机取同 episode 另一帧作"历史帧",回归
```
progress = stage_progress_gt[当前帧] − stage_progress_gt[随机历史帧]   (clamp[-1,1], MSE)
```
→ **本质是有监督的"进度差回归器"**,标签来自**人工标的 stage_progress_gt**(非 RL 回报)。value head 不看 action(`pi0_pytorch.py:608` 注释 "Not using action advantage")→ 学的是 **V(o),不是 Q(o,a)**。

**三个量怎么来**(`stage_advantage/eval_adv_est.py:236-301`,靠"换输入"复用同一 head):
| 量 | 喂给 value head 的(历史帧, 当前帧) | 含义 |
|---|---|---|
| `absolute_value` | (初始帧, 第 n 帧) | 0→n 的绝对进度 |
| `relative_advantage` | (第 n 帧, 未来 n+Δ 帧) | n→n+Δ 窗口内做出的进度 |
| `absolute_advantage` | `absolute_value[n+Δ] − absolute_value[n]`(事后差分) | AWBC 默认用它 |

**下游离散化**(`discretize_advantage.py`):把 `absolute_advantage` 按阈值/分位二值化 → `task_index` → prompt("Advantage: positive/negative")喂 AWBC 策略。

> 一句话:kai0-AE = **用人工进度标签监督出的 V(o);advantage = V 的时间差分**;失败信号弱、scalar 噪声大;离散化只发生在最后喂 prompt 时。

---

## 2. π*0.6 / RECAP(Physical Intelligence,[arXiv:2511.14759](https://arxiv.org/abs/2511.14759))

RECAP = RL with Experience & Corrections via Advantage-conditioned Policies。三阶段:示教(imitation)→ 真人遥操纠错(coaching)→ **自主练习数千次 + RL 打分**(practice)。

**Value 函数**:**分布式** V(o,ℓ),预测**离散化的蒙特卡洛回报**(到完成的负步数),**B=201 个 bin,交叉熵**:
```
min_ϕ  E_τ [ Σ_t  H( R_t^B(τ),  p_ϕ(V | o_t, ℓ) ) ]
```
是 **Monte-Carlo(整条轨迹回报)**,非 TD。backbone = 670M Gemma-3 VLM(比策略小),还掺少量网图防过拟合。

**奖励**(稀疏、含失败惩罚):
```
r_t = 0 (成功终止) ;  −C_fail (失败终止) ;  −1 (其余每步)
```
→ value 归一到 (−1,0),**天然区分成功/失败轨迹**。

**Advantage**(真 n-step 优势,N=50):
```
A^π(o_t,a_t) = Σ_{t'=t}^{t+N-1} r_{t'} + V^π(o_{t+N}) − V^π(o_t)
```
= n 步回报 + bootstrap V − baseline V。

**Advantage 条件化**(二值指示 + 无分类器引导):
```
I_t = 1( A^πref(o_t,a_t,ℓ) > ε_ℓ )      → token "Advantage: positive/negative"
min_θ E[ −log π_θ(a|o,ℓ) − α log π_θ(a|I_t,o,ℓ) ]   (30% dropout I_t → CFG)
```
真人纠错段**强制 I=True**(无视算出的 advantage)。

> 一句话:RECAP = **从自主练习的真实结果(含失败)学分布式 MC 回报 V;advantage = 真 n-step 优势;二值条件 + CFG**。是把"RL 回到真机"做成可扩展的优势条件 BC。

---

## 3. CRAVE(本工作,零训练离散 milestone-value)

跨 episode 重复态 = milestone → Viterbi-DP 在离散 bin(NB=21)上读出 0→1 单调阶梯 value。**零训练、零标签**(frozen DINOv2 + KMeans + DP)。advantage = value 的窗口差分(同 kai0 的 absolute_advantage 定义)。详见 [METHOD](cross_episode_recurrence_value_METHOD.md)。

---

## 4. 逐维对比

| 维度 | kai0-AE(监督) | π*0.6-RECAP(RL) | CRAVE(零训练) |
|---|---|---|---|
| value 信号来源 | **人工 stage_progress 标签** | **自主练习的 MC 回报(含失败)** | **跨 episode 重复结构(无标签)** |
| value 目标/损失 | 配对进度差,MSE,scalar+Tanh | 分布式 201-bin,交叉熵 | DP 在离散 bin 上的阶梯 |
| 是否 Q(o,a) | 否(V,不看 action) | **是**(A 对 action,经结果体现) | 否(V) |
| advantage 定义 | V(t+Δ)−V(t)(纯差分) | **Σr + V(t+N)−V(t)**(真优势) | V(t+Δ)−V(t)(纯差分) |
| 失败/退步敏感 | 弱(标签多在成功/dagger,scalar 噪声 32-47% 误负) | **强**(−C_fail 显式区分) | 中(结构性退步,稀疏对齐真事件 7-8%) |
| 离散化角色 | 仅最后喂 prompt 二值化 | **value 分布式离散(201bin)→ 校准** + 条件二值化 | **value 本体离散(阶梯)→ 去噪/鲁棒** |
| 校准/平滑 | 差(scalar 点估计,OOD 欠读压缩到 0.27) | **最好**(分布式,带不确定性) | 好(单调阶梯,in-dist 配 AE,OOD 更稳) |
| 能否超越示教 | 否(只拟合人标进度) | **能**(从自主结果改进,closes imitation gap) | 否(只描 demo 流形进度) |
| 成本 | 中(~1 周人工标 stage_progress + 训 value) | **高**(数千次真机自主练习 + RL + 大算力,迭代) | **极低**(零训练零标签,frozen 特征+聚类) |
| 在线可部署 | 是(学得的模型) | 是 | 需 demo cache+聚类(也可在线套用) |

**文献定位**:RECAP 的"二值优势条件 + CFG"是 **reward/advantage-conditioned BC** 一脉(RCBC、Decision-Transformer 回报条件)的鲁棒化——经典 reward-conditioned BC 在**条件到 OOD 高回报时会崩**([arXiv:2210.05158](https://arxiv.org/abs/2210.05158));用二值指示 + CFG 规避了这个失效。kai0/CRAVE 走的是 **advantage-weighted/filtered BC**(CRR/AWAC,[arXiv:2110.04698](https://arxiv.org/abs/2110.04698))那一脉:先算 advantage 再筛/加权数据。开源参照:Physical Intelligence **openpi**(π0/π0.5 base;RECAP 本身**未开源**,kai0-AE 是基于 openpi 的本地复现+人标版)。

---

## 5. 哪种更好——分场景结论(不存在单一最优)

**① 目标 = 真正突破示教上限、从经验自我改进** → **RECAP 完胜**。只有它把"真机自主练习的成败结果"灌进 value,advantage 是真 A(o,a),能让策略做出比示教者更快/更稳的动作(论文报告吞吐翻倍)。代价是最重的 RL 闭环 + 算力 + 迭代。kai0/CRAVE **做不到超越示教**,因为它们的 value 只描述"离 demo/人标进度有多远",没有自主结果信号。

**② 目标 = 给离线 AWBC 数据便宜地打 value/advantage 标签(我们的 A/B plan 场景)** → **CRAVE ≥ kai0-AE**。同域实测(kai0_base ep2302)CRAVE 与监督 AE 打平(corr 0.82)且更平滑(单调 100% vs 52%、噪声负 8% vs 32%),OOD 更稳——而 CRAVE **零标注零训练**,AE 要 ~1 周人工标 stage_progress。这正是 [AB_plan](awbc_milestone_value_AB_plan.md) 要验的:用 CRAVE 替代 AWBC 的 Stage-0+1。

**③ 目标 = value head 设计本身** → **离散/分布式 > scalar 回归**。RECAP 的 201-bin 分布式 CE 与 CRAVE 的 DP 阶梯**都靠离散化拿到鲁棒/校准**,而 kai0-AE 的 scalar+MSE 是三者里最弱的一环(我们观测到的 OOD 欠读、退步噪声多半源于此)。**离散不是 CRAVE 的劣势,反而是 RECAP 也采用的优点**;kai0-AE 恰恰输在"没离散"。

### 各自优缺点速查
- **kai0-AE** ✓ 学得的在线模型、能泛化到未见状态、可扩展为 action/失败感知;✗ 需人工进度标签、scalar 噪声大、advantage 只是 V 差分(非真优势)、无失败结果信号、夹在 CRAVE(更便宜)与 RECAP(更强)之间。
- **π*0.6-RECAP** ✓ 唯一能超越示教、真优势、失败感知、分布式校准、优势条件 BC 鲁棒(CFG);✗ 最贵(数千次真机自主练习 + RL + 大算力 + 迭代)、未开源、工程闭环复杂。
- **CRAVE** ✓ 零训练零标签、in-dist 配监督 AE 且更平滑、OOD 更稳、退步信号稀疏对齐真事件、可解释(簇间流转);✗ 只是"demo 流形进度"代理(非回报、非 A(o,a))、对**自信但错误**的动作(状态像 demo)会误高、不能超越示教、跨域需对齐挖矿域。

---

## 6. 对本工作的启示

1. **CRAVE 当 offline labeler**:在"省掉人工 stage_progress 标注"这件事上,CRAVE 是 kai0-AE 的直接、便宜、且更鲁棒的替代——A/B plan 已就绪。
2. **想要"超越示教"必须引入结果信号**:可把 CRAVE 的零训练 value 作为 RECAP 式 pipeline 的**冷启动 baseline V**(替代昂贵的 MC value 预训练),再用少量真机自主 rollout 的成败做 advantage 修正 → "CRAVE 冷启 + RL 微调",兼顾低成本与可超越示教。
3. **value head 别用 scalar MSE**:若仍要训 AE(AB 的 B 臂),应改 **分布式离散头(RECAP 式 201-bin CE)** 而非 scalar+Tanh——这是 kai0-AE 当前最该升级的一处。
