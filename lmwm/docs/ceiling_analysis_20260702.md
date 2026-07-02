# LMWM 是否到瓶颈?深度分析(2026-07-02)

> 对"LMWM 已达天花板"这一结论做严格复核。三个无训练/低成本诊断推翻了"固有 ~13 分支歧义"的说法,并精确定位真正的天花板在哪。

## 复核动机

Phase C 的结论是"单帧已到天花板,~13 分支熵是固有任务歧义"。但那个熵是 `H(next_milestone | current_milestone)`,高熵可能有三种原因,只有一种是真瓶颈:

- **H1 标签抖动**:milestone 分配 ~每 2 帧翻转,"next-unique"多是簇边界噪声,非真实阶段转移 → 可修的标签噪声。
- **H2 簇≠相位混叠**:KMeans 是视觉簇,折叠臂在不同任务阶段重访同一视觉簇 → 单帧/单 milestone 对进度歧义,但条件于"轨迹位置"可消歧。Phase C 加的是**原始帧历史**(也混叠),测错了杠杆。
- **H3 真多模态**:折叠真有分支 → 不可约,但 top-k/NLL 可处理,且远小于 13。

## 三个诊断(bias-free:train episode 估计,held-out 评估)

### 诊断 1 & 2:平滑标签 + 时间条件(`scripts/diagnose_ceiling.py`)

| 平滑窗 w | 压缩长度 | ctx=当前milestone top1/NLL | ctx=当前milestone+时间bin top1/NLL |
|---|---|---|---|
| 1(原始) | 54.1 | 0.246 / 2.556 | 0.281 / 2.414 |
| 3 | 32.1 | 0.229 / 2.672 | 0.262 / 2.517 |
| 5 | 23.9 | 0.235 / 2.661 | 0.270 / 2.514 |
| 9 | 16.9 | 0.226 / 2.626 | 0.259 / 2.483 |

- **H1 被推翻**:时序平滑把压缩长度从 54 砍到 17,但预测**没变好**(反而略差)。抖动不是瓶颈。
- **H2 部分成立**:加绝对时间 bin 一致带来 +3.5pt top1 / −0.14 NLL。轨迹位置携带 current-milestone-id 之外的真实信息。

### 诊断 3:kNN 表示天花板(`scripts/knn_ceiling.py`,GPU)

对每个 held-out 帧,用 DINOv3-H 特征 cosine 检索 train 帧,软投票其 next-milestone。这是给定表示+标签下的近最优估计器,是**任何模型的上界**。

| 估计器 | top1 | NLL |
|---|---|---|
| kNN k=50 | 0.3731 | 2.048 |
| kNN k=50 + 时间 | 0.3757 | 2.038 |
| **我们的神经 MLP(real-future)** | **0.383** | **1.98** |

- **决定性**:我们的 MLP(0.383)**已达到甚至略超 kNN 上界**(0.376)。MLP 没有欠拟合 —— 它已榨干 DINOv3-H 帧表示中关于该标签的几乎全部可预测信息。
- **时间对连续特征几乎无用**(+0.26pt),对离散 milestone-id 有用(+3.5pt)—— 因为帧特征已编码了时间/相位会提供的信息。

## 精确结论:到瓶颈了吗?

**是,也不是。**

- **是**:已到"冻结 DINOv3-H 图像特征 → 混叠离散 milestone"这一**当前表述**的天花板。MLP == kNN 上界,已实测 warmup/wd/bias-norm/β 全部无效 —— **继续在 LMWM 侧调模型是徒劳的**。
- **不是**:这不是根本天花板。上界由两个**上游**因素设定,二者都可改:
  1. **表示**:DINOv3-H 是纯图像、冻结的,它编码的"关于未来"的信息封顶。
  2. **标签空间**:KMeans 视觉簇被重访(混叠),离散 milestone-id 本身有损。

## 突破方法(按证据强度/性价比排序)

**A. 改标签空间(最大杠杆 —— 混叠封住了一切):**
1. **重定义 milestone 为相位唯一**(CRAVE 侧):聚类时拼进 progress/time(CRAVE 已有 image⊕proprio 联合聚类机制),使每个 milestone = 唯一任务阶段、不被重访 → 从源头消除 H2 混叠。
2. **换连续目标**:直接回归**未来帧特征**(LaWAM 式,无离散化、无混叠;正是 VLA subgoal 条件消费的东西),或回归**未来 progress**(标量、单调、well-defined、直接是规划/value 信号)。用特征 cosine / progress MAE 评估,而非 milestone top1。

**B. 改表示(另一个天花板因子):**
3. **加本体感觉 proprio**:CRAVE 已证 proprio 消解视觉混叠(折起态 vs 摊平态的别名)。给输入拼 arm/gripper 状态,直接打击封顶的混叠。相对便宜、可测(proprio 存在于 CRAVE 特征缓存)。

**C. 换度量(top1 是错的成功指标):**
4. 混叠离散标签上的 top1 本就是错的成功标准。对 VLA 有用的是:校准分布(已有,ECE 0.005)、top-k(top5 0.86)、latent subgoal(cosine 0.94)、progress。应汇报/优化这些,而非 top1。

## 建议

停止在"帧→离散 milestone"MLP 上调参(已证到顶)。两个最高价值、可测的下一步:
- **短期可测**:给输入加 proprio(B3),重测 kNN 上界 + MLP —— 若 top1 明显 >0.38,说明表示是瓶颈且 proprio 能抬。
- **中期重构**:把目标从离散 milestone 换成**连续未来特征 / progress 回归**(A2),这既绕开离散天花板,又是更有用的 VLA 信号。
- **根因**:重定义相位唯一 milestone(A1,CRAVE 侧)。

## 追加诊断:用 previous milestone(路径历史)条件是否更可靠?

假设:当前簇混叠时(同一视觉簇在不同阶段被重访),当前帧无法区分是哪一次访问,但"上一个 milestone"能。`scripts/reliability_prev_milestone.py`,held-out 预测真实 next-unique milestone:

| 上下文 | top1 | NLL |
|---|---|---|
| **离散** cur | 0.204 | 2.675 |
| **离散** cur+prev | **0.313** | 2.377 |
| **离散** cur+time | 0.245 | 2.508 |
| **离散** cur+prev+time | 0.321 | 2.413 |
| **kNN(当前帧)** frame-only | **0.389** | 1.942 |
| **kNN** frame+prev(α0.5) | 0.390 | 2.047 |
| **kNN** frame+prev(α1.0) | 0.390 | 2.047 |

**双向结论**:
- 在**离散 milestone-id** 层面,加 prev **大幅提升**(+11pt,0.204→0.313)—— 证实混叠是真的,prev 确实消歧,且比 time 更强。
- 但**加到当前帧特征上无增益**(0.389→0.390,NLL 反略差)。且 **frame-only(0.389)已 > cur+prev 离散(0.313)**。

**⚠️ 上表的 kNN 结论后被更正**:cosine-append 把 one-hot 拼到归一化的 1280-d 帧上再归一,信号被稀释,cosine-kNN 学不到该离散通道 —— 是**方法学假象**,不是真的无用。见下。

### 更正:用**学习型组合器**测 milestone 路径(`reliability_milestone_path.py`)

同一 MLP 类(与 LMWM 同构)公平融合 frame + milestone 路径 one-hot,held-out 预测真实 next milestone:

| 变体 | top1 | NLL |
|---|---|---|
| frame_only | 0.382 | 2.355 |
| **frame + milestone(t−1)** | **0.409** | 2.371 |
| frame + milestone(t−1, t−2) | 0.405 | 2.432 |
| path_only(K2,无 frame) | 0.319 | 2.320 |

**更正后的结论**:**milestone 路径确实有效** —— 加 milestone(t−1) 使 top1 从 0.382 → **0.409(+2.7pt)**,**突破了 frame-only kNN 上界(0.389)**。K=1 就吃满增益,K=2 不再加。

**为什么突破了"表示天花板"**:kNN 上界只衡量"仅从当前帧"的最优;milestone 路径提供的是**与帧正交的离散路径信息**(走的是哪条 fold 路线),学习型模型能融合它。所以帧不是全部 —— **milestone(t−1) + current latent → milestone(t+1) 更可靠**,与用户直觉一致。可作为 LMWM 的输入增广落地(prev-milestone 在 VLA 推理时因是上一个已提交阶段而因果可得)。

## 突破:加 state(proprio)+ milestone 路径(`reliability_state.py`)

参考 LaWM 输入 state 的做法,给输入拼 14 维 observation.state(z-score,从 kai0 parquet 按 frame_index 关联)。学习型 MLP 探针,held-out:

| 变体 | top1 | NLL |
|---|---|---|
| frame_only | 0.382 | 2.384 |
| frame+path(t−1) | 0.405 | 2.352 |
| **frame+state** | **0.409** | **2.110** |
| **frame+path+state** | **0.434** | 2.136 |

**决定性结论**:milestone 路径与 state **正交且叠加** —— frame→frame+path+state 使 top1 **0.382→0.434(+5.2pt,+14% 相对)**,NLL 2.38→2.14。state 对 NLL 提升最大(proprio 消解折起/摊平的视觉别名,正是 CRAVE 早已发现的)。

**这推翻了"到表示天花板"的判断**:frame-only kNN 上界(0.389)不是任务上界 —— proprio 和路径历史携带了帧缺失的、因果可得的正交信息。

### 已落地到完整 LMWM(`pairs_next_unique_augin.npz`,in_dim=1332=1280+38+14)

用 `export_augmented_input_pairs.py` 把 `[frame; prev-milestone one-hot; state z-score]` 作为输入,训练完整 unified LMWM(+ medoid subgoal 目标),held-out real-future:

| | 原始 frame-only | **frame+prev+state** |
|---|---|---|
| neural greedy top1 | 0.383 | **0.408**(+2.5pt) |
| neural greedy NLL | 1.977 | **1.953** |
| **+图先验融合(λ=0.3)top1** | 0.417 | **0.434** |
| +图先验融合 NLL | 1.798 | **1.776** |

(完整模型 +2.5pt 小于探针的 +5.2pt,因完整模型在**所有帧**上训练而非更干净的 stage-代表帧;但仍是确凿提升,且保留了更好的 medoid subgoal。ECE 略升到 0.116,温度 T=1.30 可修回 ~0.005。)**这是目前最佳 LMWM artifact:融合 top1 0.434 / NLL 1.776。**

## 关于时序数据(LaWM 是否用了更多时序?)

- **LaWM/LAM 本身只用 2 帧**(`num_frames: 2`,current + future,dt≈1.6s):从 (z_t, z_{t+h}) 抽 inverse 转移码 u_t,再 forward 预测 z_hat_{t+h}。它**不是长时序模型**,时序只体现在"当前↔未来"这一对上 —— 和我们的 next-unique pair 同构。
- 我们**已测更多时序**:Phase C(帧历史 H=4/6)无增益;本节(prev-milestone 路径)对帧模型无增益。两者一致 —— **更多时序上下文不能突破表示天花板**。
- 真正有效的杠杆是**目标/表示**(episode-medoid 连续目标 +0.031 cos),而非时序上下文。

## 产物

- `scripts/diagnose_ceiling.py` → `outputs/ceiling_diag/summary.json`
- `scripts/knn_ceiling.py` → `outputs/ceiling_diag/knn_summary.json`
- `scripts/reliability_prev_milestone.py` → `outputs/ceiling_diag/prev_milestone.json`
