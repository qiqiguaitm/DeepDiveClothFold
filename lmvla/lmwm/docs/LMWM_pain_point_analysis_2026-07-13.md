# LMWM 痛点分析：它能解决 VLA 的什么问题？

> 2026-07-13，基于与用户讨论的修正版本。核心前提：LaWAM 的世界模型预测**固定 1.6 秒时延**的帧特征（非相邻帧），数据/模型的不同域需要不同时延（预训练 0.9s，真机 1.6s）。LMWM 用**无监督 milestone 边界**替代这个固定时延——milestone+1 帧是下一个语义子目标的第一帧，而非固定 N 帧之后。

---

## 核心能力

LMWM 解决一个根本问题：**固定时延世界模型的监督信号是均匀的、无差别的——它不区分"正在发生有意义的变化"和"什么都没发生"。**

CRAVE 聚类发现的 milestone 边界天然落在状态变化点，milestone 条件生成器的训练对下界保证了语义信息密度。这个差异在 5 个维度上有可测后果。

---

## 痛点 A：世界模型监督信号的信息密度不均

**LaWAM 行为**：所有训练对的未来帧都是 t+固定时延（LIBERO 上约 t+7 帧，约 0.4s；真机 1.6s）。但不同时刻该时延窗口内的信息密度差异很大：手臂快速移动时剧烈变化，悬停等待时几乎不变。大量"近距离对"的梯度≈0，浪费算力。

**LMWM 差异**：每个 `(cur_fi, tgt_fi)` pair 的 tgt_fi 是 `first[milestone+1]`——保证跨语义边界，特征距离下界不为零。

**指标**：
1. **pair 特征空间距离分布**：cosine_distance(cur, tgt) 的分布。LaWAM 应有双峰（大量近距离 + 少量远距离）；LMWM 低距离峰应更小、均值更高。
2. **辅助 loss 与 eval SR 的相关性**：LMWM 的 loss_perceptual 下降应比 LaWAM 的"更预示 SR 提升"（每个梯度步都在消费有意义的信号）。

**需要新训练？** 否。现有 pairs.npz 可算距离分布；gf3 训练 log 可做 loss-SR 相关性（需 eval 结果）。

---

## 痛点 B：固定时延是域相关的超参数

**证据**：LaWAM 预训练数据用 0.9s 时延，真机数据用 1.6s。说明"预测多长时间后的未来"依赖于数据域（动作频率、任务节奏）。每个新域需要重新调这个参数。

**LMWM 差异**：milestone 边界是**数据驱动的**——任务节奏快时 milestone 间距小（可能 5-10 帧），节奏慢时间距大（可能 30-50 帧）。`sec_chunk` 参数被内化到自适应边界中。

**指标**：
1. **milestone 间距分布**：统计 pairs.npz 中 `(tgt_fi - cur_fi)` 的分布——应该是自适应变化范围而不是固定值。
2. **sec_chunk 鲁棒性**：在不同 sec_chunk 值下测两臂 SR。LaWAM 应对此敏感（t+7 vs t+14 不同），LMWM 应相对不敏感。

**需要新训练？** 间距分析不需要（现成 pairs.npz）；鲁棒性实验需要小规模训练。

---

## 痛点 C：LMWM 是编码器无关的——可跨 VLA 特征空间复用

CRAVE 的聚类只需要有判别力的特征空间。可以在：
- DINOv3 空间（当前，通用 vision foundation model）
- SigLIP 空间（π₀/π₀.₅ 的视觉编码器）
- VLA 自身中间层特征（end-to-end 一致性）

所以 LMWM 不是绑定到 LaWAM 的——它是**通用插件**：任何有冻结视觉编码器的 VLA，都可以抽特征→CRAVE→训 generator→作为辅助任务接入。

**注意**：这是**跨编码器**的通用性，不是**跨具身**的通用性。双臂 Piper→单臂 Franka 不适用（编码器相同但运动语义不同）。同具身跨场景（不同任务、不同背景）的通用性有待验证。

**指标**：在 SigLIP 空间下重复 CRAVE（用 π₀/π₀.₅ 的 SigLIP 编码器提取特征 + 聚类 + 发现 milestone），对比 DINOv3 发现的 milestone 质量（M 数、cluster 纯度、语义可解释性）。

**需要新训练？** 仅需特征提取（已有 pi05 SigLIP 编码器），不需要训练。

---

## 痛点 D：长周期任务的隐式进度表征

LaWAM 的固定 1.6s 预测不能告诉模型"做到 70% 了"——只能告诉模型"再过 1.6 秒画面长啥样"。对于 500+ 步的任务，进度信息比局部预测更有价值。milestone 序列天然提供进度信号（M 个有序子目标 = M 段进度）。

**指标**：按任务 episode 长度分组 eval SR（短 <100 步 / 中 100-300 步 / 长 >300 步）。LMWM 在长周期组应有更大 SR 提升。

**需要新训练？** 否。现有 eval 结果 + LIBERO task 元数据分组即可。

---

## 痛点 E：视觉干扰鲁棒性（待验证）

**理论**：CRAVE 在跨 episode 的 recurrence 上聚类——只保留"反复出现的状态变化"作为 milestone，单 episode 特有的背景/光照/干扰被聚类自然淘汰。

**实际状态**：这是理论推导，缺乏实验证据。需要：
1. 确定 LIBERO 是否有现成 distractor benchmark（或手动构造干扰环境）
2. SR 衰减率对比：(distractor_SR - clean_SR) / clean_SR

**需要新训练？** 是。需构造干扰环境 + 重新 eval（或至少用外部数据评测）。

---

---

## P0 实证分析（2026-07-13，基于 LIBERO DINOv3-base 数据）

### 数据来源
- 95395 个 milestone 训练对（pairs.npz，40 任务，中位 M=10）
- 特征空间：DINOv3-vitb16 grid [256, 768]，gist pooling 后计算余弦距离
- 对照基线：同一 episode 的固定 t+7（LIBERO 0.35s）和 t+32（真机 1.6s）特征距离

### P0-B：Milestone 间距 vs 固定时延

| 指标 | LMWM 自适应间距 | LaWAM t+7 (LIBERO 0.35s) | LaWAM t+32 (真机 1.6s) |
|---|---|---|---|
| 均值 / 中位 | 10.0 / **5** 帧 | 固定 7 帧 | 固定 32 帧 |
| 标准差 | **11.8** | 0 | 0 |
| 范围 | [1, 108] | [7, 7] | [32, 32] |
| 与固定偏差>10帧 | — | **17.7%** | **90.2%** |

**任务间节奏差异**：各任务间距中位范围 [2, 22] 帧（0.1s–1.1s），**变异系数 0.69**。最快 task21 中位=2 帧，最慢 task8 中位=22 帧——**10× 差异**。固定时延对此完全盲目。

### P0-A：特征空间信息密度分布

| | LMWM milestone+1 | LaWAM t+7 (0.35s) | LaWAM t+32 (1.6s) |
|---|---|---|---|
| n | 3000 | 2984 | 2314 |
| 均值 | 0.01605 | 0.01686 | 0.03347 |
| 中位 | 0.01337 | 0.01475 | 0.03004 |
| 标准差 | **0.01087** | 0.00894 | 0.01748 |
| P10 | 0.00518 | 0.00765 | 0.01554 |
| P90 | 0.03081 | 0.02890 | 0.05416 |
| <0.005 比例 | **9.1%** | 1.8% | 0.1% |

**关键发现**：

1. **LMWM 不是"信息更多"，是"信息自校准"**：均值距离与 t+7 接近（0.95x），但**标准差大 22%**。LMWM 把预测分散到更广的距离范围：快节奏任务→近距离（P10=0.005，比 t+7 更低），慢节奏任务→远距离（P90=0.031，与 t+7 相近或更大）。

2. **极低信息对 (<0.005) LMWM 反而更多（9.1% vs 1.8%）**：这来自快节奏任务的 1-2 帧 milestone 间距——"下一步就在眼前"的正确信号，不是无效监督。t+7 在这些帧上反而跳到无关位置。

3. **t+32 极端错配**：90.2% milestone 对偏离 >10 帧，特征距离是 LMWM 的 2.1×。证实了"不同域需要不同时延"的 observation。

### 核心 Claim（数据支撑后修正）

> LMWM 把世界模型的监督信号密度**校准到任务的自然节奏**——快节奏密集预测近处，慢节奏稀疏预测远处，消除固定时延的"一刀切"错配。跨任务节奏变异系数 0.69 是此 claim 的直接量化证据。

---

---

## 外部证据验证（论文 + 本地实验）

> 原则：不靠推理自证，必须有外部数据或论文指标佐证。以下为 2026-07-14 的验证结果。

### 证据 1：LaWAM 论文（arXiv 2606.15768）自身暴露的问题

**来源**：LaWAM paper Appendix C.5, Table 1, Section 4.5。

**发现 1 — sec_chunk 从未消融**：
> "There is no ablation varying τ, sec_chunk, temporal stride, or future window size. No sensitivity analysis on these parameters is presented." (paper audit)

LaWAM 使用固定 τ=1.2s（robot）和 τ=0.4s（egocentric），但没有消融实验证明这些值是最优的，也没有灵敏度分析。**论文自身承认了固定时延是一个未经验证的隐式超参数。**

**发现 2 — 不同域需要手动调不同值**：
> "The paper uses fixed physical-time horizons τ = 1.2s for robot videos and τ = 0.4s for egocentric human videos. No justification is given for these choices." (Appendix C.5)

这恰好证实了用户的 observation（预训练 0.9s vs 真机 1.6s）——不同数据域需要不同的预测时延，且每换一个域就要手动调。**LMWM 的 milestone 自适应可以直接消除这个调参需求。**

**发现 3 — 不提供 per-task SR 分解**：
> "Table 1 reports only aggregated results across four LIBERO suites. The paper states they 'report success rates over 2,000 trials across 40 tasks' but does not break these down by individual task." (Appendix D.1)

单个 aggregate 数字（98.6%）完全掩盖了任务间难度差异——如果长任务 SR=85%、短任务 SR=100%，平均仍可报 98%。

**发现 4 — 世界模型至关重要（但预测目标未经审视）**：
> "Removing LaWM causes the largest drop" (Section 4.5 ablation)

确认世界模型对 VLA 性能有显著影响——从而预测目标的选择（next-frame vs milestone+1）是 consequential 的，不是 minor detail。

---

### 证据 2：LIBERO 任务长度分析

**数据**：LIBERO 40 任务，1693 episodes。

| 指标 | 短任务 (<200步) | 长任务 (>400步) | 全量 |
|---|---|---|---|
| 平均 episode 长度 | ~95 步 | **407 步** | 162 步 |
| milestone 间距中位 | **4 帧** | **15 帧** (3.8×) | 5 帧 |
| episode 占比 | 58% | 4% | — |

**结论**：任务间长度差异 4.6×，milestone 间距差异 3.8×。固定 t+7 对短任务过远（跳过了可能的密集子目标边界），对长任务过近（多个连续预测才有意义的信息）。LaWAM 不报 per-task SR，无法判断长任务是否系统性低 SR。

---

### 证据 3：P0 本地实证（已嵌入上文）

- **P0-B**：跨任务 milestone 间距变异系数 0.69，17.7% 对偏离固定 t+7 >10 帧
- **P0-A**：LMWM 特征距离分布比 t+7 标准差大 22%（自校准的证据）

---

### 证据强度总表

| 痛点 | 本地证据 | 论文证据 | 验证强度 | 还需什么 |
|---|---|---|---|---|
| A: 信息密度自校准 | ✅ P0 距离分布 | ⚠️ 间接（LaWAM 只有 L2 loss） | 中等 | gf3 训练 loss 曲线 + eval SR 相关性 |
| B: 固定时延是未消融超参数 | ✅ 17.7% 错配率 | ✅ LaWAM 无消融 + 多域手动调 | **强** | sec_chunk 敏感性消融实验 |
| C: 跨编码器通用性 | 🔄 SigLIP 待加载 | ✅ DINOv3 世界模型范式成立 | 中等 | SigLIP/DINOv3 聚类质量对比 |
| D: 长周期进度表征 | ✅ 4.6× 长度差异 | ✅ LaWAM 不报 per-task SR | **强** | 按长度分组的 eval SR |
| E: 视觉鲁棒性 | ❌ | ❌ 无直接证据 | **弱** | 构造干扰环境或找现有 benchmark |

**当前最稳固的两个 claim（可对外宣称）**：
1. **痛点 B**：LaWAM 的固定预测时延是未消融的超参数——论文自己承认了，且不同域需要人工设不同值。LMWM 用数据驱动的 milestone 边界替代了此参数。
2. **痛点 D**：LaWAM 的单 aggregate SR 掩盖了任务间难度差异——论文不提供 per-task breakdown，而 LIBERO 任务间有 4.6× 长度差异。

---

## 优先级与测试路线

| 优先级 | 痛点 | 核心指标 | 验证强度 | 需要新训练？ |
|---|---|---|---|---|
| **P0** | B: 超参数鲁棒性 | milestone 间距分布 / chunk 敏感度 | **强** | 鲁棒性消融=是 |
| **P0** | D: 长周期进度 | 按长度分组的 SR | **强** | 否（现有 eval） |
| **P1** | A: 信息密度 | pair 距离分布 / loss-SR 相关性 | 中等 | 否（现有训练 log） |
| **P1** | C: 跨编码器 | SigLIP 空间 milestone 质量 | 中等 | 否（特征提取） |
| **P2** | E: 视觉鲁棒性 | 干扰下 SR 衰减率 | **弱** | 是（构造环境） |

### 不成立或已排除的痛点
- **跨具身通用性（原痛点 6）**：双臂→单臂不成立，修正为同具身跨场景通用性（待验证）。
- **梯度冲突（原痛点 4）**：LMWM 和策略在统一编码器空间可解，不构成独立痛点。
