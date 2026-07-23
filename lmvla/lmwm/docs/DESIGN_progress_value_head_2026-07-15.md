# 设计：LMWM 连续进度 value 头（progress-value head）

> 2026-07-15。动机来自 task 6 深度分析（见 [`RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md`](RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md) 失败模式 + 数据特征）：**弥散子目标（无独特复现终端态）落在 LMWM"离散可分 milestone"假设的盲区**。连续进度 value 是对症的根本解——它把"预测下一个独特态"换成"我完成到几成"，后者在特征平坦区依然有定义。

---

## 1. 目标与定位

- **value = 任务进度标量 `v ∈ [0,1]`**，0=起点、1=完成，单调不减。
- **不依赖 milestone 可分性**：即使某段（task 6 布丁段）没有独特 milestone，进度 GT 仍逐帧上升 → 提供非退化监督/条件信号。
- **和 milestone 世界模型互补**：milestone 清晰处走 milestone+1；弥散处靠连续 value 兜底。不是替换，是加一层鲁棒信号。

**GT 来源**（二选一，先用①）：
1. **逐帧归一化时间** `t/(n-1)`：dense、处处单调、零额外计算。缺点=假设匀速进度。
2. **CRAVE 双锚 Viterbi value**（终版读出，corr 0.943 vs 监督 stage_gt）：更贴真实相位，但需先跑终版 milestone 发现（img⊕proprio+BayesianGMM+双锚）。

---

## 2. 两种用途（决定架构）

### 用途 A · 训练辅助监督头（先做，低风险）
- 结构：`v̂ = sigmoid(MLP(h))`，`h` = 世界模型/backbone 的当前帧特征。
- 损失：`L_val = MSE(v̂, v_gt) + λ_mono · mean(ReLU(v̂_{t-1} − v̂_t + m))`（复用 `train_ablation.py --anchor progress` 的形式：L149-150）。
- **不需要历史**：训练时 `v_gt` 已知且无歧义，前馈回归即可。task 6 后 60% 特征平但 `v_gt` 仍升 → 梯度非退化。
- 收益：给世界模型/action expert 一路 dense 梯度，直接补 P1 丢弃的弥散段监督。

### 用途 B · 推理时进度条件输入（后做，需历史）
- 推理无 GT，须**在线估** `v̂_t`，再作为 token/embedding 喂 action expert（"你已 70%，推向完成"）。
- **必须用历史**：task 6 后段 obs 别名（mid≈end=0.070，60% 画面≈100%）→ 单帧前馈分不清"60% 还是已完成"。
- **形态 = 发射 + 转移（CRAVE 已验证，勿裸自回归）**：
  - 发射：`e_t = MLP(obs_t)` 出粗 value（可能别名）；
  - 转移：单调约束 `v̂_t ≥ v̂_{t-1}` + 在线 DP（SymVote / 因果 Viterbi）把历史带进来纠正别名。
  - **禁**裸 `v̂_t=f(obs_t, v̂_{t-1})`：最优解退化成 copy 上一帧，模型不读观测。
- 证据（CRAVE 侧）：无历史（最近质心）别名击穿冲 1.0；全局历史（离线 Viterbi）corr 0.943；有限历史（在线 SymVote）0.83 → **历史治别名，越全越好**。

---

## 3. 实现路径（增量、高内聚低耦合）

1. **P0 · 生成连续 value GT**：先用逐帧时间（零成本）；有余力再切 CRAVE 双锚 value（须先把 LIBERO milestone 对齐终版架构）。存 `lmwm/data/libero_progress/ep*.npy`。
2. **P1 · 加辅助头 + 损失（用途 A）**：在 LMWM 生成器/world-model forward 里挂 `value_head`，`L_val` 加权进总 loss。**先只做监督，不改推理**。
3. **P2 · 最小验证 ✅ 通过（2026-07-15）**：见下 §3.1。
4. **P3 · 混合目标（可选，治弥散段）**：世界模型目标 = milestone 清晰处用 milestone+1；当**下一 milestone 太远 / 特征位移 < 阈值**时回退固定时延 t+τ dense 目标。"有路标走路标，没路标走 dead-reckoning"。
5. **P4 · 进度条件输入（用途 B）**：加发射+转移的在线 value 估计器，`v̂_t` 作条件喂 action expert；训练用 teacher-forcing GT value + scheduled sampling 弥合 exposure bias。

### 3.1 P2 验证结果 ✅（2026-07-15，前馈 probe，无历史）

**问题**：milestone 方法在 task 6 后 60%（弥散段）塌成 1 档。那段特征轨迹到底是"单调漂移（可学出进度）"还是"围绕定点的噪声（学不出）"？决定 value 头能否救弥散段。

**方法**：task 6（弥散，M0.68）episode 级 train/val 划分；PCA128 特征 → `MLPRegressor(128,64)` 回归**逐帧归一化时间**（用途 A，前馈无历史）；测 val 集**后 60% 的 Spearman(v̂, t)**。

| task6 后 60% | 整体 spear | 前40% spear | **后60% spear** | 后60%递增步占比 |
|---|---|---|---|---|
| img-only | 0.98 | 0.97 | **0.96** | 0.57 |
| img+proprio | 0.99 | 0.99 | **0.97** | 0.59 |

**结论**：
- **后 60% Spearman 0.96 = 强单调可学信号**。milestone 方法丢弃的弥散段，其实有连续单调进度可回归 → **value 头恢复了稠密非退化监督，设计前提成立**。
- **前馈 img-only 就够**（proprio 仅 0.96→0.97）：进度回归只要**弱单调梯度**，不像 milestone 需要**独特可分簇**。mid≈end 别名（0.070）是"中点 vs 终点"的位移小，但后段逐帧的累积漂移仍单调。
- 递增步占比 0.57 = 有局部抖动（前馈无历史的预期）；**单调约束/历史（用途 B）可平滑到近 1.0**，但不影响"信号存在"的结论。
- 脚本：`/tmp/p2_probe.py`（PCA128 + MLPRegressor + 逐帧时间 GT + 后段 Spearman）。

---

## 4. 风险 / 待定

- **copy-cheat**（用途 B）：预测增量 Δv 或加 anti-persistence/lift 项，逼模型读观测。
- **exposure bias**（用途 B）：训练喂 GT 历史、推理喂预测历史 → scheduled sampling。
- **GT 选择**：逐帧时间假设匀速；真实进度非匀速时用 CRAVE 双锚 value 更准，但引入终版 milestone 依赖。
- **上限诚实**：弥散子目标类任务，value 头能把 LMWM 追平固定时延，未必超越——LMWM 的自适应优势只在"有清晰阶段结构"的任务兑现（task0/2/7）。value 头是**兜底鲁棒**，不是超越点。

---

## 5. 关联
- 数据特征与适配边界：[`RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md`](RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md)
- 已有进度锚实现：`lmwm/scripts/train_ablation.py --anchor progress`（L82/124/149）
- CRAVE value 读出（发射+转移证据）：[`../../crave/docs/final_architecture.md`](../../crave/docs/final_architecture.md) §2.10、`crave/src/crave/value/readout.py`
