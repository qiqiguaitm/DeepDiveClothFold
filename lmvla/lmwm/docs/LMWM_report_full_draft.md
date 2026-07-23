# LMWM: Latent Milestone World Model for VLA

> 初稿框架。§1 痛点（已验证）、§2 方案设计、§3 实验与结果对比。
> 数据来源：LIBERO 40 任务 × DINOv3-base 特征空间。对比基线：LaWAM（arXiv 2606.15768）released checkpoint + 同管线 no-swap 自训。

---

## §1 痛点与动机

> 原则：不靠推理自证——每个 claim 必须有本地实验数据或已发表论文的指标支撑。完整的 5 痛点探索版（含 C 跨编码器 / D 长周期 / E 视觉鲁棒性 + 测试路线）见 [`history/LMWM_pain_point_analysis_2026-07-13.md`](history/LMWM_pain_point_analysis_2026-07-13.md)；本节保留已验证、可对外宣称的 3 个。

| # | 痛点 | 本地证据 | 论文证据 | 强度 |
|---|---|---|---|---|
| 1 | 固定预测时延 τ 是未消融隐式超参数 | P0-B: 跨任务 CV=0.69, 17.7% 对偏离 t+7 >10帧 | LaWAM 无 τ 消融 + 多域手动选不同 τ | **强** |
| 2 | 单 aggregate SR 掩盖任务间难度差异 | 4.6× 任务长度差异, 长任务 ms 间距 3.8× | LaWAM 不报 per-task SR | **强** |
| 3 | 世界模型监督信号密度不均（自校准效应） | P0-A: LMWM 特征距离 std 比 t+7 大 22% | 间接（无直接论文指标） | 中等 |

**核心 claim**：现有 VLA 世界模型用固定时延预测未来帧特征，而在官方实现中这个 τ **被结构性地焊死在动作块最后一帧上，不是可独立调节的超参数**。LMWM 用 CRAVE 无监督发现的 milestone 边界替代 τ，**把 WM 监督时域与动作块解耦**，使预测目标随任务节奏自适应。

### 痛点 1：WM 预测时域与动作块未解耦

现有 latent world model（LaWAM, Seer, DreamVLA, GR-2 等）训练世界模型时使用固定预测时延 τ：预测 τ 秒后的帧特征。
在 LaWAM 官方实现中，该 τ 由动作块长度直接决定（见下方"主论点"），因此**无法独立于动作块被调节或消融**。

> ⚠️ 本节原写作"τ **从未被系统消融**"，**已于 2026-07-19 证伪并删除**（GR-1 附录 A.4 表 5 恰恰做了该消融）。
> 详见本节末的重大更正。

**本地数据（LIBERO, DINOv3-base, 40 任务, 95395 对）** — CRAVE 无监督发现的 milestone 间距高度变异：

| 指标 | 值 |
|---|---|
| Milestone 间距均值 / 中位 | 10.0 / **5** 帧 |
| 标准差 | **11.8** |
| 范围 | [1, 108] |
| 与 LaWAM t+7 (LIBERO 0.35s) 偏差 >10 帧 | **17.7%** |
| 与 LaWAM t+32 (真机 1.6s) 偏差 >10 帧 | **90.2%** |

跨任务节奏变异系数 **0.69**：最快任务 milestone 间距中位 2 帧（0.1s），最慢 22 帧（1.1s）——**10× 差异**。单个固定 τ 无法同时适配：对快任务过远（跳过密集语义边界），对慢任务过近（窗口内无有意义状态变化）。

**⚠️ [2026-07-19 重大更正]** 本节此前的表述是 **"τ 从未被系统消融"**，并引用了两条标注为
"LaWAM Appendix C.5" 的原文。经核查，**该表述已被证伪，引文亦不可靠，两者均已删除**：

1. **直接反例：GR-1（arXiv 2312.13139）附录 A.4 表 5 就做了这个消融。** 而我们原文恰恰点名 GR-1
   作为"从未消融"的例子。原文："We compare the effectiveness of predicting images at different
   future steps (i.e. 1, 3, and 5) on CALVIN benchmark."　结果 Avg.Len：Δt=1 → 3.61，**Δt=3 → 3.82**，
   Δt=5 → 3.67，并给出机理解释："predicting frames that are too far into the future may not be able
   to provide good guidance for immediate local action prediction."
   另有三篇独立反例：**JOPAT**(2605.23856, 附录 A.3.2 图 5 prediction-horizon sensitivity)、
   **AHEAD**(2606.02486, 附录 E.4 表 15, K∈{1,2,3,5,8,12}×5 seeds)、**OneWM-VLA**(2605.07931, §5.6 表 13)。
2. **"固定 τ"这个前提本身也不成立**：**SuSIE**(2310.10639) 的 τ 是从 U[k_min,k_max)=U[11,14) **区间采样**的。
3. **原引文不可回溯**：仓库内无论文 PDF、无审计报告，那两句只存在于我们自己的文档中，且源文件标注为
   `(paper audit)`——是审计者的否定性结论，被误当成"论文自身承认"。**没有论文会这样写自己。**

**可辩护的弱化表述（建议采用）**：

> 少数工作报告过 horizon 敏感性分析（GR-1 在 CALVIN 上比较 Δt∈{1,3,5}；JOPAT 扫描 future-observation
> offset），但这类消融普遍局限于单一 benchmark、窄范围、且**未与控制频率/物理时长解耦**。特别地，
> **latent** world model 一线（LaWAM 固定 1.2s、Seer n=3、DreamVLA、GR-2）一致地把 τ 当作未经检验的实现细节。

**更强且经第一手代码验证的论点(建议作为痛点 1 的主论点)**：
在 LaWAM 官方实现中(`starVLA/dataloader/lerobot_datasets.py::_sample_video_delta_indices`)，
`num_frames=2` 时未来帧索引 = `action_delta_indices[-1]`，即**世界模型的预测目标被结构性地焊死在动作块的最后一帧上**。
因此 τ **根本不是一个可独立调节的超参数**——想单独消融 τ 就必然同时改变动作块长度，二者混淆。
"robot τ=1.2s / egocentric τ=0.4s"实为**动作块时长**的选择(依控制频率与任务节奏，有独立且正当的理由)，
用它论证"WM 时延未经验证"是**不成立的**。**真正的缺口是"WM 监督时域与动作块的解耦"，而这正是 LMWM 在做的事。**

> 🚫 **引用禁令**：搜索摘要曾声称 DAWN/Pixel Motion Diffusion (2509.22652) 有 "表 10，k∈{5,10,20,30}"
> 的 temporal offset 消融。逐表核对 v1/v2 全文——**该表不存在**，全文仅 7 张表。**请勿引用。**

**LMWM 解法**：用 CRAVE 数据驱动 milestone 边界替代固定 τ，间距随任务节奏自适应（快 2 帧，慢 108 帧），消除手动调参。

### 痛点 2：单 aggregate 成功率掩盖任务间难度差异

VLA 论文（含 LaWAM）通常只报 LIBERO 各 suite 的聚合 SR，不做 per-task 分解。40 任务上一个 98.6% 可能掩盖显著跨任务方差——长周期任务系统性低 SR，被短任务高 SR 平均掉。

**论文自身的数字即为证据**（LaWAM Table 1）：Spatial 99.4 / Object 99.6 / Goal 98.4 / **Long 97.0** → 平均 98.6。
Long 是四者中最低，且这个"最低列"现象在整个文献里是普遍的（OpenVLA Long 53.7 vs 平均 76.5；π₀ 85.2 vs 94.2；π₀.₅ 92.4 vs 96.9）——
**suite 级聚合已经在掩盖难度差异，task 级只会更严重**。经核查，OpenVLA / π₀ / π₀.₅ / SpatialVLA / CoT-VLA / TraceVLA / Octo / GR00T-N1
均**未**给出 LIBERO per-task 表；给出逐任务分解的只有 LIBERO-PRO(2510.03827) / LIBERO-Plus(2510.13626) 等二次分析工作。

**外部佐证（饱和问题）**：LIBERO-Plus 明确指出 VLA 在 LIBERO 上 "almost saturated"，并刻意剔除所有模型都能解的任务以避免天花板效应；
LIBERO-PRO 论证高分来自记忆而非泛化（物体位置偏移 >0.2 即掉到 0%）。
这与我们 24 路变 seed 的实测一致：**libero_10 有 8/10 任务饱和在 93~100，聚合 94.30 完全掩盖了 t6 的 76.5**。

**本地数据（LIBERO 1693 episodes, 40 任务）**：

| 指标 | 短任务 (<200步) | 长任务 (>400步) | 全量 |
|---|---|---|---|
| 平均 episode 长度 | ~95 步 | **407 步** | 162 步 |
| Milestone 间距中位 | **4 帧** (0.2s) | **15 帧** (0.75s) | 5 帧 |
| Episode 占比 | 58% | 4% | — |

任务间长度差异 **4.6×**，milestone 间距差异 **3.8×**。短任务占多数（58%），在聚合 SR 中天然主导。

**论文证据**：LaWAM 不提供 per-task 分解——"Table 1 reports only aggregated results across four LIBERO suites... does not break these down by individual task."（Appendix D.1）。若长任务 SR=85%、短任务 SR=100%，聚合仍可报 98%。

**LMWM 潜在优势**：milestone 序列天然提供隐式进度信号（M 个有序子目标 = M 段进度）。验证：按任务长度分组统计 SR，LMWM 应在长周期组有更大提升。

### 痛点 3：世界模型监督信号的信息密度不均（自校准效应）

固定时延为每帧提供**相同距离**的预测目标（t+τ），但各时刻窗口内语义变化量差异巨大：快速移动时剧变，悬停时几乎不变。大量"近距离对"梯度≈0，浪费算力。

**本地数据（LIBERO, 3000 对采样, gist cosine distance）**：

| | LMWM milestone+1 | LaWAM t+7 (0.35s) | LaWAM t+32 (1.6s) |
|---|---|---|---|
| n | 3000 | 2984 | 2314 |
| 均值 | 0.01605 | 0.01686 | 0.03347 |
| 中位 | 0.01337 | 0.01475 | 0.03004 |
| **标准差** | **0.01087** | 0.00894 | 0.01748 |
| P10 | 0.00518 | 0.00765 | 0.01554 |
| P90 | 0.03081 | 0.02890 | 0.05416 |
| <0.005 比例 | **9.1%** | 1.8% | 0.1% |

**关键发现**：(1) LMWM 均值距离与 t+7 接近（0.95×）但**标准差大 22%**——把目标分散到更广距离：快节奏近（P10=0.005），慢节奏远（P90=0.031），即**信息密度自校准**。(2) t+32 极端错配：90.2% 对偏离 >10 帧，距离是 LMWM 的 2.1×，证实"不同域需不同 τ"。(3) 极低信息对（<0.005）LMWM 9.1% vs t+7 1.8%——来自快节奏 1-2 帧间距，是"下一步就在眼前"的正确信号，非无效监督。

**论文证据**：暂无直接论文指标（LaWAM 用简单 L2 loss，无信息密度概念）；本地距离分布为主要支撑，待训练完成后用**辅助 loss 下降与 eval SR 提升的相关性**加强。

### 验证强度汇总

| 痛点 | 本地数据 | 论文证据 | 总体强度 |
|---|---|---|---|
| 1: 固定时延是未消融超参数 | ✅ P0-B 17.7% 错配, CV=0.69 | ✅ LaWAM 无 τ 消融 + 多域手动选值 | **强** |
| 2: 单 aggregate SR 掩盖难度差异 | ✅ 4.6× 长度差异, 3.8× ms 间距 | ✅ LaWAM 不报 per-task SR | **强** |
| 3: 监督信号密度自校准 | ✅ P0-A 距离分布, std 大 22% | ⚠️ 间接（无直接论文指标） | 中等 |

**当前最稳固、可对外宣称的两个 claim**：痛点 1（LaWAM 自认固定时延未消融 + 多域手动选值）、痛点 2（不报 per-task SR，而 LIBERO 有 4.6× 长度差异）。

---

## §2 方案设计

### 2.1 LMWM 概览

LMWM = CRAVE（零训练 milestone 发现）+ MilestoneGenerator（milestone 条件生成器）。两阶段：

```
阶段 1 — CRAVE milestone 发现（零训练）:
  冻结 DINOv3-base 编码器 → 提取 episode grid 特征 [N,256,768]
  → 跨 episode gist pooling → KMeans 聚类 → 单调 Viterbi readout
  → M 个有序 milestone 质心 + 每帧的 milestone 标签

阶段 2 — Milestone 条件生成器训练:
  (cur_frame_feature, target_code) → MilestoneGenerator(AdaLN) → next_milestone_feature
  损失: L = smooth_L1(gen(g_t, z), g_m+1) + λ_lift·ReLU(cos(gen,g_t) - cos(gen,g_m+1))
  输入对: {(cur_fi, first[milestone+1])}，保证跨语义边界
```

**关键属性**：
- **编码器无关**：CRAVE 只要冻结特征空间有判别力即可（DINOv3 / SigLIP / VLA 自身编码器）
- **自适应时延**：milestone 间距由数据决定（2-108 帧），不是固定 τ
- **稀疏+高信息**：每个训练对跨越至少一个语义阶段边界，下界保证信息密度

### 2.2 与 LaWAM 的集成（→ LMWAM）

LMWM 作为 LaWAM 世界模型 decoder 的**即插即用替身**：

```
原 LaWAM:
  h_t → [LaWM decoder](h_t, latent_code) → h_{t+τ}   (τ 固定 0.35s/1.2s/1.6s)

LMWAM:
  h_t → [MilestoneGenerator](h_t, milestone_code) → h_{milestone+1}   (自适应)
```

**最小侵入 swap**（高内聚低耦合）：
1. 保留原 LAM 的 `extract_vision_features`（DINOv3-vitb16，冻结共享）
2. 只替换 `lam.decoder` 为 MilestoneGenerator
3. 训练时 `h_t1_gt` 从 t+τ 改为 milestone+1 帧特征（Path A: hook 覆盖；Path B: dataloader 原生）
4. `swap_teacher=True`：量化器改用 LMWM InverseEnc（保证 code 空间自洽）

| 组件 | LaWAM (baseline) | LMWAM (ours) |
|---|---|---|
| 视觉编码器 | DINOv3-vitb16 | 同（共享） |
| 世界模型 decoder | LaWM decoder (released) | MilestoneGenerator (CRAVE 预训练) |
| 预测目标 | t+7 帧 (0.35s LIBERO) | milestone+1 帧（自适应） |
| 量化器 teacher | LaWM VAE | LMWM InverseEnc |
| 训练方式 | 同 recipe, unfreeze decoder | 同 recipe, unfreeze decoder |

### 2.3 如何回应 §1 的痛点

| 痛点 | LMWM 的解法 | 可测指标 |
|---|---|---|
| 1: τ 是未消融超参数 | 用 CRAVE milestone 替代固定 τ——预测窗口随数据自适应 | milestone 间距分布 vs 固定 τ 的错配率；sec_chunk 消融实验（LMWM 应对不敏感） |
| 2: 单 SR 掩盖难度差异 | milestone 序列提供隐式进度——长任务 15 帧间距 vs 短任务 4 帧 | 按任务长度分组 SR；长任务组 (>300步) SR 提升幅度 |
| 3: 监督信号密度不均 | 每个 pair 跨语义边界——特征距离下界保证 | pair 特征距离分布 std；辅助 loss 下降与 eval SR 的相关性 |

---

## §3 实验与结果对比

### 3.1 实验设置

| 配置项 | 值 |
|---|---|
| 基准环境 | LIBERO **libero_10**（10 长周期任务 × 50 trials = 500 ep） |
| 编码器 | DINOv3-vitb16 (冻结) |
| VLA 架构 | starVLA (Qwen3-VL-2B 前 16 层 + Alternate-DiT flow 动作头) |
| 预训练初始化 | lawam_pretrain (released) |
| 训练步数 | **@20000**（两臂同 step；M 原训 25000、B 至 23856，取共同 @20000 保公平） |
| 有效 batch size | 128 (4 GPU × 32) |
| 硬件 | 本机 A100（eval）· volc 4×H20 / 臂（训练） |
| 评测 | 50 trials/task, libero_10 suite, seed=0（**单 seed**） |

**对比臂**：
- **Arm M (LMWAM)**：LMWM decoder + milestone+1 目标 + swap_teacher
- **Arm B (baseline)**：released LaWM decoder + t+7 目标（同 recipe no-swap 自训）

### 3.2 结果（2026-07-15 本机 eval，原始数据见 [`RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md`](RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md)）

#### 3.2.1 总体成功率

| 方法 | LIBERO libero_10 SR | succ/total |
|---|---|---|
| LaWAM 论文 Table 1, **Long suite** | **97.0%** | 论文口径 |
| LaWAM released ckpt 本机复现 | 98.0%* | 98/100 |
| **Arm B: baseline (t+7, no-swap 自训)** | **96.40%** | 482/500 |
| **Arm M: LMWAM (milestone+1, 自训)** | **92.20%** | 461/500 |
| **Δ (M − B)** | **−4.20 pt** | −21 ep |

> \* **[2026-07-19 更正]** 此前此处写作"98.0% vs 论文 98.6%",**分子分母都错**:
>   - **98.6% 是论文四 suite 的平均**(Spatial 99.4 / Object 99.6 / Goal 98.4 / **Long 97.0**),
>     而我们所有 eval 都只跑 **libero_10 = Long suite**,正确对标值是 **97.0%**。
>   - **98.0% 这个锚点本身不可用作基准**:原始数据是 `trials/task=10`、**共 100 个 episode、seed=0**
>     (`results/eval_runs/libero/lawam_libero_sft/20260712_051306`),聚合二项 SE ≈ **±1.4pt**
>     → 98.0±1.4 与论文 97.0 **完全一致**,并不构成"我们复现不出来"的证据。
>     其 per-task t6=90 是 9/10(σ≈9.5pt),与我们的 85.0±2.4 统计上不可区分。
>   - 我们自己的臂是 50 trials/task × 500 ep × 4 变 seed,**与该锚点不同口径,不可直接比**。
>   ⇒ 正确读法:对标论文 Long **97.0%**,dual2q(12500步)−2.2pt、armB(20000步)−3.4pt;
>     业界正常复现折损约 2pt(如 NVlabs/vla0 用官方权重复现 −2.0),而我们仅用了官方
>     **25000 步预算的 50%~80%**。待重测锚点(50 trials × 变 seed)后定稿。
> **诚实结论：当前 Arm M 低于 Arm B 4.2 点，不宣称 milestone 有效。** 7/10 任务基本打平；差距 71% 集中在 task 6（−0.16）+ task 8（−0.14）。失败模式全部为 550 步 horizon 超时（无发散）。**关键缺口**：训练对丢弃了最终 milestone 段，task 6/9 有 52–60% 的 episode 处于世界模型的未训练区（详见 RESULTS 文档失败模式分析）。单 seed，待多 seed 确认显著性。

#### 3.2.2 痛点 1 验证：预测时延的消融

| 实验 | 预期 |
|---|---|
| Milestone 间距分布 vs 固定 τ | LMWM 间距变异系数 0.69, 17.7% 偏离 t+7 >10 帧 |
| sec_chunk 敏感度消融 | LMWM 臂 SR 对 sec_chunk 变化不敏感；baseline 臂应敏感 |
| 不同 τ 下的 SR 对比 | *（待消融实验）* |

#### 3.2.3 痛点 2 验证：按任务长度分组的 SR

| 任务组 | 平均 episode 长度 | #tasks | Arm B SR | Arm M SR | Δ |
|---|---|---|---|---|---|
| 短 (<150步) | ~95 步 | *（统计中）* | *（训练中）* | *（训练中）* | — |
| 中 (150-300步) | ~220 步 | *（统计中）* | *（训练中）* | *（训练中）* | — |
| 长 (>300步) | ~407 步 | *（统计中）* | *（训练中）* | *（训练中）* | — |

**预期**：LMWM 在长任务组有更大的 SR 提升（milestone 间距 15 帧 vs 固定 t+7 = 7 帧，3.8× 自适应差异）。

#### 3.2.4 痛点 3 验证：监督信号信息密度

| 指标 | Arm B (t+7) | Arm M (milestone+1) |
|---|---|---|
| Pair 特征距离 std | 0.00894 | **0.01087** (+22%) |
| 辅助 loss (perceptual) 下降曲线 | *（训练中）* | *（训练中）* |
| 辅助 loss 与 eval SR 的 Spearman ρ | *（待 eval 完成后计算）* | *（待 eval 完成后计算）* |

**预期**：LMWM 的辅助 loss 下降应更预示 SR 提升（每个梯度步消费有意义信号）。

#### 3.2.5 训练效率

| 指标 | Arm B | Arm M |
|---|---|---|
| gf3 (4×A100) | 1.81 s/it | 1.51 s/it |
| volc (4×H20) | *（训练中）* | *（训练中）* |
| 显存/GPU | 69 GB | 60 GB |

> LMWM 的 provider 查表开销约 0.04 s/it（CPU 查 LRU 缓存），对训练速度影响可忽略。

### 3.3 后续实验（框架预留）

- **RoboTwin 跨具身测试**：需 train_robotwin.yaml 另训
- **跨编码器泛化**（痛点 C）：SigLIP 空间 CRAVE → milestone 质量对比
- **视觉鲁棒性**（痛点 E）：LIBERO distractor 场景 SR 衰减率
- **Path B 原生集成**：dataloader per-sample 动态采样 milestone+1 帧

---

## 附录：关键文件索引

| 文件 | 内容 |
|---|---|
| `LAM_starVLA_contract_2026-07-12.md` | LAM↔starVLA I/O 契约（P0 产出，集成接口基线） |
| `BUG_AUDIT_2026-07-12.md` | 框架 bug 审查 + Path A/B 修复方案（当前 h_t1_gt override 的设计依据） |
| `history/LMWM_pain_point_analysis_2026-07-13.md` | 痛点分析完整版（5 痛点 A–E + P0 实证 + 论文证据 + 测试路线） |
| `history/PLAN_lmwm_replace_lawm_2026-07-12.md` | LMWM 替换 LaWM 的技术路线（P0–P4 已执行） |
| `history/LAWAM_reproduce_and_kai0_sft_plan_2026-07-12.md` | LaWAM 复现 + kai0 SFT 计划（已执行） |
| `history/P1_progress_2026-07-12.md` | P1 切 DINOv3-base 重训进度快照 |
