# LMWM: Latent Milestone World Model for VLA

> 初稿框架。§1 痛点（已验证）、§2 方案设计、§3 实验与结果对比。
> 数据来源：LIBERO 40 任务 × DINOv3-base 特征空间。对比基线：LaWAM（arXiv 2606.15768）released checkpoint + 同管线 no-swap 自训。

---

## §1 痛点与动机

*（详见 `LMWM_report_sec1_pain_points.md`，此处摘要）*

| # | 痛点 | 本地证据 | 论文证据 | 强度 |
|---|---|---|---|---|
| 1 | 固定预测时延 τ 是未消融隐式超参数 | P0-B: 跨任务 CV=0.69, 17.7% 对偏离 t+7 >10帧 | LaWAM 无 τ 消融 + 多域手动选不同 τ | **强** |
| 2 | 单 aggregate SR 掩盖任务间难度差异 | 4.6× 任务长度差异, 长任务 ms 间距 3.8× | LaWAM 不报 per-task SR | **强** |
| 3 | 世界模型监督信号密度不均（自校准效应） | P0-A: LMWM 特征距离 std 比 t+7 大 22% | 间接（无直接论文指标） | 中等 |

**核心 claim**：现有 VLA 世界模型用固定时延预测未来帧特征，这个 τ 是一个**从未被消融但实际影响跨域部署的隐式超参数**。LMWM 用 CRAVE 无监督发现的 milestone 边界替代 τ，使预测目标随任务节奏自适应。

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
| 基准环境 | LIBERO (libero_10, 40 任务) |
| 编码器 | DINOv3-vitb16 (冻结) |
| VLA 架构 | starVLA (Qwen3-VL-2B 前 16 层 + Alternate-DiT flow 动作头) |
| 预训练初始化 | lawam_pretrain (released) |
| 训练步数 | 25000 |
| 有效 batch size | 128 (4 GPU × 32) |
| 硬件 (gf3) | 4×A100-80GB / 臂 |
| 硬件 (volc) | 4×H20-96GB / 臂 |
| 评测 | 50 trials/task, libero_10 suite |

**对比臂**：
- **Arm M (LMWAM)**：LMWM decoder + milestone+1 目标 + swap_teacher
- **Arm B (baseline)**：released LaWM decoder + t+7 目标（同 recipe 自训）

### 3.2 结果框架（待训练完成后填入）

#### 3.2.1 总体成功率

| 方法 | LIBERO libero_10 SR (%) |
|---|---|
| LaWAM (released ckpt) | 98.0* |
| Arm B: baseline (no-swap 自训) | *（训练中）* |
| Arm M: LMWAM (milestone, 自训) | *（训练中）* |

> * 98.0% 为本机复现值（vs 论文 98.6%）。

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
| `LMWM_report_sec1_pain_points.md` | §1 痛点详细版（含完整数据表格 + 论文引用） |
| `LMWM_pain_point_analysis_2026-07-13.md` | 痛点分析原始文档（5 痛点 + P0 实证 + 论文证据） |
| `BUG_AUDIT_2026-07-12.md` | 框架 bug 审查 + Path A/B 修复方案 |
| `PLAN_lmwm_replace_lawm_2026-07-12.md` | LMWM 替换 LaWM 的技术路线 |
| `LAWAM_reproduce_and_kai0_sft_plan_2026-07-12.md` | LaWAM 复现 + kai0 SFT 计划 |
