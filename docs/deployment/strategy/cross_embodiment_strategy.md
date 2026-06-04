# Cross-Embodiment 战略与数据分析

> **范围**: **专门探索如何在 pi05 架构上充分利用跨本体数据集** —— 让 pi05 学到 KAI0 官方数据的复杂操作知识, 同时保留 vis 数据对部署本体动作的良好适配。含 embodiment gap 实测、norm-stats/operator/混训分析、知识-本体解耦训练路线、Action Head Cond (Track C)、决策点。
> **不含**: Track X (X-VLA 官方架构, Florence2 + SoftPromptedTransformer) 已**分离**到 [`../../training/future_plans/plans/xvla_track_x_curriculum.md`](../../training/future_plans/plans/xvla_track_x_curriculum.md), 本文档不再涵盖。
> **状态**: 持续更新 (此文档替代原 `cross_embodiment_data_reuse_plan.md`, 抽离了状态表与具体训练计划)。
> **最近更新**: 2026-06-04 (**重定向**: 停 Hard/Soft prompt, 主线改为"pi05 上知识-本体解耦"; Track X 分离)。

---

## 0. 战略定位与核心问题 (2026-06-04 重定向)

### 0.1 已停止的方向 (prompt-based conditioning)
- **Hard Prompt** (`"[D405 wrist] ..."` 前缀): 之前已试, **真机效果微乎其微** (信号沿 LLM attention 隐式传播, 不显式 gate) → **停止**。
- **Soft Prompt** (X-VLA 式 VLM-input learnable soft token 移植到 pi05): **停止** (Track B 早已废弃, soft-prompt 实验线整体关闭)。
- → 下文 §5.2 Hard/Soft 行、§7 原 tri-track 的 Track X 部分、以及相关 plan 规划**均作废**。
- ⚠️ **Action Head Cond (Track C, action expert 端 domain token) 不在停止之列** —— 它是可行方案之一 (见 §5.3 / R3)。

### 0.2 ⭐ 核心问题 (本文档主线)
**如何在 pi05 模型基础上, 让模型同时:**
1. **学到 KAI0 官方数据中复杂的操作技能** —— 复杂多步叠衣、丰富抓取/折叠策略 (KAI0 base+dagger ~6.5k ep 的真正价值);
2. **在部署本体 (vis) 上正确感知与抓取** —— 关键瓶颈是 **vis 相机 (D405) 与 kai0 (D435) 图像不一致** (FOV 87° vs 69°、wrist flange 安装高度/角度、min depth 7 vs 28cm)。

> **⚠️ 2026-06-04 实测修正 (用户反馈, 推翻原"知识 vs 本体"框架)**:
> - **纯 kai0 数据训的模型在 vis 真机能跑通** —— 动作能执行 → **motor/关节映射 OK** (印证 §2.1 norm 仅差 0.5σ, 本体动作不是瓶颈);
> - **但抓取位置不精确 → 衣服无法进入下一阶段** (抓不准衣角);
> - **最大 setting 差异 = 摄像机图像不一致** (D435→D405 wrist)。
> → **真正瓶颈是「感知 (perception / 相机)」, 不是「动作 (motor / 本体)」。** 模型有 kai0 技能, 但用 D435 训出的视觉编码器去看 D405 图像 → 把衣角看错位置 → 抓偏。

> **本质 = 「技能 vs 感知」解耦** (修正版):
> - KAI0 的价值在**高层操作技能** (怎么抓、怎么多步折) —— 但其感知绑死在 **D435**。
> - vis 的价值在**部署相机 (D405) 的正确感知 + 抓取定位精度** —— 落在 **视觉编码器 / VLM 感知端**。
> - **绑定约束在视觉编码器, 不在 action expert。** 修复必须作用在**感知端** (适配 D405), 用 kai0 技能, 但让感知学 vis。

### 0.3 候选路线 (围绕「相机 / 感知 gap」)
| 路线 | 做法 | 针对 | 文档 |
|---|---|---|---|
| **P2 相机对齐 (诊断+快修)** ⭐先做 | 推理时把 D405 center-crop/resize 到 ~D435 FOV (或训练时反向模拟 D405); 先验证抓不准是否**纯 FOV/几何** | 隔离 appearance vs 视角/外参 gap | §5.6 |
| **P1 视觉增强 co-train** ⭐ | kai+vis 预合并 co-train + 激进 FOV/crop/color/distortion 增强 → 视觉编码器 camera-robust; vis 提供 D405 抓取监督 | 跨相机鲁棒感知 + 学 D405 抓取定位 | §5.6 |
| **P3 vis grasp DAgger** | 针对"抓不准"的具体 config 采 vis 示范, 直接监督 D405 下正确抓取定位 | 直接补抓取精度 | [dagger plan](../../training/future_plans/plans/dagger_validity_and_finetune_comparison.md) |
| **R1' 两阶段 (修正)** | co-train kai+vis → vis finetune, **finetune 解冻视觉/VLM 适配 D405 感知** (≠ 冻 VLM) | 末段把感知 snap 到 D405 | corrected_plan_a |
| **R0 SSL camera-invariant** | kai+vis(+xvla) visual SSL + view-conditioned token + RandomResizedCrop → camera-invariant 视觉特征 | 最 principled 的跨相机不变表示 | [`ssl_phase_pretrain_pipeline.md`](../../training/future_plans/plans/ssl_phase_pretrain_pipeline.md) |
| ~~R2 模块冻结 (冻 VLM 训 action expert)~~ | — | ❌ **已试, 不理想** —— 冻的正是要改的**感知层 (VLM)**, 抓取仍不准 (见 §5.4) | — |
| R3 Action Head Cond / R4 数据筛选 | (motor 端 / 数据侧) | 对相机感知 gap **无直接帮助**, 降为可选叠加 | §5.3/5.5 |

> **关键转向 (2026-06-04)**: 瓶颈是相机/感知 → 修复作用在**视觉编码器/感知端**, 而非 motor。**路径: 先 P2 诊断 (便宜, 看是不是纯 FOV) → 据结果上 P1 (视觉增强 co-train) 或 P3 (vis grasp DAgger)**; R0 SSL 是长线解。R2 (冻 VLM) 已证此路不通 (冻错了层)。

---

## 1. Embodiment Gap — 3 异构机器人定义

### 1.1 三机器人对比表

| 维度 | A: KAI0 (官方) | **B: vis (⭐ 部署目标)** | C: XVLA-Soft-Fold (第三方) |
|---|---|---|---|
| 机械臂型号 | piper 双臂 | piper 双臂 (同型) | piper 双臂 (待确认型号) |
| Joint DOF | 14 (7×2) | 同 | 同 (假设) |
| 控制频率 | 30 Hz | 同 | 待确认 |
| **Wrist 相机** | D435 (FOV 69°×42°, rolling, min depth 28cm) | **D405** (FOV 87°×58°, global, min depth 7cm) ⭐ | 待确认 hdf5 元数据 |
| **Wrist 安装** | 旧 flange | 新 flange (高度/角度略差) | 第三方采集, 不同 setup |
| **双臂间距** | 标准 | 略差 (毫米级) | 待确认 |
| **Top 头部相机** | D435 (76cm 高, 30° 角) | 同 (略差 <5cm/5°) | 待确认 |
| Action 语义 | "Flatten and fold the cloth." | 同 | 同 (cloth fold) |
| **Episodes** | 6,512 | **895** (B 数据量少 → 关键瓶颈) | 1,729 |
| **数据格式** | LeRobot v2.1 (14D joint) | LeRobot v2.1 (14D joint) | hdf5 (EE6D? 待确认) |
| **部署?** | ❌ 不部署 | ✅ **唯一部署目标** | ❌ 不部署 |
| domain_id (Track X) | 0 | **1** | 2 |

### 1.2 与 B 部署目标的 gap 量化

A vs B 关键 wrist gap (§2 实测):
- **R 腕 yaw+roll paired shift ~19°** (SE3 复合旋转) — 部署 B 时 wrist 视野 OOD
- L2 mean diff 0.47 rad (跨 robot effect, 剔除 operator confound 后)
- 13/14 dim 在 ±1σ 内 (大部分可被 per-dataset norm 吸收)

C (XVLA-Soft-Fold) vs B gap **待量化** (等格式适配后做 norm_stats 对比, 预期 wrist 相机差异最大)。

### 1.3 实测真机症状

1. **Cloth loop** (复杂场景): mixed_1 baseline (纯 A 训练) 部署 B 出现循环卡死 — D435→D405 视觉 OOD 累积漂移
2. **空桌面抖动**: vis SFT 后 prior 被高 jump 帧拉宽, 空桌面 condition 弱 → 抽到大 action
3. **混训抖动 > 纯 B**: 早期 naive 混训 (`mixed_pure2_1800_6000`) 真机抖 > `pure_1200_new_norm` → naive 混训创造双模式策略, chunk 间切换抖

---

## 2. 实测 Norm-stats / Operator / 混训 — 数据集对比 (2026-05-21)

### 2.1 KAI0 ↔ vis Norm-stats Δmean (joint angles, 14D)

直接从原始 parquet 重算 (kai0_base 102/3055 ep × 114k frames, vis_v2_merged 112/895 ep × 133k frames)。

| dim | label | A.mean | B.mean | Δmean (A−B) | Δ角度 | |Δ|/A.σ |
|---:|---|---:|---:|---:|---:|---:|
| 0 | L_肩 yaw | -0.062 | -0.079 | +0.017 | 1.0° | 0.09σ |
| 1 | L_肩 pit | +1.547 | +1.368 | +0.179 | 10.2° | 0.33σ |
| 2 | L_肘 | -1.301 | -1.200 | -0.100 | 5.7° | 0.22σ |
| 3 | L_腕 yaw | -0.095 | -0.159 | +0.064 | 3.7° | 0.22σ |
| 4 | L_腕 pit | +0.796 | +0.719 | +0.077 | 4.4° | 0.32σ |
| 5 | L_腕 rol | +0.031 | +0.158 | -0.127 | 7.3° | 0.45σ |
| 6 | L_grip | +0.028 | +0.029 | -0.001 | — | 0.03σ |
| 7 | R_肩 yaw | +0.115 | -0.006 | +0.121 | 6.9° | 0.71σ |
| 8 | R_肩 pit | +1.486 | +1.476 | +0.010 | 0.6° | 0.02σ |
| 9 | R_肘 | -1.461 | -1.284 | -0.177 | 10.1° | 0.33σ |
| **10** | **R_腕 yaw** ⚠️ | +0.048 | +0.341 | **-0.293** | **16.8°** | **1.05σ** |
| 11 | R_腕 pit | +0.918 | +0.899 | +0.019 | 1.1° | 0.08σ |
| **12** | **R_腕 rol** ⚠️ | +0.003 | -0.241 | **+0.244** | **14.0°** | **0.99σ** |
| 13 | R_grip | +0.035 | +0.021 | +0.013 | — | 0.41σ |

```
L1 norm: 1.44 rad   L2 norm: 0.51 rad   L∞: 0.293 rad (16.8° @ R_腕yaw)
Within ±1σ: 13/14   median 0.32σ        max 1.05σ
Per-arm L2:  Left 0.26    Right 0.44     → 右臂偏 1.67× 左臂
B/A motion range: median 1.07, mean 1.11, [0.88, 1.48]
```

### 2.2 关键发现 (4 条)

1. **整体分布高度重叠 but not identical**: 13/14 维落在 ±1σ 内, per-dataset norm 大部分可吸收。但 L2 = 0.51 rad 在 50-step chunk 上累积影响显著。
2. **右臂偏移 1.67× 大于左臂** — 双臂间距不同的直接证据。
3. **R 腕 yaw (-16.8°) + R 腕 roll (+14°) 是配对偏移** ⭐ (核心):
   - paired correlated shift, 不是独立
   - SE(3) 表示下合成 **~21° 复合旋转**
   - 物理意义: 右手 wrist 末端在 B 上比 A 整体旋转 21° → D405 wrist 视野下 cloth 出现 21° 旋转 OOD
4. **B 运动幅度比 A 大 10-30%** — 解释 "vis SFT 后 prior 被拉宽" 现象。

### 2.3 vis 内部 Operator 与时间漂移

| Group | Operator | Episodes | 占比 |
|---|---|---:|---:|
| G1 (主操作员, ztm+lym 同一人) | ztm 723 + lym 149 | 872 | 97.4% |
| G2 (助手) | gsy | 23 | 2.6% |

**G1 内时间漂移**: 同一 operator 跨 5 天 (4-24 vs 4-28) L2 drift = 0.47 rad, **与 cross-robot effect 同量级**。
**G2 影响微弱**: gsy 剔除后跨 robot L2 仅降 8.9% (0.51 → 0.47), R 腕 paired shift 仍 ~19°。结论: **不必 per-operator norm**, 真正的 cross-robot geometric effect 真实存在。

### 2.4 混训策略 6 方案对比

| 方案 | 描述 | R 腕 19° | 时间 drift | motion range | 工程量 |
|---|---|:-:|:-:|:-:|:-:|
| A. Naive joint norm | 合并算单一 norm_stats | ❌ | ❌ | ❌ | 0.5 day |
| **B. Per-dataset norm + Single model** | 每数据集 own norm, 同模型 | ⚠️ 90.7% | ⚠️ 90.7% | ✅ | 1 day |
| ~~C. Soft Prompt + Per-DS norm~~ **已停** | 显式 routing | — | — | — | — (§0.1; domain 区分改 R3 Action Head Cond) |
| D. Curriculum (A pretrain → B finetune) | 现有 mixed_1 → smooth_800 | ✅ B finetune | ✅ | ✅ | 1 day |
| E. SSL Decoupled | A 进 visual SSL, B 进 policy | 不参与 | 不参与 | 不参与 | 9 week |
| F. EE-based action | delta EE pose | ✅ **天然消除** | ⚠️ | ⚠️ | 3 day |

**MMD 实测**: per-dataset norm 后 MMD(A_norm, B_norm) = 0.00558 vs raw 0.0597 (降 90.7%), 但仍 28× self-baseline (残余高阶矩 + joint synergy 差异不可消)。

**关键 Insight**: 早期 `mixed_pure2_1800_6000` 失败的真因可能不是"混训不能", 而是用了 naive joint norm (方案 A) 而非 per-dataset norm (方案 B)。

**推荐 Layered Combination**:
- L1 Visual: 方案 E (SSL decoupled, A+B+C all in, no action loss)
- L2 Policy: 方案 D (curriculum / 两阶段, = R1/R2) + 方案 B (per-DS norm)。~~方案 C (Soft Prompt)~~ 已停 (§0.1); domain 区分改用 R3 Action Head Cond (可选)
- L3 Ablation: 方案 F (EE-based, paper 对照)

---

## 3. 4-层 ROI 战略框架

**判断**: 某 loss / objective 是否依赖 A 和 B 的 action space 对齐?

| Layer | 内容 | 依赖 action 对齐? | A 价值 | 工程复杂度 |
|---|---|---|---|---|
| **L1: Visual SSL / World Model** ⭐ | V-JEPA + point track + flow + dynamics | ❌ 不依赖 | **全功率可用** | 高 |
| **L2: Embodiment-cond Policy** | A+B 共训, embedding 区分 | ⚠️ 弱依赖 (需 conditioning) | 可用, 需对齐 | 中 |
| **L3: Auxiliary tasks** | Inverse dynamics, future frame pred | ⚠️ 部分依赖 | 可用, 不入主 loss | 中 |
| **L4: Data engine / Sim2Real** | Retargeting, replay-augmentation | ✅ 强依赖 | 低 (需高保真 retarget) | 高 |

> **核心原则**: A 的价值不在"直接帮 B 做 task", 而在 **representation / dynamics / prior** 这些更上游的层次。

---

## 4. 假说矩阵 (H1-H4)

| ID | 假说 | 关键实验 | 成功标准 |
|---|---|---|---|
| **H1** | SSL pretrain on A+XVLA+B 提供的 visual repr 在 cloth 任务上 > π0.5 default | E3.1 vs E3.0 | B finetune val MAE 降 ≥10%, 真机抖动减少 |
| **H2** | Multi-objective (V-JEPA + track + flow + xview) > 单 V-JEPA | E1.5 vs E1.1 | Downstream val MAE 进一步降低 |
| **H3** | Embodiment-conditioned dynamics 让 A 的物理 prior 不污染 B policy | E3.3 vs E3.2 | 真机平滑度 + 复杂场景成功率提升 |
| **H4** | Motion-residual: cloth_residual 部分 embodiment-invariant | E2.3 latent 分析 | A vs B cloth_residual latent MMD < 0.1 |

---

## 5. Action 表示与 Conditioning 选项

### 5.1 Action 表示决策

| 模型 | 默认 Action |
|---|---|
| π0 (老) | Delta (relative to chunk start) |
| **π0.5** | **Absolute** (默认), 可选 relative |
| OpenPI Aloha | DeltaActions transform (`use_delta_joint_actions=True`) |
| 本地 `mixed_1` ckpt | **Absolute** (norm_stats 实测: mean[1]=1.48, std=0.63) |
| KAI0 raw 数据 | **Absolute** (joint angles ±π) |
| X-VLA | EE6D absolute pose (20D = xyz+Rot6D+grip per arm) |

**实证 ([Demystifying Action Space Design 2602.23408](https://arxiv.org/abs/2602.23408))**: 单机器人/单任务/long-horizon → absolute 更稳; 多 embodiment/跨设备 → delta 更稳; 混合 mask (joint delta + gripper absolute) 是 pragmatic 选择。

### 5.2 Embodiment Conditioning 选项

| 方式 | 实现复杂度 | 本地代码状态 | 推荐度 |
|---|---|---|---|
| ~~Hard prompt~~ | 极低 | ✅ | ❌ **停止 (2026-06-04)** — 真机效果微乎其微 |
| ~~Soft prompt (VLM-input soft token, 移植 pi05)~~ | 低 | — | ❌ **停止 (2026-06-04)** — soft-prompt 实验线关闭 (官方 soft prompt 留在 Track X 文档, 与本 pi05 文档分离) |
| **Action Head Cond (Track C 方案 A)** | 极低 (action expert input concat 1 domain token) | ✅ 已实现 (`pi0.py:action_head_cond_hub`, dataset_id 透传已修) | ⭐⭐⭐ **保留** — pi05 上唯一在用的 conditioning (= R3) |

> **pi05 上 conditioning 只剩 Action Head Cond (Track C)**: 在 **action expert 输入端** concat 1 个 domain token (paligemma 不知 domain), 推理固定 vis token → 拆 kai/vis 双峰、走 vis 模式。⚠️ 历史 Track C 训练全 collapse 是因为走了 broken 的 `datasets_yaml` 复制路径, **不是 conditioning 本身失败** —— 必须走预合并单源路径重试, 见 [`corrected_plan_a_conditioning_premerge.md`](../../training/future_plans/plans/corrected_plan_a_conditioning_premerge.md) 与 [`../../training/history/experiments/conditioning_vs_action_representation_ablation.md`](../../training/history/experiments/conditioning_vs_action_representation_ablation.md) §4.4。

### 5.3 Action Head Cond (Track C, 方案 A) — 设计要点

**动机**: 在 **action expert 输入端** concat 1 个 domain token (paligemma 完全不知 domain), 让 action expert 按 domain 区分 denoise → 推理固定 vis token, 拆掉 kai/vis 双峰、走 vis 模式。这正好服务"本体层只认 vis"的目标 (§0.2)。

```
方案 A (Track C):
  d → action_head_cond_hub[d] (B,1,1024) → 拼到 action expert input (与 noise_action_token 同级)
    → action expert self-attn (4-8 层) → action  [paligemma 完全不知 domain]
```

**与 R1/R2 的关系**: R3 (本节) 用**显式 token** 在 motor 端区分 domain; R2 (§5.4) 用**冻结调度**在 motor 端只喂 vis 数据。两者都让 action expert 偏向 vis 本体, 可叠加 (token + 末段 vis-only finetune)。

> ⚠️ **训练路径必须改**: 历史 Track C "单阶段 joint balanced (vis ×7) via `datasets_yaml`" **全 collapse** —— 真因是 `datasets_yaml`/ConcatDataset 代码路径本身 broken (**不是 conditioning 失败**, 见 [`conditioning_vs_action_representation_ablation.md`](../../training/history/experiments/conditioning_vs_action_representation_ablation.md) §4.4)。**必须改走物理预合并单源路径** (corrected Plan A): domain_id 逐帧带入, 推理固定 vis。

**代码改造点**:
- `pi0_config.py`: 加 `action_head_cond_num_domains: int = 0`
- `pi0.py.__init__`: 加 `self.action_head_cond_hub = nnx.Embed(num_domains, action_expert_width)` (init N(0, 0.02))
- `pi0.py` action expert forward: `domain_token = self.action_head_cond_hub(obs.dataset_id)[:, None, :]; action_input = jnp.concat([domain_token, action_input], axis=1)`
- `transforms.py`: dataset_id 透传 ✓ (已修)

---

### 5.4 R2 — 模块冻结解耦 (冻 VLM 训 action expert) ❌ 已试, 不理想

> ⛔ **2026-06-04 实测结论 (用户反馈)**: R2 **不解决问题**。原假设"gap 在 motor/本体"被推翻 (§0.2): 纯 kai0 模型在 vis 能跑通 (motor OK), 真瓶颈是**相机/感知** (抓取不准)。R2 **冻住 VLM** —— 而 VLM 正是承载感知、需要适配 D405 的那一层 —— 所以冻错了层, 抓取依然不准。**保留本节作记录, 修复转向 §5.6 感知端。**

**(原思路, 已证不适用)**: pi05 = VLM backbone (paligemma, 感知/语义/任务知识) + action expert (flow-matching, 本体动作映射)。把两者**分阶段、分数据**训练:

| Stage | 数据 | 训练谁 | 冻结谁 | 作用 |
|---|---|---|---|---|
| **S1 知识注入** | kai+vis 预合并 (vis 加权) | VLM (+ 可选 action expert) | — | VLM 吸收 **kai0 复杂任务 + vis** 的视觉/语义知识 |
| **S2 本体对齐** | **vis-only** (轻量, 低 lr) | **仅 action expert** | **冻 VLM** | action expert re-snap 到 vis 本体动作分布; **kai 的 action 模式从 motor 端被擦掉**, 而 VLM 里的 kai 知识被冻结保留 |

> **为什么有效**: S1 让 kai 知识进了 VLM (representation); S2 冻住 VLM (锁住知识) 只在 vis 上重训 action expert → 输出动作纯 vis 本体, **不带 kai 的 21° 腕姿双峰** → 真机不抖。这正是 §0.2「知识层吃 kai+vis, 本体层只认 vis」的直接实现, 也是官方 Knowledge Insulation 的精神 (梯度/参数隔离 motor 与 knowledge)。

**实现** (openpi 已有机制):
- `TrainConfig.freeze_filter` (nnx filter) —— S2 设为"冻结除 action expert 外全部" (类似已有 `freeze_filter=nnx.Not(nnx_utils.PathRegex(".*action_head_cond_hub.*"))` 的写法, 改成匹配 action expert 参数路径)。⚠️ **需核对 pi0.py 里 action expert 的参数 path 命名** 写对 PathRegex。
- 或用 lerobot pi05 的 `train_expert_only=true` (冻 VLM 只训 action expert + projection) 做 S2。
- S1 数据走**预合并单源**路径 (不走 broken datasets_yaml, 见 R1/corrected Plan A)。

**与 R1 区别**: R1 的 S2 是"全参数轻量 vis finetune"; R2 的 S2 是"**只**训 action expert + 冻 VLM" —— R2 更外科, 显式保护 VLM 里的 kai 知识不被 vis finetune 冲掉。**建议先跑 R2**。

> 📋 待建 plan: `docs/training/future_plans/plans/` 下补一个 "pi05 module-decoupled kai-knowledge + vis-body" 实验 plan (S1/S2 超参 + freeze_filter 核对 + 对照 R1 全参 finetune + vis-only baseline + 真机)。

### 5.5 R4 — 数据筛选 (降 action 冲突, 保留新知识)

kai0 6.5k ep 里, 与 vis 重叠的**简单动作** (平移、对折) 是 action 双峰冲突的主要来源, 而**复杂多步操作 / 罕见场景**才是 vis 缺的新知识。

- **做法**: 对 kai0 episode 按"复杂度 / 与 vis 分布差异"打分, 只取**高复杂度 + vis 覆盖不足**的子集进 co-train; 丢弃与 vis 高度重复的简单 ep。
- **收益**: 同样吸收 kai 复杂知识, 但减少同观测下的 action 模式冲突 → 降低对 S2/conditioning 的依赖。
- **可与 R1/R2/R3 任意叠加**。低成本预处理, 不改模型。
- ⚠️ 复杂度打分方式 (action 方差 / 轨迹长度 / 任务阶段数 / 视觉新颖度) 待定, 需小实验标定。

---

### 5.6 ⭐ 相机 / 感知 gap 修复路线 (P1-P3, 2026-06-04 新主线)

> 真瓶颈 = D435→D405 wrist 相机不一致致**抓取定位不准** (§0.2)。修复作用在**视觉编码器 / 感知端**。

**P2 — 相机对齐 (先做, 诊断 + 可能快修, 最便宜)**
- D405 (FOV 87°, min 7cm) 比 D435 (69°, 28cm) **更宽更近**。推理时把 D405 wrist 图 **center-crop + resize 到 ~D435 FOV**, 让模型看到"D435-like"视野 → 若抓取立刻变准, 说明 gap 主要是 **FOV/几何**, 一个固定 crop 即低成本解。
- ⚠️ 历史 §8 风险3 曾否决"D405 crop 到 D435 FOV"(理由: 丢周边信息 + 双向维护)。但作为**诊断 + 抓取阶段的快修**值得重测 (中央视野对抓取够用)。
- **若 crop 后仍不准** → gap 不只是 FOV, 含**视角/外参** (flange 高度角度差) 或 **appearance** → 必须走 P1/P3 用 vis 数据学。

**P1 — 视觉增强 co-train (主力)**
- kai+vis **预合并 co-train** (绕开 datasets_yaml), 对图像加**激进增强**: RandomResizedCrop (scale 0.5-1.0 模拟 FOV 差)、color/exposure jitter、轻微透视/distortion → 逼视觉编码器学**跨相机鲁棒**特征。
- vis (D405) 样本提供**部署相机的抓取定位监督**; kai0 提供技能。增强让 kai0 的 D435 感知不至于 overfit 到 D435。
- 关键: 这是让**感知端**吃 kai+vis (与 R2 相反 —— R2 冻感知, P1 主动训感知)。

**P3 — vis grasp DAgger (直接补精度)**
- 部署中"抓不准衣角"的具体场景, 采 vis (D405) 纠错示范, 直接监督正确抓取定位 → 最直接补 D405 感知-抓取闭环。
- 与现有 dagger 流程复用 (`dagger_validity_and_finetune_comparison.md`), 但目标聚焦**抓取精度** (非动作平滑)。

**建议执行序**: **P2 诊断 (1 天)** → 若纯 FOV 则 P2 即解; 否则 **P1 视觉增强 co-train** 为主 + **P3 vis grasp DAgger** 补具体失败点; **R0 SSL camera-invariant** 作长线 principled 解。R3/R4 (motor/数据侧) 对感知 gap 无直接帮助。

---

## 6. RTC / TAC — Action Chunking 实时性方案对比 (2026-05-22)

### 6.1 三论文核心对比

| 论文 | 时间 | 路线 | 改 base 模型? | 推理 latency | 真机验证 |
|---|---|---|:-:|:-:|:-:|
| **Inference RTC** ([2506.07339](https://arxiv.org/abs/2506.07339)) | 2025-06 | 推理时 inpainting + vjp guidance | ❌ | **+28%** (97 vs 76 ms) | ✅ π0.5 6 task × 480 ep |
| **TAC** ⭐ ([2512.05964](https://arxiv.org/abs/2512.05964)) | 2025-12 | **训练时** prefix actions 作 ground-truth | ❌ (改 loss + adaLN per-token) | **0** (与 baseline 持平) | ✅ π0.6 box/espresso |
| **A2C2** ([2509.23224](https://arxiv.org/abs/2509.23224)) | 2025-09 | lightweight correction head, 每步 Δa | ❌ (base frozen, +新 module) | +4.7ms (~6%) | ❌ 仅 sim |

**三者正交可叠加**, 各解决不同子问题:
- Inference RTC = pseudo-inverse 强行约束 (老 ckpt 补救)
- **TAC** = 模型自己学会 chunk overlap (训练时一次, 推理零开销)
- A2C2 = 实时反应模块 (cloth dynamic state 时强相关)

### 6.2 本地实现状态

| 项 | 文件 | 状态 |
|---|---|---|
| **Inference RTC** | `kai0/src/openpi/models/pi0_rtc.py` | ✅ 完整 1:1 复刻 (`get_prefix_weights` 4 schedules, `jax.vjp` guidance, `min(c·inv_r2, max_gw)`) |
| **TAC training** | — | ✅ 已集成 (`pi0_rtc.py::compute_loss`, `Pi0Config.tac_enabled`, adaLN per-token, 0 新参数) — vis_v2_full v4 16gpu (hdv82) 已成功 running |
| **A2C2 correction head** | — | ❌ 未实现 (搁置) |

### 6.3 TAC Algorithm (Algorithm 1, JAX)

```python
def compute_loss(rng, obs, actions, *, max_delay=10):
    b, ah, ad = actions.shape
    noise_rng, time_rng, delay_rng = jax.random.split(rng, 3)
    time  = jax.random.uniform(time_rng, (b,))
    noise = jax.random.normal(noise_rng, (b, ah, ad))

    # TAC 新增 4 行:
    delay        = jax.random.randint(delay_rng, (b,), 0, max_delay)
    prefix_mask  = jnp.arange(ah)[None, :] < delay[:, None]
    time         = jnp.where(prefix_mask, 1.0, time[:, None])   # per-token time
    postfix_mask = jnp.logical_not(prefix_mask)[:, :, None]

    x_t = time[:, :, None] * actions + (1 - time[:, :, None]) * noise
    v_t = model(obs, x_t, time)
    loss = (v_t - (noise - actions)) ** 2
    return jnp.sum(loss * postfix_mask) / (jnp.sum(postfix_mask) + 1e-8)
```

**Architecture**: `Pi0Config.tac_enabled: bool = False` + `tac_max_delay: int = 10`; **adaLN-zero conditioning 改 per-token** (scale/shift/gate 在 sequence 维允许差异); 0 新参数。

### 6.4 训练超参 (论文披露)

| Setting | π0.6 论文值 | 本地 Cloth Task |
|---|---|---|
| Fine-tune steps | 8000 | 同 (~12h on 16 H20) |
| Batch size | 512 | 128-256 (GPU 较少) |
| Delay sampling | uniform [0, 10] | uniform [0, 6] (30Hz × 200ms = d=6) |
| Inference denoising steps | 5 | 同 |
| Init | π0.6 base | pi05_base 或 mixed_1 |

### 6.5 复现难易度

| 维度 | 评分 | 说明 |
|---|:-:|---|
| 算法清晰度 | ⭐⭐⭐⭐⭐ | Algorithm 1 完整 JAX 代码 (论文附录) |
| 代码开源 (full repo) | ❌ | 仅论文 Algorithm 1, 无 GitHub |
| 模型 ckpt 可用 (π0.6) | ❌ | 闭源 |
| 超参完整 | ⭐⭐⭐⭐⭐ | 训练 step / batch / delay 全披露 |
| 架构改动复杂度 | ⭐⭐⭐⭐⭐ | adaLN per-token, 0 新参数 |
| 对 π0.5 可移植性 | ⭐⭐⭐⭐⭐ | adaLN 同架构 |

**总体可复现性**: 不依赖 π0.6 ckpt, 可在 π0.5 + 自有 cloth 数据上复现。

---

## 7. pi05 跨本体训练路线图 (本文档主线)

> **2026-06-04 重定向**: 原 tri-track (Track A SSL / Track C cond / **Track X X-VLA**) 中, **Track X 已分离** → [`xvla_track_x_curriculum.md`](../../training/future_plans/plans/xvla_track_x_curriculum.md) (独立 X-VLA 架构, 不在本 pi05 文档范围)。Soft/Hard prompt 已停 (§0.1)。本文档聚焦 **pi05 上的「知识-本体解耦」**。

```
pi05 跨本体数据利用 (本文档)
┌──────────────────────────────────────────────────────────┐
│ 数据侧 (必做): kai+vis 物理预合并单源 + 合并/per-source     │
│   norm + vis 加权  ── 绕开 broken datasets_yaml/ConcatDS  │
├──────────────────────────────────────────────────────────┤
│ 训练路线 (§0.3):                                          │
│   R2 模块冻结解耦 ⭐  S1 VLM 学 kai+vis → S2 冻 VLM 只训   │
│                       action expert on vis (保本体)       │
│   R1 两阶段全参    co-train → 轻量 vis-only finetune       │
│   R3 Action Head Cond  domain token 拆双峰 (可叠加, 预合并)│
│   R4 数据筛选      只取 kai0 复杂任务 ep (降冲突, 可叠加)   │
├──────────────────────────────────────────────────────────┤
│ 上游独立: R0 SSL visual pretrain (kai+vis+xvla, 无 action) │
│   → 喂 pi05 backbone (ssl_phase_pretrain_pipeline.md)     │
└────────────────────────┬─────────────────────────────────┘
                         ↓  vis 真机为终判 (抖动 + 成功率)
```

**文档边界**:
- **本文档**: pi05 上如何用 kai+vis 数据 (R1–R4 + R0 指针)。
- **Track X (X-VLA)**: [`xvla_track_x_curriculum.md`](../../training/future_plans/plans/xvla_track_x_curriculum.md) —— 独立模型/架构线 (Florence2 + 官方 soft prompt + EE6D 20D + domain 槽 warm-init)。
- **数据预合并 + conditioning 实验**: [`corrected_plan_a_conditioning_premerge.md`](../../training/future_plans/plans/corrected_plan_a_conditioning_premerge.md) (R1/R3 落地)。
- **R0 SSL 预训练**: [`ssl_phase_pretrain_pipeline.md`](../../training/future_plans/plans/ssl_phase_pretrain_pipeline.md)。

**Milestone (pi05 线)**:
- **M-now**: R1/R2 预合并 + 两阶段实验 → vis 真机 vs **vis-only baseline** (要超越的对象)。
- **M-next**: R3 (Action Head Cond, 预合并路径) / R4 (数据筛选) 作增强 ablation。
- **远期**: R0 SSL backbone 注入 + 跨任务扩展 (Task B 检索 / Task C 挂衣)。

---

## 8. 风险预警 + 关键陷阱

| # | 风险 | 应对 |
|---|---|---|
| 1 | CoTracker3 在 heavy occlusion (crumpled cloth) 失败 | Pseudo-track 加 confidence filter; track loss 按 mask 加权 |
| 2 | RAFT 在 fast motion 失败 | Quasi-static 阶段训 flow, dynamic 阶段降权重 |
| 3 | **D435 FOV (69°) < D405 (87°)** Wrist sensor gap | ❌ ~~输入端 D405 crop 到 D435 FOV~~ (不可持续, 训练-推理双向维护, 跨相机不通用, 丢失 D405 周边信息). ✅ **representation-level invariance**: (a) view-conditioned token (data loader 标 `view_id`, E1.4/E1.5 xview head 自学); (b) RandomResizedCrop augmentation (scale 0.6-1.0) |
| 4 | π0.5 PaliGemma backbone continual SSL 易 catastrophic forget | Layer-wise lr decay, peak 5e-5, anchor loss on 1% LAION subset |
| 5 | 叠衣 success criterion 真机评估难自动化 | 设计 IoU / fold count / stage completion 离线 metric |
| 6 | IK 在 delta EE 推理时不连续 | Warm-start with current joints, 或训练同时输出 delta EE + delta joints |
| 7 | EE-relative 丢失绝对工作空间位置信息 | 加 base→top_camera frame 的 anchor token (从 hand-eye calibration 得来) |
| 8 | Multi-objective loss 不收敛 | Phase 1 先单 V-JEPA 5k step 预热, 再逐项加 |
| 9 | Embodiment cond 在 visual 还是 dynamics? | Phase 1 visual 不区分 view 来源 (要 invariant); Phase 2 dynamics 才区分 (要 partition) |
| 10 | Phase 0 (CoTracker) 慢 | Temporal stride 3 + batch size 优化 + 8-GPU 并行, 实测 ~17h |

---

## 9. 决策点

### 决策点 1: 是否引入 dagger?
- L1 (SSL): ✅ 引入 (3457 ep 增加 vision diversity)
- L2 (policy): ❌ 不引入 (抖动 +62%, 污染 action prior)
- L3 (aux): ⚠️ 可选 (作为 inverse dynamics 目标)

### 决策点 2: Embodiment conditioning 实现方式? (2026-06-04 三次更新)
- ❌ **Hard prompt — 停** (真机效果微乎其微)
- ❌ **Soft prompt — 停** (pi05 移植实验线关闭; 官方 soft prompt 留在 Track X 独立文档)
- ✅ **Action Head Cond Token (Track C, 方案 A) — 保留** (pi05 上唯一 conditioning, = R3; **必须走预合并单源, 不走 broken datasets_yaml**)
- pi05 主线已转向**非 conditioning 的 R2 模块解耦 / R1 两阶段** (§0.3); conditioning (R3) 作可叠加增强, 非主路径
- 真机评估: 全部终态在 vis (B 真机) 测试

### 决策点 3: EE-relative action 是否启用? ❌ 已 deprioritize (2026-05-22)
- ~~Phase 0 E0.5 EE-relative preprocessing~~ 取消
- ~~Phase 3 E3.4 / E3.8 delta EE~~ 取消
- 理由: R 腕 21° paired shift 由 R2 模块解耦 (S2 vis-only 重训 action expert) + R3 Action Head Cond 处理, 工程量更低 + 无 IK 不连续风险
- 保留作为远期 backup

### 决策点 4: 是否回看 M1 短期方案?
- 触发条件: pi05 R1/R2 (或 R3 Action Head Cond) 首轮实验完成且**真机超过 vis-only baseline** → M1 不需要
- 否则: 回看 B oversample 修复抖动 (EE-relative 不再回看)

### 决策点 5: 采纳 TAC 作为 Phase 3 ablation 新维度 ✅
- 采纳 TAC (零参数, 几乎零成本, 7-13% improvement 已 paper 验证)
- A2C2 暂搁置 — 等 TAC 结果再决定 dynamic obs response 是否必要
- 保留 Inference RTC (`pi0_rtc.py` 已实现) — 老 ckpt 部署还能用

---

## 10. 相关文档

- **Plan 详情**: `docs/training/future_plans/plans/` (`ssl_phase_pretrain_pipeline.md`, `xvla_track_x_curriculum.md`, `pytorch_native_vis_v2_full.md`)
- **Conditioning 跟踪**: `docs/training/history/experiments/xvla_conditioning_methods_results.md`, `conditioning_vs_action_representation_ablation.md`
- **实时推理**: `docs/deployment/inference/realtime_vla/strategy.md`, `docs/deployment/inference/rtc_implementation.md`
- **数据集诊断**: `docs/training/history/experiments/dataset_diagnostic_report.md`
- **Norm-stats ablation 实测**: `docs/training/history/experiments/norm_stats_ablation_apr28_450.md`
