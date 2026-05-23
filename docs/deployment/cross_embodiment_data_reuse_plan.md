# 跨本体数据复用 — 战略与执行计划 (Consolidated)

> **更新时间**: 2026-05-22 (晚 — 3-robot heterogeneous 架构明确)
> **作者**: Tim + 综合外部研究员讨论 + 项目实测数据
>
> **战略核心** (2026-05-22 晚 修订):
>
> ```
> 3 个异构机器人数据集 (heterogeneous robots):
>   • A = KAI0 (官方 piper, D435 wrist, 6,512 ep)              — 训练用, 不部署
>   • B = vis  (自有 piper, D405 wrist, 895 ep)   ⭐ 部署目标   — 训练用 + 真机评估
>   • C = XVLA-Soft-Fold (第三方 piper, 1,729 ep)              — 训练用, 不部署
>
> 部署目标: 仅 B (vis 真机) — 所有真机评估在 vis 上做
> 训练目标: 充分利用 3 个异构机器人 (~9,136 ep) 的 cross-embodiment 价值,
>           不污染 B 部署性能, 并为 CoRL/NeurIPS paper 铺路。
> ```
>
> **方法学**: 用 **X-VLA 官方架构** (ICLR 2026) — 天然 multi-domain (3 domain), Florence2 + SoftPromptedTransformer + EE6D 20D action, 论文证明在 SoftFold-Agilex 同款 Piper 上 100%。每个机器人一个 domain_id, 推理时 force `domain_id=vis(1)`。
>
> **历史路线 (已废弃)**: pi0.5 + 移植 Soft Prompt (Track B) — 训练不稳定, 弃用; 改走 X-VLA 官方原生 (Track X)。

---

## 目录

- [Part I — 战略评估](#part-i--战略评估)
  - [1. Embodiment Gap 定量](#1-embodiment-gap-定量)
  - [2. 4-层 ROI 战略框架](#2-4-层-roi-战略框架)
  - [3. 数据规模与现状校准](#3-数据规模与现状校准)
  - [4. 核心假说矩阵 (H1-H4)](#4-核心假说矩阵-h1-h4)
- [Part II — 技术参考](#part-ii--技术参考)
  - [5. EE-relative Action 可行性](#5-ee-relative-action-可行性)
  - [6. 与 π0.5 / X-VLA 默认对照](#6-与-π05--x-vla-默认对照)
- [Part III — 执行计划](#part-iii--执行计划)
  - [7. Milestone 总览 (M1-M4) — Dual-Track Parallel](#7-milestone-总览-m1-m4--dual-track-parallel)
  - [8. M2 详细计划 (Track A SSL + Track C Action Cond + Track X X-VLA 官方)](#8-m2-ssl-pretraining-详细-phase-0-4)
  - [9. 资源 + 数据 + 网络](#9-资源--数据--网络)
- [Part IV — 跟踪 + 风险](#part-iv--跟踪--风险)
  - [10. 状态跟踪 (持续更新)](#10-状态跟踪-持续更新)
  - [11. 风险预警 + 关键陷阱](#11-风险预警--关键陷阱)
  - [12. 决策点](#12-决策点)
  - [13. 修订历史](#13-修订历史)

---

# Part I — 战略评估

## 1. Embodiment Gap 定量 — **3 异构机器人**

> 2026-05-22 修订: 明确为 3 个异构机器人 (A = KAI0 / B = vis ⭐部署/ C = XVLA), 而不是 A+B 二分。

### 1.1 三机器人对比表

| 维度 | A: KAI0 (官方) | **B: vis (⭐ 部署目标)** | C: XVLA-Soft-Fold (第三方) |
|---|---|---|---|
| 机械臂型号 | piper 双臂 | piper 双臂 (同型) | piper 双臂 (待确认型号) |
| Joint DOF | 14 (7×2) | 同 | 同 (假设) |
| 控制频率 | 30 Hz | 同 | 待确认 |
| **Wrist 相机** | D435 (FOV 69°×42°, rolling, min depth 28cm) | **D405** (FOV 87°×58°, global, min depth 7cm) ⭐ | 待确认 hdf5 元数据 |
| **Wrist 安装** | 一致设计 (旧 flange) | 一致设计 (新 flange, 高度/角度略差) | 第三方采集, 不同 setup |
| **双臂间距** | 标准 | 略差 (毫米级) | 待确认 |
| **Top 头部相机** | D435 (76cm 高, 30° 角) | 同 (略差 <5cm/5°) | 待确认 |
| Action 语义 | "Flatten and fold the cloth." | 同 | 同 (cloth fold) |
| **Episodes** | 6,512 | **895** (B 数据量少 → 关键瓶颈) | 1,729 |
| **数据格式** | LeRobot v2.1 (14D joint) | LeRobot v2.1 (14D joint) | hdf5 (EE6D? 待确认) |
| **部署?** | ❌ 不部署 | ✅ **唯一部署目标** | ❌ 不部署 |
| domain_id (Track X) | 0 | **1** | 2 |

### 1.2 与 B 部署目标的 gap 量化

A vs B 关键 wrist gap (从 §3.3 实测):
- **R 腕 yaw+roll paired shift ~19°** (SE3 复合旋转) — 部署 B 时 wrist 视野 OOD
- L2 mean diff 0.47 rad (跨 robot effect, 剔除 operator confound 后)
- 13/14 dim 在 ±1σ 内 (大部分 PI per-dataset norm 可吸收)

C (XVLA-Soft-Fold) vs B gap **待量化** (等 E0.7 格式适配后做 norm_stats 对比, 预期 wrist 相机差异最大)。

### 1.3 实测真机症状回顾

引自 [dataset_diagnostic_report.md](../training/dataset_diagnostic_report.md):

1. **Cloth loop** (复杂场景): mixed_1 baseline (纯 A 训练) 部署 B 出现循环卡死 — D435→D405 视觉 OOD 累积漂移
2. **空桌面抖动**: vis SFT 后 prior 被高 jump 帧拉宽, 空桌面 condition 弱 → 抽到大 action
3. **混训抖动 > 纯 B**: `mixed_pure2_1800_6000` 真机抖 > `pure_1200_new_norm` → naive 混训创造双模式策略, chunk 间切换抖

---

## 2. 4-层 ROI 战略框架

**判断标准**: 某 loss / objective 是否依赖 A 和 B 的 action space 对齐?

| Layer | 内容 | 依赖 action 对齐? | A 价值 | 工程复杂度 |
|---|---|---|---|---|
| **L1: Visual SSL / World Model** ⭐ | V-JEPA + point track + flow, dynamics | ❌ 不依赖 | **全功率可用** | 高 |
| **L2: Embodiment-cond Policy** | A+B 共训, 通过 embedding 区分 | ⚠️ 弱依赖 (需 conditioning) | 可用, 需对齐 | 中 |
| **L3: Auxiliary tasks** | Inverse dynamics, future frame pred | ⚠️ 部分依赖 | 可用, 不入主 loss | 中 |
| **L4: Data engine / Sim2Real** | Retargeting, replay-augmentation | ✅ 强依赖 | 低 (需高保真 retarget) | 高 |

> **核心原则**: A 的价值不在"直接帮 B 做 task", 而在 **representation / dynamics / prior** 这些更上游的层次。
>
> 本文档主线: **L1 + L2 dynamics + L3** 端到端实验 (详见 §8 M2 计划)。

---

## 3. 数据规模与现状校准

### 3.1 训练数据池 (2026-05-21 实测路径)

| 来源 | Episodes | 总帧数 | Avg/ep | 视频路径 | Size |
|---|---:|---:|---:|---|---:|
| **A: Kai0 base** | 3055 | 3.36M | 1101 | `/data/shared/ubuntu/workspace/dataset/Kai0_official/Task_A/base/videos/` | 46G |
| **A: Kai0 dagger** | 3457 | 2.42M | 699 | `/data/shared/ubuntu/workspace/dataset/Kai0_official/Task_A/dagger/videos/` | 39G |
| **B: vis_v2_merged** | 895 | 1.06M | 1188 | `/data/shared/ubuntu/workspace/dataset/Task_A/vis_v2_merged/videos/` | 6.3G |
| **XVLA-Soft-Fold** | 1729 | ~? | — | 见 §9.2 多地副本表 | 444G |
| **合计** | **9136 ep** | **~7M frames** | — | 3 views (top_head, hand_left, hand_right) per ep | **~535G** |

**视角统一命名** (LeRobot v2.1 convention, 也用于 SSL data loader):
- `observation.images.top_head` — top 相机 (A 全是 D435, B 用 D435)
- `observation.images.hand_left` — 左 wrist (A 用 D435, B 用 D405) ⚠️ embodiment gap
- `observation.images.hand_right` — 右 wrist (同上)

### 3.2 当前模型 SOTA 对比 (val MAE@1)

| 实验 | Init | 数据 | Best MAE@1 | 真机表现 |
|---|---|---|---:|---|
| `task_a_new_pure_200` (js02 resume) | mixed_1 step 22k | vis 200 ep | **0.0065** ⭐ | 待测 |
| `task_a_new_pure2_1800_6000` (uc SOTA) | pi05_base | 7900 ep mix | 0.0085 | **抖动严重** |
| `task_a_new_pure2_1800_js` (js cluster) | pi05_base | 1800 ep | 0.0090 | 待测 |
| `task_a_new_smooth_800` (uc03) | mixed_1 | vis_clean 800 | 完成 | 待测 |

**重要观察**: val MAE 漂亮的 SOTA `mixed_pure2_1800_6000` 真机抖动严重 — **val MAE ≠ 真机平滑度**。

### 3.3 KAI0 ↔ vis 实测 Norm-stats 对比 (2026-05-21)

> 直接从原始 parquet 重算 (kai0_base 102/3055 ep × 114k frames, vis_v2_merged 112/895 ep × 133k frames), 不通过任何 cached 或 xvla 模块入口。XVLA-Soft-Fold 是独立第三方数据集, 不参与本对比。

#### 3.3.1 Δmean 单独表 (A=KAI0_base vs B=vis_v2_merged)

| dim | label | Δmean (A−B, rad) | Δ角度 (°) |
|---:|---|---:|---:|
| 0 | L_肩 yaw | +0.017 | +1.0° |
| 1 | L_肩 pit | +0.179 | **+10.2°** |
| 2 | L_肘 | −0.100 | −5.7° |
| 3 | L_腕 yaw | +0.064 | +3.7° |
| 4 | L_腕 pit | +0.077 | +4.4° |
| 5 | L_腕 rol | −0.127 | −7.3° |
| 6 | L_grip | −0.001 | — |
| 7 | R_肩 yaw | +0.121 | +6.9° |
| 8 | R_肩 pit | +0.010 | +0.6° |
| 9 | R_肘 | −0.177 | **−10.1°** |
| **10** | **R_腕 yaw** | **−0.293** | **−16.8°** ⭐ |
| 11 | R_腕 pit | +0.019 | +1.1° |
| **12** | **R_腕 rol** | **+0.244** | **+14.0°** ⭐ |
| 13 | R_grip | +0.013 | — |

#### 3.3.2 完整对比 (含 std + z-score)

| dim | label | A.mean | A.std | B.mean | B.std | Δmean | Δ角度 | B/A σ | |Δ|/A.σ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | L_肩yaw | -0.062 | 0.20 | -0.079 | 0.24 | +0.017 | 1.0° | 1.20 | 0.09σ |
| 1 | L_肩pit | +1.547 | 0.53 | +1.368 | 0.57 | +0.179 | 10.2° | 1.06 | 0.33σ |
| 2 | L_肘 | -1.301 | 0.46 | -1.200 | 0.46 | -0.100 | 5.7° | 0.99 | 0.22σ |
| 3 | L_腕yaw | -0.095 | 0.30 | -0.159 | 0.44 | +0.064 | 3.7° | 1.48 | 0.22σ |
| 4 | L_腕pit | +0.796 | 0.24 | +0.719 | 0.29 | +0.077 | 4.4° | 1.21 | 0.32σ |
| 5 | L_腕rol | +0.031 | 0.28 | +0.158 | 0.40 | -0.127 | 7.3° | 1.43 | 0.45σ |
| 6 | L_grip | +0.028 | 0.034 | +0.029 | 0.030 | -0.001 | — | 0.88 | 0.03σ |
| 7 | R_肩yaw | +0.115 | 0.17 | -0.006 | 0.22 | +0.121 | 6.9° | 1.30 | 0.71σ |
| 8 | R_肩pit | +1.486 | 0.57 | +1.476 | 0.58 | +0.010 | 0.6° | 1.03 | 0.02σ |
| 9 | R_肘 | -1.461 | 0.54 | -1.284 | 0.52 | -0.177 | 10.1° | 0.96 | 0.33σ |
| **10** | **R_腕yaw** ⚠️ | +0.048 | 0.28 | +0.341 | 0.32 | **-0.293** | **16.8°** | 1.13 | **1.05σ** |
| 11 | R_腕pit | +0.918 | 0.24 | +0.899 | 0.23 | +0.019 | 1.1° | 0.97 | 0.08σ |
| **12** | **R_腕rol** ⚠️ | +0.003 | 0.25 | -0.241 | 0.27 | **+0.244** | **14.0°** | 1.09 | **0.99σ** |
| 13 | R_grip | +0.035 | 0.033 | +0.021 | 0.029 | +0.013 | — | 0.88 | 0.41σ |

#### 3.3.3 汇总分析

```
L1  norm:  1.44 rad
L2  norm:  0.51 rad    ← 核心 metric
L∞ max:   0.293 rad = 16.8° @ R_腕yaw (dim 10)

分布重叠 (|Δ|/A.σ z-score):
  Median: 0.32σ    Max: 1.05σ
  Within ±1σ:   13/14
  Within ±0.5σ: 11/14
  Within ±0.3σ:  6/14

Per-arm:
  Left:  L2 = 0.26 rad, max 10.2° @ L_肩pit, max |Δ|/σ = 0.45σ
  Right: L2 = 0.44 rad, max 16.8° @ R_腕yaw, max |Δ|/σ = 1.05σ
  → 右臂偏移 1.67× 左臂

运动幅度 B/A:  median 1.07, mean 1.11, range [0.88, 1.48]
  → B 整体 motion range 略 > A
```

#### 3.3.4 关键发现 (4 条)

1. **整体分布"高度重叠 but not identical"**: 13/14 维落在 A 的 ±1σ 内, PI per-dataset norm 大部分可吸收。但 L2 = 0.51 rad 在 50-step chunk 上累积影响显著。

2. **右臂偏移 1.67× 大于左臂** — 双臂间距不同的直接证据 (§1.2 quantified)。

3. **R 腕 yaw (-16.8°) + R 腕 roll (+14°) 是配对偏移** ⭐ (核心发现):
   - 不是独立, paired correlated shift
   - SE(3) 表示下合成 ~**21° 复合旋转**
   - 物理意义: 右手 wrist 末端在 B 上比 A 整体旋转 21° → **D405 wrist 视野下 cloth 出现 21° 旋转 OOD**
   - **这是 EE-based action 的精准价值场景** — EE 在 gripper local frame, 天然消除复合旋转

4. **B 运动幅度比 A 大 10-30%** (B/A std ratio): 解释 §1.3 "vis SFT 后 prior 被拉宽" 的现象 — vis 操作员动作幅度更大, action prior 更宽。

#### 3.3.5 对三种跨 embodiment 策略的精准启示

| 策略 | 能处理 R 腕 21° 旋转? | 能处理 motion range 1.1× scale-up? | 综合 |
|---|:-:|:-:|:-:|
| **PI per-dataset norm** | ⚠️ 部分 (mean 对齐, 但 chunk 内仍有 wrist OOD) | ✅ std 缩放自动 | ⭐⭐ |
| **Soft Prompt** (X-VLA 官方, Track X) | ✅ 显式 condition, 学到 domain shift | ⚠️ 隐式 | ⭐⭐⭐⭐ |
| **EE-based** (Delta EE) | ✅ **天然消除** | ⚠️ 不直接 | ⭐⭐⭐ |
| **Soft Prompt + EE 结合** | ✅✅ | ✅ | ⭐⭐⭐⭐ |

→ EE-based 不再是"可有可无", R 腕 21° paired shift 是 joint 表示的硬伤, EE 是干净解。但仍建议**作为 Phase 3 ablation 而不是 wholesale switch** — PI norm + Soft Prompt 可能已覆盖大部分场景。

---

### 3.4 vis 内部 Operator 与时间漂移分析 (2026-05-21)

> 深入挖掘 vis_v2_merged 内部结构, 揭示 §3.3 KAI0↔vis 偏移中 operator confound 与 cross-robot effect 的分量。

#### 3.4.1 实际 Operator 结构

`meta/episodes.jsonl` 含 `operator` + `_src_dir` 字段, 实际:

| Group | Operator (alias) | Episodes | 占比 |
|---|---|---:|---:|
| **G1** (主操作员, ztm+lym 同一人) | ztm 723 + lym 149 | 872 | **97.4%** |
| G2 (助手) | gsy | 23 | 2.6% |

时间跨度: 2026-04-23 ~ 2026-05-09 (10 个采集日期)。

#### 3.4.2 跨 Group 对比 (G1 vs G2)

| 指标 | 值 |
|---|---:|
| L2 mean diff | 0.518 rad |
| max |Δ|/σ | 0.64σ @ L_腕rol |
| Within ±1σ | 14/14 |

→ G2 (gsy) 与 G1 偏移**约等于** KAI0 ↔ vis 跨 robot 偏移 (0.47-0.51)。

#### 3.4.3 G1 内时间漂移 (同一人, 不同日期)

| 日期 | L2 vs 2026-04-24 baseline | max |Δ|/σ |
|---|---:|---:|
| 04-24 | 0 (baseline) | — |
| 04-25 | **0.42** | 0.69σ |
| **04-28** | **0.47** ⭐ | **0.88σ** (peak) |
| 04-29 | 0.45 | 0.77σ |
| 04-30 | 0.32 | 0.36σ |
| 05-06 | 0.33 | 0.42σ |
| 05-07 | 0.33 | 0.42σ |
| 05-08 | 0.40 | 0.75σ |
| 05-09 | 0.25 | 0.43σ |

→ **同一 operator 跨 5 天 (4-24 vs 4-28) drift = 0.47 rad**, 与 cross-robot effect 同量级!

#### 3.4.4 真正 Cross-robot Effect (剔除 gsy 干扰)

```
KAI0_base vs G1-only (剔 gsy):  L2 = 0.4650 rad, max 0.93σ @ R_腕yaw (14.9°)
KAI0_base vs full vis (含 gsy):  L2 = 0.5105 rad, max 1.05σ @ R_腕yaw (16.8°)

→ 剔除 gsy 后 cross-robot L2 仅降 8.9%, R_腕 yaw+roll paired shift 仍是 ~19°
```

#### 3.4.5 关键发现修正

1. **gsy (2.6%) 对 norm_stats 影响极小** (current vs G1-only L2 = 0.08 rad, 0.16σ) → **不必 per-operator norm**
2. **G1 内时间漂移 ≈ cross-robot drift** (0.47 vs 0.47) → 4-24 数据可能与 4-25+ 是不同 "phase" (设备 calibration 漂移)
3. **R 腕 paired shift ~19° 真实存在** (剔除 operator confound 后仍在), 是真正的 cross-robot geometric effect

#### 3.4.6 立即可做的实验

- 用 G1 (剔 gsy) + 4-25+ 数据 (剔 warm-up phase) 重训 → 真机对比当前 smooth_800

---

### 3.5 混训策略 6 方案 + 实证一致性分析 (2026-05-21)

> 实证回答: "两数据集 (KAI0 + vis G1) 能否混训?" 通过 per-dim 归一化后多指标对齐性测量。

#### 3.5.1 6 种混训方案对比

| 方案 | 描述 | 处理 R 腕 19° | 处理时间 drift | 处理 motion range diff | 工程量 |
|---|---|:-:|:-:|:-:|:-:|
| **A. Naive joint norm** | 合并算单一 norm_stats | ❌ | ❌ | ❌ | 0.5 day |
| **B. Per-dataset norm + Single model** | 每数据集 own norm, 同一 model 不显式 condition | ⚠️ (90.7%) | ⚠️ (90.7%) | ✅ | 1 day |
| **C. Soft Prompt + Per-DS norm** | per-DS norm + domain_id 显式 routing (X-VLA) | ✅ | ✅ | ✅ | 0 (代码已就绪) |
| **D. Curriculum (A pretrain → B finetune)** | mixed_1 → smooth_800 现有路线 | ✅ (B finetune) | ✅ | ✅ | 1 day |
| **E. SSL Decoupled** | A 进 visual SSL, B 进 policy (Track A) | 不参与 action | 不参与 | 不参与 | 9 week |
| **F. EE-based action** | delta EE pose 表示, 天然 embodiment-invariant | ✅ **天然消除** | ⚠️ EE 也漂 | ⚠️ | 3 day |

#### 3.5.2 实证: Per-dataset Norm 对齐效果 (MMD 测量)

直接计算 A 和 B 自归一化后的分布距离:

```
不归一化:                    MMD(A_raw,  B_raw)  = 0.0597    (large divergence)
Per-dataset norm 后:        MMD(A_norm, B_norm) = 0.00558   (降低 90.7%)
self baseline:               MMD(A_norm, A_norm) = 0.0002
Ratio MMD(A,B) / MMD(A,A): 28×  (仍有残差)
```

→ **Per-dataset norm 消除 90.7% 的分布偏差** (主要是 per-dim mean/std), 但仍有 28× self-baseline 残差。

#### 3.5.3 残余 10% 偏差来源 — Per-dim Norm 解决不了的部分

经 per-dim self-norm 后, 5 个维度的对齐分析:

| 度量 | 对齐? | 残差 (估算) |
|---|:-:|---|
| Per-dim mean | ✅ 完美 | 0 (by construction) |
| Per-dim std | ✅ 完美 | 0 (by construction) |
| Per-dim Skewness | ⚠️ 部分 | L2 ≈ 1.5 (排除 outlier dim 后) |
| Per-dim Kurtosis | ⚠️ 部分 | 大多数 dim 在 1-3 量级差 |
| **Per-dim quantile shape** | ⚠️ 部分 | median quantile L2 0.51, max 3.7 (异常 dim 6/7) |
| **Inter-dim correlation** (joint synergy) | ❌ 显著不同 | Frobenius 2.5; B 各维联动更强 (mean |off-diag| 0.21 vs A 0.12) |

具体 finding:
- **B 的关节联动更强**: 例如 L_肩pit × L_腕rol 相关系数 A=+0.09 vs B=+0.41 (相差 0.32)
- **A 的某些 dim 几乎不动** (data quirk): L_grip 在 A 几乎全程 const (q01-q99 跨度仅 0.017), 而 B 是双模式 open/close (跨度 2.69)
- **Skewness 差异**: 大多数 dim 高阶矩仍不同

#### 3.5.4 修正方案 B 评级

| | 之前评级 | **修正后** | 修正理由 |
|---|:-:|:-:|---|
| A. Naive joint norm | ❌ 已证失败 | ❌ 已证失败 | MMD 0.06, 不变 |
| **B. Per-dataset norm + Single model** | ⭐⭐ 中性, 不推荐 | **⭐⭐⭐ 应该可行** | MMD 降至 0.006 (10× 小), 比 naive 显著好 |
| C. Soft Prompt + Per-DS norm | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 处理残余 10% 仍最优 |
| D. Curriculum | ⭐⭐⭐ | ⭐⭐⭐ | — |
| E. SSL Decoupled | ⭐⭐⭐⭐ (Track A) | ⭐⭐⭐⭐ | — |
| F. EE-based | ⭐⭐⭐ | ⭐⭐⭐ | — |

#### 3.5.5 关键 Insight (重要)

> **`mixed_pure2_1800_6000` 失败的真因可能不是"混训不能", 而是用了 joint norm (A) 而非 per-dataset norm (B)**。
>
> 如果当时用方案 B 训练, 可能效果显著好于 joint norm 但仍逊于 Soft Prompt。

#### 3.5.6 推荐策略 — Layered Combination

```
Layer 1 (Visual, Phase 1 Track A SSL):
  E. SSL decoupled — A + B + XVLA all in, no action loss
  → 学到 cross-embodiment invariant visual repr

Layer 2 (Policy, Phase 3 / Track X):
  C. Soft Prompt + Per-DS norm — 显式 routing
  + D. Curriculum (二阶段 A→B finetune)
  → 显式 routing + lock 到 B

Layer 3 (Phase 3 ablation):
  F. EE-based — joint vs EE 控制变量实验
  → paper ablation 数据点
```

#### 3.5.7 验证假说的新 Ablation Set (Phase 3 加)

| Exp | Cond Method | Norm 策略 | 用途 |
|---|---|---|---|
| **E3.0** baseline | × | per-dataset (single ds smooth_800) | 当前 SOTA |
| **E3.5** | × | **Naive joint norm** (A) | 故意复现失败假说 |
| **E3.6** | × | **Per-dataset norm + Single model** (B) | **验证 §3.5.5 insight** |
| **E3.7** | **Soft Prompt** (X-VLA 官方原生, VLM input 端) | Per-DS norm + Soft Prompt | Track X 路线 (§8.8) |
| **E3.8** ⭐ 主线 | **Action Head Cond Token** (方案 A, action expert input 端) | Per-DS norm | Track C 路线 — 与 E3.7 1:1 对照, 验证 "VLM 端 vs Action expert 端" 注入点选择 |
| ~~**E3.9**~~ Dual Cond | ~~Soft Prompt + Action Head Cond~~ | — | **2026-05-22 搁置** — 双端组合, 待 E3.7/E3.8 单端结果出来再决定是否启用 |

→ E3.5 vs E3.6 量化 "naive joint vs per-dataset norm" 的真机抖动差异。
→ **E3.7 vs E3.8 量化 conditioning 注入点选择 (VLM input vs Action expert)** — 主线 ablation (2026-05-22 决策, 双端组合 E3.9 待资源充足再启)。
→ Action Head Cond 选定方案 A (Concat token), B/C/D 暂搁置 — 详见 §6.3.1。
→ EE-relative 路线 (旧 E3.8 delta EE) 已 deprioritize, paired shift 由 conditioning 处理。

---

### 3.6 已隐式执行的 L2 (但未显式标识)

当前 SOTA 链 `pi05_base → mixed_1 → task_a_new_pure_200` 本质上是 **A-heavy pretrain → B-only finetune** 的 curriculum, 但缺失:
- ❌ 没有显式 embodiment conditioning (model 不知道哪是 A 哪是 B)
- ❌ 没有 EE-relative action (用绝对关节角)
- ❌ 没有 wrist view 对齐
- ❌ kai0_dagger 进了 init (污染抖动 prior)

→ 真机抖动是这些缺失的总和体现。

---

## 4. 核心假说矩阵 (H1-H4)

| ID | 假说 | 关键实验 | 成功标准 |
|---|---|---|---|
| **H1** | SSL pretrain on A+XVLA+B 提供的 visual repr 在 cloth 任务上 > π0.5 default | E3.1 vs E3.0 | B finetune val MAE 降低 ≥10%, 真机抖动减少 |
| **H2** | Multi-objective (V-JEPA + track + flow + xview) > 单 V-JEPA | E1.5 vs E1.1 | Downstream val MAE 进一步降低 |
| **H3** | Embodiment-conditioned dynamics 让 A 的物理 prior 不污染 B policy | E3.3 vs E3.2 | 真机平滑度 + 复杂场景成功率 提升 |
| **H4** | Motion-residual decomposition: cloth_residual 部分 embodiment-invariant | E2.3 latent 分析 | A vs B cloth_residual latent 的 MMD < 0.1 |

---

# Part II — 技术参考

## 5. EE-relative Action 可行性 ⚠️ DEPRIORITIZED (2026-05-22)

> **2026-05-22 决策**: EE-relative action 路线**整体暂停**, 不进入近期实验计划。R 腕 21° paired shift (§3.3) 由 **Soft Prompt + Action Head Cond Emb 组合 (新主线 E3.7/E3.8/E3.9)** 处理 — 在 LLM input 端 + Action expert 端双端 condition embodiment domain, 实现与 EE-based 等价的 paired shift 消除, 但工程量更低 + 不引入 IK 不连续风险。
>
> 本节内容**保留为技术参考**, 但 Phase 0 E0.5 / Phase 3 E3.4 (EE-relative) / E3.8 (delta EE) 全部移出执行计划。新主线见 §6.3。

### 5.1 可用资源 (参考)

| 资源 | 位置 |
|---|---|
| **piper URDF** | `calib/piper_local.urdf` (SolidWorks 完整导出) |
| **DH 参数 + 2° j2/j3 校正** | `/home/tim/workspace/piper_sdk/piper_sdk/kinematics/piper_fk.py` (C++) |
| **PiperFK Python 封装** | `calib/piper_fk.py` (`PiperFK().fk_homogeneous(q)` → 4×4) |
| **Hand-eye 标定 (camera↔arm base)** | `config/calibration.yml` (DANIILIDIS, reproj <0.3px) |
| **双臂 CAN 配置** | `config/pipers.yml` |

### 5.2 三种 EE-relative 方案对比

| 方案 | 公式 | 跨本体优势 | 实现成本 |
|---|---|---|---|
| **A. Delta joints** | a_t = q_t − q_{t−1} | ✅ 完全绕开几何, 最简单 | 极低 (parquet 改 1 列) |
| **B. Delta EE pose** ⭐ | a_t = T^{−1}_{t−1,EE} ⊗ T_{t,EE} (6-DOF twist) | ✅ 绕开 base 偏置, 保留 EE 物理意义 | 中 (跑 FK + log map) |
| **C. EE pose in base frame** | a_t = T_{t,EE} (绝对) | ❌ base→arm 偏置仍有 | 中 |

**推荐方案 B**: Delta EE pose 是物理最干净的 embodiment-invariant 表示 — "gripper 在自己 frame 里挪了多少", 同 piper 不同 base 安装位置完全无关。

### 5.3 实施步骤 (~1 天)

```python
# 1. 离线预处理脚本
from calib.piper_fk import PiperFK
fk = PiperFK()

for ep in dataset:
    actions = ep["action"]  # (T, 14) — joint angles
    q_left = actions[:, 0:7]; q_right = actions[:, 7:14]
    # Compute EE pose per arm
    T_left = np.stack([fk.fk_homogeneous(q[:6]) for q in q_left])
    T_right = np.stack([fk.fk_homogeneous(q[:6]) for q in q_right])
    # Delta in EE frame: dT_t = T_{t-1}^{-1} @ T_t
    dT_left = np.linalg.inv(T_left[:-1]) @ T_left[1:]
    dT_right = np.linalg.inv(T_right[:-1]) @ T_right[1:]
    # se(3) log map
    twist_left = se3_log(dT_left)
    twist_right = se3_log(dT_right)
    new_action = concat([twist_left, twist_right, gripper_L, gripper_R])
    # Write back parquet
```

### 5.4 推理时反变换

```python
# 部署时: model outputs delta EE → 累积回 EE pose → IK → joints
T_current = fk.fk_homogeneous(q_current)
for delta in predicted_chunk:
    T_next = T_current @ se3_exp(delta[:6])
    q_next = ik_solve(T_next, q_current)  # warm start
    send_to_arm(q_next)
    T_current = T_next
```

**风险**: IK 解可能不唯一 / 不连续 → 需要 warm-start。**备选**: 训练时同时输出 delta EE + delta joints, 部署时优先 delta joints (避开 IK)。

---

## 6. 与 π0.5 / X-VLA 默认对照

### 6.1 Action 表示决策 (实证调研, 见 [delta_vs_absolute_research](.))

| 模型 | 默认 Action | 备注 |
|---|---|---|
| **π0 (老)** | Delta (relative to chunk start) | OpenPI docs |
| **π0.5 (新)** | **Absolute** (默认), 可选 relative | LeRobot pi05 docs |
| **OpenPI Aloha 数据** | DeltaActions transform on (use_delta_joint_actions=True) | 内部 pipeline 转 delta |
| **本地 mixed_1 ckpt** | **Absolute** (实测 norm_stats: mean[1]=1.48, std=0.63, 与 state 同分布) | 已通过 norm_stats 分析确认 |
| **KAI0 数据集 (raw)** | **Absolute** (joint angles ±π) | 数据库实测 |
| **X-VLA** | EE6D absolute pose (20D = xyz+Rot6D+grip per arm) | ICLR 2026 |

**最权威实证研究** ([Demystifying Action Space Design, arxiv 2602.23408](https://arxiv.org/abs/2602.23408)): 13000+ rollouts 表明:
- **单机器人 / 单任务 / long-horizon** → **absolute** 更稳 (我们的场景 ✓)
- **多 embodiment / 跨设备** → delta 更稳
- **混合 mask** (joint delta + gripper absolute) 是 pragmatic 选择

### 6.2 Embodiment Conditioning 选项

| 方式 | 实现复杂度 | 本地代码状态 | 推荐度 |
|---|---|---|---|
| **Hard prompt** (`"[D405 wrist] ..."`) — 改 prompt 字符串 | 极低 (0 改 model) | ✅ 任意 config 都可用 | ⭐⭐ 弱版本 (信号沿 LLM attention 自然传播, 不显式 gate) |
| **Soft prompt** (X-VLA 官方原生) | 低 (Track X 走官方 `SoftPromptedTransformer`, 见 §8.8) | ✅ `lerobot/xvla-base` ckpt + repo `/home/tim/workspace/X-VLA/` | ⭐⭐⭐⭐ **主线 (Track X)** — 显式 inject 到 24 层 SoftPromptedTransformer, 推理时 force `domain_id=vis` |
| **Action head embedding** (Track C 方案 A — paper ablation) | 极低 (action expert input concat 1 domain token) | ✅ 已实现 (`pi0.py:action_head_cond_hub`) | ⭐⭐⭐ — paligemma 不知 domain, 信号仅在 action expert. 提供 "action expert 端 cond" 对照 |

### 6.3 Action Head Conditioning Embedding (2026-05-22 新主线 — **方案 A 选定**)

> **动机**: Soft Prompt 在 **VLM (PaliGemma) 输入端**注入 domain embedding，信号经 24 层 LLM attention + cross-attn → action expert KV cache → action。**在 action expert 输入端直接注入 domain token**，paligemma 完全不知 domain，conditioning 只调制 action expert 的 denoise 行为，与 Soft Prompt 形成 "VLM 端 vs Action expert 端" 1:1 对照。

#### 6.3.1 实现方案 — 方案 A: Concat Domain Token at Action Expert Input

> **2026-05-22 用户决策**: 4 候选方案中选定 **方案 A**（B/C/D 暂搁置）。理由: 工程最简, paper 与 Soft Prompt 形成最干净 sparse-prefix 对照, 直接验证 "domain conditioning 模块选择" 这一核心 question。

**信号路径对比 (Soft Prompt vs 方案 A)**:

```
Soft Prompt (Track X 官方原生, X-VLA-0.9B):
  d → SoftPromptedTransformer.soft_prompts[d] (B,32,1024)
      → 拼到 Florence2-VLM input
      → 24 层 SoftPromptedTransformer attention
      → action heads (DomainAwareLinear) → action

方案 A (Track C):
  d → action_head_cond_hub[d] (B,1,1024)
      → 拼到 action expert input (与 noise_action_token 同级)
      → action expert self-attn (4-8 层)
      → action
      [paligemma 完全不知 domain]
```

**关键差异**:
- Soft Prompt: 控制 *VLM 如何看世界*（domain-specific perception）
- 方案 A: 控制 *action expert 如何 denoise*（domain-specific motor output）
- 互不竞争, 但 paper E3.7 vs E3.8 验证 "perception vs motor" 注入点选择

#### 6.3.2 代码改造点

| 文件 | 改动 |
|---|---|
| `kai0/src/openpi/models/pi0_config.py` | 加 `action_head_cond_num_domains: int = 0`（默认禁用）|
| `kai0/src/openpi/models/pi0.py` | (1) `__init__` 加 `self.action_head_cond_hub = nnx.Embed(num_domains, action_expert_width)` (init N(0, 0.02)); (2) action expert forward 中读 `obs.dataset_id` → embed → reshape (B, 1, D) → 拼到 noise_action_tokens 前 |
| `kai0/src/openpi/training/config.py` | 新 config: `xvla_actcond_stage1_kai_warmup` / `xvla_actcond_stage2_vis_only` / `xvla_actcond_stage3_joint_finetune` |
| `kai0/src/openpi/transforms.py` | 已修, dataset_id 已透传 ✓ |

**伪代码**:
```python
# pi0_config.py
@dataclasses.dataclass
class Pi0Config:
    ...
    action_head_cond_num_domains: int = 0  # 0 = disabled

# pi0.py:Pi0.__init__
if config.action_head_cond_num_domains > 0:
    self.action_head_cond_hub = nnx.Embed(
        num_embeddings=config.action_head_cond_num_domains,
        features=action_expert_width,
        embedding_init=nnx.initializers.normal(0.02),
    )

# pi0.py: action expert forward (or wherever noise_action_tokens are prepared)
if self.action_head_cond_hub is not None:
    domain_token = self.action_head_cond_hub(obs.dataset_id)  # (B, D)
    domain_token = domain_token[:, None, :]                    # (B, 1, D)
    action_input = jnp.concat([domain_token, action_input], axis=1)  # (B, 1+L, D)
    # adjust attention mask + position embeddings accordingly
```

#### 6.3.3 训练流程 (修订: 单阶段 balanced, 2026-05-22 PM)

> **架构修订**: 弃用 3-stage curriculum, 改单阶段 joint training。理由见 §6.3.6。

| 步骤 | 状态 | 备注 |
|---|---|---|
| Phase 1.5 编码 + smoke test | ✅ 完成 | uc01 8 A800, step 50 mu d0=7.35e-5 PASS |
| **Single-stage balanced** | 🔄 running (flgmf) | Shanghai 16 A100, kai_base + kai_dagger + **vis × 7** (balanced sampling) joint 50k step from pi05_base |

#### 6.3.6 为什么放弃 3-stage curriculum? (2026-05-22 PM 决策)

经讨论方案 A 的实际信号路径:

| 维度 | Soft Prompt (Track X X-VLA 官方) | Action Cond (Track C 方案 A) |
|---|---|---|
| 信号传播路径 | 24 层 PaliGemma + 4-8 层 action expert | **仅 4-8 层 action expert** |
| 影响 image/text representation? | ✅ 是 (domain 信息改变 VLM attention) | ❌ 否 (paligemma 完全不知 domain) |
| 信号对齐难度 | 高 | **低** |
| Stage 2 freeze-backbone 必要性 | 高 (保护 24 层 VLM) | **中-低** (action expert 4-8 层, 短路径) |

**关键洞察**: Soft Prompt 影响 VLM attention pattern (24 层影响 image 怎么编码), 需要 stage 2 隔离训练保护; **Track C 方案 A 只影响 action expert 怎么把 latent 转 action**, 不改 image 编码, stage 2 价值边际低。

**数据不平衡 (kai 6512 ep vs vis 895 ep, 7.27×) 用 ConcatDataset over-sampling 处理** (vis × 7 在 datasets_yaml 重复路径, 49/51 split)。这比 stage 2 frozen-backbone 更直接、更轻量。

**最终方案**: 单阶段 joint kai+vis 50k step, balanced sampling (vis ×7), 12h 完成。paper 对照 E3.7 (X-VLA 官方 Soft Prompt) vs E3.8 (Action Cond joint balanced)。

#### 6.3.4 真机评估目标

> **2026-05-22 用户决策**: Track C 训练用 **kai + vis 跨本体混合数据**, 真机测试用 **vis (B 真机)** — 验证 cross-embodiment training 是否提升 B 真机表现。

| Variant | 训练数据 | 真机平台 | 关键 metric |
|---|---|---|---|
| **C3.0 (Track C 终态 = Action A Stage 3)** | kai+vis 混训 | **vis (B 真机)** | 抓衣角成功率 / 折叠成功率 / 抖动 p99 / 30 ep × 固定 + 3 OOD |

#### ~~6.3.5 方案 B/C/D — 暂搁置 (2026-05-22)~~

以下 3 个方案暂时不实施, 保留作技术参考。如未来 Track C 方案 A 真机效果不达预期, 可回看:

- ~~B. FiLM (Feature-wise Linear Modulation)~~ — per-block γ/β modulation
- ~~C. adaLN (adaptive LayerNorm)~~ — DiT-style, 与现有 adaRMS 互动复杂
- ~~D. Cross-Attention from domain emb to action layers~~ — 最 expressive 但计算 +15%

(完整设计与对比见 git history commit `4306b4c` ↔ 之前的 §6.3 版本)

---

### 6.4 RTC / TAC — Action Chunking 实时性方案对比 + 集成计划 (2026-05-22)

> 问题: chunk 边界不连续 + 推理延迟下抖动累积. 三类方案 (inference / training / 模块化), 我们考虑叠加在 Track A 或 Track C 终态上。

#### 6.4.1 三篇 RTC 论文核心对比

| 论文 | 时间 | 路线 | 改 base 模型? | 推理 latency | 真机验证 |
|---|---|---|:-:|:-:|:-:|
| **Inference RTC** (Black, [2506.07339](https://arxiv.org/abs/2506.07339)) | 2025-06 | 推理时 inpainting + pseudo-inverse vjp guidance | ❌ | **+28%** (97 vs 76 ms) | ✅ 6 task × 28h × 480 ep (π0.5) |
| **TAC** (Black 团队, [2512.05964](https://arxiv.org/abs/2512.05964)) ⭐ | 2025-12 | **训练时**把 prefix actions 作 ground-truth context | ❌ (改 loss + adaLN per-token) | **0** (与 baseline 持平) | ✅ π0.6 box building / espresso |
| **A2C2** (Sendai, [2509.23224](https://arxiv.org/abs/2509.23224)) | 2025-09 | 加 lightweight correction head, 每步基于最新 obs 输出 Δa | ❌ (base frozen, +新 module) | +4.7ms (~6%) | ❌ 仅 sim (Kinetix, LIBERO) |

#### 6.4.2 维度详细对比

| 维度 | Inference RTC | **TAC** ⭐ | A2C2 |
|---|---|---|---|
| 推理 latency | +28% | **0** | +5% |
| 重训需求 | ❌ 不需 | ✅ 需 (8k step finetune) | ⚠️ 只重训 small head |
| Backward 兼容 ckpt | ✅ | ❌ | ✅ |
| 每步用最新 obs | ❌ | ❌ | ✅ ⭐ |
| 对动态环境反应 | 低 | 低 | 高 |
| 代码改动 | 中 (vjp + scan) | **小 (<2% codebase)** | 中 (新 module) |
| 真机验证 | ✅ 充分 | ✅ 部分 | ❌ 无 |
| Smoothness 来源 | guided diffusion 朝 prev_chunk | 模型内化, 自然平滑 | 每步 residual 修正 |
| 与 Soft Prompt / 三轨叠加 | ✅ orthogonal | ✅ orthogonal | ✅ orthogonal |

**关键 insight**: 三者**正交可叠加**, 各自解决不同子问题:
- Inference RTC = pseudo-inverse 强行约束 (老 ckpt 补救)
- **TAC = 模型自己学会 chunk overlap (训练时一次, 推理零开销)**
- A2C2 = 添加实时反应模块 (cloth dynamic state 时强相关)

#### 6.4.3 本地实现状态 (2026-05-22 实测)

| 项 | 文件 | 状态 |
|---|---|---|
| **Inference RTC** | `kai0/src/openpi/models/pi0_rtc.py` (360 行) | ✅ **完整实现** (论文 1 的 1:1 复刻: `get_prefix_weights` 4 schedules — ones/zeros/linear/exp, `jax.vjp` guidance, `guidance_weight` clipping = min(c·inv_r2, max_guidance_weight)) |
| **TAC training** | — | ❌ **未实现** (compute_loss 仍标准 flow matching, 见 pi0_rtc.py:206-232) |
| **A2C2 correction head** | — | ❌ 未实现 |

#### 6.4.4 TAC 集成方案 — Algorithm 1 移植 (论文已给完整代码)

**核心改动 (~6 行 + adaLN per-token)**:

```python
# kai0/src/openpi/models/pi0_rtc.py — compute_loss 改:

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

**Architecture 改动 (Pi0Config)**:
- 加 `tac_enabled: bool = False` + `tac_max_delay: int = 10`
- **adaLN-zero conditioning 改成 per-token** (scale / shift / gate 在 sequence 维允许差异)
- **不增加可学习参数** (per-token 只是 broadcast 改 indexing)

#### 6.4.5 训练 hyper-params (论文披露完整)

| Setting | π0.6 论文值 | 我们 Cloth Task 候选 |
|---|---|---|
| Fine-tune steps | 8000 | 同 (~12h on 16 H20) |
| Batch size | 512 | 128-256 (我们 GPU 较少, 调小) |
| Delay sampling | uniform `[0, 10]` | uniform `[0, 6]` (我们 30Hz 控制 vs 50Hz, max latency 200ms 对应 d=6) |
| Inference denoising steps | 5 | 同 |
| Init | π0.6 base | pi05_base 或 mixed_1 |
| 调度 (sim) | 从 epoch 24 finetune 8 epoch | 从 baseline 22k step finetune 8k step |

#### 6.4.6 复现难易度评估

| 维度 | 评分 | 说明 |
|---|:-:|---|
| 算法清晰度 | ⭐⭐⭐⭐⭐ | Algorithm 1 完整 JAX 代码 (论文附录) |
| 代码开源 (full repo) | ❌ | 仅论文 Algorithm 1, 无 GitHub |
| 模型 ckpt 可用 (π0.6) | ❌ | 闭源 |
| 数据 ckpt 可用 | ⚠️ | Kinetix 公开, real task 闭源 |
| 超参完整 | ⭐⭐⭐⭐⭐ | 训练 step / batch / delay 全披露 |
| 架构改动复杂度 | ⭐⭐⭐⭐⭐ | adaLN per-token, 0 新参数 |
| 对 π0.5 可移植性 | ⭐⭐⭐⭐⭐ | adaLN 同架构 |

**总体可复现性**: **不依赖 π0.6 ckpt, 可在 π0.5 + 我们自有 cloth 数据上复现**, 5 day 工程量。

#### 6.4.7 TAC 与 Track A/B/C 的叠加关系

```
                      ┌─────────────────────────────────────────┐
                      │   Track A:  SSL Visual Pretrain          │
                      │   Track C:  Action Head Conditioning     │
                      │   Track X:  X-VLA 官方 (Soft Prompt LLM) │
                      │   (+) TAC training (compute_loss 改动)   │ ← Phase 3 加
                      └────────────────┬────────────────────────┘
                                       │ 推理时
                                       ↓
                      ┌─────────────────────────────────────────┐
                      │   (Optional) Inference RTC 仍可启用       │
                      │   (Optional) A2C2 correction head        │
                      └─────────────────────────────────────────┘
```

→ TAC **不与任何现有 Track 冲突**, 只是 Phase 3 训练时多一个 flag (`tac_enabled=True`)。

#### 6.4.8 集成时间线 (插入 Phase 3)

| 阶段 | 任务 | 时间 |
|---|---|---|
| **Phase 3 准备** | (1) 写 `pi0_rtc.py::compute_loss_tac` (6 行新增) <br> (2) `Pi0Config.tac_enabled` flag <br> (3) adaLN per-token broadcast patch (~30 行) | **2 day** |
| **Smoke test** | uc02 8 GPU × 5k step, smooth_800 数据, 看 loss curve | **0.5 day** |
| **Phase 3 ablation 集成** | 加入 E3.x 新变种 (见 §10.4) | 同 Phase 3 主线 |
| **真机评估** | 30 ep cloth fold, vs E3.0 baseline + E3.4 stack | **1 day** |
| **总计** | — | **~4-5 day**, 不抢主线 GPU |

#### 6.4.9 Phase 3 Ablation 新增 (待加入 §10.4)

```yaml
现 §10.4 ablation 加 RTC 维度:
  E3.0   baseline                          (no RTC)
  E3.4   Full Stack (SSL + Soft Prompt)    (no RTC)
  E3.RTC1  + Inference RTC (运行时 enable_rtc=True, 老 ckpt 即可)  ← 已 implemented, 0 训练
  E3.RTC2  + TAC training (compute_loss_tac, 8k step finetune)   ← 新加 ⭐
  E3.RTC3  + TAC + Inference RTC (训练 TAC 后推理仍启 RTC, 二次叠加)
  E3.RTC4  + TAC + A2C2 correction head    ← 终极 (若 cloth dynamic 需求强)
```

预期排序 (真机 smoothness): E3.RTC4 > E3.RTC3 > E3.RTC2 > E3.RTC1 > E3.0
预期排序 (latency): E3.0 = E3.RTC2 < E3.RTC4 ≪ E3.RTC1 = E3.RTC3

#### 6.4.10 决策 (2026-05-22)

- ✅ **采纳 TAC** 作为 Phase 3 ablation 新增维度 (零参数, 几乎零成本, 论文实证 7-13% improvement)
- ⏸️ **A2C2 暂搁置** — 等 TAC 跑完看是否还需要 dynamic obs response (cloth 主要 static deformation, 反应性需求中等)
- 🔄 **保留 Inference RTC** (`pi0_rtc.py` 已实现) — 不破坏现有 inference 路径, 老 ckpt 部署还能用

#### 6.4.11 参考文献

- [Real-Time Execution of Action Chunking Flow Policies (Black 2506.07339)](https://arxiv.org/abs/2506.07339)
- [Training-Time Action Conditioning for Efficient Real-Time Chunking (2512.05964)](https://arxiv.org/abs/2512.05964) — HF page: [huggingface.co/papers/2512.05964](https://huggingface.co/papers/2512.05964)
- [Leave No Observation Behind: Real-time Correction for VLA Action Chunks (Sendai 2509.23224)](https://arxiv.org/abs/2509.23224)
- [Daily ArXiv VLA TAC 中文分析](https://infinity4b.github.io/daily-arxiv-vla/papers/2512.05964/)
- [pi.website RTC 官方介绍](https://www.pi.website/research/real_time_chunking)
- 本地实现: `kai0/src/openpi/models/pi0_rtc.py` (Inference RTC ✓, TAC ✗)

---

# Part III — 执行计划

## 7. Milestone 总览 (M1-M4) — Tri-Track Parallel

### 📐 三轨并行架构 (2026-05-22 晚 修订, Track B 已废弃)

```
                     ┌─────────────────────────────────────┐
                     │   Track A: SSL 主线                 │
                     │   (uc02 + Robot-North-H20)          │
                     │   ├── Phase 0 Pseudo-labels         │
                     │   ├── Phase 1 V-JEPA + track + flow │
                     │   ├── Phase 2 Dynamics + Embodiment │
                     │   └── Phase 3 Policy + Ablation     │
                     ├─────────────────────────────────────┤
                     │   Track C: Action Head Cond Emb     │
                     │   (Action expert 端 cond, paper     │
                     │    ablation 对照, 16 GPU)           │
                     │   └── 单阶段 balanced joint training│
                     ├─────────────────────────────────────┤
                     │   Track X: X-VLA 官方架构 ⭐ 主线    │
                     │   (Florence2 + SoftPromptedXformer, │
                     │    EE6D/joint 20D action, 16 GPU)   │
                     │   ├── X3.A: 3-domain (A+B+C)        │
                     │   └── X3.B: 2-domain (A+B, no XVLA) │
                     └─────────────────┬───────────────────┘
                                       ↓
                          ┌──────── Final Merge ────────┐
                          │ SSL Visual Backbone +       │
                          │ X-VLA 官方 ckpt (X3.A/B 终态)│
                          │ + Dynamics-conditioned head │
                          └─────────────────────────────┘
```

**资源分配 (2026-05-22 晚)**:
- Track A: uc02 (Phase 0) + Robot-North-H20 (Phase 1-3)
- Track C: 已在 cn-shanghai / cn-beijing 跑 paper ablation 对照 (4 个 job 中)
- Track X (主线): uc01+uc02 16 A800 — X3.A / X3.B 顺序执行

### 🚀 M1 (1-2 周): 短期真机修复 — **已 deprioritize**

> 用户决策 (2026-05-21): 先专注 L1 SSL + X-VLA 路线 (M2-M3), M1 暂不展开。

### 🔬 M2 (~9 周): Multi-track 训练 ⭐ **主线**

**Track A — SSL Pretraining + Dynamics + Policy**: 详见 §8.1-8.5。
- **当前进度**: Phase 0 E0.1 完成 (kai0_base + dagger CoTracker3 已跑完)

**Track X — X-VLA 官方架构 Native 训练**: 详见 §8.8。
- **当前进度**: prep (HF ckpt 拉取 + env setup + dataset adapter)

### 🌍 M3 (M2 之后): Track A + Track X Merge + 真机 + Paper

- **Merge 策略**: SSL visual backbone (E1.5) + X-VLA 官方 Soft Prompt (X3.A/B 终态) + Dynamics-conditioned head
- **真机大规模测试** (60-100 ep per ablation)
- CoRL / NeurIPS submission

### 📝 M4 (long-tail): ATOM Policy 扩展

- ATOM stack: frozen M2 visual + dynamics → object tokenizer (per-point/region) → policy head
- 适用于跨任务扩展 (Task B 检索, Task C 挂衣)

---

## 8. M2 SSL Pretraining 详细 Phase 0-4

### 8.1 整体目标

> 训一个 cloth-folding-specific visual encoder (基于 π0.5 PaliGemma/SigLIP backbone continual-pretrain), 用作下游 B-only policy 的 vision tower, 真机性能优于直接用 π0.5 default。

**总 GPU-day 预算**: ~140 GPU-day on Robot-North-H20 (39 GPU free / 56 total)。

### 8.2 Phase 0 — 数据预处理 + 伪标签生成 (Week 1-2)

**目标**: 把 9136 ep 视频 (7M frames × 3 view = 21M frame-views) 跑过 CoTracker3 / RAFT / SAM, 生成离线 pseudo-labels 给 Phase 1 SSL 用。

**资源分配 (并行)**: uc02 跑 CoTracker + flow (8 A800 80GB), Robot-North-H20 1 节点 (8 H20) 跑 SAM。

| Exp | Tool | 输入 (有效) | 输出 | Resource | ETA |
|---|---|---|---|---|---|
| **E0.1** Pseudo-track | CoTracker3 (scaled_offline.pth, v3.0 windowed) | T=24 window, stride=12 → ~580k windows × 3 view | `tracks/{ep_id}/{view}.npz` (W, T, N=36, 2) | uc02 8 A800 | ~17h (实测) |
| **E0.2** Optical flow | RAFT-Large | adjacent pair, **temporal stride 3** → ~2.3M pairs × 3 view | `flow/{ep_id}/{view}.npz` (H/8, W/8, 2) | uc02 4 GPU 并行 | ~20h |
| **E0.3** Cloth mask | SAM2 (Hiera-L) | 1 frame/sec × 9136 ep × 3 view ≈ 820k mask | `mask/{ep_id}/{view}.npz` | Robot-North-H20 8 H20 | ~12h |
| **E0.4** ~~FOV align~~ | ~~OpenCV~~ | **取消 (用户决策, 不可持续)** | — | — | — |
| ~~**E0.5** EE-relative action~~ | ~~Python + PiperFK~~ | ~~A + B + XVLA actions~~ | ~~`action_ee_relative/{ep_id}.npz` (T, 14)~~ | ~~CPU~~ | ❌ **取消 (2026-05-22)** — EE-relative 路线整体 deprioritize, R 腕 21° paired shift 由 Soft Prompt + Action Head Cond 处理 |

> **E0.4 已取消 (2026-05-21 决策)**: D405 → D435 FOV pixel-level crop 不可持续 (训练-推理双向维护负担 + 跨相机不通用 + 丢失 D405 周边信息)。
> **替代方案**: (1) **View-conditioned token** (data loader 标记 `view_id`, model 自学区分) — 由 Phase 1 E1.4/E1.5 中的 `xview_head` 自然实现; (2) **RandomResizedCrop augmentation** (scale 0.6-1.0) 让 model 自然 robust 到 FOV 差异。原理: representation-level invariance > input-level pixel hack。详见 §11 风险 #3。

**优化策略**:
1. **Temporal stride 3**: 7M → 2.3M effective frames
2. **CoTracker windowed**: T=24 window stride=12 (避免 OOM)
3. **SAM 稀疏化**: 1 frame/sec (cloth 形态变化慢)
4. **断点续传**: .npz 已存在且 > 100 byte 即 skip

**输出位置 (统一约定)**:
```
/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/ssl_phase0/
├── tracks/<dataset>/ep_XXXXXX/{top_head,hand_left,hand_right}.npz
├── flow/<dataset>/...   (同结构)
├── masks/<dataset>/...  (同结构, 稀疏 1/sec)
├── action_ee_relative/{ep_XXXXXX}.npz
└── logs/                (每 GPU shard 一个 log)
```
(原 `rgb_d405_d435align/` 已取消)

**质量检查**: 抽样 50 ep 人工 inspect, 不合格的 ep 记 `skip_list.txt`。

### 8.3 Phase 1 — SSL Visual Encoder Pretrain (Week 2-4)

**核心**: 从 π0.5 SigLIP/PaliGemma vision tower **continual-pretrain** (不 from scratch), 输出 cloth-fold-specific encoder。

> **并行调度 (per user 决策)**: 跑 3 jobs in parallel — E1.1 (baseline) / E1.4 (xview) / E1.5 (full multi-objective)。跳过 E1.2/E1.3 中间点 (从 E1.5 内置 ablation 反推单项贡献)。

#### E1.1 — V-JEPA Baseline (单目标)
```yaml
backbone:     π0.5 SigLIP (continual-pretrain)
input:        T=16 frames, 3 views, 224×224
objective:    masked latent prediction (V-JEPA 2.1)
mask:         tube 30%, edge-saliency 2× boost on cloth edges (用 E0.3 mask)
lr:           5e-5 layer-wise decay (peak), warmup 2k, cosine to 5e-7
batch:        128 total (16 H20 × 8/gpu)
steps:        50k
data:         A + XVLA + B 全量 (~9000 ep)
embodiment:   不区分 (统一 backbone)
output:       /vePFS-North-E/vis_robot/.../ssl_ckpts/E1.1_vjepa_base/
```

#### E1.4 — V-JEPA + Track + Flow + Cross-view
```yaml
继承 E1.1, 加 3 个 head:
  - track_head: 8M param, predict 36 keypoint tracks over T=16
                loss = L2(xy) + BCE(visibility), w=0.5
  - flow_head:  predict dense flow from latent
                loss = EPE vs RAFT pseudo, masked by cloth mask, w=0.3
  - xview_head: top latent → wrist latent (autoregressive)
                loss = cosine + L2, w=0.2
weights:      固定 (w_vjepa=1.0, w_track=0.5, w_flow=0.3, w_xview=0.2)
```

#### E1.5 — Full Multi-objective + Phase Weights + Saliency + Multi-scale
```yaml
继承 E1.4:
  - Phase 1 weights (step 0-25k):  w_vjepa=1.0, w_track=0.5, w_flow=0.3, w_xview=0.2
  - Phase 2 weights (step 25k-50k): w_vjepa=0.5, w_track=1.0, w_flow=0.5, w_xview=0.3
  - Saliency mask: edges 2×, interior 0.5×
  - Multi-scale temporal: 一半 batch T=8 (short), 一半 T=48 (long)
  - Anchor loss: 1% batch on LAION subset (防 catastrophic forget)
```

**Phase 1 验收 (downstream micro-eval)**:
- 每个 E1.x 跑完 → 小规模 B-only policy finetune: 3k step, batch 32, 8 GPU on uc02
- 比较 val MAE on B val set
- E1.5 应该明显胜 E1.1 (H2 验证)

### 8.4 Phase 2 — Dynamics Pretrain (Week 5-6)

**核心**: 在 frozen visual encoder 上训 latent dynamics model, 引入 embodiment conditioning + motion-residual decomposition。

| Exp | 内容 | 关键改动 |
|---|---|---|
| **E2.1** | Latent Dynamics baseline (无 embodiment) | `(z_t, action_t) → z_{t+1}`, L2 loss |
| **E2.2** | + Embodiment Conditioning | input 加 `embodiment_emb` (A_emb, XVLA_emb, B_emb, dim=128) |
| **E2.3** ⭐ | + Motion-Residual Decomposition (paper 原创) | 分两 head: `ego_motion_head` (embodiment-specific) + `cloth_residual_head` (embodiment-invariant) |
| **E2.4** | + Inverse Dynamics Auxiliary | predict `a_t | (z_t, z_{t+1}, emb_e)`, weight 0.2 |

**E2.3 paper claim**: cloth_residual 部分在 A vs B 上分布相同 (用 MMD < 0.1 验证)。

**Phase 2 验收**:
- E2.3 latent 上 t-SNE / MMD: A vs B 在 cloth_residual 部分应近, 在 ego_motion 部分应远
- 如 motion-residual 分不开 → 回退 E2.2 全联合

### 8.5 Phase 3 — Downstream Policy + Ablation (Week 7-8)

**核心**: 把 Phase 1+2 产出接到 π0.5 policy, B-only finetune, 真机评估。

| Exp | Visual | Dynamics | Cond Method | Data | 用途 |
|---|---|---|---|---|---|
| **E3.0** baseline | π0.5 default | × | × | B (smooth_800) | 当前 SOTA, baseline |
| **E3.1** | E1.5 frozen | × | × | B | 测 H1 (visual repr 单独 value) |
| **E3.2** | E1.5 LoRA | × | × | B | 测 fine-tunable 是否更好 |
| **E3.3** | E1.5 LoRA | E2.3 frozen | × | B | 测 H3 (dynamics 额外贡献) |
| **E3.4** Full Stack | E1.5 LoRA | E2.3 frozen | Soft Prompt + Action Head Emb | B + A weighted | 终态最强 (Soft+ActionHead 组合) |

**训练设置**:
- 16 H20 × 50k step ≈ 35h/exp
- batch 128, lr 1.5e-5 → 1.5e-6, num-workers 64
- EMA 0.999

**真机测试 protocol**:
- 30 episode per exp, 固定场景 + 3 OOD 场景 (布料/姿态/光线)
- 指标: 抓衣角成功率, 完整折叠成功率, 平均执行时长, 抖动 metric (action diff p99)

### 8.6 Phase 4 — Real Machine + Paper (Week 9)

- 真机大规模测试 (60-100 episodes/exp)
- Final ablation table (见 §10.4)
- Failure case analysis
- Paper figures (architecture diagram, latent t-SNE, ablation curve)

### 8.8 Track X — X-VLA **官方架构** Native 训练 (2026-05-22 晚 新主线) ⭐

> 战略转向: 放弃 pi0.5 + X-VLA-style soft prompt 移植 (Track B 已废弃), **改用 X-VLA 官方完整架构**, 与论文 1:1 一致, 仅适配数据。
>
> **3 robot 异构设计** (per §1.1):
> - domain_id=0: A (KAI0, 训练用, 不部署)
> - **domain_id=1: B (vis, 训练用 + 唯一真机部署)** ⭐
> - domain_id=2: C (XVLA-Soft-Fold, 训练用, 不部署)
>
> X-VLA 论文用 30 个 domain × 290K ep 训过, 我们 3 个 domain × 9136 ep 完全适用其 multi-domain 范式。推理时 `force domain_id=1` 走 vis 路径。

#### 8.8.1 战略动机

| 维度 | Track B (pi0.5 + 移植 soft prompt) | **Track X (X-VLA 官方)** ⭐ |
|---|---|---|
| Backbone | π0.5 SigLIP/PaliGemma + Gemma 300M action expert | **Florence2-Large + 24-layer SoftPromptedTransformer** |
| Action 表示 | Joint 14D (与 PI 派一致) | **EE6D 20D** (xyz + Rot6D + grip, 每臂 10D) |
| Soft Prompt | 嫁接 (`pi0.py:soft_prompt_hub`, 32×2048) | **原生** (`SoftPromptedTransformer` 内置, 32×1024) |
| Init ckpt | pi05_base | **X-VLA-0.9B 官方 ckpt** (HF `2toINF/X-VLA`) |
| 训练 pipeline | 自创 3-stage 移植 | **论文 2-step adaptation** (Prompt Warmup + Joint Optim) |
| 实证 | 失败 (3× stage 2 全 fail) | ICLR 2026, IROS 2025 Champion, SoftFold (Piper) **100%** |

→ **从"嫁接式"改"原生式"**, 收益: 论文实证 + 同硬件已 100% (SoftFold-Agilex = 同款 Piper) + 跨 embodiment 是 X-VLA 论文核心 contribution。

#### 8.8.2 X-VLA 官方架构关键事实 (paper + 本地 repo)

**官方 repo**: `/home/tim/workspace/X-VLA/` (已 clone)
```
X-VLA/
├── models/
│   ├── modeling_xvla.py        ← 主类 (295 行)
│   ├── configuration_xvla.py   ← 24 层, 1024D, 30 domain
│   ├── modeling_florence2.py   ← Florence2 视觉 backbone
│   ├── transformer.py          ← SoftPromptedTransformer (403 行)
│   ├── action_hub.py           ← EE6D / Joint / Auto action space 注册
│   └── processing_xvla.py      ← 多模态处理器
├── datasets/                   ← 数据加载 + domain_handler/
├── evaluation/
│   └── SoftFold-Agilex/        ← 我们同款 Piper 评估 (100% 验证!)
├── train.py                    ← 全参数训练 (282 行)
├── peft_train.py               ← LoRA 微调
└── deploy.py                   ← FastAPI 推理服务
```

**模型架构** (来自论文):
```
High-Dimensional Stream:
  主视角图像 → Florence2-Large → VLM Features
  辅助视角 → 独立视觉编码 → Aux Visual Features

Low-Dimensional Stream:
  Proprio + Action + Time embedding → 轻量 Linear

SoftPromptedTransformer (24 层, 1024D, 16 头):
  输入 = [Action_tokens | VLM_proj | Aux_proj | Soft_Prompts(domain_id)]
  → DomainAwareLinear → action 预测

输出: 30 步动作序列 [30 × 20D EE6D]
```

**EE6D 20D action 编码** (每臂 10D):
```
左臂: xyz(3) + Rot6D(6) + gripper(1) = 10D
右臂: xyz(3) + Rot6D(6) + gripper(1) = 10D
合: 20D × 30 step chunk
```

**两步法 adaptation** (论文 §3.3):
- **Step 1 (Prompt Warmup)**: 冻 backbone (Florence2 + Transformer), 仅训新加的 soft_prompt[domain_id] + action heads. LR base × 0.1 for soft prompts, base for action heads.
- **Step 2 (Joint Optim)**: 解冻 backbone, joint finetune all (soft prompt LR × 0.1)。

#### 8.8.3 训练数据 — 2 个对照实验 (X3.A vs X3.B, 量化 XVLA 贡献)

**数据池** (无需重新采集, 3 个异构 robot 都可用):

| 机器人 | domain_id | Episodes | 当前 action 格式 | 转 EE6D 需要 | 部署? |
|---|:-:|---:|---|---|:-:|
| **A: KAI0 base + dagger** (官方 D435 wrist) | 0 | 6,512 | 14D joint | ✅ 转 EE6D (PiperFK + Rot6D) | ❌ 不部署 |
| **B: vis_v2_merged** (自有 D405 wrist) ⭐ | **1** | 895 | 14D joint | ✅ 同上 | ✅ **唯一部署** |
| **C: XVLA-Soft-Fold** (第三方) | 2 | 1,729 | 待确认 (可能 EE6D) | 检查 hdf5 元数据 | ❌ 不部署 |
| **合计** | — | **9,136 ep** | — | — | — |

#### 8.8.3.1 两组对照实验 (Curriculum: Continual Pretrain → vis Adaptation)

> **战略**: "**X-VLA Phase I extension + Phase II adaptation**" curriculum。先把我们 3 个 (X3.A) / 2 个 (X3.B) domain 知识注入官方 base, 得到 extended-base; 再用 vis 数据单独 finetune lock 到部署 robot。
>
> **核心目的**: 量化 **C (XVLA) 对 B (vis) 部署是否有增益**。X3.A 与 X3.B 仅在 Stage A 数据池上不同 (含/不含 XVLA), 其他设置完全一致 → 单变量 ablation。
>
> **2026-05-22 用户决策**: 只跑 curriculum 版本 (X3.A + X3.B), 不做 single-training 对照 (节省资源)。

##### Exp X3.A — 3-domain Curriculum (A + B + C)

```yaml
=== Stage A: Continual Pretrain on multi-domain ===
Init:           lerobot/xvla-base (官方 Phase I ckpt, 290K ep × 30 domain)
                                  ← 我们 "extend" 这个 base
Data:           A (KAI0) + B (vis) + C (XVLA-Soft-Fold), mixed
Sampling:       balanced 1×:7×:2× (per_domain_weights = {0:1.0, 1:7.0, 2:2.0})
effective ep:   6512 + 6265 + 3458 = 16,235
有效比例:        ~40% A + ~39% B + ~21% C
domain_id 槽:    用 base 中未占用的 3 个 (e.g., 19=A, 20=B, 21=C)
Goal:           把 A/B/C 3 个 domain 知识写入扩展 base
Schedule:       freeze_steps=1000, steps=20000, LR base=1e-4, VLM × 0.1
Output:         X3.A_extended_base.ckpt

=== Stage B: vis-only Adaptation (Phase II 标准) ===
Init:           X3.A_extended_base.ckpt (Stage A 输出)
Data:           B (vis) only, 895 ep
domain_id:      固定 = 20 (vis)
Goal:           target-specific lock, 强化部署 prior
Schedule:       freeze_steps=500, steps=10000, LR base=5e-5 (更低防 overfit), VLM × 0.1
EMA:            0.9999, 监控 val loss 选 best step
Output:         X3.A_deploy.ckpt (vis 部署用)
```

##### Exp X3.B — 2-domain Curriculum (A + B, **不含 XVLA**, 对照)

```yaml
=== Stage A: Continual Pretrain on 2 domains ===
Init:           lerobot/xvla-base (同 X3.A)
Data:           A (KAI0) + B (vis), 不含 C
Sampling:       balanced 1×:7× (per_domain_weights = {0:1.0, 1:7.0})
effective ep:   6512 + 6265 = 12,777
有效比例:        ~51% A + ~49% B
domain_id 槽:    用 2 个 (e.g., 19=A, 20=B)
Goal:           2 domain (无 XVLA) 注入扩展 base
Schedule:       同 X3.A Stage A (freeze_steps=1000, steps=20000)
Output:         X3.B_extended_base.ckpt

=== Stage B: vis-only Adaptation (同 X3.A) ===
Init:           X3.B_extended_base.ckpt
Data:           B (vis) only, 895 ep
domain_id:      固定 = 20 (vis)
Schedule:       同 X3.A Stage B
Output:         X3.B_deploy.ckpt
```

#### 8.8.3.2 两组实验对比 — XVLA 数据贡献 Ablation

| 指标 | X3.A (3-domain) | X3.B (2-domain) | 解读 |
|---|---|---|---|
| Stage A 数据 | A+B+C | A+B (不含 C) | 唯一变量 |
| Stage A effective ep | 16,235 | 12,777 | X3.A 多 27% |
| Stage A 训练时长 | ~15h on 16 H20 | ~12h on 16 H20 | X3.A 略长 |
| Stage B 数据 | vis only | vis only | 完全一致 |
| Stage B 训练时长 | ~8h on 16 H20 | ~8h on 16 H20 | 完全一致 |
| 总训练时长 | **~23h** | **~20h** | — |
| Soft prompt 数 | 3 个 (A/B/C) | 2 个 (A/B) | — |
| 真机评估 | vis | vis | 都一样 |

**期望结论 (3 种可能)**:
| X3.A vs X3.B (真机 vis 评估) | 推断 |
|---|---|
| **X3.A > X3.B** | ✅ XVLA 提供有用 cross-platform 多样性 → 终态用 X3.A |
| **X3.A ≈ X3.B** | ⚠️ XVLA neutral, 简化为 X3.B 即可 (降低复杂度) |
| **X3.A < X3.B** | ❌ XVLA 过于不同, dilute vis prior → 弃 XVLA, X3.B 为终态 |

#### 8.8.3.3 Sampling 实现细节 (Stage A 用)

```python
# X-VLA datasets/domain_config.py 风格 (per_domain_weights)
# Stage A (continual pretrain) 用 balanced sampling
DOMAIN_WEIGHTS_X3A = {19: 1.0, 20: 7.0, 21: 2.0}   # A:B:C = 1:7:2 (domain_id 19/20/21 用 base 未占)
DOMAIN_WEIGHTS_X3B = {19: 1.0, 20: 7.0}             # A:B   = 1:7  (无 C)

# InfiniteDataReader 内部按 weight 采样
# 每个 batch (假设 batch_size=256):
#   X3.A Stage A:  ~102 A samples + ~100 B samples + ~54 C samples
#   X3.B Stage A:  ~129 A samples + ~127 B samples

# Stage B 不用 sampling weight, 单一 domain (B = vis)
```

#### 8.8.3.4 部署设置 (两组共用)

```python
# 推理 vis 真机时 (X3.A_deploy 和 X3.B_deploy 都一样)
domain_id = 20                                       # 固定 vis domain (Stage A 时分配的 ID)
output = model(obs, action_history, domain_id=20)    # 走 vis-specific soft_prompt
action = output.actions  # 30 step × 20D EE6D
# 反变换 EE6D → joint (per-arm IK)
joint_action = ee6d_to_joint_per_arm(action, current_joints)
```

→ X-VLA 推理代码 `evaluation/SoftFold-Agilex/deploy/client_eef6d_xvla.py` 已实现 EE6D → euler → PosCmd 反变换 (见 analysis_kai0_xvla.md §8)。

#### 8.8.3.5 Stage B Overfit 防护 (vis only 895 ep)

895 ep × 10k step 训练有 overfit 风险, 需:
1. **Lower LR** (5e-5 vs Stage A 的 1e-4)
2. **Shorter steps** (10k vs Stage A 20k), 加 early stop
3. **EMA = 0.9999** (X-VLA 默认)
4. **每 1k step val MAE 检查**, 选 best step ckpt (不一定是最终 step)
5. **Inline eval val set**: vis val split (e.g., vis_v2_merged_val) — 不在 train set 中

#### 8.8.4 训练 Pipeline — 2-stage Curriculum (X3.A 和 X3.B 都用)

> **2026-05-22 晚 用户决策**: 采用 "**continual pretrain + single-domain adaptation**" curriculum, 不用单 stage training。
>
> 这是 **X-VLA Phase I' (我们做的 continual pretrain) + Phase II (vis only adaptation)** 路线, 与论文 Phase I → Phase II 框架对齐, 只是 Phase I 我们做的是 continual extension (不是 from scratch)。

##### Phase I/II 三种区分 (官方 + 我们)

| | Phase I 官方 | **Phase I' (continual pretrain)** ⭐ 我们 Stage A | **Phase II (adaptation)** ⭐ 我们 Stage B |
|---|---|---|---|
| 数据 | 290K ep × 7 platforms | 我们 3 (或 2) domain mixed | vis only |
| 用途 | 训 X-VLA-0.9B base | 把我们 domain 注入 base | target lock-in 部署 |
| 我们 | ❌ 不做 (用官方 ckpt) | ✅ Stage A | ✅ Stage B |
| Init | from scratch | lerobot/xvla-base | Stage A 输出 |
| Sampling | uniform | **balanced 1:7:2 (A) / 1:7 (B)** | single domain |

##### Stage A 训练配置 — Continual Pretrain on multi-domain ⭐

```bash
lerobot-train \
  --policy.path="lerobot/xvla-base"            # 官方 X-VLA-0.9B Phase I ckpt
  --policy.dtype=bfloat16
  --policy.action_mode=auto                     # 自动检测 dim, max=20
  --policy.max_action_dim=20                    # EE6D 兼容
  --steps=20000                                 # Stage A 用 20k step (论文推荐)
  --policy.freeze_vision_encoder=false          # 不冻 VLM (LeRobot 现行推荐)
  --policy.freeze_language_encoder=false        # 不冻 LLM
  --policy.train_policy_transformer=true        # 训 transformer
  --policy.train_soft_prompts=true              # 训 soft prompts (新加的 3/2 个 domain)
  --dataset.repo_id=<mixed_dataset_yaml>        # X3.A: A+B+C balanced 1:7:2; X3.B: A+B 1:7
  --policy.repo_id=<USER>/xvla-<X3A|X3B>-stageA-extended-base
  # 内部默认:
  # --freeze_steps=1000        # 前 1k step backbone freeze (新 soft_prompt 嵌入)
  # --learning_coef=0.1        # VLM 用 base × 0.1 (关键稳定)
  # --warmup_steps=2000        # LR warmup
```

##### Stage B 训练配置 — vis-only Adaptation (target lock)

```bash
lerobot-train \
  --policy.path=<Stage A 输出 ckpt>            # ← 关键: 从 Stage A extended-base 继续
  --policy.dtype=bfloat16
  --policy.action_mode=auto
  --steps=10000                                 # Stage B 短 (防 overfit on 895 ep)
  --policy.freeze_vision_encoder=false
  --policy.freeze_language_encoder=false
  --policy.train_policy_transformer=true
  --policy.train_soft_prompts=true              # 主要更新 vis soft_prompt
  --dataset.repo_id=<vis_only_yaml>             # 仅 vis (B) 数据
  --policy.repo_id=<USER>/xvla-<X3A|X3B>-deploy
  # 关键调整 (防 overfit):
  # --learning_rate=5e-5      # ← 比 Stage A 低一半 (从 1e-4 → 5e-5)
  # --freeze_steps=500        # ← 较短 backbone freeze
  # 监控 inline_eval val MAE, 选 best step 而非最终 step
```

##### Stage A 内部 freeze schedule (X-VLA 论文 §3.3 标准, 自动)

```
step 0     ─── 1000:    backbone frozen,  仅训 soft_prompts + action_heads
                         (相当于 "Prompt Warm-up" — 让新 soft_prompts 嵌入)
step 1000  ─── 20000:   backbone 解冻,    full joint finetune
                         (相当于 "Joint Optimization" — 全模型 fine-tune)
                       
LR scaling 始终生效:    VLM LR = base × 0.1, 其他 = base LR
```

##### Stage B 内部 freeze schedule

```
step 0    ─── 500:     backbone frozen, 短 warmup (vis soft_prompt 重新 stabilize)
step 500  ─── 10000:   backbone 解冻 (LR 5e-5 vs Stage A 的 1e-4, 防 overfit)

监控:   每 1k step 测 inline_eval val MAE on vis_v2_merged_val
        早停: 若连续 3 个 eval 点 val MAE 不降, 提前终止取 best step ckpt
```

##### 资源估算 (X3.A + X3.B, 总时长)

| 项 | X3.A (3-domain) | X3.B (2-domain) |
|---|---:|---:|
| Stage A effective ep | 16,235 | 12,777 |
| **Stage A 时长** | ~15h on 16 H20 | ~12h on 16 H20 |
| Stage B effective ep | 895 (vis only) | 同 |
| **Stage B 时长** | ~8h on 16 H20 | ~8h on 16 H20 |
| **总训练时长** | **~23h** | **~20h** |
| 真机评估 | +1 day | +1 day |
| **大颗粒度 ETA** | ~25h training + 1 day eval | ~21h training + 1 day eval |

X-VLA-0.9B vs π0.5 3B 速度对比:
| 项 | π0.5 (3B, 我们之前) | **X-VLA 0.9B** |
|---|---:|---:|
| 模型大小 | ~3B params | **~0.9B** (1/3) |
| 单 step GPU 占用 | ~32GB on H20 | **~12-16GB** on H20 |
| Batch size | 128 | **256-512** (可放大) |

#### 8.8.5 数据预处理任务 (Phase 0 新增)

| 任务 | 输入 | 输出 | 工程量 |
|---|---|---|---|
| **E0.6** Joint→EE6D action 转换 | KAI0 + vis parquet 的 14D joint action | EE6D 20D action (FK + Rot6D 编码) | 1 day (复用 calib/piper_fk.py) |
| **E0.7** XVLA-Soft-Fold 格式适配 | hdf5 dataset | LeRobot-style 或 X-VLA 原生格式 | 0.5 day |
| **E0.8** Mixed dataset YAML 构建 | 3 个 source 的路径 | X-VLA datasets/domain_config.py 风格 | 0.5 day |
| **E0.9** X-VLA env + ckpt 拉取 | conda env XVLA + HF model | 部署到 vePFS-North-E | 0.5 day |

**总 Phase 0 (Track X) 额外工程量**: ~2.5 day。

#### 8.8.6 真机评估 — **仅 vis (B 真机)**

**3 异构 robot → 1 部署目标**: KAI0/XVLA 不部署, **所有真机评估只在 vis 上做**。

**评估设置**:
- 推理设置: `domain_id = 1 (vis)` 显式 force
- 任务: cloth folding (与 X-VLA SoftFold-Agilex 几乎相同)
- 真机 metric: 抓衣角成功率, 完整折叠成功率, 抖动 p99, 执行时长
- Episode budget: **30 ep 固定场景 + 3 OOD 场景** (不同布料 / 初始姿态 / 光线)

**Reference baseline** (论文报告):
- X-VLA 官方 SoftFold-Agilex 任务在同硬件 Piper 上**已 100%** 成功率
- 我们 cloth task 与 SoftFold 内容几乎相同, 期望接近官方水平

**预期收益** (vs Track B Soft Prompt 失败):

| 指标 | π0.5 + 移植 soft prompt (Track B) ❌ | **X-VLA 官方 (Track X)** ⭐ |
|---|:-:|:-:|
| Stage 收敛 | Stage 2 × 3 fail | ✅ 论文 + ICLR 验证 |
| EE 表示 | 14D joint | 20D EE6D (cloth task 友好) |
| 跨 embodiment | 嫁接式 (A↔B 二分), 不稳 | 原生 multi-domain (A/B/C 三分类), 290K ep × 7 平台验证 |
| Multi-domain scale | 2 domain (我们 cap) | 30 domain 容量 (官方默认) |
| vis 部署 prior | 难以保证 | balanced sampling (B ×7) + soft prompt force 锁定 |

#### 8.8.7 与 Track A (SSL) 的关系

**完全 orthogonal, 仍可叠加**:
```
Track A Phase 1 SSL (V-JEPA + track + flow + xview): 
  → 视觉表征 (Florence2 之外的 backbone, 用于对照/替换)

Track X X-VLA 官方训练:
  → 主线 policy, 论文复现
  
最终 paper ablation (E3.x):
  E3.0 baseline π0.5 default
  E3.1 + SSL backbone (Track A)
  X3.0 X-VLA 官方 (Track X 主线)
  X3.1 X-VLA + SSL frozen vision (Track A + X 融合)
  X3.2 X-VLA + Track A SSL LoRA (终极)
```

#### 8.8.8 Track X 实施时间线 (新插)

| Week | 任务 |
|---|---|
| Week 1 (本周) | E0.6-E0.9 数据预处理 + 环境部署 |
| Week 2 | Phase 1 Prompt Warmup (5-10k step on 16 H20) |
| Week 3-4 | Phase 2 Joint Optim (50k step, ~15-20h) |
| Week 5 | 真机评估 + ablation 表 |

**总周期**: ~4-5 周 (X-VLA-0.9B 比 π0.5 3B 快 ~2×)

#### 8.8.9 决策 (2026-05-22 晚)

- ✅ **采纳 X-VLA 官方架构** 作为 Track X 新主线
- ❌ **放弃 Track B** (pi0.5 + 移植 soft prompt) — 实证 Stage 2 × 3 fail
- 🔄 **保留 Track A** (SSL pretrain) — 与 Track X 并行 + 可叠加 vision backbone
- 🔄 **保留 Track C** (Action Head Cond) — 已实现, 可作为 paper ablation 对照
- ⏸️ **暂停 SSL Phase 0 中的 vis_v2_merged + XVLA tracks 预处理** (等 Track X data 格式确定后再算)

#### 8.8.10 参考文献

- [X-VLA Paper (ICLR 2026, arxiv 2510.10274)](https://arxiv.org/pdf/2510.10274)
- [Project Page + Demos](https://thu-air-dream.github.io/X-VLA/)
- [HuggingFace Models](https://huggingface.co/collections/2toINF/x-vla)
- [LeRobot 集成文档](https://huggingface.co/docs/lerobot/xvla)
- 本地 repo: `/home/tim/workspace/X-VLA/`
- 详细架构分析: `analysis_kai0_xvla.md` §3 (X-VLA 深度解读)

---

### 8.9 时间线 (Gantt) (8.7→8.8 更新)

```
Week 1   ┌────[Phase 0] 数据预处理 (uc02 + Robot-North-H20 并行)
         │       ↓ Phase 0 完成
Week 2-4 │   ┌──[Phase 1] SSL (3 并发: E1.1 + E1.4 + E1.5) on Robot-North-H20
         │   │
Week 5-6 │   │  ┌──[Phase 2] Dynamics (4 串行: E2.1→E2.2→E2.3→E2.4)
         │   │  │
Week 7-8 │   │  │  ┌──[Phase 3] Policy + Ablation (5 jobs)
         │   │  │  │
Week 9   │   │  │  │  ┌──[Phase 4] 真机 + Paper
         └───┴──┴──┴──┘
```

**总 9 周** (并行可压缩到 7-8 周)。

---

## 9. 资源 + 数据 + 网络

### 9.1 GPU 资源 (2026-05-22 PM 更新)

| 资源 | GPU | 状态 | 当前任务 |
|---|---:|---|---|
| **Robot-North-H20** (cn-beijing) | 47 H20 free / 56 total | active | Stage 2 v2 (6fr6c) running 16 H20 |
| **robot-task** (cn-shanghai) | 4 A100-80G free / 28 total | active | Track C abs single-stage v5 (sqthr) running 16 A100 |
| **uc02** | 8 A800 | **idle** ✅ E0.1 CoTracker base+dagger 完成 (6512 ep tracks) | 待: Track X X3.A 训练 (uc01+uc02) |
| **uc01** | 8 A800 | **idle** ✅ Track C smoke + exp1 eval 完成 | 待: Track X X3.A 训练 |
| **gf3** | 1 H20 | active | smoke/dev |
| **gf0** | 控制平面 | active | volc + uc 任务统一管理 |
| uc03 | 8 A800 | busy (task_a_new_100, nw=32) | 不动 |

**当前可启动的并发上限** (2026-05-22 晚):
- Beijing: 5 H20 free (我占 48 / 系统占 51 / 总 56)
- Shanghai: 4 A100 free (我占 16 / 系统占 24 / 总 28)
- **uc01+uc02: 2 × 8 A800 = 16 A800 idle** ⭐ Track X 主要训练资源

**当前 Running jobs (4 个, 2026-05-22 晚)**:
| Track / Exp | 当前阶段 | Job ID | 资源 | 备注 |
|---|---|---|---|---|
| Track A (SSL Phase 0) | E0.1 ✅ done base+dagger / E0.2 待启 | — | uc02 闲 (将让位 Track X) | — |
| Track C abs single-stage | Running | t-20260522194822-sqthr | Shanghai 16 A100 | xvla_actcond_single_stage_joint, vis × 7 balanced |
| Track C × delta variant | Running | t-20260522195640-t42hs | Beijing 16 H20 | Action Cond × delta-action 对照 |
| pi05 delta Task_A/base | Running | t-20260522192932-cldrd | Beijing 16 H20 | kai-only delta baseline (no cond) |
| E3.6 per-DS norm no cond | Running | t-20260522201522-s72th | Beijing 16 H20 | norm ablation |
| **Track X X3.A** ⏳ prep | HF ckpt 下载 / env setup | — | **uc01+uc02 16 A800** (next) | X-VLA 官方 0.9B, 3-domain (A+B+C) curriculum |
| **Track X X3.B** ⏳ pending | 等 X3.A 完成 | — | 同上 (sequential) | X-VLA 官方, 2-domain (A+B, no XVLA) |

**已知踩坑** (2026-05-22):
- JAX 多机 env var: train.py:411 读 **`JAX_PROCESS_INDEX`**, 不是 `JAX_PROCESS_ID` / `JAX_PROCESS_COUNT`
- cnsh container 不能访问 github/astral.sh → uv install fail; workaround: rsync `/home/tim/.local/share/uv` 到 vePFS-cnsh + symlink
- cnbj 镜像缺 ffmpeg → vis_v2_merged mp4 解码 fail; entrypoint 加 apt-get install ffmpeg
- vis_v2_merged 数据 frame index 与 mp4 mismatch (skip ~1.4% samples, training 仍推进)
- cnbj/cnsh git checkout 易 stale → 推荐 HTTPS remote 避免 SSH key 依赖

> 控制平面: 所有 volc + uc 任务通过 **gf0** 统一管理 (见 [training_servers_knowledge_base.md §5.6.c-d](./training_servers_knowledge_base.md))。

### 9.2 XVLA-Soft-Fold 多地副本

| 服务器 | 路径 | 用途 | 状态 (2026-05-21) |
|---|---|---|---|
| **uc02 本地** | `/data/tim/datasets/xvla_soft_fold/` | 原始下载位置 | ✅ 完整 (1729 files, 444G) |
| **uc01/02/03 NFS** | `/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/xvla_soft_fold/` | uc 集群训练用 (走 NFS 到 uc01 disk) | ✅ 完整 (1729 files, 444G) |
| **gf0 vePFS-cnsh** | `/vePFS/tim/xvla/data/xvla_soft_fold/` | robot-task (cn-shanghai) volc job 共享 | 🔄 下载中 (gf0 ← hf-mirror, ~7h ETA) |
| **gf3 vePFS-cnbj** ⭐ | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/xvla/data/xvla_soft_fold/` | **Robot-North-H20** (cn-beijing) 集群 job 共享 | 🔄 下载中 (gf3 ← hf-mirror, ~8h ETA) |

> gf3 副本到位后, Phase 1 SSL pretrain on Robot-North-H20 集群 jobs 挂 vePFS-cnbj 即可见 XVLA。

### 9.3 数据 sync 架构 (TOS 为中心枢纽)

完整架构见 [training_servers_knowledge_base.md §6](./training_servers_knowledge_base.md):
```
[sim01] → TOS → {uc01-03, gf0, gf3}
   (源)        (训练消费者)
```

KAI0 原始数据从 sim01 上传 TOS, 各训练服务器从 TOS 拉到本地 mirror。

---

# Part IV — 跟踪 + 风险

## 10. 状态跟踪 (持续更新)

### 10.1 Track A Phase 0 — 数据预处理 🔄 in_progress (启动 2026-05-21)

| Sub-task | 状态 | 启动 | 完成 | 备注 |
|---|---|---|---|---|
| **环境安装** (uc02 kai0 venv) | ✅ done | 2026-05-21 06:05 | 2026-05-21 06:14 | cotracker3 (local git clone + uv pip -e), decord 0.6.0, einops 0.8.1, opencv 4.11, pyarrow 20.0. CoTracker3 ckpt 从 hf-mirror 下载 (96MB) |
| **真实视频 timing 实测** | ✅ done | 2026-05-21 06:19 | 2 ep 123s = **60s/ep × 3 view** | 8 GPU 并行预估总 ~17h |
| **E0.1 Kai0_base** (3055 ep) | ✅ done | 2026-05-21 06:20 | 2026-05-21 12:22 | uc02 8 GPU 并行, 实际 6h02. 输出 3055 ep / **2.0G** tracks |
| **E0.1 Kai0_dagger** (3457 ep) | ✅ done | 2026-05-21 12:23 | 2026-05-22 ~04 UTC | uc02 8 GPU, 共 6512 ep tracks 完成 (kai0_base + dagger), uc02 现 idle |
| E0.1 vis_v2_merged (895 ep) | 待启动 | — | — | 同上 |
| E0.1 XVLA-Soft-Fold (1729 ep) | 待启动 | — | — | hdf5 格式, 需不同 dataset adapter |
| E0.2 RAFT optical flow | 待启动 | — | — | 待 E0.1 完成, 复用 uc02 GPU |
| E0.3 SAM2 cloth mask | ✅ **done** | 2026-05-21 17:40 UTC | 2026-05-22 01:13 UTC | Robot-North-H20 1 节点 (t-20260521174041-8nhps), 6512 ep × 3 view, 19534/19536 npz, 输出 `/vePFS-North-E/.../ssl_phase0/masks/` |
| ~~E0.4 FOV alignment~~ | ❌ **取消** | — | — | 不可持续 (见 §8.2 + §11 #3), 由 view-cond token + RandomResizedCrop 替代 |
| ~~E0.5 EE-relative action~~ | ❌ **取消 (2026-05-22)** | — | — | EE-relative 路线整体 deprioritize, 由 Soft Prompt (Track X 官方) + Action Head Cond (Track C) 替代 |
| **Phase 0 整体** | 🔄 in_progress | 2026-05-21 | — | 修正 ETA: ~3-5 day (E0.4 + E0.5 取消后减少 ~8h) |

### 10.2 Phase 1 — SSL Pretrain ⏳ pending Phase 0

| Exp | 状态 | Job ID | Val Loss | Downstream MAE | 备注 |
|---|---|---|---|---|---|
| E1.1 V-JEPA baseline | — | — | — | — | 待 Phase 0 |
| E1.4 + track + flow + xview | — | — | — | — | 待 Phase 0 |
| E1.5 Full multi-objective | — | — | — | — | 待 Phase 0 |

### 10.3 Phase 2 — Dynamics ⏳ pending Phase 1

| Exp | 状态 | Job ID | Val Loss | MMD A↔B | 备注 |
|---|---|---|---|---|---|
| E2.1 Latent dyn baseline | — | — | — | — | 待 Phase 1 |
| E2.2 + Embodiment cond | — | — | — | — | 待 Phase 1 |
| E2.3 + Motion-residual | — | — | — | — | 待 Phase 1 |
| E2.4 + Inverse dyn aux | — | — | — | — | 待 Phase 1 |

### 10.4 Phase 3 — Policy + Final Ablation Table ⏳ pending Phase 2

| Variant | Visual | Dynamics | Soft Prompt | Action Head Cond | Motion-residual | Val MAE | 真机平滑度 | 真机成功率 |
|---|---|---|---|---|---|---:|---:|---:|
| **E3.0** baseline (π0.5 default) | — | — | — | — | — | TBD | TBD | TBD |
| **E3.1** + Visual SSL | E1.5 frozen | — | — | — | — | ? | ? | ? |
| **E3.2** + LoRA tune | E1.5 LoRA | — | — | — | — | ? | ? | ? |
| **E3.3** + Dynamics | E1.5 LoRA | E2.3 | — | — | ✓ | ? | ? | ? |
| **X3.A** Track X (X-VLA 官方 3-domain) ⭐ | Florence2 | — | ✓ | — | — | ? | ? | ? |
| **X3.B** Track X (X-VLA 官方 2-domain) | Florence2 | — | ✓ | — | — | ? | ? | ? |
| **C3.0** Track C (Action Head Cond only) ⭐ 新 | π0.5 default | — | — | ✓ | — | ? | ? | ? |
| **E3.7** Soft Prompt+SSL | E1.5 LoRA | — | ✓ | — | — | ? | ? | ? |
| **E3.8** ⭐ 新 Action Head Cond only + SSL | E1.5 LoRA | — | — | ✓ | — | ? | ? | ? |
| **E3.9** ⭐ 新 Dual Cond (Soft + Action Head) + SSL | E1.5 LoRA | — | ✓ | ✓ | — | ? | ? | ? |
| **E3.4** Full Stack (终态) | E1.5 LoRA | E2.3 | ✓ | ✓ | ✓ | ? | ? | ? |

(待填)

### 10.5 Track C — Action Head Cond Token (方案 A) **修订: 单阶段 balanced** (2026-05-22 PM)

> **方案选定**: 4 候选 (A/B/C/D) 中选 **A (Concat domain token at action expert input)**, B/C/D 搁置。
>
> **架构修订 (2026-05-22 PM)**: 经讨论 (§6.3.6 信号路径分析), **放弃 3-stage curriculum, 改单阶段 joint training**。理由:
> - Track C 方案 A 信号注入在 action expert input (仅 4-8 层), 信号路径远比 Soft Prompt (24 层 PaliGemma) 短, Stage 2 freeze-backbone 边际价值低
> - 训练时间减半 (~12h vs ~24h)
> - 实证验证 "stage 必要性" 也是 paper 加分项 (单 stage 行就 paper 说明 Track C 简单到不需 curriculum)
>
> **采样平衡**: kai 6512 ep vs vis 895 ep (7.27× 不平衡) → **datasets_yaml vis × 7** (ConcatDataset 重复路径) → 49/51 sample ratio。详见 stage3_kai_vis_joint_balanced.yaml。
>
> **训练数据**: kai+vis joint 7407 ep (vis × 7 后 12777 ep index space)。
>
> **真机评估**: vis (B 真机)。

| 步骤 | 状态 | Job ID | Start | End | Step | Best Val | 备注 |
|---|---|---|---|---|---|---|---|
| Phase 1.5 代码实现 | ✅ **完成** | commits 4050336 + 81d2ec8 + 5f18e3f | 2026-05-22 | 2026-05-22 | — | — | pi0_config / pi0.py / weight_loaders / configs / datasets_yaml 全套实现 + balanced sampling. 旧 ckpt 完全兼容 |
| Smoke test (kai+vis mixed) | ✅ **PASS** | uc01 actcond_smoke | 2026-05-22 04:21 | 2026-05-22 04:34 | 50 / 100 | — | uc01 8 A800 batch 16. **mu d0 absmax=7.35e-5 L2=5.57e-4** (grad flow OK) |
| ~~3-stage curriculum (S1/S2/S3)~~ | ❌ **2026-05-22 PM 弃用** | — | — | — | — | — | 用户决策: action expert 端信号路径短, 不需 curriculum. 改单阶段 |
| **Single-stage balanced** | 🔄 running | t-20260522160619-flgmf | 2026-05-22 16:06 UTC | — | — / 50k | — | Shanghai 16 A100. kai_base + kai_dagger + vis × 7 joint (datasets_yaml). pi05_base init. ETA ~12h |
| **Track C 整体 (C3.0 终态)** | 🔄 single-stage running | — | 2026-05-22 | — | — | — | 训练 ~12h. 终 ckpt → vis 真机评估 |

> Track C (方案 A) 作为 paper ablation 提供 "action expert 端 domain conditioning" 数据点 (vs Track X 主线的 X-VLA 原生 Soft Prompt 实现)。

### 10.6 Track X — X-VLA 官方架构 Native 训练 ⏳ pending (2026-05-22 晚 启动规划)

#### 10.7.1 Phase 0 数据/环境准备 (X3.A + X3.B 共用)

| 阶段 | 状态 | 备注 |
|---|---|---|
| **E0.6** Joint→EE6D action 转换 | ⏳ 待启 | 复用 `calib/piper_fk.py` + Rot6D 编码, 1 day. 输出 KAI0 + vis 的 20D EE6D action |
| **E0.7** XVLA-Soft-Fold 格式适配 | ⏳ 待启 | hdf5 → LeRobot 或 X-VLA 原生, 0.5 day. **仅 X3.A 需要**, X3.B 跳过 |
| **E0.8** Mixed dataset YAML | ⏳ 待启 | 0.5 day. 两组各写一份: `mixed_3domain.yaml` (A+B+C) + `mixed_2domain.yaml` (A+B) |
| **E0.9** X-VLA env + 官方 ckpt | ⏳ 待启 | conda env XVLA + HF `2toINF/X-VLA` 拉到 vePFS-cnbj, 0.5 day |
| **Phase 0 合计** | — | ~2.5 day (X3.B 可减 0.5 day 因不需 XVLA 适配) |

#### 10.7.2 Exp X3.A — 3-domain Curriculum (A + B + C → vis adaptation)

> 2-stage curriculum: Stage A (continual pretrain on A+B+C balanced 1:7:2) → Stage B (vis-only adaptation)

| 阶段 | 状态 | Job ID | 起 | 终 | Step | Best Val | 备注 |
|---|---|---|---|---|---|---|---|
| **X3.A.SA** Stage A Continual Pretrain | ⏳ 待启 | — | — | — | — / 20k | — | Init: lerobot/xvla-base. Data: A+B+C balanced 1:7:2 (eff 16,235 ep). 16 H20, ~15h. 内部 freeze_steps=1000 |
| **X3.A.SB** Stage B vis-only Adapt | ⏳ 待启 | — | — | — | — / 10k | — | Init: X3.A.SA 输出. Data: B (vis) only. LR 5e-5, freeze_steps=500, 监控 val MAE 选 best. 16 H20, ~8h |
| **X3.A.eval** vis 真机评估 | ⏳ 待启 | — | — | — | — | — | force domain_id=20 (vis), 30 ep + 3 OOD |
| **X3.A 整体** | ⏳ pending | — | — | — | — | — | ~23h training + 1 day eval |

#### 10.7.3 Exp X3.B — 2-domain Curriculum (A + B → vis adaptation, **不含 XVLA**)

> 同 X3.A 流程, 仅 Stage A 数据少 C (XVLA), domain 数 2 (vs 3)

| 阶段 | 状态 | Job ID | 起 | 终 | Step | Best Val | 备注 |
|---|---|---|---|---|---|---|---|
| **X3.B.SA** Stage A Continual Pretrain | ⏳ 待启 | — | — | — | — / 20k | — | Init: lerobot/xvla-base. Data: A+B balanced 1:7 (eff 12,777 ep). 16 H20, ~12h. 内部 freeze_steps=1000 |
| **X3.B.SB** Stage B vis-only Adapt | ⏳ 待启 | — | — | — | — / 10k | — | Init: X3.B.SA 输出. Data: B (vis) only. 同 X3.A.SB config. 16 H20, ~8h |
| **X3.B.eval** vis 真机评估 | ⏳ 待启 | — | — | — | — | — | force domain_id=20 (vis), 30 ep + 3 OOD |
| **X3.B 整体** | ⏳ pending | — | — | — | — | — | ~20h training + 1 day eval |

#### 10.7.4 Track X 关键决策点 (XVLA 数据贡献 ablation)

| 决策点 | 触发条件 | 行动 |
|---|---|---|
| **D1**: X3.A vs X3.B 真机评估对比 | 两实验都完成 | 量化 XVLA (C) 的贡献价值 |
| D1.结果 X3.A > X3.B | XVLA 增益显著 | 终态采用 X3.A 配置 (3-domain) |
| D1.结果 X3.A ≈ X3.B | XVLA neutral | 简化为 X3.B (减少复杂度) |
| D1.结果 X3.A < X3.B | XVLA 反而 dilute | 终态采用 X3.B, XVLA 仅用于 Track A SSL pretrain |

#### 关键路径 (启动顺序)

```
[E0.6-E0.9 数据预处理] ─── 2.5 day
        ↓
[X3.B.SA Stage A: A+B continual] ─── ~12h    ← 优先启 (pipeline 验证, 无 XVLA 依赖)
        ↓
[X3.B.SB Stage B: vis adapt]    ─── ~8h
        ↓
[X3.B 真机评估]                  ─── 1 day
                                          ↓
[X3.A.SA Stage A: A+B+C continual] ─── ~15h
        ↓
[X3.A.SB Stage B: vis adapt]    ─── ~8h
        ↓
[X3.A 真机评估]                  ─── 1 day
        ↓
[D1 决策: XVLA 是否有增益?]
```

**总周期**: ~5-7 day on Robot-North-H20 16 H20 (含数据预处理 + 真机)

**关键依赖**:
1. **E0.6 Joint→EE6D** 完成 (用 PiperFK + Rot6D, 1 day)
2. **E0.7 XVLA-Soft-Fold 格式适配** 完成 (仅 X3.A 需要)
3. **X-VLA env + 官方 ckpt** 部署到 vePFS-cnbj
4. **Stage A 完成后** 立即接 Stage B (用 Stage A 输出 ckpt 作为 init)

---

## 11. 风险预警 + 关键陷阱

| # | 风险 | 应对 |
|---|---|---|
| 1 | CoTracker3 在 heavy occlusion (crumpled cloth) 失败 | Pseudo-track 加 confidence filter; track loss 按 mask 加权 |
| 2 | RAFT 在 fast motion 失败 | Quasi-static 阶段训 flow, dynamic 阶段降权重 |
| 3 | **D435 FOV (69°) < D405 (87°)** Wrist sensor gap | ❌ ~~输入端 D405 crop 到 D435 FOV~~ (不可持续, 训练-推理双向维护, 跨相机不通用, 丢失 D405 周边信息). ✅ **改 representation-level invariance**: (a) view-conditioned token (data loader 标 `view_id`, E1.4/E1.5 xview head 自学); (b) RandomResizedCrop augmentation (scale 0.6-1.0) 让 model 自然 robust |
| 4 | π0.5 PaliGemma backbone continual SSL pretraining 易 catastrophic forget | Layer-wise lr decay, peak 5e-5, anchor loss on 1% LAION subset |
| 5 | 叠衣 success criterion 真机评估难自动化 | 设计 IoU / fold count / stage completion 离线 metric |
| 6 | IK 在 delta EE 推理时不连续 | Warm-start with current joints, 或训练同时输出 delta EE + delta joints |
| 7 | EE-relative 丢失绝对工作空间位置信息 | 加 base→top_camera frame 的 anchor token (从 hand-eye calibration 得来) |
| 8 | Multi-objective loss 不收敛 | Phase 1 先单 V-JEPA 5k step 预热, 再逐项加 |
| 9 | Embodiment cond 在 visual 还是 dynamics? | **Phase 1 visual 不区分 view 来源**, Phase 2 dynamics 才区分 (visual 要 invariant, dynamics 要 partition) |
| 10 | Phase 0 (CoTracker) 慢 | Temporal stride 3 + batch size 优化 + 8-GPU 并行, 实测 ~17h |

---

## 12. 决策点

### 决策点 1: 是否引入 dagger?
- L1 (SSL): ✅ 引入 (3457 ep 增加 vision diversity)
- L2 (policy): ❌ 不引入 (抖动 +62%, 污染 action prior)
- L3 (aux): ⚠️ 可选 (作为 inverse dynamics 目标)

### 决策点 2: Embodiment conditioning 实现方式? ✅ **重新决策 (2026-05-22, 二次更新)**
- ~~Hard prompt only~~ (信号沿 LLM attention 隐式传播, 不显式 gate)
- ✅ **Soft Prompt (X-VLA 官方原生)** — Track X, 在 VLM input 端 (Florence2 + SoftPromptedTransformer, X-VLA-0.9B 官方 ckpt)
- ⭐ **Action Head Cond Token (方案 A)** — Track C 选定, 在 action expert input 端 concat 1 domain token (待加, 见 §6.3.1)
- ~~方案 B (FiLM) / C (adaLN) / D (Cross-attn)~~ **2026-05-22 暂搁置** — 4 选 1 后选定方案 A, 工程最简 + paper 与 Soft Prompt 1:1 sparse-prefix 对照
- ~~终态 E3.9 双端 Soft + Action Cond~~ **暂搁置** — 待 E3.7 (Soft only) vs E3.8 (Action Cond only) 单端结果出来再决定是否启用双端
- 真机评估: 全部 Track C/X 终态在 **vis (B 真机)** 测试

### 决策点 3: EE-relative action 是否启用? ❌ **已 deprioritize (2026-05-22)**
- ~~Phase 0 E0.5 EE-relative preprocessing~~ 取消
- ~~Phase 3 E3.4 / E3.8 delta EE~~ 取消
- 理由: R 腕 21° paired shift (§3.3) 由 Soft Prompt + Action Head Cond 处理, 工程量更低 + 无 IK 不连续风险
- 保留作为远期 backup, 如 conditioning 路线效果不佳再启用 (§5 内容保留作参考)

### 决策点 4: 是否回看 M1 短期方案?
- 触发条件: Phase 1 (SSL) + Track C / Track X 完成首轮 ablation, 如果 E3.1 / E3.7 / E3.8 已经超过 baseline → M1 不需要做
- 否则: 回看 B oversample 修复抖动 (EE-relative 已 deprioritize, 不再回看)

---

## 13. 修订历史

| 日期 | 内容 |
|---|---|
| 2026-05-22 (晚 四次修订) | **Track X 切到 2-stage Curriculum (用户决策)**: 改用 "**continual pretrain + single-domain adaptation**" 模式 — Stage A 从 lerobot/xvla-base 续训 (multi-domain mixed, balanced sampling, 20k step) → Stage B 用 vis only 短 adaptation (10k step, LR 5e-5 防 overfit, freeze_steps=500, 监控 val MAE 选 best). 这是 X-VLA "Phase I' continual extension + Phase II adaptation" 路线, 与论文 Phase I/II 框架对齐。**只跑 X3.A + X3.B 两组**, 不做 single-training 对照 (节省资源). domain_id 改用 base 中未占用 slot (19=A, 20=B, 21=C). 更新 §8.8.3 / §8.8.4 / §10.7.2 / §10.7.3 |
| 2026-05-22 (晚 三次修订) | **Track X 训练 pipeline 改为论文 1:1 + 加 X3.A/X3.B 对照**: 修正之前规划 (取消独立 2-phase + Phase 3 vis-only lock, 取消偏离论文做法). 改用 X-VLA 论文/LeRobot 官方 single training (20k step + `freeze_steps=1000` 内部前 1k step prompt warmup + `learning_coef=0.1` VLM LR scaling). 加 §8.8.3.1 两组对照实验: **X3.A** (3-domain A+B+C balanced 1:7:2, ~15h) vs **X3.B** (2-domain A+B balanced 1:7, ~12h) 量化 XVLA 数据贡献. §8.8.4 改为 single training + Phase II 内部 schedule. §10.7.2/10.7.3 改为单 stage 状态表 + 加 §10.7.4 决策点 D1. LeRobot 集成现行推荐: 不冻 VLM, train transformer + soft prompts + VLM (LR × 0.1), action_mode=auto, dtype=bf16 |
| 2026-05-22 (晚 二次修订) | **明确 3 异构机器人 + 唯一部署目标 vis**: 头部 banner 重写, 加 "3 robot heterogeneous" 表 (§1.1: A=KAI0 dom_id=0 / B=vis dom_id=1 ⭐部署 / C=XVLA dom_id=2); §8.8.3 数据池表加 domain_id + 部署列, balanced sampling 设计 (A:B:C = 1×:7×:2×, vis ×7 上采样确保部署 prior); §8.8.3.1 推理时 force `domain_id=1`; §8.8.6 真机评估明确 vis-only, X-VLA SoftFold (同硬件) baseline 100% |
| 2026-05-22 (晚 战略转向) | **❌ Track B 完全终止 + ⭐ Track X 启动 (X-VLA 官方架构 native 训练)**: 经实证 Track B (pi0.5 + 移植 soft prompt) Stage 2 × 3 fail + e3-6 × 2 fail, 嫁接式不稳。改走 X-VLA 官方完整架构 (Florence2 + 24-layer SoftPromptedTransformer + EE6D 20D action + X-VLA-0.9B 官方 ckpt), 与论文 1:1 一致, 仅适配 KAI0+vis 数据。新增 §8.8 Track X 完整计划 + §10.7 状态跟踪表; §8.7 Track B 标注 DEPRECATED + §10.5 标注完全终止。Phase 0 新增 E0.6-E0.9 (joint→EE6D, XVLA 格式, mixed yaml, env+ckpt)。本地已有官方 repo `/home/tim/workspace/X-VLA/`. 任务 #16 删除 + #17 新建 Track X. ICLR 2026 + IROS 2025 Champion + SoftFold-Agilex (同硬件) 100% 验证 |
| 2026-05-22 (深夜) | **§6.4 RTC / TAC 实时性方案对比与集成计划**: 整理 3 篇 RTC 论文 (Inference RTC 2506.07339, **TAC 2512.05964 ⭐**, A2C2 2509.23224) 维度对比; 确认本地 `pi0_rtc.py` 已 1:1 复刻 Inference RTC, 缺 TAC training path; 移植方案: Algorithm 1 复刻 (~6 行 compute_loss 改 + adaLN per-token broadcast), Pi0Config 加 `tac_enabled` flag, 0 新参数; 复现难易度 ⭐⭐⭐⭐⭐ (算法/超参全披露, 不依赖闭源 π0.6 ckpt); 加 §6.4.9 Phase 3 ablation 新增 E3.RTC1-RTC4 行 (Inference RTC / TAC / TAC+RTC / TAC+A2C2); A2C2 暂搁置 (等 TAC 结果) |
| 2026-05-22 (PM 二次决策) | **Track C 改单阶段 balanced + Track B 终止 Stage 2/3 + E3.6 提交**: 经 §6.3.6 信号路径分析, Action Cond 方案 A 在 action expert input 端的信号路径远比 Soft Prompt 短 (4-8 层 vs 24 层), Stage 2 freeze-backbone 边际价值低 → 弃用 3-stage curriculum, 改单阶段 joint training (kai+vis 50k step, balanced sampling vis × 7). 训练时间 24h → 12h. Track B Stage 2 (6fr6c) + 3-stage curriculum 整体终止, 仅保留 Stage 1 ckpt 49999 作 paper E3.7 baseline. 新提交: E3.6 per-DS norm + no cond (Beijing 16 H20, n98pl) + Track C single-stage balanced (Shanghai 16 A100, flgmf) |
| 2026-05-22 (晚) | **Track C 方案 A 选定 + B/C/D 搁置**: 4 候选 Action Head Cond 方案 (A Concat / B FiLM / C adaLN / D Cross-Attn) 详细对比后, 选 **方案 A** (Concat domain token at action expert input)。理由: 工程最简 + 与 Soft Prompt 形成 1:1 sparse-prefix 对照 (不同模块、相同设计模式), paper E3.7 vs E3.8 直接量化 "VLM 端 vs Action expert 端" 注入点选择。B/C/D 暂搁置作技术参考。E3.9 双端组合也搁置, 待单端结果出来再决定。Track C 训练用 kai+vis 跨本体混合, 真机评估用 vis B 真机。§6.3 / §6.2 / §3.5.7 / §10.4 / §10.6 / 决策点 2 同步更新 |
| 2026-05-22 (中) | **Tri-track + Action Head Cond 启用 + EE-relative deprioritize**: §6.3 新增 Track C Action Head Conditioning Embedding (含 4 候选方案) 与 Track B Soft Prompt 互补; §10.6 Track C 状态跟踪表; §3.5.7 Phase 3 ablation 新增 E3.8 / E3.9; §10.4 ablation 表新增 C3.0/E3.7/E3.8/E3.9 列; §10.1 E0.5 + Phase 3 E3.8 delta EE 全部取消; §5 整节标 deprioritized 保留作参考; §7 三轨架构图. SAM2 (E0.3) 状态 ✅ done. 资源更新: robot-task 20 A100 free 已可用 |
| 2026-05-21 (深夜) | **§3.5 vis operator + 时间漂移分析**: 澄清 ztm+lym 同一人 (G1=872 ep, G2=gsy=23 ep); G1 内时间漂移 0.47 ≈ cross-robot drift; gsy 对 norm 影响微弱 (0.08 rad). **§3.6 混训策略 6 方案 + 实证一致性**: per-dataset norm 实测 MMD 降低 90.7% (0.06→0.006), **修正方案 B 评级 ⭐⭐→⭐⭐⭐ (实际可行!)**; 识别残余 10% 来自 higher moments + joint correlation; 推荐 layered (E + C + D); §3.6.7 加 E3.5-E3.8 ablation 验证 naive joint vs per-DS norm |
| 2026-05-21 (晚) | **Dual-track 化 + 放弃 FOV crop**: 加 §6.2.1 本地 soft_prompt_hub 代码 + ckpt 现状 (代码已实现但未训过); §6.2.2 X-VLA 3-stage 流程; §7 dual-track 架构图 (Track A SSL + Track B X-VLA 并行); §8.7 Track B 完整 X-VLA stage 1/2/3 计划 (Stage 1 t-20260521154828-76d44 已提交); §10.5 Track B 状态跟踪表; §10.4 加入 B3.0 + 改 E3.4 为 dual-track merge; 决策点 2 已决策为 soft prompt. **取消 E0.4 Wrist FOV crop** — 不可持续, 替换为 view-cond token + RandomResizedCrop (§8.2 + §11 #3) |
| 2026-05-21 (早) | **Consolidated**: 合并 `ssl_pretraining_experiment_plan.md` 到本文档 §8; 删除 X1/X2/X3 详细配置 (deprioritize M1); 加 §6 与 π0.5/X-VLA 默认对照 + 实证调研; 加 §4 假说矩阵 H1-H4; 加 §10 状态跟踪 |
| 2026-05-21 (earlier) | 加 XVLA-Soft-Fold 多地副本 (§9.2: uc02 本地 + uc NFS + gf0 vePFS-cnsh + gf3 vePFS-cnbj) |
| 2026-05-19 | 初版: 设备差异 + 4 层 ROI + EE-relative 可行性 + M1-M4 milestones + Qizhi 资源分配 |
