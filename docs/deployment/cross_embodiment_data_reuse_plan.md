# 跨本体数据复用 — 战略与执行计划 (Consolidated)

> **更新时间**: 2026-05-21
> **作者**: Tim + 综合外部研究员讨论 + 项目实测数据
> **背景**: A=官方 KAI0 双臂 piper (D435 wrist), B=自有双臂 piper (D405 wrist), 共享同一型号机械臂但 wrist 相机+机械装配有差异。当前实测发现 naive 混合训练 (A+B) 在 B 真机上抖动**反而**超过纯 B 训练 → 需要系统化复用方案。
> **目标**: 把 A 的 **6,512 ep** + XVLA-Soft-Fold **1,729 ep** 数据价值榨干, 同时不污染 B 真机部署性能, 并为 CoRL/NeurIPS paper 铺路。

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
  - [8. M2 Dual-Track 详细计划 (Track A SSL Phase 0-4 + Track B X-VLA Stage 1-3)](#8-m2-ssl-pretraining-详细-phase-0-4)
  - [9. 资源 + 数据 + 网络](#9-资源--数据--网络)
- [Part IV — 跟踪 + 风险](#part-iv--跟踪--风险)
  - [10. 状态跟踪 (持续更新)](#10-状态跟踪-持续更新)
  - [11. 风险预警 + 关键陷阱](#11-风险预警--关键陷阱)
  - [12. 决策点](#12-决策点)
  - [13. 修订历史](#13-修订历史)

---

# Part I — 战略评估

## 1. Embodiment Gap 定量

### 1.1 共享 (跨本体的"同"部分)

| 维度 | A (官方 KAI0) | B (自有) |
|---|---|---|
| 机械臂型号 | piper 双臂 (6 DOF + gripper) | 同 |
| Joint DOF | 14 (7×2 含 gripper) | 同 |
| 控制频率 | 30 Hz | 同 |
| Top 头部相机 | RealSense D435 | 同 |
| Top 高度 | ~76 cm | 同 (略差) |
| Top 俯视角度 | ~30° | 同 (略差) |
| Action 语义 | "Flatten and fold the cloth." | 同 |

### 1.2 差异 (按影响从大到小)

| Gap 类型 | A | B | 量化差异 | 严重度 |
|---|---|---|---|---|
| **Wrist 相机** | RealSense **D435** (RGB FOV 69°×42°, rolling, min depth 28cm) | RealSense **D405** (RGB FOV 87°×58°, global shutter, min depth 7cm, RGB-IR 共光路) | brightness ↓12%, sharpness ↓31%, 近距 RGB-D 行为完全不同 | 🔴 **最严重** |
| **Wrist 安装** | 一致设计 | 一致设计但高度/角度略差 (毫米级) | 待精确测量 | 🟠 第二严重 — wrist view 物体 pos/scale 受影响 |
| **双臂间距** | 标准 | 略差 (毫米级未测) | state joint3 std +61%, joint5 std +83% | 🟡 absolute EE 时 systematic bias |
| **Top 相机** | 标准 | 略差 (<5cm, <5°) | mean brightness 接近 | 🟢 < 5cm/5° 时 augmentation cover |

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
| **Soft Prompt** (X-VLA, Track B) | ✅ 显式 condition, 学到 domain shift | ⚠️ 隐式 | ⭐⭐⭐ |
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

Layer 2 (Policy, Phase 3 / Track B):
  C. Soft Prompt + Per-DS norm — 显式 routing
  + D. Curriculum (二阶段 A→B finetune)
  → 显式 routing + lock 到 B

Layer 3 (Phase 3 ablation):
  F. EE-based — joint vs EE 控制变量实验
  → paper ablation 数据点
```

#### 3.5.7 验证假说的新 Ablation Set (Phase 3 加)

| Exp | Action | Norm 策略 | 用途 |
|---|---|---|---|
| **E3.0** baseline | absolute joint | per-dataset (single ds smooth_800) | 当前 SOTA |
| **E3.5** | absolute joint | **Naive joint norm** (A) | 故意复现失败假说 |
| **E3.6** | absolute joint | **Per-dataset norm + Single model** (B) | **验证 §3.5.5 insight** |
| **E3.7** | absolute joint | Per-DS norm + Soft Prompt (C) | Track B 路线 |
| **E3.8** | delta EE | Per-DS norm + Soft Prompt | + F 维度 ablation |

→ E3.5 vs E3.6 直接量化 "naive joint vs per-dataset norm" 的真机抖动差异。

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

## 5. EE-relative Action 可行性

### 5.1 可用资源

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
| **Soft prompt** (X-VLA style, 每 domain 32×2048 learnable vec) | 0 (代码已实现) | ✅ **`pi0.py:136-181` 已有 `soft_prompt_hub` 实现** + `xvla_stage1/2/3` config 模板就绪 | ⭐⭐⭐ **正确路径** — 显式 inject 到 LLM input, 推理时可 hard-force domain |
| **Action head embedding** | 中 (改 model arch) | ❌ 待加 | ⭐⭐ 与 soft prompt 互补 |
| **Soft prompt + Action head emb 结合** | 中-高 | 部分 | ⭐ 终态最强 |

#### 6.2.1 本地代码与 ckpt 现状 (2026-05-21 实测)

```python
# kai0/src/openpi/models/pi0.py:136-181 (已实现, 默认禁用)
if config.soft_prompt_num_domains > 0 and config.soft_prompt_len > 0:
    self.soft_prompt_hub = nnx.Embed(
        num_embeddings=config.soft_prompt_num_domains,       # A=0, B=1, [XVLA=2]
        features=config.soft_prompt_len * paligemma_width,   # 32 × 2048 = 65536 per domain
    )
# forward 时:
soft = self.soft_prompt_hub(obs.dataset_id)
soft = soft.reshape(B, soft_prompt_len, llm_width)            # (B, 32, 2048)
# Prepend 到 LLM input → 与 X-VLA 论文 1:1 一致
```

| Ckpt | soft_prompt_hub weights? | 说明 |
|---|---|---|
| pi05_base, mixed_1, smooth_800, pure_200 | ❌ 无 | 全部 `soft_prompt_num_domains=0` 默认禁用 |
| xvla_stage1/2/3 config 模板 (line 1059-1146) | ✅ 已配置 `num_domains=2, len=32` | **从未实际执行训练**, 直到 2026-05-21 t-20260521154828-76d44 |

#### 6.2.2 X-VLA 3-stage 训练流程 (config.py 已就绪)

```
Stage 1: xvla_stage1_kai_warmup
   Init: pi05_base (干净起点) + soft_prompt_hub initialized N(0, 0.02)
   Data: kai0_base + kai0_dagger (全部 domain_id=0)
   全模型 + soft_prompt 联合训练, 50k step
   → 学到 domain[0] (kai) 的 soft_prompt 表示

Stage 2: xvla_stage2_soft_prompt_only_vis
   Init: Stage 1 final ckpt
   Frozen backbone + LLM, ONLY train soft_prompt_hub
   Data: vis_v2_merged (domain_id=1, vis)
   5k step, lr 5e-4
   → 仅对齐 domain[1] (vis) 的 soft_prompt slot, 不动 backbone

Stage 3: xvla_stage3_joint_finetune
   Init: Stage 2 ckpt
   Unfreeze 全模型, joint finetune soft_prompt + backbone
   Data: kai + vis 混训
   → 终态最强模型
```

> X-VLA Soft Prompt 已在 290K episodes 跨 7 个 platforms × 5 个 arm types 验证可行 (ICLR 2026)。每域仅 65K 参数 (32×2048), 整体 0.04% 非共享参数。

---

# Part III — 执行计划

## 7. Milestone 总览 (M1-M4) — Dual-Track Parallel

### 📐 双轨并行架构 (2026-05-21 起)

```
                     ┌─────────────────────────────────────┐
                     │   Track A: SSL 主线 (uc02 + H20)    │
                     │   ├── Phase 0 Pseudo-labels         │
                     │   ├── Phase 1 V-JEPA + track + flow │
                     │   ├── Phase 2 Dynamics + Embodiment │
                     │   └── Phase 3 Policy + Ablation     │
                     ├─────────────────────────────────────┤
                     │   Track B: X-VLA Soft Prompt        │
                     │   (Robot-North-H20, 16 H20)         │
                     │   ├── Stage 1 kai warmup (in_progress) │
                     │   ├── Stage 2 vis soft_prompt only  │
                     │   └── Stage 3 joint finetune        │
                     └─────────────────┬───────────────────┘
                                       ↓
                          ┌──────── Final Merge ────────┐
                          │ SSL Visual Backbone +       │
                          │ X-VLA Soft Prompt Hub +     │
                          │ Dynamics-conditioned policy │
                          └─────────────────────────────┘
```

**两条 track 互不冲突**: Track A 主要用 uc02 (Phase 0) + Robot-North-H20 (Phase 1-3); Track B 用 Robot-North-H20 16 GPU。在 Robot-North-H20 39 GPU 可用前提下并行 OK (Track A 单 exp 16 GPU, Track B 同 16 GPU, 共 32, 余 7)。

### 🚀 M1 (1-2 周): 短期真机修复 — **已 deprioritize**

> 用户决策 (2026-05-21): 先专注 L1 SSL + X-VLA 路线 (M2-M3), 暂不展开 X1/X2/X3 系列。M1 计划保留但不立即执行, 待 M2/M3 完成首轮 ablation 后回看。

简要内容: EE-relative action + embodiment prompt + B oversample 修复真机抖动 (详细方案见 git 历史)。

### 🔬 M2 (~9 周): Dual-track 训练 ⭐ **主线**

**Track A — SSL Pretraining + Dynamics + Policy**: 完整 4 个 Phase 详见 §8.1-8.5。
- **当前进度**: Phase 0 in_progress (E0.1 Kai0_base CoTracker3 跑在 uc02)

**Track B — X-VLA Soft Prompt Curriculum (3-stage)**: 详见 §8.6。
- **当前进度**: Stage 1 in_progress (t-20260521154828-76d44, Robot-North-H20 16 H20, 2026-05-21 07:48 启动)

### 🌍 M3 (M2 之后): Dual-track Merge + 真机 + Paper

- **Merge 策略**: SSL visual backbone (E1.5) + X-VLA Soft Prompt (xvla_stage3) + Dynamics-conditioned action head
- **真机大规模测试** (60-100 ep per ablation, 含 X-VLA stage 1/2/3 各自 baseline)
- **Paper figures + writing** (中心 claim: cloth_residual MMD + ablation 表)
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
| **E0.5** EE-relative action | Python + PiperFK | A + B + XVLA actions | `action_ee_relative/{ep_id}.npz` (T, 14) | CPU | 几小时 |

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

| Exp | Visual | Dynamics | Action | Data | 用途 |
|---|---|---|---|---|---|
| **E3.0** baseline | π0.5 default | × | absolute | B (smooth_800) | 当前 SOTA, baseline |
| **E3.1** | E1.5 frozen | × | absolute | B | 测 H1 (visual repr 单独 value) |
| **E3.2** | E1.5 LoRA | × | absolute | B | 测 fine-tunable 是否更好 |
| **E3.3** | E1.5 LoRA | E2.3 frozen | absolute | B | 测 H3 (dynamics 额外贡献) |
| **E3.4** Full Stack | E1.5 LoRA | E2.3 frozen | EE-relative | B + A weighted (embodiment cond) | 终态最强 |

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

### 8.7 Track B — X-VLA Soft Prompt Curriculum (并行执行)

> 与 Track A (SSL) 完全并行。Track B 是 GR00T N1.5 / X-VLA 风格的直接 policy + soft prompt 路线, 输出可作为 Track A Phase 3 的对照 baseline + 后期 merge 提供 soft_prompt_hub 权重。

#### 8.7.1 配置就绪 (config.py 已有, 见 §6.2.1)

| Stage | Config name | Init | Data | Freeze | Steps | LR |
|---|---|---|---|---|---|---|
| **1** | `xvla_stage1_kai_warmup` | pi05_base + N(0,0.02) soft_prompt | kai0_base + dagger (domain_id=0) | None (joint train) | 50k | 1.5e-5 → 1.5e-6 |
| **2** | `xvla_stage2_soft_prompt_only_vis` | Stage 1 ckpt | vis_v2_merged (domain_id=1) | Backbone frozen, **only soft_prompt** | 5k | 5e-4 |
| **3** | `xvla_stage3_joint_finetune` | Stage 2 ckpt | kai + vis 混训 | Unfreeze all | 待定 | 待定 |

#### 8.7.2 资源 + ETA

| Stage | GPU | ETA | 总 GPU-h |
|---|---:|---:|---:|
| 1 | 16 H20 (Robot-North-H20) | ~12h | ~200 |
| 2 | 8-16 H20 | ~2h | ~30 |
| 3 | 16 H20 | ~12h | ~200 |
| **合计** | — | ~26h | ~430 GPU-h |

#### 8.7.3 当前 Stage 1 实际任务 (2026-05-21)

```yaml
Job:     xvla-stage1-cnbj-kai-warmup-16gpu (t-20260521154828-76d44)
Status:  Queueing → Running (2026-05-21 07:48 启动)
Workers: 2 × ml.hpcpni3ln.45xlarge = 16 H20
Storage: vepfs-cnbj /vePFS-North-E/vis_robot/
Verify:  ✅ 路径全部 ok (data + ckpt + yaml + venv 实测)
Config 关键:
  - soft_prompt_num_domains=2 (A=0, B=1) — ⚠️ 未给 XVLA 留槽位
  - soft_prompt_len=32
  - use_delta_joint_actions=False (与 pi05_base + mixed_1 一致)
  - inline_eval_every=99999 (实际禁用, 训完手动 eval)
  - inline_eval_val_root: kai0/dagger (in-domain, 不用 vis)
```

#### 8.7.4 Track A + B 最终 Merge (M3 Week 9-)

```python
# 最强 final model 设想 (paper E3.5 / E4.0):
TrainConfig(
    name="final_ssl_xvla_dynamics",
    model=pi0_config.Pi0Config(
        pi05=True,
        soft_prompt_num_domains=2,      # 或 3 (如果引入 XVLA)
        soft_prompt_len=32,
        # 加: motion_residual_dynamics=True (Phase 2 E2.3 学到的)
    ),
    weight_loader=CheckpointWeightLoader(
        path_to_xvla_stage3_ckpt,        # X-VLA soft_prompt_hub weights
        # + 替换 vision tower with SSL E1.5 backbone
    ),
    data=...vis_only finetune,
    use_delta_joint_actions=...,         # 决策点 (待 X1 验证)
)
```

#### 8.7.5 已知 caveats (Track B)

1. **soft_prompt_num_domains=2 没给 XVLA 留槽** — 如果将来想引入 XVLA 进 X-VLA route, 必须扩 num_domains 重训; 当前 Track B 设计是 A + B 二分类, 不含 XVLA
2. **absolute joint** — 与 §6.1 实证调研一致 (π0.5 默认 absolute + KAI0 数据 absolute + mixed_1 norm_stats 验证), 但与 "尝试 delta" 假设线不重叠 (delta 假设线只能在 Track A 内验证)
3. **inline_eval 禁用** — 训完后需要手动 eval pipeline (一次性 forward + 算 MAE)

### 8.7 时间线 (Gantt)

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

### 9.1 GPU 资源 (2026-05-22 实测可用)

| 资源 | GPU | 状态 | M2 分配 |
|---|---:|---|---|
| **Robot-North-H20** (cn-beijing) | 32-40 H20 free / 56 total | active | SSL/X-VLA 主战场 (单 exp 16 GPU, 可 2-3 并发) |
| **robot-task** (cn-shanghai) ⭐ | **20 A100-80G free / 28 total** | active | Phase 1 SSL 副战场 / 备用 (单 exp 8 GPU 起) |
| **uc02** | 8 A800 | busy (Phase 0 E0.1 CoTracker3) | 数据预处理 |
| **uc01** | 8 A800 | free (exp1 已完成) | E0.5 + offline eval 等小作业 |
| **gf3** | 1 H20 | active (smoke/dev) | 单卡 smoke / debug |
| **gf0** | 控制平面 | active | volc + uc 任务统一管理 |
| uc03 | 8 A800 | busy (task_a_new_100) | 不动 |

**当前可启动的并发上限** (2026-05-22):
- Beijing: 2 × 16 GPU job 或 1 × 32 GPU job (32 H20 free)
- Shanghai: 2 × 8 GPU job 或 1 × 16 GPU job (20 A100 free)
- uc01: 1 × 8 GPU job (8 A800 idle)

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
| **E0.1 Kai0_dagger** (3457 ep) | 🔄 running | 2026-05-21 12:23 | — | uc02 8 GPU, 每 GPU 433 ep, ETA ~4.6h (avg 699 frames/ep, 比 base 短) |
| E0.1 vis_v2_merged (895 ep) | 待启动 | — | — | 同上 |
| E0.1 XVLA-Soft-Fold (1729 ep) | 待启动 | — | — | hdf5 格式, 需不同 dataset adapter |
| E0.2 RAFT optical flow | 待启动 | — | — | 待 E0.1 完成, 复用 uc02 GPU |
| E0.3 SAM2 cloth mask | 待启动 | — | — | Robot-North-H20 1 节点跑 |
| ~~E0.4 FOV alignment~~ | ❌ **取消** | — | — | 不可持续 (见 §8.2 + §11 #3), 由 view-cond token + RandomResizedCrop 替代 |
| E0.5 EE-relative action | 待启动 | — | — | CPU + PiperFK, 脚本 `/tmp/e0_5_ee_pose.py` 已就位 |
| **Phase 0 整体** | 🔄 in_progress | 2026-05-21 | — | 修正 ETA: ~5-7 day (E0.4 取消后减少 ~5h) |

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

| Variant | Visual | Dynamics | Soft Prompt | Motion-residual | Inverse Dyn | Val MAE | 真机平滑度 | 真机成功率 |
|---|---|---|---|---|---|---:|---:|---:|
| **E3.0** baseline (π0.5 default) | — | — | — | — | — | TBD | TBD | TBD |
| **E3.1** + Visual SSL | E1.5 frozen | — | — | — | — | ? | ? | ? |
| **E3.2** + LoRA tune | E1.5 LoRA | — | — | — | — | ? | ? | ? |
| **E3.3** + Dynamics | E1.5 LoRA | E2.3 | — | ✓ | — | ? | ? | ? |
| **B3.0** Track B (xvla stage 3 alone) | π0.5 default | — | ✓ (xvla) | — | — | ? | ? | ? |
| **E3.4** Full Stack (Track A + B merge) | E1.5 LoRA | E2.3 | ✓ (xvla) | ✓ | ✓ | ? | ? | ? |

(待填)

### 10.5 Track B — X-VLA Soft Prompt Curriculum ⏳ in_progress (启动 2026-05-21)

| Stage | 状态 | Job ID | Start | End | Step | Best Val | 备注 |
|---|---|---|---|---|---|---|---|
| **Stage 1 kai warmup** | 🔄 running (mu PASS) | t-20260521154828-76d44 | 2026-05-21 07:48 UTC | — | 2100+ / 50k | — | 16 H20 on Robot-North-H20. ✅ **Step 2000 ckpt mu 验证 PASS** (d0 mu absmax=1.15e-3): RepackTransform + AgilexInputs 两处 dataset_id passthrough 修复确认生效, soft_prompt_hub 真的在训练。Rate ~1.8 s/it, ETA step 50k ~10h |
| Stage 2 vis soft_prompt only | 待 Stage 1 | — | — | — | — / 5k | — | LR 5e-4, freeze backbone |
| Stage 3 joint finetune | 待 Stage 2 | — | — | — | — | — | Joint train all |
| **Track B 整体** | 🔄 stage 1 running | — | 2026-05-21 | — | — | — | 3 stages 总 ETA ~22-26h (Stage 1 mu PASS 后流程已 unblock) |

> **2026-05-21 重大 bug 修复**: 之前 Stage 2 grad_norm=0 + 旧 Stage 1 soft_prompt_hub 不训练的根因, 是 `RepackTransform` 和 `AgilexInputs` 两处都重建 data dict 时丢掉了 `dataset_id`, 导致 obs.dataset_id=None → embed_prefix soft prompt 分支被 dead-code-eliminate。修复 commits: `9d2184a` (RepackTransform) + `df23d5a` (AgilexInputs)。

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

### 决策点 2: Embodiment conditioning 实现方式? ✅ **已决策 (2026-05-21)**
- ~~Hard prompt only~~ (信号沿 LLM attention 隐式传播, 不显式 gate)
- ✅ **Soft prompt (X-VLA style)** — 代码已实现 (`pi0.py:soft_prompt_hub`), config 已就绪 (`xvla_stage1/2/3`), 已启动 (Track B Stage 1 in_progress)
- ⏭️ Track A Phase 2 (dynamics) 同时用 soft embedding 作为 conditioning input
- 终态 (E3.4): SSL backbone + xvla soft_prompt_hub + dynamics motion-residual

### 决策点 3: 是否回看 M1 短期方案?
- 触发条件: Phase 1 (SSL) 完成首轮 ablation, 如果 E3.1 / E3.2 已经超过 baseline → M1 不需要做
- 否则: 回看 EE-relative action + B oversample 修复抖动

---

## 13. 修订历史

| 日期 | 内容 |
|---|---|
| 2026-05-21 (深夜) | **§3.5 vis operator + 时间漂移分析**: 澄清 ztm+lym 同一人 (G1=872 ep, G2=gsy=23 ep); G1 内时间漂移 0.47 ≈ cross-robot drift; gsy 对 norm 影响微弱 (0.08 rad). **§3.6 混训策略 6 方案 + 实证一致性**: per-dataset norm 实测 MMD 降低 90.7% (0.06→0.006), **修正方案 B 评级 ⭐⭐→⭐⭐⭐ (实际可行!)**; 识别残余 10% 来自 higher moments + joint correlation; 推荐 layered (E + C + D); §3.6.7 加 E3.5-E3.8 ablation 验证 naive joint vs per-DS norm |
| 2026-05-21 (晚) | **Dual-track 化 + 放弃 FOV crop**: 加 §6.2.1 本地 soft_prompt_hub 代码 + ckpt 现状 (代码已实现但未训过); §6.2.2 X-VLA 3-stage 流程; §7 dual-track 架构图 (Track A SSL + Track B X-VLA 并行); §8.7 Track B 完整 X-VLA stage 1/2/3 计划 (Stage 1 t-20260521154828-76d44 已提交); §10.5 Track B 状态跟踪表; §10.4 加入 B3.0 + 改 E3.4 为 dual-track merge; 决策点 2 已决策为 soft prompt. **取消 E0.4 Wrist FOV crop** — 不可持续, 替换为 view-cond token + RandomResizedCrop (§8.2 + §11 #3) |
| 2026-05-21 (早) | **Consolidated**: 合并 `ssl_pretraining_experiment_plan.md` 到本文档 §8; 删除 X1/X2/X3 详细配置 (deprioritize M1); 加 §6 与 π0.5/X-VLA 默认对照 + 实证调研; 加 §4 假说矩阵 H1-H4; 加 §10 状态跟踪 |
| 2026-05-21 (earlier) | 加 XVLA-Soft-Fold 多地副本 (§9.2: uc02 本地 + uc NFS + gf0 vePFS-cnsh + gf3 vePFS-cnbj) |
| 2026-05-19 | 初版: 设备差异 + 4 层 ROI + EE-relative 可行性 + M1-M4 milestones + Qizhi 资源分配 |
