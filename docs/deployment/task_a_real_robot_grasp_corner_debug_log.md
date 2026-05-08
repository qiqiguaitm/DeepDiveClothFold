# Task A 真机叠衣服 — "夹不到衣角反复尝试" 排查日志

> **问题创建时间**: 2026-04-29
> **关联训练文档**: `docs/training/task_a_visrobot01_mixed_600.md`
> **责任人**: ruitao jing
> **当前状态**: 🔴 待排查

---

## 0. 问题陈述

### 0.1 三个微调 ckpt 真机表现

| ckpt | 数据成分 | best MAE@1 | 真机表现 | "反复夹同位"现象 |
|---|---|---:|---|---|
| `mix_vis600_v1` step 38000 | 310 vis + 145 kai0_base + 145 kai0_dagger | 0.0146 | 有一定成功率 | 偶发 |
| `pure_vis600_v1` step 39999 | 309 vis + 291 hflip mirror | 0.0151 | 略逊于 mix | 较明显 |
| `vis_base_40k_v1` step 36000 | 288 vis only | 0.0168 | 显著较差 | **非常明显** |

### 0.2 核心反差：与 kai0 官方 ckpt 对比 ⚠️

| 维度 | **kai0 官方 ckpt** | **我们的全参数微调 ckpt** |
|---|---|---|
| 夹取衣角 | ✅ 高成功率 | ❌ 成功率明显下降，反复夹同位 |
| 叠衣服全流程 | ❌ 流程走不到底 | ✅ 流程顺畅可走完 |

### 0.3 已知约束（缩小排查范围）

- **同套硬件**采集训练数据和部署 → cam intrinsics / pose / 单位 / 坐标系 / gripper 阈值 等硬件 mismatch 全部排除
- **mix_vis600 已经是混合训练**（自采 + kai0 官方 base + kai0 官方 dagger），结果**仍有 corner-grasp 退化** → "简单 mix 官方数据" 这条路已经验证不够，需要更深入的方案

### 0.4 问题定性

**真问题**：全参数微调过程把官方 base 已具备的 **corner-grasp sub-skill** 擦掉/稀释了，同时换来了 flow completion 能力 — 是一次 **trade-off**，不是 data scarcity。

可能机制：
- **D1** 全参微调把权重推离了 corner-grasp 的窄峰最优
- **D2** 自采数据中 corner-grasp 帧的 label 精度不如官方（teleop 时人手未必精准对准衣角）
- **D3** 自采数据中 corner-grasp 帧占比被叠衣服后段稀释；mix_vis600 中 vis : kai0 = 2:1，kai0 比例可能已经不够
- **D5** vision tower 全解冻 + LR 1.5e-5 + 40k step 太激进，损坏视觉先验
- **D7** 训练越久越糟，plateau 后 overfit 到自采域的 corner-grasp 误差
- **B1** closed-loop 死循环：obs 几乎不变 → action chunk 几乎相同 → 反复同位（解释"反复尝试"现象，但不解释成功率退化本身）

---

## ⭐ 代码层面排查结论 (2026-04-29)

**头号 actionable**：当前部署（`temp.sh:152` 选 L）显式 `enable_rtc:=false`，但同文件 211 行注释明示 RTC 对"抓取瞬间偏"预期 +15-30%。**先 `./rtc_apply.sh on`** 5 分钟零成本试一次。

✅ 无问题：A1 norm_stats md5 一致 / A2 归一化对称 / A4 pure_vis600 mirror 自洽 / A7 chunk 策略合理。详见 §1 Phase 1。

🔴 真问题：
- **P1** 当前部署关闭 RTC （立即修）
- **P3** flow-matching 接近 deterministic → 解释"反复同位"现象
- **P4** 三套微调全部全参解冻 + LR 1.5e-5 + 40k step（命中 D5）
- **D6** 自采 vs init 数据 norm_stats: mean 偏移 12-18°，std 放大 2-2.5×；vis_base_40k 偏移最大 → 真机最差（自洽）

---

## 1. 排查 Checklist

### Phase 0 — 关键对照测试（**头号要做**，几乎决定后续路径）

#### 0.A "step vs corner-grasp 成功率" 真机曲线

在同一台真机、同一衣物初始姿态下，依次跑下列 ckpt 各 10 次叠衣服尝试，记录 corner-grasp 成功率：

| ckpt | corner-grasp 成功 | full-flow 成功 | 备注 |
|---|---|---|---|
| kai0 官方原版 ckpt | `___ / 10` | `___ / 10` | baseline (已知 corner OK / flow 差) |
| `Task_A/mixed_1/params` (微调 init) | `___ / 10` | `___ / 10` | init 是否已开始退化？ |
| `mix_vis600_v1` step 5000 | `___ / 10` | `___ / 10` | 早期 |
| `mix_vis600_v1` step 10000 | `___ / 10` | `___ / 10` | 中期 |
| `mix_vis600_v1` step 20000 | `___ / 10` | `___ / 10` | 中后期 |
| `mix_vis600_v1` step 38000 (best MAE) | `___ / 10` | `___ / 10` | 当前部署版本 |
| `mixed_gf0_173_v1` step 9000 (1:1:1 等量 mix, 短训) | `___ / 10` | `___ / 10` | **额外对照**：kai0 比例更高 + 训得更短 |

- 实测日期: `_______`
- 实测人: `_______`

**结论分支**（决定后续路径）：

| 观察到的曲线 | 成立的假设 | 下一步 |
|---|---|---|
| 单调下降，早期 step 反而比 38k 好 | D7 (训练越久越糟) | 立即部署早期 ckpt (Phase 2.C7)；长期走 C2 (短训低 LR) |
| 全程都比官方差，从 step 5k 开始就差 | D1/D5 (微调机制本身在破坏能力) | 走 C1 (freeze vision tower) 或 C8 (LoRA) |
| init (mixed_1) 已经比官方原版差 | init 链条早期就出问题 | 回到更原始 base 重做 init |
| `mixed_gf0_173` 比 `mix_vis600` 显著好 | kai0 比例 / 短训 是关键 | 走 C2 + 加大 kai0 比例重训 |
| 各 ckpt 差不多，都比官方差 | D2/D3 (自采数据本身的标注问题) | 走 D2 数据审计 → C4 (corner 专项高质量数据) |

#### 0.B 失败 episode 录制（用于解释"反复同位"现象）

录制 5 个失败 episode，同步保存：
- [ ] 三路 cam 视频（hand_left / hand_right / base）
- [ ] 每帧 state (14-dim)
- [ ] 每帧 policy 输出的完整 50-step action chunk + 实际下发 action
- [ ] 时间戳同步
- 存放路径: `_______`

逐帧分析死循环期间：
- [ ] obs (state) 方差 = `_______`
- [ ] action chunk 方差 = `_______`
- [ ] 失败模式分类: ☐ 横向偏 ☐ 高度不够 ☐ 朝向偏 ☐ gripper 闭合时机错

---

### Phase 1 — 部署侧排查（代码层面已完成 2026-04-29）

#### ✅ 已确认无问题（一行带过）

- **A1 norm_stats**：md5 与训练时一致（`38bff549/d8b80670/01842ddc`）
- **A2 归一化方向**：openpi `Normalize`/`Unnormalize` 自动对称
- **A4 pure_vis600 mirror**：完全自洽训练增广，部署无需任何逆变换
- **A7 chunk 消费**：3Hz 推理 / 30Hz 发布 / drop 前 8 步 / 线性 blend，**不是死循环主因**
- **A9 sim01**：仅在 sim01 上能验证

#### 🚨 真正发现的问题（按风险排序）

| # | 问题 | 文件:行 | 风险 | 应对 |
|---|---|---|---|---|
| **P1** | 当前部署 launcher (`temp.sh` 选 L) 显式 `enable_rtc:=false`，但同文件 211 行注释明示 RTC 对"抓取瞬间偏"预期 +15-30% | `start_autonomy_temp.sh:152` | 🔴 **高** | **立刻** `./rtc_apply.sh on` 或改 launcher `:=true` |
| **P3** | pi05 flow-matching 对 noise 不敏感 → 实际**接近 deterministic** → obs 不变则 chunk 几乎相同 → 反复同位 | `pi0.py:299` `policy.py:65,75` | 🟡 中 | RTC + 缩短 chunk 执行步数 + 高 inference_rate 缓解 |
| **P4** | 三套 vis600 配置全参解冻 + LR 1.5e-5 + 40k step（命中 D5 假说，对比 awbc_from_official_mixed 是 1e-5 / 20k） | `config.py:2570-2652` | 🟡 中高 | 重训才能解决 → 见 C1/C2 |
| **D6** | 自采 vs init 数据 norm_stats: **mean 偏移 12-18°，std 放大 2-2.5×**，**vis_base_40k 偏移最大 → 真机表现也最差**（与回归假说自洽） | 见下表 | 🔴 强证据 | 重训需要考虑混入更接近 init 分布的数据 |
| P2 | `state = np.where(state > π, 0, state)` 静默清零越界关节 | `agilex_policy.py:93` | 🟢 低 | 加一行 warn log |
| P5 | `gripper_offset=0.003` rad ≈ 0.17° | `policy_inference_node.py:301` | 🟢 低 | 验证与训练标定一致 |

#### D6 norm_stats 偏移量化

| 维度 | mixed_1 (init) | mix_vis600 | pure_vis600 | vis_base_40k | 最大偏移 |
|---|---:|---:|---:|---:|---:|
| state.mean[3] L_j3 | -0.040 | -0.138 | 0.074 | -0.176 | 14° |
| state.mean[5] L_j5 | -0.038 | 0.104 | -0.055 | 0.175 | 12° |
| state.mean[10] R_j3 | 0.024 | 0.192 | 0.086 | 0.337 | **18°** |
| state.mean[12] R_j5 | 0.015 | -0.150 | -0.071 | -0.300 | **18°** |
| state.std[3-5,10-12] | ~0.21 | ~0.40 | ~0.51 | ~0.51 | std ×2~2.5 |

#### 立即可执行的修复（不重训）

1. **打开 RTC**（5 分钟）：`./start_scripts/rtc_apply.sh on` → 不行试 `rtc_tight`
2. **换 ckpt**：把 launcher 默认 ckpt 加上 `mixed_gf0_173_v1 step 9000`（MAE@1=0.0129 全系列最佳，1:1:1 等量 mix + 短训，可能就是当前最优部署候选）
3. **加 state 越界 warn**（P2，一行）：`if np.any(np.abs(qpos) > np.pi): self.get_logger().warn(...)`

---

### Phase 2 — 干预方案（按 Phase 0.A 结论选择，不要全做）

#### C7 ⭐ 早期 ckpt 部署（0 成本，今天就能做）

- [ ] 直接部署 `mix_vis600_v1` step 5000 / 10000 / 20000 真机测试（已含在 Phase 0.A）
- 选定 step: `_______`
- 决策依据: `_______`

#### C2 短训 + 降学习率重训

- 配方：复用 mix_vis600 数据成分，改 step 8k–15k + peak_lr 5e-6（远比当前 1.5e-5 / 40k 保守）
- 触发条件：Phase 0.A 显示训练越久越糟
- 计划开始日期: `_______`
- 真机测试: corner `___%`, flow `___%`

#### C1 vision tower freeze 微调

- 配方：复用 mix_vis600 数据，加 freeze_filter (PaliGemma 全 frozen)，仅训 action expert
- 触发条件：Phase 0.A 显示从早期 step 就退化
- 计划开始日期: `_______`
- 真机测试: corner `___%`, flow `___%`

#### C4 corner 专项高质量数据 100–200 ep

- 严格要求 teleop 精准对准衣角，提升 label 质量
- 触发条件：Phase D 数据审计显示 label 系统性偏差
- 计划开始日期: `_______`

#### C5 加大 kai0 比例重训

- 配方：kai0 : 自采 = 2:1 或 3:1（mix_vis600 是 1:2 反向）
- 触发条件：Phase 0.A 显示 mixed_gf0_173 (1:1:1) 显著优于 mix_vis600 (2:1:1 偏 vis)
- 计划开始日期: `_______`

#### C6 ⭐ 部署侧打开 RTC + 调高 inference_rate（**已知最高 ROI 的零成本干预**）

> 由 Phase 1 P1+P3 直接得出。当前 launcher 选 L 显式 `enable_rtc:=false`，自废 RTC 武功。temp.sh 第 211 行注释明确写 RTC tight 对"抓取瞬间偏"预期改善 15-30%。

- [ ] 试 `./start_scripts/rtc_apply.sh on` (默认 RTC + 3Hz infer + smooth=8)
  - 真机测试: corner `___%`, flow `___%`
- [ ] 试 `./start_scripts/rtc_apply.sh rtc_tight` (10Hz, exec_h=12, max_guid=0.8, smooth=3)
  - 真机测试: corner `___%`, flow `___%`
- [ ] 试 `./start_scripts/rtc_apply.sh rtc5` (6Hz, exec_h=16, max_guid=0.5)
  - 真机测试: corner `___%`, flow `___%`

#### C8 LoRA / adapter 微调（保底方案）

- 物理上限制权重漂移幅度
- 触发条件：C1 / C2 / C5 都不能恢复 corner-grasp
- 计划开始日期: `_______`

---

### Phase D — 数据审计（与 Phase 0.A 并行）

#### D2 自采 corner-grasp label 精度审计

- [ ] 抽 30 个自采 ep，逐帧人工标注"夹取触发瞬间" ee 距衣角真实距离
- [ ] 统计：mean / std = `___ / ___ cm`，偏离 > 1cm 占比 = `___%`
- 结论: `_______`

#### D3 corner-grasp 帧占比统计

- [ ] 统计每个 ep 中 corner-grasp 阶段帧数 / 总帧数
- [ ] vis_base 平均: `___%`；kai0_base 平均: `___%`
- 结论 (是否被叠衣后段大量帧稀释): `_______`

---

## 2. 决策树（已用代码层面发现更新）

```
[今天必做] C6 打开 RTC （5 分钟，零成本）━━━━━━━━━━━━━━━━━━━━━━
  │     代码层证据：当前部署 enable_rtc=false (P1) + 输出近 deterministic (P3)
  │     → temp.sh:211 注释明示对"抓取瞬间偏"预期 +15-30%
  │
  ├── RTC 打开后 corner-grasp 显著改善 → 直接上线，结案
  │
  └── RTC 打开后仍有问题
        ↓
        Phase 0.A 真机对照曲线 (含 mixed_gf0_173 step 9000 对照)
            │
            ├── 早期 step ≈ 官方，后期 step 才退化
            │     → D7 成立 → C7 (部署早期 ckpt) + C2 (短训重训)
            │
            ├── 全程都比官方差，从 step 5k 就差
            │     → D1/D5 成立 (代码已确认 P4 全参解冻+高 LR+长训)
            │     → C1 (freeze vision) → 失败兜底 C8 (LoRA)
            │
            ├── mixed_gf0_173 (1:1:1 + 短训) 显著好于 vis600 系
            │     → kai0 比例 + 训长是关键 → C5 (加大 kai0) + C2 (短训)
            │
            ├── init (mixed_1) 已经比官方差
            │     → init 链条问题 → 回到更原始 base 重做 init
            │
            └── 各 ckpt 差不多都比官方差
                  → D2/D3 数据审计 → 若 label 差 → C4 (高质量 corner 专项)

并行：
  Phase 0.B 失败 episode 录制 — 解释"反复同位"现象
  Phase D 数据审计 — 决定是否需要重采数据
  P2 加 state>π 越界监控（一行日志）
```

---

## 3. 排查记录

| 日期 | 阶段 | 操作 | 结果 | 下一步 |
|---|---|---|---|---|
| 2026-04-29 | — | 创建本日志 | — | 开始 Phase 0.A |
| 2026-04-29 | — | 用户澄清：mix_vis600 已是混合训练；同套硬件 → 排除大量假设，砍掉 Group A 硬件 mismatch / Group C 中 C0 mix 官方数据方案 | — | 立即执行 Phase 0.A 真机对照 |
| 2026-04-29 | Phase 1 (代码) | A1/A2/A4/A7 全部读代码完成 | A1/A2/A4 ✅ OK；A7 chunk 策略合理。**5 个新疑点 P1-P5**：当前 RTC=off (P1) / state>π 静默清零 (P2) / sample 接近 deterministic (P3) / 全参解冻+高 LR+长训命中 D5 (P4) / gripper_offset=0.003 (P5) | 立即试 C6 (打开 RTC) — 5 分钟零成本 |
| 2026-04-29 | Phase D (代码) | D6 norm_stats 量化对比完成 | mix/pure/vis_base 三套数据集的 norm_stats mean 比 init 偏移 12-18°，std 放大 2-2.5×。**vis_base_40k 偏移最大，恰好真机表现也最差** — 与"capability regression 由分布漂移驱动"假说自洽 | 真机执行 Phase 0.A 对照 + C6 |
| | | | | |

---

## 4. 相关资源

- 训练实验全记录: `docs/training/task_a_visrobot01_mixed_600.md`
- 部署 ckpt tar 包:
  - `mix_vis600 step 38000`: `/vePFS/tim/workspace/deepdive_kai0_tmp/data/mix_vis600_best_step38000.tar`
  - `pure_vis600 step 39999`: `/vePFS/tim/workspace/deepdive_kai0_tmp/data/pure_vis600_best_step39999.tar`
  - `vis_base_40k step 36000`: `/vePFS/tim/workspace/deepdive_kai0_tmp/data/vis_base_40k_best_step36000.tar`
- 中间 step ckpts (供 Phase 0.A 早期 ckpt 测试): `/vePFS/.../checkpoints/pi05_flatten_fold_mix_vis600/mix_vis600_v1/{2000,4000,...}/`
- 数据 build 脚本: `train_scripts/data/build_task_a_{vis_base,mix_vis600,pure_vis600}.py`
- sim01 部署文档: `docs/deployment/sim01_deployment.md`
