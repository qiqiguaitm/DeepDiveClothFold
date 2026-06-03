# Dagger 数据有效性验证 + 训练方式对比 (smooth800 + dagger)

> **目的**: 验证自采 dagger 数据 (`vis_dagger/v2`, 210 ep) 的有效性, 并对比两种把 dagger 引入的训练方式。
> **状态**: 📋 规划 (2026-06-03)
> **关联**:
> - smooth800 基线: [`../../history/experiments/task_a_new_smooth_800_new_norm_results.md`](../../history/experiments/task_a_new_smooth_800_new_norm_results.md) (811 ep, MAE@1=0.0089, best=step40k)
> - dagger 同步: [`../../../deployment/training_ops/data_sync_tos.md`](../../../deployment/training_ops/data_sync_tos.md) (sync_vis_dagger, vis_dagger/v2)
> - 表示一致性已验证 (见 §1)

---

## 0. 两个核心问题

| 问题 | 实验 | 方法 |
|---|---|---|
| **Q1 数据有效性**: dagger 数据加进去能否提升模型? | **Exp-A** (从头重训 smooth800+dagger) | 与纯 smooth800 baseline 对比 |
| **Q2 训练方式对比**: "从头重训" vs "best ckpt 上短步微调" 哪个引入 dagger 更高效? | **Exp-B** (best ckpt + 1:1 短步微调) | 与 Exp-A + baseline 三方对比 |

> ⚠️ **真机为终判** (沿用 data_root_cause_probe 铁律): offline MAE 只用于 ① 训练健康 ② 选 ckpt ③ 同验证集相对差。dagger 是真机纠错示范, 其价值 (减少走停/松手) 在 offline 逐帧 MAE 上可能不显著 → 最终需真机对比。

---

## 1. 前置: dagger vs smooth800 表示一致性 — ✅ 已验证 (2026-06-03)

直接读 parquet 对比, 结论: **表示方式完全一致, 可直接混入, 无需转换。**

| 维度 | dagger (vis_dagger/v2) | smooth800 (A_new_smooth_800/base) | 一致 |
|---|---|---|---|
| state/action | 14D absolute joint | 14D absolute joint | ✅ |
| **action == state** | ✅ 全 4 日期全 ep (max\|Δ\|=0.000000) | ✅ (0.000000) | ✅ |
| action 语义 | absolute (action[t]=同步关节读数; action 序列=真实未来轨迹) | 同 | ✅ |
| fps | 30 | 30 | ✅ |
| gripper 量纲 | 米级 0~0.058 | 0~0.057 | ✅ |
| 关节范围 | ±2.4 rad | ±2.5 rad | ✅ |

**⚠️ 2 个差异 (不影响表示, 但 build 时要处理)**:
1. **dagger 多 `intervention` 列** (int8, 全=1 → 全程人工干预的纠错轨迹)。smooth800 无此列 → **合并集 build 时删掉 intervention 列对齐 schema**。
2. **state 分布有偏移** (新采集, 如 R_j1 dagger −0.067 vs smooth +0.067, 在 std 内)。这是 dagger 要补的多样性 → **混合集必须重算 norm_stats, 不复用 smooth800 的**。

---

## 2. 数据集构建

### 2.1 dagger 规模 (vis_dagger/v2)

| date | ep |
|---|---:|
| 2026-05-29 | 64 |
| 2026-06-01 | 32 |
| 2026-06-02 | 71 |
| 2026-06-03 | 43 |
| **总计** | **210** |

> dagger 用 **v2 (不裁投放)**, 与 smooth800 (未裁) 保持一致 → 隔离"加 dagger"单变量, 不引入"裁/不裁"额外变量。

### 2.2 三个数据集 (build_*.py 待写, 参考 build_vis_v2_full.py 惯例)

| 数据集 | 组成 | ep | 用途 | 命名 |
|---|---|---:|---|---|
| **D0** baseline | smooth800 全量 | 811 | Exp-A 对照 (= 已有 smooth800, 无需重 build) | `A_new_smooth_800` (现成) |
| **D1** full-mix | smooth800 全量 + dagger 全量 | 811+210=1021 | Exp-A 重训 | `A_smooth800_dagger_full` |
| **D2** 1:1-mix | smooth800 抽 210 + dagger 全量 210 | 420 | Exp-B 微调 | `A_smooth800_dagger_1to1` |

**build 关键** (三集通用):
- **删 intervention 列** (dagger 侧), 对齐 smooth800 schema。
- **episode_index 重排** 0..N-1 (合并多源)。
- **保留 14D joint** (不转 EE6D, 不裁投放)。
- **视频**: 复用源 mp4 (符号链接, 同 build_task_a_new_100.py 的 --symlink-video; 跨数据集源不同需 copy 或 deref)。
- **norm_stats 各自重算** (openpi `compute_norm_stats.py`, 经真实 dataloader, padding 32D + 分位数)。
- **D2 抽样**: smooth800 811 ep 固定 seed (如 seed=42) 随机抽 210 → 可复现。

---

## 3. Exp-A — 从头重训 smooth800+dagger (验 Q1 数据有效性)

### 3.1 配置 (与 smooth800 完全一致, 仅数据变)

| 项 | 值 (= smooth800 baseline) |
|---|---|
| Config | `pi05_flatten_fold_A_smooth800_dagger_full` |
| 框架 | JAX/Flax NNX (`scripts/train.py`) |
| **Init** | `mixed_1_clean` (`/home/tim/local_ckpts/Task_A_init/mixed_1_clean/params`) — 与 smooth800 同 init |
| 数据 | **D1** (`A_smooth800_dagger_full`, 1021 ep) |
| use_delta_joint_actions | False (absolute) |
| LR | Cosine, warmup=1k, peak=1.5e-5, decay=50k, decay_lr=1.5e-6 |
| EMA | 0.9999 |
| Steps / Batch | 50,000 / 128, fsdp_devices=8 |
| num_workers | 16×节点数 (uc 单机 16; 见 [[feedback-uc-cluster-num-workers]]) |
| inline_eval | smooth800 val (26 ep) — **与 baseline 同 val, 可直接比** |
| 集群 | 单机 8 GPU (uc/cnsh/cnbj 视空闲) |

> **单变量铁律**: D1 vs smooth800 baseline 唯一差别 = 多了 210 ep dagger。init/lr/step/batch/val 全同。

### 3.2 判定 (Exp-A vs smooth800 baseline)

| offline MAE (smooth800 val, 同协议) | 结论 |
|---|---|
| Exp-A @1/@50 ≤ baseline (0.0089/0.0636) | ✅ dagger 有正贡献 (至少不伤) → 真机确认 |
| Exp-A ≈ baseline | dagger 对 in-distribution 无显著影响 → 看真机 (dagger 价值在 OOD 纠错) |
| Exp-A > baseline (变差) | ⚠️ dagger 引入分布偏移伤 in-dist → 查 dagger 质量 / 配比 |

> ⚠️ dagger 是纠错数据, 其价值主要在**真机减少走停/松手**, offline MAE 可能看不出 → **真机对比才是 Q1 终判**。

---

## 4. Exp-B — best ckpt + 1:1 短步微调 (验 Q2 训练方式)

### 4.1 配置 (从 smooth800 best ckpt 继续, 1:1 数据, 短步)

| 项 | 值 |
|---|---|
| Config | `pi05_flatten_fold_A_smooth800_dagger_1to1_ft` |
| **Init (resume)** | **smooth800 best ckpt step40000** (`uc03:.../pi05_flatten_fold_a_new_smooth_800_new_norm/.../40000/`; 40k≈49999 已 plateau, 见 results §5) |
| 数据 | **D2** (`A_smooth800_dagger_1to1`, smooth 抽210 + dagger 210 = 420 ep) |
| **Steps** | **20,000** (短步微调) |
| LR | Cosine, warmup=1k, peak=1.5e-5→1.5e-6 (同 smooth800; 短退火) — 或考虑更低 peak (微调常用 1/2~1/3, 见下风险) |
| EMA / batch / fsdp | 0.9999 / 128 / 8 (同) |
| inline_eval | smooth800 val (26 ep) — 同 baseline/Exp-A 可比 |

> **"1:1" = ep 数等量** (D2: smooth 210 + dagger 210)。让 dagger 在微调中获得与 smooth 同等曝光, 快速注入纠错行为。
> **从 best ckpt 继续** = 验证"已收敛模型 + 少量 dagger 短步" 能否高效引入 dagger 价值 (省算力 vs Exp-A 从头 50k)。

### 4.2 ⚠️ 微调风险

1. **LR 选择**: 从已收敛 ckpt 继续, peak 1.5e-5 可能偏高 → 灾难性遗忘 smooth 知识。**建议先 dry 一个低 peak (如 5e-6) 变体**, 或监控 smooth val 是否回退。
2. **resume 机制**: openpi resume 需 ckpt 含 train_state (optimizer/EMA), 还是只 load params 重置 optimizer? **确认 smooth800 ckpt 是否保留 train_state** (results §5 提到完整 ckpt 含 train_state)。若只 load params, optimizer 从零, warmup 重要。
3. **1:1 抽样 seed 固定** → D2 可复现。

### 4.3 判定 (Exp-B vs Exp-A vs baseline)

| 对比 | 结论 |
|---|---|
| Exp-B (20k 微调) 真机 ≈ Exp-A (50k 重训) | ✅ **短步微调更高效** (省 60% 算力达同效) → 训练方式推荐微调 |
| Exp-B < Exp-A | 从头重训更充分 → dagger 需深度融合, 微调不够 |
| Exp-B 出现 smooth 能力回退 (val MAE 升) | ⚠️ 灾难性遗忘 → 调低 LR / 加 smooth 配比 / 缩短步数 |

---

## 5. 实施步骤

| Step | 内容 | ETA | 依赖 |
|---|---|---:|---|
| T1 | 写 `build_smooth800_dagger.py` (删 intervention 列 + 合并 + 重排 + 视频处理) | 2h | — |
| T2 | build **D1** (1021 ep) + **D2** (420 ep, seed42 抽样) | 1h | T1 |
| T3 | 各自 `compute_norm_stats.py` 重算 norm_stats | 0.5h | T2 |
| T4 | 注册 2 个 config (`A_smooth800_dagger_full` / `_1to1_ft`) | 0.5h | T3 |
| T5 | **Exp-A** 从头重训 50k (单机 8 GPU) | ~5h | T4 + init 就位 |
| T6 | **Exp-B** best ckpt resume 微调 20k | ~2h | T4 + smooth800 best ckpt 就位 |
| T7 | offline eval: D0/Exp-A/Exp-B 同 smooth800 val 对比 | 1h | T5 T6 |
| T8 | ⭐ **真机测试** (baseline vs Exp-A vs Exp-B 三方) = Q1+Q2 终判 | 1 day | T7 |

### 5.1 ⚠️ 起训前依赖 (本机缺失, 需确认)

| 依赖 | 状态 |
|---|---|
| `mixed_1_clean` init (Exp-A) | gf0 本机**无**, 在 uc03/cnbj — 起训机需确认就位 |
| smooth800 best ckpt step40000 (Exp-B) | 在 **uc03** 本地, 跨机需 scp/TOS 中转 |
| 训练机选择 | uc (NFS 共享, init/ckpt 都在) 最方便; 否则 TOS 中转 init+ckpt |

---

## 6. 预期产出 + 结论矩阵

| | offline MAE (smooth val) | 真机 (走停/松手/完成) | 结论 |
|---|---|---|---|
| **D0** baseline (smooth800) | 0.0089 / 0.0636 (已知) | 已验证 work (基准) | — |
| **Exp-A** 重训+dagger | 待测 | 待测 | **Q1**: dagger 有效性 |
| **Exp-B** 微调 1:1 | 待测 | 待测 | **Q2**: 微调 vs 重训 |

**核心交付**:
- **Q1**: dagger 数据是否提升真机表现 (Exp-A/B 任一 > baseline 即证有效)。
- **Q2**: "从头重训 50k" vs "best ckpt 微调 20k" 哪个引入 dagger 更高效 (真机同效则微调胜, 省算力)。

> 后续 (视结果): 若 dagger 有效, 可推广到更多 dagger 日期 / 调配比 / 试 dagger 裁投放 (v3) 版本。

---

## 附录 — 关键路径

| 项 | 路径 |
|---|---|
| dagger 源 | `kai0/data/Task_A/vis_dagger/v2/<date>-v2` (210 ep) |
| smooth800 源 | `kai0/data/Task_A/self_built/A_new_smooth_800/{base,val}` (811 ep + 26 val) |
| smooth800 best ckpt | `uc03:/data/shared/.../pi05_flatten_fold_a_new_smooth_800_new_norm/.../40000/` (= 49999) |
| init | `mixed_1_clean` (uc03/cnbj) |
| build 脚本 (待写) | `train_scripts/kai/data/build_smooth800_dagger.py` |
| 输出数据集 | `kai0/data/Task_A/self_built/A_smooth800_dagger_{full,1to1}/` |
