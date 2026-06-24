# Dagger 数据有效性验证 + 训练方式对比 (smooth800 + dagger)

> **目的**: 验证自采 dagger 数据 (`vis_dagger/v2`, 210 ep) 的有效性, 并对比两种把 dagger 引入的训练方式。
> **状态**: 🔄 进行中 (2026-06-05) — Exp-B (1:1 微调) ✅ 训练完成 (inline @1=0.0085); Exp-A (从头重训) 🟡 cnbj 排队。**执行状态 + 实测见 §7**。
> **🆕 Exp-C (2026-06-12)**: v3 早期干净 base(≤5-10,985ep)+ dagger v3 全量(513ep)混合、**单 norm**、重训 pi05 → 📝 规划草稿见 **§8**(待确认)。
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

## 7. 执行状态 + 实测结果 (2026-06-05) — 🔄 进行中

> ⚠️ **计划 vs 实际的偏差**(以实际为准): dagger v2 四日期实际 **227 ep**(非计划的 210);build 时跳过 5 个 broken-video ep。Exp-B init 实际用 **smooth800 `step49999`**(用户提供的 `kai0/checkpoints/task_a_new_smooth_800_step49999`,非 step40000;两者已 plateau 等价)。

| 数据集 | 计划 | **实际(build 完)** | 备注 |
|---|---|---|---|
| **D1** full-mix `A_smooth800_dagger_full` | 1021 | **1033 ep** | smooth 811 + dagger 227 − 5 broken-skip;`chunks_size=1033` |
| **D2** 1:1-mix `A_smooth800_dagger_1to1` | 420 | **453 ep** | smooth 抽227(seed42) + dagger 227 − 1 skip |

### 7.1 Exp-B (D2 1:1 微调) — ✅ 训练完成 (cnsh)

- **状态**: ✅ 20k step 训完(cnsh 单节点 8 A100, 1.9s/it ≈ 9h);config `pi05_flatten_fold_A_smooth800_dagger_1to1_ft`, init `step49999`, `--overwrite`。
- **inline-eval**(smooth val 26 ep, 200-frame 子集, 每 8k):

  | step | MAE@1 | MAE@10 | MAE@25 | MAE@50 |
  |---|---|---|---|---|
  | 8000 | 0.0088 | 0.0168 | 0.0262 | 0.0378 |
  | **16000** | **0.0085** | **0.0164** | **0.0255** | **0.0368** |

  → 16k 仍在降、未 plateau,训到 20k 持续改善 → **最佳 ckpt 大概率 step 19999**(未落在 inline 点,确切须 offline eval `10000` vs `19999`)。
- **保存 ckpt**: `10000`, `19999` → `kai0/checkpoints/pi05_flatten_fold_A_smooth800_dagger_1to1_ft/smooth800_dagger_1to1_ft_cnsh/`
- ⏳ **待办**: offline 全 val eval(pi05 协议,确认最佳 ckpt) + 真机。

### 7.2 Exp-A (D1 full-mix 从头重训) — 🟡 cnbj 排队中

- **状态**: 🟡 Queueing(cnbj `Robot-North-H20` 2 节点 gang 配额长期不足)。曾尝试迁 cnsh(数据+init 已就位)但用户决定**留 cnbj 排队**(`t-20260605081112-tth46`)。
- config `pi05_flatten_fold_A_smooth800_dagger_full`, init `mixed_1_clean`, 50k, 16 H20。
- ⏳ 训练未开始 → 无结果。

### 7.3 踩坑记录(已修,详见 [`training_pitfalls_common.md`](../../../deployment/training_ops/submission/training_pitfalls_common.md))

- **info.json `total_episodes` 用 pre-skip 数** → 幽灵尾索引 → lerobot 文件 assert → offline HF 崩(§3)。已修 build 脚本 + 3 份 info.json。
- **`chunks_size=1000` < N(1033)** → ep≥1000 找 chunk-001 → 同样 assert 崩(§3)。已修(`chunks_size=max(1000,N)`)。
- **多机 ckpt-init `sync_global_devices` mismatch** = 上次失败留的残桩 ckpt 目录 → 清目录后重提即过。

---

## 8. ⭐ Exp-C (新, 2026-06-12) — v3 早期干净 base + dagger v3 全量混合重训

> **一句话**: 把 **vis_base/v3 早期段(< 2026-05-18,front-trim 已裁)** 与 **vis_dagger/v3 全量** 混成单一数据集、**一起算一个 norm**、从 `mixed_1_clean` init 重训 pi05,验证"早期干净 base + 全量 dagger 纠错"是否比 smooth800 锚 / 纯 base 真机更好。
> **状态**: ✅ **已实施完成**(2026-06-14 提交 → 06-16 跑完 50k,cnsh robot-task 8 A100)。结果见 **§8.8**。
> ⚠️ **ckpt 已误删不可恢复**(2026-06-17 清理时一条换行折断的 `rm -rf {brace}` 误删整个 `v3early_dagger_cnsh/` 含 best 49999;vePFS `.snapshots` 为空无快照)→ **Tier-3 真机需重训**(data `A_v3early_dagger` + config 均在,重跑即可复现)。MAE 历史已从训练日志留存(§8.8)。
> **为什么这么配**(综合两文档最佳结论):
> - idle 系列 → ① **v3 前端裁有效**(Step 1 ✅);② **5-18~5-27 是真机嫌疑/fail 窗**(Exp-D 假说)→ 取 **≤5-10 早期 work 段**(排嫌疑窗)。见 [`idle_data_trimming_experiments.md`](idle_data_trimming_experiments.md) §2/§3.6/§5.5。
> - 本文档 → **dagger 是有效纠错数据**(Q1)。这次用 **v3 前裁版 dagger 全量**(8 日期 513ep,比 v2 的 227ep 更多)。

### 8.1 数据范围(已实测,2026-06-12)

> ⚠️ **2026-06-16**: 下表帧数为 **pre-tailcap**(`A_v3early_dagger` 已在尾裁前 build,故这些数对该 run 成立)。**v3 现已就地尾裁**(Step 3,删 ~1.6%)→ **重新 build** 该集会用尾裁后 v3,帧数略降,届时以 build 时实测为准。trim 标签 "v3 前裁" → **"v3 前裁+尾裁"**。

| 域 | 来源 | 日期 | ep | 帧(pre-tailcap)|
|---|---|---|---:|---:|
| **base**(v3 前裁+尾裁,< 5-18)| `vis_base/v3` | 4-23 / 4-24 / 4-25 / 4-28 / 4-29 / 4-30 / 5-06 / 5-07 / 5-08 / 5-09 / 5-10(11 日期,**排 5-16 残缺**)| **985** | **1,194,427** |
| **dagger**(v3 前裁+尾裁,全量)| `vis_dagger/v3` | 5-29 / 6-01 / 6-03 / 6-04 / 6-05 / 6-08 / 6-09 / 6-10(8 日期)| **513** | **770,545** |
| **merged** | `self_built/A_v3early_dagger`(已 build,pre-tailcap)| | **≈1498** | **≈1,964,972** |

> ⚠️⚠️ **关键澄清(请确认)**: 用户说"vis_dagger/v3 **和** vis_base/v3 的 **< 5-18 日期**"。但 **dagger v3 全部日期 ≥ 2026-05-29,没有任何 < 5-18 的日期**(dagger 都是后期采的纠错轨迹)。→ 故"< 5-18"**只对 base 生效**;dagger 取**全量 8 日期**。本 plan 按此解读;若你只想要某些 dagger 日期,见 §8.6-Q1。
> - 自然比 **base:dagger ≈ 1.55:1**(帧)。**不做过采样/加权**(用户只要"混合+一起算 norm",非 1:1 平衡;如需平衡见 §8.6-Q3)。

### 8.2 schema 对齐(已核实,可直接混)

| 维度 | vis_base/v3 | vis_dagger/v3 | 处理 |
|---|---|---|---|
| state/action | 14D absolute joint | 14D absolute joint | ✅ 同 |
| **action == state** | ✅ | ✅(max\|a−s\|=0.0)| ✅ 同 |
| `intervention` 列 | ❌ 无 | ✅ 有(int8)| ⚠️ **merge 时删**,对齐 schema(同 §1) |
| 前端裁 | v3(已裁)| v3(已裁,同 pipeline)| ✅ 单变量干净:两侧都 v3 |
| fps / 视频 | 30 / 3 路 mp4 | 30 / 3 路 mp4 | symlink 复用 |

### 8.3 build(克隆 `build_smooth800_dagger.py`)

`build_smooth800_dagger.py` 已实现"base 源 + dagger 源 → 单 lerobot:删 intervention + episode_index 重排 + 视频 symlink + 单 norm"。**克隆为 `build_v3early_dagger.py`,把源换掉**:
- base 源 → `vis_base/v3` 的 **11 个 < 5-18 日期**(非 smooth800)。
- dagger 源 → `vis_dagger/v3` 的 **全 8 日期**(非 vis_dagger/v2)。
- 输出 → `kai0/data/Task_A/self_built/A_v3early_dagger`(`chunks_size=max(1000,N)`;`total_episodes`=post-skip,避免幽灵尾索引,见 §7.3 踩坑)。
- ⭐ **norm**: 对 merged 集**整体算一个 norm**(`compute_norm_states_fast.py`,经真实 dataloader,padding 32D + 分位数)——用户明确"一起算一个 norm",**不分域、不复用任何旧 norm**。

### 8.4 训练 config(克隆 flatten-fold pi05)

新建 `pi05_flatten_fold_v3early_dagger`(克隆 `pi05_flatten_fold_A_smooth800_dagger_full` §3.1):
- `repo_id` → `A_v3early_dagger`;`use_delta_joint_actions=False`(absolute);`default_prompt="Flatten and fold the cloth."`(横向折,与 base/dagger 同 SOP)。
- init `CheckpointWeightLoader("mixed_1_clean/params")`;LR cosine warmup1k / peak1.5e-5 / decay50k→1.5e-6;EMA0.9999;**50k step**;bs128;fsdp8;norm 重算;inline-eval `vis_v2_merged_val`(与 idle/v3 系列同 val,可横比)。
- **集群**: cnbj(Robot-North-H20)或 cnsh 视空闲,8/16 卡(global batch 128 不变)。

### 8.5 评估(真机为终判)

| Tier | 做法 |
|---|---|
| Tier 1 offline | `vis_v2_merged_val` inline-eval 逐 ckpt MAE → 收敛 + 选 best。⚠️ idle/慢轨迹 MAE 反指(§铁律),只 sanity。 |
| Tier 3 真机 | best ckpt 真机叠衣:成功率 / 走停 / 松手脱落 / 抓取到位 = 终判。 |
| 对照 | ① **smooth800 锚**(真机 work 基准,MAE@1=0.0089);② Exp-A `A_smooth800_dagger_full`(smooth+dagger-v2,§7.2,若训完);③(可选)**纯 v3 base ≤5-10**(无 dagger)control 隔离 dagger 增量。 |

**判据**: merged 真机 **> smooth800 锚 + ≥ Exp-A** → "早期 v3 base + 全量 v3 dagger"是更好配方;若 ≈ → dagger v3 增量被 base 吃掉;若 < → 查 dagger v3 质量 / base 早期段覆盖。

### 8.6 待确认(动手前)
1. **dagger 范围**: 用 vis_dagger/v3 **全 8 日期 513ep**(默认,因无 <5-18 dagger)?还是只取某些日期?
2. **base 5-16**: 确认**排除**(16ep 残缺,无争议)?
3. **配比**: 自然混(base:dagger≈1.55:1,默认)还是要 1:1 平衡(需 per-domain 加权,改用 `KaiVisMergedDataConfig`,见 [`pi05_task_a_av1_mixed_1to1_plan.md`](pi05_task_a_av1_mixed_1to1_plan.md))?**用户只说"一起算一个 norm"→ 默认自然混 + 单 norm。**
4. **集群 / 步数**: cnbj or cnsh?50k 确认?
5. **init**: `mixed_1_clean`(沿用)确认?

### 8.7 实施步骤(待"开始实施")
1. 克隆 `build_v3early_dagger.py`(base=v3 11日期 + dagger=v3 8日期,删 intervention)→ build `A_v3early_dagger`(≈1498ep)。
2. `compute_norm_states_fast.py` 对 merged **整体算单 norm**。
3. 注册 config `pi05_flatten_fold_v3early_dagger`,commit/push。
4. 提交训练(cnbj/cnsh,50k,inline-eval)。
5. offline 选 ckpt → **真机对比**(vs smooth800 锚 / Exp-A / 纯 base)。
6. 回填结论 + results.md + master history。

### 8.8 ⭐ 实施结果(2026-06-16 跑完 50k)

**实测配置**(均已核对):
- 数据 `A_v3early_dagger`:**1498 ep / 1,964,972 帧**(base 985ep/1,194,427f + dagger 513ep/770,545f)。
- **采样比 = 帧比 base:dagger = 1.55:1**(单一合并数据集、均匀采样、单 norm,**无加权 sampler**;按 ep 是 1.92:1 但训练按帧采)。
- config `pi05_flatten_fold_v3early_dagger`,`default_prompt="Flatten and fold the cloth."`,init `mixed_1_clean`,50k / bs128 / fsdp8。
- job `t-20260615074032-qg8m8`(cnsh robot-task),inline-eval on `vis_v2_merged_val`。

**inline-eval MAE 历史**(val = `vis_v2_merged_val`,单调收敛、无过拟合回升):

| step | MAE@1 | MAE@10 | MAE@25 | MAE@50 |
|---|---|---|---|---|
| 8000 | 0.0110 | 0.0271 | 0.0504 | 0.0815 |
| 16000 | 0.0095 | 0.0230 | 0.0398 | 0.0586 |
| 24000 | 0.0090 | 0.0211 | 0.0348 | 0.0490 |
| 32000 | 0.0086 | 0.0197 | 0.0316 | 0.0439 |
| 40000 | 0.0084 | 0.0187 | 0.0296 | 0.0408 |
| 48000 | 0.0083 | 0.0181 | 0.0284 | 0.0391 |
| **49999 (best)** | **0.0083** | **0.0180** | **0.0281** | **0.0388** |

> 对照 smooth800 锚 MAE@1=0.0089(§8.5)→ Exp-C 49999 @1=0.0083 略优;长 horizon @50=0.0388 收敛干净。**但 offline MAE 非终判**(idle/慢轨迹反指,§铁律)→ **dagger 价值仍须真机对比**(走停/松手/完成),而真机所需 best ckpt 已误删 → **需重训后才能 Tier-3**。

- best ckpt 原路径(**已删除**):`kai0/checkpoints/pi05_flatten_fold_v3early_dagger/v3early_dagger_cnsh/49999/params`

### 8.9 ⭐ 重训复现(2026-06-17,重构/裁尾后 v3 数据 → Robot-GPU开发机队列)
> 误删 ckpt 后,用**重建版 `A_v3early_dagger`**(v3 源被 TOS 重构原地裁尾 → 重建:video 改 copy、parquet==video 校验、1498ep/**1,938,865** 帧 vs 原 1,964,972,裁掉 ~26k=1.3% 帧)重训。job `t-20260617160628-xzj9v`,exp `v3early_dagger_devq`,inline-eval `vis_v2_merged_val`。

| step | MAE@1 | @10 | @25 | @50 |
|---|---|---|---|---|
| 8000 | 0.0109 | 0.0269 | 0.0502 | 0.0813 |
| 24000 | 0.0090 | 0.0210 | 0.0345 | 0.0484 |
| 40000 | 0.0085 | 0.0187 | 0.0294 | 0.0405 |
| **49999 (best)** | **0.0083** | **0.0180** | **0.0281** | **0.0387** |

- best ckpt(**已找回**,仅留 49999):`kai0/checkpoints/pi05_flatten_fold_v3early_dagger/v3early_dagger_devq/49999/params`
- **offline MAE 与误删前几乎一字不差**(@50 0.0387 vs 0.0388)→ 裁尾重建数据统计上一致。
- ⚠️⚠️ **真机回归(2026-06-18 用户观察)**:`v3early_dagger_devq` 真机**抓衣角成功率略降、容易抓不到**,虽 offline MAE 不变。offline 不敏感于抓取精度(§铁律再次印证)。**根因待查 = 裁尾/前裁是否动了抓取相位帧**(见 §8.10)。
- 重训命令:`submit_yaml.py train_scripts/kai/volc/v3early_dagger_devq_8gpu.yaml`

### 8.10 🔴 真机抓取回归根因(2026-06-19 查实)= 重裁 v3 视频 **PTS 未归零 → 训练时视觉与动作错位**

**结论(决定性证据)**:重构里被原地重裁的 v3 视频,**前裁切了头帧却没把 PTS 归零**,lerobot 按时间戳解码取错帧。
- 实测 04-30 ep1 top_head:`time_base=1/15360, fps=30`,首帧 `pts=17403=1.133s`=**34 帧偏移**(≈该集前裁帧数)。
- 复现 lerobot 解码:请求帧序号 200(`query_ts=200/30=6.667s`)→ 实际取到**第 167 帧**(像素差 18.9)→ **静默偏移 −34 帧、不报错**(6.667s 仍落在视频时间轴内,tolerance 通过)。
- 普查:Exp-C base v3(pts=6144=12 帧)、AWBC vis_dagger v3(pts=6144)、AH1 源 06-15-v3 —— **凡 TOS 重构重裁的 v3 全部非零 PTS**。
- ⚠️ **纠正旧认知**(memory 曾记"常规 pi05 按帧序号解码不受影响"——**错**):`data_loader.py:191-195` 常规 pi05 训练用 LeRobotDataset 也是 `delta_timestamps`(按时间戳解码),且 **`tolerance_s=30.0`**(为容忍 kai/vis 抖动时间戳放这么松)→ 30s 超大 tolerance **正好吞掉 1.1s 的 PTS 偏移** → 不报错、静默取错帧。**= 常规训练同样中招**,不止 AE。

**机制(串起全部现象)**:训练时 parquet 第 t 行(state/action[t])被配上**早 ~12–54 帧(0.4–1.8s)的图像** → 策略学成"画面比动作滞后" → 真机视觉实时对齐时按滞后补偿 → 伸向衣角**过去的位置** → **抓不到衣角**。偏移量 = 每集前裁帧数 → **逐集不一致**,更难学。
- **为何 offline MAE 盲**:train + val **同一套错位解码**,模型自洽拟合"滞后映射"→ val MAE 照样低(0.0387);真机视觉对齐才暴露 → **"offline MAE 非终判"教科书案例**。
- **为何旧 Exp-C(误删那版)真机更好**:旧 v3 当时已被 `reset_video_pts.py` 修过(PTS=0 对齐);本次重裁**重新引入** bug → 回归。两版 offline MAE 几乎相同(各自自洽)却真机不同,完全自洽。

**修复**:① `train_scripts/kai/data/reset_video_pts.py` 对所有重裁 v3 归零 PTS(确认无损改 PTS、非重编码);② 重建受影响数据集(`A_v3early_dagger`/`Task_AH1`/`vis_dagger v3`);③ 重训 Exp-C(AH1 真机前同样须修)。**根本修法**:重裁工具 `trim_video_pyav` 裁剪时即归零 PTS(memory 记过已修,本轮重构走了旧路径又犯)。

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
