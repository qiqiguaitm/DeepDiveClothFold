# AWBC / RECAP Advantage 升级实施方案

> **目的**: 把当前 deepdive_kai0 简化版的 "全 dagger 标 positive" 升级为完整 RECAP 4-step pipeline — 用 advantage estimator 给每帧打 ground-truth advantage 值, 再做 advantage-weighted behavior cloning.
> **现状 (2026-06-06)**: 🟢 **启动传统路线** — smooth800 SFT 已 plateau (MAE@1=0.0089) + dagger 已积累 6 个日期, 触发条件达成. **复用已训好的 advantage estimator (跳过 Stage 0-1), 在 smooth800+全dagger 上做一次 AWBC**. 见下方 [⭐ 当前执行计划](#-当前执行计划-2026-06-06--传统路线-复用估计器--smooth800全dagger). ViVa 路线 (5B 视频生成 value, 算力昂贵) **暂缓**, 见 [`../../training/future_plans/plans/awbc_viva_value_comparison_plan.md`](../../training/future_plans/plans/awbc_viva_value_comparison_plan.md).
> **下一步 (2026-06-12)**: 🔜 **用 vis 的 Task_A 数据集重新构建打标、完整走一遍 AWBC 流程**(Step 0→4: stage_progress_gt 标注 → 训 estimator → eval 打 advantage → discretize → AWBC 训练)。具体执行 plan: [`../../training/future_plans/plans/awbc_vis_task_a_full_pipeline_plan.md`](../../training/future_plans/plans/awbc_vis_task_a_full_pipeline_plan.md)(📝 待确认决策)。本文档为活跃总纲。
> **🗄️ 历史归档 (2026-06-12)**: 已跑完/废弃的 AWBC 实验迁入 history —— [awbc_pi07style_experiment](../../training/history/experiments/awbc_pi07style_experiment.md)(⛔FAILED)、[awbc_v2_training_plan](../../training/history/experiments/awbc_v2_training_plan.md)、[gf2_advantage_awbc_plan](../../training/history/experiments/gf2_advantage_awbc_plan.md)、[gf0_awbc_baseline_v2_results](../../training/history/experiments/gf0_awbc_baseline_v2_results.md)。索引见 [history/README §AWBC](../../training/history/README.md)。
> **关键依赖**: 需要 dagger 段 (≈ Form C 的 inference/纠错信号) 给 estimator 区分高/低 advantage — 本次靠**加 dagger** 解决 (smooth_800 纯 demo-only 是 ViVa 计划里的天花板).
> **上游参考**:
> - 论文: [RECAP (Physical Intelligence 2025-11, arXiv 2511.14759)](https://arxiv.org/abs/2511.14759)
> - KAI0 上游实现: `/data1/tim/workspace/kai0/stage_advantage/` (已开源, 完整 4-step)
> - 预标注数据集: `Task_A/advantage/` (KAI0 官方 HuggingFace / ModelScope, 跳过 Stage 0-3 直接 Stage 4 训练可用)

---

## ⭐ 当前执行计划 (2026-06-06) — 传统路线: 复用估计器 + smooth800+全dagger

> **决策**: 先不做 ViVa(5B WAN value 模型,单卡 ~19 天/模型 + 跨集群,见 viva 对比计划),先用**已训好的 pi0-AdvantageEstimator** 出一个 AWBC 基线。**唯一改进 = 数据从 demo-only `smooth_800` 扩成 `smooth800 + 全部 dagger`** —— dagger 段提供 estimator 区分"高/低 advantage"所需的非示范/纠错信号(正是 ViVa 计划 §3 指出的 demo-only 天花板)。**复用估计器 → 跳过 Stage 0-1**,只跑 Stage 2(打标)→ 分析对齐 → Stage 3(离散化)→ Stage 4(AWBC 训练)。这次只做**一次新训练**(非多臂对比)。

### 资产(已核实就位)

| 项 | 路径 | 状态 |
|---|---|---|
| **Advantage Estimator(复用,跳过 Stage 0-1)** | `kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/{99999, 100000}` | ✅ 训到 100k(215G) |
| Stage 2/3 脚本 | `kai0/stage_advantage/annotation/{eval.py, evaluator.py, discretize_advantage.py}` | ✅ |
| **warm-start init**(AWBC 续训起点) | `kai0/checkpoints/task_a_new_smooth_800_step49999/params`(smooth800 SFT,MAE@1=0.0089) | ✅ |
| smooth800 源 | `kai0/data/Task_A/self_built/A_new_smooth_800/{base 811ep, val 26ep}` | ✅ |
| **全 dagger(6 日期)** | `kai0/data/Task_A/vis_dagger/v2/{2026-05-29,06-01,06-02,06-03,06-04,06-05}-v2` = 64+32+71+60+73+13 = **313 ep** | ✅ |

### 步骤

1. **建数据集 `A_smooth800_dagger_all`**(smooth 811 + 全 6 dagger 日期 313 ≈ **1124 ep**)。⚠️ 现成的 `A_smooth800_dagger_full` 只含**前 4** dagger 日期(227ep),不是"全 dagger" → **需扩 `build_smooth800_dagger.py` 的 `DAGGER_DATES` 到 6 个日期重建**;沿用其 build 惯例(squeeze、symlink、auto norm_stats、`chunks_size=max(1000,N)`)。
2. **V0 sanity(必做)**: 在 3-5 ep 上跑估计器,画 advantage vs GT 进度,确认估计器在 **smooth800+vis-dagger 域**上方向对、corr 合理 —— ViVa 计划 R8 担心"pi0-AE 与 vis 域不符",**必须先验证再全量**。
3. **Stage 2 打标**: `python kai0/stage_advantage/annotation/eval.py Flatten-Fold KAI0 <A_smooth800_dagger_all>`(多卡用 `--num-workers/--worker-id` 切片)→ 每帧 +`relative_advantage` / `absolute_advantage` 列。
4. **对齐 / 分析处理**: advantage 分布直方图 + per-episode corr(advantage vs 进度)过滤差 episode(参 viva 计划 `corr_filter`,|corr|<阈值剔除)+ 据分布定 discretize 阈值(median vs top-25%)。
5. **Stage 3 离散化**: `discretize_advantage.py <ds> --discretion-type binary --advantage-source absolute_advantage` → `task_index ∈ {0,1}` + `meta/tasks.jsonl`("Flatten and fold the cloth. Advantage: positive/negative")。
6. **Stage 4 AWBC 训练**: ⚠️ **本仓库 config.py 暂无 `pi05_flatten_fold_awbc`(那是 KAI0 上游的)→ 需新建 AWBC TrainConfig**:`prompt_from_task=True` / init = `task_a_new_smooth_800_step49999` / `repo_id` = labeled 集 / batch128 / warm-start 续训 ~15-20k step(SFT plateau 后精修,步数少)。推理永远喂 positive prompt。
7. **评估**: Tier1 离线 MAE(对照 SFT 0.0089,仅 sanity —— MAE 对 AWBC 不敏感)+ **Tier3 真机 rollout(决定性,成功率/throughput)**。

### 与 SFT 基线对照
warm-start 从 0.0089 SFT 续训 → 任何 < 0.0089 的离线改进 + 真机成功率提升,干净归因到 advantage 加权。**主判据 = 真机 rollout**(positive-prompt 推理)。

### ✅ 结果回填 (2026-06-09) — `pi05_flatten_fold_awbc` (smooth800+全dagger, 1117ep)
- **数据集**: `A_smooth800_dagger_all_awbc`(1117 ep / 1.468M frames;advantage 按帧 `ra≥0` 离散:**25.2%neg / 74.8%pos**)。warm-start `task_a_new_smooth_800_step49999`,50k/bs128/fsdp8。
- ⚠️ **inline-eval 全程失败**(`Prompt is required`:config `default_prompt=None` + `prompt_from_task=True`,inline-eval 路径不读 per-frame task)→ 训练正常、但**无 inline MAE**。**离线补评**(gf0 GPU,val=`vis_v2_merged_val` 30ep,prompt=`Advantage: positive`,MAE@1/10/25/50):

| step | MAE@1 | MAE@10 | MAE@25 | MAE@50 |
|---|---|---|---|---|
| 10000 | 0.0084 | 0.0153 | 0.0236 | 0.0331 |
| 20000 | 0.0082 | 0.0147 | 0.0223 | 0.0308 |
| 30000 | 0.0081 | 0.0142 | 0.0215 | 0.0294 |
| 40000 | 0.0080 | 0.0139 | 0.0210 | 0.0287 |
| **49999** ⭐ | **0.0079** | 0.0136 | 0.0205 | 0.0278 |

- **最佳 ckpt**(单调下降,末步最好):`kai0/checkpoints/pi05_flatten_fold_awbc/pi05_awbc_cnsh/49999`(@1=0.0079,略优于 SFT 基准 0.0089)。
- ⚠️ MAE 对 AWBC 不敏感(positive-prompt 离线 MAE 看不出 advantage 加权的真实收益)→ **主判据仍是真机 rollout**。结果 json:`logs/awbc_eval_mae.json`。

### 🔬 新实验 (2026-06-09) — AWBC ablation: 只用 smooth800 (无 dagger)
> **动机(控制变量)**: 本方案核心假设 = "dagger 段提供 estimator 区分高/低 advantage 所需的失败/纠错信号;纯 demo-only smooth800 的 advantage 方差 η²≈3%(天花板),信号弱"(见上 + viva 计划 §3)。本实验**去掉 dagger,只留 smooth800 的 advantage-labeled 帧**,直接对照 `pi05_flatten_fold_awbc`(smooth800+全dagger)→ 量化 dagger 段对 AWBC 的贡献。

- **数据集 `A_smooth800_awbc`**(**806 ep / 925k frames**):从已标注的 `A_smooth800_dagger_all_awbc` 里**按内容指纹抽出 smooth800 那部分**(estimator 是 **per-episode 独立**打标 → 抽出的 advantage 标签 = 重跑 smooth800-only 估计器的结果,等价且零成本)。811 中 806 唯一匹配(另 5 ep 因 build 时 squeeze 改了 action 数组、指纹不命中,占 0.6%,已记录)。norm_stats 在 806 子集上**重算**。
- **advantage 分布**: **22.0%neg / 78.0%pos** —— 比含 dagger 版(25.2%neg)**neg 更少**,正印证"demo-only 失败信号更弱"的假设(可作实验观察点之一)。
- **config**: `pi05_flatten_fold_awbc_smooth800only`(与 `pi05_flatten_fold_awbc` **逐字段一致,仅 repo_id 不同** → 唯一变量 = 有无 dagger)。warm-start 同、50k/bs128/fsdp8、`default_prompt=None`(inline-eval 同样 skip,训完离线评 MAE 口径一致)。
- **提交**: cnsh 8卡 `train_scripts/kai/volc/pi05_awbc_smooth800only_cnsh_8gpu.yaml`。
- **判据**: ① 离线 MAE 对照(sanity);② **真机 rollout 成功率 vs 含-dagger AWBC**(决定性)。预期:若含-dagger 显著赢 → 证实 dagger 失败信号是 AWBC 关键;若打平 → demo-only 的 advantage 已够(或两者都没真正用上 advantage,需回看 estimator 信号强度)。

---

## 1. 算法定位 (一句话)

VLA 训练时不再"每帧等权 BC", 而是**按每帧 advantage 值加权 BC** — 高 advantage 帧 (人类示范 / 策略成功段) 权重大, 低 advantage 帧 (策略失败段) 权重小或负. Advantage 由一个独立训练的 estimator (基于 pi0 fine-tune) 学习得到, 监督信号是手动标的 stage_progress_gt.

**与 DAgger 简化版的差别**: 简化版按 dataset 来源 (dagger=positive, inference=negative) 二值标签; 完整版按 frame 级 advantage 连续值标签.

---

## 2. AWBC 适配 / 不适配场景

| 场景 | 是否适合 | 理由 |
|---|---|---|
| **DAgger 已迭代 N 轮, val MAE@1 平台 (不再下降)** | ✅ 首选 | 数据增加边际收益递减, 需要更细致的 frame-level 加权 |
| **失败模式仍是"分布外 / 不会"类 (DAgger 适配)** | ✅ | RECAP 设计目的就是 boost DAgger 数据的训练效率 |
| **多 sub-phase 任务** (Task_A flatten-fold 多步骤) | ✅ | stage_progress_gt 天然按 subtask 标注 |
| **失败是 action edit 类 (精度差 mm)** | ❌ | 走 RLT (论文实证 RLT 解 action edit 远优于 BC 类) |
| **新任务从零建立能力** | ❌ | AWBC 假定已有 baseline policy, bootstrap 阶段用纯 DAgger |
| **单 sub-phase 任务 (无明显 stage 划分)** | ⚠️ | stage_progress_gt 标注难, 退化为按 progress 单调递增标 |

**适配的失败模式特征**: (a) DAgger 加 demo 效果递减, (b) 任务有 ≥ 2 个清晰 sub-phase 边界 (易于人工标 stage_progress_gt), (c) policy 已能完成 30%+ 任务 (有 inference rollout 可用作 negative samples).

---

## 3. 4-Step Pipeline 详解

### Stage 0 — 手动标 `stage_progress_gt` (一次性)

**目标**: 为每帧打 stage progress 值, 作为 Stage 1 estimator 的监督信号.

**流程**:
1. 对每个 episode, 人工标:
   - Episode 起止时间戳
   - Subtask 边界时间戳 (例如 Task_A 2 阶段任务: Stage 1 = 抓 → Stage 2 = 对折)
2. 脚本根据时间戳自动算每帧的 `stage_progress_gt` (0-1 浮点, 阶段内单调递增)
3. 写入 parquet 列

**工具**: KAI0 上游 `stage_advantage/annotation/README.md` 提供脚本 + 标注规范.

**输入**: Form C dataset (推荐用 inference + dagger 混合, 至少 200 episode)
**输出**: 同 dataset, parquet 多一列 `stage_progress_gt`
**工作量**: 标 500 episode (Task_A 当前体量) ~ 1 周 (操作员 + 标注员协作), 或者**用 KAI0 预标注的 `Task_A/advantage/`** 跳过此步.

### Stage 1 — 训 Advantage Estimator

**目标**: Fine-tune pi0 模型, 用 stage_progress_gt 监督回归 advantage 值.

**命令** (KAI0 上游):
```
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
    --exp_name=run1 --save_interval 10000
```

**关键 config 项** (来自 `kai0/src/openpi/training/config.py` ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD):
- 基础模型: pi0 (不是 pi0.5)
- 数据集: Form C inference + dagger 混合
- Loss: 2-step relative advantage regression (frame n vs frame n+50)
- `skip_norm_stats=True` — estimator 不需要 norm
- 通常 50k-100k step, 1-2 天 (8×A100)

**关键原理**:
- 输入: (images, state) — 与训 policy 一致
- 输出: advantage 标量 (取代 action 输出 head)
- 训练时见到 inference 段的"失败状态" + dagger 段的"成功状态", 学会区分

**为什么需要 inference 段**: 这一步是 Form C 的核心依据. 没有 inference 段 → estimator 没法学到"什么算低 advantage" → 跳过这步会让 Stage 2 输出全部 positive (失去意义).

**输出**: trained estimator ckpt (~7GB)
**工作量**: 1-2 天集群训练

### Stage 2 — 预测 Advantage 标到 dataset

**目标**: 用 Stage 1 的 estimator 给整个 dataset 每帧打 advantage 值.

**命令**:
```
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset
```

**输出**: 新 parquet (路径如 `data_KAI0_100000/data/chunk-000/`), 含两列额外:
- `relative_advantage`: frame n 与 frame n+50 的 progress 差值
- `absolute_advantage`: frame n+50 absolute value 与 frame n 的差值, 截到 [-1, 1]

**实现细节** (`kai0/stage_advantage/annotation/evaluator.py`):
- 批处理 GPU 推理 + 并行视频解码
- 整个 Task_A 500 episode ~ 几小时

**工作量**: 半天

### Stage 3 — Discretize Advantage → task_index + tasks.jsonl

**目标**: 把连续 advantage 值二值化 / N-bin, 转成 `task_index` (int64), 配合 `meta/tasks.jsonl` 给每个 task_index 一个 prompt 字符串.

**命令**:
```
bash stage_advantage/annotation/discretize_advantage.sh
```

**典型 binning** (`discretize_advantage.py`):
- 二值: advantage > median → "Advantage: positive"; ≤ median → "Advantage: negative"
- 也支持 N-slice 分箱 (例如 top-25% / mid-50% / bot-25%)

**输出**: 同 dataset, 每帧 `task_index` 指向 `tasks.jsonl` 中的 prompt
**工作量**: 半天

**与 `label_dagger_positive.py` 对比**:
- `label_dagger_positive.py`: 把所有 dagger 帧标 positive, 不看 advantage 值 — 等价于 "用 dataset 路径粗暴当 advantage"
- Stage 3 完整版: 按真实 ground-truth advantage 值打标, frame-level 精度

### Stage 4 — AWBC 训练

**目标**: 用 advantage-weighted BC 训 policy.

**命令**:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc \
    --exp_name=run1
```

**核心机制 — `prompt_from_task`**:
- 训练时不用固定 prompt, 而是从每帧的 `task_index` 查 `tasks.jsonl` 得到 prompt
- 即 "Advantage: positive" 帧用 positive prompt 学 (强 BC)
- "Advantage: negative" 帧用 negative prompt 学 (相当于 "学到的关联是: negative prompt → 这类动作")
- 推理时永远 prompt = "Advantage: positive" → 模型只生成 positive 风格动作

**与现有 `pi05_flatten_fold_awbc` config** ([`../../train_scripts/launch/run_awbc_daggeronly_gf0.sh`](../../train_scripts/launch/run_awbc_daggeronly_gf0.sh)):
- 现有: 数据来源 `Task_A/dagger_labeled` (label_dagger_positive 输出), 所有帧 positive
- 升级后: 数据来源 `Task_A/advantage` (Stage 3 输出), 按真实 advantage 区分

**工作量**: 与现有 SFT 训练相同, 30k-60k step, 2-3 天 (gf1 8×A100)

---

## 4. 现有简化版 vs 完整版对比

| 维度 | `label_dagger_positive.py` (现有简化版) | RECAP 完整版 (本方案) |
|---|---|---|
| Advantage 来源 | 数据集路径 (`Task_A/dagger/*` 全标 positive) | Estimator 预测的 frame-level 值 |
| 是否需要 inference 段 | ❌ 否 (当前 Form A 没有) | ✅ 是 (训 estimator 需要) |
| 是否需要 Stage 0 标注 | ❌ 否 | ✅ 是 (~1 周人工; 或用 KAI0 预标注) |
| Advantage 分辨率 | 二值 (整段 positive / 不录 negative) | 连续值 → 二值或 N-bin |
| Negative 样本 | 无 | inference 段失败帧 |
| 训练 config | `pi05_flatten_fold_awbc` (现成) | 同 config, 不同数据集路径 |
| 工作量 | 0 (已实现) | 4-5 天 (跳过 Stage 0 用预标注) / 2 周 (含 Stage 0) |
| 训练效果 (论文) | baseline | +显著提升 (RECAP 论文实测 ~2× throughput on hard tasks) |

---

## 5. Phase 拆分

| Phase | 任务 | 工作量 | 关键文件 / 命令 | 前置依赖 |
|---|---|---|---|---|
| **A0** | **决策点检查**: DAgger 简化版 val MAE 是否平台? 失败模式是否仍是 DAgger 适配? | 0.5 天 | 看 [`../../training/history/00_action_only_finetune_history.md`](../../training/history/00_action_only_finetune_history.md) | DAgger 已迭代 ≥ 3 轮 |
| **A1** | **数据准备**: 确认 Form C dataset (inference + dagger) 已积累 ≥ 200 ep | 0.5 天 | DAgger Phase D5 完成 | [`dagger_implementation_plan.md`](dagger_implementation_plan.md) Phase D1-D5 |
| **A2a (路径 a)** | 用 KAI0 预标注 `Task_A/advantage/` 直接 Stage 4 | 0 天数据准备 | `scripts/download_dataset.py` 拉 HuggingFace dataset | 任务结构与 Task_A 兼容 |
| **A2b (路径 b)** | 自有数据 Stage 0 手动标 stage_progress_gt | 1 周 | `stage_advantage/annotation/README.md` SOP | 操作员 + 标注员 |
| **A3** | Stage 1 训 Advantage Estimator (gf1 集群) | 1-2 天 | `scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` | A2a 或 A2b 完成 |
| **A4** | Stage 2 Predict advantage on 自有 dataset | 0.5 天 | `stage_advantage/annotation/eval.py Task-A KAI0 <dataset>` | A3 完成 |
| **A5** | Stage 3 Discretize → task_index | 0.5 天 | `discretize_advantage.sh` | A4 完成 |
| **A6** | Stage 4 AWBC 训练 + sim01 部署评估 | 2-3 天 | `scripts/train.py pi05_flatten_fold_awbc` (config 改 repo_id) | A5 完成 |

**两条路径总工作量**:
- **路径 a (用 KAI0 预标注)**: A0 + A1 + A2a + A3 + A4 + A5 + A6 = **5-6 天**
- **路径 b (完整自标)**: A0 + A1 + A2b + A3 + A4 + A5 + A6 = **2-3 周**

**推荐**: 先走路径 a (用 KAI0 预标注 estimator 测试效果). 若效果好 → 路径 b 投入自标 stage_progress_gt 给自有数据.

---

## 6. 资源与时间预算

| 资源 | 路径 a | 路径 b |
|---|---|---|
| **工程时间** | 5-6 天 | 2-3 周 |
| **人力 (操作员/标注员)** | 0 | ~1 周 (Stage 0 标 500 ep × subtask 边界) |
| **GPU 时间** | Stage 1 ~1-2 天 (8×A100) + Stage 4 ~2-3 天 (8×A100) | 同 |
| **存储** | Estimator ckpt ~7GB + AWBC ckpt ~7GB + dataset 重复 ~10GB | 同 |
| **数据集体量** | ≥ 200 ep Form C (inference + dagger) | 同 |
| **真机时间** | 0 (训练 + 离线 eval) + sim01 部署评估 ~2h | 同 |

---

## 7. 风险与兜底

| 风险 | 概率 | 影响 | 兜底 |
|---|---|---|---|
| KAI0 预标注 `Task_A/advantage/` 与自有 Task_A 任务结构不一致 | 中 | 大 | 路径 b 自标 stage_progress_gt |
| Estimator 在 inference 段失败状态拟合不充分 (因为 inference ep 比 dagger 少) | 中 | 中 | 多收一轮 DAgger 增加 inference ep 比例; 或者从 RLT replay buffer 抽 transition 反向喂 estimator |
| Stage 3 discretize 阈值不合适 (advantage 分布偏) | 低 | 中 | 改 binning 阈值 (top-25% vs median), 多试几个 config |
| AWBC config `prompt_from_task` 与 deepdive_kai0 现有 pi0.5 sidecar 不兼容 | 中 | 中 | 复用 sim01 部署的 sidecar 注册机制 (已有模板) |
| 训完 ckpt 效果不如简化版 | 低 | 中 | 退回 `label_dagger_positive.py` 简化版; 留 RECAP 数据归档备用 |

---

## 8. 与 DAgger / RLT 的关系 (重要)

| 阶段 | DAgger 主轨 | AWBC 升级 (本方案) | RLT 升级 |
|---|---|---|---|
| **数据来源** | 持续录 dagger (人示范) | DAgger Form C 累积的 inference + dagger | DAgger ckpt 当 RLT 的 frozen VLA |
| **修改对象** | VLA 全模型 (~5B) | VLA 全模型 (~5B) | actor 1M, VLA 冻结 |
| **训练成本** | 30-60k step / 实验 | Stage 1 + Stage 4 双训 | 几小时真机 + 极小算力 |
| **适合的失败** | 分布外 / 模型不会 | DAgger 已平台, 同类失败 | action edit / 单 channel offset |
| **触发顺序** | 第 1 轮 | DAgger 平台后 | DAgger 或 AWBC 平台后, 走 critical phase |

**完整升级路径**:
```
DAgger (Form C) → DAgger 加 demo 持续 → val MAE 平台 → AWBC 升级 → val MAE 再平台 → RLT 修关键 phase
```

每一步都是 **independent + composable** — RLT 用 AWBC 训出的 ckpt, AWBC 用 DAgger 录的数据, DAgger 不变.

---

## 9. 与上游 / 相关文档跳转

- 主轨 DAgger 实施 → [`dagger_implementation_plan.md`](dagger_implementation_plan.md) (Form C 决策 + Phase D1-D5)
- 另一支升级 (RLT) → [`rlt_implementation_plan.md`](rlt_implementation_plan.md)
- KAI0 上游 advantage pipeline → `/data1/tim/workspace/kai0/stage_advantage/README.md`
- RECAP 论文 → [arXiv 2511.14759](https://arxiv.org/abs/2511.14759)
- 现有简化版实现 → [`../../train_scripts/data/label_dagger_positive.py`](../../train_scripts/data/label_dagger_positive.py)
- 现有 AWBC 训练入口 → [`../../train_scripts/launch/run_awbc_daggeronly_gf0.sh`](../../train_scripts/launch/run_awbc_daggeronly_gf0.sh)
- KAI0 预标注数据集 → <https://huggingface.co/datasets/OpenDriveLab-org/Kai0> / <https://www.modelscope.cn/datasets/OpenDriveLab/Kai0>
- 跨本体战略主文档 → [`cross_embodiment_strategy.md`](cross_embodiment_strategy.md)
