# AWBC 完整流程 on vis Task_A(Stage 0→4 重建打标)— 执行 plan

> **建立**: 2026-06-12
> **目的**: 在 **vis 的 Task_A 数据集**上**重新构建打标、完整走一遍 RECAP/AWBC 4-step pipeline**(Stage 0 标注 → Stage 1 训 estimator → Stage 2 打标 → Stage 3 离散化 → Stage 4 AWBC 训练),训出 vis-native 的 AWBC 策略。
> **上游(活跃总纲)**: [`../../../deployment/strategy/awbc_implementation_plan.md`](../../../deployment/strategy/awbc_implementation_plan.md) §3(4-Step Pipeline 详解)。代码真相源: [`../../../../kai0/stage_advantage/README.md`](../../../../kai0/stage_advantage/README.md)。
> **历史归档**: 已跑完/废弃的 AWBC 实验见 [history §AWBC](../../history/README.md)。
> **状态**: 📝 **规划草稿**,待确认决策(§9)。
> ⚠️ **铁律**: 真机为终判;val MAE 仅作收敛 sanity。

---

## 0. 与"已做过的"区别(为什么再走一遍)

| | 活跃 plan 已做(2026-06-09) | **本 plan(全流程)** |
|---|---|---|
| estimator | **复用 KAI0 官方 estimator**(`adv_est_v1/99999`,跨本体)→ 跳过 Stage 0-1 | **在 vis 上重训**(Stage 0 标 + Stage 1 训)= **vis-native** |
| 打标质量 | KAI0 估计器迁移到 vis,可能不准 | vis 自标 stage_progress_gt → 估计器更贴 vis 任务进度 |
| 成本 | 低(跳过标注) | 高(**Stage 0 标注是大头**,见 §4) |
| 价值 | 快出基线 | 验证"vis-native 打标能否让 AWBC 比复用版/SFT 更好" |

> **核心假设**: vis 自己标注 + 训练的 advantage estimator,给出的高/低 advantage 信号比复用 KAI0 估计器更可靠 → AWBC 真机更好(尤其抓取/对折关键帧加权更准)。

---

## 1. 数据集(vis Task_A)

- **用 `A_smooth800_dagger_all`(1117 ep / 1.47M 帧,vis 单本体,3相机/14D,30Hz)** = smooth800 demo + 全 dagger。
- ⚠️ **dagger 段是 Stage 1 的关键**:estimator 要学"什么算**低** advantage"必须见过 inference/纠错轨迹;纯 demo(smooth800)是天花板。dagger = Form C 信号(见上游 plan §3 Stage 1)。

---

## 2. Stage 0 — 标 `stage_progress_gt`(逐帧任务进度 GT)

**目标**: 给每帧打 stage progress(0→1),作 Stage 1 的监督。公式(stage_advantage/README): 帧在第 k/K 个 subtask 内 → `stage_progress_gt = k/K + (1/K)·(帧在该 subtask 内位置 / 段长)`。

**Task_A subtask 定义**(2 阶段):**Stage 1 = 展平+抓 → Stage 2 = 对折压平**(可细化,见 §9 待确认)。

**两种标法(成本差很大,§9 待选)**:
| | A. 全人工 | **B. 半自动(推荐)** |
|---|---|---|
| 做法 | 逐 episode 人工标 subtask 边界时间戳 | **用夹爪状态切 subtask 边界**(抓取闭合=Stage1→2 边界),线性插值 progress;人工只抽检/修正 |
| 依据 | stage_advantage/annotation README SOP | 我们已分析的夹爪分布(抓取闭合在 1-3mm,夹爪 transition 清晰可检)+ `stage_classifier_plan` |
| 成本 | ~1 周(1117 ep) | ~1-2 天(脚本 + 抽检) |
| 风险 | 准但慢 | 边界靠夹爪近似,复杂折法可能不准 → 抽检修正 |

→ **产物**: parquet 加 `stage_progress_gt` 列(写回 `A_smooth800_dagger_all` 或副本 `A_smooth800_dagger_all_sp`)。

---

## 3. Stage 1 — 训 Advantage Estimator(vis-native)

- **脚本**: `scripts/train_pytorch.py`;**config 新建** `ADVANTAGE_TORCH_VIS_TASK_A`(克隆 `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` config.py:1018,改 `repo_id`→vis 标注集、`pytorch_weight_path`→pi05 base)。
- 机制:`AdvantageEstimator`(pi0 架构,从 pi05 base 初始化);`loss_value=1.0 / loss_action=0.0`(只回归进度);`skip_norm_stats=True`;数据 `AdvantageLerobotDataset`(读 task_index 取 prompt + 采同 episode 比较帧 `his_-100_`,回归 `progress = stage_progress_gt − his_stage_progress_gt`)。
- **跑**: `torchrun --nproc_per_node=8 scripts/train_pytorch.py ADVANTAGE_TORCH_VIS_TASK_A --exp_name=run1 --save_interval 10000`(8卡,~1-2 天,到 ~100k)。
- **产物**: estimator ckpt(`experiment/ADVANTAGE_TORCH_VIS_TASK_A/run1/<step>/model.safetensors`)。

---

## 4. Stage 2 — 用新 estimator 打标(predict advantage)

- **脚本**: `kai0/stage_advantage/annotation/eval.py`(`SimpleValueEvaluator`,KAI0=双时刻 stage 级)。
- **改** `eval.py` 的 `MODELS_CONFIG_MAP` → 指向 Stage 1 新 estimator ckpt + config。
- **跑**: `python stage_advantage/annotation/eval.py Flatten-Fold KAI0 <A_smooth800_dagger_all>`(多卡用 `--num-workers/--worker-id` 切片)。
- **产物**: 每帧加 `relative_advantage` / `absolute_value` / `absolute_advantage` 列(写到 `data_KAI0_<step>/`)。
- **对齐核验(必做)**: advantage 与 GT progress corr(参考 ViVa 验证经验,需正相关);抽检高/低 advantage 帧是否符合直觉(抓取/对折关键帧 advantage 高)。

---

## 5. Stage 3 — Discretize → task_index + tasks.jsonl

- **脚本**: `discretize_advantage.py <ds> --threshold <T> --discretion-type binary --advantage-source absolute_advantage [--stage-nums 2]`。
- **产物**: 每帧 `task_index ∈ {0,1}` + `meta/tasks.jsonl`:
  - `task_index=1` → `"Flatten and fold the cloth. Advantage: positive"`;`0` → `"... Advantage: negative"`。
- **阈值 T**: top `T%` advantage 帧为 positive(默认 30);stage-aware 可 `--stage-nums 2`(每 subtask 独立分位)。§9 待定。

---

## 6. Stage 4 — AWBC 训练

- **config**: 复用/克隆 **`pi05_flatten_fold_awbc`**(活跃 plan 2026-06-09 已建)→ 改 `repo_id` 指向 Stage 3 labeled 集;`base_config=DataConfig(prompt_from_task=True)`(读 task_index→prompt)。
- **超参(沿用 flatten-fold pi05 + AWBC 续训)**: init = **warm-start `task_a_new_smooth_800_step49999`**(SFT plateau 后精修);batch128 / fsdp8 / EMA0.9999;**续训 ~15-20k step**(AWBC 是 plateau 后 frame-level 加权,步数少);norm 重算;8卡。
- ⚠️ **推理永远喂 positive prompt**: `"Flatten and fold the cloth. Advantage: positive"`(train==deploy 一致)。

---

## 7. 评估(真机为终判)

| Tier | 做法 |
|---|---|
| **Tier 1 offline** | val(留出)逐 ckpt val MAE(positive-prompt 推理)→ 收敛 + 选 ckpt。⚠️ MAE 对 AWBC 不敏感,只 sanity。 |
| **Tier 3 真机(决定性)** | sim01 部署,**positive prompt** 跑叠衣 → 成功率 / 完成帧数 / 关键 sub-phase 通过率 / 夹持稳定。 |
| **对照** | ① SFT 基线(MAE@1=0.0089);② **复用-estimator 版 AWBC**(2026-06-09 结果);③ 本 vis-native 版。→ 判"vis 自标是否更好"。 |

**判据**: vis-native AWBC 真机 > SFT 且 ≥ 复用版 → vis 自标有效;若 ≈ 复用版 → 复用够用,Stage 0-1 不值;若 < → 查标注/估计器质量。

---

## 8. Phase 拆分 + 工期

| Phase | 任务 | 工期 | 前置 |
|---|---|---|---|
| **S0** | Stage 0 标 stage_progress_gt(半自动 B 推荐)| 1-2 天(B)/ ~1 周(A)| 数据就位 |
| **S1** | Stage 1 训 vis estimator(8卡 ~100k)| 1-2 天 | S0 |
| **S2** | Stage 2 打标 + 对齐核验 | 0.5 天 | S1 |
| **S3** | Stage 3 discretize | 0.5 天 | S2 |
| **S4** | Stage 4 AWBC 训练(8卡 ~15-20k)| 1 天 | S3 |
| **S5** | eval + 真机对比(vs SFT / vs 复用版)| 1-2 天 | S4 |
| **合计** | | **~1 周(半自动 Stage 0)** | |

---

## 9. 待确认(动手前)
1. **Stage 0 标法**: 半自动 B(夹爪切边界,推荐,~1-2 天)还是全人工 A(~1 周)?
2. **subtask 定义**: Task_A 2 阶段(抓 / 对折)够吗,还是要细分(展平/抓/第1折/第2折/压平)?
3. **数据集**: `A_smooth800_dagger_all`(1117ep,含 dagger)确认?还是要别的 vis 集?
4. **discretize**: binary top-30% + stage-aware(--stage-nums 2)?
5. **AWBC init**: warm-start `task_a_new_smooth_800_step49999`(默认)?
6. **是否同时保留"复用-estimator 版"作对照**(已有 2026-06-09 结果可直接比,推荐)?

---

## 关联
- 活跃总纲: `docs/deployment/strategy/awbc_implementation_plan.md`(§3 4-step 详解 + 2026-06-09 复用版结果)
- 代码: `kai0/stage_advantage/{README.md, annotation/{eval.py, evaluator.py, discretize_advantage.py}}` · `scripts/train_pytorch.py`
- estimator config: `kai0/src/openpi/training/config.py:1018`(`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`,克隆改 vis)
- 夹爪分析(半自动 Stage 0 依据): `temp/gripper_zoom_*.png` + [`gripper_action_clip_experiment.md`](gripper_action_clip_experiment.md)
- 数据: `kai0/data/Task_A/self_built/A_smooth800_dagger_all`(1117ep)
