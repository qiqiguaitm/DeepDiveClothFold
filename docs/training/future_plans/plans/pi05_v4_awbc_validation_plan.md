# pi05 × v4(base+dagger)KAI0 AE AWBC — v4 数据可用性验证 plan

> **建立**: 2026-06-23
> **目的**: 验证 **TOS v4 新框架数据可用** —— 用**全部 v4 base + dagger** 跑完整 **KAI0 AE AWBC 流程**(Advantage Estimator → 打标 → discretize → AWBC 训练),offline + 真机验证 v4 能训出可部署策略。**重点验证 v4 的新夹爪约定(action≠state,取主臂指令)能否解决"夹持松手"问题。**
> **状态**: 📋 **核心决策定档(§7)** — 全 v4 / AE=adv_est_v1(最早 kai0 AE)/ init=**pi05_base** / 50k;**仅待 ① discretize 阈值 ② 集群 确认 + 发话"开始实施"**。本次仍只更新文档,不实施。
> **🔀 变体**: 从 PaliGemma base 冷启动 + 100k + cnbj H20 闲时的 AWBC 变体见 [`pi05_v4_awbc_from_paligemma_plan.md`](pi05_v4_awbc_from_paligemma_plan.md)。
>
> ### ⚠️⚠️ v4 两个关键要点(实施时务必)
> 1. **norm_stats 必须对 v4 重算** —— v4 动作分布变了(夹爪 action≠state,取主臂指令)→ **绝不能复用旧 v2/v3 的 norm**,否则夹爪维归一化错位 → 静默训坏(offline MAE 也看不出)。用 `compute_norm_states_fast.py` 对 merged v4 集重算。
> 2. **夹爪不裁(原始 v4 action)** —— v4 已是"主臂意图指令",之前的夹爪裁剪(≤5mm→0)对 v4 **无意义/有害**;v4 从数据源头改了夹爪语义,正是要验证它能否解决松手问题。
> **上游**: 总纲 [`../../deployment/strategy/awbc_implementation_plan.md`](../../deployment/strategy/awbc_implementation_plan.md)(§3 4-step)· vis-native AWBC plan [`awbc_vis_task_a_full_pipeline_plan.md`](awbc_vis_task_a_full_pipeline_plan.md)· 同步脚本/v4 框架变更见记忆 [[project_tos_sync_paused_restructure]]。
> ⚠️ **铁律**: 真机为终判;VLA 报告先看 val MAE(不是 train loss);idle 轨迹 MAE 反指。

---

## 0. 为什么做这个(v4 是什么 + 验什么)

**v4 = TOS 2026-06-23 框架变更后的新标准**,与 v2/v3 三处不同:
| 维度 | v2/v3(旧)| **v4(新)** |
|---|---|---|
| trim | v2 未裁 / v3 前裁(+本地尾裁)| **前裁 + 尾裁**(采集端已做)|
| **夹爪 action** | `action == state`(被物体限位的实际位置)| **`action ≠ state`(取主臂指令,gripper-from-master)**,max\|a−s\|≈0.004 |
| intervention 列 | dagger 有 | dagger 有(base 无)|

→ **核心验证点**:v4 的"夹爪意图指令"动作是否让真机**夹持更稳(不松手/不脱落)** —— 这正是 [`gripper_action_clip_experiment.md`](gripper_action_clip_experiment.md) 想解决的问题,v4 从数据源头改了夹爪语义,**可能让夹爪裁剪实验不再需要**。本实验用 AWBC(已验证的训练范式)在 v4 全量上跑一遍,既验数据可用、又验新夹爪约定。

---

## 1. 数据(全部 v4,已落地校验)

| 域 | 来源 | 日期 | ep | 帧 |
|---|---|---|---:|---:|
| **base**(demo)| `vis_base/v4` | 13(4-23~6-04)| **1207** | **1,348,869** |
| **dagger**(纠错,含 intervention)| `vis_dagger/v4` | 12(5-29~6-23)| **789** | **1,020,660** |
| **merged** | `self_built/A_v4_base_dagger`(新建)| | **≈1996** | **≈2,369,529** |

- ⚠️ dagger 段是 AWBC 的关键:AE 要靠 inference/纠错轨迹学"低 advantage"(总纲 §3 Stage 1)。v4 dagger 有 `intervention` 列(全程干预)。
- schema 对齐:base v4 无 intervention、dagger v4 有 → **merge 时删 intervention 列**(同旧 dagger 合并惯例);两者均 action≠state、14D、前裁+尾裁、3 相机 → 可直接混。

---

## 2. KAI0 AE AWBC 流程(复用现有 pipeline)

| Stage | 做什么 | v4 注意 |
|---|---|---|
| **0–1 AE** | ✅ **复用最早 kai0-trained AE** `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1`(用户定;2026-04 训,eval.py 默认 step **100000**)| AE 从 **图像+state** 预测 stage_progress;v4 的 state 与旧一致(action 才变)→ **复用应可**;⚠️ Stage 2 后**核验 advantage 与 GT 进度正相关**,不行再考虑在 v4 上重训 AE |
| **2 打标** | `stage_advantage/annotation/eval.py Flatten-Fold KAI0 <A_v4_base_dagger>` → 每帧加 `absolute_advantage` 列 | 多卡 `--num-workers/--worker-id` 切片 |
| **3 discretize** | `discretize_advantage.py --discretion-type binary --advantage-source absolute_advantage [--stage-nums 2]` → `task_index∈{0,1}` + tasks.jsonl(`Advantage: positive/negative`)| top-30% + stage-aware |
| **4 AWBC 训练** | 克隆 `pi05_flatten_fold_awbc`(config.py:2095)→ `repo_id`=v4 labeled 集,`prompt_from_task=True` | init/步数/集群见 §3 |

---

## 3. 训练规格(克隆 `pi05_flatten_fold_awbc`)

- **config** 新建 `pi05_v4_awbc`(克隆 config.py:2095):
  - `repo_id` → `A_v4_base_dagger`(Stage 3 labeled);`base_config=DataConfig(prompt_from_task=True)`;`use_delta_joint_actions=False`(absolute)。
  - ⚠️⚠️ **norm_stats 必须对 v4 重算**(`compute_norm_states_fast.py`)—— v4 动作分布变了(夹爪 action≠state),**绝不能复用旧 v2/v3 的 norm**。
  - **夹爪 = v4 原始 action(不裁)**:v4 已是"意图指令",正是要验证的对象;裁剪(≤5mm→0)对 v4 无意义/有害。
  - ✅ **init = pi05_base**(用户定)= `CheckpointWeightLoader("/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params")`(PI 官方 pi05 机器人预训练 base)。⚠️ **不是 `mixed_1_clean`**(那是 KAI0 的 warm-start,非 pi05 base)。✅ **50,000 step**;batch 128;fsdp 8;EMA 0.9999;save 每 2k / keep 10k;`inline_eval_val_root` → v4 留出 val。
  - **推理永远喂 positive prompt** `"Flatten and fold the cloth. Advantage: positive"`(train==deploy)。
- **集群**:单节点 8 卡(cnbj Robot-North-H20 / cnsh A100,见空闲;`submit-training-job` skill)。

---

## 4. 评估(真机为终判)

| Tier | 做法 |
|---|---|
| Tier 1 offline | v4 留出 val 逐 ckpt **val MAE**(整体 + **夹爪维单列**,因为夹爪是验证重点)+ loss → 收敛 + 选 best。⚠️ AWBC 对 MAE 不敏感,只 sanity。 |
| Tier 2 标注核验 | Stage 2 后:advantage vs GT 进度 corr + 抽检高/低 advantage 帧合理性(确认 AE 在 v4 上没失效)。 |
| Tier 3 真机(决定性)| 部署 best ckpt 跑叠衣:成功率 / 各 sub-phase 通过 / **夹持稳定性(松手/脱落)= v4 主验证点**。 |

**判据(v4 可用性)**:
- ✅ **可用** = 训练收敛 + 真机能叠衣 + **夹持比旧 v3 明显更稳**(v4 新夹爪约定有效)。
- ⚠️ ≈ 旧 = v4 夹爪改动无真机增量(但数据仍可用)。
- ❌ 更差 / 不收敛 = 查 v4 pipeline(norm / 动作语义 / AE 失配)→ v4 数据/框架有问题。
- **对照**:旧 AWBC `pi05_flatten_fold_awbc`(smooth800+全dagger,action==state,2026-06-09 结果)→ 直接比"v4 新夹爪 vs 旧 action==state"。

---

## 5. 落地步骤
1. **build** `A_v4_base_dagger`(合并 v4 base 13 + dagger 12,删 intervention,episode_index 重排,视频 symlink)。
2. **重算 norm_stats**(v4 动作分布)。
3. **Stage 2** AE 打 advantage（复用 adv_est_v1）+ **核验对齐**。
4. **Stage 3** discretize → labeled 集。
5. **注册 config** `pi05_v4_awbc`,commit/push。
6. **提交 8 卡训练**。
7. **eval**:val MAE → 选 ckpt → **真机**(对照旧 AWBC),落 §4 判据。
8. 回填 results.md + 更新 master history。

---

## 6. 风险 / 注意
- **AE 在 v4 上是否失效**:AE 用图像+state 预测进度,理论上 v4 可复用;但 v4 是新采集批次,分布可能漂移 → Stage 2 后必须核验(Tier 2),不过关则在 v4 上重训 AE(回到完整 Stage 0–1,工作量大,见 [`awbc_vis_task_a_full_pipeline_plan.md`](awbc_vis_task_a_full_pipeline_plan.md))。
- **norm 复用陷阱**:v4 动作分布 ≠ 旧 → 必须重算 norm,否则夹爪维归一化错位 → 静默训坏。
- **action_dim padding**:pi05 action_dim=32,v4 是 14D → padding 逻辑与旧一致(沿用 flatten-fold config)。
- **base 早期段**:v4 base 是 4-23~6-04(含早期"work 段" + 6-04),非全量后期 → 偏干净;若要更全可等 TOS 补 v4 后期日期。

---

## 7. 决策定档(✅ 2026-06-23 用户确认)
1. ✅ **数据 = 全部 v4** base(13)+ dagger(12)= ~1996ep/2.37M 帧。
2. ✅ **AE = 最早 kai0-trained AE** `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1`(复用,step 100000;非 vis AE、不在 v4 重训)。Stage 2 后核验对齐。
3. ✅ **init = pi05_base**(PI 官方 pi05 机器人预训练 base,`openpi_cache/openpi-assets/checkpoints/pi05_base/params`)。⚠️ **不是 `mixed_1_clean`**(=KAI0 warm-start,非 pi05 base)。
4. ✅ **步数 = 50,000**。
5. 🔲 **discretize**(待定,默认建议 binary top-30% + stage-aware `--stage-nums 2`)。
6. 🔲 **集群**(待定,cnbj Robot-North-H20 / cnsh A100,见空闲)。

> 主配置已定;仅 ⑤⑥ 待定 + "开始实施" → ① build A_v4_base_dagger + 重算 norm → ② Stage 2 打标(adv_est_v1)+ 核验 → ③ Stage 3 discretize → ④ 注册 config pi05_v4_awbc → ⑤ 8 卡 50k 训练 → ⑥ eval(真机对照旧 AWBC)。

---

## 关联
- AWBC 总纲 + 复用 AE 路线:`docs/deployment/strategy/awbc_implementation_plan.md`
- config 克隆源:`kai0/src/openpi/training/config.py:2095`(`pi05_flatten_fold_awbc`)· AE config :1027(`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`)
- AE ckpt:`kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/`
- stage_advantage 脚本:`kai0/stage_advantage/annotation/{eval.py, evaluator.py, discretize_advantage.py}`
- 数据:`kai0/data/Task_A/vis_base/v4`(13日期)+ `vis_dagger/v4`(12日期)
- v4 夹爪约定背景:[`gripper_action_clip_experiment.md`](gripper_action_clip_experiment.md)(action≡state→松手问题;v4 从源头改 gripper-from-master)
