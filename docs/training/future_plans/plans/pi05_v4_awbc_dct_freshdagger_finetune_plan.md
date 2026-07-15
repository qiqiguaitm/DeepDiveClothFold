# pi05_v4_awbc_dct × 最新采集数据 KAI0 AE AWBC 续训微调 — 训练 plan

> **建立**: 2026-07-03
> **目的**: 用**最新采集的 dagger v4(2026-06-29 ~ 07-03)** + **同等数量的最新 base v4**,经 **KAI0 AE 打标 + discretize**,在 **`pi05_v4_awbc_dct/49999` ckpt 基础上续训微调 30k 步**,验证"补最新纠错数据"能否**修掉真机夹爪"微微张开"**问题。
> **状态**: 📋 **plan 草拟**(本文件只规划,不实施)。
> **动机(本 session 诊断)**: `pi05_v4_awbc_dct/49999` 训练用的 `A_v4_base_dagger` 是**新旧 v4 混合集** —— 61% 的 ep(旧 4-5 月 base)是 `action==state`(旧语义),仅 39%(6 月 dagger)是 `action≠state`(gripper-from-master 新语义)。混合导致模型学不到一致的夹爪行为 → 真机抓取中夹爪"微张开"。本实验用**最新的、100% 新语义**的 dagger 续训修正。
> ⚠️ **铁律**: 真机为终判;VLA 报告先看 val MAE(不是 train loss);夹爪维单列看。

---

## 0. 核心思路(1 句话)

在已收敛的 DCT-loss v4 AWBC ckpt 上,**只用最新 5 天(06-29~07-03)的 dagger 纠错数据 + 配平的最新 base**,走同一条 KAI0 AE AWBC 打标流程,**低 LR 续训 30k** → 把夹爪行为"拉"回一致的 gripper-from-master 语义,同时保留 DCT 平滑。

---

## 1. 数据(全部已落地校验)

### 1.1 dagger(最新 5 天,100% 新语义)✅
| 日期 | ep | 夹爪 \|a-s\| | nonzero |
|---|---:|---|---|
| 2026-06-29-v4 | 100 | 0.00151 | 99% |
| 2026-06-30-v4 | 107 | 0.00153 | 99% |
| 2026-07-01-v4 | 106 | 0.00137 | 98% |
| 2026-07-02-v4 | 68 | 0.00153 | 99% |
| 2026-07-03-v4 | 125 | 0.00136 | 89% |
| **小计** | **506** | | **均 action≠state** |

### 1.2 base(配平 ~506ep,取最新日期累加)
base v4 **没有 6 月的成规模数据**(06-04/18/28 仅 1~21ep),故"最新 base"实为回溯到 5 月:

| 累加(新→旧)| ep | 累计 | 夹爪语义 |
|---|---:|---:|---|
| 06-28 + 06-18 + 06-04 | 1+1+21 | 23 | ⚠️ 06-28/18 是新语义(共2ep), 06-04 旧语义 |
| + 05-18 | 201 | 224 | action==state(旧)|
| + 05-10 + 05-09 + 05-08 | 95+30+98 | 447 | action==state(旧)|
| + 05-07 | 20 | 467 | action==state(旧)|
| (+ 05-06 取 ~39ep 补到 506)| ~39 | ~506 | action==state(旧)|

→ **base ≈ 506ep**(06-28 回溯到 05-06 部分);merged ≈ **1012ep**。

### ⚠️⚠️ 关键数据风险(必须知晓)
**"最新 base" 里除 06-28/06-18(共 2ep)外全是 `action==state` 旧语义** —— 这正是原 ckpt 夹爪问题的来源。三个应对(§7 决策待定):
- **方案 A(字面执行,用户默认)**: 就用最新 ~506ep base 配平。理由:base 是**专家 demo**,夹爪"抓取时闭合"这一 GT 行为本身正确(旧/新语义在闭合值上差异极小 ~0.0015);base 提供"别忘记专家行为"的锚,新 dagger 提供夹爪纠正信号。
- **方案 B(最干净语义)**: **只用 2ep 新语义 base + 506ep 新 dagger**(dagger 主导)。语义最纯,但样本失衡、base 太少。
- **方案 C(dagger-only)**: 干脆不配 base,506ep 纯新 dagger 续训。最聚焦夹爪修正,但可能过拟合纠错分布/遗忘 demo。
> **建议**: 先 A(符合用户"同等数量最新 base");若真机夹爪仍不稳,再对照 C。

---

## 2. KAI0 AE AWBC 流程(复用现有 pipeline)

| Stage | 做什么 | 注意 |
|---|---|---|
| **0 build** | 合并 506 dagger + ~506 base → `self_built/A_v4_freshdagger_ft`(删 intervention 列、episode_index 重排、视频 symlink、**norm_stats 对该集重算**)| 复用 `build_v4_awbc_merged.py`(改 BASE_DATES/DAGGER_DATES 为本 plan 的日期)|
| **1 AE** | ✅ 复用 `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1`(step 100000)| 与原 v4 AWBC 同一 AE,保证可比;Stage 2 后核验 advantage vs GT 进度正相关 |
| **2 打标** | `stage_advantage/annotation/eval.py Flatten-Fold KAI0 <A_v4_freshdagger_ft>` → 加 `absolute_advantage` 列 | 多卡 `--num-workers/--worker-id` |
| **3 discretize** | `discretize_advantage.py --discretion-type binary --advantage-source absolute_advantage --threshold 30 --stage-nums 2` → task_index∈{0,1} + tasks.jsonl | 与原 v4 AWBC 同 top-30%,保持可比 |
| **4 续训** | 克隆 `pi05_v4_awbc_dct`,**换 init=续训 ckpt + repo_id=新集 + 30k + 低 LR**(见 §3)| 保留 `use_dct_loss=True` |

---

## 3. 训练规格(克隆 `pi05_v4_awbc_dct` → 续训微调)

- **config** 新建 `pi05_v4_awbc_dct_freshft`(克隆 config.py `pi05_v4_awbc_dct`):
  - `repo_id` → `self_built/A_v4_freshdagger_ft`(Stage 3 labeled);`base_config=DataConfig(prompt_from_task=True)`;`use_delta_joint_actions=False`。
  - ✅ **model 保留 `use_dct_loss=True`**(继承 DCT 平滑,唯一目的是别丢已有的抗抖能力)。
  - ✅ **init = 续训 ckpt(非 pi05_base)** = `CheckpointWeightLoader("/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_v4_awbc_dct/pi05_v4_awbc_dct/49999/params")`。⚠️ 这是 finetune-from-ckpt,不是 warm-start base。
  - ⚠️⚠️ **norm_stats 对新 merged 集重算**(数据分布变了;夹爪不裁,原始 v4 action)。
  - **LR(低,续训收敛 ckpt)**:CosineDecay warmup **500** / peak **1e-5** / decay **30k** / end **1e-6**(比原训 1.5e-5 更低,避免破坏已收敛权重;冷启动配方绝不用)。
  - ✅ **30,000 step**;batch 128;fsdp 8;EMA 0.9999;save 每 2k / keep 全程;`inline_eval_val_root` → v4 留出 val。
  - **推理永远喂 positive prompt** `"Flatten and fold the cloth. Advantage: positive"`(train==deploy)。
- **集群**: 单节点 8 卡(cnbj Robot-North-H20 / cnsh A100,见空闲;`submit-training-job` skill)。

---

## 4. 评估(真机为终判 —— 夹爪是本实验主判据)

| Tier | 做法 |
|---|---|
| Tier 1 offline | v4 留出 val 逐 ckpt val MAE(整体 + **夹爪维 idx 6/13 单列**)+ loss。⚠️ AWBC 对 MAE 不敏感,只 sanity。 |
| Tier 2 标注核验 | Stage 2 后 advantage vs GT 进度 corr;抽检高/低 advantage 帧。 |
| Tier 3 真机(决定性)| 部署 best ckpt 跑叠衣:**抓取过程夹爪是否还"微张开"**(= 本实验主判据)+ 成功率 + 夹持稳定性。 |

**对照**:
- **续训前 `pi05_v4_awbc_dct/49999`**(同真机协议)→ 直接比"补最新 dagger 前后夹爪稳定性"。
- (可选)方案 C dagger-only 续训 → 隔离 base 配平的影响。

**判据**:
- ✅ 成功 = 真机夹爪不再微张开 + 成功率 ≥ 续训前。
- ⚠️ ≈ = 夹爪略好但仍有,考虑叠加部署端夹爪 clamp(见 [[gripper_action_clip_experiment.md]] / 本 session 诊断的 clamp idx6/13→state 方案)。
- ❌ 更差 = 数据/语义未修好 → 查 norm / 走方案 C。

---

## 5. 落地步骤
1. **build** `A_v4_freshdagger_ft`(506 dagger[06-29~07-03] + ~506 最新 base,删 intervention,重排,symlink)。
2. **重算 norm_stats**(新 merged 集)。
3. **Stage 2** AE 打 advantage(adv_est_v1)+ 核验对齐。
4. **Stage 3** discretize top-30% → labeled。
5. **注册 config** `pi05_v4_awbc_dct_freshft`(克隆 dct → init=49999/params + repo_id + 30k + LR peak1e-5 + 保留 DCT),commit/push。
6. **提交 8 卡 30k 续训**(`submit-training-job`)。
7. **eval**:val MAE(夹爪单列)→ 选 ckpt → **真机**(对照续训前 ckpt,看夹爪微张开)。
8. 回填 results.md + 更新 master history。

---

## 6. 风险 / 注意
- ⚠️ **base 语义污染(最大风险)**: "最新 base" 除 2ep 外全是 action==state 旧语义(§1.2)→ 可能稀释夹爪修正。真机不达标就转方案 C(dagger-only)对照。
- **续训 vs 重训**: 从收敛 ckpt 续训 30k,LR 必须低(1e-5),否则破坏已学好的手臂运动/DCT 平滑。
- **AE 复用**: adv_est_v1 在最新 dagger 上是否仍对齐 → Stage 2 后核验(新采集批次分布可能漂移)。
- **样本失衡**: 506 dagger(纠错)vs 506 base(demo)≈ 1:1;若 dagger 过采导致过度纠正,可调配比。
- **夹爪量级极小**(全量程 ~0.07)→ 任何残余误差都会真机可见;续训修不干净时,部署端 clamp(idx6/13→state)是 0 成本兜底。

---

## 7. 决策定档(⏳ 待用户确认)
1. ✅ **数据 = 最新 dagger 5 天(506ep,06-29~07-03)+ 配平最新 base(~506ep)**。
2. ✅ **AE = adv_est_v1**(复用,step 100000)。
3. ✅ **init = 续训 `pi05_v4_awbc_dct/49999/params`**(finetune,非 base)。保留 `use_dct_loss=True`。
4. ✅ **步数 = 30,000**。LR = warmup500 / peak1e-5 / decay30k / end1e-6。
5. 🔲 **base 配平方案**(A 用户默认最新~506 / B 仅2ep新语义 / C dagger-only)—— 待选,建议先 A。
6. 🔲 **集群**(cnbj H20 / cnsh A100)—— 待定。

> 主配置已定;仅 ⑤⑥ 待定 + "开始实施" → build → 重算 norm → AE 打标+核验 → discretize → 注册 config → 8 卡 30k 续训 → eval(真机对照续训前,看夹爪)。

---

## 关联
- 续训源 ckpt: `kai0/checkpoints/pi05_v4_awbc_dct/pi05_v4_awbc_dct/49999`(DCT-loss v4 AWBC,从 pi05_base 训 50k)
- DCT config 来源: `kai0/src/openpi/training/config.py`(`pi05_v4_awbc_dct`)· DCT plan [`vlanext_dct_then_soft_connection_plan.md`](vlanext_dct_then_soft_connection_plan.md)
- 姊妹 plan(全 v4 AWBC): [`pi05_v4_awbc_validation_plan.md`](pi05_v4_awbc_validation_plan.md) · [`pi05_v4_awbc_from_paligemma_plan.md`](pi05_v4_awbc_from_paligemma_plan.md)
- build 脚本: `train_scripts/kai/data/build_v4_awbc_merged.py`(改 BASE_DATES/DAGGER_DATES)
- AE: `kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/` · `kai0/stage_advantage/annotation/{eval.py, discretize_advantage.py}`
- 夹爪问题背景(本 session 诊断 + clamp 兜底): [`gripper_action_clip_experiment.md`](gripper_action_clip_experiment.md)
- 数据: `kai0/data/Task_A/vis_dagger/v4/{2026-06-29~07-03}` + `vis_base/v4/{最新回溯}`
