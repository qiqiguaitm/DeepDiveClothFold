# 修正版 Plan A — Conditioning 预合并单源路线 (kai + vis_base/v3)

> **目的**: 干净地重做 Plan A (embodiment conditioning),验证 **domain token 消歧能否压住 kai/vis 双模态抖动并超越 vis-only baseline**。之前的 conditioning 实验全跑在 broken 的 `datasets_yaml`/ConcatDataset 路径上,从未真正检验过 conditioning,本 plan 改走已证健康的**物理预合并单源路径**。
> **状态**: 📋 规划 (2026-06-04)
> **vis 数据**: `kai0/data/Task_A/vis_base/v3` (按日期分,19 目录,合计 **1940 ep / 2.53M frames**)
> **关联**:
> - 根因核查 + 官方对比: [`../../analysis/pi05_cross_embodiment_training_deep_dive.md`](../../analysis/pi05_cross_embodiment_training_deep_dive.md)
> - 崩溃实证: [`../../history/experiments/conditioning_vs_action_representation_ablation.md`](../../history/experiments/conditioning_vs_action_representation_ablation.md)
> - conditioning 方法对比: [`../../history/experiments/xvla_conditioning_methods_results.md`](../../history/experiments/xvla_conditioning_methods_results.md)
> - 跨本体战略 (21° 腕姿双模态): [`../../../deployment/strategy/cross_embodiment_strategy.md`](../../../deployment/strategy/cross_embodiment_strategy.md) §1.3 / §2.1
> - 健康预合并模板: `train_scripts/kai/data/build_xvla_exp1_hard_merged.py`

---

## 0. 背景与纠错 (为什么之前的 Plan A 不算数)

代码级核查 (2026-06-04) 结论:

1. **之前 conditioning 全崩,但凶手不是 conditioning** —— E3.6 (无 cond) / Track C (cond) / Action-delta 三连崩 (val MAE ≈ 0.47 predict-zero),共同点是**都走 `datasets_yaml` → `_create_concat_torch_dataset`** (`data_loader.py:161/203`)。conditioning 接线本身正确 (`pi0.py:247-251`,05-21 passthrough bug 已修)。
2. **per-source norm 在代码里不存在** —— `transform_dataset` (`data_loader.py:340`) 对整个 ConcatDataset 套**单一** `norm_stats` (从 `repo_id=kai0_base` 单源加载,`config.py:351`)。`InjectDatasetId` 只塞 domain_id,**不切换 norm**。E3.6 config 名 `xvla_e3_6_per_ds_norm_no_cond` 与 yaml 注释 "per-source norm" **均为误导,机制从未实现**。
3. **单一 norm 不是充分原因** —— Hard Prompt 混训 (`xvla_exp1_hard_prompt_merged_uc`) 同样单一 norm,但走**物理预合并单数据集** (`repo_id=.../xvla_exp1_hard_merged`,无 datasets_yaml),**健康** (vis val ~0.008)。分水岭 = **预合并单源路径 (健康) vs datasets_yaml ConcatDataset 路径 (崩)**。
4. **连 vis×1 也崩** —— E3.6 实际用 `e3_6_no_cond_kai_vis_joint.yaml` 是 vis×1 (ablation 文档写 "×7" 是笔误),说明崩与过采样无关,是**路径本身**的问题。

→ **两个不同的坑**:

| 坑 | 触发 | 症状 | 性质 |
|---|---|---|---|
| **A: offline collapse** | datasets_yaml ConcatDataset + 单 norm | offline MAE 0.47 + 真机不动/抖 | 管线 bug |
| **B: 真机抖动** | 物理预合并 (`mixed_pure2`、1:1 merge) | offline 健康但真机抖 > 纯 vis | 真实 kai/vis 双模态冲突 (21° 腕姿) |

**本 plan 的设计**:走预合并单源路径**绕开坑 A**,然后用 **conditioning (domain token) 专门测能否压住坑 B**。这是之前从没干净测过的命题。

---

## 1. 核心问题与假说

| # | 问题 | 假说 | 判据 |
|---|---|---|---|
| **Q1** | conditioning token (kai=0/vis=1) 能否在推理 (固定 token=vis) 时把 kai/vis 双峰拆开,消除真机抖动? | H1: 能 (token 显式消歧 > 观测隐式消歧,因 kai/vis 观测太像) | 真机抖动 metric ≤ vis-only |
| **Q2** | 加 kai 数据 co-train (有 conditioning) 能否**超越** vis-only baseline (成功率 / OOD 泛化)? | H2: 边际提升 (kai 提供场景多样性) 或持平 | 真机成功率 ≥ vis-only |
| **Q3** | conditioning 是否必要? (vs 预合并无 cond) | H3: 必要 (无 cond 的预合并 = 坑 B,会抖) | 对照 control 组 |

> ⚠️ **真机为终判** (沿用 data_root_cause_probe 铁律): offline per-source MAE 只用于 ① 训练健康闸门 ② 选 ckpt ③ 同验证集相对差。conditioning 的价值 (消歧/减抖) 在逐帧 MAE 上未必显著 → **必须真机对比**。

---

## 2. 数据构建 — 预合并 `kai_vis_v3_merged` (单数据集 + dataset_id 列)

**改写 `build_xvla_exp1_hard_merged.py` 为 `build_kai_vis_v3_merged.py`**,产出**一个** LeRobot 数据集:

| 源 | domain_id | 角色 | 约量 |
|---|---:|---|---:|
| `kai0/data/Task_A/kai0_base` | 0 | kai 官方 | 3055 ep |
| `kai0/data/Task_A/kai0_dagger` | 0 | kai dagger (可选 knob,见注) | 3457 ep |
| `kai0/data/Task_A/vis_base/v3/*-v3` (19 日期全合) | 1 | **vis 部署目标** | 1940 ep |

产出: `kai0/data/Task_A/self_built/kai_vis_v3_merged/`

构建要点 (照搬模板的正确做法 + 一处新增):
1. **重写 parquet** 的 `episode_index` / `index` (全局帧) / `task_index` 为合并后连续值 (模板已做,必须对,因 lerobot `__getitem__` 从 parquet 读这些列)。
2. **新增 `dataset_id` 列** = domain_id (kai→0, vis→1),逐帧写入 parquet。
   - 透传链已就绪: `transforms.py:105` 的 repack 会把 frame 里的 `dataset_id` 透传到 obs → `pi0.py:247` 消费。
   - ⚠️ **验证点 V1**: 确认 lerobot `LeRobotDataset.__getitem__` 会把自定义 parquet 列 (`dataset_id`) 带进 frame dict。若不会,则加一个极简 transform `ReadDatasetIdFromTaskIndex` (从 `task_index` 映射 0/1 → `data["dataset_id"]`),插在 repack 之前。
3. `meta/tasks.jsonl`: 2 条 (0=kai prompt, 1=vis prompt) 或统一 prompt — **conditioning 用 token 区分,prompt 可统一为 `"Flatten and fold the cloth."`** (避免 prompt 与 token 双重信号混淆)。
4. videos 全用 **symlink** (模板已做,省空间)。
5. **vis 验证集**: 从 vis_base/v3 **留出最后 ~50-100 ep** (建议取最近日期 `2026-05-28-v3` 整段) 作 `vis_v3_val`,**不进训练合并集**,重算其 meta。inline_eval 用它。

> **注 (dagger knob)**: kai domain 默认含 base+dagger (与历史 Hard Prompt 对齐)。若想隔离 dagger 影响,可出一个 `kai_vis_v3_merged_nodagger` 变体 (kai 仅 base 3055)。与 [`dagger_validity_and_finetune_comparison.md`](dagger_validity_and_finetune_comparison.md) 正交,本 plan 不展开。

**norm_stats** (合并后单一,够用 —— kai/vis 仅差 0.5σ):
```bash
kai0/.venv/bin/python kai0/scripts/compute_norm_stats.py --config-name=xvla_plan_a_premerge_cond_v3
# 产出 assets/<asset_id>/norm_stats.json,含 q01/q99 (pi05 quantile norm 需要)
```
- ⚠️ **验证点 V2**: 确认 norm_stats.json 含 `q01`/`q99` 字段 (pi05 `use_quantile_norm=True`)。

---

## 3. 实验矩阵 (4 组,真机对比)

| 组 | config (建议名) | init | 数据 | cond | 阶段 | 作用 |
|---|---|---|---|---|---|---|
| **Baseline** | `pi05_vis_v3_base` | pi05_base | **vis_base/v3 only** (单源) | 无 | 单阶段 50k | 要超越的对象 |
| **Plan A** | `xvla_plan_a_premerge_cond_v3` | pi05_base | `kai_vis_v3_merged` (预合并) | **token 2域** | 单阶段 50k | 主实验 (Q1/Q2) |
| **Control** | `xvla_plan_a_premerge_nocond_v3` | pi05_base | `kai_vis_v3_merged` | **无** | 单阶段 50k | 隔离 conditioning 价值 (Q3,= 坑 B 复现) |
| **Plan A+ (可选)** | `..._cond_v3_visft` | Plan A 终 ckpt | **vis_v3 only** | token (固定 vis) | 轻量 ~5k, lr↓ | 两阶段: co-train → 贴 vis 分布 |

> 最小闭环 = Baseline + Plan A + Control 三组 (一次性回答 Q1/Q2/Q3)。Plan A+ 视前三组结果再决定。

---

## 4. 训练配置 (新增 TrainConfig,镜像现有约定)

**Plan A 主 config** (`config.py` 新增):
```python
TrainConfig(
    name="xvla_plan_a_premerge_cond_v3",
    model=pi0_config.Pi0Config(pi05=True, action_head_cond_num_domains=2),
    data=LerobotAgilexDataConfig(
        repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/kai_vis_v3_merged",
        # ❌ 不要 datasets_yaml — 走单源路径 (绕开坑 A)
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,   # abs joint (与健康 Hard Prompt 一致)
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1.5e-5, decay_steps=50_000, decay_lr=1.5e-6),
    ema_decay=0.9999,
    num_train_steps=50_000,
    keep_period=10_000, save_interval=2_000,
    num_workers=16,          # ⚠️ = 16 × 节点数 (1节点16 / 2节点32 / 3节点48), 勿默认64
    batch_size=128, fsdp_devices=16,
    inline_eval_val_root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/vis_v3_val",
    inline_eval_n_frames=200, inline_eval_every=4,
    inline_eval_dataset_id=1,   # vis (inline eval 必须传, 否则 pi0.py:247 跳过 token)
),
```
- **Baseline / Control**: 同上,Baseline 改 `repo_id=vis_base/v3`(需先合并 v3 为单源)+ 去掉 `action_head_cond_num_domains`;Control 去掉 `action_head_cond_num_domains` 但保留合并集。
- 严格控制变量:四组同 init / 同 step / 同 batch / 同 lr schedule。

---

## 5. 关键代码改动 / 验证清单

| ID | 项 | 动作 |
|---|---|---|
| C1 | merge 脚本 | 改 `build_xvla_exp1_hard_merged.py` → 新增 `dataset_id` 列写入 + 指向 vis_base/v3 19 日期 |
| V1 | dataset_id 透传 | 确认 parquet 列 `dataset_id` → frame dict → `obs.dataset_id`;否则加 `ReadDatasetIdFromTaskIndex` transform |
| V2 | norm quantile | 确认合并集 norm_stats.json 含 q01/q99 |
| V3 | 训练健康 | step 2000 ckpt mu absmax check (沿用 smoke 习惯);train loss 正常下降 |
| **D1** | **部署** | vis 真机 client **必须传 `dataset_id=1`** (否则 token 被跳过 → 退化为无 cond → 抖) |

---

## 6. 评估协议

**Offline (健康闸门,非终判)**:
- per-source val MAE@1/10/25/50,**分别用各源自己的 norm 算** (sanity);健康标准: vis MAE@1 与 Baseline 同量级 (~0.008),**绝不能 ≈0.47** (那就是又踩坑 A)。
- conditioning sanity: 同一帧切 `dataset_id=0` vs `1`,输出应**明显不同** (证明 token 起作用)。

**真机 (终判,vis 机器)**:
| metric | 说明 |
|---|---|
| 抖动 (action diff p99 / 空桌面抖动目测) | **Q1 核心** —— Plan A vs Baseline vs Control |
| 抓衣角成功率 (30 ep 固定场景) | Q2 |
| 完整折叠成功率 | Q2 |
| OOD 场景成功率 (3 OOD × N) | Q2 (kai 多样性是否帮泛化) |

---

## 7. 决策树

```
Plan A offline vis MAE ≈ 0.47 ?
  是 → 又踩坑 A (预合并/列写错或 norm 错) → 查 V1/V2,勿继续真机
  否 (健康) → 真机评估
        Plan A 抖动 ≤ Baseline 且 成功率 ≥ Baseline ?
          是 → ✅ conditioning 路线成立 → 考虑 Plan A+ 两阶段进一步贴 vis
          否,但 Control 抖得更厉害 → conditioning 有效 (压住了部分坑 B),但 kai 数据净收益为负 → 回到 vis-only,conditioning 留给未来真异构
          否,且 Control 与 Plan A 差不多 → conditioning 对这对"双胞胎"无用 (观测/token 都难消歧) → 结论: 同任务近同构场景 kai0 边际价值低,vis-only 即最优,把 conditioning 机器留给未来真异构数据
```

> 无论哪个分支,都得到一个**确定结论**:要么 conditioning 路线可用,要么坐实"同任务近同构下 kai0 无用、vis-only 最优"。两者都解决了用户最初的困惑。

---

## 8. 执行步骤 checklist

- [ ] **S1** 改写 merge 脚本,产出 `kai_vis_v3_merged` (含 dataset_id 列) + `vis_v3_val` (vis 留出) + `vis_v3_base` (Baseline 单源合并)
- [ ] **S2** 验证 V1 (dataset_id 透传) — 小样本 dataloader dump 一条样本看 `obs.dataset_id`
- [ ] **S3** `compute_norm_stats.py` 三个 config,验证 V2 (q01/q99)
- [ ] **S4** 新增 4 个 TrainConfig
- [ ] **S5** smoke (1-2 ep × 100 step) 验证 mu PASS + conditioning sanity (dataset_id=0/1 输出不同)
- [ ] **S6** 提交 4 组训练 (submit-training-job;num_workers=16×节点数)
- [ ] **S7** offline 健康闸门 (per-source MAE,勿 0.47)
- [ ] **S8** 选 ckpt → vis 真机对比 (抖动 + 成功率)
- [ ] **S9** 回填结果到 `xvla_conditioning_methods_results.md` + 更新 deep-dive 文档结论

---

## 附: 避坑清单 (本 plan 专属铁律)

1. **不走 `datasets_yaml`** —— 任何 kai+vis 混训都先**物理预合并成单数据集**。datasets_yaml/ConcatDataset 路径已证三连崩,在修好它之前禁用。
2. **dataset_id 必须逐帧带** —— 预合并单源没有 InjectDatasetId,domain 信息必须以 parquet 列形式存在并透传。部署同样必须传 `dataset_id=1`。
3. **prompt 统一** —— conditioning 用 token,prompt 别再加 `[KAI]/[VIS]` 前缀,避免双重信号。
4. **跨 session 数字 fresh 复测** —— 本 plan 引用的 baseline MAE (~0.008) 仅作量级参考,正式对比前重测。
5. **真机为终判** —— offline 只做健康闸门与选 ckpt。
