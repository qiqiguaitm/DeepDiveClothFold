# Conditioning × Action Representation 2×2 Ablation

> **作用**: 在同 init (pi05_base) / 同 step (50k) / 同 lr schedule / 同 batch=128 下, 系统对比 **conditioning (有 / 无 domain conditioning)** × **action representation (absolute / delta joint)** 四个 cell 对 vis 真机折叠的影响。
>
> **背景**: 2026-05-22 用户提出 "condition delta 训废了在初始状态震动不做操作" — Action Cond × delta ckpt 真机表现退化。需要把 conditioning 与 action representation 两条因素拆开, 单独量化各自贡献, 避免归因到错的轴上。
>
> **范围**: 4 个训练 run 的 val MAE + 真机抓衣 / 折叠 / 抖动观察。Conditioning 仅指 **Action Head Cond Token (Track C 方案 A)**, 即 action expert input 端 concat 1 个 learnable domain token; 不涉及 Soft Prompt / Hard Prompt (那些在 `xvla_conditioning_methods_results.md` 跟踪)。
>
> **最近更新**: 2026-05-23
>
> **关联文档**:
> - `xvla_conditioning_methods_results.md` — Hard / Soft / Action Head 三种 conditioning 方式对照 (相同 abs action 下)
> - `../../../deployment/strategy/cross_embodiment_strategy.md` §5.3 — Track C 方案 A (Action Head Cond Token) 设计与决策
> - `00_training_history.md` — 全量训练历史榜单
> - `task_e_master_plan.md` §2.1 — Task E 早期 abs vs delta 结论 (v14 delta 同步数劣于 abs)

---

## 1. 实验矩阵

|  | **No Conditioning** | **Action Head Cond (Track C, 方案 A)** |
|---|---|---|
| **Absolute joint** (14D, kai/vis joints + grippers) | **E3.6**: `xvla_e3_6_single_norm_no_cond` (kai+vis, **vis×1, 单一 norm**) | **Track C abs**: `xvla_actcond_single_stage_joint` (kai+vis × 7 balanced) |
| **Delta joint** (12 joint delta + 2 gripper abs, `make_bool_mask(6,-1,6,-1)`) | **pi05 delta**: `pi05_flatten_fold_task_a_base_delta` (kai0_base 3055 only, single-source) | **Action Cond × delta**: `xvla_actcond_single_stage_joint_delta` (kai+vis × 7 balanced) |

> ⚠️ **不严格对称**: 左下 cell (pi05 delta) 训练数据是 kai0_base 3055 ep 单源, 没有走 datasets_yaml。这是 baseline for "delta 在 kai 单源能否收敛"。
>
> ⚠️ **2026-06-04 更正 (重要)**: 本文档此前多处把 **E3.6 标为 "kai+vis × 7 balanced + per-DS norm"——这是错的**。核查代码后确认:
> 1. E3.6 实际用 `e3_6_no_cond_kai_vis_joint.yaml` = **vis×1**(无 ×7 过采样);
> 2. **per-DS norm 在代码里根本不存在** —— `transform_dataset` (data_loader.py:340) 对整个 ConcatDataset 套**单一** norm (kai0_base 的)。原 config 名 `xvla_e3_6_per_ds_norm_no_cond` 是误导,已更名 `xvla_e3_6_single_norm_no_cond`。
> 3. **这反而强化结论**: 连 vis×1 也崩 → 凶手是 **`datasets_yaml`/ConcatDataset 路径本身**,与过采样、per-DS-norm 都无关。
> 详见 [`../../analysis/pi05_cross_embodiment_training_deep_dive.md`](../../analysis/pi05_cross_embodiment_training_deep_dive.md) 与 [`../../future_plans/plans/corrected_plan_a_conditioning_premerge.md`](../../future_plans/plans/corrected_plan_a_conditioning_premerge.md)。下方表格中关于 E3.6 的 "×7 / per-DS norm" 字样均以本更正为准。

### 1.1 共享超参 (全 4 cell 一致)

| 项 | 值 |
|---|---|
| Init | pi05_base (HF `lerobot/pi05_base`) |
| Backbone | pi05 (PaliGemma + Action Expert) |
| Train steps | 50,000 |
| lr schedule | CosineDecay, warmup 1k, peak 1.5e-5, decay→1.5e-6 |
| Batch size | 128 (global) |
| ema_decay | 0.9999 |
| keep_period | 10,000 |
| inline_eval_val_root | `Task_A/vis_v2_merged_val` (9 ep × 200 frames) |
| inline_eval_every | 每 4k step (12 个采样点 / 50k) |
| num_workers | 8 per worker × N nodes (volc 默认) |
| Compute | Beijing/Shanghai 16 H20 (FSDP) |

### 1.2 仅四个变量差异

| Config | conditioning | action | datasets_yaml | inline_eval_dataset_id |
|---|---|---|---|---|
| E3.6 (no-cond × abs) | `action_head_cond_num_domains=0` | `use_delta_joint_actions=False` | kai+vis × 7 balanced + per-DS norm | (n/a) |
| pi05 delta (no-cond × delta) | `action_head_cond_num_domains=0` | `use_delta_joint_actions=True` | (single repo kai0_base) | (n/a) |
| Track C abs (cond × abs) | `action_head_cond_num_domains=2` | `use_delta_joint_actions=False` | kai+vis × 7 balanced | 1 (vis) |
| Action Cond × delta (cond × delta) | `action_head_cond_num_domains=2` | `use_delta_joint_actions=True` | kai+vis × 7 balanced | 1 (vis) |

---

## 2. 训练状态与已知 MAE

> 2026-05-23 截图。Eval 跑完后更新此表。

| Cell | Job ID | Step | 状态 | Best Val MAE@1 | MAE@10 | MAE@25 | MAE@50 | Val set | 备注 |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| **E3.6** no-cond × abs | t-20260522201522-s72th | **50k ✅** | ❌ **COLLAPSE** | **0.4706** | **0.4732** | **0.4758** | **0.4750** | vis_v2_merged 前 50 ep | Beijing 16 H20. **与 cond × abs (0.4699) 和 cond × delta (0.4663) 几乎一致**. 同 "predict-zero" 坍塌. **=> conditioning 不是因, datasets_yaml + balanced sampling 才是真正的 culprit** |
| **pi05 delta** no-cond × delta | t-20260522192932-cldrd | **50k ✅** | DONE | **0.0116** | **0.0423** | **0.0874** | **0.1447** | vis_v2_merged 前 50 ep | Beijing 16 H20, 19h15m. (kai0_dagger 127 ep 上 @1=0.0120/@50=0.1041 — 两 val 数字接近, 说明 vis val 对 pi05 delta 不算 OOD) |
| **Track C abs** cond × abs | t-20260522194822-sqthr | **48k ✅** (训练 crash @ 48100/50000) | ❌ **TRAIN COLLAPSE** | **0.4699** | **0.4724** | **0.4749** | **0.4740** | vis_v2_merged 前 50 ep (`dataset_id=1`) | Shanghai 16 A100. **训练 SIGSEGV 在 step 48100**, 用 48000 ckpt eval. **与 cond × delta (0.4663) 几乎一致** — 同 "predict-zero" 坍塌. **=> conditioning 实现本身有问题, 不限 delta** |
| **Action Cond × delta** cond × delta | t-20260522195640-t42hs | **50k ✅** | ❌ **TRAIN COLLAPSE** | **0.4663** | **0.4690** | **0.4717** | **0.4712** | vis_v2_merged 前 50 ep (`dataset_id=1`, prompt="Flatten and fold the cloth.") | Train loss 0.005 看似收敛, 但 val MAE 全 horizon ~0.47 几乎不变 ≈ `mean(\|gt_abs\|)`. **模型坍塌到 "predict zero delta"** local minimum. 详见 §3.3 |

---

## 3. 量化对比

### 3.1 No-cond × delta on vis val (pi05 delta 在 vis 上行为正常)

| Horizon | pi05 delta on **vis_v2_val50** | pi05 delta on **kai0_dagger 127 ep** | 一致性 |
|---|---:|---:|---|
| MAE@1 | **0.0116** | 0.0120 | ✅ kai/vis 接近 |
| MAE@10 | 0.0423 | 0.0330 | vis 略高 (+28%) |
| MAE@50 | 0.1447 | 0.1041 | vis 略高 (+39%) |

**结论**: pi05 delta (no cond) 在 vis 数据上的 MAE 与 kai 数据接近, 说明 model 跨 vis/kai generalize OK。Delta 的长程退化模式正常 (@1=0.012 → @50=0.14, 12× 放大, 与 chunk-50 误差累积一致)。

### 3.2 Cond × delta 真机震动 = 训练坍塌 (核心发现 🔥)

> 用户 2026-05-22 反馈: **"condition delta 训废了在初始状态震动不做操作"**。Offline eval 提供了 smoking gun:

**Action Cond × delta on vis_v2_val50 (`dataset_id=1`, correct prompt)**:

| Horizon | MAE | 跟 pi05 delta on vis 比 |
|---|---:|---|
| MAE@1 | **0.4663** | **40×** 差 |
| MAE@10 | **0.4690** | **11×** 差 |
| MAE@25 | **0.4717** | **5.4×** 差 |
| MAE@50 | **0.4712** | **3.3×** 差 |

**关键观察**:
1. **MAE 全 horizon 几乎不变** (0.466 → 0.472), 不是典型 delta 累积模式 (应该 @1 → @50 大幅放大)。
2. **0.47 ≈ mean(|gt_abs|)** — Piper joint normalize 后绝对值平均 ~0.5。这是 "model 输出 ≈ 零" 时与 abs target 比对的 MAE。
3. **Train loss 看似正常** — step 49000 loss=0.005, gnorm=0.06, param_norm=1805. 收敛到 "predict zero delta" 局部最小值。
4. **Conditioning IS 被应用**: 测试 `dataset_id=0` vs `dataset_id=1`, MAE 分别 0.42 / 0.47 — domain_token 确实进入 model, 只是 model 本身废了。
5. **Eval 脚本 bug 已修**: 原脚本忘了传 `dataset_id` 到 obs, 触发 `pi0.py:246` 跳过 `action_head_cond_hub` token → suffix 结构异常。已 patch + commit (新增 `--dataset-id` + `--prompt`)。

**坍塌机制 (hypothesis)**:
- Delta target 在大多数 timestep 数值很小 (joint 静止 / 慢移时 delta ≈ 0)
- Action Expert input 前置 1 个 `domain_token` (init N(0, 0.02)), 在训练初期是 noise, 部分扰动 action expert 的 attention pattern
- 在 cond × abs 路线上, target ~rad 量级, 必须学到有用 motion 才能压低 loss
- 在 cond × delta 路线上, "全部输出 0" 已经能压到 loss ≈ E[delta²] ≈ 1e-4 (步态平均的 RMS 很小), 接近正常 flow-matching 残差噪声
- 模型陷入 "predict zero delta" 局部最小值, 失去 active motion 能力

**对应真机现象**:
- 初始状态 (静止) → model 输出 ≈ 0 delta → controller 跟随当前 state → 不动 ✓ (用户观察吻合)
- 真机需要主动启动 motion (e.g. 接近衣服) → model 仍输出 ≈ 0 delta → state hold → "在原地震动" (controller 在 setpoint 附近 dither) ✓

→ **Action Cond + delta 是危险组合, 不应用于部署**。

### 3.3 No-cond × abs vs No-cond × delta (delta 的"纯成本", 老 SOTA 参考)

| Horizon | pi05 delta on vis | abs ref (mixed_pure2_1800_6000, pi05_base + 7900 ep) | Δ |
|---|---:|---:|---|
| MAE@1 | 0.0116 | 0.0085 | +36% worse |
| MAE@10 | 0.0423 | 0.0168 | +152% worse |
| MAE@50 | 0.1447 | 0.0337 | +330% worse |

> ⚠️ 不严格公平 (abs ref 数据多 2.5×, val 也不完全一致), 但方向明确: delta 在 long-horizon 严重退化。Task E 早期 (master plan §2.1) 已发现同样模式。**保守估计 delta 在 @1 ~+25%, 在 @50 ~+200%**。

---

## 4. 2×2 ablation 汇总 (val MAE on vis_v2_val50)

### 4.1 单步精度 MAE@1

| Data 路线 | No Cond | Action Cond |
|---|---:|---:|
| Absolute + **kai+vis × 7 balanced (datasets_yaml)** | **E3.6: 0.4706** ❌ | **Track C: 0.4699** ❌ |
| Delta + **kai+vis × 7 balanced (datasets_yaml)** | (未跑) | **0.4663** ❌ |
| Delta + **single-source kai0_base 3055** | **pi05 delta: 0.0116** ✓ | (未跑) |

### 4.2 长程精度 MAE@50

| Data 路线 | No Cond | Action Cond |
|---|---:|---:|
| Absolute + datasets_yaml | **0.4750** ❌ | **0.4740** ❌ |
| Delta + datasets_yaml | — | **0.4712** ❌ |
| Delta + single-source | **0.1447** ✓ | — |

### 4.3 vis 真机 (B 真机) 折叠任务

| Cell | 真机表现 |
|---|---|
| E3.6 abs + datasets_yaml | 预期同 cond × delta: 静止/震动 (offline 0.47 collapse) |
| Track C abs + datasets_yaml | 同上 |
| Action Cond × delta + datasets_yaml | **在初始状态震动, 不操作** ❌ (与 offline collapse 吻合) |
| pi05 delta single-source | 真机待测 (offline @1=1.2%, @50=14.5%, 长程估计欠平滑) |

### 4.4 ⭐⭐⭐ 关键发现 (重新归因 2026-05-25) — `datasets_yaml + balanced sampling` 本身 broken, 与 conditioning 无关

之前归因 "Action Head Cond Token (方案 A) 实现整体失败" **被 E3.6 推翻**. E3.6 **没有任何 conditioning** (vanilla pi05), 但用 kai+vis × 7 balanced datasets_yaml 训出来的 ckpt MAE@1=0.4706, 与两个 Action Cond cells 几乎一致:

| Cell | Cond | Action | Data | MAE@1 | MAE@50 |
|---|---|---|---|---:|---:|
| **E3.6 (no-cond × abs)** | ❌ none | abs | balanced yaml | 0.4706 | 0.4750 |
| Track C abs (cond × abs) | ✅ Action Head | abs | balanced yaml | 0.4699 | 0.4740 |
| Action Cond × delta | ✅ Action Head | delta | balanced yaml | 0.4663 | 0.4712 |
| pi05 delta | ❌ none | delta | **single-source** | **0.0116** | **0.1447** |

3 个 collapsed cells 共有特征: **kai+vis × 7 balanced via `datasets_yaml`**. Conditioning + action representation 都不是因。

**新 candidate root causes** (待诊断):
1. ~~**Per-DS norm 错配** — E3.6 显式启用 per-DS norm~~ **(2026-06-04 已排除)**: per-DS norm **从未实现**, E3.6 与其他 cell 一样是**单一 norm (kai0_base)**;且 E3.6 是 vis×1 (无 ×7)。所以"per-DS norm 计算出错"不成立。real candidate 见下方 §结论更新。
2. **ConcatDataset 路径本身** — 连 E3.6 (vis×1, 单一 norm, 无 cond) 都崩, 说明问题在 `datasets_yaml`→`_create_concat_torch_dataset` 这条**代码路径**, 而非过采样。对照: 物理预合并单数据集 (Hard Prompt `xvla_exp1_hard_merged`) 同样 kai+vis 混训却**健康** (vis ~0.008)。**最强候选**: ConcatDataset 路径下 inline-eval 的 norm/denorm 不一致, 或多源 batch 把优化推向 predict-mean 局部最优 (需实跑诊断)
3. **InjectDatasetId 与 model 期望不匹配** — datasets_yaml 注入 dataset_id 进 obs, 但 vanilla pi05 (no cond) 不消费, 是否 transform chain 改写了 action target?
4. **Asset_id 解析错** — datasets_yaml 模式下 asset_id 取 first repo, norm_stats load 路径可能错位

**对应真机现象 ("predict zero" pattern)**:
- 3 个 collapse cell 全 horizon MAE ≈ 0.47 ≈ `mean(|gt_abs|)` (normalized)
- 即 model 输出 ≈ 常量 (近 0), 与 GT abs 比对得 mean(|target|)
- 真机表现: 不动 / 震动 (controller 在 setpoint 附近 dither)

**重新归因**:
- ❌ ~~delta 累积错误~~ (这只是次要因素; pi05 delta no-cond 工作正常)
- ✅ **Conditioning gate / sparse-prefix 设计破坏 action expert** — 两 action 表示都一样坍塌
- 真机 "震动" 也是 conditioning 问题, 不是 delta 问题

---

## 5. 早期 Task E 实验背景 (历史经验)

`task_e_master_plan.md` §2.1 早在 2026-04 已提:

> **abs vs delta**: v14 delta (batch=8) loss 曲线好看但 val MAE 劣于 abs 同步数; 且**部署时 delta 需要在每步用当前观测 state 加回去, 存在误差积累**。**所有差异化实验统一用 abs**。

并在 `00_training_history.md`:
> `pi05_stand_box_delta`: abs vs delta 对比, delta 输

→ 本次 Task A delta vs abs 复现了同样结论, 并且首次量化了 long-horizon 的退化幅度 (@50 +200%)。

---

## 6. 结论 (2026-05-25 重写, E3.6 完成后)

1. ⭐⭐⭐ **`datasets_yaml + balanced sampling` 路线 broken**: 3 个用 kai+vis × 7 balanced yaml 训出的 ckpt (E3.6 / Track C abs / Action Cond × delta) 全 collapse 到 MAE@1≈0.47, 跨 horizon 几乎不变。Conditioning + action representation **都不是因**。Single-source pi05 delta (kai0_base 3055 ep only) 是唯一健康样本。
2. **真机震动 ≠ conditioning bug**: 之前归因 "Action Head Cond 实现 broken" 错了。真机震动 = `datasets_yaml` 路线训出的所有 ckpt 通病, 与是否有 conditioning 无关。
3. **不应用部署**: 任何用 `kai+vis × 7 balanced datasets_yaml` 训的 ckpt 都不能部署。
4. **可部署候选**: 只有 single-source pi05 delta (kai0_base, 0.0116) 和 single-source pi05 abs (老 SOTA 0.0085, 7900 ep mixed)。两个都不用 datasets_yaml。
5. **Conditioning eval 细节坑** (仍然有效): 推理时必须传 `obs.dataset_id` 给 conditioning 模型 (否则 `pi0.py:246` 跳过 domain_token)。但这只是 conditioning 模型才需要, 与本节 collapse 不相关。
6. **新候选 root cause** (§4.4): Per-DS norm 错配 / ConcatDataset balanced sampling / InjectDatasetId transform / asset_id 解析。需要单独 debug — 但 **不在本 doc 当前 scope** (本 doc 主线是 conditioning 对比, 现已结论可量化)。

---

## 7. Eval 命令速查 (gf3 cnbj)

### 7.1 No-cond model (pi05 delta) eval
```bash
ssh gf3 'cd /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0 && \
  .venv/bin/python -u ../train_scripts/kai/eval/eval_val_action_mse.py \
      --config pi05_flatten_fold_task_a_base_delta \
      --ckpt /vePFS-North-E/.../pi05_flatten_fold_task_a_base_delta/49999 \
      --val /tmp/vis_v2_val50 \
      --n-sample-frames 50'
# 不需要 --dataset-id (no conditioning model)
```

### 7.2 Conditioning model (Action Cond) eval — **必须** 传 dataset_id 和 correct prompt
```bash
ssh gf3 'cd /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0 && \
  .venv/bin/python -u ../train_scripts/kai/eval/eval_val_action_mse.py \
      --config xvla_actcond_single_stage_joint_delta \
      --ckpt /vePFS-North-E/.../xvla_actcond_single_stage_joint_delta/49999 \
      --val /tmp/vis_v2_val50 \
      --n-sample-frames 50 \
      --dataset-id 1 \
      --prompt "Flatten and fold the cloth."'
# dataset_id=1 = vis (deploy target)
# 不传 --dataset-id 时 pi0.py:246 跳过 domain_token, suffix 结构异常
```

### 7.3 vis_v2_val50 构建 (50 ep symlink, 在 gf3 /tmp)
```bash
ssh gf3 'mkdir -p /tmp/vis_v2_val50/meta && cd /vePFS-North-E/.../Task_A/vis_v2_merged &&
  cp meta/{info.json,tasks.jsonl} /tmp/vis_v2_val50/meta/ &&
  head -50 meta/episodes.jsonl > /tmp/vis_v2_val50/meta/episodes.jsonl &&
  head -50 meta/episodes_stats.jsonl > /tmp/vis_v2_val50/meta/episodes_stats.jsonl &&
  ln -sfn $PWD/data /tmp/vis_v2_val50/data &&
  ln -sfn $PWD/videos /tmp/vis_v2_val50/videos'
```

### 7.4 真机推理代码检查
```bash
# vis 真机 client 必须传 dataset_id, 否则同样 collapse
grep -nr "dataset_id\|obs\\[\"dataset_id\"\\]" client_inference_path/
```
