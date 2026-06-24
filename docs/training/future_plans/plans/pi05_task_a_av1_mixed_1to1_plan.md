# pi05 × (Task_A + Task_AV1) 混合 1:1 过采样训练 — 执行 plan

> **建立**: 2026-06-12
> **目的**: 把 **Task_A `A_smooth800_dagger_full`**(横向折,1033ep)与 **Task_AV1 `base`**(竖向折 Vertical Fold v1 新 SOP,304ep)**混合**提交 **BJ 集群**(Robot-North-H20),通过**提高 Task_AV1 采样概率**让两者 **frame-level 1:1**,co-train 出 pi05 策略。
> **机制**: 复用已验证的 **`KaiVisMergedDataConfig` + `domain_weights`**(单一 pre-merged 数据集 + 按帧加权采样,**无磁盘复制**;同 `pi05_kaivis_perdsnorm_cond`)。
> **状态**: 📋 **规划定稿(6 项决策已定,§7)** — 仅待用户发话"开始实施"即冻快照→build→注册 config→提交 BJ。**本次仍只更新文档,不实施**。
> ⚠️ **铁律**: 真机为终判;val MAE 先于 train loss。
> **关联**: Task_AV1 单独基线 [`pi05_fold_sop_paradigm_baselines.md`](pi05_fold_sop_paradigm_baselines.md) · 模板 config `pi05_kaivis_perdsnorm_cond`(config.py:1066)。

---

## 0. 为什么这么做(动机 + 风险)

- **动机**: Task_AV1 是新 SOP、数据少(304ep/0.447M 帧);单独训易过拟合/不稳。用 **3.3× 大的 Task_A**(1033ep/1.455M 帧)co-train 提供叠衣通用先验 + 正则;**1:1 过采样**保证新 SOP 信号不被淹没。
- ⚠️ **两者是不同折法**(Task_A=横向 `"Flatten and fold the cloth."` / Task_AV1=竖向 `"... Vertical Fold v1."`)。**靠 domain 条件(task_index→soft-prompt token + per-domain norm)+ 各自 prompt 区分**,部署时选竖向折那一支。**这是 co-train 助力小数据集,不是要一个模型乱折** → ✅ 保留 per-domain prompt(§7-2)。

---

## 1. 数据规模 + 1:1 权重(已实测)

| 域 | 数据集 | ep | **帧数(实测)** |
|---|---|---|---|
| domain 0 | `Task_A/self_built/A_smooth800_dagger_full` | 1033 | **1,455,235** |
| domain 1 | `Task_AV1/base/v2/{2026-06-11-v2, 2026-06-12-v2}` | 304(133+171)| **446,955**(206,502+240,453)|

- **frame-level 1:1 权重**: `domain_weights = (1.0, 1455235/446955) = (1.0, 3.256)`。
  - ⚠️ **不是 ep 比**(1033/304=3.40);AV1 单 ep 略长 → 帧比 3.256。**最终以 norm-build 实数为准**(同模板注释 "norm-build exact")。
- ⚠️ Task_AV1 仍在增长(TOS watchdog 每 10min 同步)→ **冻结一个快照**再算权重(§7-Q1)。

---

## 2. Step 1 — 合并成单一 lerobot(`a_av1_merged`)

复用 **`train_scripts/kai/data/build_kai_vis_merged.py`** 的机制,新建 `build_a_av1_merged.py`(克隆改源):
- **源**:`A_smooth800_dagger_full`(domain 0)+ Task_AV1 两个日期目录(domain 1);先把 AV1 两日期并入 domain 1(同 `build_task_av1_200_split.py` 的日期合并逻辑)。
- **机制**:ONE physical LeRobot dataset,单 chunk;**每帧 `task_index` = domain_id**(kai 路线 0 / av1 路线 1);`observation.state`/`action` 保留,其余列(frame/index/episode/timestamp/task_index)重建;**视频 symlink 到 realpath**(不放大磁盘)。
- **输出**:`kai0/data/Task_A/self_built/a_av1_merged/`。
- ⚠️ **prompt**:merged 的 `meta/tasks.jsonl` 应**保留两条 prompt**(domain0="Flatten and fold the cloth." / domain1="Flatten and fold the cloth. Vertical Fold v1."),让两支可区分(见 §3)。

## 2.5 Step 2 — per-domain norm stats

复用 **`build_kai_vis_norm.py`**(C2)新建 `build_a_av1_norm.py`:在 merged 根下算 **两套 norm**:
- `norm_domain0_taskA`(用 domain0 帧)+ `norm_domain1_av1`(用 domain1 帧)。
- 训练时 `DomainNormalize` 按 `obs.dataset_id` 选对应 norm(各域独立标准化,避免互相污染)。

---

## 3. Step 3 — 注册 config(克隆 `pi05_kaivis_perdsnorm_cond`)

新建 `pi05_task_a_av1_mixed_1to1`(克隆 config.py:1066 `pi05_kaivis_perdsnorm_cond`):
```python
TrainConfig(
    name="pi05_task_a_av1_mixed_1to1",
    data=KaiVisMergedDataConfig(
        repo_id=".../kai0/data/Task_A/self_built/a_av1_merged",
        domain_weights=(1.0, 3.256),   # FRAME-level 1:1: Task_A 1.455M / Task_AV1 0.447M (norm-build 校正)
        # ✅ per-domain prompt 保留(§7-2): domain0 横向 / domain1 竖向
    ),
    weight_loader=CheckpointWeightLoader("mixed_1_clean/params"),  # warm-start, 沿用 flatten-fold pi05 配方
    # cosine warmup 1k / peak 1.5e-5 / decay→1.5e-6; EMA 0.9999; batch 128; fsdp 8
    # num_train_steps: 见 Q2
)
```
- **机制说明**:`KaiVisMergedDataConfig`(config.py:651)→ `domain_weights` 转 `domain_sample_weights={0:1.0, 1:3.256}` → **`_DomainWeightedJAXSampler`**(data_loader.py:454)按帧 `torch.multinomial(replacement=True)` 采样 → 期望每 batch domain0:domain1 ≈ 1:1。
- ⚠️⚠️ **必须走 JAX 训练路径**(`scripts/train.py`)。**PyTorch 路径(`train_pytorch.py`)不实现加权采样**(data_loader.py:534 只对 framework="jax" 生效)→ 用 PyTorch 会退化成 ConcatDataset 均匀采样,**1:1 失效**。

---

## 4. Step 4 — 提交 BJ 集群训练

- **集群**:**Robot-North-H20**(cnbj / BJ),单节点 **8 卡**(数据/代码走 vePFS-cnbj 共享,见 [[reference_uc_cluster_nfs_layout]] 类比)。
- **命令**(JAX):`python scripts/train.py pi05_task_a_av1_mixed_1to1 --exp_name=run1`(经现有 volc/BJ 提交封装;参考 `submit-training-job` skill 选 Robot-North-H20 队列)。
- **步数/超参**:warm-start + co-train,建议 **50k step**(plateau 后融合新 SOP;Q2 可调);batch128 / fsdp8 / EMA0.9999 / save 每 2k。
- **日志**:开训核对 `[domain-weighted sampler] weights={0:1.0,1:3.256} frames=... ` 一行确认权重生效。

---

## 5. 评估(真机为终判)

| Tier | 做法 |
|---|---|
| Tier 1 offline | **两域分别留出 val** 逐 ckpt val MAE:domain1(竖向折,主目标)+ domain0(横向折,看不退化)。⚠️ 喂对应域 prompt + domain_id。 |
| Tier 3 真机 | 部署 best ckpt:**竖向折(domain1 prompt)** 成功率/各 sub-phase 通过/夹持稳定 = 主判据;可选横向折 sanity。 |
| 对照 | ① Task_AV1 **单独基线**(`pi05_fold_sop_paradigm_baselines`,200ep);② 本混合 1:1。→ 判"co-train 是否提升竖向折"。 |

**判据**:混合 1:1 的竖向折真机 **> 单独 AV1 基线** → co-train 有效;若 ≤ → 横向先验干扰,考虑调权重(偏向 AV1)或只在 encoder 共享。

---

## 6. 落地步骤
1. **冻结 Task_AV1 快照**(停 watchdog 或记录 ep 列表)+ 复算精确帧权重。
2. `build_a_av1_merged.py` 合并 → `a_av1_merged`(task_index=domain,视频 symlink)。
3. `build_a_av1_norm.py` 算 per-domain norm。
4. 注册 config `pi05_task_a_av1_mixed_1to1`(§3),commit/push。
5. **BJ(Robot-North-H20)8 卡 JAX 训练**(`scripts/train.py`,50k,核对加权采样日志)。
6. eval:两域 val MAE → 选 ckpt → 真机竖向折 rollout,对照 AV1 单独基线。
7. 回填 results.md + 更新 master history。

---

## 7. 决策定档(✅ 2026-06-12 用户确认)
1. ✅ **Task_AV1 范围 = 冻结当前 304ep 快照**(133+171)。开训前停 watchdog / 记录 ep 列表,后续新数据**不进本实验**;以此快照复算精确帧权重。
2. ✅ **prompt = 保留 per-domain**:domain0 = `"Flatten and fold the cloth."`(横向)/ domain1 = `"Flatten and fold the cloth. Vertical Fold v1."`(竖向)。部署分别选;**train==deploy 一字不差**。
3. ✅ **权重 = 严格 frame-level 1:1** → `domain_weights=(1.0, 3.256)`(Task_A 1.455M / Task_AV1 0.447M;最终以 norm-build 实数微调)。
4. ✅ **步数 = 50k**。
5. ✅ **init = pi05 warm-start `mixed_1_clean/params`**(沿用 flatten-fold pi05 配方)。
6. ✅ **Task_A = 含 dagger 版** `A_smooth800_dagger_full`(1033ep)。

> 全部定档。仅待"开始实施":① 冻 AV1 快照 + 复算权重 → ② `build_a_av1_merged.py` → ③ `build_a_av1_norm.py` → ④ 注册 config `pi05_task_a_av1_mixed_1to1` → ⑤ BJ(Robot-North-H20)8卡 **JAX** 训练。

---

## 关联
- 机制模板:`config.py:651`(`KaiVisMergedDataConfig`)/ `config.py:1066`(`pi05_kaivis_perdsnorm_cond`,domain_weights=(1.0,3.970))
- 加权采样:`kai0/src/openpi/training/data_loader.py:437-474`(`_build_domain_weights` / `_DomainWeightedJAXSampler`,**JAX-only**)
- build 脚本模板:`train_scripts/kai/data/{build_kai_vis_merged.py, build_kai_vis_norm.py, build_task_av1_200_split.py}`
- 数据:`kai0/data/Task_A/self_built/A_smooth800_dagger_full`(1033ep)+ `kai0/data/Task_AV1/base/v2/`(304ep,watchdog 同步)
- Task_AV1 单独基线对照:[`pi05_fold_sop_paradigm_baselines.md`](pi05_fold_sop_paradigm_baselines.md)
