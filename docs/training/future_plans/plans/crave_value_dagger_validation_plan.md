# CRAVE value 模型打标 × 新 dagger(chunk-001) 可用性验证 plan

> **建立**: 2026-07-14
> **目的**: 用 **CRAVE 在线 GRU value 模型**(corr 0.975)给新的 chunk-001 dagger(拼接完整 ep, 387 ep) + 等量 base 数据**打 stage_progress_gt**, 走 AWBC pipeline 训 pi05 → 真机验证:**新格式 dagger 是否可用、是否依然回折冻结**。
> **本轮范围**: plan + **数据打标**(§1-3) → 后续提交 AWBC 训练 + 真机。
> **上游**: [`dagger_launchpoint_trim_freeze_fix_plan.md`](dagger_launchpoint_trim_freeze_fix_plan.md) 证明了旧 dagger 的迟疑边界段是冻结根因。**本实验验证新格式 dagger(拼接完整 ep,无迟疑起手)是否从源头解决了问题。**
> ⚠️ **铁律**: 真机为终判; AWBC 无 val MAE(prompt_from_task=True)。

---

## 0. 定位:为什么用 CRAVE value 模型打标

### 旧路线(已死路)
```
人工 stage label → pi0-AE (value_head) → absolute_advantage → discretize → AWBC
```
pi0-AE 的 PyTorch ckpt 全部 dead(value-only 饿死 backbone), JAX 原版 `adv_est_v1` 已全网删除。**没有可用的 AE。**

### 新路线(本 plan)
```
CRAVE polyline pipeline (DINOv3-base + proprio + BGMM + Viterbi) → stage_progress_gt → advantage → discretize → AWBC
```
CRAVE 的在线 GRU 已证明 corr **0.975**(`gru_polyline_heldout.png`)。**直接用它给 dagger+base 打逐帧 progress label**, 绕开 pi0-AE。

### 与 launchtrim 实验的关系

| | launchtrim(已证不冻) | 本实验 |
|---|---|---|
| dagger 格式 | 旧 chunk-000(短 clip,有迟疑起手) | **新 chunk-001(拼接完整 ep,无迟疑起手)** |
| dagger 处理 | 起爆点前裁(修法) | **不裁**(原始完整 ep) |
| 标签来源 | 复用旧 labeled 源的 task_index | **CRAVE GRU 重新打标** |
| AWBC | pi05 v4 AWBC | 同配方 |
| 判据 | vs 任务②(裁前,已冻) | **vs launchtrim(裁后,不冻)**:新格式能否**不裁也天然不冻** |

**核心假设(H)**: 新 chunk-001 dagger 是拼接完整 ep(83s mean),前端无 idle、无迟疑起手、无边界段污染 → CRAVE 打出的 progress 标签天然干净 → AWBC 训练后**不冻、且成功率 ≥ launchtrim**。

---

## 1. CRAVE value 打标 pipeline

### 1.1 链

```
dagger chunk-001 videos (top_head)
    ↓
DINOv3-base 768D pooled 特征
    ↓
apply 现有 PCA→128D (来自 kai0_base 全量, 复用 gen_final_v3/polyline dump 的 pca_mean/components)
    ↓
⊕ proprio position 14D (标准化, from parquet observation.state)
    ↓
= 142D img⊕proprio joint
    ↓
daw() 双锚 Viterbi → polyline (去阶梯)
    ↓
stage_progress_gt ∈ [0,1] 逐帧 30Hz native
```

**关键复用**:
- **PCA transform**: 来自 `dump_polyline_labels_kai_full.py` 的 `pca_mean` + `pca_components`(kai0_base 3055ep 全量拟合)
- **Milestones**: M=8, 来自同次生成的 BGMM clustering(proven corr 0.947)
- **Proprio 标准化**: 同 SMU/SSD(14D joint position, kai0_base 全量统计)
- **双锚参数**: λ=16·FPS/3 = 160

### 1.2 数据范围

| 来源 | ep 数 | 说明 |
|---|---|---|
| **dagger** | **387** | chunk-001 全量(12 日期,对齐 TOS,含 `dagger_frame_class`) |
| **base** | **387** | 从 kai0_base 随机采样(等量,控变量) |
| **total** | **774** | merged → `self_built/A_v4_chunk001_dagger_crave_labeled` |

### 1.3 需新增的特征提取

chunk-001 dagger 视频**尚未提取 DINOv3 特征**。需要:
1. 同步 chunk-001 视频(目前仅 pq 已同步,视频大部分缺失)
2. 对 387 个 dagger ep 逐帧跑 DINOv3-base → 768D pooled 特征
3. 存为 `lmvla/crave/data/dagger_chunk001_dinov3base/`(仿 kai_dinov3base 格式)

**kai0_base 的 387 ep 不需重新提取**——已有 `data/kai_dinov3base/shard_0.npz`(3055 ep 全量),从中按 episode_index 切片复用。

### 1.4 标签生成脚本

新脚本 `lmvla/crave/experiments/label_dagger_chunk001_crave.py`:
1. 加载现有 PCA + milestones + SMU/SSD(从 `dump_polyline_labels_kai_full.py` 的中间状态 pickle 或重跑 milestone discovery)
2. 对于每个 dagger/base ep:
   - 读取 DINOv3 特征
   - PCA→128D + L2
   - 拼接 proprio 14D(标准化)
   - daw() 双锚 Viterbi → polyline
3. 输出 per-ep npy: `temp/crave_ae_labels/chunk001_dagger/ep*.npy` + `.../base/ep*.npy`
4. sanity: 抽 6 ep 画 polyline vs norm-time 图,验证 corr≥0.85

### 1.5 数据集构建

`train_scripts/kai/data/build_chunk001_dagger_crave_labeled.py`:
- 底座: 387 kai0_base ep(symlink videos) + 387 chunk-001 dagger ep(symlink 或 cp videos)
- 列: 标准 lerobot 7 列 + `stage_progress_gt`(= polyline value, 0→1)
- 重排 episode_index,重算 norm_stats(action_dim=32)
- 落: `kai0/data/Task_A/self_built/A_v4_chunk001_dagger_crave_labeled`

---

## 2. AWBC 训练(打标完成后)

### 2.1 Advantage discretize

```bash
python stage_advantage/discretize_advantage.py \
    --data A_v4_chunk001_dagger_crave_labeled \
    --advantage-source stage_progress_gt \
    --discretion-type binary --top 0.3
```
输出 `task_index` ∈ {0,1} + `tasks.jsonl`(Advantage: positive/negative)。

### 2.2 Config

新建 `pi05_v4_awbc_chunk001_dagger_crave`:
- 克隆 `pi05_v4_awbc_launchtrim`(无 DCT, init pi05_base, warmup1k/peak1.5e-5/50k/bs128/fsdp8/EMA0.9999)
- `repo_id` → `A_v4_chunk001_dagger_crave_labeled`(discretized)
- `prompt_from_task=True`
- ⚠️ norm_stats 必须对 merged 集重算

### 2.3 对照矩阵

| 臂 | dagger 格式 | 标签来源 | 预期冻结 |
|---|---|---|---|
| 任务②(plus_freshdagger) | 旧 chunk-000 短 clip | 旧 AE-C task_index | ❌ 冻(已知) |
| **launchtrim(已证)** | 旧 chunk-000 短 clip,前裁 | 切片复用 task_index | ✅ **不冻** |
| **本实验** | **新 chunk-001 完整 ep,不裁** | **CRAVE-GRU stage_progress_gt** | **? 待验证** |

单变量 vs launchtrim = **(dagger 格式 + 标签来源)**。

---

## 3. 评估(真机为终判)

| Tier | 做法 |
|---|---|
| offline | val MAE(整体+夹爪维单列)sanity; CRAVE label 质量(corr vs T, 单调率) |
| **真机** | **回折过渡是否冻结 >5s** + 夹爪微张开 + 成功率 |

**判据**:
- ✅ **H 成立** = 不冻 + 夹爪修复 → 新格式 dagger(拼接完整 ep)天然干净,可直接用,不需要起爆点前裁
- ⚠️ 缓解未消 = 冻结缩短但仍存在 → 新格式部分解决了问题,但标签或训练还需调
- ❌ 仍冻 → 归因到标签质量(CRAVE-GRU label 在 dagger 域上泛化差)或新格式本身的问题

---

## 4. 落地步骤

### Phase A: 打标(本次)
1. 同步 chunk-001 dagger 视频(TOS → local)
2. 提取 DINOv3-base 特征:新脚本 `extract_dagger_chunk001_d3b.py`
3. 复用 PCA + milestones + SMU/SSD:从 `dump_polyline_labels_kai_full.py` 继承
4. 标签生成:新脚本 `label_dagger_chunk001_crave.py` → `stage_progress_gt` npy
5. 数据集 build: base(387) + dagger(387) → `A_v4_chunk001_dagger_crave_labeled`

### Phase B: AWBC(后续)
6. discretize top-30% binary
7. 注册 config + 8 卡 50k 训练
8. 真机 vs launchtrim(§3 判据)

---

## 5. 风险 / 注意
- **CRAVE-GRU 在 dagger 域泛化**:GRU 训在 kai0_base 域,dagger 域含人类手臂/不同光照 → DINOv3 特征可能偏移 → label 质量可能不如 base。Phase A sanity 图必须核验
- **视频缺失**:chunk-001 视频只同步了 07-01 的部分(249 mp4),其余需从 TOS 拉(可能很大,~10-50GB)
- **DINOv3 提取慢**:387 ep × mean 83s × 30fps = ~960k frames, 单 GPU 提取预计 1-2 小时
- **base 采样**:387 ep 从 3055 ep kai0_base 随机采,需固定 seed 保证可复现
- **标签不 faithful 于旧 AE**:CRAVE-GRU label 和旧 AE-C task_index 不可直接比较(不同 label 源),只能真机判

## 执行记录 (2026-07-14)

**Phase A 实施中**:
| 步骤 | 脚本 | 状态 |
|---|---|---|
| 1. 视频同步 TOS→local | `tosutil cp -r -u` (1161 mp4) | 🔄 进行中 (60%+) |
| 2. DINOv3 特征提取 | `extract_dagger_chunk001_d3b.py` | 🔄 进行中 (~2h ETA) |
| 3. CRAVE labeling | `label_chunk001_dagger_crave.py` | 🔲 待提取完成 |
| 4. 数据集 build | `build_chunk001_dagger_crave_labeled.py` | 🔲 待打标完成 |

**数据路径**:
- 源数据: `vis_dagger/v4/<date>/data/chunk-001/` (387 ep, TOS 对齐)
- DINOv3 特征: `lmvla/crave/data/dagger_chunk001_dinov3base/`
- 标签 npy: `lmvla/crave/temp/crave_ae_labels/chunk001_val/{base,dagger}/`
- 数据集输出: `self_built/A_v4_chunk001_dagger_crave_labeled`

**特征提取配置**: DINOv3-base `encode_pooled`, fp16, ~77fps RTX 5090, ~960k frames total.
**标签管线参数**: PCA 128D (kai0_base 全量), BGMM M=8 milestones, SMU/SSD 14D proprio, λ=160, daw() 双锚 Viterbi→polyline.

---

## 关联
- 冻结修法验证: [`dagger_launchpoint_trim_freeze_fix_plan.md`](dagger_launchpoint_trim_freeze_fix_plan.md) (launchtrim,已证不冻)
- 冻结诊断: [`pi05_v4_awbc_modeB_freeze_diagnosis_plan.md`](pi05_v4_awbc_modeB_freeze_diagnosis_plan.md)
- CRAVE polyline 标签: [`crave_polyline_kai_ae_retrain_plan.md`](crave_polyline_kai_ae_retrain_plan.md)
- CRAVE 最终架构 + 淘汰索引: `lmvla/crave/docs/HISTORY.md` · [[project_crave_history_index]]
- 新 dagger 数据: `vis_dagger/v4/<date>/data/chunk-001/` (387 ep, TOS 对齐)
- kai0_base DINOv3 特征: `lmvla/crave/data/kai_dinov3base/` (shard_0.npz 5G)
- daw() 双锚 Viterbi: `lmvla/crave/experiments/render_kai_online_gru.py` · `dump_polyline_labels_kai_full.py`
