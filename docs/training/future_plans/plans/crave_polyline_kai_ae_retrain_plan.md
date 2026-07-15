# CRAVE 折线(去阶梯)标签 × 重训 KAI0 pi0-AE —— CRAVE-KAI-AE 可行性实验 plan

> **建立**: 2026-07-11
> **目的**: 用 **CRAVE 已跑通的最终 polyline(去阶梯)逐帧 value 标签**当监督,**重训 KAI0 的 pi0-AE(Advantage Estimator, torch value_head)**,产出一个新的 **CRAVE-KAI-AE**;离线看它打出的 `absolute_value / absolute_advantage` 是否比现役人工标签 AE-C(`adv_est_v1/100000`)**更平滑、advantage 更干净**,判断"CRAVE 干净监督 + pi0-AE 部署接口"这条路是否可行。
> **本轮范围**: 只到"训出 CRAVE-KAI-AE + 离线对照",**暂不接 AWBC/真机**(那是 [`crave_ae_distill_plan.md`](crave_ae_distill_plan.md) 的 Phase 3 / AB_plan Tier3)。
> **前身**: [`crave_ae_distill_plan.md`](crave_ae_distill_plan.md) 的 **Phase 1(AE-A/AE-B)**。⚠️ 那两套 `crave_stage_{A,B}` 数据集(7/3 落地)用的是**已淘汰的 DINOv3-H + norm01 + anchor-linear/时间先验 Viterbi 标签**(见 `lmvla/crave/docs/HISTORY.md` §2 A1/C1/F3),**效果不理想且不可复用**。本 plan = 把标签换成 **CRAVE 收口后的 DINOv3-base img⊕proprio → 双锚 Viterbi → polyline** 再重训一次。
> ⚠️ **铁律**: 判据用 **P/N 干净度 / 单调 / advantage 信噪比 / 跨-ep 方差 / 完成态 value≈1**,**不用 circular MAE**(AE 各自拟合自己目标)。
> **状态**: ⚠️ **v1 (value-only, loss_action=0.0) 已确认为 dead-value 塌缩** —— mono 50k + raw 20k 两个 ckpt 均输出常数 ≈0 (std<0.05, Spearman=0.19). **根因已确诊, v2 修法已落地** (§0a). ✅ **v2 MT 版 (multi-task, loss_action=1.0) config+数据就绪, 待提交** (2026-07-14). 新 config: `ADVANTAGE_TORCH_CRAVE_POLY_MT`(raw) / `ADVANTAGE_TORCH_CRAVE_POLY_MONO_MT`(mono). 提交信息见 §8.

---

## 8. ✅ v2 MT 版 数据/配置已就绪 —— 提交信息 (给任务提交 agent)

**一句话**: 8 卡 torchrun 跑 `ADVANTAGE_TORCH_CRAVE_POLY_MT` / `_MONO_MT` (pi0-AE **multi-task: action+value**, 50k), 下面全部就位.

> ⚠️ **别再用旧 config** `ADVANTAGE_TORCH_CRAVE_POLY` / `_POLY_MONO` — 那些是 v1 value-only 版 (loss_action=0), 已确认 dead-value, ckpt 不可用.

### 8.1 提交命令

```bash
cd /vePFS/tim/workspace/deepdive_kai0/kai0

# 主实验: raw polyline × multi-task AE
uv run torchrun --standalone --nproc_per_node=8 \
    scripts/train_pytorch.py ADVANTAGE_TORCH_CRAVE_POLY_MT \
    --exp_name=crave_poly_mt --save_interval 10000

# 消融对照: mono polyline × multi-task AE (raw-vs-mono)
uv run torchrun --standalone --nproc_per_node=8 \
    scripts/train_pytorch.py ADVANTAGE_TORCH_CRAVE_POLY_MONO_MT \
    --exp_name=crave_poly_mono_mt --save_interval 10000
```
- 单节点 8 卡 (gf3 / cnbj 择空闲; 可走 `submit-training-job` skill).
- ⚠️ **PyTorch DDP 路线** (非 JAX `train.py`), model=`AdvantageEstimatorConfig` (`train_pytorch.py` 有断言).
- 两臂**同集群、同参数** (唯一变量 = raw vs mono polyline 标签), 出结果后直接 raw-vs-mono 对照.

### 8.2 就位清单

| 项 | 值 | 状态 |
|---|---|---|
| **config ① POLY_MT** | `ADVANTAGE_TORCH_CRAVE_POLY_MT` (config.py, raw polyline) | ✅ 已注册 |
| **config ② MONO_MT** | `ADVANTAGE_TORCH_CRAVE_POLY_MONO_MT` (config.py, mono polyline) | ✅ 已注册 |
| **数据 ① raw** | `kai0/data/Task_A/self_built/crave_stage_poly` (3055ep, polyline raw labels 0→1, meta/videos symlink→kai0_base) | ✅ build 完成 |
| **数据 ② mono** | `kai0/data/Task_A/self_built/crave_stage_poly_mono` (3055ep, cummax mono labels) | ✅ build 完成 |
| **监督列** | `stage_progress_gt` (polyline value, ∈[0,1], 末值全=1.0) | ✅ |
| **norm_stats** | copy 自同底座 crave_stage_A (state/action 14D) | ✅ |
| **init 权重** | `kai0/checkpoints/pi05_base/pytorch/model.safetensors` (14.4G), strict=False (value head 新增随机) | ✅ |
| **⭐ 损失 (修法)** | `loss_action_weight=1.0 / loss_value_weight=1.0` (**action+value 多任务**) | ✅ |
| **规格** | 50k step · bs144 · num_workers24 · save/keep 10k · 8卡 DDP | ✅ |
| **数据底座** | kai0_base (action==state, 14D 绝对关节位置) | ✅ (辅助任务够用) |

### 8.3 数据集来源 (可复现, 不需要重新 build)

标签数据**完全复用 v1**:
- 生成链: dump_polyline_labels_kai_full.py → write_crave_stage_poly.py (含 `--mono`)
- 全量验证: 3055 ep, 3-cam full coverage (0 missing), poly 全对齐 (0 interp/fallback), video frame==parquet row
- 特征源: lmvla/crave/data/kai_dinov3base/{index.npz, shard_0.npz} (DINOv3-base bank, native 30Hz)
- 标签链: shard → PCA128 ⊕ proprio14 (142D) → BGMM (cov≥0.5) → M=8 milestones → 双锚 Viterbi → daw 去阶梯 polyline
- 验证: polyline-vs-T corr **0.947**, 末值 median **1.000**, mono 0.891

### 8.4 训练后 eval

```bash
cd /vePFS/tim/workspace/deepdive_kai0/kai0

# 跑 advantage 质量评估 (P/N 翻转/zero-cross/Spearman vs GT)
.venv/bin/python stage_advantage/eval_advantage_quality.py \
    --ckpt checkpoints/ADVANTAGE_TORCH_CRAVE_POLY_MT/crave_poly_mt/50000 \
    --config ADVANTAGE_TORCH_CRAVE_POLY_MT \
    --data data/Task_A/self_built/crave_stage_poly \
    --n 15 --horizon 50 --out eval_crave_poly_mt

# mono 臂同理, 换 config/data
.venv/bin/python stage_advantage/eval_advantage_quality.py \
    --ckpt checkpoints/ADVANTAGE_TORCH_CRAVE_POLY_MONO_MT/crave_poly_mono_mt/50000 \
    --config ADVANTAGE_TORCH_CRAVE_POLY_MONO_MT \
    --data data/Task_A/self_built/crave_stage_poly_mono \
    --n 15 --horizon 50 --out eval_crave_poly_mono_mt
```

期望: value 曲线有 0→1 趋势 (backbone 被 action loss 驱动), advantage P/N 翻转远少于 AE-C (256 flips), mono-vs-raw 对照判标签变体.

### 8.5 历史记录

| 版本 | config | 状态 | 结果 |
|---|---|---|---|
| v1 raw | `ADVANTAGE_TORCH_CRAVE_POLY` (loss_action=0.0) | t-20260712084931-2b57p (8×A100, raw 20k 仍在训) | ⚠️ 预期 dead |
| v1 mono | `ADVANTAGE_TORCH_CRAVE_POLY_MONO` (loss_action=0.0) | t-20260713111452-b8qzl (16×A100, 50k done) | ❌ dead-value 确诊 |
| **v2 raw** | **`ADVANTAGE_TORCH_CRAVE_POLY_MT`** (loss_action=1.0) | 🔲 待提交 | — |
| **v2 mono** | **`ADVANTAGE_TORCH_CRAVE_POLY_MONO_MT`** (loss_action=1.0) | 🔲 待提交 | — |

---

## 0a. ⚠️ v1 死值根因确诊 (2026-07-14)

**现象**: `ADVANTAGE_TORCH_CRAVE_POLY_MONO` 训到 50k 后,`absolute_value` 全部塌在 −0.2~+0.1 区间,无 0→1 趋势. 即使传真关节位置 (state≠0), 输出 std 仅 0.041, Spearman 0.23 — **模型学到常数函数**.

**根因**: **value-only 训练 (`loss_action_weight=0.0`) 把 Gemma-2B backbone 饿死了.** 前人已在代码里明确诊断, 见 `config.py:1061–1065`:

```
# 诊断: vis AE multi-task (action+value) — 判定 vis value 弱是"配置"还是"vis感知地板"
# 唯一变量 vs ADVANTAGE_TORCH_VIS_AWBC: loss_action_weight 0.0→1.0 (加回 action flow-matching 辅助任务).
# 假设: value-only 把 backbone 视觉特征饿瘦 → vis value 卡在 loss 0.073 (≈常数基线).
# 对照锚: KAI0 AE (多任务) value-vs-GT corr=0.93 / loss=0.002; 现 vis (value-only) corr=0.67 / loss=0.073.
```

**两条路径的本质差异**:

| | AE-C (能用, corr 0.93) | CRAVE-AE v1 (死, corr ≈0) |
|---|---|---|
| **model config** | JAX `Pi0Config` (多任务) | PyTorch `AdvantageEstimatorConfig` (value-only) |
| **action loss** | ✅ 开 (action flow-matching) | ❌ **关** (`loss_action_weight=0.0`) |
| **backbone 梯度来源** | action head + value head | **只有 value head** |
| **结果** | backbone 被迫学视觉特征 → value head 搭便车 | backbone 无动力学 → 输出 MSE 最优常数 ≈0 |

**修法**: `loss_action_weight=0.0 → 1.0`,加回 action flow-matching 辅助任务.
kai0_base 数据 `action==state` (14D 绝对关节位置) 作为辅助信号够用 —— backbone 只需从图像预测关节位置就能得到丰富梯度,
value head 搭在这个视觉表征上学习进度. 这正是 `ADVANTAGE_TORCH_VIS_AWBC_MT` (line 1067) 的设计.

**新 config**: `ADVANTAGE_TORCH_CRAVE_POLY_MT` (raw polyline) / `ADVANTAGE_TORCH_CRAVE_POLY_MONO_MT` (mono).
旧 v1 config (`POLY`/`POLY_MONO`, loss_action=0.0) 标记为已弃用, 其 ckpt 不可用于 downstream.

**验证方法**: 用 `stage_advantage/eval_advantage_quality.py` 出 advantage 质量图 (P/N 翻转/zero-cross-rate/Spearman vs GT).
预期: backbone 被 action loss 驱动后, value head 能学到 0→1 趋势, advantage 抖动远少于 AE-C (256 flips).

**数据不变**: `crave_stage_poly` + `crave_stage_poly_mono` 数据完全复用 v1,无需重建.

---

## 0. 定位:为什么"用 KAI0 pi0-AE + CRAVE polyline 标签"

**读图结论(CRAVE 最终在线 value 图)**:
- `docs/visualization/online_value/dualhead_vs_kai0ae_6ep.png` —— 左列 value:CRAVE 双头在线(蓝)平滑贴 GT(vsGT 0.95–0.98);**KAI0-AE `absolute_value`(紫)剧烈抖**,ep2341 甚至 KAI0vsGT=0.858 掉队。右列 advantage:**KAI0-AE `absolute_advantage`(紫)剧烈抖动/频繁过零**;CRAVE 头②(红)平滑贴 GT,ep2341 双头vsGT=0.641 而 **KAI0vsGT=−0.111(KAI0-AE 直接失效)**。
- `gru_polyline_heldout.png` —— polyline teacher(去阶梯)+ 因果 GRU,held-out **mean corr 0.975**。这是 CRAVE 已跑通、稳定的进度监督信号。

**→ KAI0-AE 抖 = 它用人工 stage 标签训(段内平台/阶跃,advantage 段内≡0 或噪声主导 → 256 次 P/N 翻转)。假设:把同一 pi0-AE 架构换成 CRAVE 的干净 polyline 标签重训 → 输出的 value/advantage 变平滑、翻转变少。**

**为什么不直接用 CRAVE 在线 GRU、非要蒸进 pi0-AE?**
- 下游 **AWBC pipeline 消费 pi0-AE 的 `absolute_advantage` 列**(→ `discretize_advantage.py` → `task_index∈{0,1}` → advantage-conditioned VLA)。pi0-AE 与部署 VLA 同源(gemma_2b backbone),接口已打通。
- 把 CRAVE 干净标签**蒸进 pi0-AE** = **CRAVE 标签质量 + pi0-AE 部署兼容**,一步到位、无需给 AWBC 另接一个 DINOv3+GRU 旁路。

**诚实天花板(先写在前面)**: pi0-AE 是 **per-frame VLA** → 视觉相同、相位不同的帧消不了歧(和 cluster 一样)。CRAVE polyline 标签本身**已把双锚 Viterbi 的路径消歧烘进标签值**,但蒸进 per-frame AE 时,AE 只见单帧、可能**丢掉部分时序消歧** → value 曲线会比 CRAVE 在线 GRU(带 GRU 时序记忆)略糙。这是**架构地板**,不是 bug;若离线对照显示 CRAVE-KAI-AE 仍明显优于 AE-C,即达标。

---

## 1. ⭐⭐ 头号坑:CRAVE 标签全链 **temp/ 中间产物已被清空**,必须从特征重生成

> 本会话核查(2026-07-11):`lmvla/crave/temp/` 已被清理,**下列中间产物全部不存在**,只剩少量 png/log:
> - ❌ `temp/crave_full_dinov3h/index.npz`(帧索引)
> - ❌ `temp/crave_d3b_pca128/{milestones.npz, feats/}`(PCA128 逐帧特征)
> - ❌ `temp/crave_final_v3.npz`(milestone spec)
> - ❌ `temp/crave_ae_labels/{final,polyline,polyline_mono}/`(标签 npy)
>
> ✅ **仅存活**: `lmvla/crave/data/kai_dinov3base/{index.npz(3.36M帧,E/FR/T schema), shard_0.npz(5G 特征)}`。
> ✅ on-disk `kai0/data/Task_A/self_built/crave_stage_{A,B}/`(**7/3 旧标签,勿复用**)。

**⇒ 结论:必须重跑整条标签生成链(下 §2)。不要以为能直接复用旧 npz/labels/数据集。**

**替代/省算力要点**:
- `fullextract_d3b_pca128.py` 里硬编码读 `temp/crave_full_dinov3h/index.npz`(已删)→ **改指向存活的 `data/kai_dinov3base/index.npz`**(经核验 keys=E/FR/T/n 完全同 schema,可直接替)。
- 该脚本默认**从视频重解码 DINOv3-base**(av+GPU,3055 ep,重);`data/kai_dinov3base/shard_0.npz` 可能已含等价 pooled 特征 → **可选:核验特征等价后直接喂 shard,免重解码**(省数小时 GPU)。⚠️ 未核验前别盲用,两套抽取(multitask base bank vs 逐帧 PCA 链)裁剪/池化可能不同。

---

## 2. 标签重生成链(canonical 脚本,顺序执行;env=`/home/tim/miniconda3/envs/srpo`,`PYTHONPATH=lmvla/crave/src:lmvla/crave/experiments`)

| # | 脚本 | 输入 | 输出 | 坑 |
|---|---|---|---|---|
| 1 | `fullextract_d3b_pca128.py` | kai0_base 视频(或复用 `data/kai_dinov3base`)+ **`data/kai_dinov3base/index.npz`(替已删 dinov3h index)** | `temp/crave_d3b_pca128/{milestones.npz, feats/ep*.npy}` | 硬编码 index 路径要改;GPU 重解码;确认 D3B 权重 `/vePFS/shock/.CACHE/.../dinov3-vitb16-pretrain-lvd1689m` 在 |
| 2 | `gen_final_v3.py` | `crave_d3b_pca128/{milestones.npz, feats/}` | `temp/crave_final_v3**b**.npz` | ⚠️ **存成 `crave_final_v3b.npz`(带 b),下游 `gen_anchored_labels.py` 读 `crave_final_v3.npz`(无 b)→ 必须重命名/软链**;milestone 用 **img⊕proprio joint + BGMM diag 自适应K + median + per-mode cov≥0.5**(HISTORY A6/B1/B4,已是最终版,别退 img-only 否则 M≈3 塌对角线)|
| 3 | `gen_anchored_labels.py` | `crave_final_v3.npz` | `temp/crave_ae_labels/final/ep*.npy`(双锚 Viterbi,**无 smooth·无 norm01**,真实 0→1)| 双锚已 0→1,**不再 per-ep norm01**(norm01 掩盖达顶失败,HISTORY C1)|
| 4 | `gen_polyline_labels.py` | `crave_final_v3.npz` + `crave_d3b_pca128/feats` | `temp/crave_ae_labels/polyline/ep*.npy` + `polyline_mono/ep*.npy` | **本 plan 的主标签**;polyline=代表帧折线(corr **0.957** vs 阶梯 0.944,GT 本身是平滑 ramp)。raw(带真实回落)/ mono(cummax 只进不退)两版 |
| 5 | `write_crave_stage_datasets.py`(**改 `LAB` 指向 polyline**)| kai0_base + polyline 标签 | `kai0/data/Task_A/self_built/crave_stage_poly/`(+ 可选 `_mono`)| 现脚本 `LAB=temp/crave_ae_labels/final`(阶梯)→ **改成 `polyline`**;base=kai0_base,symlink meta/videos,逐 ep 加 `stage_progress_gt`,长度不符则 interp 对齐 |

**sanity(每步必看,别静默过)**:
- 步 2 后:打印 milestone 数 M(期望 ≈10,不是 3)+ median 值单调不倒挂。
- 步 4 后:抽 6~10 held-out ep 画 polyline vs 监督 GT,重现 `gru_polyline_heldout` 级别(per-ep corr ≥0.9);polyline 应平滑无阶梯、无塌缩/别名。
- 步 5 后:抽 ep 核 `len(stage_progress_gt)==parquet 行数`、值域 [0,1]、段内**连续**(非纯阶跃)。

---

## 3. 数据集(`crave_stage_poly`,唯一变量=标签)

- **底座**: `kai0_base`(3055 ep,干净、不带旧 AE 输出列)。meta/ videos/ **symlink 复用 kai0_base**(AE 只读 state+image+`stage_progress_gt`,**不裁视频、无 PTS 问题**;标签是 native 30Hz 逐帧对齐)。
- **列**: 标准 lerobot 列 + `stage_progress_gt`(= polyline value,0→1)。⚠️ 只保留标准列(个别 ep 如 ep104 残留 prediction 列 → HF CastError 崩,write 脚本已过滤)。
- **norm_stats.json**: AE 训练归一化只作用于 state/action,`stage_progress_gt` **不归一**。copy 自 `kai0_base`(或复用 `crave_stage_A/norm_stats.json`,同底座)。
- **变体(可选消融)**: `crave_stage_poly`(raw,含真实回落)vs `crave_stage_poly_mono`(cummax 单调)。默认主实验用 **raw polyline**(corr 最高、GT-ramp 最贴);mono 作对照(回落是否干扰 advantage 符号)。

---

## 4. 训练规格(克隆现有 CRAVE_A config → 换 repo_id)

- **config** 新建 `ADVANTAGE_TORCH_CRAVE_POLY`(克隆 `config.py:1092` 的 `ADVANTAGE_TORCH_CRAVE_A`,唯一改 `repo_id`):
  - `model = AdvantageEstimatorConfig(pi05=True, action_dim=32, action_horizon=50, max_token_len=200, loss_action_weight=0.0, loss_value_weight=1.0)` —— ⚠️ **必须 `AdvantageEstimatorConfig`**(`train_pytorch.py` 有断言);**纯回归 stage-progress 差**(action loss 关)。
  - `repo_id → .../self_built/crave_stage_poly`;`default_prompt="Flatten and fold the cloth."`。
  - `pytorch_weight_path = kai0/checkpoints/pi05_base/pytorch`(**init pi05_base,strict=False**,value head 新增随机)。
  - `advantage_estimator=True`;`num_train_steps=50_000`(AE value 收敛快;AE-C baseline 是 100k,对照取各自收敛 ckpt);`save_interval=10_000`;`batch_size=144`;`num_workers=24`(pyav 3-cam 逐帧解码,少 worker 喂不动 8×A100)。
- **启动**: `uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py ADVANTAGE_TORCH_CRAVE_POLY --exp_name=crave_poly`。
- **集群**: 8 卡(gf3 / cnbj 择空闲;`submit-training-job`)。commit/push config 后提交。

**AE 目标机制(理解为何要平滑标签)**: `advantage_dataset.py` 对每帧**在线采同 ep 随机帧**,回归 `progress = spg[t] − spg[random]`(不需预 join `his_-100` 列)。→ **spg 越平滑,pairwise advantage 越干净**;若给纯阶跃标签,段内 `Δspg≡0`(平台无梯度)→ 训成 dead value(见 memory `ae-stage-label-collapse`:旧 vis 平阶跃 0.25/0.75 → 94% 目标为零 → dead)。**polyline 段内连续 = 天然避开这个塌缩**,正是选它而非阶梯的核心理由。

---

## 5. 评估(离线,非 circular)

用 `stage_advantage/annotation/eval.py`(Flatten-Fold KAI0,指向新 ckpt)在 **held-out 成功 ep** 上打 `absolute_value / absolute_advantage`,与 baseline 对照:

| 指标(痛点) | 期望(CRAVE-KAI-AE vs AE-C) |
|---|---|
| P/N 翻转次数 / NEG 帧占比(advantage 抖) | **明显↓**(AE-C 有 256 次翻转) |
| value 单调率 mono | ↑ |
| `relative_advantage` 信噪比(std / 过零率) | ↓ |
| 跨-ep 同 score 段 value 方差(全局阈值可用前提) | ↓ |
| 完成态 value | 接近 1 |

**对照臂**:
- **AE-C**(现役 baseline)= `adv_est_v1/100000`(人工 stage 标签)。
- (可选)**AE-poly-mono** = mono 标签变体。
- (可选)**旧 AE-A/AE-B** = `crave_stage_A/B`(H+norm01,若想量化"换 base+polyline 的增量")。

**复现对照图**:仿 `dualhead_vs_kai0ae_6ep.png`,把紫色 KAI0-AE 换成新 CRAVE-KAI-AE,看是否从"剧烈抖"变"平滑贴 GT"。

**判据**:
- ✅ **可行** = CRAVE-KAI-AE 的 value/advantage 比 AE-C 明显更平滑、翻转显著少、完成态 value≈1、跨-ep 对齐好 → CRAVE polyline 是好监督,进 [`crave_ae_distill_plan.md`](crave_ae_distill_plan.md) 下游(discretize→AWBC / Phase 2 ranking)。
- ⚠️ **≈ AE-C** = polyline 蒸进 per-frame AE 后优势被架构地板吃掉 → 诊断是**标签**(重看 §2 sanity)还是 **AE 容量/per-frame 天花板**(→ 考虑 Phase 2 ranking g_φ,或直接用 CRAVE 在线 GRU 旁路喂 AWBC)。
- ❌ **更差 / dead value** = 查标签段内是否退化成阶跃 / norm_stats / init strict / 是否误用旧 crave_final_v3b 命名。

---

## 6. 落地步骤
1. **改 2 处硬编码**:`fullextract_d3b_pca128.py` 的 index 路径 → `data/kai_dinov3base/index.npz`;`write_crave_stage_datasets.py` 的 `LAB` → `temp/crave_ae_labels/polyline`,输出名 → `crave_stage_poly`。
2. **重跑标签链** §2 步 1→5(每步 sanity)。⚠️ 步 2 后重命名 `crave_final_v3b.npz → crave_final_v3.npz`。
3. **build** `crave_stage_poly`(+ 可选 `_mono`),copy norm_stats,抽 ep 核对齐。
4. **注册 config** `ADVANTAGE_TORCH_CRAVE_POLY`,commit/push。
5. **8 卡 torchrun 训 50k**。
6. **eval**(§5,非 circular)vs AE-C,落判据。
7. 回填本 plan 结果 + 更新 `crave_ae_distill_plan.md`(标注旧 A/B 被 polyline 版取代)+ master history。

---

## 7. 风险 / 注意(踩坑清单)
1. **temp/ 全清空** → 从 `kai_dinov3base` 重生成全链;别复用任何旧 npz/labels(§1)。
2. **别复用 on-disk `crave_stage_A/B`(7/3)** = H+norm01 旧标签,已淘汰(HISTORY A1/C1/F3)。
3. **index 缺失** → 用 `data/kai_dinov3base/index.npz` 替(schema 已核验)。
4. **`crave_final_v3b.npz` vs `crave_final_v3.npz` 命名不一致** → 步 2/3 间必须重命名。
5. **milestone 必须 img⊕proprio joint**(HISTORY A6):单数据集 img-only 只出 M≈3 → teacher 塌对角线;`gen_final_v3.py` 已是 joint 版,别改回。
6. **双锚不 norm01**(C1):标签本身 0→1,再 norm01 会掩盖达顶失败。
7. **标签段内必须连续**(polyline 天然连续):纯阶跃 → 段内 Δ≡0 → dead value(memory `ae-stage-label-collapse`)。这是**选 polyline 不选阶梯的根本原因**。
8. **AE `his_-100` 是在线同-ep 随机采样**(非固定 −100 帧、非预 join 列):`progress=spg[t]−spg[rand]`;平滑 spg → 干净 advantage。
9. **init strict=False + `AdvantageEstimatorConfig`**:value head 新增随机;`train_pytorch.py` 断言 model 类型,克隆时别退回 `Pi0Config`。
10. **video/PTS 无关**:AE 复用 kai0_base 视频 symlink,不裁不改 → 无 v3 那类 PTS 坑;但要确认 kai0_base 视频与 spg 帧对齐(native 30Hz 逐帧)。
11. **判据非 circular**:AE/GRU 各拟合自己目标,别用"对谁的 MAE";用 P/N 干净度/mono/信噪比/跨-ep 方差/完成态 value。
12. **per-frame 天花板(诚实边界)**:多值歧义消不掉(§0);polyline 已含路径消歧标签,但蒸进 per-frame AE 可能丢部分 → 若不达标,退 CRAVE 在线 GRU 旁路 or Phase 2 ranking。
13. **D3B 权重就位**:`/vePFS/shock/.CACHE/hf_cache/hub/dinov3-vitb16-pretrain-lvd1689m`(local_files_only)提交前确认在。

---

## 关联
- 上游标签方案收口 + 淘汰索引(**动手前必读**): `lmvla/crave/docs/HISTORY.md` §1(canonical)/§2(ledger)· [[project_crave_history_index]]
- 最终架构 / polyline 依据: `lmvla/crave/docs/final_architecture.md`(§2.11 踩坑, §2.12 polyline)
- 在线读出(polyline teacher): `lmvla/crave/docs/online_readout_route.md`(D3)· 图 `docs/visualization/online_value/{gru_polyline_heldout,dualhead_vs_kai0ae_6ep}.png`
- 前身 plan(Phase 1 A/B, 旧标签): [`crave_ae_distill_plan.md`](crave_ae_distill_plan.md)
- canonical 脚本: `lmvla/crave/experiments/{fullextract_d3b_pca128,gen_final_v3,gen_anchored_labels,gen_polyline_labels,write_crave_stage_datasets}.py`
- AE config / dataloader: `kai0/src/openpi/training/config.py`(`ADVANTAGE_TORCH_CRAVE_A` = 克隆源)· `kai0/src/openpi/training/advantage_dataset.py`(his_-100 在线采样)
- baseline AE-C: `kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1`(step 100000)
- 存活特征: `lmvla/crave/data/kai_dinov3base/{index.npz, shard_0.npz}`
