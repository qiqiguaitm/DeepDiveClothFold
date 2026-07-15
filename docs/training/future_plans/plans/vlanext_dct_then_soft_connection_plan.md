# VLANeXt → pi05 两步走实验规划:① 频域 DCT loss(优先) → ② Soft Connection

> **建立**: 2026-06-26
> **目的**: 借鉴 VLANeXt(*Recipes for Building Strong VLA Models*,[arXiv 2602.18532v2](https://arxiv.org/abs/2602.18532),ICML 2026,NTU S-Lab)两条与折叠最相关的配方,提升 pi05 折叠的**动作平滑度与鲁棒性**。**分两步,有先后依赖**:先做低成本高杠杆的**频域 DCT loss**;验证有效后再做高成本的 **Soft Connection** 架构实验。
> **状态**: 规划(Step 1 待 sanity → 提交训练)
> **关键发现**: pi05 的 DCT 频域 loss **早已完整实现、端到端打通,但从未在任何 config 启用** → Step 1 是「开开关 + 跑 + 评测」,不是从零实现。

---

## 0. 为什么是这两条(VLANeXt 依据)

VLANeXt 在统一框架下消融出 12 条 VLA 配方,其中两条对我们折叠任务最相关,且**正交**(一个训练目标、一个架构):

| 配方 | 类型 | VLANeXt 实测增益 | 对折叠的价值 |
|---|---|---|---|
| **#12 频域 DCT loss** | 训练辅助 loss(零结构改动) | LIBERO→LIBERO-plus(扰动)**+5.4 鲁棒**,训练开销≈0 | 把动作 chunk 当时间序列,压高频抖动 → **更平滑、更鲁棒**,直接利于 smooth 折叠 + 夹爪稳定 |
| **#6 Soft Connection** | 架构(VLM↔policy 接线) | **+0.8**(soft 91.8 vs tight 90.0) | pi05 现为 tight(π₀ 系列式);soft 略优但增益小、改造大 |

**结论**:先吃 #12(高杠杆/零成本),#6 作为 gate 后置。

---

## 1. 现状盘点(已逐条核对代码,canonical 路径 `/vePFS/tim/workspace/deepdive_kai0`,`/home/tim/...` 为其软链)

### 1.1 DCT 频域 loss —— **已实现,未启用**
全链路已打通,仅差一个 config 开关:

| 环节 | 位置 | 说明 |
|---|---|---|
| DCT-II helper | `kai0/src/openpi/models/pi0.py:25` `_dct2_last_time_axis` | 沿时间轴做 DCT-II |
| config→model | `pi0.py:170-174` | `self.use_dct_loss` / `self._dct_loss_weight` / `_dct_low_freq_weight` / `_dct_high_freq_weight` |
| loss 计算 | `pi0.py:300-396` `compute_loss` | `use_dct_loss=True` 时返回 `{main_loss, dct_loss, dct_weight}`;低频权重高、高频权重低(`weights = low·(1-f) + high·f`) |
| 总损失聚合 | `scripts/train.py:372-385` | `total = main + dct_weight·dct_loss`,且 log `dct_loss` / `dct_loss_weighted` |
| 配置字段 | `kai0/src/openpi/models/pi0_config.py:44-49` | `use_dct_loss=False`(默认关);`dct_loss_weight=0.1` / `dct_low_freq_weight=1.0` / `dct_high_freq_weight=0.2` —— **与 VLANeXt 配方一致** |
| 动作表示 | `pi0_config.py:26-27` | `action_horizon=50`,`action_dim=32`(DCT 沿 50 步时间轴) |

> **缺口**:`grep use_dct_loss=True src/openpi/training/config.py` → **空**。无任何 TrainConfig 开启 → **从未训练/评测过**。因此首要任务是 sanity + 建启用 config + 跑对照。

### 1.2 VLM↔policy 连接 —— pi05 = **tight**
- `pi0.py:359-362`:`(prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens], …, adarms_cond=[None, adarms_cond])` —— prefix(图像+语言)与 suffix(动作 token)**喂进同一个 PaliGemma.llm 共享注意力** = VLANeXt 定义的 "tight"(π₀ 系列式)。
- `adarms_cond` 已是现成的 **timestep 注入通道**(adaLN 思路),Soft 改造可复用。

### 1.3 训练/评测/文档约定
- 训练入口:`scripts/train.py`(JAX)/ `scripts/train_pytorch.py`;config 注册 `src/openpi/training/config.py` 的 `_CONFIGS`(line~1020)+ `get_config`。
- 折叠基线:SFT warm-start `task_a_new_smooth_800_step49999`(MAE@1=0.0089);折叠 config 家族 `pi05_flatten_fold*`(config.py:2095 区)。
- 提交训练:走 `submit-training-job`(cnsh robot-task 闲时 8×A100);本地 2 卡仅 sanity/评估(见 memory「No local 2-GPU long training」)。
- 评测:Tier1 离线 MAE(`eval_val_action_mse.py`)+ Tier3 sim01 rollout。
- 产出约定:per-exp `results.md` + 更新 master `docs/training/history/experiments/00_training_history.md` + commit/push。

---

## 2. Step 1 · 频域 DCT loss 实验(**优先**)

**假设**:在不损失成功率的前提下,DCT 频域 loss 让折叠动作更平滑、对扰动更鲁棒(对应 VLANeXt 的 +5.4 鲁棒)。

### 2.1 Sanity(本地 2 卡,先做)
1. **`_dct2_last_time_axis` 正确性**:对常数序列只有 freq-0 非零;对正弦序列能量集中在对应频点;与 `scipy.fft.dct(type=2)` 数值对齐。
2. **一步训练**:取折叠基线 config 临时设 `use_dct_loss=True`,跑 1 step,确认 `dct_loss` 数值量级合理、`total` 含 DCT 贡献、梯度非 NaN、log 出现 `dct_loss_weighted`。

### 2.2 建启用 config(单变量)
- 新建 `pi05_flatten_fold_dct` = 折叠 SFT 基线 config **逐字段一致**,**仅** `use_dct_loss=True`(weight/freq 用默认 0.1 / 1.0 / 0.2)。同 warm-start、同数据、同步数、同 seed/val。
- 注册进 `_CONFIGS`。

### 2.3 (可选)轻量扫参
- `dct_loss_weight ∈ {0.05, 0.1, 0.2}`,`dct_high_freq_weight ∈ {0.2, 0.4}`;先只跑论文默认 `0.1 / 0.2`,持平再扫。

### 2.4 训练
- `submit-training-job` → cnsh robot-task 闲时 8×A100(Preemptible + self-heal),步数与基线一致。

### 2.5 评测(决定性 = sim01)
| Tier | 指标 | 作用 |
|---|---|---|
| Tier1 | val MAE@1/10/25/50 vs 基线 | sanity（MAE 对平滑收益不敏感，仅看不退化） |
| **Tier3** | sim01 rollout:**成功率** + **动作平滑度(jerk / 速度方差 / 频谱高频能量)** + 夹爪稳定性 | **决定性**;DCT 主打平滑,必须显式量平滑指标 |
- 同口径对照 **no-DCT 基线**(同 warm-start/同步数)。

### 2.6 判据
| 结果 | 结论 |
|---|---|
| 成功率 ≥ 基线 **且** 更平滑 | ✅ 采纳,把 `use_dct_loss=True` 设为折叠默认 |
| 成功率持平、平滑更好 | 记录;按是否在意平滑决定是否默认开 |
| 成功率下降 | 查 weight 过大(高频被过度压制→动作糊)/`freq_split`,回退或降 weight |

### 2.7 产出
- `results.md` + 更新 `00_training_history.md` + commit/push。

---

## 3. Step 2 · Soft Connection 实验(**gate:Step 1 完成后再启动**)

**仅当** Step 1 验证频域有效、且仍想进一步榨性能时启动。VLANeXt 实测 soft 仅 +0.8,而这是**真模型手术**,优先级低。

### 3.1 改造点
- 现状 tight:`pi0.py:359-362` 共享注意力。
- 改为 soft:在 suffix(动作)侧插一组**可学 query buffer**,policy 通过**逐层 cross-attention** 读 PaliGemma 各层特征(而非直接共享 hidden state),timestep 用 **adaLN**(复用现成 `adarms_cond`)。改 `embed_suffix`(`pi0.py:229-297`)+ llm 前向。

### 3.2 风险与兜底
- 改注意力/插新模块可能破坏 PaliGemma 预训练对齐 → **保留 tight 作 baseline**;新增参数小步、warm-start;必要时只在末若干层加 cross-attn。
- 预期增益小:把它当探索项,不阻塞主线。

### 3.3 评测
- 同 Step 1 口径(Tier1 + Tier3),vs **tight baseline**。若叠加 Step 1 的 DCT,则与「tight+DCT」对照以隔离架构变量。

---

## 4. 机器 / 资源
- **Sanity / 评估**:本地 2 卡(不跑长训练)。
- **正式训练**:cnsh robot-task 闲时 8×A100(`submit-training-job`)。
- **评测**:sim01。
- DCT 为纯训练 loss,无额外显存/数据需求;Soft 增少量参数。

---

## 5. 风险与兜底(汇总)
| 风险 | 兜底 |
|---|---|
| DCT 已实现但从未跑,可能有隐藏 bug | Step 2.1 sanity 先验证 helper 正确性 + 一步训练 |
| weight 过大 → 高频被压、动作糊/反应迟钝 | 从默认 0.1 起;下降则降 weight / 调 `freq_split` |
| DCT 在折叠真机/ sim 的收益不及 LIBERO(刚体 vs 可变形) | 以 Tier3 sim01 为准,offline MAE 仅 sanity |
| Soft 改坏 backbone 对齐 | 保 tight baseline,小步解冻末层,gate 在 Step 1 之后 |

---

## 6. 链接
- 论文:[VLANeXt arXiv 2602.18532](https://arxiv.org/abs/2602.18532) · 代码 `github.com/DravenALG/VLANeXt` · 设计空间 `DESIGN_SPACE.md`
- 代码锚点:`kai0/src/openpi/models/pi0.py:25,170-174,300-396` · `pi0_config.py:44-49,26-27` · `scripts/train.py:372-385`
- 相关 plan:[AWBC×milestone-value AB_plan](cross_episode_recurrence_value/awbc_milestone_value_AB_plan.md) · [CRAVE-RPO](cross_episode_recurrence_value/crave_rpo_minimal_validation_plan.md)
