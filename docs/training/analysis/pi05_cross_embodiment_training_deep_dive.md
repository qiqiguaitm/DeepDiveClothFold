# pi0.5 跨本体训练深度研究 — 官方方法 vs 我们的方法

> **目的**: 深度研究 pi0.5 架构跨本体 (cross-embodiment) 数据应该如何训练, 对比官方 (Physical Intelligence / openpi / lerobot) 方法与我们当前的训练方法, 定位我们 `datasets_yaml + balanced sampling` 跨域训练全 collapse 的根因。
> **建立**: 2026-06-03
> **方法**: 联网研究官方 pi0.5 论文/blog/lerobot 文档 + 逐行读我们的 openpi 跨域训练代码 + 实测 norm_stats 差异。
> **关联**:
> - 我们的 collapse 实验: [`../history/experiments/conditioning_vs_action_representation_ablation.md`](../history/experiments/conditioning_vs_action_representation_ablation.md) (3 个 datasets_yaml cell 全 collapse 到 MAE≈0.47)
> - 跨本体战略: [`../../deployment/strategy/cross_embodiment_strategy.md`](../../deployment/strategy/cross_embodiment_strategy.md)

---

## 0. TL;DR — 核心差异 + collapse 真因方向

| 维度 | 官方 pi0.5 | 我们的做法 | 问题 |
|---|---|---|---|
| **跨本体混训定位** | 大规模**预训练** (10k+ 小时, 7+ 机器人, 68 任务) 学通用先验; 单任务靠**post-train 微调** | 在已 finetune 的小数据 (kai 6.5k ep + vis 0.9k ep) 上做 **balanced 混训** | **混训用错了阶段** — 官方混训在预训练, 我们在微调阶段混两个近似机器人 |
| **数据混合** | co-training 多模态 (web/VQA/detection/subtask/action) + 跨本体, curriculum 加权 | datasets_yaml: vis **重复 7 次**与 kai concat (vis 占 ~51%) | 重复实例 = 粗糙过采样, 非真正加权采样器 |
| **归一化** | 每数据集自带 stats; lerobot 每 dataset 独立 norm; quantile (q01/q99) | **单一 norm_stats** (kai0_base 的) 应用到全部数据集 (含 vis) | 🔴 vis 用 kai 的 quantile 归一化 (虽差异小, 但管线本身错) |
| **新机器人接入** | 从 base 微调, lr 2.5e-5, batch 32, 几千 step, train_expert_only 可选 | 从 pi05_base 50k step, batch 128, 混训 | 步数/数据规模与官方"微调"不匹配 |
| **结果** | Libero 97.5% (SOTA) | **3 个 datasets_yaml cell 全 collapse 到 predict-zero** | — |

**核心结论**:
1. **方法论错位**: 官方"跨本体"是**预训练**手段 (海量异构数据学通用 manipulation 先验), 我们把它当**微调**手段 (混两个近似 Agilex 机器人)。两者目标不同。
2. **真因方向** (2026-06-04 更正): collapse **不是** conditioning (no-cond 的 E3.6 也崩)、**不是** delta vs abs (单源 delta 正常)、**不是** norm_stats 大错配 (kai/vis 实测仅差 0.5σ)、**也不是过采样** (E3.6 实为 **vis×1** 也崩,不是 ×7)。真因在 **`datasets_yaml`/`_create_concat_torch_dataset` 这条代码路径本身**。**决定性对照**: 物理预合并单数据集 (Hard Prompt `xvla_exp1_hard_merged`,同样 kai+vis 混训、同样单一 norm) **健康** (vis ~0.008);凡走 datasets_yaml ConcatDataset 的全崩。**修复 = 绕开该路径,改物理预合并** (见 corrected Plan A)。具体崩的机制 (eval norm 不一致 / 多源 batch 推向 predict-mean) **尚未 100% 锁定** (见 §4),但**不影响修复方向**。
3. **务实建议**: 对你们"近似同构机器人 (都是 Agilex Piper) + 小数据"的场景, **官方的正解不是 balanced 混训, 而是 "在一个机器人上训好 → 另一机器人数据微调"** (= 你们 dagger plan 的思路)。绕开有 bug 的 datasets_yaml 路线。

---

## 1. 官方 pi0.5 怎么做跨本体训练

### 1.1 跨本体是「预训练」手段, 不是「微调」手段 ⭐

pi0.5 ([blog](https://www.pi.website/blog/pi05), [paper](https://www.pi.website/download/pi05.pdf)) 的核心创新 = **co-training on heterogeneous data**:

> "diverse training mixture creates a 'curriculum' that enables generalization across physical, visual, and semantic levels simultaneously."

混合的数据源 (官方明确列出):
1. **Multimodal Web Data** (image captioning / VQA / object detection)
2. **Verbal Instructions** (人类逐步指导)
3. **Subtask Commands** (高层语义标签)
4. **Cross-Embodiment Robot Data** (来自 π0 训练集的多机器人数据)
5. **Multi-Environment Data** (静态机器人 × 多家庭)
6. **Mobile Manipulation** (~400h)

π0 基础: **~10,000 小时机器人数据, 7 种机器人配置, 68 任务** 的预训练。

→ **关键认知**: 官方"跨本体混训"的目的是**在预训练阶段学通用 manipulation 先验** (这只勺子怎么抓、衣服怎么叠的物理常识), 让模型 zero/few-shot 泛化到**新环境/新物体**。它**不是**用来"把机器人 A 的数据和机器人 B 的数据混起来提升 B 在某个固定任务上的精度"。

### 1.2 两阶段: pre-train (混) → post-train (单任务微调)

- **Pre-training**: 海量异构 co-training (上述 6 类), 学通用先验。
- **Post-training / fine-tuning**: 在**目标任务/机器人**的数据上微调。lerobot 官方微调命令 ([HF doc](https://huggingface.co/docs/lerobot/en/pi05)):
  ```bash
  lerobot-train --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
    --steps=3000 --batch_size=32 \
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false
  ```
  - **steps 3000, batch 32** (远小于我们的 50k/128) — 微调是"少量 step 适配", 不是从头大训。
  - `train_expert_only=true` 可选: 冻 VLM 只训 action expert + projection (省显存, 防遗忘)。

### 1.3 归一化: 每数据集独立 + quantile

lerobot pi05 归一化 ([HF doc](https://huggingface.co/docs/lerobot/en/pi05)):
- pi05 默认用 **quantile normalization** (需 dataset 含 quantile stats; `augment_dataset_quantile_stats.py` 补)。
- 或显式 `--policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'`。
- **每个 dataset 自带自己的 stats** — lerobot 的设计是 dataset-level normalization, 一个 dataset 一份 stats。
- pi05 默认 **absolute action**; 可选 `use_relative_actions=true` (relative/delta), 需用 `recompute_stats --relative_action true --chunk_size 50 --relative_exclude_joints gripper` 在 relative 空间重算 stats。**注意: relative 时 gripper 排除在 delta 之外** (保持 absolute)。

### 1.4 Knowledge Insulation (π0/π0.5 稳定混训的关键)

NeurIPS 2025 *Knowledge Insulating VLA Models*: 混训 web 数据 + 机器人动作时, 用**梯度隔离**防止 action 流的梯度破坏 VLM 的语义知识 (action expert 与 VLM backbone 之间的梯度通路被"绝缘")。这是官方能稳定混异构数据**而不 collapse** 的核心技术之一。

---

## 2. 我们怎么做的 (datasets_yaml + balanced sampling)

### 2.1 实现 (代码路径)

- **config**: `xvla_actcond_single_stage_joint` 等用 `LerobotAgilexDataConfig(repo_id=kai0_base, datasets_yaml=stage3_kai_vis_joint_balanced.yaml)` (config.py:1239-1242)。
- **balanced yaml** (`xvla/data/stage3_kai_vis_joint_balanced.yaml`): kai0_base + kai0_dagger + **vis_v2_merged 重复 7 次** → vis 占 ~51%。
- **数据加载** (data_loader.py:203-243): 每个 repo_id 建一个 LeRobotDataset → `torch ConcatDataset` 拼接 (无加权采样器, 重复实例=粗过采样)。
- **归一化** (config.py:334-353 + data_loader.py:324-343): asset_id 默认取 `repo_ids[0]` (kai) → **加载单一 norm_stats (kai 的)** → 在 ConcatDataset **之后**统一 Normalize (全部样本用 kai 的 stats)。pi05 → `use_quantile_norm=True` (config.py:352)。

### 2.2 与官方的 4 个关键差异

| # | 官方 | 我们 |
|---|---|---|
| D1 **阶段** | 跨本体在**预训练** | 我们在**微调**阶段混 |
| D2 **采样** | curriculum 加权 / 比例采样 | **重复 7 次实例** + 朴素 ConcatDataset (无 WeightedSampler) |
| D3 **归一化** | **每 dataset 独立** stats | **单一** stats (kai) 套全部 |
| D4 **稳定技术** | Knowledge Insulation 等 | 无 |

---

## 3. Collapse 现象回顾 (来自 ablation 文档)

3 个用 `datasets_yaml + balanced` 的 cell **全 collapse**, 1 个单源正常:

| Cell | conditioning | action | 数据路线 | MAE@1 | 状态 |
|---|---|---|---|---:|---|
| E3.6 | ❌ none | abs | **balanced yaml** | 0.4706 | ❌ collapse |
| Track C abs | ✅ ActionHead | abs | **balanced yaml** | 0.4699 | ❌ collapse |
| Action Cond × delta | ✅ ActionHead | delta | **balanced yaml** | 0.4663 | ❌ collapse |
| pi05 delta | ❌ none | delta | **single-source** kai0_base | **0.0116** | ✅ 正常 |

- MAE 全 horizon ≈ 0.47 ≈ `mean(|gt_abs|)` → 模型输出 ≈ 常量 (predict-zero)。
- **已排除**: conditioning (no-cond E3.6 也崩)、action 表示 (单源 delta 正常)。
- **共因**: 全走 `datasets_yaml`/ConcatDataset 路径 (E3.6 vis×1、Track C/Action-delta vis×7 —— **过采样与否都崩**,所以共因是路径,不是 ×7)。

---

## 4. Collapse 真因分析 (诚实标注: 方向明确, 未 100% 锁定)

### 4.1 已排除的假说

| 假说 | 证据 | 判定 |
|---|---|---|
| conditioning 实现 bug | no-cond E3.6 也 collapse | ❌ 排除 |
| delta vs abs | 单源 delta 正常 (0.0116) | ❌ 排除 |
| **norm_stats 大错配** | **实测 kai vs vis 仅差 0.5σ** (13/14 维在 ±1σ 内, max 1.05σ, std ratio 0.86~1.26; cross_embodiment_strategy §2.1 + 本次复算) | ⚠️ **差异太小, 不足以单独致 collapse** |

> ⚠️ **修正 ablation 文档/agent 的过度归因**: 两次 agent 分析都倾向"单一 norm_stats 套到不同机器人致 collapse", 但**实测 kai/vis norm 差异极小** (都是 Agilex Piper, 关节范围接近)。0.5σ 量级的归一化错位**不可能**让 flow-matching 模型完全坍塌到 predict-zero。**norm 错配是管线缺陷, 但不是 collapse 的充分原因。**

### 4.2 仍存疑的真因候选 (按可疑度)

> ⚠️ **2026-06-04 更正 — 过采样不是因**: 此前把"vis 重复 7 实例"列为最可疑。但 **E3.6 实为 vis×1 也崩** → 过采样被排除。下表已重排,凶手聚焦在**路径本身**。

1. **🔴 `datasets_yaml`/ConcatDataset 代码路径本身** (最可疑): 凡走 `_create_concat_torch_dataset` (data_loader.py:203) 的全崩 (E3.6 vis×1 / Track C vis×7),凡走**物理预合并单源** (Hard Prompt `xvla_exp1_hard_merged`) 的健康。两者都是 kai+vis、都是单一 norm,**唯一差别是数据加载路径**。最可能的子机制: (a) 该路径下 **inline-eval 的 norm/denorm 与训练不一致** → val 0.47 是 eval 假象 (但真机也废,故非纯 eval); (b) 多源 batch + 单一 norm 把优化推向 predict-mean 局部最优。**需实跑诊断** (打印 batch 组成 / loss 分 domain / 对比同 ckpt 用单源 eval 脚本)。
2. **🟡 单一 quantile norm 边界**: pi05 用 quantile (q01/q99 → [-1,1])。vis 用 kai 的 q01/q99 归一化**轻微超界** (实测 -1.19~1.08)。单独看不致命 (Hard Prompt 同样单一 norm 却健康),可能叠加路径问题放大。
3. **🟡 vis_v2_merged 含 hflip mirror** (左右臂对称): mirror 数据 + 跨域混是否产生模式冲突, 待查 (与坑 B 真机双模态相关)。

### 4.3 定位真因的最小实验 (建议)

1. ~~去掉重复 (vis×1)~~ **已被 E3.6 回答**: vis×1 仍 collapse → 过采样不是因。
2. **物理预合并 vs datasets_yaml**: 同数据 (kai+vis) 一个走预合并单源、一个走 datasets_yaml,对比 → 隔离"路径"因素 (= corrected Plan A 的 baseline 已隐含此对照)。
3. **打印 batch 诊断**: 训练时 log 每 batch 的 domain 占比 + 分 domain loss → 看是不是某 domain 梯度异常。
4. **MEAN_STD 替 quantile**: `use_quantile_norm=False` 重训 → 隔离 quantile 边界因素。
- 单变量逐个排除, 即可锁定真因。

---

## 5. 对我们场景的方法论建议 ⭐

### 5.1 核心判断: 我们的场景**不需要**官方式 balanced 混训

- 官方跨本体混训解决的是 **"用海量异构数据学通用先验, 泛化到新环境"**。
- 我们的场景是 **"两个近似同构机器人 (都 Agilex Piper, norm 差 0.5σ) + 固定叠衣任务 + 小数据"** —— 这**不是**跨本体泛化问题, 是**域适应 (domain adaptation)** 问题。
- 用 balanced 混训 (官方预训练手段) 来做小数据域适应 = **杀鸡用牛刀且踩了 datasets_yaml 的坑**。

### 5.2 推荐路线 (绕开 broken 的 datasets_yaml)

| 方案 | 做法 | 适用 |
|---|---|---|
| **A. 单源 + 微调 (推荐)** | 在主数据 (kai 或 vis-clean) 上单源训好 → 用另一机器人/dagger 数据**微调** (单源 dataloader, 不走 datasets_yaml) | = 你们 dagger plan 的思路, 已验证单源管线健康 |
| **B. 预合并数据集** | 离线把 kai+vis 合成**一个** lerobot 数据集 (统一 episode_index + **重算合并 norm_stats**), 当单源训 | 想真混但绕开 datasets_yaml 的多实例/单 norm bug |
| **C. 修 datasets_yaml** | per-dataset norm + WeightedRandomSampler 替代重复实例 (需改 openpi data_loader) | 长期想要真正的加权多域训练 |

> **对当前 dagger 实验 (`dagger_validity_and_finetune_comparison.md`)**: 它走的是**单源合并数据集**路线 (build 一个 smooth800+dagger 合并集, 重算 norm_stats, 当单源训) = **方案 B**, 正好**绕开了 datasets_yaml 的 collapse 坑**。✅ 方向正确。

### 5.3 若将来要真正跨 3 本体 (含 XVLA-Soft-Fold EE6D)

- 官方启示: 跨本体要么 (a) 统一 action 空间 (EE6D 跨臂通用, = Track X 做的) + per-domain soft prompt, 要么 (b) 预合并 + 统一 norm。
- **避免**: 朴素 datasets_yaml 重复实例 + 单 norm (已证 collapse)。

---

## 6. 结论

1. **官方 pi0.5 跨本体 = 预训练手段** (海量异构学通用先验 + Knowledge Insulation 稳定), **微调是少量 step 单任务适配**。我们把混训用在了微调阶段, 且用了 broken 的 datasets_yaml 路线。
2. **collapse 真因方向明确但子机制未锁死**: 不是 conditioning / delta / norm 大错配 / **过采样** (E3.6 vis×1 也崩), 而是 **`datasets_yaml`/ConcatDataset 代码路径本身** (健康对照 = 物理预合并单源 Hard Prompt)。子机制 (eval norm 不一致 / 多源 batch 推向 predict-mean) 待诊断, 但修复方向明确: **绕开该路径,改物理预合并**。
3. **务实建议**: 我们"近似同构 + 小数据 + 固定任务"场景应走**单源训 + 微调** 或 **预合并单源**, 绕开 datasets_yaml。**当前 dagger 实验已是正确路线 (方案 B)**。
4. **不要**在这个场景照搬官方 balanced co-training — 那是为海量异构预训练设计的, 不适合两个近似机器人的小数据域适应。

---

## 附录 — 信息源

| 项 | 来源 |
|---|---|
| pi0.5 blog (co-training/curriculum) | https://www.pi.website/blog/pi05 |
| pi0.5 paper PDF | https://www.pi.website/download/pi05.pdf |
| lerobot pi05 微调/归一化/relative | https://huggingface.co/docs/lerobot/en/pi05 |
| Knowledge Insulation (NeurIPS 2025) | 搜索结果 (官方稳定混训技术) |
| 我们的 collapse 实证 | `conditioning_vs_action_representation_ablation.md` |
| kai/vis norm 实测 0.5σ | `cross_embodiment_strategy.md` §2.1 + 本次复算 |
| 跨域代码路径 | `config.py:334-353`, `data_loader.py:203-243/324-343`, `transforms.py:206-210` |
