# X-VLA Conditioning Methods — 三种方式混合数据集训练对照实验

> **作用**: 系统跟踪三种 embodiment conditioning 方式在 **kai (官方) + vis (自建)** 混合数据集上的训练效果与对比, 作为 cross_embodiment_data_reuse_plan.md Track B/C 主线实验的结果汇总。
>
> **背景**: kai (KAI0 base+dagger, 6512 ep) + vis (vis_v2_merged, 895 ep) 跨本体混合时 naive joint norm 失败 (`mixed_pure2_1800_6000` 真机抖动严重)。本实验汇总三种显式 conditioning 方式 (改变模型对 domain 的感知方式) 的真机/val 表现, 用于 paper E3.7 / E3.8 / E3.9 ablation 主线。
>
> **范围**: hard prompt / soft prompt (X-VLA style) / action head conditioning embedding 三种, 同数据 (kai+vis 混合 7407 ep) + 同 init (pi05_base) + 同步数 (50k) + 同 batch (128) + 同 num_workers (节点数 × 16) + 同 lr schedule, 严格控制变量。
>
> **最近更新**: 2026-05-22 (Stage 1 76d44 step 2000 ckpt mu PASS 验证, Soft Prompt 实现端到端正确; Hard Prompt exp1 完成 step 49999)
>
> **关联文档**:
> - `docs/deployment/cross_embodiment_data_reuse_plan.md` — Track B / Track C 完整执行计划 + 假说 H1-H4
> - `docs/deployment/cross_embodiment_data_reuse_plan.md` §6.2 — Soft Prompt (Track B) 设计
> - `docs/deployment/cross_embodiment_data_reuse_plan.md` §6.3 — Action Head Cond Emb (Track C) 设计
> - `docs/training/00_training_history.md` — 全量训练历史索引
> - `docs/training/training_paradigm_comparison.md` — 一阶段 vs 两阶段范式对比

---

## 1. 三种 Conditioning 方式核心区别

| 方式 | 注入模块 | 注入位置 | 实现 | 参数量 | gate 强度 |
|---|---|---|---|---|---|
| **Hard Prompt** | VLM | input (prompt 字符串前缀) | 改 `InjectDefaultPrompt` 走 per-dataset prefix, e.g. `"[KAI] Flatten and fold the cloth."` vs `"[VIS] ..."` | 0 (复用 paligemma tokenizer + 已有 LLM weights) | ⭐⭐ 弱 (信号经 LLM attention 隐式传播) |
| **Soft Prompt** (X-VLA) | VLM | input (learnable prefix token, 32 个) | `soft_prompt_hub = nnx.Embed(num_domains, len × paligemma_width)`, 在 `Pi0.embed_prefix` prepend soft tokens | 65K × num_domains | ⭐⭐⭐ 中 (显式 gate, 在 VLM input 端, 信号经 24 层 paligemma attention) |
| **Action Head Cond Token** (方案 A) | **Action Expert** | input (1 domain token, concat 前置) | `action_head_cond_hub = nnx.Embed(num_domains, action_expert_width)`, 拼到 action expert input 之前 | ~1K × num_domains | ⭐⭐⭐ 中 (action expert input 端 sparse-prefix, 仅 4-8 层 self-attn, **paligemma 不知 domain**) |

> **关键 question (本文档要回答)**: domain conditioning 信号应该注入到 **VLM (perception 端)** 还是 **action expert (motor 端)**?
>
> Track B vs Track C 是 1:1 sparse-prefix 对照实验:
> - 同设计模式 (input concat 一个/几个 learnable token)
> - 不同模块 (VLM vs action expert)
> - 直接量化 "module 选择" 的真机效益

---

## 2. 实验汇总表 (持续填入)

### 2.1 Hard Prompt (Track A baseline)

| Config | Job ID / 路径 | Init | 数据 | Step | num_workers | rate | Best Val MAE@1 (per-source) | 真机平滑度 | 真机成功率 | 备注 |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| `xvla_exp1_hard_prompt_merged_uc` | uc01 local (49999 ckpt) | pi05_base | kai0_base+dagger + vis_v2_merged = **7407 ep** | 49999 / 50k | 64 (initial) / 16 (resume) | 1.8 s/it (initial) / 5 s/it (resume from 42k) | kai0_base: **0.0077** / kai0_dagger: **0.0130** / vis_v2_merged: **0.0082** | TBD | TBD | 训练完成 2026-05-22 05:57 (Beijing). Offline eval 完成 2026-05-22 11:35 (Beijing) on uc01 GPU 0/1/2 并发, 50 ep × 20 query/ep. ckpt 42GB at `/data/shared/ubuntu/local_ckpts/xvla_exp1_hard_prompt_merged_uc/xvla_exp1_hard_prompt_merged_uc/49999/` |

#### 2.1.1 Hard Prompt 三数据集 per-horizon Val MAE (49999 ckpt, 2026-05-22)

| 数据集 | n_ep | MAE@1 | MAE@10 | MAE@25 | MAE@50 |
|---|---:|---:|---:|---:|---:|
| kai0_base (官方) | 50 | **0.0077** ⭐ 最低 | 0.0141 | 0.0216 | 0.0292 |
| kai0_dagger (官方) | 50 | **0.0130** ❌ 最差 (+69% vs base) | 0.0267 | 0.0435 | 0.0597 |
| vis_v2_merged (自建) | 50 | **0.0082** | 0.0188 | 0.0329 | 0.0517 |

> **观察**: dagger 在所有 horizon 都明显差 — 验证 cross_embodiment_data_reuse_plan.md 决策点 1 "dagger 不引入 policy" 是合理的。vis 仅微差于 base (+6.5% @1), hard prompt 对自建数据 transfer OK。
>
> **参考对照** (00_training_history.md):
> - 老 SOTA `mixed_pure2_1800_6000` (vis val): MAE@1=0.0085 — exp1 vis 0.0082 微胜 ✓
> - NEW SOTA `task_a_new_pure_200_new_norm` (200 ep -new): MAE@1=0.0065 — exp1 vis 0.0082 差 +21% (kai 数据"拖累" prior)

> ⚠️ **Val 偏误差注**: val 是 "last 50 ep of each source" — 与训练集近似分布, 实际 generalization 评估会更难。OOD 真机评估是唯一可靠 ground truth。

### 2.2 Soft Prompt (Track B — X-VLA style)

| Config | Job ID | Init | 数据 | Step | num_workers | rate | Best Val MAE@1 (per-source) | 真机平滑度 | 真机成功率 | 备注 |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| **Stage 1**: `xvla_stage1_kai_warmup` | t-20260521154828-76d44 (Beijing 16 H20) | pi05_base + soft_prompt_hub init N(0, 0.02) | kai0_base+dagger (domain_id=0) | **49999 / 50k ✅ done** | 32 | 1.4 s/it | kai0_base: **0.0083** / kai0_dagger: **0.0136** | TBD | TBD | ✅ Step 2000 ckpt mu PASS (d0 absmax=1.15e-3). Offline eval 完成 2026-05-22 11:51 (gf3 1 H20 sequential), dataset_id=0 |
| **Stage 2**: `xvla_stage2_soft_prompt_only_vis` | t-20260522113931-vc7q8 (Beijing 16 H20, auto-submitted) | Stage 1 ckpt 49999 | vis_v2_merged (domain_id=1) | running ~0 / 5k | 32 | TBD | TBD | TBD | TBD | Freeze backbone, only `soft_prompt_hub` trainable. LR 5e-4, batch 128, 16 H20, ETA ~1-2h. Auto-submit triggered 2026-05-22 11:39 |
| **Stage 3**: `xvla_stage3_full_finetune_vis` | (Pending Stage 2) | Stage 2 ckpt | kai + vis 混训 | — / 50k | 32 | TBD | TBD | TBD | TBD | Unfreeze all, joint finetune. ETA ~12h |
| **B3.0**: Track B 最终 ckpt (=Stage 3 49999) | (Pending) | — | — | — | — | — | TBD | TBD | TBD | Track B 最终模型用于 paper ablation E3.7 |

### 2.3 Action Head Conditioning Embedding (Track C — 方案 A 选定)

> **方案 A 选定 (2026-05-22)**: 4 候选 (A Concat / B FiLM / C adaLN / D Cross-Attn) 中选 **A (Concat domain token at action expert input)**。B/C/D 暂搁置。
> **真机评估目标**: vis (B 真机). 训练数据: kai + vis 跨本体混合。
> 设计原理详见 `cross_embodiment_data_reuse_plan.md` §6.3.1。

| Config | Job ID | Init | 数据 | Step | num_workers | rate | Best Val MAE@1 (per-source) | 真机平滑度 | 真机成功率 (B 真机) | 备注 |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| Phase 1.5 代码 | — | — | — | — | — | — | — | — | — | 待实现: `pi0_config.py: action_head_cond_num_domains` + `pi0.py: action_head_cond_hub: nnx.Embed(num_domains, action_expert_width)` + action expert input concat 1 domain token. ~半天 编码 |
| **Smoke test** | (Pending code) | pi05_base | kai+vis mix (1-2 ep × 1k step) | — / 1k | 16 | — | — | — | — | uc01 1-2 GPU, 验证 `mu(action_head_cond_hub)` non-zero (与 Soft Prompt PASS 同 pattern) |
| **C-Stage 1**: kai warmup | (Pending) | pi05_base + action_head_cond_hub init N(0, 0.02) | kai0_base+dagger (domain_id=0) | — / 50k | 32 | — | — | — | — | Shanghai 16 A100 (推荐, 与 Track B Beijing 不抢资源), ETA ~12-15h |
| **C-Stage 2**: vis cond only | (Pending) | C-Stage 1 ckpt | vis_v2_merged (domain_id=1) | — / 5k | 32 | — | — | — | — | Freeze backbone, only `action_head_cond_hub` trainable, LR 5e-4 |
| **C-Stage 3**: joint finetune | (Pending) | C-Stage 2 ckpt | kai + vis 混训 | — / 50k | 32 | — | — | TBD | TBD | Unfreeze all. 终 ckpt → vis 真机评估 |
| **C3.0**: Track C 最终 | (Pending) | — | — | — | — | — | TBD | TBD | TBD | paper ablation E3.8 baseline, 与 Track B B3.0 (Soft Prompt) 1:1 对照 |

### 2.4 双端组合 (Soft Prompt + Action Head Cond) — **2026-05-22 暂搁置**

> 等 E3.7 (Soft only) vs E3.8 (Action Cond only) 单端结果出来再决定是否启用。

| Config | Job ID | Init | 数据 | Step | Best Val MAE@1 | 真机平滑度 | 真机成功率 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---|
| ~~**E3.9**: Dual Cond~~ | — | — | — | — | — | — | — | **2026-05-22 搁置**, 待 Track B/C 单端结果出来 |

---

## 3. 三方式 Head-to-Head 对比 (待填)

> 实验完成后, 此处汇总三方式在**同等条件下** (kai+vis 混合 7407 ep, pi05_base init, 50k step, batch 128) 的真机 + val 表现。

### 3.1 Val MAE 对比

| 方法 | Best Val MAE@1 | @10 | @25 | @50 | vs Hard Prompt baseline |
|---|---:|---:|---:|---:|---|
| Hard Prompt (B3 baseline) | TBD | TBD | TBD | TBD | (baseline) |
| Soft Prompt (Track B) | TBD | TBD | TBD | TBD | ?% better/worse |
| Action Head Cond (Track C) | TBD | TBD | TBD | TBD | ?% better/worse |
| Dual Cond (Soft + Action Head) | TBD | TBD | TBD | TBD | ?% better/worse |

### 3.2 真机表现对比 (B 真机 — vis)

> 所有 Track 终态 ckpt 在 vis (B 真机) 上 evaluate. 训练数据均为 kai+vis 跨本体混合 (7407 ep), 验证 cross-embodiment training 是否提升 B 真机表现.

| 方法 | 抓衣角成功率 (30 ep × 固定场景) | 完整折叠成功率 | 平均执行时长 | 抖动 metric (action diff p99) | OOD 场景成功率 (3 OOD × 30 ep) |
|---|---:|---:|---:|---:|---:|
| Hard Prompt | TBD | TBD | TBD | TBD | TBD |
| Soft Prompt (Track B) | TBD | TBD | TBD | TBD | TBD |
| Action Cond (Track C, 方案 A) | TBD | TBD | TBD | TBD | TBD |
| ~~Dual Cond~~ (2026-05-22 搁置) | — | — | — | — | — |

### 3.3 资源消耗对比

| 方法 | 训练总 GPU-hour | 训练阶段数 | 推理推理时是否 query domain_id? | ckpt 占用 |
|---|---:|---:|---|---:|
| Hard Prompt | ~180 GPU-h (1 stage 50k × 16) | 1 | 否 (信号已 baked into LLM weights) | 42 GB |
| Soft Prompt | ~430 GPU-h (3 stages) | 3 (kai warmup → vis only → joint) | 是 (dataset_id 走 obs) | TBD GB |
| Action Head Cond | ~430 GPU-h (3 stages) | 3 | 是 | TBD GB |
| Dual Cond | ~12h (init from best B/C, finetune) | 1 stage final | 是 | TBD GB |

---

## 4. 关键假说与 paper claim

| Hypothesis | 验证实验 | 当前状态 |
|---|---|---|
| **H_HP-vs-SP**: Soft Prompt 显式 gate 强于 Hard Prompt 隐式信号传播 | Track B Stage 3 vs Hard Prompt baseline | 待 Stage 3 完成 |
| **H_VLM-vs-Action**: Action expert 端 conditioning > VLM 输入端 conditioning (信号离 action 输出更近) | C3.0 (方案 A) vs B3.0 (Soft Prompt) | 待 Track C 完成 |
| ~~**H_Dual-Synergy**~~ 双端 condition 互补 | ~~E3.9~~ 暂搁置 | **2026-05-22 搁置**, 待单端结果再启 |
| **H_PairedShift-Recoverable**: 21° R wrist paired shift (§3.3 in plan) 可由 conditioning 恢复, 不需要 EE-relative | 真机 OOD wrist 评估 (B 真机 vis) | 待真机评估 |
| **H_CrossEmbodiment-Useful**: kai+vis 跨本体训练在 B 真机优于单 vis 训练 | Track C C3.0 vs `task_a_new_pure_200_new_norm` (NEW SOTA 0.0065, vis only) | 待 Track C 完成 |

---

## 5. 时间线 & 决策点

### 已发生 (按时间)

| 日期 | 事件 |
|---|---|
| 2026-05-19~21 | Hard Prompt baseline (Track A) 训练 — `xvla_exp1_hard_prompt_merged_uc` |
| 2026-05-21 | **重大 bug 修复**: RepackTransform + AgilexInputs 两处 dataset_id passthrough 漏掉 → soft_prompt_hub grad=0. Commits `9d2184a` + `df23d5a` |
| 2026-05-21 (晚) | Track B Stage 1 (`xvla_stage1_kai_warmup`) Shanghai 队列 14h queue 后迁移到 Beijing |
| 2026-05-21 (晚) | Track B Stage 1 t-20260521154828-76d44 step 2000 ckpt mu PASS, soft_prompt_hub 真训练 ✅ |
| 2026-05-22 (早) | Hard Prompt exp1 ckpt 49999 save 完成 (Track A 主线) |
| 2026-05-22 | Track C (Action Head Cond) 计划升级为主线; 取消 EE-relative |

### 已决策 (2026-05-22)

- ✅ 议题 3 (Action Head Cond 实现): **选定方案 A** (Concat domain token at action expert input). B/C/D 暂搁置.
- ✅ Track C 训练数据: kai + vis 跨本体混合 (同 Track B)
- ✅ Track C 真机评估: vis (B 真机)
- ✅ E3.9 双端组合: 暂搁置, 待 E3.7/E3.8 单端结果出来

### 待决策

- 议题 8 (真机 eval 规模): 30 ep × 3 OOD vs 60-100 ep, 当前推荐 30 + 30×3 = 120 / ablation
- 议题 9 (E3.5 norm ablation): 是否跑 50k 长训, 看资源决定

---

## 6. 命令速查

### 6.1 提交 Track B Stage 2 (Stage 1 完成后)

> 已 armed Monitor `bxvi2073d` (persistent), auto-submit on Stage 1 ckpt 49999.
> 手动提交参考 `/tmp/stage2_cnbj_16gpu.yaml` + `submit_yaml.py`。

### 6.2 提交 Track C Stage 1 (代码实现后)

```bash
# 推荐 Shanghai 16 A100 (与 Track B Beijing 不抢资源)
# 方案 A: Concat domain token at action expert input
/skill submit-training-job --task xvla_actcond_stage1_kai_warmup
```

### 6.3 Offline eval (val MAE 数字)

```bash
# 在 uc01 上跑
ssh uc01
cd /data/shared/ubuntu/workspace/deepdive_kai0/kai0
.venv/bin/python ../train_scripts/eval/eval_val_action_mse.py \
    --config xvla_exp1_hard_prompt_merged_uc \
    --ckpt /data/shared/ubuntu/local_ckpts/xvla_exp1_hard_prompt_merged_uc/xvla_exp1_hard_prompt_merged_uc/49999 \
    --val <val_path>
```

> 注: exp1 数据集无 pre-split val。需要先从训练集 hold out ~50 ep + 重算 meta。详见 task #58。
