# Task A new_pure_200 训练结果 ⭐⭐ 全任务 NEW SOTA

> **结论先行**: js02 (单机 8 GPU) 上 `task_a_new_pure_200_new_norm` 从 step 22000 resume 到 49999, final **MAE@1 = 0.0065** ⭐, 全任务新 SOTA, 比之前 best (`task_a_new_pure2_1800_6000_new_norm` 0.0085, uc 集群 7900 ep) 低 **24%**。**200 ep 精选 + mixed_1_clean init** 推翻"数据规模 > 数据质量"假设, 数据质量 (`-new` 限定 + mirror 增强) 占主导。

## 1. 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_pure_200_js` |
| Model | pi05 (`Pi0Config(pi05=True)`) |
| **Init** | `/mnt/data/tim/kai0_ckpts/Task_A/mixed_1_clean/params` (MA-merged Task_A warmed-up) |
| **Dataset** | `/mnt/data/tim/data/Task_A/self_built/A_new_pure_200` (200 ep 精选, JuiceFS:visincept) |
| - 来源 | `-new` 日期限定 + hflip mirror 增强, 仅高质量子集 |
| Val | `/mnt/data/tim/data/Task_A/self_built/A_new_pure_200_val` |
| Prompt | "Flatten and fold the cloth." |
| `use_delta_joint_actions` | False |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50k, decay_lr=1.5e-6 |
| EMA decay | 0.9999 |
| Steps | 50,000 |
| Batch | 120, fsdp_devices=8 (单机全 FSDP `[1,8]`) |
| Save | every 2,000 step, keep_period=2,000, max_to_keep=1 |
| inline_eval | every 2 saves (= 每 4k 步), 200 frames |
| Seed | 42 |
| Server | **js02** (A800-SXM4-80GB × 8) |
| 启动 | **从 step 22000 resume** (原 ckpt 在 `/mnt/data/tim/checkpoints/.../22000`), 续训到 49999 |
| WandB | offline (`--no-wandb-enabled`) |
| 训练时长 | resume window 16h11m (2026-05-14 08:08 → 2026-05-15 00:18 CST) |

### 1.1 关键差异 (vs 老 SOTA `pure2_1800_6000`)

| 维度 | pure_200 (本实验) | pure2_1800_6000 (老 SOTA) |
|---|---|---|
| **数据规模** | **200 ep** (1/40 of SOTA) | 7900 ep |
| **Init** | mixed_1_clean (Task_A 微调过) | pi05_base (原始) |
| 集群规模 | 单机 8 GPU | 24 GPU (uc01+02+03) |
| Batch | 120 | 120 |
| Final MAE@1 | **0.0065** ⭐ | 0.0085 |
| Gap | — | **+30.8% (worse)** |

## 2. 完整 inline-eval MAE@{1,10,25,50} 曲线

> 因 resume from step 22000, 仅记录 step 24000 起的 eval。

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 24000 | 0.0071 | 0.0086 | 0.0095 | 0.0115 | (resume baseline) |
| 28000 | 0.0069 | 0.0082 | 0.0088 | 0.0102 | -2.8% |
| 32000 | 0.0067 | 0.0078 | 0.0084 | 0.0094 | -2.9% |
| 36000 | 0.0066 | 0.0076 | 0.0080 | 0.0088 | -1.5% |
| 40000 | 0.0066 | 0.0074 | 0.0078 | 0.0085 | 0.0% |
| 44000 | 0.0065 | 0.0073 | 0.0076 | 0.0081 | -1.5% |
| 48000 | 0.0065 | 0.0072 | 0.0075 | 0.0080 | 0.0% |
| **49999** | **0.0065** | **0.0072** | **0.0075** | **0.0079** | 0.0% |

**Best**: step 49999, MAE@1 = **0.0065** ⭐ (44k/48k 也是 0.0065, @50 在 49999 最低 0.0079)

## 3. 训练动力学

- resume @ 22k 的起点未直接 eval, 但 step 24k 已达 0.0071 — mixed_1_clean init 下经过 24k step (相当于从 mixed_1 起步训了 24k) 已显著优于 pi05_base init 同步数
- 收敛极快: step 24k → 36k 在 12k 步内从 0.0071 跌到 0.0066 (-7%), 但 36k 后进入 long plateau
- @1 在 44k-49999 持续 0.0065 (4 个连续 eval 持平), 训练已饱和
- @50 (long-horizon) 持续小幅下降到 step 49999 (0.0115 → 0.0079, **-31%**) — 长 horizon planner 后期仍在精修, 与 pure2_1800_6000 趋势一致
- 全程 0 NaN, 无 div, eval 时长 754-962s (均值 768s)

## 4. SOTA 关键洞察

1. **数据质量 > 数据规模**: 200 ep (`-new` 限定 + mirror) 比 7900 ep 大杂烩 MAE 低 24%。**关键变量**: pure_200 是高度精选的 `-new` 日期数据 + hflip mirror 双对增强, 比 mix b6000+p1200 大量 `kai0_base / kai0_advantage` 旧数据干净得多。
2. **mixed_1_clean init 上限被低估**: 之前 pure2_1800_6000 用 pi05_base init 得到 0.0085 (老 SOTA), 推断 "干净起点天花板更高"。但 pure_200 用 mixed_1_clean (Task_A warmed-up) 反而打到 0.0065 — 说明在**高质量小数据**场景下, Task_A 适配过的 init 反而是更好的起点 (节省 24k step 的"重新学 Task_A"开销)。
3. **小数据 + 长训** 配合 EMA decay 0.9999 不发生 overfit: 200 ep × 50k step = ~250 epoch, 仍单调下降, 没看到 val MAE 反弹。
4. **resume 不影响收敛趋势**: 22k resume 起点没破坏 cosine LR schedule (continued from same step counter), 后续 28k step 平滑收敛。
5. **@1 vs @50 差异启示**: @1=0.0065 (单步精度), @50=0.0079 (50 步长 horizon) — gap 仅 22%, 明显小于其他实验 (pure2_1800_6000 @1=0.0085 @50=0.0337, gap 296%)。说明该模型不仅单步准, 长 horizon planner 也极稳。

## 5. 最佳 ckpt 位置

```
js02:/mnt/data/tim/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_new_pure_200_new_norm/49999/
```

⚠️ JuiceFS:visincept 共享路径 — js01-js04 任一节点都能直接读, 不需 SSH 跨节点拷贝。

**完整 ckpt 列表** (每 2k 步): 22000, 24000, 26000, ..., 48000, 49999  
**附属文件**: 同目录 `norm_stats.json`, `wandb_id.txt`

## 6. 后续计划

- **对照实验**: 改用 pi05_base init (而非 mixed_1_clean) 跑同样的 pure_200 数据, 验证"高质量小数据下哪个 init 更优"。新 exp `task_a_pure200_new_norm_base_pi0.5` 已在 js04 单卡启动。✅ **2026-05-26 完成, 见 §7**
- **deploy 到 sim01**: ckpt 49999 应优先 pack + 上 sim01 测试 (老 SOTA pure2_1800_6000 已上, 这个有望大幅优化 chunk planner)。
- **解析为啥 -24%**: 是数据质量, 还是数据量本身就够? 跑 pure_200 / pure_400 / pure_800 sweep 即可 disentangle。

---

## 7. 2026-05-26 更新 — pi05_base init 对照实验完成 + 关键修正

> 🔥 **本节推翻 §1-5 中的 3 个核心假说**. 新 dual eval 数据揭示: init **不是主导因素**, pure_200 的 0.0065 是**严重过拟合 to 2 dates** 的产物, vis_v2_full 在 generalization 上反而**比 pure_200 好**.

### 7.1 pi05_base init 实验完成

| 项 | 值 |
|---|---|
| Ckpt | `/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/task_a_pure200_base_pi05_step49999` |
| 训练 | gf3 (1 H20 smoke), batch=16, fsdp_devices=1 |
| 数据 | 同 A_new_pure_200 (100 unique × 2 mirror, 2 dates 05-08/09) |
| Init | **pi05_base 冷启** (vs NEW SOTA 的 mixed_1_clean warm init) |
| Steps | 50000 (full schedule) |

### 7.2 Dual eval 结果

| Ckpt | val 数据集 | MAE@1 | MAE@10 | MAE@25 | MAE@50 |
|---|---|---:|---:|---:|---:|
| **pure_200 + mixed_1_clean** (NEW SOTA) | native A_new_pure_200_val | **0.0065** ⭐ | 0.0072 | 0.0075 | 0.0079 |
| **pure_200 + pi05_base** (新对照) | native A_new_pure_200_val | **0.0065** ⭐ | **0.0074** | **0.0078** | **0.0087** |
| **pure_200 + pi05_base** (cross-val) | vis_v2_val50 (来自 04-23/24) | **0.0207** ❌ | 0.0507 | 0.0900 | 0.1348 |
| **vis_v2_full + pi05_base** (1406 ep, 16 dates) | vis_v2_val50 (同上) | **0.0131** ✓ | 0.0386 | 0.0714 | 0.1138 |

### 7.3 🔥 推翻的 3 个假说

#### 推翻 1: ❌ ~~"mixed_1_clean init 在小数据下更优"~~ (§1, §5)

**新数据**: pure_200 + pi05_base 在 native val 上 **0.0065 == pure_200 + mixed_1_clean 0.0065**.

→ **Init 不影响 final native MAE**. 之前 §5 关于 init 的所有讨论都需修正. 关键在数据 (pure_200 精选 + mirror) 而非 init.

#### 推翻 2: ❌ ~~"pure_200 是真正的 SOTA"~~

**新数据**: 同 ckpt 在 cross-val (vis_v2_val50) 上 MAE@1 从 0.0065 → **0.0207 (3.2× 退化)**.

→ **0.0065 是"过拟合 to 2 dates" 的产物**, 不代表真泛化能力. 在 vis_v2_full 训练集分布的早期 dates (04-23/24) 上, pure_200 model 性能差.

#### 推翻 3: ❌ ~~"vis_v2_full 数据量增加反而效果差"~~

**新数据**: 在同 val (vis_v2_val50) 上对比:
- vis_v2_full + pi05_base: **0.0131**
- pure_200 + pi05_base: **0.0207** (差 36%)

→ vis_v2_full 在跨日期 val 上**比 pure_200 好 36%**. 1406 ep 跨 16 dates 训练让 model 学到 broader generalization prior. 训练没坏, 反而是 pure_200 在 cross 表现不好.

### 7.4 真正的解释 — Val distribution 决定 MAE 数量级

| Train | Native val (in-distribution) | Cross val (vis_v2_val50, 04-23/24) |
|---|---|---|
| pure_200 (05-08/09, 2 dates) | 0.0065 ✓ (val ∈ train dates) | 0.0207 ❌ (val ∉ train dates, OOD) |
| vis_v2_full (04-23~05-22, 16 dates) | n/a (no test) | **0.0131 ✓ (val ∈ train dates)** |

**vis_v2_val50 (04-23/24) ∈ vis_v2_full 训练分布, ∉ pure_200 训练分布**.
- vis_v2_full 在 vis_v2_val50 上的 0.0131 是 in-distribution 精度
- pure_200 在 vis_v2_val50 上的 0.0207 是 OOD 精度

之前的 NEW SOTA 0.0065 vs vis_v2_full 0.0131 是**不公平比较** — 两 model 各自在自己分布的核心做 eval.

### 7.5 真机表现 vs Offline MAE 的真正关系

```
真机部署 in conditions ≈ 05-08/09 dates 风格 (近期采集):
  → pure_200 表现 ≈ 0.0065 (优秀)
  → vis_v2_full 表现 ≈ 0.0131 (一般, 因为是 16 dates 的"平均策略")
  → pure_200 在此特定场景胜出

真机部署 in conditions varied / OOD:
  → pure_200 表现 ≈ 0.0207+ (3× 退化)
  → vis_v2_full 表现 ≈ 0.0131 (稳定)
  → vis_v2_full 在 generalization 上胜出
```

### 7.6 vis_v2_full 真机不好的修正解释

之前归因 (错): "训练没问题, init 弱 / mirror 缺失 / 协议漂移"

现在归因 (对): **vis_v2_full 训练没问题, generalization 实际比 pure_200 好**. 真机不好可能因为:

1. **真机评估场景与训练 dates 不匹配**: vis_v2_full 训练分布跨 04-23~05-22, 真机可能在更新的 dates 风格下评估 (e.g., 05-26) → OOD
2. **vis_v2_full 学到"平均策略", 不锐利**: 跨 16 dates 的"折中" → 在任何具体场景上都不是最锐利, 但比 pure_200 在 OOD 时稳定
3. **Real-world vs offline gap**: offline val 是录制好的数据, 真机有 lighting/setup variations 等 noise

---

## 8. 2026-05-31 更新 — PyTorch 原生训练对照 (隔离 JAX vs PyTorch 框架变量)

> 实验 `A_mirror200_pi05_pytorch`: 同 pure_200 数据 + **同 pi05_base init** + 同 50k schedule, 唯一变量 = **训练框架 PyTorch DDP (`scripts/train_pytorch.py`) vs JAX (§7.1)**。设计见 [`../../future_plans/plans/A_mirror200_pi05_pytorch.md`](../../future_plans/plans/A_mirror200_pi05_pytorch.md)。

### 8.1 配置

| 项 | 值 |
|---|---|
| Config | `pi05_pytorch_a_new_pure_200` |
| 框架 | **PyTorch DDP** (vs §7 的 JAX) ⭐ 唯一变量 |
| Init | **pi05_base** (与 §7.1 JAX pi05_base 对照组**完全一致**) |
| Dataset / Val | `A_new_pure_200` / `A_new_pure_200_val` (与 §7 同) |
| Steps / batch | 50000 / 128, peak_lr 1.5e-5 |
| 集群 | cnsh robot-task 8× A100, 训练耗时 **~68.5h** (DDP + dataloader 慢) |
| Ckpt 目录 | `kai0/checkpoints/pi05_pytorch_a_new_pure_200/A_mirror200_pi05_pytorch/{step}/` (含 model.safetensors + optimizer.pt) |

### 8.2 完整 MAE@{1,10,25,50} 曲线 (native A_new_pure_200_val, 20 ep × 200 frames, 全 26 个 ckpt)

> 🔴 **2026-05-31 作废重测**: 本表初版的 26-step 曲线 (@50 末段 ~0.0646) 是**用错 prompt 评的** — `eval_val_action_mse.py --prompt` 默认 `"stand up the fallen box"` (Task E), batch eval 未覆盖 → 叠衣模型被喂叠箱指令, MAE 全程虚高 ~1.8×。**正确 prompt `"Flatten and fold the cloth."` 重测见 §8.4.4 (plain 50k @50=0.0350)**。完整正确曲线重测中 (下表待填)。

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| … | (正确-prompt 重测中) | | | |
| **50000** (correct prompt) | **0.0100** | 0.0174 | 0.0258 | **0.0350** |

**Best ckpt 位置**: `kai0/checkpoints/pi05_pytorch_a_new_pure_200/A_mirror200_pi05_pytorch/{step}/` (含 model.safetensors + optimizer.pt, 每 2k 步, 2000…50000)。正确-prompt best step 待完整曲线确认 (plateau ~38k 后)。

### 8.3 🔥 关键发现 — PyTorch 比 JAX 显著差 (同 init 纯框架对照)

| 框架 (同 pi05_base init, 同数据, 同 val) | MAE@1 | @50 |
|---|---:|---:|
| **JAX** (§7.1, native val) | **0.0065** | 0.0087 |
| **PyTorch** (本节, native val) | **0.0121** | 0.0646 |
| **Δ (PyTorch / JAX)** | **+86% (1.86×)** | +643% |

> 🔴 **本表 (8.3) 的 PyTorch 0.0121/0.0646 是训练 inline-eval 数字, 与 JAX 行 (0.0065/0.0087, 训练记录) 协议不一致 — 此对比已被 §8.4 推翻为 "苹果比橘子" (同一 50k ckpt 独立 eval 实为 0.0100/0.0350)。真实框架 gap 见 §8.4.4 三方同协议对比 (@50 真差 4.1× 而非 6.4×)。本表保留作错误轨迹。**

> ⚠️ §7 已证 init (pi05_base vs mixed_1_clean) 在 native val 上对 final MAE 无影响 (都 0.0065)，故本对比是**纯框架变量** — PyTorch 路径在 pure_200 上 offline MAE 显著劣于 JAX：
> - **@1 差 86%** (0.0121 vs 0.0065)，**@50 差 6.4×** (0.0646 vs 0.0087) — 长 horizon 退化尤其严重，PyTorch 版 @1→@50 gap 达 434% (0.0121→0.0646) vs JAX 仅 34% (0.0065→0.0087)。
> - **可能原因** (待查): (a) PyTorch flow-matching sampler / num_steps 与 JAX 实现差异; (b) DDP gradient sync / mixed-precision 数值差异; (c) PyTorch 侧 EMA 未启用或实现不同 (JAX ema_decay=0.9999); (d) preprocessing (image norm / action norm) 在 `models_pytorch/preprocessing_pytorch.py` 与 JAX transform 不一致。
> - **结论**: **PyTorch 原生路径当前尚不能等价 reproduce JAX 的 pure_200 表现** — 在用 PyTorch 跑生产/其他实验前需先排查上述 gap。真机对照 (PyTorch ckpt vs JAX ckpt 同场景) 可进一步确认是 offline-only 伪差还是真实退化。

> 评估方法: JAX `eval_val_action_mse.py` 经 `create_trained_policy` 自动检测 `model.safetensors` → 走 `load_pytorch` 分支, 同一脚本即可评 PyTorch ckpt (无需单独 PyTorch eval 工具)。

### 8.4 🔬 根因排查 — **EMA 假说被实测证伪 (2026-05-31)**

> ⚠️ **重大修正**: 本节早期版本断言 "EMA 缺失是 PyTorch 差的主因"。**2026-05-31 model-soup 实测直接证伪了这个假说**。以下是诚实的修正记录 — 保留错误推断的轨迹以警示。

#### 8.4.1 代码事实 (仍成立): train_pytorch.py 确实不支持 EMA

`train_pytorch.py:513` 写死 `logging.info("EMA is not supported for PyTorch training")` — PyTorch 训练**确实没有 EMA** (config 的 ema_decay=0.9999 被静默忽略, ckpt 存 raw 末步权重)。这是**事实**, 但下面证明它**不是** MAE gap 的主因。

#### 8.4.2 Model-soup 实测 (decisive, 2026-05-31)

**方法**: 均匀平均末段 6 个 ckpt (40k/42k/44k/46k/48k/50k) 模拟 EMA(0.9999) 末 ~10k 步窗口 (脚本 `train_scripts/kai/eval/model_soup_ema_probe.py`)。同 `eval_val_action_mse.py` (20 ep × 200 frame) 评估 soup vs plain-50k。

**权重确实平均了** (diff 探针): `|soup−50k| ≈ 0.5 × |40k−50k|` (action_in_proj.weight: 2.3e-5 vs 4.9e-5), 数学正确, soup ≠ 50k。

**结果** (native A_new_pure_200_val, 20 ep × 200 frame, 同协议):

| ckpt | @1 | @10 | @25 | @50 |
|---|---:|---:|---:|---:|
| **PyTorch plain 50k** (no EMA) | **0.0100** | 0.0174 | 0.0258 | **0.0350** |
| **PyTorch soup 40k-50k** (≈EMA) | **0.0101** | 0.0175 | 0.0259 | **0.0349** |
| Δ (soup vs plain) | +1% | +0.6% | +0.4% | **−0.3%** |

→ **soup ≈ plain, 全 horizon 差异 < 1%**。模拟 EMA **没有任何改善**。**EMA 缺失假说证伪。**

#### 8.4.3 同时发现: §8.2 训练 inline-eval 数字 ≠ 独立 eval (协议不同)

| 来源 | 同一个 50k ckpt 的 @50 |
|---|---:|
| §8.2 训练时 inline-eval 记录 | **0.0646** |
| 本次独立 `eval_val_action_mse.py` (20ep×200f) | **0.0350** |

差 1.8× — **训练内 inline-eval 与独立 eval 脚本的采样/帧选取/val 子集不同**。

→ **这推翻了 §8.3 "PyTorch @50 比 JAX 差 7.4×" 的整个对比基础**: 那是拿 PyTorch 训练 inline-eval (0.0646) 比 JAX 训练 inline-eval (0.0087), 但两者 eval 协议未对齐 + ckpt 也未同协议重测。**是苹果比橘子。**

#### 8.4.3b Preprocessing / norm 对齐复核 (2026-05-31, 仍有效)

| 环节 | 对齐? |
|---|---|
| action/state Normalize (共享 data loader) | ✅ 一致 |
| norm_stats 来源 / eval 反归一化 | ✅ 同脚本 |
| image 像素域 [-1,1] (eval train=False) | ✅ 一致 |
| **image augmentation (train)** | ⚠️ PyTorch train 有 crop/rotate/color aug, JAX 待确认 |

#### 8.4.4 ✅ 三方同协议对比 — 框架 gap 真实存在 (2026-05-31 闭环)

JAX ckpt (`task_a_pure200_base_pi05_step49999`, 自带 `assets/a_new_pure_200/norm_stats.json`) **就在本机**, 用同一 `eval_val_action_mse.py` (20ep×200f, prompt 一致) 重测, 三方终于可比:

| ckpt | @1 | @10 | @25 | @50 |
|---|---:|---:|---:|---:|
| **JAX pure_200** (同协议) | **0.0066** | **0.0074** | **0.0078** | **0.0085** |
| PyTorch plain 50k | 0.0100 | 0.0174 | 0.0258 | 0.0350 |
| PyTorch soup (≈EMA) | 0.0101 | 0.0175 | 0.0259 | 0.0349 |
| **Δ (PyTorch / JAX)** | **+52%** | +135% | +231% | **+312% (4.1×)** |

> JAX 同协议 @1=0.0066 ≈ §7.1 训练记录 0.0065 → JAX eval 协议自洽, 数字可信。

**铁结论**:
1. ✅ **EMA 不是主因** (soup≈plain, 证伪)。
2. ✅ **PyTorch 确实显著差于 JAX** — 同协议 @50 真差 **4.1×** (不是 §8.3 的 7.4× 伪数, 但绝非无差)。
3. ✅ **gap 随 horizon 单调放大** (@1 +52% → @50 +312%) — 典型 **chunk rollout 误差累积**, 单步就偏 (52%) + 越滚越偏。

**修正后的根因候选** (EMA 已排除, 按 horizon-scaling 签名重排):

| 候选 | 与"@1 就偏 + horizon 放大"签名吻合? | 优先级 |
|---|---|:-:|
| **flow-matching sampler / denoising 实现差异** (num_steps / dt / noise schedule, PyTorch `sample_actions` vs JAX) | ✅ 高度吻合 — 采样器偏差每步累积 | ⭐⭐⭐ |
| **train-time image aug** (PyTorch 有 crop/rotate/color, §8.4.3b) 拉低 in-distribution 锐度 | ⚠️ 能解释 @1 偏 52%, 但不完全解释 horizon 放大 | ⭐⭐ |
| bf16 数值 / DDP grad sync | ⚠️ 通常均匀抬高, 不强烈 horizon-scaling | ⭐ |

**待办 (要定位 PyTorch 真实缺陷)**:
- ⭐ 对比 PyTorch `pi0_pytorch.sample_actions` vs JAX `pi0.sample_actions` 的 denoising loop (num_steps, dt 符号, noise→action 方向) 是否逐行等价
- ⭐ 关掉 PyTorch train-time image aug 重训一小段, 看 @1 是否回到 ~0.0066
- 这些是 PyTorch 路径能否用于生产的前置, 不是 EMA。

#### 8.4.5 教训 (诚信)

1. **跨协议对比 = 苹果比橘子**: §8.3 拿 PyTorch 训练 inline-eval 比 JAX 训练 inline-eval, 未确认两者 eval 协议一致 → 得出 7.4× 的伪结论。**任何框架/方法对比, eval 协议必须逐字对齐, ckpt 用同一脚本重测。**
2. **假说要先证再写**: 早期版本把 "EMA 缺失是主因" 当结论写进文档 (还一度写了未实测数字), 实测 soup≈plain 直接打脸。**机理推断 (EMA 平滑长 horizon) 听起来合理, 但本数据集/本规模下不成立。**
3. **soup 是诊断 EMA 假说的正确工具**, 这次用对了 — 它廉价、确定地证伪了假说, 避免了白做 EMA patch + 68h 重训。


### 7.7 修正后的归因表

| 维度 | 之前 (错误) | 现在 (正确) |
|---|---|---|
| Init 影响 | 主导因素, mixed_1_clean 优 | **无影响**, 同 0.0065 |
| Mirror 缺失 | 主因之一 | 主要 boost in-distribution 精度, 对 generalization 影响小 |
| 数据量更大反而差 | 是 (基于 native val) | **错** — vis_v2_full 在 vis_v2_val50 上 0.0131 < pure_200 cross val 0.0207 |
| 16 dates 协议漂移 | 主因之一 (训练不锐利) | 部分对 — 但同时给了 generalization prior |
| Val 选择 | (未考虑) | **决定性因素** |
| 真机表现 | offline MAE 越低越好 | 取决于真机条件与 train distribution 距离 |

### 7.8 下一步可立即跑的实验

| 优先级 | 行动 | 预期发现 |
|---|---|---|
| ⭐⭐⭐ P0 | **真机评估 pure_200 vs vis_v2_full 在近期/旧 dates 多场景** | 验证 §7.5 假设 |
| ⭐⭐ P1 | 构建 universal val (从 vis_v2_full 16 dates 各抽 ~2 ep) eval 所有 ckpt | 公平 generalization 比较 |
| ⭐⭐ P2 | **vis_v2_full ckpt 49999 + 100 unique pure_200 ep finetune 10k step** | 兼顾 broad prior + 锐利 |
| ⭐ P3 | pure_200 用更多 dates (e.g., 05-08~05-22 = 5 dates curated × mirror) 重训 | 测 broader train + curated data 是否最优 |

### 7.9 关键 Takeaway (修正版)

1. ✅ **小 curated 数据 + 重复 = 在 in-distribution val 上极佳**, 但是**严重过拟合, 在 cross-val 3.2× 退化**
2. ✅ **大 diverse 数据 + 跨日期 = 在 broader val 上更稳** (vis_v2_full 0.0131 < pure_200 cross 0.0207)
3. ❌ ~~Init 决定上限~~ — 数据决定上限, init 影响收敛速度但不影响 final 精度
4. ❌ ~~vis_v2_full 训练失败~~ — 训练 OK, 在跨日期 val 上反而比 pure_200 好 36%
5. ⚠️ **必须用 train/val distribution 匹配的 val 比较 MAE**, 跨分布的 MAE 不可比
6. ⚠️ **真机 deployment 应根据真机场景 vs train distribution 距离选择 model**: 近期场景 → pure_200, 跨日期/OOD → vis_v2_full

---

## 8. 2026-05-27 Chunk/Noise diagnostic 更新

为定位 vis_v2_full 真机 "走几步退几步 + 夹爪犹豫" 问题, 在 3 个 ckpt 上跑了 P0/P1/P2 + fixed-noise 诊断. pure_200 这边的 fresh measurement:

| 指标 | pure_200 | vis_v2_full | TAC v7 |
|---|---:|---:|---:|
| P1 random L | 0.0390 🟡 | 0.0265 🟢 | 0.0666 🔴 |
| P1 fixed-noise L | 0.0389 🟡 | 0.0206 🟢 | (未测) |
| **Noise contribution ΔL** | **+0.0001** | +0.0059 | — |
| P2 mean variance | **0.0117** | 0.0234 | 0.0231 |
| P2 max variance | **0.1903** | 0.6688 | 0.6525 |

**pure_200 的特征**:
- ✅ **几乎完全 deterministic** — noise 贡献 0.0001 (vis_v2_full 是 0.006, 大 60×)
- ✅ **P2 max 0.19** (低 — 所有时刻所有 dim 都 deterministic)
- 🟡 **P1 = 0.039 MARGINAL** — chunk-to-chunk diff 主要来自**输入变化** (state@k → state@k+5), 不是 noise

**为什么 pure_200 真机 work 而 vis_v2_full 不 work**: pure_200 的 deterministic 行为意味着同一 obs 始终输出相同 action chunk, **关键时刻不会跳 mode**. vis_v2_full 大多数时刻 deterministic 但在 rare 关键时刻 (P2 max 0.67) 多 mode → 真机看到的 oscillation.

**修复路线**: vis_v2_full 走 [G0 fixed-noise inference fix](../../../deployment/inference/fixed_noise_inference_fix.md), 不需要重训. 详见 [analysis §11](../../analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md#11-2026-05-27-重大数据更正--chunknoise-诊断-定位真机-oscillation).
