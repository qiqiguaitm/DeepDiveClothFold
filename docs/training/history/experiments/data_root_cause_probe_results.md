# 数据问题排查实验 — 结果记录 (Data Root-Cause Probe Results)

> **作用**: 记录 [`../../future_plans/plans/data_root_cause_probe_experiments.md`](../../future_plans/plans/data_root_cause_probe_experiments.md) 系列实验的训练 / offline MAE / 真机结果。
> **状态**: 🔄 Exp-1 (裁投放) 训练+MAE 完成; **真机 no-release 比 raw 明显改善 (用户 2026-06-02) → H1 初步成立** (机理见 §4); Exp-1b (不裁对照) 训练中, 待并排真机严格坐实。**H2 数据侧已定位 (2026-06-07, 见文末 §H2)**: 5-19~5-27 的失败根因是**抓取前接近段"双臂同停"idle 翻倍** (34–43% vs 好数据 15–20%), 已数据侧**证伪**"坏抓取 demo / 反复重抓 demo / 抓取点固定"三种假设。
> **建立**: 2026-06-02
>
> ⚠️ **方法学铁律 (来自 plan §0)**: 本系列**以真机为终判, offline MAE 系统性反指** (慢/停顿轨迹逐帧误差低却真机灾难; gripper/wrist 问题被 12D arm 稀释)。下面的 MAE **只用于** ① 确认训练健康收敛 ② 选真机测试用的 best ckpt ③ Exp-1 vs Exp-1b 同验证集的相对差。**MAE 不能单独判定 H1** —— 走停/犹豫 (症状①) 在 offline 逐帧 MAE 上几乎不可见。

---

## Exp-1 — `A_0522_0526_no_release` (裁投放, 验证 H1) ✅ 训练+MAE 完成

### 1. 训练配置 (实跑)

| 项 | 值 |
|---|---|
| Config | `pi05_flatten_fold_A_0522_0526_no_release` |
| 集群 | **cnsh 16×A100** (Volc robot-task), FSDP effective batch=128 |
| Init | `mixed_1_clean` |
| 数据 | `A_0522_0526_no_release` (5-22+5-26 共 200 ep, 裁投放后 ~313k frames) |
| Prompt | `"Flatten and fold the cloth."` / abs joints (`use_delta_joint_actions=False`) |
| Steps | **50,000** (plan 写 40k, config 实跑 50k); save_interval=2000, keep_period=10000 |
| 速度 | ~46 步/min (2000 步/43min), 全程稳定 |
| ckpt 根 | `/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_A_0522_0526_no_release/A_0522_0526_no_release_cnsh/` (保留 step `10000 20000 30000 40000 49999`) |

### 2. Offline MAE — saved ckpt 逐点重测 (2026-06-02)

验证集 `vis_v2_merged_val` (30 ep, 与训练 inline-eval 同集, **cross-val**: 训练数据 ≠ 验证数据), prompt `"Flatten and fold the cloth."`, 200 frames, gf0 A100。

| step | MAE@1 | @10 | @25 | @50 | |
|---|---|---|---|---|---|
| **20000** ⭐ | **0.0160** | **0.0378** | **0.0686** | **0.1093** | **全 horizon 最优 → 真机首选** |
| 30000 | 0.0160 | 0.0384 | 0.0695 | 0.1101 | @1 平, 长程更差 |
| 49999 | 0.0163 | 0.0393 | 0.0704 | 0.1110 | 最差 (轻微过拟) |

**训练期 inline-eval 曲线** (同验证集, 每 8k 一次, 交叉印证):

| step | 16000 | 24000 | 32000 | 40000 | 48000 |
|---|---|---|---|---|---|
| MAE@1 | 0.0161 | **0.0159** | 0.0161 | 0.0163 | 0.0163 |
| MAE@50 | 0.1090 | 0.1096 | 0.1103 | 0.1107 | 0.1110 |

> 交叉印证: offline `49999`(@1=0.0163 @50=0.1110) ≈ inline `48000`(@1=0.0163 @50=0.1110) 完全吻合 → offline 重测可信。

### 3. 分析

- **曲线在 16k–24k 触底后单调微劣化** (@1 0.0159→0.0163, @50 0.1090→0.1110): 该数据集 (200 ep / 313k frame) **~20k 步即收敛, 之后轻微过拟**。50k 步对这个规模偏多。
- **best 可部署 ckpt = step 20000** (落在甜区、且是保存点)。已按 `checkpoints_layout.md` 扁平拓扑 A 打包 (剥 train_state, norm_stats 烘进 `assets/A_0522_0526_no_release/`):
  - `TAR: /vePFS/tim/pkg/A_0522_0526_no_release_best_step20000.tar` (11.6 GB)
  - ⚠️ 真机 config 需 `AssetsConfig(asset_id="A_0522_0526_no_release")`, 见打包说明。
- **MAE ≈ 后期 baseline 水平、无显著改善** —— 这**符合预期, 不构成 H1 的证据**: 裁投放只删开头 ~7% 静止帧, 而走停/犹豫 (症状①) 是**推理时的时序行为**, offline 逐帧 teacher-forcing MAE 看不见。H1 成不成立**只能靠真机**。

---

## Exp-1b — `A_0522_0526_raw` (不裁对照) 🔄 训练中, MAE 待出

**对照意义** (plan §1.6): 同两天数据、同 config、同 init、同 step, **唯一差别 = 不裁投放**。排除"只是用了 2 天/200ep 规模效应"的混淆, 让 H1 判定干净。

| 项 | 值 / 状态 |
|---|---|
| Config | `pi05_flatten_fold_A_0522_0526_raw` (50k step) |
| 集群 | **uc02 + uc03 2-node 16×A800** (JAX 多机 FSDP) |
| 数据 | `A_0522_0526_raw` (200 ep, 336,917 frames, 不裁) |
| Init / Prompt | `mixed_1_clean` / `"Flatten and fold the cloth."` (与 Exp-1 一致) |
| 状态 | 🔄 **稳定训练中** (2026-06-02 起); **step-2000 首个 ckpt save 已验证通过** (见下), MAE/ckpt 待训练完成回填 |
| ckpt 根 | `/data/shared/ubuntu/workspace/multinode_ckpts/pi05_flatten_fold_A_0522_0526_raw/A_0522_0526_raw_uc16/` (共享 NFS) |

> **✅ 多机稳定性已实测通过** (2026-06-02 09:12): step-2000 ckpt 在共享 NFS 落成 finalized `2000/` (12G params + metadata + assets + train_state, 无 tmp 残留), orbax `Wrote NNN array_metadata` 写入共享 NFS 成功 (= 原崩溃点), 训练继续到 Step 2200 loss 0.0075。**这才是多机真正的稳定判据** (非 Step100 loss 下降)。
> **基建踩坑** (迁 uc 多机时): 首跑崩在 step-2000 orbax 落盘 (ckpt 落节点本地盘), 换节点重跑又连挂 3 次 (JAX 编译缓存不对称致跨主机 clique init 死锁)。根因+修复见 [`../../backup/uc_cluster_jobs.md §12.11 坑 9/10`](../../backup/uc_cluster_jobs.md)。

### MAE (待回填)

| step | MAE@1 | @10 | @25 | @50 |
|---|---|---|---|---|
| _待训练完成_ | | | | |

---

## §4 机理 — 为什么"开头静止段"会致真机走停 (文献支撑) ⭐

> **背景**: 用户 2026-06-02 真机观察 **no-release (裁投放) 模型比 raw 走停/犹豫明显减少**, 但**两者 offline MAE 几乎相同** (§2: no-release @1=0.0160 ≈ 后期 baseline)。这看似矛盾, 实则**完全符合模仿学习理论** —— 走停是闭环推理的时序行为, offline 逐帧 MAE 测不到。以下为机理 + 文献。

### 4.1 核心论点: MAE 不变但真机变好 = 投放静止段是症状① (走停/犹豫) 主因

| | raw (含投放静止段) | no-release (裁掉) |
|---|---|---|
| 训练标签 | 大量 "不动→不动" 样本 (后期 ep 静止帧% 38 vs smooth 32.7) | 全是有效操作动作 |
| 学到的起手策略 | "先等一会儿" (idling) | "直接起手" |
| 惯性捷径 (copycat) | 被强化 → 真机易锁死在静止态 | 弱化 |
| chunk 内容 (pi05 chunk=30/50) | 可能含停顿子段 → 整块开环复现 | 全是有效运动 |
| **offline MAE** | — | **几乎不变** (逐帧 teacher-forcing 看不到时序停滞) |
| **真机时序行为** | 走停 / 犹豫 / loop | 流畅 / 果断 |

**为什么 offline MAE 测不出**: MAE 是**开环逐帧** (每帧喂真值 obs), 策略永远不会进入"停滞累积"状态; 走停是**闭环 rollout** 时策略进入"像静止的状态"就停 → 只有真机/闭环能暴露。**这正是 plan §0 "offline MAE 系统性反指" 铁律的具体实例。**

### 4.2 四个机制 (按相关度)

**① Policy Idling (策略停滞) — 最直接** ⭐⭐⭐
DeepMind 2025 *Exploiting Policy Idling for Dexterous Manipulation*: "成功的人类演示常含操作前的细微停顿; 用未过滤数据训练 SOTA 模仿学习策略时, 策略会在 rollout 时**复现这些停顿** (idling behavior)。" BC 是纯监督模仿, **不区分"该停"和"不该停"**, 忠实把静止学进策略。论文明确缓解办法 = **过滤训练数据** (正是本实验做的)。

**② Idle Frames 是 BC 已知毒药, 标准做法就是删** ⭐⭐⭐
π *Real-Time Chunking*: "需保持静止的任务 (如倒酱汁) 产生的 **idle actions 对 BC 是已知难点, 通常被避免或过滤掉**。" 毒性来源: (a) **数据失衡** — 静止段贡献大量"输出≈0"样本, 先验偏向不动; (b) **多解歧义** — 同一静止观测在不同 ep 后接不同动作 (继续等 vs 起手), 条件分布多峰 → BC 模态平均 → 边界处犹豫抖动。

**③ Causal Confusion / Copycat — 为什么"开头"尤其致命** ⭐⭐
UC Berkeley *Causal Confusion in Imitation Learning*: BC 非因果, 分不清真因与相关量。对开头静止段: (a) **inertia 捷径** — 策略易学到"上一刻不动→继续不动" (静止帧前后高自相关, 最易拟合的伪因果); 开头一长段静止 = 大量"不动→不动"强相关样本 → 强化惯性停滞 → 真机进入类静止态就锁死; (b) **起手锚点** — 模型对"开头→怎么动"学习权重高, 所有 ep 开头都"先等" → 真机一开始就犹豫。

**④ Action Chunking 把静止"焊死"进 chunk** ⭐
pi05 = flow-matching + chunk (30/50 步)。chunk 内若含静止子段 → 学到"预测含停顿的动作块" → 推理整块开环复现停顿, 比单步 BC 更难纠 (单步下帧可修, chunk 焊死)。RTC inpainting 是治标, **治本是训练数据无静止段**。

### 4.3 推论 (对所有数据集的指导)

1. **所有数据集都应裁掉开头投放/等待静止段** (motion-onset 检测 + margin, 是标准做法)。
2. **不止开头** — 机制①② 说明**中途长停顿、反复重抓 loop 段也该清** (= plan H2 / Exp-2 方向)。建议把 motion-onset 推广成"全程静止段检测"。
3. **smooth_800 为何真机 work**: 早期数据节奏紧凑 (ep 中位 1091 vs 后期 1600+, 静止帧% 32.7 vs 38) = **天然已过滤**, 无需裁。
4. **gripper 松手 (症状②) 独立**: 裁投放治不了 (没动 gripper 维), 需单独排查 (H3)。

### 4.4 文献

- [Exploiting Policy Idling for Dexterous Manipulation (DeepMind 2025)](https://arxiv.org/pdf/2508.15669) — policy idling 定义 + 数据过滤缓解
- [Real-Time Execution of Action Chunking Flow Policies (π)](https://arxiv.org/html/2506.07339) — idle actions 是 BC 难点, 常被过滤
- [Causal Confusion in Imitation Learning (UC Berkeley, 1905.11979)](https://arxiv.org/pdf/1905.11979) — copycat / inertia 捷径
- [What Matters in Learning from Offline Human Demonstrations (robomimic, Stanford)](https://robomimic.github.io/study/) — 演示数据质量对 BC 的影响

> **结论**: "MAE 不变 + 真机变好" 不是矛盾, 而是 **H1 (投放静止段致走停) 成立的标志性证据** —— 配合真机终判可坐实。理论上裁投放对症状①有效有充分文献支撑。

---

## H1 终判 — 🟢 初步支持 (用户真机观察), 待 Exp-1b 对照坐实

| 比较 | 状态 |
|---|---|
| **Exp-1 (裁) 真机走停/犹豫** | 🟢 **用户 2026-06-02 观察: no-release 比 raw 明显改善** (走停/犹豫减少) |
| Exp-1 vs Exp-1b **同验证集 offline MAE** | ⏳ 待 Exp-1b 训完 (预期两者 MAE 相近 — 不影响 H1, 见 §4.1) |
| Exp-1 (裁) vs Exp-1b (不裁) **真机并排对照** = H1 严格终判 | ⏳ 待 Exp-1b ckpt 真机测 |

**判定规则** (plan §1.7):
- 裁后真机走停/犹豫显著改善 → ✅ **H1 成立** (投放静止段是症状①主因)。**← 用户初步观察落此档**
- 无改善 → ❌ H1 排除, 转 Exp-2 (H2 整段慢节奏 / H4 wrist)。
- 改善但残留 loop → ⚠️ H1 部分成立, 与 H2/H4 叠加。

**当前判断**: 用户真机观察 (no-release 明显好) + §4 文献机理 (policy idling / idle-frame 过滤是标准做法) → **H1 高置信成立, 投放静止段是症状① 走停/犹豫的主因**。⏳ 严格终判仍需 Exp-1b (不裁同数据) 并排真机, 以排除"只是 2 天/200ep 规模效应"。

> 松手 (症状②) 真机未改善 = 符合预期 (没动 gripper), 佐证症状①② 独立 → 走 H3。
> ⚠️ **offline MAE 几乎相同 ≠ 裁投放无效** (§4.1): 走停是闭环时序行为, 逐帧 teacher-forcing MAE 测不到; "MAE 不变 + 真机变好" 正是 H1 成立的标志。

---

## H2 — 数据侧定位: 抓取前接近段"双臂同停"翻倍 (2026-06-07)

> **触发**: 用户真机观察 — 5-19~5-27 数据训出的模型**死记关节顺序**: 无论是否真夹到衣服, 都把"伸手→闭合"这段开环走完, 之后才"发现"没夹到, 再回到衣服位置重抓。即**抓取及后续一段动作忽视视觉, 只学到动作顺序**。这是 plan §H2 ("整段慢节奏/多停顿/反复重抓") 的真机症状。
>
> **方法**: 直接扫 600 个 episode 的 action parquet (无需解码视频), 脚本 [`train_scripts/kai/data/scan_grasp_retry.py`](../../../../train_scripts/kai/data/scan_grasp_retry.py) + 内联相位/分臂分析。对照组 = 早期真机能跑的好数据 (4-28 / 5-06 / 5-08 / 5-10)。

### H2.1 先证伪三个直觉假设 (都不是元凶)

| 假设 | 5-19~5-27 | 好数据 | 判定 |
|---|---|---|---|
| **相机冻结** (帧重复→图不变动作变→学忽略视觉) | freeze 0.01–0.1%; 仅 5-27 hand_left 1 ep 卡 207 帧 | 同量级 | ❌ 排除 (见 `Task_A/base/analysis/freeze`) |
| **数据含"抓空→重抓"演示被 BC 照抄** | aborted-in-place **0.04/ep**, regrasp-loop **0.30/ep** | 0.04 / 0.29 | ❌ 与好数据几乎相同 → 真机重抓是**涌现的因果混淆, 非抄自数据** |
| **抓取点被固定** (衣服总在同位→不需视觉) | 首抓臂位姿 std **0.27–0.35 rad** | 0.15–0.35 (5-19~5-27 反而更分散) | ❌ 抓取点正常随衣服变化 |

> 结论: 失败**不是少数坏抓取 episode**, 而是分布级缺陷。**修正 plan §H2 原假设**: "反复重抓示范风格"被证伪 (重抓 demo 未升), 真正抬高的是下面的**抓取前 idle**。

### H2.2 决定性证据 — idle 全部集中在"抓取前接近段"

按第一次真正夹衣服 (首个 open→close, gripper 米值 0=闭 0.08=开) 为界拆相位; **idle 帧 = 双臂 12 关节最大速度 < 0.002 rad/frame (即 L 和 R 同时几乎不动)**:

| 相位 | **5-19~5-27** | 好数据 |
|---|---|---|
| **抓取前 approach idle** | **34–43%** | 15–20% |
| 抓取及之后 idle | 4–9% | 4–10% (**一致**) |

多出来的 idle **全部在第一次抓取之前**; 抓取后段与好数据无差异。按天: 5-20 / 5-21 / 5-22 最重 (pre 41–43%), 5-19 最轻 (34%)。全体 **63% 的 ep pre-grasp idle >35%, 37% >45%**; 极端 ep (5-22 ep90 / 5-20 ep75 / 5-21 ep81) 接近段 **60–74%** 的帧双臂全停。

**分臂拆解 (排除"一臂等另一臂"的正常协同)** — pre-grasp 段:

| 口径 | 好数据 | 5-19~5-27 |
|---|---|---|
| **BOTH 双臂同停** (本节用的) | 15–20% | **34–43%** ↑↑ |
| either 至少一臂停 | ~60% | 68–77% (仅略升) |
| L only (左停右动 = 协同) | 23–25% | **11–16%** ↓ |
| R only (右停左动 = 协同) | ~21% | 18–24% (持平) |

抬高的**只有"双臂同时彻底停"**; 代表正常双臂协同的 L-only/R-only 不升反降 → 多出的是**纯死时间 / 悬停犹豫**, 不是协同等待。

### H2.3 机理 — 为何"接近段 idle"专门毒化抓取

1. 采集时第一次去夹衣角前, 双臂悬停/犹豫/等待时间是好数据 **~2×**。
2. 该段 idle 帧**视觉-动作几乎零相关** (手不动, 画面随便变, action≈0) 且各 ep 高度雷同。
3. pi05 = flow-matching + **action chunk(30/50)**: 成块的"悬停→伸手→闭合"被**焊进一个 chunk** → 模型学成"接近 = 放一段固定开环序列", 时序/本体感受即可预测, **不需看图**。
4. 推理时它不看是否真夹到, 照走"伸手+闭合", 直到后段视觉重新起作用才"发现"失败 → 回去重抓。= 用户症状①的精确数据来源。

> 与 §4 机理一致: §4.2 ①Policy Idling + ③Causal Confusion (copycat/inertia) + ④Chunk 焊死。本节把 §4.3 #2 "不止开头" 量化并**定位到抓取前接近段** (而非中途停顿或末端重抓循环)。
> **模型侧已独立印证** (memory `reference_vision_ablation_openloop.md`): `eval_vision_ablation.py` 确诊 "5day_recent" 开环 — 夹爪置黑 SNR **5.7×** vs 健康 26.6×。数据侧 (本节) 与模型侧互证。

### H2.4 "哪些数据有问题" + 修法

- **范围**: 5-19~5-27 共有缺陷, 严重度 **5-20 ≈ 5-21 ≈ 5-22 > 5-26 ≈ 5-27 > 5-19**。
- **最该优先裁的极端 ep** (pre-grasp idle, 全表 `/tmp/openloop_risk_suspect.json`): `5-22/ep090(74%)` `5-20/ep075(74%)` `5-21/ep081(72%)` `5-26/ep087(68%)` `5-22/ep057(67%)` `5-20/ep008(66%)` `5-19/ep094(66%)` …
- **修法 (= Exp-2, H2 验证实验)**: 不删 episode, 而做 **motion-onset 裁剪 + 接近段内部 idle 压缩**, 把抓取前 idle 从 34–43% 拉回 ~15% 基线。代价极小 — **裁剪到 15% 基线仅删约 4.9% (~5 万帧)**, 完全不碰抓取及之后的有效动作。是 Exp-1 `no_release` ("只裁开头投放") 向"裁整个接近段双臂同停"的推广。
- **判定**: 重训 (Exp-2) 后真机"伸手开环走完再回抓"显著减少 → H2 成立。

### H2.5 复现

```bash
PY=kai0/.venv/bin/python
# 抓取循环 / 短抓 / 重抓-loop 扫描 (证伪假设 2)
$PY train_scripts/kai/data/scan_grasp_retry.py \
    --dates 2026-05-19-v2,2026-05-20-v2,2026-05-21-v2,2026-05-22-v2,2026-05-26-v2,2026-05-27-v2 \
    --top 25 --out /tmp/grasp_retry_suspect.json
# 相位/分臂 idle 对比见本节内联脚本 (gripper 0=闭 0.08=开; idle 阈 0.002 rad/frame; 双臂同停)
```
