# AWBC × ViVa Value Model 对比实验方案

**创建时间**: 2026-05-31
**状态**: 🔶 **暂缓 (2026-06-06)** — ViVa(5B WAN value,单卡 ~19 天/模型 + 跨集群算力)成本太高,**先走传统 pi0-AdvantageEstimator 路线出 AWBC 基线**,并把数据从 demo-only `smooth_800` 扩成 **smooth800+全dagger**(dagger 段正好补上本文 §3 指出的 demo-only 天花板)。传统路线见 [`../../../deployment/strategy/awbc_implementation_plan.md`](../../../deployment/strategy/awbc_implementation_plan.md) ⭐当前执行计划。ViVa 待传统路线出基线后,作为"换更强 value 模型"的对照再上(本文档保留为 ViVa 方案蓝本)。
**负责人**: Tim
**一句话**: 把 AWBC pipeline 里产 advantage label 的 estimator 换成 **ViVa video-generative value model**,做**只换 label 来源、其余全锁死**的受控对比 —— 且对比**两个 ViVa 变体**(official 线性进度 value vs DSM-r30 里程碑加权 value),看哪种 value 信号能让同一套 AWBC 训出更好的 policy,并与 smooth_800 SFT 基准对照。

**上游/关联**:
- AWBC pipeline 全貌 → [`../../../deployment/strategy/awbc_implementation_plan.md`](../../../deployment/strategy/awbc_implementation_plan.md)(4-step RECAP)
- 历史失败教训 → [`awbc_pi07style_experiment.md`](../../history/experiments/awbc_pi07style_experiment.md)(π0.7-style，全失败，根因:demo-only advantage 方差 η²≈3%、prompt 信号弱)
- 已实施的 v2 数据扩充 → [`awbc_v2_training_plan.md`](../../history/experiments/awbc_v2_training_plan.md)(base+dagger+mirror，12,024 ep)
- ViVa 论文 → [arXiv 2604.08168](http://arxiv.org/abs/2604.08168) / 仓库 `/vePFS/zundong/ViVa`(GigaAI-research/ViVa)

### 参与对比的 value 模型(2 个 ViVa 变体,均 WAN2.2-TI2V-5B / 5.00B / step 7000 / task_a_0509v2)

| 代号 | ckpt 路径 | value target | 关键 config | 说明 |
|---|---|---|---|---|
| **ViVa-official** | `/vePFS/zundong/checkpoint_step_7000` | **线性进度** `frame_idx/(ep_len-1)`(0→1 单调) | run `viva-TaskA0509-official-0521.0225`;cam_key 默认 `cam_high` | value = 朴素任务进度 |
| **ViVa-DSM-r30** | `/vePFS/zundong/robot` | **Dual-Slope Milestone**(`dual_slope_r=30`,critical 帧陡升)| run `viva-TaskA0509-DSMr30-0530.0415`;cam_key 直接用 `top_head/hand_left/hand_right`;需 `milestones_task_a_0509v2.jsonl`(仅训练用)| value 在 milestone/critical 帧斜率放大 → AWBC 给关键帧更高权重 |

> 两者**架构完全相同**(`video_model.wan_model.*`,5.00B,BF16),只在 value target 形状不同。**核心假设**:DSM-r30 的"关键帧加权"value 比 official 的"线性进度"value 更能让 AWBC 抓住任务关键步骤 → 训出更好 policy。这正是本对比要验证的。
> **cam_key 差异**:official 用默认 `cam_high` → 需 lerobot-compat 视图改名;DSM-r30 config 已显式 `cam_key=top_head` → 可直接读 smooth_800 原始相机名(也可统一走视图,二选一,见 §2)。

---

## 0. 为什么这个对比有意义(动机)

历史 AWBC 全线失败的**根因不在 AWBC 训练侧,而在 advantage label 侧**(见 `awbc_pi07style_experiment.md` 第十节):

1. pi0-AdvantageEstimator 的 `absolute_value` 与 GT progress corr=0.896(本身不错),
2. 但 AWBC 真正用的是 `absolute_advantage = absolute_value(t+50) − absolute_value(t)`(**二阶差分**),差分把噪声放大 → corr 掉到 ~0.3-0.4,
3. demo-only 数据 advantage 方差本就只有 η²≈3%(弱-中),再叠加噪声 → prompt 信号弱到模型直接学会忽略。

**ViVa 的卖点正好打这个痛点**:它用预训练视频生成器(WAN)的时空先验,联合预测"未来 proprioception + 标量 value",把 value 估计 grounding 在**预期的 embodiment dynamics** 上,而不是 pi0 那种 static-snapshot 回归。论文实测 ViVa 接入 RECAP 在真机 box assembly 上有 substantial improvement,且 value 曲线更可靠(更贴合真实任务进度)。

→ **假设**: ViVa value 信号的信噪比 > pi0-AdvantageEstimator,因此**同一套 AWBC 训练**用 ViVa label 应当训出 ≥ pi0-label baseline 的 policy。这是本实验要证伪/证实的核心命题。

---

## 1. 实验设计:受控多臂(单变量 = value 模型)

**唯一变量 = 给 smooth_800 打 advantage label 的 value 模型**。其余一切(训练数据 episode 集合、AWBC config、超参、seed、discretize 方式、eval val split、评估协议、warm-start init)全部锁死。

| | **Baseline** | **Arm V1 (ViVa-official)** | **Arm V2 (ViVa-DSM-r30)** | (可选)**Arm A (pi0-AE)** |
|---|---|---|---|---|
| Label 来源 | 无(纯 SFT)| ViVa-official value → `viva_adv=value(t+Δ)−value(t)` | ViVa-DSM-r30 value → 同式 | pi0-AdvantageEstimator `absolute_advantage` |
| Label 工具 | — | `ViVa inference_half_8gpu.py` + ckpt `checkpoint_step_7000` | 同,ckpt `/vePFS/zundong/robot` | `kai0/stage_advantage/annotation/eval.py` |
| 离散化 | — | `discretize_advantage.py binary` | **同脚本同阈值** | **同脚本同阈值** |
| 训练 config | (SFT,已训完)| AWBC config,`repo_id`→V1-labeled | 同,`repo_id`→V2-labeled | 同,`repo_id`→pi0AE-labeled |
| **基准数据集** | **`A_new_smooth_800`(811 ep)** | 同 | 同 | 同 |
| **Init(warm-start)** | — | **smooth_800 SFT 49999 ckpt** | 同 | 同 |
| 训练超参 | — | batch=128,fsdp=8,nw=64,同 seed | **完全相同** | **完全相同** |
| Eval | doc 数字 MAE@1=0.0089 | smooth_800/val(26 ep)同协议 | **完全相同** | **完全相同** |
| exp_name | (已有 49999)| `awbc_viva_official_7k` | `awbc_viva_dsmr30_7k` | `awbc_pi0ae` |

> **主对比 = V1 vs V2**(两个 ViVa value 变体哪个更好),**都对照 Baseline**(AWBC 是否比纯 SFT 强)。pi0-AE 臂(Arm A)为可选历史对照,优先级最低。

> **基准 / init 定档(2026-05-31 核实)**:
> - 数据集 = `uc03:/data/shared/ubuntu_old/data/Task_A/A_new_smooth_800/{base,val}`(811 ep base + 26 ep val,Agilex 三相机 top_head/hand_left/hand_right + state-14,**纯 SFT、tasks.jsonl 仅 1 条 prompt → 两臂都需从零打 advantage 标**)
> - Init = `uc03:/data/shared/ubuntu_old/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_smooth_800_new_norm/task_a_new_smooth_800_new_norm/49999/params`
> - **参考基准数字**(该 SFT run,见 [`../../history/experiments/task_a_new_smooth_800_new_norm_results.md`](../../history/experiments/task_a_new_smooth_800_new_norm_results.md)): MAE@1=**0.0089** / @10=0.0221 / @25=0.0404 / @50=0.0636,step 40k 起 plateau。
> - **为什么 warm-start 而非 pi05_base 冷启**:AWBC 的前提就是"SFT 已 plateau 后做 frame-level 加权精修"(见 [`../../../deployment/strategy/awbc_implementation_plan.md`](../../../deployment/strategy/awbc_implementation_plan.md) §2)。从已收敛的 0.0089 policy 续训,(a) 任何低于 0.0089 的改进都干净归因到 advantage 加权,(b) 续训步数少(10-20k 即可),迭代快。

> **为什么必须单变量**:历史 π0.7 实验一次改了 3-4 个东西(n_slices+stage-aware+dropout),失败后无法归因。本次每臂只改 value 模型,任何差异都能干净归因。

---

## 2. Pipeline 映射:ViVa value → AWBC task_index

AWBC 训练侧(Stage 4)完全不动,它只认数据集里每帧的 `task_index` + `meta/tasks.jsonl` 的 prompt。我们要做的是**用 ViVa 重新生成 task_index 这一列**,流程对齐现有 Stage 2→3:

```
每个 ViVa 臂 (V1 official / V2 DSM-r30, 流程相同, 仅换 ckpt):
  dataset(lerobot-compat 视图) ── inference_half_8gpu.py (ViVa ckpt) ──► +列 prediction (=ViVa value)
          ── [薄脚本] viva_value → viva_advantage = value(t+Δ) − value(t)  ──► 写回 absolute_advantage 列
          ── discretize_advantage.py (binary, 同阈值) ──► task_index ∈ {0,1} + tasks.jsonl(同 prompt 文本)
          ── AWBC 训练 (warm-start 49999, repo_id 换成该臂 labeled 数据集)

(可选历史对照 Arm A, pi0-AE):
  dataset ── eval.py (pi0 AE) ──► absolute_advantage ── 同一 discretize ── task_index ── 同一 AWBC 训练
```

**关键适配点(3 个薄脚本/检查,不碰 AWBC 训练代码)**:

1. **lerobot-compat 视图**: ViVa 期望 ALOHA 风格 3 相机(`cam_high / cam_left_wrist / cam_right_wrist`)+ state-14。Task_A(Agilex)是 top + 双 hand 相机 + state-14。ViVa ckpt 训练用的 `task_a_0509v2_lerobot_compat` 已经是这个转换的产物 → 复用同一转换脚本,把 deepdive_kai0 的 AWBC 训练数据集转成 ViVa 能读的视图(只为推理喂数据,不改原训练数据集)。
2. **viva_advantage 计算 + 符号**: ViVa value 语义需先验证(见 §6 风险 R3 — findings.md 里有"递减 fraction-remaining"和"递增 progress"两种说法)。写一个薄脚本对齐到现有 `absolute_advantage` 语义(progress 变化率,越大越好),必要时翻符号。Δ 默认取 ViVa 自己的 `future_offset=30`,并做 Δ=50 的对照(对齐 pi0 AE 的窗口)。
3. **写回 + discretize**: 把 `viva_advantage` 写进 parquet 的 `absolute_advantage` 列(列名复用,这样 `discretize_advantage.py` 零改动),用与 Arm A **完全相同**的 `--threshold` / `--discretion-type binary` / `--stage-nums` 跑离散化。`tasks.jsonl` 的 prompt 文本两臂保持一致("Flatten and fold the cloth. Advantage: positive/negative")。

---

## 3. 数据集 / Init / 集群(已定档)

**数据集 = `A_new_smooth_800`**(811 ep base + 26 ep val,uc03)。选它的理由:
- 小(811 ep)、训练快、有**已知 SFT 基准数字**(MAE@1=0.0089)可直接对照;
- 与 init ckpt 同源(下一条),warm-start 干净;
- ViVa ckpt 也是在 Task_A 同域(`task_a_0509v2`)训的 → 域匹配。

**Init = smooth_800 SFT 49999 ckpt(warm-start,两臂共用)**。AWBC 是 SFT-plateau 后的精修,从 0.0089 的已收敛 policy 续训。

**集群 = uc03**(AWBC 训练);**ViVa labeling 不在 uc03**(uc 集群无 vePFS,跑不了 ViVa env)→ 见 §6 R1 跨集群流程。

> ⚠️ **smooth_800 是 demo-only 的固有局限**(必须正视):无 dagger/inference rollout 段。pi0-AdvantageEstimator 学"什么算低 advantage"**依赖见过失败段**(见 `awbc_implementation_plan.md` Stage 1 "为什么需要 inference 段")。纯 demo 上 pi0-AE 的 label 可能偏弱、advantage 方差 η²≈3%(历史天花板)。
> → 这恰恰是 **ViVa 的潜在优势点**:ViVa 用视频生成先验估 value,理论上对"没见过失败段"的依赖更小。本实验正面测这一点。但也要预期:若 ViVa 也吃不到 demo-only 的信号,两臂可能都打不过 0.0089 SFT 基准 → 那就转含 dagger 的数据集(见 §4 判据"打平"分支)。

**后续规模化(可选)**: 若 smooth_800 上 ViVa 显著赢,再上含 dagger/mirror 的 `advantage_v2`(12,024 ep)复现规模化收益。

---

## 4. 评估协议(决定胜负的标准)

> **用户定档(2026-05-31)**: 评估 = **Tier 1(MAE sanity)+ Tier 3(rollout,决定性)**;**不做 Tier 2**(label 质量离线诊断);baseline 直接用 smooth_800 SFT 的 doc 数字(MAE@1=0.0089),不单独重训 SFT。

**Tier 1 — 离线 in-training eval(便宜,sanity)**:
- 同 val split(smooth_800/val 26 ep),各臂 `mae_joint_{1,10,50}` / `mae_grip_{1,10,50}` 逐 checkpoint 曲线,对照 SFT 基准 MAE@1=0.0089。
- ⚠️ 纯 action MAE 对 AWBC 不够敏感(positive-prompt 推理下各臂 MAE 可能接近,历史如此)。MAE 只作 sanity,不作主判据。

**Tier 3 — Rollout(决定性主判据)**:
- sim01 部署各臂 ckpt(ViVa-official / ViVa-DSM-r30 [/ pi0-AE]),positive-prompt 推理,跑 Task_A flatten-fold rollout。
- 指标:成功率、完成帧数(episode 长度,越少越快)、关键 sub-phase(抓→对折)通过率。
- **主对比 = V1(official) vs V2(DSM-r30)**,均对照 SFT 基准。
- 这是 AWBC 真正的目标指标(论文也是看 throughput / 成功率,不是 MAE)。

**判据**(主看 Tier 3 rollout):
- **ViVa-AWBC 赢** = 某臂 rollout 成功率/throughput 明显 > SFT 基准 → 该 value 形状有效;若 V2(DSM-r30)> V1(official),则"关键帧加权 value"假设成立 → 推 advantage_v2 规模化。
- **打平** = 各臂 rollout ≈ SFT 基准 → demo-only 数据已触天花板,转含 dagger/失败段的数据集(ViVa 时空先验在 OOD 上才有空间)。
- **输** = 各臂均 < SFT → ViVa value 在 Task_A demo-only 域无增益,记录归档。

---

## 5. Phase 拆分

| Phase | 任务 | 工作量 | 关键文件/命令 | 前置 |
|---|---|---|---|---|
| **V0** | **可达性 + 健全性核对**: ① smooth_800 的 videos 是否 symlink 断链(重装后)、base/val 可被 lerobot 正常加载;② ViVa env(`viva` conda)+ ckpt 7000 在哪台机可跑、能加载;③ ViVa value 语义方向(递增/递减)在 smooth_800 的 3-5 ep 上肉眼确认(画 value 曲线) | 0.5 天 | `ViVa/scripts_repro/plot_value_curve.py` / `inference_half_8gpu.py` | — |
| **V1** | **lerobot-compat 视图**: 把 gf0 上重建好的 smooth_800 转成 ViVa 视图(top_head→cam_high / hand_left→cam_left_wrist / hand_right→cam_right_wrist + state-14 + 生成 state_stats.txt + task_a t5 embedding) | 0.5-1 天 | `/vePFS/tim/viva_work/scripts/make_lerobot_compat_view.py` | V0 |
| **V2** | **ViVa labeling ×2**: 在 smooth_800 视图上跑两个 ViVa 推理,各产 `prediction`(value)列 —— ① ViVa-official(`checkpoint_step_7000`),② ViVa-DSM-r30(`/vePFS/zundong/robot`)| 1 天(gf0 单卡逐 ckpt;或借多卡)| `inference_half_8gpu.py` + 各自 patched config(WAN 路径→gf0;DSM-r30 cam_key=top_head 可省视图)| V1 |
| **V3** | **三模型 label(同样处理)**: 每个模型(ViVa-official / ViVa-DSM-r30 / pi0-AE)各跑: ① `viva_value_to_advantage.py`(`viva_advantage=value(t)−value(t+Δ)`,Δ=30,符号已翻)写回 `absolute_advantage`;② **`corr_filter.py`** 按 \|corr\|≥0.5 产 keep-list 过滤差 episode;③ `discretize_advantage.py` binary 同阈值 → task_index + tasks.jsonl | 0.5 天(脚本)| `/vePFS/tim/viva_work/scripts/{viva_value_to_advantage,corr_filter}.py` + `kai0/.../discretize_advantage.py` | V2 |
| **V4** | **AWBC 续训 ×2**: 同 config/init(smooth_800 49999/params)/超参,`repo_id` 分别指 V1/V2-labeled,`exp_name=awbc_viva_official_7k` / `awbc_viva_dsmr30_7k`,uc03 8×A800,batch=128/fsdp=8/nw=64,续训 ~15-20k step | 2-3 天(两臂,uc03 串行或借 uc01/02 并行)| uc03 launcher(套 `run_uc03_*`)| V3 |
| **V6** | **评估 + 结论**: Tier1 MAE 曲线(对照 0.0089 基准)+ Tier3 sim01 rollout(成功率/throughput);主比 V1 vs V2;写 results.md + 更新 master history | 1-2 天 | `train_scripts/kai/eval/eval_awbc_compare.py` + sim01 部署 | V4 |

**总工作量**: **~7-9 天**(warm-start + smooth_800 小数据;两个 ViVa 变体各 labeling + 续训 + eval)。
**注**: 按用户最新决定,**评估纳入 Tier 3 rollout,不做 Tier 2**(label 质量离线诊断);baseline 直接用 [`task_a_new_smooth_800_new_norm_results.md`](../../history/experiments/task_a_new_smooth_800_new_norm_results.md) 的 SFT 数字,不单独重训 SFT。

---

## 5.5 V0 执行结果(2026-05-31,健全性核对)

| 检查项 | 结果 | 证据 / 后果 |
|---|---|---|
| **smooth_800 videos** | 🟢 **已修复** | 原 uc03 上 2433/2433 symlink 断链(重装后源删)。**已重建到 gf0**:rsync meta+parquet(base 811 + val 26 ep)→ `kai0/data/Task_A/self_built/A_new_smooth_800/{base,val}`,2511 个 video symlink 前缀 `/data/shared/dataset/KAI0/Task_A/base/`→`.../vis_base/` 重写,**0 缺失 0 broken**,抽样解析到真实 mp4。meta 一致(929,942 frames) |
| **源视频** | 🟢 | gf0 `vis_base/` 含全部 10 源日期,相机 `top_head/hand_left/hand_right` 对应 |
| **ViVa ckpt 结构(×2)** | 🟢 **均完好** | **official** `checkpoint_step_7000` 与 **DSM-r30** `/vePFS/zundong/robot` 各 825 tensors / **5.00B** / BF16,全 `video_model.wan_model.*`,架构一致。**value 经 denoise latent 帧产出(非标量 head),方向验证须跑完整推理(含 VAE)** |
| **ViVa value 方向** | 🟢 **已验证通过(official ckpt)** | 端到端推理跑通(模型加载→645/1949 帧→写 `prediction` 列)。**真实 t5 + in-distribution episode(05-09)上 corr=−0.885**,10 段 `0.881→0.127` 单调递减 → **value = 任务剩余比例(start≈1→end≈0,递减)**。**AWBC advantage 须符号翻转:`viva_advantage = value(t) − value(t+Δ)`**(R3 已 close)|

**V0 净结论(更新)**: 已就绪的 = 数据(smooth_800 自包含)✅ + **viva env(gf0 `/vePFS/tim/viva_env`,torch2.7.1+cu126/cuda✅/lerobot0.3.2/diffusers/deepspeed,flash-attn 非必需 graceful fallback)** ✅ + **lerobot-compat 视图**(相机 top_head/hand_left/hand_right→cam_high/cam_left_wrist/cam_right_wrist,`/vePFS/tim/viva_work/views/smooth800_base_vivaview`,ep0/1 实体隔离防污染)✅ + **state_stats**(从 811 parquet 算,`/vePFS/tim/viva_work/smooth800_state_stats.txt`)✅ + **patched config**(把 config 里过期的 `/vePFS-East/...` WAN 硬编码路径 sed 成 gf0 实际 `/vePFS/zundong/ViVa/weights`)✅ + ckpt 结构 ✅。

> **关于 `/vePFS-East`**:ViVa 的 config/脚本里硬编码了 `/vePFS-East/comp_robot/zundong/ViVa/...` 这类绝对路径,但本集群文件系统就是 **`/vePFS`**(无独立 vePFS-East 挂载),这些是从别处带来的**过期路径字符串**(WAN 权重路径已 sed 成 gf0 实际值)。

**V0 收尾 = ✅ 通过(official ckpt)**。补齐的依赖链(用户拷文件 + gf0 装包):`viva_model.py` + `viva_utils.py`(用户从原仓库补)→ `giga_datasets==1.0.0`(清华源)→ `flash-attn==2.7.4.post1` abiTRUE wheel(github 经代理断点续传 414MB,运行时硬依赖)。env 完整,5B 模型加载+推理全通。

🟡 **关键发现:ViVa value 质量逐 episode 波动(非系统性日期偏差)**。10 日期各取 1 代表 episode、真实 t5 推理,|corr(value, 进度)|:

| 日期 | ep | \|corr\| | | 日期 | ep | \|corr\| |
|---|---|---|---|---|---|---|
| 04-23 | 0 | **0.108** 🔴 | | 05-06 | 588 | **0.269** 🔴 |
| 04-25 | 192 | 0.806 🟢 | | 05-08 | 696 | 0.785 🟢 |
| 04-29 | 432 | 0.776 🟢 | | 05-09★训练 | 785 | 0.876 🟢 |

→ **不是 OOD-by-date**:05-06(离训练日期最近)反而差(0.269),04-25(更远)却好(0.806)。约 **2/3 episode 好(\|corr\| 0.78-0.88)、1/3 差(<0.3)**,是 per-episode 噪声。可视化:`/vePFS/tim/viva_work/viva_value_curves.png`。
→ **缓解策略(已定)**:全量打标后用 **`corr_filter.py`** 按 \|corr\|≥阈值(默认 0.5)过滤掉差 episode,只用好的喂 AWBC。三个模型同样处理。
→ **advantage 符号(V0 已定)**:value=剩余比例(递减),故 `viva_advantage = value(t) − value(t+Δ)`(脚本 `viva_value_to_advantage.py`,Δ=30)。

⚠️ **算力现实(动手前必读)**: 单卡 gf0 实测 ~0.6 s/帧;smooth_800 = 929,942 帧 → **~155 小时/模型,3 模型 ≈ 19 天**(单卡不可行)。全量 labeling **必须多卡机**(`inference_half_8gpu.py` 跨 GPU 切帧,或跨 GPU 切 episode):8 卡 ≈ 19h/模型。需在有 viva env + 多卡的机器跑(见 §7 开放问题)。

**工件落点(tim 可写)**: 脚本/视图/配置在 `/vePFS/tim/viva_work/`(`scripts/make_lerobot_compat_view.py`、`views/smooth800_base_vivaview`、`smooth800_state_stats.txt`、`config_taskA_official_gf0.yaml`、`run_infer_ep0.sh`);env 在 `/vePFS/tim/viva_env`(注:`/vePFS/zundong/ViVa` root-only 只读,故工件不放那)。

**资产分布(关键拓扑问题)**:

| 资产 | 位置 | 可达性 |
|---|---|---|
| smooth_800 meta+parquet | uc03 `/data/shared/ubuntu_old/...`(102M)| ✅(videos 断链)|
| 源视频 vis_base(10 日期)| **gf0** `/vePFS/.../Task_A/vis_base/` | ✅ 实体 |
| ViVa repo + ckpt7000 + WAN 权重 | **gf0** `/vePFS/zundong/`(共享 vePFS cnsh)| ✅ |
| ViVa conda env + `viva_model.py` + task_a t5/state_stats | **未随 `/vePFS/zundong/ViVa` 拷全**(config 里写的是过期 `/vePFS-East/...` 路径)| env 已在 gf0 重建✅；t5/state_stats 已自备✅；仍缺 `viva_model.py` |
| GPU | gf0 仅 1×A100(且现已占满 79938/81920)/ uc03 8×A800 / ViVa 机 H20 | 分散 |

**V0 结论**:数据 videos 是真阻塞但**可从 gf0 vis_base 重建**;ViVa 加载/方向验证卡在"viva env + 空闲 GPU + task_a t5/state_stats 不在 gf0"。→ 需先确定 **ViVa 在哪台机跑**(见 §7 开放问题 3 升级版),再续 V0 check 3 与 V1-V2。

**推荐落地拓扑(基于 V0)**:
1. **数据重建在 gf0**:把 smooth_800 meta+parquet 从 uc03 拉到 gf0,videos 按 symlink 编码的 `(date, episode)` 从 gf0 vis_base **实体重建** → 落到 `kai0/data/Task_A/A_new_smooth_800/`(用户已授权 copy 到本地)。产出一份自包含、co-located 于 ViVa repo 的 smooth_800。
2. **ViVa labeling 直接在 gf0 跑**(viva env 已在 gf0 `/vePFS/tim/viva_env` 重建,ckpt/WAN 权重/config 都在 `/vePFS/zundong/ViVa`):smooth_800 视图、state_stats 已就绪;补上 `viva_model.py` 即可。
3. **AWBC 训练在 uc03**(8×A800,数据本地)或 ViVa 机:labeling 完的带 task_index 数据集搬到训练机。

## 6. 风险与兜底

| # | 风险 | 概率 | 影响 | 兜底 |
|---|---|---|---|---|
| R1 | **跨集群训练**: ViVa labeling + 数据重建在 **gf0**(`/vePFS`);AWBC 8 卡训练拟在 **uc03**(无 vePFS、独立 4TB SSD,但 gf0 仅 1 卡)。labeling 与 training 不在同一机 | 中 | 中 | labeling 完只需把带 task_index 的 parquet+tasks.jsonl 从 gf0 搬到 uc03(走 TOS 中转或 rsync);数据小(~102M meta+parquet) |
| R0 | **smooth_800 videos symlink 断链**(uc 重装 tim→ubuntu,老 symlink 可能指向不存在的 tim 路径)| 中 | 高 | V0 先验证 base/val 能被 lerobot 加载 + 视频可读;断链则从 vis_base 源重建 symlink 或实体拷贝 |
| R8 | **Arm A 的 pi0-AE estimator ckpt 找不到/与 smooth_800 域不符** | 中 | 中 | V3b 先定位 `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` run ckpt;若无,用 KAI0 预标注 estimator 或退而用 `label_dagger_positive` 简化版作 Arm A(需在文档标注口径差异)|
| R2 | **相机/schema 不匹配**: ViVa 要 ALOHA 3-cam,Task_A 是 Agilex | 中 | 高 | V1 lerobot-compat 转换(ViVa ckpt 训练时已做过同样转换,脚本可复用);转换后抽样核对 cam key + state 维度 |
| R3 | **ViVa value 语义方向不确定**: findings.md 同时出现"fraction-remaining(递减)"与"progress(递增)"两种描述,DSM 变体又不同 | 中 | 中 | V0 在 3-5 episode 上画 value 曲线肉眼定方向;`viva_advantage` 符号据此设;再用 GT progress 验证 corr 为正 |
| R4 | **future_offset 不一致**(ViVa=30 vs pi0 AE=50)| 中 | 低 | Δ 主用 30(ViVa 原生),加 Δ=50 对照;离散化阈值两 Δ 各自按自身分布定 |
| R5 | **ViVa 推理太慢/太贵**(5B WAN,逐帧 denoise)| 中 | 中 | `inference_half_8gpu.py` 已按帧切片多卡并行;`num_inference_steps=1`;先在 200-ep 子集验证 pipeline 再全量 |
| R6 | **MAE 不敏感,Tier1 看不出差异** | 高 | 低 | 早就预期(历史教训);主判据放 **Tier3 rollout**,MAE 只作 sanity |
| R7 | **baseline 不可比**(老 ckpt 数据/seed/config 已漂移)| 中 | 高 | 不确定就 V4 重训一臂;两臂同时跑保证同环境同 vePFS I/O |

---

## 7. 关键开放问题(评审时确认)

1. **复用还是重训 baseline?** — `gf0_awbc_baseline_v2` 的 ckpt + eval 日志是否还在、是否与 `Task_A/advantage` 当前版本同源?能复用省 2-3 天。
2. **数据集选 advantage/ 还是 advantage_v2/?** — 推荐先小后大(§3)。
3. **全量 labeling 在哪台多卡机跑?** — gf0 仅 1 卡,全量 3 模型 ≈19 天不可行。需多卡(8 卡 ≈19h/模型)。但多卡机(uc01/02/03、Robot-North H20)无 viva env + ViVa repo → 要么把 viva env(`/vePFS/tim/viva_env`)+ repo + flash-attn wheel 同步过去,要么在那边重建。**待裁决**。
4. **第三个模型 = ?** — `/vePFS/zundong` 仅 2 个 ViVa ckpt(official + DSM-r30)。"三模型"的第三个推断为 **pi0-AE**(`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`,另一条 kai0 pipeline,需先定位其 estimator ckpt)。**待确认**。
4. **Tier 3 rollout 算力/真机窗口** — sim01 部署评估是否纳入本轮(决定性但占真机/sim 时间),还是先只做 Tier1+2 出快结论。

---

## 8. 落地命令骨架(待 V0 核对后填实参)

```bash
# ── V2: ViVa labeling(在 ViVa 所在机,viva conda env)──
cd /vePFS/zundong/ViVa
torchrun --nproc_per_node=8 inference_half_8gpu.py \
    --checkpoint /vePFS/zundong/checkpoint_step_7000 \
    --config config/train_viva-TaskA0509-baseline-bs192-0522.0528.yaml \
    --data_path <Task_A/advantage 的 lerobot-compat 视图> \
    --t5_embedding data/t5_task_a_0509v2.pt \
    --state_txt <state_stats.txt> \
    --num_inference_steps 1
# → parquet 多出 prediction(ViVa value)列

# ── V3: value→advantage→task_index ──
python tools/viva_value_to_advantage.py \        # 新增薄脚本
    --dataset <labeled view> --future-offset 30 \
    --out-col absolute_advantage                  # 写回复用列名
python kai0/stage_advantage/annotation/discretize_advantage.py \
    <dataset> --threshold <与Arm A同> --discretion-type binary \
    --advantage-source absolute_advantage --stage-nums 2

# ── V5: Arm B 训练(gf0,复制 baseline launcher 改 repo_id + exp_name)──
#   CONFIG=pi05_flatten_fold_awbc ; EXP_NAME=awbc_label_viva7k
#   data.repo_id → ViVa-labeled 数据集
bash train_scripts/kai/launch/run_awbc_label_viva7k_gf0.sh

# ── V6: 对比评估 ──
python train_scripts/kai/eval/eval_awbc_compare.py \
    --arm-a <baseline ckpt> --arm-b <viva ckpt>
```

---

## 9. 与现有文档的关系

- 本方案是 `awbc_implementation_plan.md` 的 **Stage 1-2 替换实验**:不动 Stage 4 训练,只把"产 advantage label 的模型"从 pi0-AE 换成 ViVa。
- 是对 `awbc_pi07style_experiment.md` 失败根因("label 信噪比低")的**正面攻击**:不再在 prompt 侧做花活(那条路已证死),而是从源头换一个更强的 value 模型。
- 若 ViVa 赢,后续可与 `awbc_v2_training_plan.md` 的数据扩充(base+dagger+mirror)叠加,形成 "ViVa-label × 全量数据" 的最终 AWBC 配方。
