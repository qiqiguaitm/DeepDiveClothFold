# AWBC × ViVa Value Model 对比实验方案

**创建时间**: 2026-05-31
**状态**: 📝 **PLAN — 待评审 / 未启动**
**负责人**: Tim
**一句话**: 把 AWBC pipeline 里产 advantage label 的 **pi0-based AdvantageEstimator** 换成新的 **ViVa video-generative value model**(`/vePFS/zundong/checkpoint_step_7000`),做一次**只换 label 来源、其余全锁死**的受控 A/B,看 ViVa 的 value 信号能否让同一套 AWBC 训出更好的 policy。

**上游/关联**:
- AWBC pipeline 全貌 → [`../../../deployment/strategy/awbc_implementation_plan.md`](../../../deployment/strategy/awbc_implementation_plan.md)(4-step RECAP)
- 历史失败教训 → [`awbc_pi07style_experiment.md`](awbc_pi07style_experiment.md)(π0.7-style，全失败，根因:demo-only advantage 方差 η²≈3%、prompt 信号弱)
- 已实施的 v2 数据扩充 → [`awbc_v2_training_plan.md`](awbc_v2_training_plan.md)(base+dagger+mirror，12,024 ep)
- ViVa 论文 → [arXiv 2604.08168](http://arxiv.org/abs/2604.08168) / 仓库 `/vePFS/zundong/ViVa`(GigaAI-research/ViVa)
- ViVa value 模型 = `/vePFS/zundong/checkpoint_step_7000`(WAN2.2-TI2V-5B backbone，在 `task_a_0509v2_lerobot_compat` 上训到 step 7000)

---

## 0. 为什么这个对比有意义(动机)

历史 AWBC 全线失败的**根因不在 AWBC 训练侧,而在 advantage label 侧**(见 `awbc_pi07style_experiment.md` 第十节):

1. pi0-AdvantageEstimator 的 `absolute_value` 与 GT progress corr=0.896(本身不错),
2. 但 AWBC 真正用的是 `absolute_advantage = absolute_value(t+50) − absolute_value(t)`(**二阶差分**),差分把噪声放大 → corr 掉到 ~0.3-0.4,
3. demo-only 数据 advantage 方差本就只有 η²≈3%(弱-中),再叠加噪声 → prompt 信号弱到模型直接学会忽略。

**ViVa 的卖点正好打这个痛点**:它用预训练视频生成器(WAN)的时空先验,联合预测"未来 proprioception + 标量 value",把 value 估计 grounding 在**预期的 embodiment dynamics** 上,而不是 pi0 那种 static-snapshot 回归。论文实测 ViVa 接入 RECAP 在真机 box assembly 上有 substantial improvement,且 value 曲线更可靠(更贴合真实任务进度)。

→ **假设**: ViVa value 信号的信噪比 > pi0-AdvantageEstimator,因此**同一套 AWBC 训练**用 ViVa label 应当训出 ≥ pi0-label baseline 的 policy。这是本实验要证伪/证实的核心命题。

---

## 1. 实验设计:受控 A/B(单变量)

**唯一变量 = advantage label 的来源**。其余一切(训练数据 episode 集合、AWBC config、超参、seed、discretize 方式、eval val split、评估协议)全部锁死。

| | **Arm A (baseline)** | **Arm B (ViVa)** |
|---|---|---|
| Label 来源 | pi0-AdvantageEstimator `absolute_advantage` | ViVa value → `viva_advantage = value(t+Δ) − value(t)` |
| Label 产出工具 | `kai0/stage_advantage/annotation/eval.py` | `ViVa/inference_half_8gpu.py` |
| 离散化 | `discretize_advantage.py --discretion-type binary` | **同一脚本、同一阈值** |
| 训练 config | AWBC config(prompt_from_task）| **同一 config**,仅 `repo_id` 指向 ViVa-labeled 数据集 |
| **基准数据集** | **`A_new_smooth_800`(811 ep)** | **同一份** |
| **Init(warm-start)** | **smooth_800 SFT 49999 ckpt** | **同一 init** |
| 训练超参 | batch=128, fsdp=8, num_workers=64, 同 seed | **完全相同** |
| 训练集群 | **uc03**(8×A800-80GB)| **同一集群** |
| Eval | smooth_800/val(26 ep), 同 MAE@1/10/50 协议 | **完全相同** |

> **基准 / init 定档(2026-05-31 核实)**:
> - 数据集 = `uc03:/data/shared/ubuntu_old/data/Task_A/A_new_smooth_800/{base,val}`(811 ep base + 26 ep val,Agilex 三相机 top_head/hand_left/hand_right + state-14,**纯 SFT、tasks.jsonl 仅 1 条 prompt → 两臂都需从零打 advantage 标**)
> - Init = `uc03:/data/shared/ubuntu_old/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_smooth_800_new_norm/task_a_new_smooth_800_new_norm/49999/params`
> - **参考基准数字**(该 SFT run,见 [`../../history/experiments/task_a_new_smooth_800_new_norm_results.md`](../../history/experiments/task_a_new_smooth_800_new_norm_results.md)): MAE@1=**0.0089** / @10=0.0221 / @25=0.0404 / @50=0.0636,step 40k 起 plateau。
> - **为什么 warm-start 而非 pi05_base 冷启**:AWBC 的前提就是"SFT 已 plateau 后做 frame-level 加权精修"(见 [`../../../deployment/strategy/awbc_implementation_plan.md`](../../../deployment/strategy/awbc_implementation_plan.md) §2)。从已收敛的 0.0089 policy 续训,(a) 任何低于 0.0089 的改进都干净归因到 advantage 加权,(b) 续训步数少(10-20k 即可),迭代快。

> **为什么必须单变量**:历史 π0.7 实验一次改了 3-4 个东西(n_slices+stage-aware+dropout),失败后无法归因。本次只改 label 源,任何 A/B 差异都能干净归到 ViVa 上。

**两条提交臂的命名**(checkpoint/exp_name 绑定,提前定好避免覆盖):
- Arm A: `exp_name=awbc_label_pi0ae`(若已有 `gf0_awbc_baseline_v2` ckpt 可直接复用作 A,见 §5 决策点)
- Arm B: `exp_name=awbc_label_viva7k`

---

## 2. Pipeline 映射:ViVa value → AWBC task_index

AWBC 训练侧(Stage 4)完全不动,它只认数据集里每帧的 `task_index` + `meta/tasks.jsonl` 的 prompt。我们要做的是**用 ViVa 重新生成 task_index 这一列**,流程对齐现有 Stage 2→3:

```
现有 (Arm A):
  dataset ── eval.py (pi0 AE) ──► +列 absolute_value / absolute_advantage
          ── discretize_advantage.py (binary, percentile threshold) ──► task_index ∈ {0,1} + tasks.jsonl
          ── pi05_flatten_fold_awbc 训练

ViVa (Arm B):
  dataset(lerobot-compat 视图) ── inference_half_8gpu.py (ViVa 7k) ──► +列 prediction (=ViVa value)
          ── [新增薄脚本] viva_value → viva_advantage = value(t+Δ) − value(t)  ──► 写回 absolute_advantage 列
          ── discretize_advantage.py (binary, 同阈值) ──► task_index ∈ {0,1} + tasks.jsonl(同 prompt 文本)
          ── pi05_flatten_fold_awbc 训练 (repo_id 换成 ViVa-labeled 数据集)
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

**Tier 1 — 离线 in-training eval(必做,便宜)**:
- 同 val split,Arm A vs Arm B 的 `mae_joint_{1,10,50}` / `mae_grip_{1,10,50}` 逐 checkpoint 对比曲线。
- 参照已有 baseline 数字:`gf0_awbc_baseline` Eval@21000 `mae_joint_1=0.0048 / @10=0.0084 / @50=0.0125`。
- ⚠️ **注意**: 纯 action MAE 对 AWBC 不够敏感(positive-prompt 推理下两臂 MAE 可能很接近,历史就是这样)。MAE 只作 sanity,不作主判据。

**Tier 2 — Advantage label 质量离线诊断(必做,直接验证 ViVa 卖点)**:
- 在 held-out episode 上,ViVa value vs GT `progress(t)` 的 per-episode corr / R²,直接对标 pi0 AE 的 corr=0.896。
- 更关键:`viva_advantage`(差分后)vs GT progress-rate 的 corr,对标 pi0 AE 差分后的 ~0.3-0.4。**这是 ViVa 是否真的更好的最直接证据**,即使 policy 没拉开也能给结论。
- η²(组间/组内 action 方差比):ViVa-binary 分桶后是否 > pi0 的 3.1%。

**Tier 3 — Rollout(决定性,贵)**:
- sim01 部署两臂 ckpt,positive-prompt 推理,跑 Task_A flatten-fold rollout。
- 指标:成功率、完成帧数(episode 长度)、关键 sub-phase(抓→对折)通过率。
- 这是 AWBC 真正的目标指标(论文也是看 throughput / 成功率,不是 MAE)。

**判据**:
- ViVa **明确赢** = Tier 2 advantage corr 显著高 **且** Tier 3 rollout 成功率/throughput ≥ baseline → 写结论,推 advantage_v2 规模化。
- ViVa **打平** = Tier 2 赢但 Tier 3 ≈ baseline → 说明 demo-only 数据已触天花板,转 dagger/失败段数据(ViVa 时空先验在 OOD 上才有空间)。
- ViVa **输** = Tier 2 都不赢 → ViVa value 在 Task_A 域没有信噪比优势,记录归档,不再投入。

---

## 5. Phase 拆分

| Phase | 任务 | 工作量 | 关键文件/命令 | 前置 |
|---|---|---|---|---|
| **V0** | **可达性 + 健全性核对**: ① smooth_800 的 videos 是否 symlink 断链(重装后)、base/val 可被 lerobot 正常加载;② ViVa env(`viva` conda)+ ckpt 7000 在哪台机可跑、能加载;③ ViVa value 语义方向(递增/递减)在 smooth_800 的 3-5 ep 上肉眼确认(画 value 曲线) | 0.5 天 | `ViVa/scripts_repro/plot_value_curve.py` / `inference_half_8gpu.py` | — |
| **V1** | **数据搬运 + lerobot-compat 视图**: smooth_800 在 uc03,ViVa 在 vePFS-East → 把 smooth_800(含 videos 实体,非 symlink)同步到 ViVa 所在机;转成 ViVa 视图(top_head→cam_high / hand_left→cam_left_wrist / hand_right→cam_right_wrist + state-14 + 生成 state_stats.txt + task_a t5 embedding) | 0.5-1 天 | 复用 ViVa 侧 `task_a_0509v2_lerobot_compat` 转换脚本 | V0 |
| **V2** | **ViVa labeling**: 在 smooth_800 视图上跑 ViVa 推理,产 `prediction`(ViVa value)列 | 0.5 天(多卡 H20）| `torchrun --nproc_per_node=N inference_half_8gpu.py --checkpoint /vePFS/zundong/checkpoint_step_7000 --config config/train_viva-TaskA0509-baseline-bs192-0522.0528.yaml --data_path <smooth_800 视图> --t5_embedding data/t5_task_a_0509v2.pt --state_txt <state_stats.txt>` | V1 |
| **V3a** | **Arm B label**: 薄脚本 `viva_value→viva_advantage=value(t+Δ)−value(t)`(Δ=30 主、50 对照)写回 `absolute_advantage`;`discretize_advantage.py` binary 离散化 → task_index + tasks.jsonl;搬回 uc03 | 0.5 天 | 新脚本 + `kai0/stage_advantage/annotation/discretize_advantage.py` | V2 |
| **V3b** | **Arm A label(并行)**: 定位/确认 pi0-AdvantageEstimator ckpt(`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`),用 `eval.py` 在 smooth_800 上产 `absolute_advantage`;**同一 `discretize_advantage.py` 同阈值** → task_index | 0.5-1 天(含 AE 推理)| `kai0/stage_advantage/annotation/eval.py` | V0(AE ckpt 就位)|
| **V4** | **Arm A 训练(baseline 重训)**: AWBC config,`repo_id`=smooth_800-pi0AE-labeled,**init=smooth_800 49999/params**,`exp_name=awbc_smooth800_pi0ae`,uc03 8×A800,batch=128/fsdp=8/nw=64,续训 ~15-20k step | 1-1.5 天 | uc03 launcher(套 `run_uc03_*` 模板)| V3b |
| **V5** | **Arm B 训练**: 同 config/init/超参,`repo_id`=smooth_800-ViVa-labeled,`exp_name=awbc_smooth800_viva7k` | 1-1.5 天 | 复制 V4 launcher 改 2 行(repo_id+exp_name)| V3a, (V4 并行)|
| **V6** | **三层评估 + 结论**: Tier1 MAE 曲线(对照 0.0089 基准）+ Tier2 label 质量诊断 + Tier3 sim01 rollout;写 results.md + 更新 master history | 1-2 天 | `train_scripts/kai/eval/eval_awbc_compare.py` + sim01 部署 | V4, V5 |

**总工作量**: **~6-8 天**(warm-start + smooth_800 小数据,labeling + 双臂续训 + eval;两臂训练可在 uc03 串行或借 uc01/02 并行)。

---

## 5.5 V0 执行结果(2026-05-31,健全性核对)

| 检查项 | 结果 | 证据 / 后果 |
|---|---|---|
| **smooth_800 videos** | 🟢 **已修复** | 原 uc03 上 2433/2433 symlink 断链(重装后源删)。**已重建到 gf0**:rsync meta+parquet(base 811 + val 26 ep)→ `kai0/data/Task_A/self_built/A_new_smooth_800/{base,val}`,2511 个 video symlink 前缀 `/data/shared/dataset/KAI0/Task_A/base/`→`.../vis_base/` 重写,**0 缺失 0 broken**,抽样解析到真实 mp4。meta 一致(929,942 frames) |
| **源视频** | 🟢 | gf0 `vis_base/` 含全部 10 源日期,相机 `top_head/hand_left/hand_right` 对应 |
| **ViVa ckpt 结构** | 🟢 **完好** | `checkpoint_step_7000/model.safetensors`:825 tensors / **5.00B 参数** / BF16,全 `video_model.wan_model.*`。**value 经 denoise latent 帧产出(非标量 head),方向验证必须跑完整推理(含 VAE)** |
| **ViVa value 方向** | ⏳ **阻塞** | 需 viva conda env + task_a 的 t5/state_stats —— **均在 /vePFS-East,gf0 挂不到**(config `train_viva-TaskA0509-official-0521.0225.yaml` 在 gf0✅,WAN 权重在 gf0✅,但 env 本体 + `t5_task_a_0509v2.pt` + state_stats 不在)。gf0 现 GPU 0 空闲(~60G free)|

**V0 净结论**: 数据 videos 阻塞 **已解除**(smooth_800 在 gf0 自包含可用)。剩唯一阻塞 = **ViVa env 不在 gf0**,二选一:(a) 在 gf0 重建 viva env(requirements.txt + WAN T5/VAE 权重都在 gf0,需 conda create py3.11.10 + flash-attn 编译 + regen t5_task_a + state_stats,~1h);(b) 在真正的 ViVa 训练机(挂 /vePFS-East + 8×H20 + env 就绪)上跑 labeling。

**资产分布(关键拓扑问题)**:

| 资产 | 位置 | 可达性 |
|---|---|---|
| smooth_800 meta+parquet | uc03 `/data/shared/ubuntu_old/...`(102M)| ✅(videos 断链)|
| 源视频 vis_base(10 日期)| **gf0** `/vePFS/.../Task_A/vis_base/` | ✅ 实体 |
| ViVa repo + ckpt7000 + WAN 权重 | **gf0** `/vePFS/zundong/`(共享 vePFS cnsh)| ✅ |
| ViVa conda env + task_a t5/state_stats | **/vePFS-East**(ViVa 训练机)| ❌ gf0 看不到 |
| GPU | gf0 仅 1×A100(且现已占满 79938/81920)/ uc03 8×A800 / ViVa 机 H20 | 分散 |

**V0 结论**:数据 videos 是真阻塞但**可从 gf0 vis_base 重建**;ViVa 加载/方向验证卡在"viva env + 空闲 GPU + task_a t5/state_stats 不在 gf0"。→ 需先确定 **ViVa 在哪台机跑**(见 §7 开放问题 3 升级版),再续 V0 check 3 与 V1-V2。

**推荐落地拓扑(基于 V0)**:
1. **数据重建在 gf0**:把 smooth_800 meta+parquet 从 uc03 拉到 gf0,videos 按 symlink 编码的 `(date, episode)` 从 gf0 vis_base **实体重建** → 落到 `kai0/data/Task_A/A_new_smooth_800/`(用户已授权 copy 到本地)。产出一份自包含、co-located 于 ViVa repo 的 smooth_800。
2. **ViVa labeling 在 ViVa 训练机**(有 viva env + H20 + /vePFS-East):需把重建好的 smooth_800(或其 lerobot-compat 视图)同步过去;t5_task_a/state_stats 复用 ViVa 训练时的产物。
3. **AWBC 训练在 uc03**(8×A800,数据本地)或 ViVa 机:labeling 完的带 task_index 数据集搬到训练机。

## 6. 风险与兜底

| # | 风险 | 概率 | 影响 | 兜底 |
|---|---|---|---|---|
| R1 | **跨集群**: smooth_800 + AWBC 训练在 **uc03**(无 vePFS、独立 4TB SSD);ViVa env + ckpt 在 **vePFS-East / `/vePFS/zundong`**。labeling 与 training 不在同一机 | 高 | 中 | smooth_800 仅 102M meta+parquet(videos 是 symlink)→ 先把 videos 实体化再 rsync 到 ViVa 机做 labeling;labeling 完只需把带 task_index 的 parquet+tasks.jsonl 搬回 uc03(走 TOS 中转或直接 rsync) |
| R0 | **smooth_800 videos symlink 断链**(uc 重装 tim→ubuntu,老 symlink 可能指向不存在的 tim 路径)| 中 | 高 | V0 先验证 base/val 能被 lerobot 加载 + 视频可读;断链则从 vis_base 源重建 symlink 或实体拷贝 |
| R8 | **Arm A 的 pi0-AE estimator ckpt 找不到/与 smooth_800 域不符** | 中 | 中 | V3b 先定位 `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` run ckpt;若无,用 KAI0 预标注 estimator 或退而用 `label_dagger_positive` 简化版作 Arm A(需在文档标注口径差异)|
| R2 | **相机/schema 不匹配**: ViVa 要 ALOHA 3-cam,Task_A 是 Agilex | 中 | 高 | V1 lerobot-compat 转换(ViVa ckpt 训练时已做过同样转换,脚本可复用);转换后抽样核对 cam key + state 维度 |
| R3 | **ViVa value 语义方向不确定**: findings.md 同时出现"fraction-remaining(递减)"与"progress(递增)"两种描述,DSM 变体又不同 | 中 | 中 | V0 在 3-5 episode 上画 value 曲线肉眼定方向;`viva_advantage` 符号据此设;再用 GT progress 验证 corr 为正 |
| R4 | **future_offset 不一致**(ViVa=30 vs pi0 AE=50)| 中 | 低 | Δ 主用 30(ViVa 原生),加 Δ=50 对照;离散化阈值两 Δ 各自按自身分布定 |
| R5 | **ViVa 推理太慢/太贵**(5B WAN,逐帧 denoise)| 中 | 中 | `inference_half_8gpu.py` 已按帧切片多卡并行;`num_inference_steps=1`;先在 200-ep 子集验证 pipeline 再全量 |
| R6 | **MAE 不敏感,Tier1 看不出差异** | 高 | 低 | 早就预期(历史教训);主判据放 Tier2 label-corr + Tier3 rollout,MAE 只作 sanity |
| R7 | **baseline 不可比**(老 ckpt 数据/seed/config 已漂移)| 中 | 高 | 不确定就 V4 重训一臂;两臂同时跑保证同环境同 vePFS I/O |

---

## 7. 关键开放问题(评审时确认)

1. **复用还是重训 baseline?** — `gf0_awbc_baseline_v2` 的 ckpt + eval 日志是否还在、是否与 `Task_A/advantage` 当前版本同源?能复用省 2-3 天。
2. **数据集选 advantage/ 还是 advantage_v2/?** — 推荐先小后大(§3)。
3. **ViVa labeling 在哪台机跑?** — ViVa env 在 `/vePFS-East/comp_robot/.../conda_envs/viva`,需确认那台机的 GPU/盘可用,以及能不能访问到要标的 Task_A 数据(否则先把数据集 rsync 过去)。
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
