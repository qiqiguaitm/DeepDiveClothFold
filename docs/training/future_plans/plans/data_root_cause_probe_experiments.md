# 数据问题排查实验系列 (Data Root-Cause Probe)

> **目的**: 用一系列**最小变量**的训练实验, 逐个证实/证伪 A_0423_0527 / 后期数据真机失败 (走停/犹豫/cloth loop + 拉取松手) 的数据侧假设。每个实验只改一个数据维度, 真机表现为终判 (offline MAE 不可信, 见下)。
>
> **状态**: 🔄 进行中 (Exp-1 训练+MAE ✅ 完成; Exp-1b 训练中; H1 终判待真机)
> **建立**: 2026-06-01
> **结果记录** ⭐: [`../../history/experiments/data_root_cause_probe_results.md`](../../history/experiments/data_root_cause_probe_results.md) — Exp-1 完整 MAE + best ckpt 打包 + Exp-1b 状态
> **关联**:
> - 根因分析: [`../../history/experiments/A_0423_0527_run_a_b_results.md`](../../history/experiments/A_0423_0527_run_a_b_results.md) §3.5 (offline SOTA 但真机差)
> - 反证 (纯后期单训也犯病): [`../../history/experiments/task_a_vis_curated_subset_experiments.md`](../../history/experiments/task_a_vis_curated_subset_experiments.md) (vis_5day_recent 单独训练同样真机 fail)
> - 数据侧 audit: [`../../analysis/vis_v2_full_data_audit.md`](../../analysis/vis_v2_full_data_audit.md) §0.NEXT-v7/v8
> - work 锚点: [`../../history/experiments/task_a_new_smooth_800_new_norm_results.md`](../../history/experiments/task_a_new_smooth_800_new_norm_results.md) (smooth_800 真机 work)

---

## 0. 背景 — 已坐实的现象与候选根因

**核心悖论**: 后期数据 (5-18 起) offline MAE 反而更低 (5day_recent cross-val @1=0.0086, A_0423_0527 @1=0.0073 = offline SOTA), 但**真机全部 fail**; 早期 smooth_800 (4-23~5-09) offline 略差 (@1=0.0089) 却**真机 work**。

**两类真机症状 (用户 2026-06-01 反馈)**:
1. **臂走走停停 / 犹豫时间长 / 来回重复 (cloth loop 循环)**
2. **夹爪夹取衣服后, 拉取过程中容易松手**

**已实测的数据侧签名 (smooth=work vs 所有后期段=fail, 见 A_0423_0527 results §3.5.4b)**:

| 段 | 真机 | ep中位长 | 静止帧%(<2e-3) | 抓取循环/ep | L_腕yaw σ ratio | L_腕yaw Δmean |
|---|---|---:|---:|---:|---:|---:|
| smooth 4-25~5-09 | ✅ work | **1091** | 32.7 | 3.9 | 1.00× | 0° |
| 5day核心 5-18~21 | ❌ fail | 1610 | 37.7 | 4.9 | 0.82× | +9.0° |
| 5-22 | ❌ fail | 1566 | 38.0 | 4.7 | 0.75× | +15.3° |
| 5-26 | ❌ fail | 1709 | 39.9 | 4.6 | 0.78× | +15.5° |
| 5-27 | ❌ fail | 1634 | 38.1 | 4.4 | 0.86× | +16.3° |

**候选根因 (待逐一排查)**:
- **H1 投放过程污染** (本系列 Exp-1): 后期 episode 开头都有一段长静止 (motion onset 中位 ~127 帧 = 投放等待), 后期 ep 整体长 50%。模型可能学到"开头长时间不动"→ 真机走停/犹豫。🟢 **真机初步成立** (用户 2026-06-02: no-release 明显改善) + 文献机理 (policy idling / idle-frame 过滤), 见 results §4。
- **H2 慢节奏/多停顿/反复重抓的示范风格** (整段轨迹, 非仅开头): 后期 ep 静止帧%↑、抓取循环↑ → BC 忠实模仿 → loop。
- **H3 gripper 校准漂移** (症状2 松手): 后期 R_grip 表征偏离真机 firmware (见 §3.5.4b)。
- **H4 wrist prior 收窄 + 腕yaw 漂移** (OOD): 后期单 operator → σ 0.75~0.86× + mean 漂 +15°。
- ❌ **已证伪 — 双模态冲突**: 曾假设 smooth(−8.5°)+后期(+7°) 左腕双峰致犹豫, 但**纯后期单模态的 5day_recent 单独训练也犯病** → 双峰不是主因 (最多是混合数据的附加放大项)。

> ⚠️ **方法学铁律**: 本系列**以真机为终判**。offline MAE 系统性反指 (慢/停顿轨迹逐帧误差低但真机灾难; gripper/wrist 问题被 12D arm 稀释)。每个实验出 ckpt 后必须真机测, 不能只看 MAE。

---

## Exp-1 — 裁掉投放过程 (验证 H1)

### 1.1 假设

后期 episode 开头的长静止段 = **投放衣物的等待过程** (机械臂静止, 操作员往工作台投放衣物)。这段"长时间不动"被 BC 学进策略 → 真机推理时模型在开头/中途也"等"→ 走停、犹豫、循环。

**裁掉投放过程后**, 若真机走停/犹豫显著改善 → H1 (投放污染) 成立; 若无改善 → H1 被排除, 转向 H2 (整段慢节奏) / H4 (wrist)。

### 1.2 数据集构建 — `A_0522_0526_no_release`

**源**: `kai0/data/Task_A/vis_base/{2026-05-22-v2, 2026-05-26-v2}` (各 100 ep, 共 200 ep)
**目标**: `kai0/data/Task_A/self_built/A_0522_0526_no_release/`

**为什么选这两天**:
- 都是 ❌ fail 后期段, 且都已实测含投放静止段 (见 §1.3)。
- 避开 5-22 gripper 重度过度归零干扰? **否** — 故意保留 gripper 现状, 本实验只动"投放裁剪"这**一个变量**, gripper/wrist 不变, 以隔离 H1。(gripper 留给后续 Exp 单独排查。)
- 5-27 暂不纳入, 留作"裁剪方案泛化性"的 hold-out。

### 1.3 投放段检测 — 已实测验证 ✅

用机械臂运动状态自动检测 motion onset (臂 12 维逐帧 |Δaction| 均值持续 > 阈值的首帧):

```
检测参数: thr=3e-3 (rad/帧), win=10 (持续 10 帧才算真运动), margin=15 帧
```

实测结果 (2026-06-01, 脚本 `/tmp/onset_scan.py` + `/tmp/onset_grip.py`):

| date | ep | motion onset 中位 | onset p10~p90 | 开头静止 gripper | 裁剪切点 cut 中位 | 砍掉占比 |
|---|---:|---:|---:|---|---:|---:|
| 2026-05-22 | 100 | 129 帧 | 106~164 | L=0.001 R=0.002 (闭合待命) | 114 帧 | 7.5% |
| 2026-05-26 | 100 | 126 帧 | 104~164 | L=0.001 R=0.001 | 112 帧 | 6.5% |

- **100% 的 episode 都有开头静止段** (onset 最小也 60+ 帧, 无一例外), 分布高度集中 (95% 落在 90~200 帧) → 投放假设**坐实**。
- 开头静止段 gripper ≈ 0 (闭合/未张开待命), 臂不动 → 确为投放等待, 非有效操作。
- 裁后总帧: 5-22 161k→149k, 5-26 176k→164k, 共 337k→313k (砍 ~7%)。

### 1.4 裁剪规则

对每个 episode:
1. 计算 `onset` = 臂运动起始帧 (上述检测)。
2. `cut = max(0, onset - margin)`, margin=15 帧 (保留一点投放结束前的余量, 避免切到操作起手)。
3. **parquet**: 删除前 `cut` 行, `frame_index` / `index` / `timestamp` 重排 (timestamp 从 0 重新累加, fps=30)。
4. **video** (per-episode mp4, 3 路 top_head/hand_left/hand_right): 用 `ffmpeg -ss <cut/30>s -i in.mp4 -c:v libx264 ... out.mp4` 重编码裁掉开头, **必须保证裁后视频帧数 == 裁后 parquet 行数** (逐 ep assert)。depth 路 (`observation.depth.top_head`) 同样处理。
5. **meta**: `episodes.jsonl` 的 `length` 改为裁后长度; `episodes_stats.jsonl` 重算 (或先复制, 训练前用 openpi `compute_norm_stats` 重算 norm_stats); `info.json` 的 total_frames/total_episodes/splits 更新。

> ⚠️ **视频对齐是本实验最大工程风险** — lerobot dataloader 按 frame_index 取视频帧, 裁开头后视频与 parquet 必须严格同步。**逐 ep 校验帧数一致**, 不一致则该 ep 报错跳过 + log。若 ffmpeg 重编码帧数不可控, 退路: 用 `-vf select` 按帧精确截取, 或保留视频不裁但在 parquet 里标记 valid 起始 (需 dataloader 支持, 风险更高 → 优先 ffmpeg 方案)。

### 1.5 检测脚本要点 (待写 `train_scripts/kai/data/build_no_release.py`)

```python
ARM = list(range(0,6)) + list(range(7,13))   # 12 臂维 (排除 dim6 L_grip, dim13 R_grip)
def motion_onset(action, thr=3e-3, win=10):
    da = np.abs(np.diff(action[:, ARM], axis=0)).mean(axis=1)  # (T-1,)
    run = 0
    for i, moving in enumerate(da > thr):
        run = run + 1 if moving else 0
        if run >= win:
            return i - win + 1   # 持续运动段的起始帧
    return len(action)           # 全程未动 (异常 ep, 应 log)
# cut = max(0, onset - 15)
```

参考现有 build 脚本惯例: [`train_scripts/kai/data/build_task_a_new_100.py`](../../../../train_scripts/kai/data/build_task_a_new_100.py) (lerobot v2.1 layout: data/chunk-000 + videos/chunk-000/{cam} + meta/{info,episodes,episodes_stats,tasks})。

### 1.6 训练配置

与 smooth_800 / A_0423_0527 对齐, 只让数据变 (单变量):

| 项 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_A_0522_0526_no_release` |
| Model | pi05 (`Pi0Config(pi05=True)`) |
| 框架 | JAX/Flax NNX (`scripts/train.py`) — 与 work 锚点 smooth_800 同框架 (避开 PyTorch 框架 gap, 见 pytorch_vs_jax postmortem) |
| Init | `mixed_1_clean` (与 smooth_800 work 锚点一致) |
| Dataset | `A_0522_0526_no_release` (~200 ep, ~313k frames 裁后) |
| Prompt | "Flatten and fold the cloth." |
| use_delta_joint_actions | False (absolute) |
| LR | Cosine, warmup=1k, peak=1.5e-5, decay=50k→1.5e-6 |
| EMA | 0.9999 |
| Steps / Batch | 40,000 / 128 (smooth_800 实测 40k 已 plateau, 省 20%) |
| norm_stats | 裁后**重算** (openpi compute_norm_stats), 不复用源 |
| 集群 | 单机 8 GPU (uc / cnsh / cnbj 视空闲, 见 submit-training-job) |

> ⚠️ **对照基线**: 必须有一个"**同两天数据但不裁投放**"的对照 (Exp-1b), 否则无法区分"裁投放生效" vs "只用 2 天数据/200ep 的规模效应"。Exp-1b = 同 config 同 init 同 step, 数据 = `A_0522_0526_raw` (不裁)。两者真机对比才是 H1 的干净判定。

### 1.7 判定

| Exp-1 (裁投放) 真机 vs Exp-1b (不裁) 真机 | 结论 |
|---|---|
| 走停/犹豫显著改善 | ✅ **H1 成立** — 投放静止段是症状①主因 (或主因之一)。推广: 对所有后期数据裁投放后重训 |
| 无改善 / 仍走停 | ❌ H1 排除 — 投放不是主因。转 Exp-2 验 H2 (整段慢节奏: 裁掉中途停顿段 + 抑制重抓循环) 或 H4 (wrist) |
| 改善但仍有残留 loop | ⚠️ H1 部分成立, 与 H2/H4 叠加 → 继续 Exp-2 |

> 松手 (症状2) **预期本实验不改善** (没动 gripper) — 若真机松手依旧, 正好佐证症状①②独立, 符合 §3.5 双层污染结论。

### 1.8 实现状态 (2026-06-01)

| 步骤 | 状态 | 备注 |
|---|---|---|
| 投放检测逻辑 (motion onset) | ✅ 已写 + dry-run 验证 | median cut 114 帧, 共裁 7.0%, 200 ep, 裁后 313,419 frames; min cut=44 (无未检出) |
| PyAV 裁剪帧对齐 | ✅ 3 ep 抽测通过 | 源 video==parquet (1755==1755), 裁后 video==裁后 parquet (1638==1638), 含 max-cut ep99 |
| build 脚本 | ✅ `train_scripts/kai/data/build_no_release.py` | 已并行化优化 (见下) |
| **`A_0522_0526_raw` (对照)** | ✅ **已生成** | 200 ep, 336,917 frames, 视频 symlink, 4 秒完成 |
| **`A_0522_0526_no_release` (裁)** | ⏳ **代码就绪, 未运行转换** | 用户要求先不转换; 跑一次约十几分钟 (600 mp4 重编码) |
| 两个训练 config 注册 | ✅ `config.py` | `pi05_flatten_fold_A_0522_0526_{no_release,raw}`, 40k step |
| norm_stats 重算 | ⏳ 待数据集就绪后跑 | 各自 `compute_norm_stats.py` |

### 1.9 转换脚本要点 (`build_no_release.py`)

一个脚本两模式 (`--mode no_release` 裁 / `--mode raw` 对照), 把 5-22+5-26 合成单个 lerobot-v2.1 数据集 (episode_index 重排 0..199)。

**核心逻辑**:
```python
ARM_DIMS = list(range(0,6)) + list(range(7,13))   # 12 臂维 (排 dim6 L_grip / dim13 R_grip)
THR, WIN, MARGIN = 3e-3, 10, 15
def motion_onset(action):                          # 臂运动起始帧
    da = np.abs(np.diff(action[:,ARM_DIMS],axis=0)).mean(axis=1)
    run = 0
    for i, moving in enumerate(da > THR):
        run = run+1 if moving else 0
        if run >= WIN: return i-WIN+1
    return len(action)                             # 全程未动 (异常 ep)
# cut = max(0, onset - MARGIN);  parquet 删 [0:cut] 行 + 重排 frame_index/index/timestamp/episode_index
# 3 路 RGB mp4 用 PyAV 重编码裁头, 逐 ep assert 裁后帧数 == 裁后 parquet 行数
```

**关键工程决定**:
- **只裁 3 路 RGB** (top_head/hand_left/hand_right) — 训练只读这 3 路。**depth (top_head_depth zarr) 不裁不带** (训练不用深度, 用户确认)。
- **源 meta 适配**: vis_base 的 `episodes.jsonl` 用 `episode_id` 字段且**无 `episodes_stats.jsonl`** → 脚本重新生成 self_built 所需的 `episodes_stats.jsonl` (标量 feature 的 min/max/mean/std/count) + 标准 `episodes.jsonl` (episode_index/tasks/length) + patch `info.json` (total_*/splits, 删 depth feature)。
- **视频重编码参数**: `libx264 crf=18 preset=veryfast threads=4` (近视觉无损, 比默认 medium 快 5-8×)。
- **并行**: 600 个 mp4 重编码用 `ProcessPoolExecutor` (14 worker × 4 线程 = 56 核), 逐 ep `_trim_job` 内部 assert 帧对齐。

**运行命令** (用户放行后):
```bash
cd /vePFS/tim/workspace/deepdive_kai0
kai0/.venv/bin/python train_scripts/kai/data/build_no_release.py --mode raw --symlink-video   # 已跑
kai0/.venv/bin/python train_scripts/kai/data/build_no_release.py --mode no_release            # 待跑
# 然后各自重算 norm_stats:
kai0/.venv/bin/python kai0/scripts/compute_norm_stats.py pi05_flatten_fold_A_0522_0526_no_release
kai0/.venv/bin/python kai0/scripts/compute_norm_stats.py pi05_flatten_fold_A_0522_0526_raw
```

### 1.10 ⚠️ 注意事项 / 风险 (起训前必看)

1. **init 路径本机缺失** — config 写的 `mixed_1_clean` (`/home/tim/local_ckpts/Task_A_init/mixed_1_clean/params`) **在本机 (cnsh) 不存在**, 它在 uc03/cnbj。**起训前必须**: 在目标训练机确认 init 就位, 或换本机有的 init (但换 init 会引入变量, 应与对照 Exp-1b 用同一 init)。norm_stats / build 不需要 init, 不受影响。
2. **对照 Exp-1b 不可省** — `A_0522_0526_raw` 已建好待用。若只跑 no_release 不跑 raw, "裁投放→真机变好" 会与 "只用 200ep 小数据" 混淆, H1 判不准。两者必须**同 init / 同 step / 同 hparams**, 仅数据裁不裁这一个变量。
3. **norm_stats 必须各自重算, 不可复用** — 裁投放改变了 action 分布 (开头静止段被删 → 静止帧占比下降), raw 与 no_release 的 norm_stats 不同, 且都不能用 A_0423_0527 / smooth 的。重算走官方 `compute_norm_stats.py` (经真实训练 dataloader, padding 到 32D + 分位数, 与训练一致)。
4. **视频帧对齐是硬约束** — `trim_video_pyav` 已逐 ep assert `裁后视频帧数 == 裁后 parquet 行数`, 任一不符即抛错中断 (不会静默产出错位数据)。若 PyAV 在某 ep 报错, 查该 ep 源 mp4 是否损坏。
5. **MARGIN=15 帧的取舍** — 保留 onset 前 15 帧 (0.5s), 避免切到"伸手起手"动作。若想更激进裁干净投放可调小, 但有切到有效操作起手的风险; 当前 15 是保守值。
6. **5-27 留作 hold-out** — 本实验只用 5-22+5-26。若 Exp-1 证明裁投放有效, 用 5-27 (同样 fail 的后期段, 未参与本实验) 验证裁剪方案的泛化性。

---

## Exp-2+ — 后续实验 (占位, 视 Exp-1 结果展开)

| Exp | 假设 | 数据操作 | 触发条件 |
|---|---|---|---|
| **Exp-2** | H2 整段慢节奏/多停顿 | 裁掉 episode **中途**长静止 run (>N 帧连续不动) + 截断末端反复重抓循环 | Exp-1 无改善或部分改善 |
| **Exp-3** | H4 wrist OOD | 仅用 wrist σ/mean 接近 smooth 的 ep 子集; 或对 L_腕yaw 做 smooth 对齐 | Exp-1/2 后仍走停 |
| **Exp-4** | H3 gripper (症状2) | 回滚到校准前原始 gripper (`/data2/visrobot_backup/.../Task_A_backup/base/`) 重训 | 症状②松手专项 |

---

## 附录 A — 检测/裁剪参数速查

```
fps                  = 30
action dims (14D)    = [0-5 L_arm, 6 L_grip, 7-12 R_arm, 13 R_grip]
                       L_腕yaw=dim3, R_腕yaw=dim10
motion onset 检测     = 臂12维 |Δa| 均值 > 3e-3 持续 10 帧 (THR/WIN)
裁剪切点 cut          = max(0, onset - 15)  (MARGIN=15)
实测 cut (build dry-run): median=114  mean=117.5  p90=149  max=248  min=44  共裁 7.0%
裁后规模 (no_release)  = 200 ep, 313,419 frames
对照规模 (raw)        = 200 ep, 336,917 frames (已建好)
视频编码             = libx264 crf=18 preset=veryfast threads=4, ProcessPool 14×4
源数据               = kai0/data/Task_A/vis_base/{2026-05-22-v2, 2026-05-26-v2}
```

## 附录 B — 文件/路径

| 项 | 路径 | 状态 |
|---|---|---|
| 源数据 | `kai0/data/Task_A/vis_base/{2026-05-22-v2,2026-05-26-v2}` | — |
| build 脚本 | `train_scripts/kai/data/build_no_release.py` | ✅ 已写 + 验证 |
| 裁后数据集 | `kai0/data/Task_A/self_built/A_0522_0526_no_release/` | ⏳ 待转换 (代码就绪) |
| 不裁对照 | `kai0/data/Task_A/self_built/A_0522_0526_raw/` (Exp-1b) | ✅ 已生成 |
| 训练 config | `config.py` → `pi05_flatten_fold_A_0522_0526_{no_release,raw}` | ✅ 已注册 (40k step) |
| norm_stats 脚本 | `kai0/scripts/compute_norm_stats.py <config_name>` | ⏳ 待数据集就绪后跑 |
