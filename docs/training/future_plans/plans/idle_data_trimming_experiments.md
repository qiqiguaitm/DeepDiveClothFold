# Idle(静止/投放)数据裁剪对训练的影响 — 汇总 + 分步规划

> **核心目的(本系列的真正主线,之前文档未点明)**: 验证**裁掉 episode 里的 idle(静止)帧能否让模型真机表现更好**。idle 帧分两类:① **前端**"投放等待"长静止段(机械臂不动、操作员往台上放衣服);② **中段**操作里的停顿/犹豫/反复。假设:idle 帧被 BC 忠实模仿 → 真机走停 / 犹豫 / cloth loop / 拉取松手。
> **分步走**: **Step 1 前端投放裁剪**(= 之前的 v2→v3 / no_release,已做)→ **Step 2 中段 idle 裁剪**(未来)→ Step 3 节奏归一(可选)。
> **状态**: Step 1 ✅ 真机初步成立(裁前端真机明显改善);Step 2 📋 规划。
> **建立**: 2026-06-07(**合并自** [`v2v3_data_window_scaling_experiments.md`](v2v3_data_window_scaling_experiments.md) + [`data_root_cause_probe_experiments.md`](data_root_cause_probe_experiments.md) H1,围绕 idle 主题重组)。
> ⚠️ **方法学铁律**: **真机为终判,offline MAE 系统性反指** —— idle 多的慢/停顿轨迹逐帧 teacher-forcing MAE 反而低,真机却灾难。MAE 仅用于确认训练健康 + 选 ckpt。

---

## 0. 为什么怀疑 idle 数据(已坐实的签名)

后期数据 offline MAE 更低却真机 fail;早期 smooth_800 offline 略差却真机 work。两者最大的数据侧差异之一就是 **idle/静止帧占比 + episode 长度**:

| 段 | 真机 | ep 中位长 | 静止帧 %(\|Δ\|<2e-3) | 投放 onset 中位 |
|---|---|---:|---:|---:|
| smooth 4-25~5-09(work) | ✅ | 1091 | **32.7%** | 短 |
| 后期 5-18~5-27(fail) | ❌ | 1600+ | **37~40%** | ~127 帧 |

→ 后期 ep **长 50%、静止帧多 5~7pp、开头投放等待长**。假设这些 idle 被学进策略 → 真机犹豫/走停。**本系列就是逐步裁掉 idle 验证之。**

---

## 1. idle 数据 + 裁剪机制(代码)

- **检测 motion-onset**(`build_no_release.py`):12D `|Δaction|` 均值持续 > `thr=3e-3`(rad/帧)达 `win=10` 帧的首帧 = 真运动起点;`margin=15` 帧。
- **前端裁**:`cut = max(0, onset - margin)`,删 parquet 行 `[0:cut]` + 同步裁 3 路 mp4(`assert video_frames == parquet_rows`)。
  - `--mode no_release`:对指定 2 天做前裁(单实验对照)。
  - `--per-date`(v3):对 `vis_base/v2/<date>` 每个 ep 前裁 → 输出 `vis_base/v3/<date>-v3`。
- **版本含义**: **v2 = 未裁原始;v3 = 前端投放已裁**。(只裁前端,中段 idle 仍在 → Step 2。)

---

## 2. Step 1 — 前端投放裁剪:已完成实验汇总

| 实验 | 数据 | 裁剪 | offline | 真机 | 结论 |
|---|---|---|---|---|---|
| **no_release probe**(data_root_cause Exp-1)| `A_0522_0526`(后期 fail 2 天 200ep)| no_release(前裁)vs raw(未裁)| best step20k MAE@1 **0.0160**(与 raw 持平,offline 看不出)| ✅ **no-release 明显改善**(用户 2026-06-02)| 🟢 **H1 投放污染初步成立**:前裁真机更流畅 |
| v2/v3 window 系列 | v2/5-18(Exp-A 未裁)· v3 窗口/全量/去脏(Exp-B/C/D 前裁)| 混合(见下注)| 各 horizon MAE | ⏳ 真机待做 | 量数据量/时窗/去脏,**顺带都在 v3 前裁基础上** |

> ⚠️ **诚实标注(之前的混淆)**: v2/v3 window 系列(Exp-A~D)其实**混了多个变量**(trim v2/v3 × 窗口 1日/多日/全量 × 去脏 Exp-D),**没把"idle 裁剪"作为单变量隔离**。真正干净的"裁 vs 不裁"单变量对照是 **no_release probe(Exp-1 no_release vs raw,同 2 天同量)** → 这才是 Step 1 的关键证据。Exp-A~D 的窗口/去脏结论与 idle 主题正交(Exp-D `t-20260607104053-jgbgw` 仍在跑,细节见原 v2v3 文档)。

**Step 1 结论**: **前端投放裁剪(no_release / v2→v3)真机明显改善**(H1 初步成立)。offline MAE 看不出(反指)。→ **idle 数据确实伤真机,裁了有用;下一步裁中段。**

---

## 3. ⭐ Step 2 — 中段 idle 裁剪(未来,单变量隔离设计)

> 前端裁已验证有效。下一步:清除 episode **中段**的停顿/静止/反复帧(操作过程里的犹豫),看真机能否进一步改善。

### 3.1 检测 + 裁剪(待实现)
- 检测中段静止段:`|Δaction|` 持续 `< thr`(静止)且**不在开头/结尾**的连续段 → 候选删除。
- ⚠️ **关键风险:轨迹连续性**。中段删段后,action chunk 可能**跨被删段跳变**(前后帧不连续)→ 反而引入 jump。需:(a) 删段后重排 frame_index + 重建 chunk;(b) 保留段边界的过渡帧(渐入渐出);(c) 只删"完全静止"(很严阈值),不动慢速有效动作。
- 参数从严起步(只删明显停顿),宁可少删不可删坏轨迹。

### 3.2 单变量实验设计(吸取 Step 1 教训:一次只动一个变量)
- 选**一份固定数据**(建议 smooth_800 或某后期段),**唯一变量 = 中段 idle 裁/不裁**,其余(参数/init/pipeline/部署)完全相同。
- 对照:`base`(仅前裁 v3)vs `mid_trim`(前裁 + 中段裁)。
- **真机为终判**:走停/犹豫时长、cloth loop 次数、完成率;offline 仅看健康。

### 3.3 分步执行
- **2a 小规模验证裁剪正确**:1~2 ep 上跑检测+裁剪,人工/可视化确认"只删静止、轨迹不断裂、chunk 不跳变"。
- **2b 单段单变量真机**:一段数据 base vs mid_trim,同参数训 → 真机 A/B。
- **2c 推广**:若 2b 真机改善 → 全量数据中段裁 + 重训。

---

## 4. 关联文档
- **合并来源(已并入本文档)**: `v2v3_data_window_scaling_experiments.md`(v2/v3 + 窗口 Exp-A~D,细节与 Exp-D 任务跟踪留存其中)。
- **前端裁根因线**: `data_root_cause_probe_experiments.md`(H1=投放污染=本系列 Step 1;H2 慢节奏 / H3 gripper 漂移 / H4 wrist OOD 是**其它**真机失败根因,不属 idle 主题,各自单查)。
- **裁剪脚本**: `train_scripts/kai/data/build_no_release.py`(`--mode no_release` 前裁 / `--per-date` v3 / 待加中段裁模式)。
- **work 锚点**: `task_a_new_smooth_800_new_norm_results.md`。
