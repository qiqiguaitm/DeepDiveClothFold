# LIBERO-10 全方案 per-task 成功率总表(2026-07-19)

全部为 **本机 egl 单worker、libero_10、50 trials/task、500 ep、25 worker、无 chunk 覆盖** 的同口径 eval。
数据源: `lmvla/lawam/results/eval_runs/libero/*/suites/libero_10/episodes.jsonl`(逐 episode 重算, 非转录)。

## 0. 任务名(列头对照)

| 列 | 任务 | 特征 |
|---|---|---|
| t0 | put both the alphabet soup and the tomato sauce in the basket | 双物入筐 |
| t1 | put both the cream cheese box and the butter in the basket | 双物入筐 |
| t2 | turn on the stove and put the moka pot on it | 开灶+放壶 |
| t3 | put the black bowl in the bottom drawer of the cabinet and close it | 抽屉+关闭 |
| t4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 双杯分盘 |
| t5 | pick up the book and place it in the back compartment of the caddy | 书入格 |
| **t6** | put the white mug on the plate and put the chocolate pudding to the right of the plate | **弥散/相对定位** ⭐区分度高 |
| t7 | put both the alphabet soup and the cream cheese box in the basket | 双物入筐 |
| **t8** | put both moka pots on the stove | **重复物体别名+精放** ⭐区分度高 |
| **t9** | put the yellow and white mug in the microwave and close it | **遮挡/长视野** ⭐区分度高 |

---

## 1. ⚠️ 先看噪声标尺 — 同一 checkpoint 重复评测

**这是解读下面所有数字的前提。** 同 ckpt、同协议、只是跑了两次:

| ckpt | eval | agg | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| armB_baseline **20000** | #1 (`v4`) | **96.4** | 98 | 100 | 100 | 96 | 98 | 100 | 84 | 100 | 100 | 88 |
| armB_baseline **20000** | #2 (`armB_baseline_20k`) | **94.4** | 94 | 98 | 98 | 98 | 100 | 100 | 78 | 100 | 100 | 78 |
| | **同ckpt差** | **2.0** | 4 | 2 | 2 | 2 | 2 | 0 | **6** | 0 | 0 | **10** |
| hintdrop015 **12500** | #1 | **94.6** | 92 | 98 | 96 | 96 | 100 | 100 | 88 | 100 | 84 | 92 |
| hintdrop015 **12500** | #2 | **94.4** | 94 | 98 | 96 | 100 | 100 | 100 | 90 | 96 | 84 | 86 |
| | **同ckpt差** | **0.2** | 2 | 0 | 0 | 4 | 0 | 0 | 2 | 4 | 0 | **6** |

**⚠️ 2026-07-19 晚 修正 — 上面这个"噪声标尺"的解释是错的。**

后续排查发现 `libero_eval_core.py:50` 有 `seed: int = 0`, 且 shell 脚本**从不透传 seed** →
**所有历史 eval 与并行重复全部 seed=0**。seed 同时驱动 `env.seed()` 与 `np.random.seed()`,
故同 seed 重复 = 同初始状态 = **近确定性**。实测: 机制② 4 路同 seed 聚合 std 仅 **0.14**,
而二项分布预期 σ≈**0.97** → 同 seed 并行重复**不是独立样本**, 误差棒是假的。

由此反推, armB 那对 96.4 vs 94.4:
- ckpt mtime = 07-14 23:24, **早于两次 eval**(07-15 02:17 / 07-17 21:49)→ 权重未被覆盖
- 同 seed=0、同 ckpt、同协议、近确定性 → **那 2.0pt 不可能是随机噪声**
- 最可能原因 = **两次 eval 之间的代码漂移**(07-15→07-17 期间改过 `lawam.py` / `flowmatching_expert.py` 等共享路径)
- ⇒ **跨日期的历史数字未必同口径**, 本表纵向比较需谨慎

**修复**: 已给 `run_libero_suite_benchmark.sh` 增加 `EVAL_SEED` 透传(不设则行为与旧版一致):
```bash
if [ -n "${EVAL_SEED:-}" ]; then eval_cmd+=(--args.seed "${EVAL_SEED}"); fi
```
此后重复评测必须**每路不同 seed** 才能得到真误差棒。已按此重提 24 路(见 §6)。

**理论噪声量级(变 seed 时适用):**
- **聚合**(n=500): 二项 σ≈1.0pt → **±2pt 内不可区分**
- **per-task**(n=50): 二项 σ≈4.2pt → **±8pt 内不可区分**

---

## 2. 全方案总表

按聚合排序。**灰色区间 = 与噪声不可区分**。

| # | 方案 | WM 类型 | steps | **agg** | t0 | t1 | t2 | t3 | t4 | t5 | **t6** | t7 | **t8** | **t9** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | armB baseline (eval#1) | LaWM t+7 | 20000 | **96.4** | 98 | 100 | 100 | 96 | 98 | 100 | 84 | 100 | **100** | 88 |
| 2 | 2Q + CFG1.5 | LaWM∥LMWM 2query + 推理放大ms | 12500 | **95.0** | 92 | 98 | 98 | 98 | 100 | 100 | 82 | 100 | 94 | 88 |
| 3 | E1 双尺度 (1 query) | LaWM∥LMWM 共享query | 12500 | **94.8** | 92 | 100 | 100 | 92 | 100 | 100 | 88 | 100 | 90 | 86 |
| 4 | Plan B 双 query | LaWM∥LMWM 2query | 12500 | **94.8** | 92 | 100 | 96 | 98 | 96 | 100 | 80 | 98 | 94 | **94** |
| 5 | hintdrop015 (eval#1) | LMWM 替换 | 12500 | **94.6** | 92 | 98 | 96 | 96 | 100 | 100 | 88 | 100 | 84 | 92 |
| 6 | armB baseline (eval#2) | LaWM t+7 | 20000 | **94.4** | 94 | 98 | 98 | 98 | 100 | 100 | 78 | 100 | **100** | 78 |
| 7 | hintdrop015 (eval#2) | LMWM 替换 | 12500 | **94.4** | 94 | 98 | 96 | 100 | 100 | 100 | 90 | 96 | 84 | 86 |
| 8 | **no-WM(纯VLA地板)** | **无 WM** | 12500 | **93.8** | 94 | 96 | 98 | 100 | 96 | 100 | 78 | 98 | 90 | 88 |
| 9 | adaptive horizon | LMWM 自适应目标 | 12500 | **92.8** | 96 | 98 | 98 | 100 | 94 | 100 | 90 | 100 | **78** | **74** |
| 10 | armM milestone (v5) | LMWM 替换 | 20000 | **92.2** | 98 | 96 | 100 | 94 | 100 | 98 | **68** | 100 | 86 | 82 |

> 另有 `lawam_libero_sft` = 98.0 但 **n=100(仅2任务)**,样本不足不可比,未入表。

---

## 3. ⛔ 本节(基于 seed=0 单次)的结论已被 §6 变 seed 实验推翻 — 仅存档

> 下面 3.1 的 ①③ 大体仍成立, **② 已被明确证伪**(t8 恰恰是全表方差最大的任务, 不是最稳的)。
> **以 §6 的变 seed 结果为准。** 保留此节仅为记录"单次评测会如何系统性误导"。

<details><summary>展开存档内容</summary>

**① 全部 WM 方案的聚合都落在 no-WM 地板的噪声带内。**(✅ 变 seed 后仍成立)

**② 唯一超出噪声的信号在 t8。**(❌ **已证伪** — 见 §6: armB 的 t8 变 seed 后是 88.5±12.0, 范围 [68,98];
所谓"两次重复都 100"是 seed=0 确定性造的假象)

**③ adaptive horizon 是唯一明确变差的方案。**(⚠️ 未做变 seed 复核, 但 t8/t9 双低方向与 §4.10 一致)

</details>

---

## 5. 机制②(diffusion-t 调度 ms drop)首次结果 — seed=0, 4 路

北京 `t-20260719142447-sddzr`, 4 路并行(**均 seed=0, 故 std 无效**, 见 §1 修正):

| | agg | t0 | t1 | t2 | t3 | t4 | t5 | **t6** | t7 | **t8** | **t9** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 机制② 均值 | **95.80** | 99.0 | 98.5 | 99.5 | 99.0 | 98.0 | 100 | 84.5 | 99.5 | **92.0** | 88.0 |

**对判据**(t8: LaWM=100 / LMWM替换=84):
**t8 = 92.0 — 高于 LMWM 替换的 84, 但未回到 LaWM 的 100。机制② 部分修复"远端 hint 伤精细落点", 未完全解决。**
聚合 95.80 是所有 12500 步方案里最高, 但需等 armB 的变 seed 均值出来才能做同口径对比。

---

## 6. ★ 定稿 — 24 路变 seed 重复评测(2026-07-19)

**本节取代 §2/§3/§5 的一切单次结论。** 北京 8×H20 并行, 每路独立 seed(`EVAL_SEED` 透传),
500 ep/路, 同口径。任务: `t-...164803-vksv5`(armB+noWM) / `t-...164806-vsdfl`(dual2q+hintdrop) / `t-...162220-rxpqs`(机制②×8)。

### 6.1 主表(均值±std)

| 方案 | n | **聚合** | t6 弥散 | t7 精度 | **t8 双壶** | t9 遮挡 |
|---|---|---|---|---|---|---|
| **机制②** tsched | 8 | **95.22**±0.91 | **85.0**±2.4 | 98.0±1.4 | 90.0±8.1 | 85.8±5.0 |
| dual2q 并联双query | 4 | 94.80±0.71 | **85.0**±1.7 | 99.5±0.9 | 88.5±9.8 | 87.5±3.8 |
| **no-WM 纯VLA** | 4 | 94.30±0.54 | 76.5±1.7 | 99.5±0.9 | **93.5**±1.7 | **89.5**±3.6 |
| hintdrop LMWM替换 | 4 | 94.25±0.84 | 82.5±2.2 | 98.5±1.7 | 87.5±5.5 | 88.5±4.8 |
| armB LaWM(t+7) | 4 | 93.60±1.12 | 78.5±3.0 | **100.0**±0.0 | 88.5±12.0 | 82.5±3.6 |

(std 为 population std;下面显著性用样本 std 与 SEM=s/√n 计算)

### 6.2 ~~✅ 唯一稳健的机理结论 — LMWM 显著提升 t6~~ ⛔ **已被本机复核推翻(2026-07-20)**

> **本节结论不再成立。** 本机独立复核(3 seed×2 臂×500ep,roadmap §4.18):t6 Δ=+7.33 但 **t=1.15**
> (dual2q t6 std **10.26**,逐路 70/90/84),**远非 p<0.001**。
> 且本节报的 t6 std **1.7 低于 per-task n=50 的二项下限 5.51** —— 欠散无物理机制可解释(P=0.037),
> 与本表 t8 的过散(9.8~12.0)不自洽 → 「变 seed」在 per-task 层面可能仍有残余相关性。
> §6.3「不得用 t8 作判据」同样依据不足:本机 t8 方差正常且效应最大(Δ=+9.33, t=2.65)。
> **保留下文仅为记录。以 roadmap §4.18 为准。**

| 方案 | t6 | vs no-WM |
|---|---|---|
| no-WM | 76.5±1.7 | — |
| armB LaWM | 78.5±3.0 | +2.0 (ns) |
| hintdrop | 82.5±2.2 | +6.0 |
| dual2q | 85.0±1.7 | **+8.5, t≈6.1, p<0.001** |
| 机制② | 85.0±2.4 | **+8.5, t≈6.3, p<0.001** |

**含 LMWM 的方案在 t6 系统性高 6~8.5pt, 且 std 仅 1.7~3.0** → 这是整个 V8 研究里
**第一个经得起变 seed 检验的机理结论: milestone 全局指引确实改善弥散/相对定位任务**。
(t6 = "把白杯放盘上 + 布丁放盘右侧", 需相对空间参照, 正是全局指引该起作用的地方。)

### 6.3 ❌ 证伪: "t8 是唯一诊断任务"

| 方案 | t8 均值±std | 范围 |
|---|---|---|
| armB LaWM | 88.5±**12.0** | [68, 98] |
| dual2q | 88.5±**9.8** | [72, 96] |
| 机制② | 90.0±**8.1** | [74, 100] |
| hintdrop | 87.5±5.5 | [78, 92] |
| no-WM | **93.5**±1.7 | [92, 96] |

**t8 恰恰是全表方差最大的任务(std 5.5~12), 各方案不可区分**;而 no-WM 反而最高最稳。
此前"LaWM t8=100 零方差 vs LMWM 84, 16pt 差距是唯一稳健机理"的结论**完全是 seed=0 假象**。
⇒ **不得再用 t8 作为任何机制的判据。**

### 6.4 机制② 判定 — 未证明有效

| 对比 | Δ聚合 | SE | t | 结论 |
|---|---|---|---|---|
| 机制② vs armB LaWM | +1.62 | 0.73 | 2.2 | 临界(p≈0.05) |
| 机制② vs no-WM | +0.92 | 0.46 | 2.0 | 临界(p≈0.07) |
| **机制② vs dual2q(自身基座)** | **+0.42** | 0.53 | **0.79** | **不显著** |

**机制②(diffusion-t 调度 ms drop)相对它的基座 dual2q 没有可测量增益。**
聚合虽是全表最高(95.22), 但差异被压在 ~1pt 而 SEM 就有 0.3~0.6 → LIBERO 上测不出。

### 6.5 意外发现: LaWM(t+7)可能是负贡献

armB(LaWM)聚合 **93.60**, 低于 no-WM 的 **94.30**; t9 更是 82.5 vs 89.5(SE≈2.5, t≈2.8)。
⇒ **相对纯 VLA, t+7 世界模型未见收益、在遮挡长视野任务上可能有害**;
WM 的正贡献集中体现在 **LMWM 的 t6**, 而非 LaWM。

### 6.6 方法论(硬约束, 后续必须遵守)

1. **重复评测必须变 seed**。`EVAL_SEED` 已透传(`run_libero_suite_benchmark.sh`)。
   同 seed 重复逐条一致率 95.6%, 有效 σ≈0.20;变 seed 恢复完整 σ≈0.97(**误差棒差 5 倍**)。
2. **聚合 SEM ≈ 0.3~0.6(n=4~8)** → 小于 ~1.5pt 的聚合差异不可声称。
3. **per-task 判据只用低方差任务**(t6 std 1.7~3.0 可用;t8 std 5.5~12 不可用)。
4. ckpt 跨集群搬运必须带 **`config.yaml` + `dataset_statistics.json`**(`read_mode_config` 的断言;
   只带 config.json 会让 8 路空跑到超时)。

复用模板: `train_scripts/kai/volc/libero_eval_2ckpt_x4_8h20.yaml`(双 ckpt×4路)/ `libero_eval_mech2_x8_seeds.yaml`(单 ckpt×8seed)。
停任务工具: `train_scripts/kai/volc/volc_job_stop.py`。

---

## 7. 复现

```bash
cd lmvla/lawam/results/eval_runs/libero
python - <<'EOF'
import json,glob
from collections import defaultdict
for f in sorted(glob.glob('**/episodes.jsonl', recursive=True)):
    s=defaultdict(list)
    for line in open(f):
        d=json.loads(line); s[d['task_id']].append(bool(d['success']))
    tot=sum(len(v) for v in s.values()); ns=sum(sum(v) for v in s.values())
    if tot<500: continue
    print(f"{ns/tot*100:5.1f} | "+" ".join(f"t{k}={sum(v)/len(v)*100:.0f}" for k in sorted(s))+f" | {f.split('/suites/')[0]}")
EOF
```
