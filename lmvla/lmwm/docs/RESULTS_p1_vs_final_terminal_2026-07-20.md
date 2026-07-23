# P1 支线 vs 终版 arch:最终 milestone 处理的同空间头对头(2026-07-20)

> **结论先行**:LMWM 有**两条平行管线**。P1 支线丢弃最终 milestone 的全部帧,是 task「白杯+布丁」失效的根因;
> 终版 arch(`build_pairs_abl`)本就 Viterbi 分割 + 最终段 self-loop,**同一任务、同一 DINOv3-base 编码空间下 lift 反而全场最高**。
> 本文同时**证否**了此前两个被广泛沿用的假设(见 §4)。
>
> 修正对象:[`RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md`](RESULTS_lmwm_vs_lawam_libero10_2026-07-15.md) §未训练尾巴分析。

---

## 1. 两条管线的差异(代码事实)

| | 分割算法 | 最终 milestone | 入口 |
|---|---|---|---|
| **终版 arch** | `viterbi_forward`(带转移代价) | **self-loop 建对** | `train_multitask.py:242` → `train_ablation.build_pairs_abl` |
| **P1 支线** | `argmin` + 中值 w=5 + `maximum.accumulate` | **`line 69: if m+1 in first` 全部丢弃** | `p1_libero_milestone_pairs.py`(独立脚本,**未 import** `build_pairs_abl`) |

P1 是为 LaWM 头对头单独写的 DINOv3-base 空间支线,因为没有复用 `build_pairs_abl`,退化成了更弱的标签构造。
**两处缺陷叠加:最终 milestone 又大(欠分割)又空(被丢弃)。**

---

## 2. 同任务头对头(均为 DINOv3-base 空间)

任务:**「put the white mug on the plate and put the chocolate pudding to the right of the plate」**
(终版 `libero_10` = **idx 8**;P1 `merged-40` = **idx 1**)

| 管线 | 最终段处理 | 未训练尾巴 | 指标 |
|---|---|---|---|
| **P1** | 丢弃 | **60%** | 末端 lift **−0.098** |
| **终版** | self-loop | 0(有监督) | lift **+0.0655** ← **10 任务中最高** |

### 终版 per-task(`job1_final_pertask.json`)

| task | deploy | persist | lift | 描述 |
|---|---|---|---|---|
| **8** | 0.9529 | 0.8874 | **+0.0655** | **白杯+布丁** |
| 2 | 0.9571 | 0.9007 | +0.0564 | 杯进微波炉 |
| 6 | 0.9535 | 0.9021 | +0.0514 | cream cheese + butter |
| 9 | 0.9655 | 0.9194 | +0.0461 | book → caddy |
| 5 | 0.9597 | 0.9156 | +0.0441 | soup + tomato sauce |
| 4 | 0.9598 | 0.9166 | +0.0432 | soup + cream cheese |
| 7 | 0.9604 | 0.9179 | +0.0425 | 双杯双盘 |
| 3 | 0.9595 | 0.9239 | +0.0356 | 两 moka 壶 |
| 1 | 0.9544 | 0.9226 | +0.0318 | bowl → drawer |
| 0 | 0.9546 | 0.9265 | +0.0281 | stove + moka |
| **ALL** | 0.9567 | 0.9123 | **+0.0444** | |

**10 个任务全部正 lift。** 缺陷可归因到 **P1 的标签构造**,不是架构、不是编码空间。

---

## 3. P1 线 LIBERO-40 普遍性(`job2_libero40_taillift.json`)

- **37/40 任务(92%)尾巴 lift 为负**
- 均值:`tail_frac=0.22`,**`tail_lift=−0.0618`** vs **`trained_lift=+0.0124`**,`end_lift=−0.0784`
- task 1(白杯+布丁)`tail=0.60`、`end_lift=−0.098`

---

## 3.5 同口径闭合(全帧对末帧,两线可直接相减)

`job3_final_allframes.json` —— 终版模型跑 §3 的同一口径:

| 位置 | P1 lift | **终版 lift** | 差 |
|---|---|---|---|
| 0-10% | +0.0550 | +0.0060 | P1 占优 |
| 40-50% | +0.0280 | +0.0280 | 持平 |
| 60-70% | 0.0000 | **+0.0237** | 终版 |
| 80-90% | −0.0410 | **+0.0327** | 终版 |
| 90-100% | **−0.0960** | **−0.0102** | 终版 |
| **平均 end_lift** | **−0.0784** | **−0.0110** | **终版好 7×** |

**两点读数:**

1. **末端负 lift 是结构性的,两线都负** —— 终版 90-100% 也是 −0.0102。这**印证了 §4.2** 的证否:`persist→1.0` 时任何生成都赢不过恒等。**不能把末端负 lift 当作缺陷证据。**
   > ⚠️ **2026-07-20 补充修正**:此判断只对**无终端监督**的标签成立。加 CRAVE per-task 双锚(终点锚→1)后,末端 lift 可做到 **+0.0043±0.0019**(非负)——
   > 所以末端负 lift 是**可修的**,不只是度量假象。见 [`RESULTS_crave_dualanchor_lmwm_2026-07-20.md`](RESULTS_crave_dualanchor_lmwm_2026-07-20.md) §4。
2. **但 P1 的未训练区(60-90%)是真塌** —— P1 从 0.000 掉到 −0.041,**终版同区间稳在 +0.024 ~ +0.033**。这才是两条线的实质差距,且**恰好落在 P1 丢弃最终 milestone 的区间**。

**反直觉附注**:终版的训练覆盖只有 **2-5%**(`seglast` 模式每段仅 1 对,~6 对/250 帧),
P1 是**逐帧建对**的稠密覆盖 —— 但终版仍在尾部全面胜出。
→ 说明 P1 的问题**不是覆盖密度不足,而是尾部结构性缺席**;`seglast + self-loop + proto teacher` 的泛化明显更好。

终版 per-task `end_lift` 全部贴近 0(−0.024 ~ +0.007),P1 为 −0.05 ~ −0.11。

---

## 4. ⚠️ 两个被证否的假设(重要)

### 4.1 「换 Viterbi 能根治大尾巴」—— 否

同数据、同原型、只换分割算法(`seg_cmp.py`):

| task | Viterbi 尾巴 | cummax 尾巴 | 结论 |
|---|---|---|---|
| 4 put both soup **and** cream cheese | **0.146** | 0.352 | ✅ 腰斩 |
| 5 put both soup **and** tomato sauce | **0.163** | 0.341 | ✅ 腰斩 |
| **8 白杯+布丁** | **0.233** | 0.227 | ❌ **几乎相同** |
| 0 stove + moka | 0.468 | 0.465 | ❌ 两者都巨大 |
| 9 book → caddy | 0.436 | 0.436 | ❌ 两者都巨大 |
| 均值 | 0.244 | 0.288 | 段数 6.4 vs 5.2 |

Viterbi **只在「put both A and B in basket」类有效**,救不了 task 8。
task 0/9 在两种算法下**都是 ~45% 尾巴** → 这是 **milestone 粒度不足**(第三个独立缺陷),不是解码算法问题。

### 4.2 「尾巴越大越差」—— 否

LIBERO-40 上 **`Spearman(tail_frac, tail_lift) = +0.538`(正相关)**:尾巴**越大**,lift 反而**越不负**。

原因是**位置**而非尾巴大小:越接近 episode 结尾,`persist → 1.0`(当前帧≈末帧),任何生成都赢不过「原地不动」。
尾巴小的任务其尾巴全落在最末端,反而负得最狠(task 33 尾巴仅 0.10 但 `end_lift=−0.098`)。

> **因此「task 6 因为 60% 大尾巴才失败」的叙事在 40 任务尺度上不成立。**
> 真正稳的信号是同位置下:**未训练区 −0.062 vs 已训练区 +0.012**。

---

## 5. 瓶颈在 generator,不在 predictor

P1 全帧诊断中 **`oracle ≈ deploy`,gap 恒为 0.001**:

| 位置 | 训练覆盖 | persist | deploy | oracle | oracle−deploy |
|---|---|---|---|---|---|
| 0-10% | 0.97 | 0.675 | 0.730 | 0.731 | 0.002 |
| 40-50% | **0.15** | 0.763 | 0.791 | 0.792 | 0.001 |
| 60-70% | **0.00** | 0.810 | 0.810 | 0.811 | 0.001 |
| 90-100% | **0.00** | 0.918 | **0.822** | 0.823 | 0.001 |

即使喂**真实目标 code**(teacher-forced),生成器也只能到 **0.822**,而 persist 一路升到 0.918 —— **deploy 在 0.82 处饱和,够不到完成态**。

**code 条件化已验证没有坍缩**(`code_sens.py`):vs 零码 0.687、vs 随机码 0.628(打乱 0.9999 属设计内:同目标同 code)。

> **对修复的含义**:终端监督不能只「给最终段建对」,**必须让 generator 见到 (尾巴帧 → 完成态) 样本**,否则 predictor 再准也没用。

---

## 6. 复现

全部脚本已归档到 `lmwm/scripts/`,从 `lmvla/` 目录运行:

```bash
python lmwm/scripts/seg_cmp.py                                    # §4.1 分割对照(CPU)
CUDA_VISIBLE_DEVICES=0 python lmwm/scripts/job1_final_pertask.py  # §2 终版 per-task
CUDA_VISIBLE_DEVICES=1 python lmwm/scripts/job2_libero40.py       # §3 P1 LIBERO-40
CUDA_VISIBLE_DEVICES=0 python lmwm/scripts/job3_final_allframes.py # §3.5 同口径闭合
python lmwm/scripts/p1_tail_diag.py                               # §5 oracle/deploy gap
python lmwm/scripts/code_sens.py                                  # §5 code 条件化检验
```

产物:`lmwm/outputs/job{1,2,3}_*.json`

---

## 7. 待办

- [x] ~~Job3:终版同口径~~ → 见 §3.5,终版 end_lift −0.011 vs P1 −0.078
- [ ] **milestone 粒度(第三缺陷)**:task 0/9 在 Viterbi/cummax 下**均** ~45% 尾巴,需提高 per-task K
- [ ] P1 支线定位:是**退役**(改走 `build_pairs_abl`)还是**拉齐**(把 Viterbi+self-loop 移植进 `p1_libero_milestone_pairs.py`)
- [ ] RoboTwin 铺开受阻:`task_index` 是**指令级**(921k 变体,多数 <5ep),需先解决任务分组
- [x] ~~若要改终端目标:generator 必须见到 (尾巴帧 → 完成态) 样本(§5)~~ → **已由 CRAVE per-task 双锚兑现**,
      末端 lift −0.029→+0.004、10/10 任务改善,见 [`RESULTS_crave_dualanchor_lmwm_2026-07-20.md`](RESULTS_crave_dualanchor_lmwm_2026-07-20.md)
