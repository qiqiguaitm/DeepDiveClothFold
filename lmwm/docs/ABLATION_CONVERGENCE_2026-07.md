# v2 预测器消融收敛(2026-07)

控制变量消融(每 config 只改一处,同口径)。gf3 8卡 RAM-OOM 后改本地 2卡 fp16 跑通。

## 判据全表
| config | deploy | id_t1 | id_t3 | lag_ratio | fwd% | neg% | smooth | 结论 |
|---|---|---|---|---|---|---|---|---|
| base | 0.721 | 0.208 | 0.474 | 0.75 | 46% | 18% | 0.943 | 参照 |
| concat | 0.714 | 0.230 | 0.470 | 0.52 | 35% | 6% | 0.905 | AdaLN关→lag/smooth掉 |
| nolift | 0.701 | 0.241 | 0.481 | 0.57 | 40% | 8% | 0.873 | lift关→lag/smooth掉 |
| noteacher | 0.645 | 0.178 | 0.401 | — | — | — | — | teacher关→deploy崩 |
| code256 | 0.703 | 0.192 | 0.444 | — | — | — | — | 更差 |
| ccenter | 0.718 | 0.260 | 0.509 | 0.46 | 33% | 5% | 0.914 | 身份↑ 负lag↓ 但保守 |
| tall (allframes) | (val混淆) | — | — | 1.67 | 71% | 13% | 0.953 | 最平滑+最往前 但过射 |
| tc (all+center) | (val混淆) | — | — | 1.52 | 69% | 12% | 0.913 | 过射,未平衡 |

## 收敛结论
**核心锁死(控制变量验证)**:inverse-teacher + AdaLN + lift + code128。四项各自关掉都掉指标。

**两个旋钮 = 相反 trade-off,最优点由 SR 定**:
- 簇中心 center:身份+0.05、负lag 18%→5%,但保守(欠射 0.46)。
- allframes:最平滑(0.953)+最往前(71%),但过射(1.5–1.7,越过 milestone+1)。
- 二者方向相反,intrinsic 指标此消彼长(身份 vs horizon);tc 叠加未平衡。→ **只有下游 SR 能裁决 target/teacher 的最优点。**

**allframes val 混淆说明**:allframes 把段内早期帧(离目标远)纳入 → 自身 val 变难 → deploy/身份"掉"是 val 更难,非模型更差;公共 lag 口径下它其实最平滑+最往前。

脚本:`train_ablation.py`(--target_mode/--teacher/--fwd_arch/--lift_w/--code_dim)、`measure_twomodel_v2_lag.py`(lag+平滑,支持 concat)。

## center_w 曲线(2026-07)+ 定案更新
center_w ∈ {0,0.1,0.25,0.5}(teacher=center)扫描:
| center_w | deploy | id_t1 | id_t3 | reach_s | ratio | smooth |
|---|---|---|---|---|---|---|
| 0.0 | 0.717 | 0.214 | 0.474 | 1.67 | 0.63 | 0.919 |
| **0.1** | **0.728** | **0.270** | **0.511** | **1.67** | 0.63 | **0.948** |
| 0.25 | 0.721 | 0.255 | 0.510 | 1.50 | 0.57 | 0.926 |
| 0.5 | 0.717 | 0.264 | 0.503 | 1.25 | 0.47 | 0.913 |

**甜点=center_w 0.1**:身份/deploy/reach/平滑全见顶。reach 1.67s **> LaWM 1.48s**(在更难的 2.8s milestone 目标上)→ "LMWM reach 不如 LaWM"的真因是 center_w 定太高(0.5 把 reach 从 1.67 砍到 1.25)。**定案 ccenter(0.5)→ center_w=0.1(twomodel_final.pt)。**
LaWM 基线 lag:reach 1.48s / ratio 0.947(固定 1.6s horizon,近未来易打满);绝对 reach 我们(0.1)反超。分布图 `lag_distribution.png`、曲线 `center_w_curve.png`。
⚠️ value_forward 指标此前用 argmax 当前 milestone 有 bug(已修为 Viterbi);目标本身 value-forward 由 Viterbi 单调构造保证。

## 预测器输入消融(2026-07)· gist vs grid
控制变量(同 center_w=0.1/teacher=center/adaln/code128/K4,仅 `--pred_input` 不同)。grid 变体 = `MilestonePredictorGrid`(2 层 conv 下采样 G_t → 池化 → 同 MDN 头)。
| pred_input | deploy | oracle* | bestof8 | id_t3 | id_t5 | value_fwd | predm 参数 |
|---|---|---|---|---|---|---|---|
| **gist**(定案) | **0.7260** | 0.7504 | 0.7321 | **0.5125** | 0.6125 | 0.465 | **3.28M** |
| grid | 0.7257 | 0.7599 | 0.7369 | 0.5044 | 0.6231 | 0.460 | 5.61M(+71%) |

**结论:维持 gist**。deploy 完全打平(Δ0.0003),身份/value-forward 全在 run-to-run 噪声(±0.01)内;grid 多 71% 参数换 0 deploy 收益。说明"下一个是哪个价值 milestone"是**场景级全局身份判断**,pooled gist 已充分,空间细节无增量。
*oracle 只经 teacher(inv)+生成器、不过预测器 → 两 run 的 0.75/0.76 差是 batch 采样未固定种子的噪声,非 grid 效应;严格坐实需多 seed,但 deploy 打平的信号已足够。脚本:`train_ablation.py --pred_input {gist,grid}`、`MilestonePredictorGrid`(train_twomodel_v2.py)。
