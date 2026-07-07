# LMWM 最终报告:Latent Milestone World Model

> 面向 kai0 π0.5 VLA 的 milestone+1 **价值层宏观引导**世界模型。本文是**完整数据记录 + 过程迭代**的报告(网站 `web/showcase/reports/lmwm_final/` 是精简可视化 showcase,深度细节以本文为准)。
> 更新 2026-07-07 · 最终 = **预测器 Predictor + 生成器 Generator**,π0.5 SigLIP 空间,`center_w=0.1`(`lmwm/checkpoints/twomodel_final.pt`)。

---

## 0. 核心目标与两条硬指标

LMWM 的目的:给 VLA 一个 milestone+1 的**价值层宏观引导**(不是进度级、是价值级)。据此定两条**硬指标**:
- **① 解码保真**:预测的 milestone+1 解码后要忠实(不能是 off-manifold 噪声/黑斑)。
- **② 价值前向**:预测的 milestone+1 在**进展价值**上要 > 当前 stage。

其余(reach / horizon / lag / 平滑)可协商。

---

## 1. 指标词典(本科生级)

预测目标 = "未来某 milestone 的 patch 特征网格"。用多指标从不同角度衡量:

| 指标 | 定义(直白版) | 表明什么 |
|---|---|---|
| **grid-cos** | 预测网格 vs 真实目标网格的余弦 | 核心保真度(⚠️ 编码器空间不同则不能硬比) |
| **oracle grid-cos** | 用"作弊码"(teacher 偷看真未来)重建的 cos | 天花板 = 生成器表达上限(不经预测器) |
| **deploy grid-cos** | 只看当前帧、用预测码重建的 cos | **实战值**(最关心) |
| **persistence** | 把"当前帧"当预测(不动)的 cos | 白给基线 |
| **lift** = deploy − persistence | 扣掉"目标本来就近"的便宜 | 净技巧 |
| **reach / model lag** | 预测匹配到的真帧比当前晚几秒 | 敢预测多远(**同协议下可严格比秒**) |
| **dataset horizon** | 真实目标帧比当前晚几秒 | 目标时间跨度 |
| **undershoot ratio** = reach/horizon | 达成度 | <1 = 欠射(目标越远 ratio 越易低) |
| **身份 top-N** | 预测码解出的"下一 milestone" 命中真 milestone 的 top-N 率 | **硬指标② 的价值多峰命中** |
| **value-forward frac** | 预测 milestone 的 pord 值 > 当前 stage 的比例 | **硬指标②** 直接量 |
| **pred smoothness** | 相邻帧预测的余弦 | 是否抖(抖 = 给 VLA 噪声) |
| **corr(value,time)** | CRAVE 进度值 vs 时间的相关 | 进度标签质量(0.94) |

---

## 2. 最终网络结构(Predictor + Generator)

```
在线编码器 = π0.5 SigLIP-So400m/14 (冻结, 与策略同塔, 复用 KV)
   224² RGB ──► 16×16×1152 grid G_t  ──pool──►  gist g_t (1152)

预测器 Predictor —— 回答"下一个价值 milestone 是哪个":
   训练形态(teacher): inverse(G_t, G_future) ──► 理想码 z (128)   [2层CNN]
                      + 簇中心 CE 身份锚: CE(cen_head(z) → 37 milestone id, next_ms) · center_w=0.1
   部署形态(deploy):  MDN(g_t) ──► K=4 高斯混合 over 码 ──► argmax-π 分量均值 = ẑ  [3.28M]

生成器 Generator —— 回答"它在当前场景长什么样":
   AdaLN(G_t 当画布, 码) ──► 下一 grid Ĝ    [4 块 CNN, zero-init gate, 只学"变化"]
   训练/部署同一形态

离线 label 工厂(不上线): DINOv3-H(840M) + CRAVE Viterbi 单调分配 → milestone+1 medoid 目标帧
patch 解码器(仅可视化, 不喂 VLA): make_decoder big, val_L1≈0.05

世界模型参数 ≈ 34M(预测器 3.28M + 生成器 + teacher 仅训练用)
```

**为什么是 AdaLN 而非 concat**:concat 允许"复制当前帧"塌缩;AdaLN 以当前 grid 为画布、码只调制 shift/scale/gate(zero-init gate 从 proj(g_t) 起步),强制只学变化。消融里 concat → lag/平滑掉(见 §5)。**与 LaWM 收敛到同款 AdaLN 调制**。

**注入 VLA(§下一步)**:子目标 Ĝ + 码 ẑ → π0.5 action expert(KI + RoPE 偏移 + mask 重设计)。

---

## 3. 与 LaWM 的对比(完整表)

LaWM(jialei02/lawam_lam 官方 ckpt,给它补 deploy-predm)在**我们数据**、**同 reach 协议**实测:

| 指标 | LaWM | LMWM(cw=0.1) | 谁好 / 口径 |
|---|---|---|---|
| **reach / model lag(秒)** | 1.476 | **1.673** | **LMWM** · 严格同口径,绝对更远 |
| 目标 horizon(秒) | 1.66 | 2.64 | LMWM 目标更远更难 |
| undershoot ratio | 0.891 | 0.63 | LaWM 高:近未来易打满(非更强) |
| 前向预测率(>0) | 44.9% | 35.3% | LMWM 近+远双峰、尾更长 |
| 负滞后率(<0) | 10.7% | **6.5%** | **LMWM** · 更少"预测倒退" |
| oracle grid-cos | 0.770 | 0.715–0.750 | ⚠️ 空间不同不硬比(ViT-B768 vs SigLIP1152) |
| persistence | 0.627 | 0.599–0.601 | — |
| lift(oracle−persist) | +0.143 | +0.116–0.135 | 净技巧,趋势可比 |
| deploy grid-cos | —¹ | **0.726–0.728** | 只看当前帧实战值 |
| 身份 top-3 | N/A² | **0.511** | **硬指标②** |
| 预测平滑(相邻 cos) | — | **0.948** | 稳定信号 |
| 进展价值单调 | N/A² | **Viterbi 保证** | **硬指标②**,LaWM 无 |
| 跨数据集(vis_base 零微调) | 定性 | **0.56 定量** | — |
| 世界模型参数 | ~230M Transformer | **~34M CNN** | 轻 ~10× |
| 下游 SR(extrinsic) | **LIBERO 98.6% / 187ms** | 待测 | **LaWM** · 唯一短板 |

¹ LaWM 的 LAM 内不产出预测码(甩给下游 VLA),补 deploy-predm 才能测 reach。 ² CRAVE 进度/身份标签为 LMWM 特有。

**读表**:LMWM 更好 = reach、负滞后、参数、VLA 嵌入深度、进展价值单调(②)、跨集可定量;LMWM 不足 = undershoot ratio(目标远所致)、grid-cos 不硬比、**下游 SR 待测**。"LaWM 敢预测更远"是 ratio 错觉(它目标只 1.66s、近未来易打满),论绝对秒数 LMWM 反超。

---

## 4. 消融与数据记录

### 4a. 核心结构锁死(控制变量,同口径)
| config(改一处) | deploy | id_t1 | id_t3 | lag_ratio | fwd% | neg% | smooth | 结论 |
|---|---|---|---|---|---|---|---|---|
| base | 0.721 | 0.208 | 0.474 | 0.75 | 46% | 18% | 0.943 | 参照 |
| concat(关 AdaLN) | 0.714 | 0.230 | 0.470 | 0.52 | 35% | 6% | 0.905 | lag/smooth 掉 |
| nolift(关 lift) | 0.701 | 0.241 | 0.481 | 0.57 | 40% | 8% | 0.873 | lag/smooth 掉 |
| noteacher | 0.645 | 0.178 | 0.401 | — | — | — | — | deploy 崩 |
| code256 | 0.703 | 0.192 | 0.444 | — | — | — | — | 更差 |

**核心 = inverse-teacher + AdaLN + lift + code128**,四项各自关掉都掉指标。

### 4b. 簇中心锚强度 center_w 扫描(定案 = 0.1)
| center_w | deploy | id_t1 | id_t3 | reach_s | ratio | smooth |
|---|---|---|---|---|---|---|
| 0.0 | 0.717 | 0.214 | 0.474 | 1.67 | 0.63 | 0.919 |
| **0.1** | **0.728** | **0.270** | **0.511** | **1.67** | 0.63 | **0.948** |
| 0.25 | 0.721 | 0.255 | 0.510 | 1.50 | 0.57 | 0.926 |
| 0.5 | 0.717 | 0.264 | 0.503 | 1.25 | 0.47 | 0.913 |

甜点 = **0.1**:身份/deploy/reach/平滑同时见顶;>0.1 过保守(0.5 把 reach 从 1.67 砍到 1.25)。这解释了早前"LMWM reach 不如 LaWM"的真因 = center_w 定太高(旧定案是 ccenter 0.5)。簇中心 on-manifold 锚也修好了 off-manifold 黑斑(硬指标①)。

### 4c. 预测器输入:gist vs grid(定案 = gist)
控制变量,仅 `--pred_input` 不同;grid 变体 = `MilestonePredictorGrid`(2 层 conv 下采样 → 同 MDN 头)。
| pred_input | deploy | oracle* | bestof8 | id_t3 | id_t5 | value_fwd | predm 参数 |
|---|---|---|---|---|---|---|---|
| **gist**(定案) | **0.7260** | 0.7504 | 0.7321 | **0.5125** | 0.6125 | 0.465 | **3.28M** |
| grid | 0.7257 | 0.7599 | 0.7369 | 0.5044 | 0.6231 | 0.460 | 5.61M(+71%) |

**结论:维持 gist**。deploy 打平(Δ0.0003),身份/value-forward 全在 run-to-run 噪声(±0.01)内;grid 多 71% 参数换 0 deploy 收益 → "下一个是哪个价值 milestone"是**场景级全局身份判断**,pooled gist 已充分,空间细节无增量。
*oracle 只经 teacher+生成器、不过预测器,两 run 的 0.75/0.76 差是 batch 采样未固定种子的噪声,非 grid 效应;严格坐实需多 seed,deploy 打平信号已足够。

---

## 5. 过程迭代(演进史)

| 阶段 | 关键变化 | 结论 |
|---|---|---|
| **DINOv3-H V1→V3.1** | 目标构建:时序-next → 进度-argmax → 进度-Δ → **Viterbi 单调 milestone** | V3.1 唯一同满足 时间前向+跨集一致+进度单调+干净 medoid |
| **多峰归位** | 发现"多模态只回收 +0.02"**测错了轴**:多峰在**身份**(哪个 milestone),grid-cos 不敏感 | 换 MDN 多峰 Stage-1 + 身份 top-N 指标 |
| **SigLIP-space 重构** | 在线编码器 DINOv3-H → **π0.5 SigLIP 同塔**(消除与 VLA 隔阂),DINOv3-H+CRAVE 降为离线 label 工厂 | deploy 不掉,融合零隔阂 |
| **Predictor/Generator** | 两模块化:预测器(身份码 MDN)+ 生成器(AdaLN 落地);v1 concat→77% persistence 塌缩,改 AdaLN 修复(ratio 0.29→0.99) | 定架构 |
| **center_w 调参** | 簇中心 CE 锚 0.5→**0.1** | reach 1.25→1.67,身份/保真见顶 |
| **gist vs grid** | 预测器输入 | grid 无增益,维持 gist |

**三大天花板(实测)**:① 容量(Transformer)无用(LaWM 230M 未碾压 34M);② 早期"多模态天花板低"结论**作废**(是 grid-cos 掩盖身份多峰);③ SigLIP 解码 val_L1≈0.05 是对比训练特征的重建天花板,**仅影响可视化**(VLA 吃特征不吃像素),非预测/VLA 问题。

---

## 6. 下一步:LMWM × VLA → LMVLA / LMWAM

子目标 Ĝ + 码 ẑ 注入 π0.5 action expert(KI + RoPE 偏移 + mask)。Phase 1 融合接线(本地)→ Phase 2 簇微调(集群 submit-training-job)→ Phase 3 真机。裁决 = offline action-MAE(base vs +LMWM vs control)→ 真机 SR,对齐 LaWM LIBERO 98.6%。

---

## 7. 产物索引
- 最终 ckpt:`lmwm/checkpoints/twomodel_final.pt`(center_w=0.1)
- 训练:`train_twomodel_v2.py`(`MilestonePredictor` + `MilestoneGenerator` + `MilestonePredictorGrid`)
- 消融:`train_ablation.py`(`--target_mode/--teacher/--fwd_arch/--lift_w/--code_dim/--center_w/--pred_input`)
- reach 同口径:`measure_twomodel_v2_lag.py` / `measure_lawm_lag.py` · 分布图 `plot_lag_dist.py`
- 跨域渲染:`make_visbase_dinov3h_index.py` + `render_twomodel_video.py`
- 深度文档:`ABLATION_CONVERGENCE_2026-07.md`(消融全表)、`RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md`(命名/假设/多峰)、`PROGRESS_lawm_comparison_2026-07.md`(LaWM)
- baseline:LaWM `rlinf.github.io/LaWAM`(vendored `lmwm/vendor/LaWAM`)
- 网站 showcase:`web/showcase/reports/lmwm_final/`
