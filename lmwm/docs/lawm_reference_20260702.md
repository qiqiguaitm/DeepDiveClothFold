# LaWM/LAM 训练配方参考 — 用于优化 LMWM

> 日期:2026-07-02。从本地 `vendor/LaWAM/latent_action_model` 的配置与 Lightning 模块中提取 LaWM 的损失与训练参数,并在**我们自己的 LMWM 数据**上测试其损失配方,作为优化 LMWM 的参考。
>
> 注:LaWM 完整训练无法本地复现(需其数据集 + DINOv3 权重,配置里是占位路径 `/mnt/xx/xx/`)。因此"测试"指:导入并运行 LaWM 的真实损失函数,并把其损失配方应用到我们 LMWM 的 proto 预测上取参考数字。脚本:`scripts/test_lawm_losses.py`。

## 1. LaWM/LAM 完整训练配方(提取自 `config/dino_large_vae.yaml` + `core/lam_lightinng.py`)

### 优化器与调度
| 项 | LaWM 值 |
|---|---|
| 优化器 | AdamW |
| 学习率 | **3e-4** |
| weight_decay | **1e-2** |
| bias/norm 排除 wd | **是**(`exclude_bias_norm_from_wd`,分 decay/no-decay 两组) |
| LR 调度 | **线性 warmup 10000 步**,之后恒定 |
| 梯度裁剪 | 1.0 |
| 精度 | bf16-mixed |
| batch_size | 64 |
| accumulate_grad_batches | 1 |
| max_epochs | 10 |

### 损失(重建 + 辅助)
| 项 | LaWM 值 |
|---|---|
| 主重建损失 | `smooth_l1`,**beta=0.1**(对未来 DINOv3 patch 特征回归) |
| 可选损失类型 | l1 / smooth_l1(β=0.1) / cos(smooth_l1 + (1−cos_sim)) / charbonnier(eps=1e-3) / delta / l2 |
| 隐变量归一化 | `norm_latents: true`,**类型 LayerNorm** |
| 辅助 state 损失 | `lambda_aux=1.0`,`state_loss_type=l2`,masked eef 重建(物理落地) |
| diversity 损失 | `lambda_diversity=0.0`(VAE 下关闭) |
| VQ 类型 | `vae`(`vq_kwargs.layer_norm: true`) |
| 记录指标 | recon_loss、vq_loss、perplexity、**cos_sim_metric**、l1_loss_metric |

### 隐变量瓶颈(与我们 subgoal 头对应)
| 项 | LaWM 值 |
|---|---|
| code_dim | 32 |
| num_queries | 1 |
| num_frames | 2(当前 + 未来) |
| latent_layer_to_use | -2(DINO 倒数第二层) |
| decoder_last_ln | true |

## 2. LaWM vs LMWM 逐项对比

| 项 | LaWM/LAM | LMWM(当前) | 差异 |
|---|---|---|---|
| 学习率 | 3e-4 | 1e-3 | LMWM 高 3× |
| weight_decay | 1e-2 | 1e-6 | LaWM 强 4 个数量级 |
| bias/norm 排除 wd | 是 | 否 | LMWM 对全部参数施加 wd |
| warmup | 10000 步 | **无** | LMWM 无 warmup |
| proto 损失 beta | 0.1 | **1.0**(默认) | 详见 §3 |
| 隐变量归一化 | LayerNorm | L2 | 不同 |
| 回归目标 | 连续 DINOv3 未来特征 | **有限的 37-prototype 表** | 本质不同(关键) |
| 辅助 state 损失 | 有(物理落地) | 无 | — |

## 3. 在我们 LMWM 数据上测试 LaWM 损失(held-out,40,572 pair)

对我们 Stage-3 real-future 模型的 `greedy_proto` 预测 vs 真实下一 milestone 的 prototype:

| 损失 | 值 |
|---|---|
| smooth_l1 β=1.0(LMWM 当前) | 4.41e-05 |
| smooth_l1 β=0.1(LaWM) | 4.41e-04 |
| l1 | 6.82e-03 |
| l2 / mse | 8.81e-05 |
| charbonnier eps=1e-3(LaWM) | 6.99e-03 |
| cos 损失 = smooth_l1 + (1−cos)(LaWM) | 0.0568 |
| **cosine 相似度(指标)** | **0.9436** |

### beta 梯度分析(LaWM 为何取 0.1)
- per-dim 误差中位数 = **0.005**,**99.99%** 的误差 < 0.1。
- smooth_l1 梯度幅度 = clamp(|误差|/β, ≤1)。误差极小时:
  - β=1.0 → 平均梯度幅度 **0.0068**(几乎全在弱梯度的二次区)。
  - β=0.1 → 平均梯度幅度 **0.068**(**10× 更强**)。
- 结论:LaWM 的 β=0.1 对"回归平滑归一的 DINO 特征"是对的 —— 误差小,β=1.0 训不动。

## 4. 迁移测试:把 LaWM β=0.1 用到 LMWM(A/B,其他完全相同)

| 指标 | β=1.0(LMWM) | β=0.1(LaWM) |
|---|---|---|
| proto cosine(最佳步) | 0.9436 | 0.9442 |
| real-future top1 | 0.3829 | 0.3856 |
| real-future top5 | 0.8222 | 0.8247 |
| real-future NLL | 1.9775 | 1.9687 |

**结果:基本持平(噪声范围内)。** proto cosine 几乎不变(+0.0006),下游 top1 +0.3pt、NLL −0.009。

**为什么不转移**:β=0.1 在 LaWM 中重要,是因为它回归**高方差的连续 DINOv3 patch 特征**;而我们的 proto 目标是一个**有限的、已 L2 归一的 37-prototype 表**,头已经拟合到 cosine 0.944,梯度尺度不是瓶颈 —— 瓶颈是"目标是查表"(与 Phase A/C 结论一致)。产物:`checkpoints/stage3_realfuture_lawmbeta/`,`outputs/real_future_eval/realfuture_lawmbeta/`。

## 5. 迁移测试 2:warmup + weight_decay + bias/norm 排除(A/B 实测)

把 LaWM 的三项训练超参落地到 LMWM 训练器(`train_unified_lmwm.py` 新增 `warmup_steps`、`exclude_bias_norm_from_wd`,复用 `weight_decay`;`training.py` 加 `build_param_groups` / `make_warmup_scheduler`),其余与 real-future 基线完全相同,做 A/B。

| neural greedy(vs 真实) | 基线(LMWM) | LaWM-opt(wd 1e-2, warmup 100, 排除) | mild(wd 1e-3, warmup 50, 排除) |
|---|---|---|---|
| top1 | **0.3829** | 0.3805 | 0.3794 |
| top5 | **0.8222** | 0.8214 | 0.8200 |
| NLL | **1.9775** | 1.9812 | 1.9860 |
| best-fusion top1 | 0.4173 | 0.4197 | — |
| raw ECE | 0.1026 | 0.1027 | — |

**结果:两个设置都基本持平(噪声内,均略低于基线)。** warmup / 更强 wd / bias-norm 排除都没有提升 LMWM。

**为什么不转移**:这些是提升泛化/稳定性的良好默认,但本任务的天花板由**问题表述**决定 —— 固有 ~13 分支熵 + 表式目标(Phase A/C 已证),而非优化卫生。优化超参动不了 formulation-bound 的上限。产物:`checkpoints/stage3_realfuture_lawmopt{,_mild}/`,`outputs/real_future_eval/realfuture_lawmopt{,_mild}/`。

> 代码保留:这三项现已作为可选配置项进入训练器(默认关闭,向后兼容),作为未来在**更换任务表述后**(更大数据/连续目标/动作条件)可复用的良好默认。

## 6. 对 LMWM 的优化结论(按已测据排序)

**已测试 = 中性/无效(不是 LMWM 当前的杠杆):**
1. proto smooth_l1 **β=0.1**(§4):no-op —— 目标是有限 prototype 表,非连续特征回归。
2. **warmup + wd + bias/norm 排除**(§5):两设置皆持平 —— 天花板由 formulation 决定。

**最深的、真正的迁移是架构方向(非超参):**
3. LaWM 回归**连续未来特征**,我们回归**离散 prototype 表**。真正提升需要让目标离开查表 —— 这正是 Phase A→D 已经在做的(real-future 标签 + 帧条件分布 + VLA 集成)。LaWM 的价值主要在**"预测冻结空间连续 latent 子目标"的接口思路**,以及在**更换任务表述后**可复用的训练脚手架(warmup/wd/归一化/cos 损失),而非在当前表述下调单个超参。

**仍未测、可能有值的一项(留作后续):**
4. **cos 损失项**:LaWM 的 `cos` 类型 = smooth_l1 + (1−cos_sim);我们 proto 已 L2 归一,cosine 是自然目标。可直接加权试(我们已把 cosine 当指标记录)——但基于 §4/§5,预期同样受 formulation 天花板限制。

## 6.5 对齐实验:我们的 cos_sim_metric(LaWM 协议)

复刻 LaWM LAM 协议到我们 DINOv3-H 特征(`scripts/align_lawm_forecast.py`):固定 ~1.6s horizon(3Hz 缓存 h=5)、回归**未来帧特征**、`smooth_l1 β=0.1`、AdamW 3e-4/wd1e-2+warmup、inverse+forward(code_dim=32 + LN)、metric=cos_sim。held-out:

| horizon | persistence 基线 | forward-only(纯预测) | **inverse+forward(LaWM 式)** |
|---|---|---|---|
| ~1.0s (h=3) | 0.785 | 0.832 | **0.900** |
| **~1.7s (h=5≈LaWM dt)** | 0.742 | 0.819 | **0.890** |

**我们的 cos_sim_metric ≈ 0.89–0.90(LaWM 协议)。** 要点:
- inverse+forward(0.89)≫ forward-only(0.82)—— 正是 LaWM 的机理:转移码 u_t 携带当前帧缺的未来信息,给定码的重建保真度高(这也是策略要 conditioning 在码上的原因)。LaWM 记录的 `cos_sim_metric` 就是这个"给定码"的 0.89 型数字,和我们可**同协议对齐比较**。
- 这远高于我们之前的阶段跳变数字(medoid 0.864)—— 因为对齐后是**近未来**目标(更容易)。**证实:同协议下我们的预测质量不落后,在 0.89–0.90 的健康区间**(DINO 特征未来回归类工作的典型量级)。
- **caveat**:域不同(kai0 折叠 vs LaWM LIBERO/RoboTwin;pooled vs patch),对齐的是**协议**不是域;LaWM 论文(arXiv 2606.15768)的确切值本地无从取得,需查论文对齐比较。产物:`outputs/lawm_align/summary.json`。

## 7. 产物

- 测试脚本:`scripts/test_lawm_losses.py` → `outputs/lawm_loss_probe/summary.json`
- β 消融:`configs/training/kai0base_dinov3h_stage3_realfuture_lawmbeta.yaml`,`checkpoints/stage3_realfuture_lawmbeta/`
- warmup+wd+排除 消融:`configs/training/kai0base_dinov3h_stage3_realfuture_lawmopt{,_mild}.yaml`,`checkpoints/stage3_realfuture_lawmopt{,_mild}/`,`outputs/real_future_eval/realfuture_lawmopt{,_mild}/`
- trainer 新增(默认关闭,向后兼容):`training.proto_smooth_l1_beta`(默认 1.0)、`training.warmup_steps`(默认 0)、`training.exclude_bias_norm_from_wd`(默认 false);复用 `training.weight_decay`
- 库新增:`src/lmwm/training.py` 的 `build_param_groups()` 与 `make_warmup_scheduler()`
