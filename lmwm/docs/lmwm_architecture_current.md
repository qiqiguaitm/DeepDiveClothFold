# LMWM 当前架构 + 与 LaWM 核实对比(2026-07-05 收紧落盘)

面向 LMWAM(接入 kai0 π0.5 的 world-model-augmented VLA)的**冻结** LMWM。全部 stop-grad,作子目标/条件源。
LaWM 侧数字**来自官方仓库实测**(`lmwm/vendor/LaWAM`,clone 自 `github.com/RLinf/LaWAM`,配置 `latent_action_model/config/dino_base_vae.yaml`)+ 论文 arXiv 2606.15768。

---

## 1. LMWM 组件 + 参数(ours,ckpt 实测)

| 组件 | 实现 | 输入→输出 | 参数 |
|---|---|---|---|
| 冻结编码器 | DINOv3-H (ViT-H+/16),末层 | 256²RGB → 16×16×1280 | **~840M(冻结)** |
| milestone 结构 | recurrence 原型逐帧 argmax | grid → stage id / 段 medoid | ⚠️细粒度(~70-90段/ep) |
| 子目标 A:milestone+1 | forward-from-current(predm→code→fwd) | 16×16×1280 → 16×16×1280 | deploy cos **0.725**(cd128) |
| 子目标 B:near-future | 同上,h3(~1s) | 同上 | deploy **0.705** |
| 逆动力学 inverse(teacher) | 2 层 CNN,**确定性** | 2×grid → code | 6.5M |
| 正动力学 forward | 3 层 CNN | grid+code → grid | 17M |
| 转移码 ẑ | inverse/predm | → 64/128 | — |
| 解码器(仅可视化) | make_decoder big+GDL | grid → 128²RGB | 5M;L1 0.0206/sharp574 |

部署推理(predm+fwd)/路 ≈ **21M**;两路 ≈ 54M(inverse 仅训练用)。

## 2. LaWM 组件 + 参数(官方代码实测)

| 组件 | 实现(dino_base_vae.yaml) | 输入→输出 | 参数 |
|---|---|---|---|
| 冻结编码器 | DINOv3 **ViT-B/16**,取**第 -2 层**特征 | 256²→ 16×16×768 | ~86M(冻结) |
| 逆动力学 inverse | **transformer 12 层**,dim768,heads12,3D pos,**num_queries=1** | 2 帧 grid → **VAE** 潜在动作 | — |
| 正动力学 decoder | **transformer 6 层** | (u, z) → û_T grid | — |
| 潜在动作 z | **VAE bottleneck(随机)**,LN 归一 | 单 token → **code_dim=32** | — |
| LAM 合计 | (headline 230M 的是放大版;base cfg 12enc/6dec 更小) | — | **~230M(论文口径)** |
| 策略 | Qwen-GR00T:Qwen3-VL 前16层 + 4 Alternate-DiT | u,û_T,language → action + ‖ẑ−z‖² | 总 2.3B |

## 3. 核心对比(核实后)

| 维度 | LaWM | LMWM(ours) | 差异 |
|---|---|---|---|
| **latent 空间** | **patch-grid 16×16**(非 pooled) | **patch-grid 16×16** | ✅ **相同**(我早前"LaWM用pooled"是错的,已纠正) |
| 编码器 | ViT-B/16,dim**768**,~86M,层-2 | ViT-H+/16,dim**1280**,~840M,末层 | 我们编码器 **~10× 大** |
| LAM 结构 | **transformer**(12enc/6dec),230M | **CNN**(2enc/3dec),27M/路 | LaWM ~8× 大且 attention |
| 潜在动作码 | **32-d,VAE(随机)** | 64/128,**确定性** | ⚠️见下"多模态" |
| num_frames | 2(pair) | 2 | 相同 |
| **horizon** | 单一**物理时间 1.6s**(机器人)/0.4s(人) | **双路**:near-future h3(~1s,帧步未校准)+ milestone(事件,可变) | 我们多语义路;但未按秒校准 |
| 子目标条数 | 1 | 2 | 我们多一路 |
| 像素解码入环 | 无(纯 latent) | 无(解码仅可视化) | 相同(都不重建像素) |

## 4. 【落盘】最终注入给 VLA 的信号差异

**LaWM 注入 policy(Alternate-DiT 动力学流)**:
- `u` = 当前观测 latent(16×16×**768**)
- `û_T` = 预测子目标 latent(16×16×768,horizon 1.6s)
- `z` = 潜在动作(**32-d VAE**),policy 学 ẑ 驱动 WM,蒸馏 `‖ẑ−z‖²`
- → 动力学流 token 量 = u+û_T = **2×256 = 512**,dim 768

**LMWM 注入 π0.5(计划 M2)**:
- `u` = obs_grid(16×16×**1280**)
- `û_T^milestone`(16×16×1280,事件 horizon)
- `û_T^nearfuture`(16×16×1280,~1s)
- `ẑ`(**64-d 确定性**),蒸馏 `‖ẑ−z‖²`
- → 子目标 token 量 = u+2 子目标 = **3×256 = 768**,dim 1280

**信号差异清单**:
1. **特征维 768 vs 1280**。
2. **子目标 1 条 vs 2 条**;prefix token 量 512(dim768)vs 768(dim1280)→ 我们进 VLA 的信号体量 **~2.9×**(逆 LaWM 效率路线)。
3. **潜在动作 32-d VAE(随机)vs 64-d 确定性** —— **这是最实质差异**:LaWM 的 VAE 本身就是"多模态头",采样能覆盖多分支未来;我们确定性回归→收敛到条件均值→ oracle→deploy 的 0.07 gap 正是这里丢的。
4. LaWM **无 milestone/语义信号**;我们额外注入语义大目标。
5. horizon:LaWM 单 1.6s(物理秒,可移植);我们 near-future 帧步未校准 + milestone 事件。

## 5. 两个待验证杠杆

- **LAM 容量阶梯**(CNN→conv-attn→transformer × 尺寸档):oracle 0.79 是**容量**还是**任务**天花板?gf3 扫 milestone/near-future 两路(`--arch`,进行中)。
- **多模态潜在动作头**:把确定性 code 换成 **VAE/flow**(对齐 LaWM),压 oracle→deploy 的 0.07 gap。⚠️ 这两条正交,都该测。
- **horizon 校准到物理秒**(near-future 对齐 LaWM 1.6s,按各数据集 fps 换算),消跨数据集漂移。

## 6. 跨数据集证据(kai0→vis_base 零微调,实测)
| 数据 | mean pred cos |
|---|---|
| kai0 in-dist | 0.725 |
| vis_base 04-23 ep8 | 0.663 |
| vis_base 06-18 ep0 | 0.676 |

预测器**特征空间跨本体迁移成立**(掉~0.06);⚠️ 06-18 暴露**解码器 kai0 外观域偏**(橙衣→解码成青),是解码器侧非预测器(预测/真实一致故 cos 有效)。

## 7. 复现 / ckpt 状态
- **官方代码已 vendored**:`lmwm/vendor/LaWAM`(RLinf/LaWAM)。含 `latent_action_model`(LAM stage1)+ `starVLA`(policy)+ LIBERO/RoboTwin examples。
- **官方开源 ckpt**:HF `jialei02/lawam-checkpoints`;数据集 `jialei02/{libero_merged_no_noops_20hz, robotwin_merged}`。
- 早前对齐工作:`lawm_reference_20260702.md`、`align_lawm_forecast.py`(同协议 cos 0.89-0.90)、`lmwm_vs_lawm_and_patch_scheme_20260704.md`(4 层对比)。
- **未做**:用官方 ckpt/代码在**我们数据(kai0/vis_base)**上跑 LaWM LAM,直接对比 oracle/deploy grid-cos —— 待办。

## 8. 交付件(gf3)
预测器 `lmwm/outputs/subgoal_opt/{milestone_cd128,nearfuture_h3_cd64}.pt` · 解码器 `lmwm/checkpoints/patch_decoder/patch_dec_{big,xl}_gdl0.5.pt` · 编码器 `crave/encoders/_dino_vit_standalone.py` · 渲染 `render_milestone_predict_video.py`(`--raw_video` 跨数据集)
