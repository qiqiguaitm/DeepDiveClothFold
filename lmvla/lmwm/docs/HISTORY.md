# LMWM 文档历史索引 · HISTORY

> ⚠️ **动手前先读这里。** 本目录 `history/` 下都是**历史 / 被取代 / 未来阶段**的文档,**不要**照搬其中的旧方案或旧代码(编码器、teacher、指标口径很多已变)。
> **当前最终方案**(顶层 live,完整入口见 [`README.md`](README.md)):
> - 线 A 独立 LMWM = [`RESULT_newcrave_final_arch_2026-07-11.md`](RESULT_newcrave_final_arch_2026-07-11.md)（统一 DINOv3-base + shared PCA128 teacher）+ [`DECODER_dinov3base_video_2026-07-11.md`](DECODER_dinov3base_video_2026-07-11.md)。
> - 线 B LMWM×LaWAM 集成 = [`LMWM_report_full_draft.md`](LMWM_report_full_draft.md)（+ 支撑参考 `LAM_starVLA_contract` / `BUG_AUDIT`）。
> 本索引目的:**防误用旧方案 / 旧代码**,同时**留档可回溯**(想复用/查踩坑能快速定位)。CRAVE 侧对应 [`../../crave/docs/HISTORY.md`](../../crave/docs/HISTORY.md)。

---

## 0. 当前最终（顶层,live）

| 文档 | 内容 |
|---|---|
| ⭐ [`RESULT_newcrave_final_arch_2026-07-11.md`](RESULT_newcrave_final_arch_2026-07-11.md) | **单一事实源**:统一 DINOv3-base LMWM(`train_multitask.py --encoder dinov3base --teacher proto --teacher_code shared_pca`)· 新 CRAVE 方法 milestone · deploy 0.910/id3 0.940 · 闭环可视化 · 修复表 · 三方 teacher 码消融(§4.6)|
| ⭐ [`DECODER_dinov3base_video_2026-07-11.md`](DECODER_dinov3base_video_2026-07-11.md) | 可视化解码器:pooled(软) + **grid(锐)** 两法 + 时序一致 |

**交付产物(码/ckpt)**：ckpt `lmwm/checkpoints/dinov3base_lmwm_sharedpca_kaicoffee.pt`(+消融 `_kaicoffee.pt`=rand / `_pca_kaicoffee.pt`=per-task)· 解码器 `dinov3base_decoder/{kai_grid_dec,kai_video_dec}.pt`。
**脚本**：`train_multitask.py`(`--encoder dinov3base --teacher_code {shared_pca,rand,pca}`)· `gen_newcrave_spec.py` · `train_dinov3base_grid_decoder.py` / `train_dinov3base_video_decoder.py` · `make_neural_pred_decode_video.py` · `measure_dinov3base_lag.py`。

---

## 1. 🔴 已被取代（旧最终,勿照搬——SigLIP-era,已被 DINOv3-base 统一版取代）

| 文档(history/) | 曾是什么 | 为何淘汰 / 可打捞的 |
|---|---|---|
| [`LMWM2_FINAL_ARCHITECTURE.md`](history/LMWM2_FINAL_ARCHITECTURE.md) | SigLIP-era "最终定档架构"(proto teacher + MDN K4 + prev_ẑ + 密度弃权,P1/P2 探针裁决表) | 编码器换 DINOv3-base 统一空间。**可打捞**:prev_ẑ/密度弃权/proto teacher 的逐项裁决理由(仍适用) |
| [`ARCHITECTURE_AND_BASELINE.md`](history/ARCHITECTURE_AND_BASELINE.md) | SigLIP 架构 + LaWM baseline 单页速查 | 同上被取代。**可打捞**:LaWM 结构对照、reach 口径定义 |
| [`FINAL_CROSSTASK_PREDICTOR.md`](history/FINAL_CROSSTASK_PREDICTOR.md) | SigLIP proto teacher 跨任务交付(deploy 0.753/id3 0.710) | 迁 DINOv3-base 后指标口径变。**可打捞**:proto vs inv、union_ce vs progress 消融、开放词表论证 |
| [`FINAL_REPORT.md`](history/FINAL_REPORT.md) | SigLIP-era 完整技术报告 | 被 RESULT 取代。**可打捞**:方法叙述、配图 |
| [`REDESIGN_LMWM2_2026-07.md`](history/REDESIGN_LMWM2_2026-07.md) | LMWM-2 重设计方案(三路文献调研) | 被 LMWM2_FINAL 再被 RESULT 取代。**可打捞**:JEPA/LaWM/扩散 调研 |
| [`ABLATION_CONVERGENCE_2026-07.md`](history/ABLATION_CONVERGENCE_2026-07.md) | v2 预测器消融收敛(SigLIP) | 消融基于 SigLIP。**可打捞**:teacher/anchor/fwd_arch 消融方法 |
| [`PROGRESS_lawm_comparison_2026-07.md`](history/PROGRESS_lawm_comparison_2026-07.md) | LaWM 实测对比(reach **1.67s** = SigLIP 版) | **reach 已被 DINO 版 0.811s 取代**(RESULT §3.2)。**可打捞**:LaWM 同口径 reach 协议 |
| [`PLAN_new_crave_on_lmwm_2026-07-10.md`](history/PLAN_new_crave_on_lmwm_2026-07-10.md) | "用新 CRAVE 在 LMWM 跑一版"的计划 | **已执行 → 成果即 RESULT**。留作规划留痕 |

## 2. 🟡 未来阶段参考（SigLIP 融合,尚未做——到那阶段再取用,勿当现状）

| 文档(history/) | 内容 | 何时用 |
|---|---|---|
| [`MASTER_PLAN_lmwm_vla_2026-07.md`](history/MASTER_PLAN_lmwm_vla_2026-07.md) | E0→E3 总执行规划(含 SigLIP 注入 π0.5) | 迁 SigLIP 空间 + VLA 融合阶段(RESULT §4.4/未来) |
| [`INJECTION_DESIGN_2026-07.md`](history/INJECTION_DESIGN_2026-07.md) | milestone 注入 π0.5 主方案(虚拟图像 token 进 prefix + KI stop-grad) | VLA 融合 Phase 1 |
| [`INJECTION_DEEP_ANALYSIS_latent_milestone_2026-07-10.md`](history/INJECTION_DEEP_ANALYSIS_latent_milestone_2026-07-10.md) | 注入机制深度分析(π*0.6 + KI) | VLA 融合设计参考 |

## 3. 📌 仍有价值（防踩坑 / 研究方向,随时可查）

| 文档(history/) | 内容 |
|---|---|
| [`PITFALLS_AND_HISTORY.md`](history/PITFALLS_AND_HISTORY.md) | **LMWM 踩坑表 + 版本演进史 + 验证方法库** —— 建模负结果(帧历史无增益/7B 均值方差被否/EM-HMM 塌缩等)+ gf3 工程坑。**动手前必查,防重复踩坑** |
| [`RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md`](history/RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md) | 普适 milestone 定义 + 融合研究方向探讨(决策点 A:走 SigLIP 同空间) |

## 4. 🧩 集成线已执行计划 / 快照（LMWM×LaWAM,2026-07-12~13,归档于 2026-07-15）

> 这些是"把 LMWM 换进 starVLA 世界模型槽"这条线的**计划书 / 进度快照 / 完整痛点分析**。P0–P4 均已执行完成,**当前对外事实源是顶层 [`LMWM_report_full_draft.md`](LMWM_report_full_draft.md)**(已吸收其中验证过的痛点子集 1/2/3)。留档供回溯方法与未采纳的痛点方向。

| 文档(history/) | 曾是什么 | 可打捞的 |
|---|---|---|
| [`PLAN_lmwm_replace_lawm_2026-07-12.md`](history/PLAN_lmwm_replace_lawm_2026-07-12.md) | LMWM 替换 LaWM 的技术路线(P0–P4 干净 A/B 设计) | 控制变量设计、待替换接口清单、风险表 |
| [`LAWAM_reproduce_and_kai0_sft_plan_2026-07-12.md`](history/LAWAM_reproduce_and_kai0_sft_plan_2026-07-12.md) | 复现外部 LaWAM + kai0 SFT 计划 | LaWAM 资源清单、v2.1→3.0 转换要点、基座拍板 |
| [`P1_progress_2026-07-12.md`](history/P1_progress_2026-07-12.md) | P1 切 DINOv3-base 重训进度快照 | per-task milestone 修正、特征抽取/帧读取踩坑 |
| [`LMWM_pain_point_analysis_2026-07-13.md`](history/LMWM_pain_point_analysis_2026-07-13.md) | 痛点分析**完整版**(5 痛点 A–E + P0 实证 + 论文证据 + 测试路线) | 报告只保留验证过的 1/2/3;此处含 **C 跨编码器 / D 长周期 / E 视觉鲁棒性** 的未来方向与优先级 |

## 5. 📦 更早期归档（history/archive/,27 docs,2026-07-01~04）

7 月 1-4 日的探索期文档(阶段架构 20260703、优化日志、天花板分析、Phase A/B/C、decoder 迭代、medoid 目标分析、LaWM 参考等)。已有子索引 [`history/archive/README.md`](history/archive/README.md)。**均为探索期留痕,勿照搬**;查特定负结果/迭代过程时进去。

---

## 快速回溯指引

- **想复用某段代码** → 先看本文对应文档的"可打捞"列 + RESULT 的脚本清单,再去 `lmwm/scripts/`。
- **想避免重复踩坑** → 读 [`history/PITFALLS_AND_HISTORY.md`](history/PITFALLS_AND_HISTORY.md) 和 CRAVE 的 HISTORY。
- **要做 VLA 融合** → §2 的三份 SigLIP 注入文档。
- **判断某数字是否最新** → 一律以顶层 RESULT 为准(SigLIP-era 的 reach 1.67/deploy 0.753 等都是旧口径)。
