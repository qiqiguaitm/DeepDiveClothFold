# LMWM 文档体系(索引)

> LMWM = Latent Milestone World Model,面向 kai0 π0.5 VLA 的 milestone+1 价值层子目标预测器。
> 本文件是**唯一入口**:先看这里,按需下钻。**当前版**在顶层,**历史版**在 `archive/`。
> 最近整理:2026-07-08。

---

## 🎯 想快速了解 → 按这个顺序读

| # | 文档 | 一句话 |
|---|---|---|
| 0 | [`REDESIGN_LMWM2_2026-07.md`](REDESIGN_LMWM2_2026-07.md) | ⭐ **重设计方案 LMWM-2**(2026-07-08;三红线:弃检索/部署零CRAVE/结构入表示;价值几何化码空间——价值=方向、达成=码距、验证=MDN自身密度;部署仍两网络34M;SR 裁决计划 + vs 现 LMWM 对比定案) |
| 1 | [`FINAL_REPORT.md`](FINAL_REPORT.md) | **最终架构 + 各指标含义 + LaWM baseline 对比**(先看这个) |
| 2 | [`FINAL_CROSSTASK_PREDICTOR.md`](FINAL_CROSSTASK_PREDICTOR.md) | **最优跨任务预测器**(多任务联合、teacher=proto 簇中心、交付 ckpt) |
| 3 | [`ARCHITECTURE_AND_BASELINE.md`](ARCHITECTURE_AND_BASELINE.md) | **最终架构总表 + 参数量 + I/O + vs LaWM 指标一览**(单页速查) |
| 4 | [`PITFALLS_AND_HISTORY.md`](PITFALLS_AND_HISTORY.md) | **版本演进史 + 踩坑/错误路径**(避免重复犯错,踩坑前必看) |

## 📚 专题(深挖时看)

| 文档 | 范围 |
|---|---|
| [`ABLATION_CONVERGENCE_2026-07.md`](ABLATION_CONVERGENCE_2026-07.md) | 所有控制变量消融全表(teacher/anchor/fwd_arch/lift/code_dim/center_w/pred_input/proto) |
| [`RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md`](RESEARCH_DIRECTION_milestone_universal_fusion_2026-07.md) | 研究方向:普适 milestone、身份多峰、命名与模块、融合调研 |
| [`PROGRESS_lawm_comparison_2026-07.md`](PROGRESS_lawm_comparison_2026-07.md) | LaWM 官方 ckpt 在我们数据上的实测对比 |

## 🗄️ 历史 / 已归档

`archive/` 下是被上面的当前版**取代**的旧文档(阶段计划、早期架构、单点诊断、迭代日志)。**内容已提炼进 `PITFALLS_AND_HISTORY.md` 的时间线与踩坑表**,除考古外不必读。见 [`archive/README.md`](archive/README.md)。

---

## 🧭 一句话现状(2026-07-08)

- **最终架构**:π0.5 SigLIP 冻结共享塔 → 预测器(teacher 出码 / MDN 部署)+ 生成器(AdaLN,当前 grid 当画布)→ milestone+1 子目标 grid。部署主体 ~34M CNN。
- **teacher 定案 = proto(簇中心)**:码 = 下一 milestone 的 SigLIP 中心固定投影;predm 蒸馏它;生成器渲染到当前画布。与 inv(逆向动力学)打平但更简/更轻/开放词表。
- **跨任务**:多任务并集联合训练(kai0/coffee/xvla[/vis]),一个模型同时干 3 种任务(mean deploy ~0.75)。
- **vs LaWM**:reach 1.67s > 1.48s、参数轻 ~10×、与 VLA 同塔嵌入更深;唯一未覆盖 = 下游 SR。
- **下一步(唯一真缺口)**:接 π0.5 action expert 测 action-MAE / 真机 SR。

## 🔧 关键产物(代码/模型/数据)

- 训练:`lmwm/scripts/train_multitask.py`(多任务 3锚 3teacher)、`train_twomodel_v2.py`(单任务)、`train_ablation.py`(消融)
- 模型:`lmwm/checkpoints/teach_proto_3task.pt`(⭐推荐)、`teach_proto_4task.pt`、`final_crosstask_{3,4}task.pt`(inv 对照)
- bank:`temp/{crave_full_dinov3h(kai0),coffee_dinov3h,xvla_dinov3h,vis_dinov3h}` + `lmwm/data/recurrence_graphs/*`
- bank 构建:`make_visbase_dinov3h_index.py`(通用)、xreb 缓存捷径(coffee/xvla)
- 网站 showcase:`web/showcase/reports/lmwm_final/`
