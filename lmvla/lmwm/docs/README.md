# LMWM 文档体系（索引）

> LMWM = Latent Milestone World Model,面向 kai0 π0.5 VLA 的 milestone+1 价值层子目标预测器。
> 本文件是**唯一入口**。**当前最终版在顶层,历史/被取代/未来阶段的文档在 [`history/`](history/) 且由 [`HISTORY.md`](HISTORY.md) 索引。**
> 最近整理:2026-07-12（收口到统一 DINOv3-base 版,历史归档 + 索引化)。

---

## ⭐ 当前最终方案（只读这两份 = 单一事实源）

| 文档 | 一句话 |
|---|---|
| [`RESULT_newcrave_final_arch_2026-07-11.md`](RESULT_newcrave_final_arch_2026-07-11.md) | **统一 DINOv3-base LMWM**:全链路一个编码器(挖矿/预测器/生成器/teacher/解码同空间)· proto teacher 码=簇中心的 **shared PCA128** 投影 · deploy 0.910/id3 0.940 · 闭环可视化 · 修复表 · teacher 码三方消融(§4.6)|
| [`DECODER_dinov3base_video_2026-07-11.md`](DECODER_dinov3base_video_2026-07-11.md) | 可视化解码器:**pooled(软) + grid(锐)** 两法 + 视频流时序一致 |

**运行**:`train_multitask.py --encoder dinov3base --teacher proto --teacher_code shared_pca`
**交付 ckpt**:`lmwm/checkpoints/dinov3base_lmwm_sharedpca_kaicoffee.pt` + `dinov3base_decoder/kai_grid_dec.pt`
**web 展示**:`web/showcase/reports/lmwm_final/index.html`

---

## 🗄️ 历史 / 被取代 / 未来阶段 → 全在 [`history/`](history/),索引见 [`HISTORY.md`](HISTORY.md)

⚠️ **动手前先读 [`HISTORY.md`](HISTORY.md)**,防照搬旧方案/旧代码(SigLIP-era 的"FINAL/CROSSTASK/REPORT"、reach 1.67s、deploy 0.753 等都是**旧口径**,已被顶层 RESULT 取代)。HISTORY 分四类:
- 🔴 **已被取代**(SigLIP-era 旧最终:LMWM2_FINAL / ARCHITECTURE_AND_BASELINE / FINAL_CROSSTASK / FINAL_REPORT / REDESIGN / ABLATION / PROGRESS_lawm / PLAN)——标注了各自"可打捞"的内容
- 🟡 **未来阶段参考**(SigLIP 融合,尚未做:MASTER_PLAN / INJECTION_DESIGN / INJECTION_DEEP_ANALYSIS)
- 📌 **仍有价值**(PITFALLS_AND_HISTORY 踩坑表 · RESEARCH_DIRECTION)——**踩坑前必查**
- 📦 **更早期归档**([`history/archive/`](history/archive/) 27 docs,2026-07-01~04 探索期)

CRAVE 侧对应索引:[`../../crave/docs/HISTORY.md`](../../crave/docs/HISTORY.md)。
