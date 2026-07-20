# CRAVE 迭代史 & 淘汰方案索引(agent 必读)

> **日期**: 2026-07-11
> **用途**: CRAVE 经历了大量消融/迭代,`docs/` 与 `experiments/` 里**混着最终方案和已淘汰方案**。
> 本文件是**唯一的"什么是最终 / 什么已淘汰"总索引**——防止 agent 误用旧代码/旧图/旧结论,
> 同时保留每个淘汰方案的脚本·文档·图路径,方便回溯与复用。
>
> **规则**:动手前先看 §1「当前权威」。凡 §2 列为"已淘汰/被取代"的方法、脚本、图、结论,
> **不要当作当前方案复用**;要复用其代码请按 §2 的路径找,但务必知道它为何被取代。

---

## 1. 当前权威(canonical)—— 只认这些

> ⚠️ **2026-07-20 scope 分界(先读)**:下面 §1.1 的 BGMM 管线只在 **kai0 任务A 的 value/AWBC 读出** scope 内是权威。
> **阶段代表帧/边界/世界模型目标/OOD 等"普适信号"用途,当前权威 = v2 复现密度场 r(o)**(r-谷=边界、**r-脊=阶段代表帧**,无聚类无K):
> 见 [recurrence_field_architecture_v2](recurrence_field_architecture_v2.md),唯一详源 `lmvla/lmwm/docs/RECURRENCE_UNIVERSAL_goals_and_roadmap.md`。
> BGMM 三大件(视觉聚类/coverage≥0.5/双锚 Viterbi)在 LIBERO 上已被证不普适(塌 M=1),**勿用于新数据集/新用途**。

### 1.1 最终 pipeline(一句话,scope=kai0 任务A value 读出)

**DINOv3-base(768D 冻结)→ pooled → PCA→128D ⊕ proprio 位置14(各自 L2, 能量 1:1)
→ 贝叶斯GMM(diag, 自适应K)+ per-mode 覆盖率≥0.5 → 簇成员帧 T 中位数(median)定 milestone 进度
→ 双锚 Viterbi(起点锚→0 / 终点锚→1, λ=16, 无 smooth · 无 norm01);
在线 = 单向因果 GRU 蒸馏【去阶梯 polyline】teacher(零未来 + 首帧 warmup)。**

### 1.2 权威文档(current)

| 文档 | 管什么(唯一真相) |
|---|---|
| [final_architecture](final_architecture.md) | **离线主线收口**:编码器/降维/聚类器/proprio/多峰/读出/双锚 全消融 + 最终配置。§2.11 踩坑记录、§2.12 polyline。 |
| [multitask_value](multitask_value.md) | **在线 multitask value**:共享 0.72M 跨任务;§3.6 DINOv3-base 对齐、§3.7 扩 12 任务。 |
| [online_readout_route](online_readout_route.md) | **在线因果读出**:三档(否决固定滞后 / forward-DP / 因果 GRU 蒸馏)。⚠️ teacher 最终用 **polyline**(见 §2 本表 D3)。 |
| [decoder_benchmark](decoder_benchmark.md) | **解码器最终基准**:检索最优(viz)、flow/L1 归档(合成);§6 base 解码、§6b xvla 覆盖分析。 |
| [cross_dataset_validation](cross_dataset_validation.md) | 跨数据集(xvla/coffee)最终泛化数。 |
| Web 报告 | `web/showcase/reports/crave_report/index.html`(已按最终架构更新)。 |

### 1.3 权威脚本(用这些, 都在 `experiments/`)

| 阶段 | canonical 脚本 | 产物 |
|---|---|---|
| 编码器注册表 | `src/crave/config/encoders.py`(`dinov3-base` 条目) | — |
| milestone 发现(kai 标签) | `gen_final_v3.py` | `temp/crave_final_v3.npz`(img⊕proprio, BGMM, median, coverage) |
| 双锚 Viterbi 标签(离线) | `gen_anchored_labels.py` | `temp/crave_ae_labels/final/`(真实 0→1) |
| 去阶梯 polyline 标签 | `gen_polyline_labels.py` | `temp/crave_ae_labels/polyline/` |
| 写数据集 | `write_crave_stage_datasets.py` | `kai0/data/Task_A/self_built/crave_stage_{A,B}/` |
| **在线单任务 GRU(最终)** | **`render_kai_online_gru.py`** | 142D img⊕proprio + **polyline teacher** + warmup(§2 D3 修正版) |
| base 特征抽取(多任务) | `extract_base_bank.py`(vis/coffee)· `kai_extract_base.py` · `aloha_extract_base.py`(pyav 解 AV1) | `lmvla/crave/data/*_dinov3base` |
| multitask 训练 | `train_multitask_base.py`(4 任务)· `train_multitask_12task.py`(12 任务) | 共享 value model |
| 解码器 | `train_decoder_base.py` · `decode_centroids_xvla.py` · `decode_milestone_centroids.py` | 簇中心/自重建解码图 |
| xvla 覆盖分析 | `xvla_coverage_analysis.py` | milestone 覆盖诊断图 |

> `experiments/` 共 ~180 个脚本,**绝大多数是一次性诊断/消融**(命名多为 `*_diag / *_test / tsne_* / *_sweep / *_compare / v4_*`),
> 不是"当前方案",复用前先对照本表。真正的当前方案只有上表这些。

---

## 2. 已淘汰 / 被取代方案 ledger(不要当当前方案复用)

> 每行:**方法 → 为何淘汰 → 被谁取代**,末列给旧脚本/文档/图路径(仍在仓库, 供回溯/复用)。
> 很多结论的数据依据见 [final_architecture](final_architecture.md) §2.11「踩坑记录」。

### A. 编码器 / 特征

| # | 已淘汰 | 为何 | 取代为 | 旧脚本/图 |
|---|---|---|---|---|
| A1 | **DINOv3-H (1280D)** | 更大反而相位更脏(Tstd 0.219 > base 0.195);multitask §2–3.5 早期用它 | **DINOv3-base 768D→PCA128** | `extract_xvla_dinov3h.py`; `temp/*dinov3h*`; multitask_value §3.6 是对齐修正 |
| A2 | Wan-VAE / SigLIP2 / DINOv3-7B 作编码器 | 外观/对齐空间 ≠ 任务相位(Tstd 0.27/0.28) | DINO 自监督 | `crave_wanvae_*.py`, `wan_decode_*.py`, `crave_full_7b_centroid.py` |
| A3 | raw 高维(不降维) | 65–95% 维是相位噪声 | PCA→128(甜点) | `pca_all_encoders.py`, `dinov2_scale_ablation.py` |
| A4 | grid/pixel 空间特征 | 按"布的位置"碎片化同相位 | **pooled 1×1** | `extract_grid_200.py`, `extract_30hz_*` |
| A5 | proprio 用速度 / 不用 proprio / 降权 | 速度瞬时含噪碎片化;proprio 是最强相位锚且首末构型不同 | **位置 14D, img:pos 能量 1:1** | `proprio_weight_sweep.py`, `proprio_weight_diag.py` |
| A6 | **milestone 发现用 img-only** | **单数据集 img-only PCA 只聚出 M≈3 → teacher 塌成对角线**(2026-07-11 本会话踩坑) | milestone 用 **img⊕proprio**(→M≈10),或 multitask 用**跨数据集 shared-PCA**(kai M=11) | 诊断图 `visualization/online_value/diag_kai_teacher_imgonly_M3.png` |

### B. 聚类器 / milestone 数

| # | 已淘汰 | 为何 | 取代为 | 旧脚本/文档 |
|---|---|---|---|---|
| B1 | **KMeans-K10 / K₀ 饱和 + 手调 K** | 早期 1280D 结论;img⊕pos+PCA128 近高斯后 GMM 反超(Tstd 0.176→0.141) | **BayesianGMM(diag,Dirichlet)自适应K** | [clustering_method_comparison](clustering_method_comparison.md); `cluster_method_*.py`, `milestone_count_sweep.py`, `tsne_1280D_k*.py` |
| B2 | **EM-HMM 统一概率框架** | hmm-cluster 在 768D collapse | BGMM + Viterbi 分离 | [em_hmm_negative_result](em_hmm_negative_result.md); `em_hmm_vs_kmeans.py`, `milestone_hmm.py` |
| B3 | HDBSCAN | 高维崩塌 / O(n²) | BGMM | `cluster_method_full.py` |
| B4 | milestone 进度用 mean / mode | mean=幽灵值(如 0.455 无帧)、mode 排序倒挂 5 处 | **median(0 倒挂)** | `crave_milestone_isotonic.py`, `crave_milestone_order.py` |

### C. 读出(离线)

| # | 已淘汰 | 为何 | 取代为 | 旧脚本/文档 |
|---|---|---|---|---|
| C1 | **per-ep norm01** | 把每条曲线各自拉满 → 掩盖达顶失败(raw 峰值 median 仅 0.32,52% ep raw<0.5) | **双锚(起点→0/终点→1)** | final_architecture §2.10/§2.11; 旧 `crave_ramped_value.py` |
| C2 | **SymVote(Viterbi-free)** | 当前 12 粗 milestone+3Hz upsample+因果 下 corr 上限 ~0.83,持续别名击穿 | 离线 = **双锚 Viterbi**;在线另见 D | [sym_adaptive_vote](sym_adaptive_vote.md); `gen_viterbi30_labels.py`, `crave_ep2302_30hz_final.py` |
| C3 | 硬阶梯 step 当唯一输出 | 过渡生硬;监督 GT 本身是平滑 ramp | **可选 polyline 去阶梯**(corr 0.957 反超阶梯 0.944) | final_architecture §2.12; `gen_polyline_labels.py` |

### D. 读出(在线 / 因果)

| # | 已淘汰 | 为何 | 取代为 | 旧脚本/图 |
|---|---|---|---|---|
| D1 | **固定滞后 Viterbi** | 要输出第 t−L 帧必须先见第 t 帧 = **偷看 L 帧未来**,非完全在线 | forward-DP(①②)/ 因果 GRU(③) | online_readout §0;图 `visualization/online_value/online_fixedlag_9ep.png` |
| D2 | **零训练 forward-DP 路线(①对称 + ②非对称前向偏置)** | 只靠手写因果先验补未来 → 末段登顶抬不上去(①末值泊 0.5/corr 0.83;②甜点末值封顶 ~0.82/corr 0.86)。**整条零训练路线已弃用**,不再作为并列上线选项(2026-07-11 本会话) | **因果 GRU 蒸馏 polyline teacher(③)** = `render_kai_online_gru.py` | online_readout §1;图 `visualization/online_value/online_vs_offline_curves.png` |
| D3 | **阶梯-teacher GRU 蒸馏** | 蒸硬阶梯 → student 需后处理平滑;**最终蒸【去阶梯 polyline】teacher**(2026-07-11 本会话修正) | **polyline-teacher GRU** = `render_kai_online_gru.py`(142D img⊕proprio + warmup) | 旧 `train_online_gru.py`(`offline_teacher`=硬阶梯);旧图已被 `visualization/online_value/gru_polyline_heldout.png` 取代 |

### E. 解码器 / 簇中心

| # | 已淘汰 | 为何 | 取代为 | 旧脚本/文档 |
|---|---|---|---|---|
| E1 | GAN / patch decoder | 凭空编造细节(幻觉) | flow(合成归档)/ L1(高一致) | `crave_patch_gan.py`, `crave_patch_decoder.py` |
| E2 | GDL 再编码一致性损失 | 训练失败 / 对抗噪声 | 同上 | decoder_benchmark; `roundtrip_consistency.py` |
| E3 | **KMeans15-无覆盖门槛 抽 xvla 簇中心** | 非最终聚类方案(给 10 个 milestone, 与部署不一致) | **最终方案(1000采样+shared-PCA+BGMM+cov≥0.5=M7)** | `decode_centroids_xvla.py`(已是最终版) |
| E4 | 簇中心平均图当 viz | 平均 → 软/鬼影手臂 | **检索(最近真实帧)** cos 0.84 | [milestone_centroid_decoding](milestone_centroid_decoding.md); `retrieval_decode_gallery.py` |

### F. 早期整体路线 / 数据集

| # | 已淘汰 | 为何 | 取代为 | 旧路径 |
|---|---|---|---|---|
| F1 | 连续 value 路线 | 转离散 milestone | 离散主线 | `docs/archive/cross_episode_recurrence_value_CONTINUOUS.md` |
| F2 | 3-path(img/proprio/wan 分路)/ 3-level 分类 | 合并为 joint 142D + progress | joint | `crave_full_3path*.py`, `crave_freq_3way.py`, `crave_3level_*.py` |
| F3 | **旧 norm01 版 crave_stage_{A,B} 数据集** | 用了已弃的 norm01(见 C1) | 需按双锚重生成(见 final_architecture §5, **待办**) | `kai0/data/Task_A/self_built/crave_stage_{A,B}/`(旧) |

---

## 3. 历史/过渡文档一览(仍在 `docs/`, 顶部已加历史横幅)

| 文档 | 状态 | 当前替代 |
|---|---|---|
| [cross_episode_recurrence_value_METHOD](cross_episode_recurrence_value_METHOD.md) | 离散主线 V2.4(叙事仍有用) | 收口见 final_architecture |
| [viterbi_computation](viterbi_computation.md) | DP 计算详解仍有效;**固定滞后附录已否决** | 在线见 online_readout(§2 D1) |
| [clustering_method_comparison](clustering_method_comparison.md) | KMeans-era 对比 | final_architecture §2.5(B1) |
| [sym_adaptive_vote](sym_adaptive_vote.md) | 在线投票探索 | online_readout(C2/D) |
| [em_hmm_negative_result](em_hmm_negative_result.md) | ❌ 负结果 | final_architecture(B2) |
| [greedy_vs_maxprod_aliasing](greedy_vs_maxprod_aliasing.md) | 别名诊断(概念仍参考) | — |
| [milestone_centroid_decoding](milestone_centroid_decoding.md) | 早期解码方案对比 | decoder_benchmark(E) |
| [encoders](encoders.md) | 编码器综述 | final_architecture §2.1(A1) |
| `docs/archive/` | 更早探索索引 · 网站大纲 · 连续 value | — |

---

## 4. 本会话(2026-07-11)修正的两处易错点(给未来 agent 的警示)

1. **在线 GRU 的 teacher 是【去阶梯 polyline】,不是硬阶梯**(D3)。用 `render_kai_online_gru.py`,别用 `train_online_gru.py` 的 `offline_teacher`。
2. **milestone 发现必须带 proprio(或跨数据集 shared-PCA)**(A6)。kai 单数据集 img-only 只出 M=3,teacher 会塌成对角线;img⊕proprio 出 M≈10,teacher 才有真实阶段结构。
