# CRAVE 簇中心解码 与 解码器对比试验

> 2026-06-18/19。把 milestone 簇中心从"取最近真实帧"扩展为"**解码出合成质心图**",并系统对比各编码器/解码器。
> **图集统一放** `docs/visualization/cross_episode_recurrence_value/centroid_decoder/`(本文件所有图引用均在该子目录)。
> 选型结论(标准配置)单列见 [`centroid_representation_config.md`](centroid_representation_config.md);完整实验留痕见 [`center_representative_research.md`](../../../../visualization/cross_episode_recurrence_value/center_representative_research.md)。

---

## 1. 背景与目标
milestone 代表图原来用"离簇心最近的真实帧(medoid)"。本线探索:**能否训一个解码器,把簇中心(latent)直接解码成一张合成质心图**,作为去具体化、不绑单 episode 的"可读原型"。核心要回答两件事:① 用什么编码器/解码器最好;② 合成质心能否替代/补充 medoid。

## 2. 解码器对比试验(实验线)

| 阶段 | 方法 | 结论 | 图(在 `centroid_decoder/`) |
|---|---|---|---|
| 2.1 | **池化向量解码**(mean-pooled latent → 图) | 鬼影/糊(池化丢空间)→ 基线否定 | `crave_center_decoder_compare.png` / `_recon.png` |
| 2.2 | **patch-token 空间解码**(16×16 网格 → 图) | **重建变清晰**(保空间);但簇内 grid 平均仍软 | `crave_patch_decoder_compare.png` / `_recon.png` |
| 2.3a | 刚性对齐再平均(掩码质心+主轴) | 比朴素平均好,仍软原型(可形变布料平均 ill-posed) | `crave_aligned_centroid.png` |
| 2.3b | pix2pix-GAN(含低频 L1 修复掉色) | 单帧/medoid 更锐;簇平均仍软;对抗会掉色(已修) | `crave_patch_gan_compare.png` / `_recon.png` |
| 2.4 | **规模消融 A–I**(编码器 22M→300M、解码器 0.76M→12.8M、数据 9k→24k 全顶满) | **加规模救不了簇中心**:单帧 medoid 冲到 234,簇中心始终 80–112 → 瓶颈是"平均输入 ill-posed",与规模无关 | `crave_scale_ablation_AtoI.png` |
| 2.5a | **解码器倒 U 选型**(固定 large+9k:tiny→small→medium→big→xl) | 簇中心结构 52 / **112(small 峰)** / 107 / 79 / 88 → **small(0.92M)最优**(太小糊、太大噪) | `crave_large_decoder_ladder.png` |
| 2.5b | 编码器选型 + 可读候选 | small→base→**large** 结构 92→89→112,large 最优;可读候选 B/D/E | `crave_readable_centroid_candidates.png` |

**两条硬结论**:① "清晰的平均质心"对可形变布料**数学上 ill-posed**,加规模/换损失都救不动;② 清晰合成只能来自**单帧**(medoid 渲染),本质是美化 exemplar。→ 代表图最清晰仍用 **medoid(最近真实帧)**;合成质心定位为"平滑可读原型"。

> 附:挖矿 episode 数对 value 质量的影响(~100–200 ep 饱和)`crave_mine_episode_sweep.png`。

## 3. ep2302 30Hz 端到端演示(本标准配置)
脚本 `train_scripts/kai/data/crave_ep2302_30hz_decoded.py`。全程 DINOv2-large 编码 + small 解码器:

- **自适应 milestone**(KMeans-96→覆盖率→Otsu)+ **自适应 value bins**(bins=milestone 位置,value↔milestone 精确对应)+ **报告式平滑**(`smooth_monotone`,fps 标定移动平均)。
- 产物:`crave_ep2302_30hz_decoded.png`(value 曲线 + milestone 变化 + 每 milestone 簇中心解码图)、`crave_ep2302_30hz_decoded.mp4`(左相机 / 右上 value 游标 / 右下解码质心,**并行 compose + libx264** 编码,**逐帧 2960 全验证 ALL_PASS**)、`crave_ep2302_transition_3row.png`(跳变帧四行:原始 / encode-decode 重建 / 簇中心解码 / 最近真实 medoid)。

## 4. 标准配置(定稿)
**DINOv2-large 编码器 + small(0.92M)空间解码器 + patch-grid 输入 + L1 + 9k + 自适应 milestone + 自适应 value bins**。详见 [`centroid_representation_config.md`](centroid_representation_config.md)。

## 5. 图集清单(`centroid_decoder/`)
- 解码器对比:`crave_center_decoder_*` / `crave_patch_decoder_*` / `crave_aligned_centroid` / `crave_patch_gan_*`
- 规模/选型:`crave_scale_ablation_AtoI`(A–I 总图)+ 各 per-config tag / `crave_large_decoder_ladder` / `crave_readable_centroid_candidates`
- 演示:`crave_ep2302_30hz_decoded.{png,mp4}` / `crave_ep2302_transition_3row.png`
- value/数据:`crave_mine_episode_sweep`

---

## 6. milestone 聚类 / 排序 / value 方法定稿(2026-06-20)

> **最终架构(TL;DR)**:**DINOv2-large 图像 ⊕ proprio 聚类(语义 milestone + 起末消歧)→ precedence 定序 + isotonic 度量 value(排序正确、保信息)→ Wan2.2-VAE 渲染 medoid(锐利代表)**。Wan 只做渲染,不做编码器/聚类。

### 6.1 特征:DINOv2-large 图像 ⊕ proprio(消起末别名)
**问题**:纯图像特征(DINOv2-large)有**起末视觉别名** —— 折好的布(紧凑平整方块)≈ 摊平的布,DINOv2 难分 → 折好态被吸到早期 milestone,value 到不了 1.0(实测 ep763 0.15、ep1527 0.32,均为标准完整折)。
**修复**:聚类/value 特征 = **DINOv2-large 图像(1024, L2)⊕ proprio(state+Δstate 28维, z-score+L2)**(等权)。proprio(臂/夹爪状态)靠"臂伸入 vs 收回"消歧,即使图像别名。全量 3055ep 实测(`crave_full_3path.py`):**ep763 0.15→1.00、ep1527 0.32→1.00、ep2302→1.00**。生产 `crave_value.py` 三路(raw⊕armmask⊕proprio)亦验证 ep763→1.00。
**渲染只取图像那一路**:medoid = 离三路簇心最近真实帧 → 取其真实图像 → Wan 解码(不碰 proprio)。图 `crave_3path_value_test.png` / `crave_3path_gallery.png`。
**对照否决 Wan 编码器**:把 Wan-latent⊕proprio 套同架构(`crave_full_3path_wan.py`,GPU KMeans 加速 12316 维)—— 虽也到 1.0,但**排序逆序 54(vs DINOv2 ~13)、ep2302 value 卡 0.27 平台到 90% 才暴冲**(Wan-latent 偏外观、中段进度跟不动)→ **不采纳**。图 `crave_3path_wan_value.png` / `crave_3path_wan_gallery.png`。

### 6.2 排序 + value:precedence 定序 + isotonic 度量 value
**问题**:milestone 排序/value 原按"逐 ep 时间统计"(`Pk=首达中位` / `tpos=均值`),是**时间分位非因果先后** → 进度相近的相邻 milestone 前后颠倒(实测 13 逆序对,但 Kendall-τ 0.96、零大错位 → 真实但局部)。
**方案(排序与 value 量纲解耦,保信息)**:① 排序用**跨 ep precedence**(成对 `首达(A)<首达(B)` 比例,Copeland 聚合;只看 ep 内相对先后,**对节奏/归一化不变 → 鲁棒可泛化**);② value 保留度量量纲 `Pk`,沿 precedence 序做 **isotonic(PAVA)保序回归**——仅把逆序/并发的几个并到单调,其余精确保留;③ 并发态(环)自然并成平局。
**实测(全量 34 milestone)**:value 被改 16/34 但**单点最大动 0.026**、**所有大 advantage 间距精确保住**;逐帧读出 old vs new 几乎重合(Viterbi-DP 本就单调,**readout-neutral**)。脚本 `crave_milestone_order.py` / `crave_milestone_isotonic.py` / `crave_milestone_value_test.py`,图同名 + `crave_milestone_order_strip.png`,顺序表 `temp/crave_full/milestone_order_info.json`。caveat:小数据集设最小共现阈值 + 向时间序 shrinkage。

### 6.3 渲染框架:Wan2.2-VAE medoid(否决 RAE / 统一 latent)
代表图锐利靠 **Wan2.2-VAE 单帧重建**(L1 0.003,照片级);合成"平均质心"对可形变布料 ill-posed(任何解码器都软,§2 已证)→ 代表图用 **medoid**,Wan 仅渲染。
**为何不用统一单 latent**:Wan-latent 聚类按外观(corr 0.54/混相位 50%,§6.1 再证);VA-VAE/REPA-E 对齐稀释语义;V-JEPA2 无解码器(§7.1);**RAE**(冻结 DINOv2+ViT 解码器,调研排名第 1)实测**重建坏**(16×16 网格 L1 0.063,官方 `bytetriper/RAE` 代码逐项复现仍同——`Dinov2withNorm` 关 layernorm 仿射 + 剥 CLS+4register 后仍逐位相同,解码近乎与 latent 无关)→ 渲染 ≪ Wan 且聚类零增量,**否决**。ckpt 经 gf3 加速下载(见 [[reference_gf3_fast_download]])。证据图 `crave_wanvae_*`,调研 task `whcg24l2y` / `wzge796p5`。

## 7. 未来规划

> 以下两节由 deep research(2026-06-19,100 agents / 18 源 / 对抗校验)定调。结论先行,后续按"便宜首验"推进。

### 7.1 V-JEPA 2 编解码器测试 —— **结论:聚类/解码不用换;只在做时序预测时评估 V-JEPA 2-AC**
- **V-JEPA 2 / 2.1 是 latent-only,无原生像素解码器**(masked latent 预测,encoder + predictor)。要出簇中心**图**仍得**另训一个像素解码器**(Meta 自己也是单独训了个 ViT-L decoder 仅作可视化)→ 在"解码"这步 V-JEPA 2 **不省事**,相对 DINOv2-large + 训练解码器无优势。
- **dense 语义聚类 DINO 系仍更强**(ADE20K seg 47.9 vs 55.9 mIoU);V-JEPA 2.1 仅在 depth 上追平 DINOv3;"视频编码器聚类胜 DINOv3"的说法被**否决(0-3)**。→ **milestone 聚类 + 簇中心解码:保持 DINOv2-large + small 解码器(现标准配置)。**
- **V-JEPA 2 唯一差异化价值 = V-JEPA 2-AC**(动作条件世界模型,潜空间预测/规划)→ **只在 7.2 要做"时序 milestone 预测 / 世界模型规划"时才值得评估**,不是为聚类/解码。
- 工程:V-JEPA 2 ckpt 在 HF(`facebook/vjepa2-*`),但本机 HF/镜像被墙(同 dinov2-large,需另找权重)。
- **便宜首验(若评估)**:用 V-JEPA 2-AC 在 latent 空间预测 ep 的下一段 → 比对 CRAVE milestone 序列,看时序预测是否比 DINOv2 逐帧聚类更早/更准锚定相位;**不替换现有聚类**,只作时序预测候选。

### 7.2 milestone 预测赋能 VLA —— **结论:可行,但作"补充"而非替代;Design 1 优先 + 强制子目标过滤**
范式成熟(子目标/目标图条件化:SuSIE / UniPi / GR-2 / V-JEPA 2-AC),所以"预测 milestone+1 子目标条件化 VLA"**可行**。两设计实测建议:

- **Design 1 · goal-image(解码 milestone+1 簇中心 → 目标图喂 VLA)** —— **首选、范式最成熟、最低风险**。
  - ⚠️ **关键风险**:我们的解码质心是**平滑软原型**(非写实),生成子目标带伪影/非真进度会**拖垮**低层策略(已证 3-0)。→ **必须配 GHIL-Glue 式子目标过滤器**(过滤"无进度/有害"子目标),否则适得其反。
- **Design 2 · latent(milestone+1 簇中心 latent 作子目标 embedding 注入 VLA)** —— 更省(无需解码),但**前提是 milestone latent 在 VLA 编码器空间**。我们 milestone 用 DINOv2-large,pi0/pi05 用别的视觉编码器 → **不共享** → 需让 **milestone 聚类改用 VLA 同款编码器** 或 **训一个对齐投影**(额外工程)。
- **诚实红线**:"子目标图条件 > 语言/纯 BC"被**否决(0-3)** → milestone 条件化**不一定**优于我们现有的**稠密 value(优势加权 BC)** → 定位为**补充**(尤其长程/多模态 cloth fold),先验证有无增量,别当替代。

**便宜首验(各设计)**:
1. **Design 1**:离线用 **GT milestone+1 的解码图**当目标图(先用 GT、不用预测,排除预测误差),训 goal-conditioned BC vs 无目标 BC,sim01 比成功率 + 套 GHIL-Glue 过滤。增量为正再上"预测 milestone+1"。
2. **Design 2**:把 milestone **聚类改用 VLA 的编码器**(SigLIP/DINO,与 pi0 对齐)→ 子目标 latent 天然在 VLA 空间 → 注入一个子目标 token,同样先用 GT milestone+1。
3. 两者都先回答"**子目标条件化相对现有稠密 value 有没有正增量**";有,再投入预测器 + 过滤器工程。

> 详细引用见 deep research 输出(task `wzge796p5`)。

### 7.3 仿真验证方案 —— **验"方法有效性",与叠衣域解耦**
**原则**:milestone/子目标条件是**跨任务通用**方法,先在标准 VLA 基准上验"有没有效",不必迁就 cloth;验通再回真机 kai0(叠衣)验域迁移。

**选定环境:LIBERO-Long(openpi/pi0 原生闭环,本仓已接好)**——长程多阶段(milestone 天然有意义)+ pi0 闭环零集成。
- 现成资产:`kai0/src/openpi/policies/libero_policy.py`(pi0 LIBERO 策略)、`fastwam/experiments/libero/`(`run_libero_manager.py` / `eval_libero_single.py` / 并行评测 + `libero_{uncond,idm,joint}_2cam224` configs)、`xvla/X-VLA/evaluation/libero/`。
- 为何不选 cloth sim 首验:DexGarmentLab/SoftGym 等重/慢/域特化,**方法有效性**不需要布料;布料 sim(DexGarmentLab)留作最终域确认。

**对照臂(干净隔离"子目标条件是否有正增量")**:
- **A0 baseline**:原版 pi0(无子目标)。
- **A1 = Design-1**:pi0 + **GT milestone+1 解码图**作目标图(先用 GT、不预测 → 上界,排除预测误差)。
- **A2 = Design-2**:pi0 + **GT milestone+1 latent**子目标(用 pi0 同款编码器抽 milestone 特征,保证 latent 同空间)。
- **A3(若 A1/A2 有正增量再做)**:学一个 milestone+1 预测器 + **GHIL-Glue 式子目标过滤**,换掉 GT。

**流程**:CRAVE 套到 LIBERO demo(逐帧特征→聚类→milestone→每帧取 milestone+1 子目标)→ 训 A0/A1/A2(集群,长训不走本地 2 卡)→ LIBERO-Long 闭环比**成功率 + 子阶段通过率**。
**判据**:A1/A2 > A0 → 方法有头部空间 → 投预测器(A3);A1/A2 ≈ A0 → 子目标条件在此不优于纯 BC,回退/只做稠密 value。**先答"GT 子目标有没有用",再谈预测**。
