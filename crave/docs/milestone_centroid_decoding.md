# CRAVE 簇中心解码 与 解码器对比试验

> 2026-06-18/19。把 milestone 簇中心从"取最近真实帧"扩展为"**解码出合成质心图**",并系统对比各编码器/解码器。
> **图集统一放** `docs/visualization/centroid_decoder/`(本文件所有图引用均在该子目录)。
> 选型结论(标准配置)单列见 [`centroid_representation_config.md`](centroid_representation_config.md);完整实验留痕见 [`center_representative_research.md`](visualization/center_representative_research.md)。

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

## 2b. DINOv3-H 解码器 · pooled vs patch-grid 保真对比(2026-07-02)

> LMWM 侧需要"把预测的 latent 解回图像"。解码器本身属 CRAVE 方法,故结论沉淀于此;LMWM 侧的具体应用(解码预测 latent)见 `lmwm/docs/decoder_blur_diagnosis_20260702.md`。
>
> **目标澄清**:不是要"清晰",而是要**解码保留原图相同信息(保真)**。据此重排三种解码器,held-out 单帧自重建(decode 帧自己的 latent vs 真实帧,L1↓ 越忠实):

| 解码器 | 输入表示 | 自重建 L1 | 定位 |
|---|---|---|---|
| pooled L1 | DINOv3-H pooled 1280 | 6.2% | 颜色对但丢空间细节、糊(L1 求均值天生糊) |
| pooled GAN(PatchGAN+hinge) | 同上 | ~7% | 锐但**幻觉+掉色**(粉→tan)→ **保真更差,弃** |
| **DINOv3-H patch-grid**(CRAVE 空间解码器) | **patch grid 1280×16×16** | **2.7%** | **~2.3× 更忠实**,颜色/形状/臂位全保住 |

**结论**:与 §2 一致并量化 —— **保真的杠杆是"换表示(pooled→patch-grid)",不是换损失/加规模**。pooled 把 16×16 patch 平均掉 → 空间信息损失 → 自重建天花板 ~6%;patch-grid 保住空间 → 2.7%。GAN 用幻觉换锐度,反伤保真(印证 §2.4:平均输入 ill-posed,单帧才能清晰;而"忠实"只能靠保住空间的 patch-grid)。

- 复现:`crave.encoders.load_encoder("dinov3-h").encode_grid(imgs)` → (N,1280,16,16);`crave.decoding.decoder.train_dec` 训 small 空间解码器。脚本 `lmwm/scripts/train_patch_decoder.py`;数据 `lmwm/outputs/patch_decoder/{summary.json,patch_recon_compare.png}`。
- 数值:DINOv3-H patch-grid recon L1 2.7%,与 §2.2/§2.4 早先 DINOv2 patch 版 recon L1 ~2.9% 同量级 —— 换编码器到 DINOv3-H 结论不变。

## 3. ep2302 30Hz 端到端演示(本标准配置)
脚本 `train_scripts/kai/data/crave_ep2302_30hz_decoded.py`。全程 DINOv2-large 编码 + small 解码器:

- **自适应 milestone**(KMeans-96→覆盖率→Otsu)+ **自适应 value bins**(bins=milestone 位置,value↔milestone 精确对应)+ **报告式平滑**(`smooth_monotone`,fps 标定移动平均)。
- 产物:`crave_ep2302_30hz_decoded.png`(value 曲线 + milestone 变化 + 每 milestone 簇中心解码图)、`crave_ep2302_30hz_decoded.mp4`(左相机 / 右上 value 游标 / 右下解码质心,**并行 compose + libx264** 编码,**逐帧 2960 全验证 ALL_PASS**)、`crave_ep2302_transition_3row.png`(跳变帧四行:原始 / encode-decode 重建 / 簇中心解码 / 最近真实 medoid)。

## 4. 标准配置(定稿)
**DINOv2-large 编码器 + small(0.92M)空间解码器 + patch-grid 输入 + L1 + 9k + 自适应 milestone + 自适应 value bins**。详见 [`centroid_representation_config.md`](centroid_representation_config.md)。

> **⚑ 2026-07-03 统一基准更新 → 最优"latent→图"方案 = 检索(最近真实帧)**:120 帧自重建、同一套指标横评检索/patch/pooled-L1/pooled-GAN。**检索在语义保真(再编码 cos 0.84)+ 锐度(940≈真实)双赢**,合成解码器语义保真**结构性封顶 ~0.47**(GAN 锐 833 但幻觉、cos 仅 0.47;patch L1 最低但软、cos 0.42;L1 指标偏爱模糊会骗人)。合成只在"必须造 demo 集外新状态"时用 patch-grid。完整表 + 图 + 决策矩阵见 **[`decoder_benchmark.md`](decoder_benchmark.md)**。

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

## 6.4 循环簇 + 操作回退 调查与处理(2026-06-20)

> 用户提出两个边界问题,"执行边验证边回退"。脚本 `crave_cyclic_detect.py` / `crave_multimode_test.py` / `crave_regression_test.py` / `crave_condend_validate.py` / `crave_truncate_test.py`。

### 循环簇(任务内重复动作)—— 真实,多模态可缓解(小幅)
- **量化**:用"**同一 ep 内访问 ≥2 次**"(分离段)而非跨 ep 时间双峰(后者会把 partial-start 污染误判为循环)→ **119 个真循环簇**(avg 2–3.5 次/ep、最高 88% 的 ep 重复),其中 **104 个被纯度闸(tstd≤P60)丢掉** → 框架在扔重复态。图 `crave_cyclic_detect.png`。
- **相对 value 规则(采纳用户设计)**:循环簇放**多个 value 锚点**(成员时间 GMM 多峰);读出时"**取 >当前 value 的最小模式**(无则保持当前)"= **沿循环模式单调向前匹配**,用当前进度作上下文 → 相对 value(第几次迭代)。比"多模态 DP 自由选 bin"稳(**杜绝伪回退**)。caveat:此规则假设向前 → 须被回退检测门控。
- **多模态放置验证**:advDensity **+0.05 均**(循环密集 ep0/3054 +0.08/0.09),**成功 ep 仍到 1.0 无回退**。图 `crave_multimode_test.png`。**已 test 验证,未落生产**(增益小,待定)。

### 操作回退/失败 —— 走过的弯路 + 最终解(解耦 flag)
- **弯路①:end_bonus 条件化(cond_end)被证伪**。先以为"end_bonus 无条件拉 value→1"是真凶,加了 `cond_end`(默认 False)。但 **`crave_condend_validate.py` 回退测试 ON=OFF 无差异**(3-path 里 end_bonus 太弱),且 **`crave_truncate_test.py` 反证 cond_end 是错的**:它用 residual **去压 value**,把 ep763 截 90%(布已折好 ~90%、value 应 ~0.9)**误压到 0.35**——把"末帧非最终态"当"低进度"。→ **cond_end 弃用(留作 off 选项),不能用 residual 压 value**。
- **弯路②:reverse-replay 早期结论部分是测试 bug**。倒放只反帧序、proprio Δstate 没反 → 特征错乱。修正(按轨迹重算 proprio)后 3-path 行为混合:ep2302 倒放 → 0.25(✓降)、ep763 停 0.85(✗,转移惩罚所致)。
- **根因 = 回退/失败态 OOD**:布料"反向展开"等动作不在成功 demo 词表 → milestone 匹配只为 in-distribution 成功态设计 → 失败态跟踪不可靠。**非调参可解**。

### ✅ 最终解(已落地):progress value 与 failure flag **解耦**
`crave_truncate_test.py`(截断成功 ep = 未完成 rollout)验证决定性:**progress value 正确停在真实进度(截断曲线贴合完整成功曲线),alignment-residual 干净分离完成/未完成**。
- **已落 `crave_value.py::DiscreteValue.status(a,r,s)` / `value(..., ret_status=True)`**(与 cond_end 解耦,`de_end_thr` 永远算):
  - `is_complete` / `complete_conf∈[0,1]`:末帧到 endK(完成态)残差 `de_end` vs 阈 `de_end_thr`。实测 thr=1.16,**截断 de_end 1.37–1.81 判未完成、完整 0.66–0.85 判完成**,阈值清晰且分级(ep2291 90% conf 0.51 临界)。
  - `ood`(每帧到最近 milestone 距离)/ `ood_frac`:脱轨/OOD 粗信号。
  - **value 不动**(ep763 截 90% 仍 0.90,正确)—— 残差只作独立 flag,**绝不压 value**。
- **AWBC 用法**:progress value 给"走多远"的 advantage;failure flag 对未完成/脱轨段给负 advantage 或门控。对应 METHOD App②失败定位 / App④ OOD。
- **便宜首验(下一步)**:真机失败 rollout 上算 status,看 `is_complete=False` / `ood` 高的段是否对齐人工标注失败段;吻合再接进 advantage。

### 🔁 验证方法库(可复用,沉淀)
两种"无需真失败数据"造失败/回退 rollout 的方法,未来验 value/failure 处理直接复用:
1. **截断法(`crave_truncate_test.py`,推荐)**:成功 ep 截到 X%(末帧=中间态=未完成)。**全程 in-distribution**,干净。验 ① value 应停在真实进度(贴合完整曲线、不被拉到 1.0);② 完成/失败 flag 应判"未完成"。**最适合验 failure flag 校准**。
2. **倒放法(`crave_regression_test.py`)**:成功 ep 前进到中段再倒放(造真实"操作回退")。验 value 能否"先升后降"跟踪回退。**⚠️ 关键坑:proprio 含 Δstate,必须按轨迹顺序重算 mkp(Δ 才正确反向),否则倒放段特征错乱、结论失真**(本线踩过)。适合验回退灵敏度,但倒放态含 OOD 成分。

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

---

# 附录: 簇中心代表图 · 标准配置

> 2026-06-18 定稿。经编码器扫 + 解码器倒 U 阶梯 + 数据/频率消融后,**确定簇中心可视化(解码质心)的标准 enc/dec 配置**。

## 标准配置(当前最优)

| 组件 | 选择 | 参数 | 理由 |
|---|---|---|---|
| **编码器** | **DINOv2-large** | 1024d / ~300M | 簇中心语义结构最丰富(112 vs base 89 / small 92) |
| **解码器** | **small**(空间卷积) | **0.92M** | 解码器倒 U 的**峰值**(tiny 52 / **small 112** / medium 107 / big 79 / xl 88) |
| **解码器输入** | DINOv2-large **patch grid**(16×16×1024,非池化) | — | 保空间才不糊;池化向量解码必鬼影(已证) |
| **解码器损失** | L1 + 0.5·MSE(纯 L1 即可) | — | 对抗(GAN)会掉色/幻觉高频,可读性反降 |
| **训练数据** | ~9k 帧(挖矿 ~200 ep)| — | 9k > 24k(small 解码器吃 24k 会过度平滑) |
| **milestone 选择** | **自适应**:KMeans-96 → 覆盖率 → Otsu 自动阈值 | 数据自适应 | 不固定 K;V2.4 思路 |
| **value bins** | **自适应**:bins = milestone 进度位置 `Pord` + 端点{0,1} | NB = #milestone + 2(无超参) | value↔milestone 构造上精确一一对应 |
| **value 读出频率** | 30Hz,DP 窗 `lam=80, medw=45` | — | 频率窗按 fps 标定(挖矿仍 3Hz 即够) |
| **代表图本身** | **簇中心 = 簇内 large-grid 平均 → small 解码器解码** | — | 平滑、去具体化的"可读质心原型" |

**一句话**:`DINOv2-large 编码器 + small(0.92M)空间解码器 + patch-grid 输入 + L1 + 9k 数据 + 自适应 milestone`。

## 适用范围与诚实边界

- **用途**:整洁、去具体化的"可读质心原型"——做 milestone 词表/示意、不绑某条 demo 的布料颜色。
- **不替代最近真实帧**:要**最清晰**的代表图,仍用 **最近真实帧(medoid)**(锐度 476 vs 解码质心 ~112)。
- **不要追"合成平均质心的高锐度"**:可形变布料的"平均"数学上 ill-posed,加规模/换损失都救不动;small 解码器给的是**平滑可读原型**,这已是该路线的最优点。
