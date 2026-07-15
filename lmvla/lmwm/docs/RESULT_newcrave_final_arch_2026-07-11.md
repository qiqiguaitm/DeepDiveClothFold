# 用最终架构 + 新 CRAVE 方法在新 DINOv3-base 数据上跑一版（2026-07-11）

> **目标**：用 **LMWM 最终架构**（`train_multitask.py --teacher proto`，见 [`history/FINAL_CROSSTASK_PREDICTOR.md`](history/FINAL_CROSSTASK_PREDICTOR.md)）+ **新 CRAVE 方法**（[`../../crave/docs/final_architecture.md`](../../crave/docs/final_architecture.md)，BGMM img⊕pos）在用户新备的 **DINOv3-base 768D 全帧率**特征上，出 per-task deploy/id_top3，对照旧基线。
> **结论**：kai0+coffee 2 任务，新方法全面优于旧基线——**deploy kai0 0.703→0.837(+0.134)、coffee 0.788→0.934(+0.146)**；id_top3 kai0 0.473→0.901、coffee→1.0。ckpt `newcrave_base_kaicoffee.pt`。（deploy 为最稳健信号；id_top3 部分受类数变化影响，见 §4.3。）

---

## 1. 方案（两条更新的合成）

| 环节 | 采用 | 出处 |
|---|---|---|
| **milestone 发现** | DINOv3-base 768→PCA128 ⊕ proprio位置14（各 L2，1:1）→ BayesianGMM(自适应K) → per-mode coverage≥0.5 + median | 新 CRAVE `final_architecture.md` |
| **LMWM 预测器** | π0.5 SigLIP gist → proto teacher(码=下一 milestone SigLIP 中心) → MDN K=4 → AdaLN 生成器 | 最终架构 `history/FINAL_CROSSTASK_PREDICTOR.md` |
| **裁决指标** | per-task **deploy**(SigLIP grid-cos) + **id_top3** | 同上 |

milestone 在 **img⊕pos 联合空间发现**（proprio 改善分割），但**表示为 DINOv3-base 768D 质心** + 预测器输入走 **SigLIP gist**（保持与 π0.5 同塔、部署零第二编码器）。

## 2. 数据（用户备，本地+gf3）

`lmvla/crave/data/{kai,coffee,vis,xvla_full}_dinov3base`：DINOv3-base 768D 全帧率特征（index.npz{E,FR,T,n}+shard）。

| 数据集 | 帧数 | eps | proprio 源 | 溯源 |
|---|---|---|---|---|
| kai | 3,362,369 | 3055 | kai0_base parquet observation.state | ✅ 干净 |
| coffee | 55,000 | 50 | aloha_static_coffee parquet | ✅ 干净 |
| vis | 219,350 | 289 | (kai0/data，待定位 289ep 源) | ⚠️ 待确认 |
| xvla | 2,833,007 | 1532 | 多个 xvla_soft_fold hdf5 之和 | ⚠️ 多源待接线 |

**本轮测试范围 = kai0 + coffee**（溯源干净、单 root）。

## 3. 管线（可复现）

```bash
# 1) 新方法 milestone_file（DINOv3-base，PCA128⊕pos BGMM median；大数据集 --max_frames 下采样）
python lmwm/scripts/gen_newcrave_spec.py --dataset kai0   --max_frames 500000 --out temp/newcrave_specs_base/kai0_milestones_newmethod.npz
python lmwm/scripts/gen_newcrave_spec.py --dataset coffee --max_frames 0      --out temp/newcrave_specs_base/coffee_milestones_newmethod.npz
# 2) recurrence_graph（proto=DINOv3-base 768D 质心 + pord）
python lmwm/scripts/build_recurrence_graph.py --config lmwm/configs/datasets/newcrave_kai0_base_recurrence_graph.yaml
python lmwm/scripts/build_recurrence_graph.py --config lmwm/configs/datasets/newcrave_coffee_base_recurrence_graph.yaml
# 3) 最终架构训练（从 cwd=lmvla 跑，读帧 root 是 cwd 相对）
cd lmvla && python lmwm/scripts/train_multitask.py --datasets kai0,coffee --teacher proto \
    --tag newcrave_base_kaicoffee --per_task_cap 8000 --val_cap 4000 --steps 9000 --save_ckpt
```

milestone 数（新方法，DINOv3-base）：**kai0 16 / coffee 21**（旧 37 / 15）。全 0 倒挂、coverage≥0.5。

## 4. 结果

### 4.1 Smoke（cap2000/steps1500，快速验证管线）

| task(新ms) | deploy | persist | id_top1 | id_top3 | id_top5 |
|---|---|---|---|---|---|
| kai0(16) | 0.824 | 0.742 | 0.536 | **0.836** | 0.916 |
| coffee(21) | 0.932 | 0.901 | 0.927 | **0.995** | 1.000 |
| mean | 0.878 | — | — | 0.915 | — |

### 4.2 正式（cap8000/steps9000）— **交付**

ckpt：`lmwm/checkpoints/newcrave_base_kaicoffee.pt`

| task(新ms) | deploy | persist | lift | id_top1 | id_top3 | id_top5 | vfwd |
|---|---|---|---|---|---|---|---|
| kai0(16) | **0.837** | 0.745 | +0.092 | **0.686** | **0.901** | 0.964 | 0.701 |
| coffee(21) | **0.934** | 0.903 | +0.031 | 0.878 | **1.000** | 1.000 | 0.889 |
| **mean** | **0.886** | — | — | — | **0.951** | — | — |

### 4.3 vs 旧基线（`history/FINAL_CROSSTASK_PREDICTOR.md`，DINOv3-H + 旧 milestone·proto teacher·12k step）

| task | 旧 deploy | 旧 id_top3 | **新 deploy** | **新 id_top3** | Δdeploy | Δid_top3 |
|---|---|---|---|---|---|---|
| kai0 | 0.703(37ms) | 0.473 | **0.837**(16ms) | **0.901** | **+0.134** | **+0.428** |
| coffee | 0.788(15ms) | 0.985 | **0.934**(21ms) | **1.000** | **+0.146** | +0.015 |

**读法（诚实）**：
- **deploy（SigLIP grid-cos，受 milestone 类数影响小）→ 最稳健信号**：kai0 **+0.134**、coffee **+0.146**，且 kai0 lift over persistence +0.092（预测器确实学到东西）。方向明确：新方法优。
- **id_top3 的 kai0 大幅提升（+0.43）部分来自 milestone 数变少**（37→16 类，top3 命中天然更易），非纯方法增益；但 id_top1 0.686 亦为强身份，且 coffee(21>15ms 反而更多类)仍 id_top3=1.0，说明身份质量真实提升。
- 仍非单因素对照（编码器 base vs H、全帧率 vs 3Hz、milestone 方法、类数同时变）。要严格隔离，见 §6"同口径对照"。

## 4.4 统一 DINOv3-base 版 LMWM（预测器也在 DINO 空间）

把预测器/生成器/teacher **全部搬到 DINOv3-base 空间**（不用 SigLIP）：`train_multitask.py --encoder dinov3base`（crave DINOv3-base `encode_grid` 768 替 SigLIP 1152，接口相同）。好处：无需 SigLIP 权重、预测器输入用已缓存 DINO 特征、**神经预测可被 DINO 解码器直接可视化（闭环）**。

**正式（cap8000/steps9000，teacher_code=shared_pca 定档）**：ckpt `lmwm/checkpoints/dinov3base_lmwm_sharedpca_kaicoffee.pt`（fwd+predm+gmu/gsd）

| task | deploy | id_top1 | id_top3 | vs SigLIP 版 |
|---|---|---|---|---|
| kai0 | **0.865** | 0.678 | 0.886 | 0.837 / 0.901 |
| coffee | **0.955** | 0.926 | 0.995 | 0.934 / 1.00 |
| **mean** | **0.910** | — | **0.940** | 0.886 / 0.951 |

→ **统一 DINO 空间的 LMWM 稳定，deploy 0.910 略超 SigLIP 版 0.886**，id 持平。deploy>persistence 预测器确实学到。teacher 码用 **shared PCA128**（全任务联合 PCA·去噪+共享，见 §4.6 消融；rand 版 0.910/0.939 打平）。

**闭环可视化神经预测**（`make_neural_pred_decode_video.py`，srpo）：当前帧→DINO grid→gist→`predm.deploy_mean`→码 ẑ→`fwd`生成器→预测下一 grid→grid 解码器→图。视频 `docs/assets/neural_pred_decode_kai.mp4`，三联 `[real(t) | neural pred m+1 | m+1 代表帧(encode→decode)]`：
- **neural pred m+1** = 生成器 `fwd(cur, ẑ)` 解码——模型对下一里程碑的预测（前瞻，非照抄当前帧）；
- **m+1 代表帧** = **下一 segment 的 medoid**，且**分段/medoid 与 LMWM 训练 `build_pairs_abl` 完全一致**：Viterbi-monotone 分段 → 每段取**本 episode 该段内与簇中心最相似的帧** → 目标 = **下一段的 medoid**。即这一 panel **就是训练时生成器被监督去还原的那一帧**，与神经预测直接可比。
- ⚠️ **medoid 必须 per-episode 选**（本 episode 自己该段内），不能全局：kai0 各 episode 布**颜色不同**，milestone 簇抓 stage/姿态非颜色，全局 medoid 会跨到别色布 → 颜色不一致。同 episode=同布=颜色一致。这也正是训练 `build_pairs_abl` 的做法（`seg_med = order[s + argmax(Fq[s:e]·protoL[m])]`，`tgt = seg_med[i+1]`）。
- 空间对齐：grid 解码器训在 per-patch-L2 grid；LMWM 在标准化 (Gn−gmu)/gsd；fwd 输出 ×gsd+gmu → 重新 per-patch-L2 → 解码。

## 4.6 消融（负结果）：CRAVE PCA128 簇中心作 teacher 码

**假设**：proto teacher 码目前是 milestone 中心的**随机投影**（`spL@Wproj`, 768→128）。改用 CRAVE 的 **PCA128 簇中心**（`(spL−pca_mean)@pca_components.T`，milestone 判别性最强空间）作码，是否更好？（`train_multitask.py --teacher_code pca`）

**三方对比**(`--teacher_code rand|pca|shared_pca`,cap8000/steps9000):

| teacher 码 | mean deploy | mean id3 | kai0 id1 | coffee id1 | 特性 |
|---|---|---|---|---|---|
| **rand**（采用） | **0.910** | 0.939 | 0.688 | 0.910 | 随机投影·共享·含噪·免拟合 |
| per-task PCA | 0.908 | 0.937 | **0.712** | 0.836↓ | CRAVE PCA·去噪·**碎片化**(per-task 基不共享) |
| shared_pca | **0.910** | **0.940** | 0.678 | **0.926** | 全任务联合 PCA·去噪+共享(理论最优) |

→ **三者基本打平**(deploy rand=shared_pca=0.910;差异均在噪声内)。观察:① **per-task PCA 在 kai0 id1 +0.024 但 coffee id1 −0.074**(per-task 基不共享→跨任务码碎片化);② **shared_pca 修好了 coffee**(0.836→0.926,验证碎片化诊断)但丢了 kai0 gain,net 平;③ shared PCA explained_var=**0.891**(DINOv3-base gist 本身低秩,没多少噪声可去)→ 所以 ≈ rand。

**根因/洞察**:teacher 码只是 milestone 的**唯一标识符**,rand(JL 保距)/PCA(保方差)/shared_PCA(联合方差)都保留了足够可区分几何 → predictor/generator 学得一样好;**码空间的具体结构对 in-dist 指标不敏感**。注意 **CRAVE 的"PCA128 提升聚类"是关于聚类归属质量(哪些帧归一类),不是 teacher 码几何**,故该增益不转移到码。

**结论(定档):采用 `shared_pca` 为默认最终方案。** 理由:① 指标 ties/marginally-best(mean id3 0.940、生成 decode-L1 最低 0.228 vs rand 0.237),且**理论最正确**(去噪+共享,修好 per-task 的 coffee 掉分);② **同时兼容单任务(退化为该任务 PCA)与多任务(联合共享 PCA),开放词表/跨任务泛化友好**;③ **计算负担仅训练期一次 sklearn PCA fit(秒级),部署零增加**。`--teacher_code` 默认已设 `shared_pca`(`rand`/`pca` 留作消融)。工程注:PCA 码 raw norm 小(shared 0.486,簇中心接近全局均值),训练期须 scale 到 ~1 与 rand 同尺度。

**生成质量三方对比**(ep100,grid 解码器解码神经预测):deploy grid-cos 三者打平(0.910/0.908/0.910);目视四联(real/rand/pca/shared)**无可见质量差异**(锐度由 grid 解码器定,与码无关)。→ teacher 码对 predictor 指标与 generator 生成质量**均无实质区别**,故按"理论最正确+兼容性"取 shared_pca。

> ⚠️ 另:reach 指标已用统一 DINOv3-base 版实测更新 = **0.811s**(model lag, 120ep, undershoot 0.26),低于旧 SigLIP 版的 1.67s、也低于 LaWM 1.48s(偏近未来)。`measure_dinov3base_lag.py` / `lmwm/outputs/dinov3base_lag.json`。web 报告 §3.2 已同步。

## 4.7 同环境 vs **官方 LaWM** —— 9 arena 头对头（kai0 + 3 LIBERO + coffee + 4 aloha）

**公平前提**:官方 LaWM LAM 视觉塔 = `facebook/dinov3-vitb16-pretrain-lvd1689m`(768D)= **我们 LMWM 的 `dinov3-base` 逐权重同一编码器**(旧"异空间"免责声明因搬到 base 而消失)。每个 arena 都在**同 768D 空间 · 同 val · 同指标公式**下比:官方 LaWM 真权重(LAM+外挂 deploy-predm,固定 1.6s 未来目标)vs LMWM(proto teacher + milestone+1)。LIBERO 是多任务→挖矿用 `min_cov=0.06`(得 spatial 29 / goal 40 / long 50 milestone)。双机并行(kai0 arena 在 gf3·8×H20;LIBERO suite 在本机·数据在此)。

| arena(同 DINOv3-base) | LaWM oracle(TF) | LaWM deploy | **LaWM lift** | **LMWM deploy** | **LMWM lift** | LMWM id3 / vfwd |
|---|---|---|---|---|---|---|
| **kai0**(我们任务·LaWM **OOD**) | 0.781↓ | 0.672 | 0.025 | 0.865 | **0.084** | 0.886 / 0.744 |
| **LIBERO-10**(分布内) | 0.950 | 0.832 | 0.004 | 0.961 | **0.049** | 0.969 / 0.882 |
| **LIBERO-spatial** | 0.956 | 0.833 | 0.003 | 0.971 | **0.029** | 0.963 / 0.918 |
| **LIBERO-goal** | 0.956 | 0.859 | 0.002 | 0.978 | **0.025** | 0.959 / 0.915 |
| **coffee**(aloha 双臂) | 0.870 | 0.822 | 0.002 | 0.955 | **0.021** | 0.995 / 0.926 |
| **aloha_candy** | 0.887 | 0.846 | 0.003 | 0.965 | **0.019** | 0.989 / 0.947 |
| **aloha_cups** | 0.867 | 0.824 | 0.005 | 0.965 | **0.032** | 0.988 / 0.938 |
| **aloha_ziploc** | 0.870 | 0.879 | **0.059** | 0.964 | 0.025 | 1.00 / 0.972 |
| **aloha_screw** | 0.860 | 0.800 | 0.006 | 0.966 | **0.027** | 1.00 / 0.896 |

（lift = deploy − persistence;LaWM id3/vfwd 全 N/A;参数 LaWM ~230M vs LMWM ~34M,轻 10×。）

**结论(9 arena)**:
- **① 部署 lift:LMWM > LaWM,9 中 8 个**(仅 aloha_ziploc 例外)。LaWM 部署 lift 普遍 **≈0**(0.002–0.006):固定视野未来从单帧当前**不可预测**(歧义→predm 退化照抄当前,deploy≈persistence);LMWM 的价值前向 milestone **可预测**。
- **② 诚实反例 aloha_ziploc**:LaWM lift **0.059 > LMWM 0.025**。滑动任务固定视野未来恰可预测,且 LaWM 固定视野目标持恒基线更低(0.820 vs LMWM 0.939)→lift 空间大。**lift 因两模型目标不同并非完美可比**,ziploc 正是这个口径差异的体现——这也是为何主指标应看"每个模型对自己目标的 lift + 语义能力",而非跨模型 lift 绝对值。
- **③ OOD 崩塌**:kai0 LaWM oracle 从 ~0.95 塌到 0.781(未训过);**coffee/aloha 双臂任务 LaWM oracle 也仅 ~0.87**(LIBERO/RoboTwin 训练的 LAM 对 aloha 双臂偏 OOD)。而 LMWM 在每个任务重挖 milestone 重训,deploy 稳定 0.95+。
- **④ 语义**:LMWM 每个 arena 都有 id3(0.89–1.0)+ value-forward(0.74–0.97);LaWM 结构上无(N/A)。
- ⚠️ **对 LaWM 公平声明**:LaWM 的 LAM 设计上把部署码甩给下游 VLA、原生无 predm,外挂 standalone predm 非其强项→deploy 对 LaWM 略不利,应与 oracle 并读。但 9 arena 里 8 个一致表明 **LMWM 自包含部署预测实用性稳定更高**。
- **执行**:双机并行(kai0 arena 在 gf3·8×H20;LIBERO+coffee+aloha 在本机)。**踩坑**:gf3 共享队列杀多 job / **线程超订 thrash**(须 `OMP_NUM_THREADS` 限制,单/双 job 才稳)/ v3.0 aloha 用 `meta/episodes/` parquet(非 jsonl,已加 fallback)/ 任务名 vs bank 名映射(cups→cups_open 等)。产物 `lmwm/outputs/arena_comparison_9.json` + 各 `{arena}_lmwm.json` / `eval_lawm_{arena}.json`。脚本 `eval_lawm_libero.py`(--frames_kind kai0/libero/lerobotv3)、`train_multitask.py`/`gen_newcrave_spec.py`(REPO 读 `CRAVE_REPO`)。

**目标时滞定量(为什么 persistence 差异大、为什么裁决只能用 lift)**:各模型"目标帧"离当前帧的平均时滞(秒),用 `build_pairs_abl` 的 (cur, milestone+1 medoid) 对实测:

| arena | LMWM 目标时滞 均值/中位(s) | vs LaWM 固定 1.6s | LMWM/LaWM persist |
|---|---|---|---|
| **kai0** | **3.11 / 2.43** | **1.9× 更远** | 0.781 / 0.647 |
| LIBERO-10 | 1.18 / 0.95 | 0.7× 更近 | 0.912 / 0.828 |
| LIBERO-spatial | 0.49 / 0.35 | 0.3× 更近 | 0.942 / 0.830 |
| LIBERO-goal | 0.45 / 0.30 | 0.3× 更近 | 0.953 / 0.857 |
| coffee | 0.52 / 0.36 | 0.3× 更近 | 0.934 / 0.820 |
| aloha_candy/cups/ziploc/screw | 0.40–0.63 / 0.28–0.32 | 0.3–0.4× 更近 | 0.93–0.95 / 0.79–0.84 |

**洞察**:① LMWM 目标时滞**不固定**(0.4–3.1s,取决于阶段长短),与 LaWM 的固定 1.6s **根本不是同一视野**;② persistence 更高来自**两个因素叠加**——时间距离(多数 arena 更近)+ **medoid 代表帧效应**;③ **kai0 是关键隔离**:目标 **3.11s 比 LaWM 1.6s 远 1.9×,persistence 却更高**(0.781 vs 0.647)——**时间更远、视觉更近**,只能是 medoid(canonical 阶段姿态)比 LaWM 的任意 1.6s 中途帧更像当前。→ 目标难度(persistence)受"离多远 + 是不是代表帧"双重影响、**跨模型完全不可比**,故裁决必须用 **lift = deploy − persistence**(去掉持恒基线)而非 deploy 绝对值。数据 `lmwm/outputs/target_lag_per_arena.json`。

> **deploy vs lift(量纲相同、含义不同)**:两者都是无量纲 grid-cos。**deploy** = cos(预测,目标),绝对保真,但**含"目标难度"污染**(目标近→照抄当前就 0.9+,能被刷高);**lift** = deploy − persistence,**去掉目标难度**后的净前向增益(照抄当前 → lift=0,刷不了)。例:aloha_screw deploy 差 0.166 看似碾压,但真实 lift 差仅 0.021(deploy 夸大 8×);aloha_ziploc deploy 说 LMWM 赢、lift 说 LaWM 赢。**单模型看 deploy(实际保真),跨模型/判断真本事看 lift。**

## 4.8 客观对比总评(LMWM vs LaWM)

**本质区别**:LaWM 学"固定 ~1.6s 后长什么样"(独立 DINOv3 ViT-B·inverse-dynamics VAE code32·12 层 AdaLN-DiT·~230M·部署码甩给下游 VLA·**有真实 SR** LIBERO 98.6%);LMWM 学"下一个价值 milestone 长什么样"(统一 DINOv3-base·proto teacher 0 参数·4 块 AdaLN CNN·~34M·内置 MDN predm 自包含·**尚未接策略**)。

**LMWM 优势(客观)**:
1. **部署预测更实用**:9 arena 里 **8 个 deploy-lift > LaWM**。根因——固定视野未来从单帧当前**不可预测**(歧义→LaWM predm 退化照抄当前,lift≈0),而价值前向 milestone **可从当前预测**。
2. **提供 LaWM 结构上没有的语义**:milestone 身份(id3 0.89–1.0)+ 价值单调(vfwd 0.74–0.97,vs 监督 GT corr 0.943)。
3. **参数轻 ~10×** · **预测可直接解码回帧(闭环)** · **自包含部署**(不依赖策略产码) · **编码器一致性潜力**(未来迁 SigLIP 与 π0.5 同塔复用 KV)。

**LMWM 缺点 / LaWM 更强(客观)**:
1. **决定性短板:无下游 SR 证据**。LaWM 是完整系统、有真机/仿真 SR;LMWM **还没接进任何策略**,"价值引导对策略的真实增益尚未验证"——目前这点 LaWM 明确领先。
2. **oracle 重建 LaWM 分布内很强**(0.95,给定未来帧时);LMWM 用 proto teacher 无对应 teacher-forced oracle 口径。
3. **诚实反例 aloha_ziploc**:LaWM deploy-lift 0.059 > LMWM 0.025(滑动任务固定视野恰可预测)——固定视野并非在所有任务都差。
4. **成熟度**:LaWM 有公开权重/数据/多本体完整 pipeline;LMWM 是研究原型。

**公平性声明(必读)**:① **lift 跨模型非完美可比**(目标不同→持恒基线不同→lift 空间不同,ziploc 即此口径差异),稳健读法是"各模型对自己目标的 lift + 各自语义能力";② **deploy 对 LaWM 略不利**(LAM 原生无 predm、部署码设计上甩给 VLA),应与 oracle 并读;③ **OOD 因素**:coffee/aloha 双臂上 LaWM oracle 仅 ~0.87(LAM 偏 OOD),而 LMWM 每任务重挖重训——LMWM 的强项部分来自"每任务专训",LaWM 是通用 LAM。

**一句话**:世界模型自身预测质量上,**LMWM 用 1/10 参数拿到更实用的部署预测(8/9 lift 更高)+ 价值/身份语义 + 闭环可解码**;但 **LaWM 作为已落地完整 VLA 有真实下游 SR,LMWM 对策略的真实增益尚未验证**——这正是 §04(迁 SigLIP 接 π0.5)要补的唯一、也是最关键短板。**方向明确但不宜过度:LMWM 赢在 intrinsic 预测与效率,LaWM 赢在 extrinsic 已验证,二者尚未在同一"下游 SR"标尺上正面比过。**

## 4.9 不同编码空间对比:SigLIP(π0.5)vs DINOv3-base

**动机**:LMWM 最终版用 DINOv3-base(统一单空间);但 SigLIP 是 π0.5 的塔(未来 VLA 融合要用)。把**整套 LMWM 流程**也在 SigLIP 空间跑一版(`train_multitask.py --encoder siglip`,权重 `pt_224.npz` 从 pi05_base 提取的 PaliGemma img 塔),milestone 挖矿/分段仍在 DINOv3-base(复用同 graph)、grids 用 SigLIP 编码,同 `teacher_code=shared_pca`,kai0+coffee 联合训练,与 DINOv3-base 版直接对比。

| task | 指标 | **SigLIP(1152)** | **DINOv3-base(768)** |
|---|---|---|---|
| kai0 | deploy / persist / **lift** | 0.837 / 0.745 / **0.092** | 0.865 / 0.781 / **0.084** |
| kai0 | id3 / vfwd | 0.894 / 0.688 | 0.886 / 0.744 |
| coffee | deploy / persist / **lift** | 0.934 / 0.903 / **0.031** | 0.955 / 0.934 / **0.021** |
| coffee | id3 / vfwd | 0.995 / 0.937 | 0.995 / 0.926 |
| **mean** | deploy / id3 | **0.885 / 0.944** | **0.910 / 0.940** |

**结论(客观)**:
- **两空间 LMWM 预测质量相当,不存在"哪个明显更好"**。
- ⚠️ **deploy/persist 绝对值跨空间不可直接比**:SigLIP 1152D raw vs DINOv3-base 768D **每 patch L2 归一**,grid-cos 的几何尺度不同。DINO deploy 略高(0.910 vs 0.885)有一部分是"空间/归一化差异",非纯预测优劣。
- **lift(去持恒基线,较可比)SigLIP 反而略高**(kai0 0.092>0.084、coffee 0.031>0.021):SigLIP persistence 更低(目标在 SigLIP 空间离当前"更远"),但预测器净增益反而更大。**id3 打平**(0.944 vs 0.940),**vfwd 混合**(kai0 DINO 高 / coffee SigLIP 高)。
- **为什么最终版仍选 DINOv3-base**——不是因为它预测更好,而是**用途分工**:① 关键差异在**解码可视化锐度**——**两空间都有 grid 解码器可闭环**(SigLIP 解码器 `train_siglip_decoder.py`/`siglip_decoder/dec.pt` 早已训好,`render_twomodel_video.py` 可出四栏闭环视频 `siglip_neural_pred_kai.mp4`),但 **DINOv3-base 解码锐利**(重建式/VLA-JEPA 特征,布形褶皱可辨),**SigLIP 解码明显更糊**(对比学习特征、重建力弱,val_L1 数值近 0.053 但视觉软+栅格伪影);② 不需 SigLIP 权重、用已缓存 DINO 特征。而 **SigLIP 版的价值在未来 VLA 融合**:它就是 π0.5 的塔,可同塔复用 KV(§04),子目标即策略消费空间。⚠️ 更正:早前"SigLIP 无解码器做不到闭环"表述不准——**有解码器、能闭环,只是更糊**;中间两栏共用同一 SigLIP 解码器隔离出"预测对、糊在解码"。
- 产物:ckpt `lmwm/checkpoints/siglip_lmwm_sharedpca_kaicoffee.pt`;对比 `lmwm/outputs/encoder_space_comparison.json`。SigLIP 编码器 `_siglip_bigvision.py`(纯 torch,读 pt_224.npz)。⚠️ 踩坑:SigLIP 编码 heavy(kai0 23715 帧 1152D ~15min);本机 GPU 被他人进程挤时会瞬时 OOM 静默 SIGKILL,须**干净 GPU**跑。

## 4.10 最后一个 stage:Self-loop 收敛(设计完善)

**问题**:`build_pairs_abl` 只为段 0→1,...,N-2→N-1 产生训练对,最后一段被排除——"最后一段+1"不存在,模型从未被训练如何处理终点。

**方案(user 提出)**:最后一段做 **self-loop**——目标=自己 medoid,teacher 码=自己簇中心。预测器学"停在这",生成器学"保持当前不变"。

**前提验证**:全 9 arena terminal milestone 都是最高 pord(0.84-0.98),每 episode 最后一段 pord 中位值 0.84-0.96(足够高)。→ terminal self-loop 是干净的收敛行为,不是噪音。

**改动**:`build_pairs_abl` 在 `for i in range(len(seg_m)-1)` 后追加一行 self-loop:(cur, seg_med[end], cur_ms, cur_ms)。

**对比(kai0+coffee DINOv3-base 全量)**:

| task | ver | deploy | lift | id3 | vfwd |
|---|---|---|---|---|---|
| kai0 | 旧(无) | 0.865 | 0.084 | 0.886 | 0.744 |
| kai0 | **新(selfL)** | 0.867 | **0.092** | **0.893** | 0.717 |
| coffee | 旧(无) | 0.955 | 0.021 | 0.995 | 0.926 |
| coffee | **新(selfL)** | 0.955 | 0.021 | 0.985 | 0.891 |

**结论**:① **deploy/lift/id3 持平或微升**(kai0 lift +0.007/id3 +0.007,coffee 持平),self-loop 不损害预测;② **vfwd 下降(−0.03)是预期且正确的**——self-loop 帧 `progn[cur]==progn[cur]` 不满足 `>`,天然"不前向",**不是倒退,是之前 terminal 帧不在 val 里、vfwd 分母漏了它们**;③ self-loop 的收益是完整性——每帧有训练信号,terminal 有收敛行为。数据 `lmwm/outputs/selfloop_comparison.json`;ckpt `dinov3base_selfloop_kaicoffee.pt`。

## 5. 过程中修复（避免重踩）

| # | 问题 | 修复 |
|---|---|---|
| 1 | `load_index`/`build_recurrence`/`gen` 写死 feat dim **1280**，新数据 768D 崩 | 从首个 shard 推断 dim（`train_lawm_patch.py`/`build_recurrence_graph.py`/`gen_newcrave_spec.py`）|
| 2 | `index["T"]` 是**原始秒数**非 [0,1]，破坏 mode_split/median | 按 episode 归一化 T→[0,1] |
| 3 | **SigLIP `pt_224.npz` 本机缺失**（最终架构预测器输入必需）| 从 `pi05_base` orbax params 的 `PaliGemma/img` 塔提取（23 arrays，big_vision 命名对齐）→ `openpi_cache/paligemma_weights/pt_224.npz`，验证 encode_grid OK |
| 4 | `train_multitask` `REPO=parents[2]=lmvla`（移动后），temp/kai0/xvla 解析错 | `lmvla/{temp,kai0,xvla}` 符号链接（gitignore）；且**从 cwd=lmvla 跑**（read_frames root 是 cwd 相对）|
| 5 | 大数据集(kai 336万)BGMM 不可行 | `gen_newcrave_spec --max_frames` 按 episode 分层下采样（保每 ep 有帧）|
| 6 | DINOv3-base 版 **deploy=NaN + id 崩**（0.27）| DINOv3 grid 是 float16 大量级 → `G.std()` fp16 溢出 NaN + 空间未归一。修：grid **per-patch L2 归一**（CRAVE 惯例）+ gmu/gsd **fp32 累加**。修后 id 0.27→0.89 |

## 6. 待办 / 下一步

- 正式版 kai0+coffee 数字回填 §4.2。
- **xvla(1532, 多 hdf5 源)** + **vis(289, kai0/data 源待定位)** 接线 → 扩到 3/4 任务（对齐旧 3-task 基线 0.753/0.710）。
- 严格同口径对照：可在**旧 DINOv3-H** 上用新方法重训一版，隔离"编码器 vs milestone 方法"。
- 唯一未测的终判 = 下游 SR（接 π0.5 测 action-MAE），见 `history/FINAL_CROSSTASK_PREDICTOR.md` §4。
