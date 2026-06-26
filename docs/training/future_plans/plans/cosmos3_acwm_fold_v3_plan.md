# Cosmos3-Nano AC-WM (forward-dynamics) on wam_fold_v3 — 训练与评测方案

> 立项 2026-06-25。骨干 = **Cosmos3-Nano 16B**(`cosmos-framework`,FD/forward_dynamics 模式);
> 数据 = **wam_fold_v3 visrobot+kairobot 混训**;算力 = **5 节点 × 8×A100-80G**(Baidu AIHC)。
> 这是 [[cosmos3-wam-fold-world-model-plan]](v1 版)的 v3 续作 —— 通路已就绪、仅冒烟级验证(v1 上**从未完成一次完整训练**,见 §1.3),核心工作是**把数据通路从 v1 切到 v3 混库 + 完成首次完整训练 + 补齐评测**。
> 与 `giga_world_policy/.../acwm_clothfold_vis6/vis8`(Wan2.2-5B 版 AC-WM)是**两套独立实现**,本方案走 Cosmos3 骨干。
>
> **术语**:本方案的目标是 **AC-WM(action-conditioned world model)**。cosmos-framework 里它对应 `mode="forward_dynamics"`(FD 是该 mode 的内部名);下文 AC-WM 与 FD 指同一件事 —— 动作作 clean 条件、扩散 loss 只在视频。

---

## 0. TL;DR(选型与结论)

| 维度 | 决策 |
|---|---|
| 骨干 | `nvidia/Cosmos3-Nano`(16B MoT,中训版,**非** Policy-DROID),DCP 已转好 |
| 范式 | **forward_dynamics**:动作 14-D 作 clean 条件,扩散 loss 只在视频 token → P(未来视频 \| 首帧, 给定动作, 文本) |
| 数据 | `wam_fold_v3`:visrobot01_v3_train(2353ep/3.16M帧)×3 + kairobot01_v3(6512ep/5.78M帧)×1,domain 16/17,per-rig 分位归一化 |
| 算力 | 5n8g(40×A100),FSDP-8 节点内 + DDP replicate=5 跨节点 |
| 步数 | M1 10k 步(≈24–30h)出动作可控性证据 → M2 延到 20–30k 步收敛 |
| 评测 | ①视频保真 PSNR/SSIM/LPIPS vs GT;②动作可控性 ΔPSNR(GT-action − 扰动/零动作)>1dB + IDM 探针;③长程 AR(10–30s)FVD;④下游策略对齐 Pearson r≥0.8 |
| 基建现状 | 5n8g AIHC job + run 脚本 + recipe + 数据类 **均已存在**;v1 上仅冒烟级验证(smoke 30步 + 5n8g 跑到 ~926 步即中断,仅存 iter_500),**尚未完成完整训练**。本方案=切 v3 混库 + **首次完整训练** + 评测落地 |

---

## 1. 资料梳理(现状盘点)

### 1.1 模型(`cosmos/models/modelscope/`)
| 模型 | 大小 | 用途 |
|---|---|---|
| **Cosmos3-Nano** | 35G | **本方案基座**(16B MoT 中训版),已 `convert_model_to_dcp` → `wam_fold_wm_runs/checkpoints/Cosmos3-Nano-dcp` |
| Cosmos3-Nano-Policy-DROID | 31G | 策略专精版,本方案不用 |
| Cosmos3-Super / Super-I2V | 126G/122G | 可选大模型对照(零样本 I2V 基线已测,Nano>Super) |
- VAE:`WAN_VAE_PATH = .../Wan2.2-TI2V-5B/Wan2.2_VAE.pth`(4×时序/8×空间;cosmos-framework 的 ActionDataPacker 内部编码视频→latent,缓存到 `WAM_WM_LATENT_CACHE`)。

### 1.2 数据(`kai0/data/wam_fold_v3/`,LeRobot v2.1,3 相机 480×640@30fps,14-D 动作)
| 切分 | episodes | frames | 相机键 | 视频 | 备注 |
|---|---|---|---|---|---|
| visrobot01_v3_train | 2,353 | 3.16M | **top_head / hand_left / hand_right** | symlink→vis_base/dagger_v3 | domain 16 |
| visrobot01_v3_val | 100 | 140K | 同上 | symlink | **评测集** |
| kairobot01_v3 | 6,512 | 5.78M | **cam_high / cam_left_wrist / cam_right_wrist** | 实体文件 83G | domain 17 |
- 任务文本统一:`"Flatten and fold the cloth."`;state/action 均 14-D float32(6 关节+1 夹爪/臂)。
- ⚠️ 两机型**相机键命名不同**——这是切 v3 的主要代码改点(见 §3.2)。
- 注:`vae_latent_uni/` 等预抽 latent 是 **giga/Wan2.2 通路**的(布局/schema 不同),**Cosmos3 FD 通路不复用**,其自建 `WAM_WM_LATENT_CACHE`。

### 1.3 基建(`cosmos/wam_fold_wm/` + `packages/cosmos3/`)
- recipe:`wam_fold_wm/train/recipe_wm_nano.toml`(45056 token、FSDP、selective AC、EMA off、lr 2e-4、grad_clip 0.1)。
- 实验 Python:`packages/cosmos3/.../posttrain_config/wam_fold_wm_nano.py`(数据源 `build_wm_data_source` = vis×3+kai;动作模块 5× lr;`keys_to_skip_loading` fresh-init action I/O)。
- 数据类:`packages/cosmos3/.../datasets/wam_fold_dataset.py`(`WamFoldLeRobotDataset`,支持 forward_dynamics / inverse_dynamics / policy / joint;**`_DATA_ROOT` 硬编码 v1**)。
- AIHC 5n8g:`wam_fold_wm/train/aihc/{aijob_cosmos_wamfold_wm_5n8g.json, run_train_aihc_cosmos_wm.sh, submit_cosmos_wm_aihc.sh}` —— **已就绪**,replicas=5、a100_80g×8、RDMA、MAX_STEPS/SAVE_ITER/REPLICATE_DEGREE 走环境变量。
- 评测:`wam_fold_wm/eval/{fd_infer.py(Δaction 可控性), baseline_from_i2v.py, export_*.sh, make_report.py}`。
- 已有产物(v1,磁盘实证):smoke 单机 iter 30/100/200/300 ✅;iter300 导出 29G ✅;iter300 评测 `eval_results.jsonl` GT PSNR=18.68(比 I2V 基线 +5.8dB)但 **ΔPSNR=−0.15/−0.21,verdict=WEAK**(动作尚未被遵循)。
- ⚠️ **5n8g `train_out_5n8g` 未完成**:2026-06-13 启动,最远到 **iteration 926**(loss ~0.10–0.15,单步 9.7s)即中断,只落了 **iter_000000500** 一个 ckpt。**v1 从未跑完一次完整训练** → v3 的 M1 是这条 AC-WM 通路的**首次完整训练**。

### 1.4 参照基线(已有,供对齐/对比)
- 零样本 I2V 地板:Cosmos3-Nano PSNR 12.88/12.61/12.22 dB @1/3/7s。
- 三方 v3_val 离线对比(MAE 口径,policy):kai0 π₀.₅ < FASTWAM-v6 < GWP_ABS_v5(见 [[wam-three-way-v3val-compare]])——这是**下游策略**指标,FD WM 的可控性需另立 PSNR/ΔPSNR 口径。

---

## 2. 目标与范式

训练一个**动作条件世界模型**(AC-WM):给定首帧(+可选历史帧)+ 一段 14-D 动作 chunk + 文本,生成未来视频。
- **forward_dynamics**:动作 = clean 条件(不加噪/不监督),扩散 loss 只回传视频 token(`wam_fold_wm_nano` 已是此配置)。
- chunk_length=32(≈1.07s@30fps);obs 窗 33 帧 = 4k+1,精确匹配 Wan VAE 时序 stride。
- 成功定义(分层门禁):
  1. **学会场景**:GT-action 生成 PSNR ≫ I2V 基线(已达:+5.8dB@300步)。
  2. **动作可控**:ΔPSNR(GT-action − 错误动作)>1.0 dB 且 ΔPSNR(GT − 零动作)>1.0 dB(M1 目标)。
  3. **长程稳定**:10–30s AR rollout 不崩(FVD 可控,逐相机退化曲线平缓)。
  4. **下游可用**:闭环 policy-in-the-loop,WM 评分 vs 真机评测 Pearson r≥0.8(终极门禁)。

### 2.1 跨机型(visrobot01 vs kairobot01)区分机制与混训策略

visrobot01 与 kairobot01 是**两个不同 setting**(不同相机、不同场景/标定)。代码核实:训练里**已在三个层面显式区分**,不是简单堆在一起。

| 层面 | 机制 | visrobot01 | kairobot01 |
|---|---|---|---|
| **动作 I/O 头** | `action2llm`/`llm2action` = **`DomainAwareLinear`**(`nn.Embedding[num_domains=50]` 存每域独立权重+偏置),按 `domain_id` 逐样本选权重(`cosmos3_vfm_network.py:188-189, 755`) | **domain 16**(`wam_fold`) | **domain 17**(`kairobot01`) |
| **归一化** | per-rig 分位 stats,各一份(`_RIG_DEFAULTS[*].stats_path`) | `visrobot01(_v3).json` | `kairobot01(_v3).json` |
| **相机键** | per-rig 取键(§3.2),拼同一 concat_view:顶=主图全分辨率,底=两腕部各半幅横拼(`wam_fold_dataset.py:366-374`) | top_head / hand_left / hand_right | cam_high / cam_left_wrist / cam_right_wrist |

**共享**:MoT 主干、Wan VAE latent 空间、`action_modality_embed`/`action_pos_embed`(非 domain-aware)。
→ 本质 = **跨机型联合训练**:共享世界模型主干 + 各机型独立动作接口头 + 各自归一化。

**⚠️ 视觉侧无显式 domain 条件**:`domain_id` 只路由**动作头**,不路由视觉分支。两机型视觉域靠**首帧内容 + 文本**隐式区分(对 AC-WM OK——首帧恒为条件,模型据此判定机位顺生)。代价:共享主干被两个视觉域分摊容量,可能稀释部署目标 visrobot 的视觉保真。

**混训策略(本方案锁定 A 为默认,M1 评测后再决定是否上 C):**

| 策略 | 做法 | 适用 / 代价 |
|---|---|---|
| **✅ A 联合混训(默认)** | `ConcatDataset([vis,vis,vis,kai])` → 有效配比 ≈ 52:48(visrobot 过采样 3×,2353×3 vs kai 6512),偏向部署目标 | 最大化叠衣动力学数据 + 跨机型迁移;visrobot 视觉保真可能被 kairobot 域稀释 |
| B visrobot 单训 | 只挂 visrobot01_v3_train | 数据少 2.5×、无迁移;仅当 kairobot 域差异被证实有害时退守 |
| C 两阶段 | A 联合预训 → 再 visrobot 单独微调收尾 | 兼顾动力学与视觉回域;多一阶段。**M1 若分机型评测显示 visrobot 保真被拖累,即启用** |

**M1 评测务必分机型出指标**(visrobot vs kairobot 的 PSNR/ΔPSNR 分开看),作为是否从 A 升级到 C 的判据。

---

## 3. 数据准备(v3 混库)—— 本方案的核心改动

### 3.1 数据根切到 v3
`build_wm_data_source` 显式传 v3 root(优于改全局 `_DATA_ROOT`,可与 v1 实验并存):
```python
vis = WamFoldLeRobotDataset(rig="visrobot01", split="train", mode=mode,
        root="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3/visrobot01_v3_train",
        chunk_length=chunk_length, fps=fps)
kai = WamFoldLeRobotDataset(rig="kairobot01", mode=mode,
        root="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3/kairobot01_v3",
        chunk_length=chunk_length, fps=fps)
return ConcatDataset([vis, vis, vis, kai])   # 维持 3:1 跨机型配比
```
- `_apply_split` 对 `visrobot01_v3_train` rsplit("_",1)→ 前缀 `visrobot01_v3`,split="val"→`visrobot01_v3_val` ✅(评测自动指向 v3 val)。
- kairobot01_v3 无 train/val 切分,split=None ✅。

### 3.2 ⚠️ 相机键 per-rig(**必须改 `wam_fold_dataset.py`**)
现状 `_LEROBOT_VIDEO_KEYS` 是**全局单套** = `cam_high/cam_left_wrist/cam_right_wrist`(对 kairobot 正确,对 **visrobot01_v3 错误**——v3 visrobot 用 `top_head/hand_left/hand_right`)。
**改法**:把 `_LEROBOT_VIDEO_KEYS` 改成按 rig 取:
```python
_VIDEO_KEYS_BY_RIG = {
    "visrobot01": {"high": "observation.images.top_head",
                   "left": "observation.images.hand_left",
                   "right": "observation.images.hand_right"},
    "kairobot01": {"high": "observation.images.cam_high",
                   "left": "observation.images.cam_left_wrist",
                   "right": "observation.images.cam_right_wrist"},
}
```
在 `__init__` 按 `rig` 选表;主图(primary/high)= top_head(俯视,叠衣最关键)——避免 [[fastwam-v3-latent-camera-bug]] 同类相机错位坑。改完跑 §6 的 1-step smoke 验证三相机张量形状。
> 注意:v1 的 visrobot01 也用 cam_high 命名,改成 per-rig 后**v1 实验需回归测一次**(或用 v3 专用 rig 名隔离)。

### 3.3 归一化统计(per-rig 分位)
- v1 用 `visrobot01.json`/`kairobot01.json`(delta-arm + abs-gripper,分位 q01/q99)。同机器人同任务,v3 分布近似,但 v3 含更多 dagger → **重算更稳**:
  `python wam_fold_wm/train/compute_ext_norm_stats.py --root .../visrobot01_v3_train --out .../assets/visrobot01_v3.json`(kairobot 同理)。
- `_RIG_DEFAULTS[*]["stats_path"]` 指向 v3 stats(或在 build 时传 `stats_path=`)。

### 3.4 latent 缓存(自建,非复用 uni)
- FD 通路首个 epoch 在线编码视频→latent 落 `WAM_WM_LATENT_CACHE`(key 带 `L32`)。无需提前抽。
- 可选预热:用一个 1n8g 短作业先跑数百步把热点 episode 的 latent 写入共享 PFS,降低正式训练 I/O 抖动。
- 容量估算:13.8M 窗口若全缓存约数 TB —— 默认按 LRU/按需缓存,不要求全量;监控 PFS 余量。

---

## 4. 训练方案

### 4.1 拓扑与超参(沿用已验证 recipe)
| 项 | 值 | 来源 |
|---|---|---|
| 并行 | FSDP shard=8(节点内)+ DDP replicate=5(跨节点)→ 40 GPU | run 脚本 `REPLICATE_DEGREE=NNODES` |
| token 预算 | `max_num_tokens_after_packing=45056`,`max_batch_size=8`/pack | recipe |
| 有效 batch | ~40 packs/step ≈ 与 giga 全局 batch 320 同量级 | — |
| 精度/AC | bf16 + selective AC(save fmha)+ compile(language region) | 16B@45k 必须,否则 OOM |
| 优化器 | AdamW(0.9,0.95),lr **2e-4**,action 模块 **5×** lr,wd 0 | 官方 action 配方;发散回退 5e-5 |
| 调度 | LambdaCosine,**cycle=max_iter**(必须等于总步,否则 cycle lookup 崩),warmup 30 | recipe CRASH FIX |
| grad clip | 0.1 force_finite;EMA off(16B CPU 拷贝 OOM) | recipe |
| 初始化 | 载 Cosmos3-Nano-dcp,`keys_to_skip_loading` fresh-init action2llm/llm2action/embed/pos(domain 16/17 + 14-D 新) | 实验 Python |

### 4.2 阶段计划
- **M0 冒烟(1n8g,~30 步)**:切 v3 后必跑。验证三相机加载、domain 16/17、FD token 流、loss 下降。命令见 §6。
- **M1 主训(5n8g,10k 步,≈24–30h)**:`MAX_STEPS=10000 SAVE_ITER=500 SCHED_CYCLE=10000`。每 500 步存 ckpt;watcher 导出最新 ckpt 跑 fd_infer 看 ΔPSNR 是否破 1.0。
- **M2 收敛(续训到 20–30k 步)**:M1 若可控性达标但视频仍糊,延长 cosine(注意 cycle 要重设=新总步数,见 §4.1 坑)。
- **M3(可选)长程/混外部数据**:接 `docs.../open_deformable_datasets.md` 外部叠衣数据扩域;或开 inverse_dynamics 联合训 IDM 探针。

### 4.3 监控
- loss(视频 flow-matching)、grad norm、iter speed(单步~10s 参考)、`mem_node*.log`(CPU-RAM OOM 看护已内置 drop_caches)。
- 每 ckpt 触发 §5.1 快评(ΔPSNR);wandb offline,落 `CKPT_DIR/wandb`。

---

## 5. 评测方案

> FD 世界模型 ≠ 策略,**主指标是视频可控性**,不是 policy 的 action-MAE。下游对齐才用 MAE/SR。

### 5.1 动作可控性(M1 门禁,`eval/fd_infer.py` 已实现)
- 协议:v3_val 取 N≥10 ep,从首帧出发,分别注入 ①GT 动作 ②他 ep 动作 ③零动作,各生成 32 帧,算 PSNR/SSIM vs GT 未来。
- 指标:**ΔPSNR(gt − other) > 1.0 dB** 且 **ΔPSNR(gt − zero) > 1.0 dB** → 动作被真正遵循(iter300 时为 −0.21,需训练后翻正)。
- 命令:`bash eval/run_fd_infer.sh --n-episodes 10 --num-steps 8 --guidance 3.0`(CFG/步数见 §5.4 扫参)。

### 5.2 视频保真(逐 horizon)
- v3_val 全集:PSNR/SSIM/**LPIPS**(感知)+ 可选 **FVD**(分布),@1s/3s/7s 三档。
- 基线对照:零样本 I2V 地板(Nano 12.88dB)、GT-action 应远超之。逐相机(top_head/hand_left/hand_right)分别出曲线,定位哪个视角先崩。
- **分机型出指标**(visrobot vs kairobot 各一组 PSNR/SSIM/LPIPS + §5.1 ΔPSNR):这是 §2.1 混训策略 A→C 的判据 —— 若 visrobot 保真显著低于单训预期/被 kairobot 拖累,启用策略 C(visrobot 微调收尾)。

### 5.3 长程 AR rollout(M2/M3)
- 自回归拼接到 10–30s(每段 ref=上段末帧 latent),画逐相机退化曲线、漂移/穿模目检 K 个 ep。
- 抗漂移可借鉴 giga ACWM 的 cond_noise_aug(条件帧加噪)——若 Cosmos3 FD 长程漂移严重,在 ActionDataPacker/数据侧引入等效条件帧扰动。

### 5.4 推理扫参(M1 后)
- CFG guidance ∈ {1,2,3,5} × 去噪步 ∈ {8,10,20,50},网格出 PSNR×时延,选可控性/保真/速度折中点。action CFG dropout 训练侧 0.1(可控性不足再升 0.15)。

### 5.5 下游策略对齐(终极门禁)
- 闭环:policy(kai0 π₀.₅ 或 giga)→ WM 生成 → stage 分类器/VLM 评判 → 估计成功率;与真机/真离线评测算 **Pearson r ≥ 0.8**(WorldEval 同类 0.942)。
- IDM 探针(辅助):用 inverse_dynamics 模式从生成视频反推动作,与注入动作算 MAE,定量验证"动作真的被渲染进了像素"。

### 5.6 横向对比(可选,统一口径)
- 同 v3_val 上把 Cosmos3 FD AC-WM 与 giga `acwm_clothfold`(Wan2.2-5B)做**可控性 ΔPSNR + 时延**对比,沉淀到 `eval/` 报告;注意两者 latent/分辨率口径需对齐再比。

---

## 6. 执行清单(命令)

```bash
# (a) 切 v3:改 wam_fold_dataset.py(per-rig 相机键 §3.2)+ build_wm_data_source(v3 root §3.1)
#     重算 norm stats(§3.3)

# (b) M0 冒烟(1n8g,30步)—— 切 v3 后必跑
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
export BASE_CKPT_DCP=wam_fold_wm_runs/checkpoints/Cosmos3-Nano-dcp
export WAN_VAE_PATH=.../Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export WAM_WM_LATENT_CACHE=wam_fold_wm_runs/latent_cache_v3
packages/cosmos3/.venv/bin/torchrun --nproc_per_node=8 \
  -m cosmos_framework.scripts.train --sft-toml=wam_fold_wm/train/recipe_wm_nano.toml -- \
  trainer.max_iter=30 checkpoint.save_iter=30 \
  model.config.parallelism.data_parallel_replicate_degree=1
# 看:三相机加载无报错、loss 下降、ckpt 落盘

# (c) M1 正式(5n8g):改 aijob json 的 CKPT_DIR→train_out_v3_5n8g 后提交
#     (run 脚本已读 MAX_STEPS/SAVE_ITER/SCHED_CYCLE/REPLICATE_DEGREE)
bash wam_fold_wm/train/aihc/submit_cosmos_wm_aihc.sh   # replicas=5, MAX_STEPS=10000

# (d) 评测(每 ckpt / 收尾)
bash wam_fold_wm/eval/export_ckpt.sh <ckpt_dir>/iter_010000   # DCP→HF
bash wam_fold_wm/eval/run_fd_infer.sh --n-episodes 10 --num-steps 8 --guidance 3.0
# 检查 reports/fd_eval/fd_daction_report.json → ΔPSNR(gt-other) > 1.0
```

---

## 7. 风险与回退

| 风险 | 缓解 |
|---|---|
| 相机键改动影响 v1 实验 | 用 per-rig 表 + v1 回归 1-step smoke;或给 v3 起独立 rig 名隔离 |
| 16B@45k token OOM | selective AC + expandable_segments 已开;再炸则降 max_batch_size→6 或 chunk_length→24 |
| 跨机型相机错位(俯视被压扁) | 主图固定 top_head;smoke 目检三相机帧;对齐 [[fastwam-v3-latent-camera-bug]] 教训 |
| 动作不可控(ΔPSNR≤0) | 升 action CFG dropout 0.1→0.15;查 action 归一化/14-D mask;延长训练;确认 5× lr 生效 |
| LambdaCosine cycle 崩 | cycle_lengths 恒等于总步数(续训时同步改) |
| PFS I/O / CPU-RAM OOM | drop_caches 看护已内置;num_workers≤4;latent 按需缓存监控余量 |
| 混库归一化漂移 | per-rig 分位 stats(domain 16/17 各一份),不共享 |

---

## 8. 与既有工作的关系
- v1 版方案:[[cosmos3-wam-fold-world-model-plan]](本方案直接续作,基建复用)。
- giga Wan2.2-5B ACWM(`acwm_clothfold_vis6/vis8`):平行实现,作横向对比基线(§5.6)。
- 下游策略基线:[[wam-three-way-v3val-compare]]、[[kai0-pi05-local-eval]](提供闭环对齐的 policy 与真值口径)。
</content>
</invoke>
