# Cosmos3-Nano-Policy-DROID → 叠衣 AC-WM 适配方案

> **目的**: 把 **Cosmos3-Nano-Policy-DROID**(Cosmos3 的策略变体:首帧+任务+state → 未来视频 + 动作轨迹)适配到 `wam_fold_v1` 叠衣任务,得到一个动作条件世界模型(AC-WM),在 val 上评测,并用 **giga_world_policy 同款报告**与 GigaWorld-Policy / tau-0-wm 横向对比。
> **状态**: 📝 设计待评审 ｜ **创建**: 2026-06-06
> **资源**: 本机 8×A100 80G(+ b1 8×A100 可选) ｜ **模型**: 下载中 → `cosmos/models/modelscope/Cosmos3-Nano-Policy-DROID`
> **已决策 (2026-06-06)**: 基座 = **仅 Cosmos3-Nano-Policy-DROID**(不做 base-Nano A/B);动作口径 = **14D 双臂关节**(与 GigaWorld-Policy / tau-0-wm / 数据原生同口径,三方可直接对比)。

---

## 1. 为什么用 Policy-DROID(而非 base Nano)

| | base Cosmos3-Nano | **Cosmos3-Nano-Policy-DROID** |
|---|---|---|
| 输出 | 视频(+sound) | **视频 + 动作轨迹**(policy) |
| 动作头 | 有通道但未策略化 | **已在 DROID 机器人上训练成策略** |
| 适配距离到"叠衣 AC-WM" | 远(要从头训策略) | **近**(已是 AC-WM,只需换 embodiment) |

Policy-DROID 已是"给首帧+指令+state,产未来 obs+动作"的 AC-WM —— 与 giga_world_policy、tau-0-wm 同类。适配 = **换 embodiment(DROID 单臂 → Agilex 双臂)+ fold 域微调**。

## 2. 核心适配 gap

1. **动作表示**: DROID = 单末端 **10D**(9D EEF pose + 1D grasp)。本数据 = **双臂 14D 关节**(`action`/`observation.state` 各 [N,14] = L_j0..5+L_grip+R_j0..5+R_grip,@30fps)。
   - → **重置/重训动作头**(`action_proj_in`/`action_proj_out`/`action_modality_embed`,checkpoint 里 5 个张量,见 cosmos3-i2v-eval-setup memory)为 14D 双臂关节,**与 giga_world_policy/tau-0-wm 同口径(14D raw joint)**,保证三方可直接对比。
2. **运行框架**: Policy-DROID 是 `Cosmos3ForConditionalGeneration`(**cosmos-framework / omni_mot_model**),**不是 diffusers**(diffusers 端无 action 模块,见上轮 I2V 评测发现)。→ **P0: 必须装 cosmos-framework**(`github.com/NVIDIA/cosmos-framework`,uv sync cu124;cookbook 注明需 SSH 访问)。训练用其内置 **rectified-flow 配方**(`action_loss_weight=10`, logitnormal time, base_fps 24, patch_spatial 2)。
3. **VAE**: Nano VAE = `AutoencoderKLWan` z=48, 4× 时间下采样。tau0 预存的 `vae_latent` 是 Wan2.2 VAE,**需用 Cosmos VAE 重算**(或确认同构后复用)。

## 3. 数据管线 (LeRobot → cosmos-framework)

- **划分**: 训练 `visrobot01_train`(1898 ep / 2.81M frame),评测 `visrobot01_val`(200 ep / 296k frame)。
- 每帧三视角 480×640 → Cosmos VAE latent(z48,时间 4× 压缩);14D action chunk;任务文本 → Cosmos text encoder。
- 训练窗:(首帧/参考帧 latent + state)→ 未来 num_frames 视频 latent + action_chunk。块几何对齐 cosmos-framework(base_fps 24, chunk≈48 步动作)。
- 适配 cosmos-framework 的 dataloader 到 LeRobot 格式(或预转其 webdataset/格式)。

## 4. 训练/适配策略 —— **全量微调 (已定 2026-06-06)**

- **warm-start**: 全模型(视频主干 + 文本/视觉编码器 + 扩散专家)用 Policy-DROID 预训练权重;**仅动作头(`action_proj_in/out` + `action_modality_embed`)重置为 14D 双臂**(单臂10D→双臂14D)。
- **全量微调**: **所有 16B 参数可训**(非 LoRA、非冻结),fold 域 rectified-flow 联合微调(action_loss_weight=10,视频:动作 5:1 可调,logitnormal time)。
- **并行/显存**(16×A100, FSDP-16): 16B 全量 ≈ bf16权重32G + fp32master64G + Adam(m,v)128G + grad32G ≈ 256G / 16 分片 ≈ 16G/卡 + 激活 → 80G 充裕;开 **activation checkpointing**(config 自带 full)、bf16。
- **LR**: 全量微调用小 LR(~1e-5–5e-5)+ warmup-cosine(参考 [[wam-p3-fullft-recipe]]:full-FT + warmup-cosine + 5:1 loss);**不要**沿用 SFT 配置的 5e-4(那偏动作头)。
- **顺序**(同一全量微调,分热身→主训,不再冻结):先 ~数百 step 小 LR 让重置的动作头对齐,再放开到目标 LR 全程全量训。
- 在 `visrobot01_train`(1898 ep)微调;`visrobot01_val`(200 ep)留作评测(零泄漏)。

## 5. 评测 (镜像 giga_world_policy/eval_watch.py)

开环逐窗(可选闭环 SR),与 GWP 完全同口径:
- **视频**: PSNR / SSIM / temporal_absdiff_ratio / LPIPS。
- **动作**: action_mae / action_mse / **mae@{1,10,24,48}**(对标 PI05 基线 {1:.022,10:.043,24:.074,48:.116})。
- coverage: `exec`(部署步长)与 `episode`(非重叠全覆盖)两模式;窗口构建复用 `build_window_indices`。

## 6. 报告 (镜像 giga_world_policy/episode_report.py) + 对比

产 `report.html`,每抽样 episode:
- **14D 动作曲线**(pred raw vs GT,沿 exec_horizon 拼接的部署式轨迹)。
- **3 视角视频**(2 行 GT / pred(raw),cam_high|cam_left|cam_right 横排,帧上标行名):全 episode 长视频 + 代表窗短视频。
- **横向对比表**: **Cosmos3-Policy(adapted) vs GigaWorld-Policy vs tau-0-wm vs PI05**,按 video+action 指标 × horizon。复用 GWP 的 HTML/CSS 风格(`episode_report.py` 分片 + `--aggregate`)。

## 7. 阶段与里程碑

| 阶段 | 内容 | 产出 | 预估 |
|---|---|---|---|
| P0 | 下载模型 ✓ + 装 cosmos-framework + 跑通其 Policy-DROID 原生推理(冒烟) | 环境就绪 | 0.5–1 day(取决于 SSH/网络) |
| P1 | 数据转换(wam_fold_v1→JSONL, 14D action) + 动作头重置 + 单卡 smoke | 数据+配置就绪 | 1–2 day |
| P2 | **全量微调**(16×A100 FSDP, rectified-flow, warmup-cosine) | 适配 AC-WM ckpt | 2–4 day(算力相关) |
| P3 | val 评测(eval_watch 口径)+ 报告(episode_report 口径)+ 三方对比 | report.html | 0.5 day |

## 8. 风险 / 待确认(P0 级)

1. ~~cosmos-framework 可获取性~~ **已解**: 公开 HTTPS 可达(代理与直连均 OK),已 `git clone --depth1` 到 `cosmos/packages/cosmos3`。框架自带:`scripts/train.py`、SFT 配置 `configs/base/experiment/sft/vision_sft_nano.py`、OSS 数据层 `DataPackerDataLoader`(写自定义 DataPacker 即可,无需内部基建,见 `docs/custom_dataset.md`)、动作策略推理 `inference/action.py` + `scripts/action_policy_server_*`。
1b. **🔴 新 P0 风险: torch/驱动**: 框架要求 **torch==2.10.0 + cu128/cu130**(无 cu124 组),而本机 **driver 535.261(CUDA 12.2)**——正是当初把 diffusers 环境锁 torch2.6/cu124 的原因。cu128 理论上靠 CUDA minor-version 兼容(driver≥525.60)可跑,但 torch2.10 较新,**正在做最小 probe 验证**(装 torch2.10+cu128 跑一次 GPU op)。若 driver 535 跑不动 cu128 → 需换更新驱动的节点,或退而用自研 torch2.6 训练 loop。
2. ~~动作口径~~ **已定 = 14D raw joint**(与 GWP/tau0/数据同,直接可比;动作头重置训练)。
3. ~~训练范围~~ **已定 = 全量微调**(16B 全参,16×A100 FSDP;仅动作头重置)。
4. **算力/时长**: 16B 微调,2.8M frame 训练集;先小子集(如 100–200 ep)跑通 P1 再放大。
5. **VAE 复用**: 确认 Cosmos AutoencoderKLWan 与 tau0 Wan2.2 VAE 是否同构(同则复用 latent,省大量预处理)。

## 8b. 具体实现入口 (cosmos-framework, 已 clone 到 cosmos/packages/cosmos3)
- **训练入口**: `python -m cosmos_framework.scripts.train`(+ `_train.py`)。
- **AC-WM 微调配置**: `cosmos_framework/configs/base/experiment/action/posttrain_config`(动作模型 post-train,正是"在预训练 policy 上继续微调"的入口)。
- **Nano 模型配置**: `configs/base/experiment/sft/models/nano_model_config.py`(`action_gen=True`, `max_action_dim=64` → 14D padding 到 64, `action_loss_weight=10`)。
- **数据层**: `DataPackerDataLoader` + 自定义 DataPacker(`docs/custom_dataset.md`);数据集格式 `docs/dataset_jsonl.md`(JSONL + `videos/` + `images/` + `videos_5frames/`)。→ 写 wam_fold_v1(LeRobot)→ JSONL 转换器:每窗 = {首帧 image, 任务文本, 14D action chunk, 未来视频}。
- **推理/策略**: `cosmos_framework/inference/action.py` + `scripts/action_policy_server_*.py`;原生推理 `scripts/inference.py`(cookbook 已验证 Policy-DROID T2V/I2V/fd 用法)。
- **环境**: `uv sync --all-extras --group=cu128-train`(driver 535 → cu128;**正在 probe 验证 torch2.10+cu128 能否在 driver 535 跑**)。
- **评测/报告**: 复用 `giga_world_policy/scripts/wam_pipeline/{eval_watch,episode_report}.py`,把 pred 源换成 fold-AC-WM 输出,三方(Cosmos-Policy/GWP/tau0)同口径对比。

## 8c. 双节点 b1+b2 训练/评测拓扑 (已定)
- 两节点同构: **driver 535.261, 8×A100 80G, 共享 /mnt/pfs gpfs**。framework 与 .venv 装在 gpfs(`cosmos/packages/cosmos3/.venv`)→ **一次构建,两节点共用**。
- **训练**(FSDP-16, torchrun 多机): rank0=b2(本机, 192.168.20.128), rank1=b1(192.168.20.169, `ssh -p 429`)。
  ```bash
  # b2 (rank0)
  NCCL_SOCKET_IFNAME=eth0 torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=192.168.20.128 --master_port=29504 -m cosmos_framework.scripts.train <recipe.toml>
  # b1 (rank1, via ssh)
  ssh -p 429 root@120.48.99.93 "cd .../cosmos3 && NCCL_SOCKET_IFNAME=eth0 torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=1 --master_addr=192.168.20.128 --master_port=29504 -m cosmos_framework.scripts.train <recipe.toml>"
  ```
  `data_parallel_shard_degree=-1`(自动 = WORLD_SIZE 16);16B + FSDP16 + activation-checkpointing 显存充裕。
- **评测/报告**(16 GPU 分片): 沿用上轮 cosmos_eval / giga_world_policy 的 2 节点分片(b2+b1),val 200ep 横扫,出 report.html + 三方对比。
- 网络参数复用 `tau0_wm/finetune/launch_gweval_2node.sh`(eth0, 主 IP 192.168.20.128)。

## 9. 立即可做(待你拍板后)
- 验证 cosmos-framework 能否 clone + 装(P0 最高优先,决定整个方案可行性)。
- 同时:确认 Cosmos VAE vs tau0 latent 是否可复用。
- 动作口径定 14D joint(建议)。
