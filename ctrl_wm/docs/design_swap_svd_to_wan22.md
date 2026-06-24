# 设计方案：将 Ctrl-World 的 SVD 骨干替换为 Wan2.2

> 目标：把 Ctrl-World 世界模型的视频扩散骨干从 **Stable Video Diffusion (SVD)** 换成 **Wan2.2 (TI2V-5B)**。

## 0. 结论 (TL;DR)

可行，但 **不是"改个 from_pretrained 路径"那么简单**——SVD 与 Wan2.2 在 VAE 潜空间、骨干结构（UNet vs DiT）、条件注入、训练目标上都不同，等价于**重建一个基于 Wan 的世界模型**。

好消息：本仓库的**兄弟项目 `fastwam` 已经把这件事做完了**——它就是 Wan2.2-TI2V-5B + ActionDiT 的"世界动作模型"，且已适配同一套叠衣服 LeRobot 数据。因此推荐 **方案 B：复用 fastwam 的 Wan2.2 栈**，而不是在 Ctrl-World 里手工把 SVD-UNet 改写成 Wan-DiT（方案 A，工作量大且与 fastwam 重复）。

---

## 1. 现状梳理：Ctrl-World 如何依赖 SVD

| 组件 | 实现 | 关键假设 |
|---|---|---|
| VAE | `AutoencoderKLTemporalDecoder`(SVD VAE) | 4 通道、空间 8× 下采样、**逐帧编码（无时间压缩）** → latent `(T,4,24,40)` |
| 骨干 | `UNetSpatioTemporalConditionModel`(自定义) | 空时 UNet；`added_time_ids`；**frame-level 动作条件**加到时间层 |
| 文本/图像编码 | CLIP text + CLIP image | 动作 hidden + CLIP 文本 embed |
| 条件范式 | image-to-video（当前帧 latent 通道拼接）+ history buffer + 多视角 | 3 相机竖直拼成 `(T,4,72,40)`；逐帧动作与 latent 帧一一对齐 |
| 训练目标 | EDM / v-prediction（SVD 噪声调度） | `P_mean=0.7,P_std=1.6` |
| 动作条件 | `Action_encoder2`: Linear(action_dim→1024)，逐帧加到 UNet | action 与 latent 帧严格对齐 |

→ **整套代码都建立在"latent 帧 = 视频帧（除以 rgb_skip）、4 通道、逐帧动作对齐"之上。**

## 2. SVD vs Wan2.2 架构差异（gap 分析）

| 维度 | SVD（现状） | Wan2.2-TI2V-5B | 影响 |
|---|---|---|---|
| VAE | 4ch，空间8×，**逐帧** | `AutoencoderKLWan` 48ch，空间×、**时间4× 因果压缩** | ⚠️ latent 帧数 ≠ 视频帧数；逐帧动作对齐机制失效 |
| 骨干 | UNet（卷积空时） | `WanTransformer3DModel` DiT（30层，3072 dim，patch[1,2,2]，RoPE） | ⚠️ 动作注入机制完全不同 |
| 文本 | CLIP（512/1024） | umT5-xxl（text_dim 4096） | 文本条件接口变 |
| 目标 | EDM/v-pred | flow-matching（UniPC） | 训练/采样循环重写 |
| 图像条件 | 通道拼接当前帧 | TI2V：首帧 latent + mask（`first_frame_causal`） | history/自回归范式需重设计 |
| 规模 | ~1.5B UNet | 5B DiT | 显存/算力上一个量级 |

## 3. 核心难点

1. **时间压缩破坏逐帧动作对齐**：Wan VAE 把 4 帧压成 1 个 latent 帧。Ctrl-World 的"每个 latent 帧配一个 7/14 维动作"不再成立——必须把动作序列重采样/插值到 latent 帧率，或用"动作 token 序列 + 交叉注意力/MoT"的方式注入（fastwam 的 ActionDiT 正是这么做：`linear_interp` 线性插值动作到 latent 时序）。
2. **DiT 的条件注入**：UNet 里"加到时间层"的做法在 DiT 里要换成 adaLN 调制 / 额外动作 token / **Mixture-of-Transformers 混合注意力**（fastwam `mot_checkpoint_mixed_attn`：视频 DiT 与动作 DiT 双塔，混合注意力耦合）。
3. **自回归 + history + 多视角**：Wan TI2V 用首帧条件 + 因果注意力（`video_attention_mask_mode: first_frame_causal`）原生支持滚动生成；多视角需决定是"竖直拼接"还是"分视角 batch / 多条件"。
4. **权重无法迁移**：SVD checkpoint 与 Wan 完全不兼容，必须从 Wan2.2 预训练（或 fastwam 的 `ActionDiT_..Wan22..pt` 蒸出的 backbone）重新微调。
5. **数据要重新预处理**：需用 **Wan VAE（48ch）** 重抽 latent + **T5** 文本 embedding（不能复用我们为 Ctrl-World 抽的 SVD 4ch latent）。注意：叠衣服数据里**已存在** `vae_latent/`、`t5_embedding/` 预计算目录——很可能就是为 fastwam/giga_world 抽好的 Wan latent，可直接复用。

## 4. 方案 A：在 Ctrl-World 内把 SVD 换成 Wan2.2（不推荐）

需改写：`models/ctrl_world.py`(骨干/VAE/编码器)、`models/unet_*`→Wan DiT、`models/pipeline_*`(flow-matching 采样)、`Action_encoder2`(改为动作 token + DiT 注入)、`dataset_*`(Wan VAE latent + T5)、`train_wm.py`/`rollout_*`。
- 工作量：≈ 重写整个模型层，2–4 周，且与 fastwam 重复造轮子。
- 仅在"必须保留 Ctrl-World 仓库结构/接口"时才选。

## 5. 方案 B：复用 fastwam 的 Wan2.2 ActionDiT 栈（推荐）

`fastwam`（Fast-WAM, arXiv:2603.16666）= **Wan2.2-TI2V-5B + ActionDiT**，已实现：
- `fastwam/models/wan22/action_dit.py` ActionDiT（双塔 MoT 动作注入，`linear_interp` 动作插值到 latent 时序）
- Wan VAE(48ch) + umT5 文本，`load_wan22_ti2v_5b_components`
- `scripts/preprocess_action_dit_backbone.py`（从 Wan2.2 蒸 backbone）、`precompute_text_embeds.py`、`train.py`（deepspeed zero1）
- LeRobot 数据管线（`datasets/lerobot/...`，proprio/action 变换），且 `giga_world_policy/assets_visrobot01_v3` 就是同款数据

**落地步骤**：
1. 环境：按 `fastwam/README` 装环境；模型用本地 `checkpoints/Wan2.2-TI2V-5B-Diffusers` + `Wan2.2-T5`（配置里 `redirect_common_files: false` 已指向本地 dreamzero 原始 .pth，勿改）。
2. 数据：把叠衣服 LeRobot 数据接入 fastwam 的 `configs/data/`（14 维双臂 proprio/action，3 相机映射同我们 Ctrl-World 的处理：visrobot=top_head/hand_left/hand_right，kairobot=cam_high/cam_left_wrist/cam_right_wrist）；用 `precompute_text_embeds.py` 抽 T5、Wan VAE 抽 latent（或直接复用数据集里已有的 `t5_embedding/`、`vae_latent_*`）。
3. backbone：`preprocess_action_dit_backbone.py` 生成/确认 `ActionDiT_..Wan22..pt`（`action_dim=14`）。
4. 训练：`scripts/train_zero1.sh` + `configs/model/fastwam.yaml`，在 8×A100 上微调；用 `eval_fold.py` 在 `visrobot01_v3_val` 验证。
5. 对齐 Ctrl-World 的评测口径（GT vs 预测对比视频、instruction-following）。

**优点**：production-grade、动作-时间对齐/自回归/MoT 已解决、数据已大半就绪；**风险**：5B 模型显存/算力更大，需 deepspeed；与 Ctrl-World 评测脚本需做一层适配。

## 6. 推荐路线与里程碑

- **M1（调研/打通，1–2 天）**：跑通 fastwam 在叠衣服小子集上的训练 1 个 step + 一次 eval（确认数据接入、Wan VAE/T5、ActionDiT 加载）。
- **M2（数据，1 天）**：复用/补齐 `vae_latent`(Wan) + `t5_embedding`，对齐 14 维 proprio 与 3 相机。
- **M3（训练，数天）**：8×A100 微调 ActionDiT；定期 eval_fold 出对比视频。
- **M4（对齐评测）**：把结果接回我们现有的"GT vs 预测"评测口径，与 SVD 版 Ctrl-World 对比指令跟随/视频质量。

> 一句话：**不在 Ctrl-World 里硬改 SVD→Wan（方案A），而是用 fastwam 这套已经做好的 Wan2.2-ActionDiT 当作"Wan 版世界模型"，把叠衣服数据接进去微调（方案B）。**
