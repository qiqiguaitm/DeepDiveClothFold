# 方案:FastWAM 在 visrobot01 叠衣服数据上的 abs-angle 训练与多模型对比评测

状态:**执行中(2026-06-12)**。基线与协议引用 `giga_world_policy/docs/wam_mae_root_cause_and_optimization.md`(终局结果 v2)。

## 执行日志

| 项 | 状态 | 备注 |
|---|---|---|
| M1 venv | ✅(返工一次) | python3.10 缺系统头文件 → triton JIT 编译失败 → **改 python3.11**(box 有 py311-dev);torch 2.7.1+cu128 |
| M1 权重 | ✅ 零下载 | dreamzero 有完整 Wan-AI 原始 repo(DiT 3 shards/VAE.pth/T5.pth/tokenizer),软链至 `checkpoints/Wan-AI/Wan2.2-TI2V-5B`;**`configs/model/fastwam.yaml` 已改 `redirect_common_files: false`**(用本地 .pth,勿改回) |
| M1 ActionDiT 插值 | ✅ | 2.0GB payload,copied=300/interpolated=520/skipped=4(skip 为 action_dim 相关层,训练时按 14 维新建) |
| M1 T5 缓存 | ✅ | 单 prompt("Flatten and fold the cloth.")1 个缓存文件 |
| M1 dataset_stats | ✅ | `data/visrobot01_fold/dataset_stats.json`;1879 eps / 2.78M 窗口;action q01/q99 为 abs 关节角量纲(±3 rad、夹爪 [0,0.08])✓;注意首次计算需 `mkdir runs`(save 到 work_dir 的已知小坑) |
| M1 config | ✅ | `configs/data/visrobot01_fold.yaml` + `configs/task/visrobot01_fold_uncond_1e-4.yaml`(chunk48=num_frames49) |
| M2 smoke | ⏳ | 首次因 triton/py3.10 头文件失败;venv311 重建后重跑 |
| M3 AIHC 5n8g | 未起 | 多机要点:`train_zero1.sh` 仅同步 RUN_ID,**accelerate launch 需补 --num_machines/--machine_rank/--main_process_ip/--main_process_port**;wrapper 待写 |
| M4 eval 适配器 | 未起 | |

旁支:gwp_ori(abs-best)200ep 复测(同 ckpt 重采样)→ .0912@48 vs 原 .0916,**协议运行间方差 ±0.0005~0.0015**,
为终表 v2 的"打平"结论提供不确定度标尺。

## 0. 科学问题(为什么值得做)

FastWAM(arXiv 2603.16666)与我们 ANS 实验是**同一问题的两条对立路线**:

| | 训练期视频 | 测试期想象 | 动作头 | 实测/声称延迟 |
|---|---|---|---|---|
| gwp_ori(GigaWorld 切断,原 abs-best) | 共享 3072 backbone 联合去噪 | 无(action_only) | per-token MLP | 532ms |
| **gwp_ans(我们)** | 同上 + 非对称噪声耦合 | **有**(attend 半去噪视频) | per-token MLP | 283ms |
| **FastWAM** | MoT:视频专家(3072)+动作专家(1024)联合训练 | **无**(动作专家只 attend 首帧 KV) | 专用 1024-hdim ActionDiT×30层 | ~190ms(论文) |

FastWAM 论点:"视频预测的价值在训练期表征,测试期想象可砍";我们 TF/ANS 证据:"测试期想象修通后是净增益(难样本上)"。在**同一份真实双臂数据 + 同一 200-ep 协议**下对比,直接回答:**专用小动作专家(无想象)vs 共享大 backbone(有想象),谁是部署最优**。FastWAM 自带 `infer_joint`(带想象)与 `infer_action`(无想象)两条路径,还能在它自己的架构内做二次消融。

## 1. 基线锚点(已就绪,200-ep 严格同协议)

| 模型 | @1 | @10 | @24 | @48 | act 延迟 |
|---|---|---|---|---|---|
| gwp_ans step50000 | 0.0063 | 0.0288 | 0.0574 | 0.0918 | 283ms |
| gwp_ori(切断) | 0.0053 | 0.0298 | 0.0595 | 0.0916 | 532ms |
| naive lookahead | 0.0044 | 0.0303 | 0.0600 | 0.0969 | 673ms |
| delta-5x | 0.0028 | 0.0347 | 0.0720 | 0.1128 | 636ms |
| pi0.5 | 0.0219 | 0.0425 | 0.0743 | 0.1155 | — |

FastWAM 的目标线:@48 ≤ 0.092(与 gwp_ans/gwp_ori 打平)且延迟 < 283ms 即有部署价值;@48 明显更差则支持"测试期想象/大 backbone 有必要"。

## 2. 数据适配(运气好:robotwin 配置几乎是为我们写的)

fastwam 的 RoboTwin 配置 = **14 维双臂 + 3 相机且 key 同名**(cam_high/cam_left_wrist/cam_right_wrist)+ z-score + LeRobot 格式——与 wam_fold_v1 高度对齐。改动点:

1. **`configs/data/visrobot01_fold.yaml`**(抄 robotwin.yaml 改):
   - `dataset_dirs`: `../kai0/data/wam_fold_v1/visrobot01_train`(×3 oversample 可后补)+ `kairobot01`(可选,先单数据集起步);
   - **abs angle 输出**:`delta_action_dim_mask: null`(全 absolute,即用户要求的 abs 关节角)+ `norm_default_mode: z-score`;
   - **chunk 48 对齐我们协议**:`num_frames=49, action_video_freq_ratio=4` → action 48 步、视频 13 帧(约束 `((49-1)/4)%4=0` ✓,需 smoke 验证显存,fallback `num_frames=33`→chunk 32 则 mae@48 不可测,**不可接受**,必要时降视频帧率 ratio);
   - 分辨率:3 相机 robotwin 布局,起步用其释出值(240×320 族),smoke 后可升;
   - `raw_shape [3,480,640]`,instruction 用我们的 fold prompt。
2. **预处理三件套**(一次性,dev box 跑,产物落 PFS 供 AIHC 离线用):
   - `preprocess_action_dit_backbone.py` → ActionDiT 初始化权重(从 Wan2.2 插值);
   - `precompute_text_embeds.py` → T5 缓存(我们 prompt 单一,秒级);
   - `dataset_stats.json`:首次训练 `pretrained_norm_stats=null` 自动统计(**必须确认是 abs 空间统计**,对照 `assets_visrobot01/norm_stats_vis_abs.json` 的 q01/q99 ±2.3 特征做 sanity)。
3. **权重格式注意**:fastwam 经 diffsynth/modelscope 加载 **Wan-AI 原始格式**(非 Diffusers)。`../checkpoints/` 下需确认有原始格式 Wan2.2-TI2V-5B,没有则 dev box 先下载(AIHC pod 无外网)。

## 3. 环境(独立 venv,不动 cosmos 主环境)

- `python 3.10 + torch==2.7.1+cu128 + deepspeed 0.18.5 + transformers 4.49.0 + hydra`(pyproject 锁死;与 cosmos venv 的 torch 不兼容 → **必须独立 venv**;A100=sm_80,cu128 兼容);
- AIHC 镜像是否带 torch 2.7.1:不带则 venv 装 PFS 上、pod source(同 GWP 做法)。

## 4. 训练方案

- **唯一训练运行**:`fastwam-uncond` 配方(释出默认:AdamW lr=1e-4 cosine、betas=(0.9,0.95)、bf16、ZeRO-1、λ_video:λ_action=1:1),数据=visrobot01(+kai 可选)、abs mask、chunk 48;
- **公平性锚**:abs 系基线都是 ~16M 样本(50k×320)。fastwam 释出 batch=16,等样本要 1M 步不现实 → smoke 实测吞吐后定 batch/epoch,目标**同量级样本数**(配 8×A100 本地起步,不够再上 AIHC 5n8g;注意它在线 VAE 编码、无 latent 缓存,吞吐是最大风险,见 §7);
- 流程:`smoke(单机 8 卡 ~50 步,验 shape/显存/吞吐/loss 降)→ 短训 sanity(~2k 步评一次 MAE)→ 正式训练`;
- ckpt 周期落盘,复用我们 watcher 思路逐 ckpt 评(可选)。

## 5. 评测方案(关键:严格同协议)

1. **离线 MAE 适配器**(fastwam 仓库无离线评测,需新写 `scripts/eval_offline_fold.py`):
   加载 fastwam ckpt → 对 val 的同一组窗口(`episode_report` 的 exec coverage、200 eps、同 GT 对齐)跑 `infer_action` → denormalize(其 stats)→ 与 GT 算 mae@{1,10,24,48}。**输出 summary.json 兼容 cmp 工具**;
2. **延迟**:同一台 A100、同协议计时(它声称 190ms 是其硬件/NFE=20 下的数);
3. **设置矩阵**(一次训练,多设置推理):

| 设置轴 | 取值 | 回答什么 |
|---|---|---|
| 推理路径 | `infer_action`(无想象)vs `infer_joint`(带想象) | FastWAM 论点在我们数据上的架构内消融 |
| 动作 NFE | 20(默认)/ 10 / 5 | 精度-延迟曲线,与 gwp_ans 的 T_a 档位对齐比较 |
| (可选)训练数据 | vis 单源 vs vis×3+kai 混合 | 与 abs 系同数据配比的公平性升级 |

4. 汇总进 official 终表(v3),给部署建议。

## 6. 里程碑

| # | 内容 | 估时 |
|---|---|---|
| M1 | venv + 权重格式确认 + 三件套预处理 + data config | 0.5 天 |
| M2 | smoke + 吞吐实测 → 定 batch/epoch/资源 | 0.5 天 |
| M3 | 正式训练 | 1-2 天(吞吐定) |
| M4 | 离线评测适配器 + 设置矩阵评测 + 终表 v3 + 报告 | 0.5 天 |

## 7. 风险与对策

- **在线 VAE 编码吞吐低**(无 latent 缓存)→ smoke 实测;过低则给 RobotVideoDataset 加 latent 缓存路径(参照 GWP `compute_latents.py`,中等工作量);
- **chunk 48 的窗口约束/显存**(视频 13 帧 vs 释出 9 帧)→ smoke 第一项验证;不行再谈 ratio;
- **torch 2.7.1 与 AIHC 镜像**→ PFS venv;
- **样本数不完全对齐**的公平性质疑 → 报告中明示样本数,辅以训练曲线(MAE-vs-样本数)而非只报终值;
- **stats/单位错位**(z-score vs 我们 q01/q99 习惯)→ 适配器里做 round-trip 校验(参照 `check_delta_abs_roundtrip.py`)。

## 8. 交付物

- `configs/data/visrobot01_fold.yaml` + `configs/task/visrobot01_fold_uncond_1e-4.yaml`
- `scripts/eval_offline_fold.py`(200-ep 同协议适配器)
- 训练 ckpt + dataset_stats.json
- 终表 v3 + 结论报告(并入 giga_world_policy 根因文档体系)
