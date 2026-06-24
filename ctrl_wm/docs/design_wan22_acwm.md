# 设计方案（修订）：基于 Wan2.2 的 AC-WM（动作条件世界模型）

> 目标更新：我们要的是 **AC-WM**——Ctrl-World 范式的**纯动作条件世界模型**：
> `P(未来视频 | 当前帧, history, 给定动作, 文本)`，动作来自**外部策略**(π₀.₅)，模型**只生成视频**，用于策略在环 (policy-in-the-loop) 想象 rollout。骨干从 SVD 换成 **Wan2.2**。

## 0. 结论 (TL;DR)

不必从零把 SVD-UNet 改写成 Wan-DiT。**FastWAM 的"视频专家"本身已经是一个 Wan2.2 版 AC-WM**：其 `training_loss` 里 `video_expert.pre_dit(x=noisy_video, action=action, ...)` 吃的是 **clean action**（不是加噪动作），即"视频以给定动作为条件去噪"——正是 AC-WM。FastWAM 额外的 **action 专家 + action 损失** 才把它变成"联合世界-动作模型 (WAM)"。

→ **方案：复用 FastWAM 的 Wan2.2 视频骨干 + 动作条件，退化掉动作生成分支，得到纯 Wan2.2-AC-WM；外部接 π₀.₅ 在环 rollout。**

## 1. 证据（FastWAM 视频分支=AC-WM）

`fastwam/src/fastwam/models/wan22/fastwam.py::training_loss`：
- 视频：`add_noise(video)` → `video_expert.pre_dit(x=latents, timestep=timestep_video, context=text, action=action)`，**`action` 为 clean**；首帧用 `first_frame_latents` 覆盖（TI2V 图像条件）。
- 动作：`add_noise(action)` → `action_expert.pre_dit(noisy_action,...)`（这才是"生成动作"的策略分支）。
- MoT 混合注意力 `_build_mot_attention_mask` 把 video / action token 耦合。
- 损失分两路：`loss_video`（flow-matching on video）与 `action_loss`（动作 MSE），加权合并。

`video_expert.pre_dit(..., action: Optional=None, ...)` 原生支持把动作喂进视频 DiT。

## 2. SVD-Ctrl-World 与 Wan2.2-AC-WM 的对应关系

| Ctrl-World (SVD) | Wan2.2-AC-WM (基于 FastWAM 视频专家) |
|---|---|
| `Action_encoder2`: Linear(action_dim→1024) 逐帧 | 动作经 `action`→video_expert 注入 + （可选）MoT 动作 token；动作时序用 `linear_interp` 对齐到 latent 帧 |
| SVD VAE 4ch 逐帧 `(T,4,72,40)` | Wan VAE 48ch + 时间4×压缩；多视角拼接方式需定（见 §4） |
| UNet + v-pred | Wan DiT(30层) + flow-matching |
| CLIP 文本 | umT5-xxl(4096)，可预计算 `t5_embedding` |
| image-to-video + history buffer | TI2V `first_frame_latents` + `first_frame_causal` 因果注意力做自回归 |
| 外部 π₀.₅ 提供动作 | **保持一致**：外部策略提供动作，模型只出视频 |

## 3. 把 FastWAM 退化为纯 AC-WM 的两种实现

**实现 A（最小改动，推荐先做）——"冻结动作生成"模式**
- 训练时设 `action_loss` 权重 = 0（只保留 `loss_video`）。
- 动作以 clean 传入（动作专家相当于一个动作编码器，给视频分支提供动作 token，不被监督生成）。
- 推理 rollout：动作来自外部 π₀.₅（GT/策略动作），只跑视频去噪。
- 改动面：训练 loss 加权 + 推理脚本传外部动作；几乎不动模型结构。
- 缺点：仍加载/前向 action 专家（略多算力）。

**实现 B（更纯/更轻）——"去掉动作专家"模式**
- 只保留 `video_expert`，动作仅通过 `video_expert.pre_dit(action=...)` 注入（去掉 MoT 与 action 专家）。
- 需确认 `video_expert` 在无 MoT 时能独立消费 action（看 `pre_dit`/`post_dit` 对 action 的使用路径）；可能要把动作 token 改为直接 cross-attn / adaLN。
- 模型更小更快，是"干净的 Wan2.2-AC-WM"。
- 改动面：模型前向裁剪 + backbone 载入只取 video 专家。

> 建议：**先 A 打通**（验证 Wan-AC-WM 在叠衣服上可训可评），**再视需要做 B 瘦身**。

## 4. 需要明确的设计决策

1. **多视角**：Ctrl-World 把 3 相机竖直拼进 latent。Wan 高分辨率 + patch[1,2,2] 下，选 (a) 竖直拼接成一张大图，或 (b) 3 视角各自 latent 后在 token 维拼接 / 加视角 embedding。建议 (b) 更契合 DiT。
2. **动作-时间对齐**：Wan VAE 时间 4× 压缩 → 用 fastwam 的 `linear_interp` 把 14 维动作序列插值到 latent 帧时序（已有实现）。
3. **history/自回归**：用 TI2V `first_frame_latents` + `first_frame_causal` mask 做滚动；与 Ctrl-World 的 history buffer 等价。
4. **动作维度**：`action_dim=14`（双臂），与我们 SVD 版一致。
5. **数据**：用 **Wan VAE(48ch)** latent + **T5** 文本——叠衣服数据已自带 `vae_latent*/`、`t5_embedding/`（很可能就是 Wan 格式，需核对 shape/scaling 后复用），否则用 fastwam 的 `precompute_text_embeds.py` + Wan VAE 重抽。

## 4b. M0 核对结果（已完成）：Wan latent / T5 可复用 + 更优基座 GigaWorld-Policy

- `kai0/data/wam_fold_v3/*/vae_latent`(及 `vae_latent_v3fix`)= **48 通道 Wan2.2 latent**，`(N_win, 48, 4, 24, 20)` bf16，滑窗 `{starts, stride}`（每窗 4 帧=Wan VAE 时间块），2351 episodes 全覆盖（各 131G）。`vae_latent_c33/c9` 为空。→ **直接可复用为 Wan AC-WM 输入。**
- `t5_embedding/*.pt` = `(seq, 4096)` umT5-xxl，逐 episode（eval 用 `target_len` padding）。→ **可复用。**
- 这些缓存由 **`giga_world_policy` (GigaWorld-Policy)** 生成：一个**基于 Wan2.2-TI2V-5B 的世界-动作模型(WAM)**，`CasualWorldActionTransformer`，**动作为主输出、视频可选**；自带 visrobot01_v3 配置(`view_keys=top_head/hand_left/hand_right`、`LatentEpisodeSampler`、`delta_frames=range(0,49,4)`)、`compute_t5_embedding`/`compute_norm_stats`/`train`/`inference_server`/`eval_fold_gwp`。与 fastwam 同源(配置注释提到 fastwam-v4 对齐)。

**对方案的影响**：Wan 基座从 fastwam **切到 giga_world_policy**（为这套叠衣服数据量身、latent 已就绪）。AC-WM = 把 giga 的 WAM 由"动作为主、视频可选"**翻转为"视频为主、动作为给定条件"**：
- 训练以视频 flow-matching 损失为主（λ_action→0 或仅作 clean 条件，类似其 `independent_action_sigma` 路径但动作不去噪）；
- 推理时动作来自外部 π₀.₅，模型只 rollout 视频；
- 把 §5b 的 Ctrl-World trick（带噪多帧 history、条件帧噪声增强、随机步长）加到其视频分支。

## 4c. M1 已验证(smoke 通过)

复用共享 venv `/mnt/pfs/p46h4f/cosmos/.venv`(torch2.6/cu124、giga-*、diffusers0.37、deepspeed0.19)+ 本地
Wan2.2-TI2V-5B + `vae_latent_v3fix`。新增 `giga_world_policy/world_action_model/configs/acwm_clothfold.py`
(+`_smoke.py`)与 `trainer/acwm_trainer.py`(`ACWMTrainer`)。1-step smoke(单卡, batch=1, 2 步)结果:
```
Step[1/2] visual_loss 0.7340 / total 0.7340
Step[2/2] visual_loss 0.0054 / total 0.0054   (cuda 15.4/26.9G)
Save transformer + ACWMTrainer state  →  Total_time 0:30, 无报错
```
→ AC-WM(仅视频损失、动作 clean 注入、条件帧噪声增强)端到端可训可存。run 命令见 `configs/ACWM_README.md`。

## 4d. ③ 多帧 history(已实现+验证)与 ② 统一重抽(已就绪)

**③ 带噪多帧 history —— 已实现并 smoke 通过(K=2)**:
- transformer `transformer_wa_casual.py` 三个 forward 变体把 `num_ref_tokens = pw*ph` 改为
  `num_ref_frames = ref_latents.shape[2]//p_t; num_ref_tokens = num_ref_frames*pw*ph`(K=1 时行为不变,向后兼容)。
- `ACWMTrainer.forward_step`:前 K 帧取 `visual_latents[:,:,:K]` 作条件、逐帧独立加噪(`_augment_ref`),
  `first_frame_mask[:,:,:K]=0`,`num_clean_latent_tokens=frame_per_tokens*K`,损失只在未来帧。
- smoke(num_history=2, visrobot, 单卡 2 步):`visual_loss 0.5262→0.0035`,无形状错误,checkpoint 正常。
- 约束:K < latent 窗口 T_lat(visrobot v3fix T=4→K≤3;kairobot T=2→K≤1)。

**② 统一重抽 —— 已就绪(脚本+配置)**:
- `compute_latents.py` 加环境覆盖 `GWP_VIEW_KEYS / GWP_OFFS / GWP_OUT_SUBDIR`(默认不变,向后兼容)。
- `scripts/wam_pipeline/run_unify_latents.sh`:按各库正确相机序 + 13 帧窗(`OFFS=range(0,49,4)`→T_lat=4)
  重抽两库到 `vae_latent_uni`(都 12×48, T_lat=4)→ 可混训 + 两库都支持 K>1。
- `configs/acwm_clothfold_mix.py`:混训 visrobot+kairobot(各自 view_keys)、val=visrobot01_v3_val、num_history=2。
- 注:全量重抽是 GPU 重活(~9000 eps 过 Wan VAE),与当前 SVD 提取争用;建议提取空闲后用
  `NS=8 bash scripts/wam_pipeline/run_unify_latents.sh` 启动。

## 4e. 跨本体(kairobot01 vs visrobot01 不同机械臂)处理

两库:同任务(叠衣)、同模态(14维双臂关节+3视角),但**运动学/外观/相机命名不同**。giga 的处理(已对齐到 `acwm_clothfold_mix.py`):
1. **每本体独立归一化**(关键):`robotype_to_embed_id={visrobot01:0,kairobot01:1}` + `norm_path=[norm_stats_vis_abs, norm_stats_kai_abs]`(`norm_paths[embed_id]`)。否则动作条件被错误归一化。
2. **视觉 ref/history 帧自带本体信息**:模型无显式 robot-id embedding,本体区分靠视觉锚点(对世界模型足够)。
3. **一致相机语义序**(②保证):两库都 [俯视,左腕,右腕]。
理由:布料形变物理机器人无关 → 混训共享物理;AC-WM 下动作仅条件,per-emb 归一化 + 视觉锚点即良定义。备选:分别训练 / 加 robot-id token(通常非必需)。

## 4f. 全自主编排(已启动)
`ctrl_wm/logs/acwm_master_orchestrate.sh`(选项1):等 SVD 提取结束 → `run_unify_latents.sh` 重抽 vae_latent_uni →
`acwm_clothfold_mix_smoke`(4步,验证跨本体 dataloading+两套 norm+K=2)→ 全量 `acwm_clothfold_mix`(8卡)。
日志:`logs/{acwm_master,unify_full,acwm_mix_smoke,acwm_mix_train}.log`。

## 5. 实施阶段

- **M0 复用确认**：核对 `kai0/data/wam_fold_v3/*/vae_latent*`、`t5_embedding` 是否为 Wan2.2 格式（48ch / umT5）→ 能否直接喂 fastwam。
- **M1 小子集打通（实现 A）**：fastwam 配置里 `action_loss_weight=0`、接叠衣服数据（14维 proprio、3相机映射）、跑 1 个 step + 一次 `eval_fold`（看视频 rollout PSNR/SSIM）。
- **M2 训练**：8×A100 deepspeed zero1 微调视频专家（动作 clean 条件），定期出"GT vs 预测"对比视频，`visrobot01_v3_val` 验证。
- **M3 在环 rollout**：外接 π₀.₅，复刻 Ctrl-World 的 policy-in-the-loop 想象评测（指令跟随）。
- **M4（可选）实现 B 瘦身**：去 action 专家，得纯 AC-WM，重训/微调并对比。

## 5b. 要从 Ctrl-World 吸收的设计 trick

FastWAM 已具备:文本/动作 CFG、多相机拼接、首帧条件、动作-时序 `linear_interp`、future-only loss、(固定) temporal stride。**需要从 Ctrl-World 额外吸收**：

- ✅ **带噪多帧 history 条件(最重要)**：Ctrl-World 用 `num_history=6` 个过去帧作条件、每帧加独立噪声 `sigma_h~|N|*0.3`，loss 只在未来帧；FastWAM 只用单个 clean 首帧。世界模型做长程自回归 rollout 时，多帧历史给时间一致性、history 加噪缩小 train/test gap（推理时 history 是模型自生成的不完美 latent）。Wan 下用 flow-matching 等价做法（把条件帧向噪声插值一个小 τ）。
- ✅ **条件帧噪声增强**（当前/首帧 `sigma≤0.2`）：抗自回归漂移，与上一条配套。
- ✅ **随机时间步长** `skip=randint(1,2)`：训练覆盖多种动作速度/帧率，提升指令/动作跟随泛化（FastWAM 是固定 stride）。
- 🔸 可选：history dropout(15%)、非均匀多尺度历史采样 `[0,0,-12,-9,-6,-3]`。
- ❌ 不照搬：EDM 噪声调度/v-pred（Wan 用 flow-matching）、多视角"竖直拼一张大图"（DiT 下改各视角独立 latent + 视角 embedding）。

## 6. 与当前 SVD 版叠衣服微调的关系

当前正在跑的 **SVD 版 Ctrl-World 叠衣服微调**（kairobot 提取中→自动训练）是 baseline；Wan2.2-AC-WM 是其升级骨干版，二者评测口径（GT vs 预测、指令跟随）保持一致以便对比。

> 一句话：**Wan2.2-AC-WM = 取 FastWAM 的视频专家(本就是动作条件世界模型)，关掉动作生成分支(loss=0/去 action 专家)，外接 π₀.₅ 在环 rollout。** 不重写骨干，复用成熟 Wan2.2 栈。
