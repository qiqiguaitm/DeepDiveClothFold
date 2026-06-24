# τ₀-WM 叠衣服微调方案(关节空间 · visrobot01 部署)

> 状态:设计方案(future plan) · 目标本体:visrobot01 · 任务:Flatten and fold the cloth.
> 数据:`kai0/data/wam_fold_v1` · 训练框架:tau0 自有框架(需自建 trainer)
> 末态目标:在 visrobot01 上**关节空间在线直控**验证

---

## 0. TL;DR(最终结论)

- **在 tau0 自己的框架里微调,真·复用其预训练权重**;但**只做关节空间(joint-14),不兼容 eef6d**——目标只是 visrobot01 关节直控,eef6d 是多余约束。
- 利用 tau0 动作分支「**小投影 + 大共享干**」的结构:把 `action_proj_in`/`action_head` 重置为 14 维(随机初始化),**`action_blocks×30` + 视频主干 + VAE + T5 全部加载预训练权重**。这才是"真·复用"——价值在干,不在投影。
- 数据用 `wam_fold_v1` 的 **GigaWorld 构建版直接用**(已净化 + 标准化 + 可复现划分),关节-14 无需任何转换;VAE latent 用 **tau0 自己的 VAE 重抽**。
- 训练两段式:**P1 冻结干暖启 joint 头(兼做先验 go/no-go ablation)→ P2 解冻 `action_blocks` 专精**。
- 部署:新建 `TauPolicyJoint`,**关节进、关节出,免 FK/IK**,直接对接 visrobot01 / Piper 关节控制栈。
- **分水岭**:P1 的 ablation。若冻结干 + 训 joint 头的 vis_val MSE 明显优于"随机干"基线 → tau0 先验有效,继续 P2;若持平 → 退回 **GigaWorld 关节-14**(更省)。

---

## 1. 背景与现状盘点(已核实)

### 1.1 两个 WAM,务必区分
- **τ₀-WM(tau-0-wm,sii-research)**:本工作区为**纯推理发布**(无训练代码),动作空间 **eef6d-20**(`action_in_dim=20`)。11GB 权重已从 hf-mirror 下载到 `tau0_wm/checkpoints/tau-0-wm/`。
- **GigaWorld-Policy(`giga_world_policy/`,gigaai-research,arXiv 2603.17240)**:完整 WAM **训练框架**,动作空间 **关节-14**(模型里 `nn.Linear(14,...)` 硬编码),已有 `visrobot01_fold*` 配置并跑过多轮(`runs/` 有 eval 日志)。
- 二者**架构近亲、代码不同源**,权重不能直接互相加载。本方案选择**在 tau0 框架内**微调(因用户要复用 tau0 预训练先验),GigaWorld 作为**对照/兜底**。

### 1.2 数据 `kai0/data/wam_fold_v1`(LeRobot v2.1,agilex,30fps,480×640,3 相机)
| 子集 | episodes | VAE latent | T5 | 角色 |
|---|---|---|---|---|
| `visrobot01_train` | 1898 | ✅(GigaWorld 版) | ✅ | **目标域** |
| `visrobot01_val` | 200 | ❌(仅 T5) | ✅ | 离线评估 held-out |
| `kairobot01` | 6512 | ✅ | ✅ | **辅助域** |
| `visrobot01` | 2098 | ❌ | ✅ | = train+val 全集 |

- state/action = **14 维关节**(每臂 6 关节 + 1 夹爪;夹爪 idx 6/13 绝对、余者 delta)。
- 三相机键:`cam_high` / `cam_left_wrist` / `cam_right_wrist`(训练 resize 256×192)。
- **GigaWorld「构建」≠ 框架锁定**:`build_wam_dataset.py` 只做无损标准化(相机改名 `top_head→cam_high` 等、去 depth、时间戳规整、连续重编号、av1→h264 硬链接)+ **有益净化**(`repair_action_spikes` 修 kai 编码器尖峰)+ **可复现 held-out**(seed=42、sha256 指纹、`split_map.json`)。parquet+视频是标准 LeRobot → **任何训练代码可用**。
- **kai 仍有 7 个残留脏 episode**:102,1310,1482,1709,1979,4153,5144 → 训练排除。
- **原始 Task_A 此节点不存在**(build 源路径是另一台机)→ 无法、也无必要回原始数据训练。

### 1.3 硬件
当前节点 **8×A100-80GB**(适合 5B 微调);更大规模可上 gf0/gf1。

---

## 2. 关键决策与依据

### 2.1 为什么关节空间而非 eef6d
| | 关节-14 | eef6d-20 |
|---|---|---|
| 数据 | `wam_fold_v1` **直接用**(本就关节) | 需 Piper URDF **FK 转换**(新管线 + 误差源) |
| 部署 | **关节进关节出,免 FK/IK**,贴合 Piper 栈 | 需 IK(时延 + 失败模式) |
| 复用 tau0 先验 | ✅(干照样加载,仅换投影) | ✅ |
| 跨 rig 不变性 | 用 per-embodiment norm + 专精消化 | EEF 天然更不变(理论略优) |
| 工程成本 | 低 | 高 |
> visrobot01 单一目标本体,EEF 的不变性优势可用 norm + 专精廉价拿到;FK/IK 成本不值。**选关节空间。**

### 2.2 为什么不兼容 eef6d(放弃 dual-head/LoRA-保-eef6d)
- 真正的先验在 `action_blocks×30` + 视频主干,**与动作空间无关**;eef6d 头只是 ~3 万参数小投影,丢之不可惜。
- visrobot01 关节直控**不需要 eef6d 输出**。保 eef6d 只在「一份 ckpt 服务多机型/多空间」时才值,当前不沾边。
- dual-head **不降低**核心不确定性(见 §6),只增复杂度 → 不做。

### 2.3 VAE latent 缓存的可复用性(核实结论)
- GigaWorld 缓存用 **diffusers `AutoencoderKLWan`**(Wan2.2-TI2V-5B-Diffusers)+ `(latent−mean)×(1/std)` 归一,256×192。
- tau0 用 **raw `Wan2.2_VAE.pth`**(不同实现,同家族),归一形式相同但**不保证逐值一致**。
- → **不要假设可复用;用 tau0 自己的 VAE 重抽 latent**(或训练时在线解码)。可用「同 clip 双 VAE 编码比 MSE」一测定论。

---

## 3. 架构设计:小投影换新,大共享干复用

### 3.1 tau0 动作分支结构(`model.py`,`use_ae` 分支)
| 模块 | 维度 | 绑动作空间? | 处理 |
|---|---|---|---|
| `action_proj_in` | `Linear(20→1024)` | ✅ | **重置为 `Linear(14→1024)`,随机初始化** |
| `action_head` | `Head(1024→20)` | ✅ | **重置为 `Head(1024→14)`,随机初始化** |
| `action_blocks×30` | `WanAttentionBlock(1024, cross_attn_dim=3072)` | ❌ | **加载预训练**(核心先验,逐层 cross-attn 视频) |
| `action_time_embedding/projection`、`action_freqs` | 1024 | ❌ | 加载预训练 |
| 视频 `blocks×30`+`head`、VAE、T5 | 3072 | ❌ | 加载预训练 |

forward 关键(line 702–814):`history_action_state`(当前 state token,joint 模式下 14 维)拼到噪声动作前 → `action_proj_in` → 30 `action_blocks` → `action_head` → 丢 state token。

### 3.2 改动量
- 子类 `WanModel`,`action_in_dim=14`,`strict=False` 部分加载。
- **新增随机参数仅 ~3 万**(两个 14 维投影);**30 层 action_blocks 与整个世界模型主干一个不丢**。
- 这就是「关节头从零、干是预训练」,**不是从零训练**。

---

## 4. 训练器(tau0 无 trainer,需自建)

新建 `train_tau0.py`,复刻 GigaWorld `CasualWATrainer.forward_step` 的 flow-matching 思路,套 tau0 `WanModel.forward`:

单步训练(非迭代去噪):
1. 3 视图帧 → Wan VAE → 视频 latent;
2. 采样 σ(flow-matching,`flow_shift` 与推理一致);
3. 视频、动作用**同一 σ** 加噪:`x_t = σ·noise + (1−σ)·x0`;
4. `WanModel(..., action_states=noisy_action, action_timestep=σ, history_action_state=state, return_action=True, return_video=可选)`;
5. 目标速度 `noise − x0`;`loss = λ_act·MSE(pred_act) + λ_vid·MSE(pred_vid, 首帧mask)`(纯动作微调可 `λ_vid=0` 省算力)。

- tau0 forward 已内置 `gradient_checkpointing` 与 `store_buffer`(视频特征缓存)。
- DDP:accelerate + deepspeed ZeRO-2(同 GigaWorld,8×A100)。
- **时序对齐(必核)**:tau0 config `chunk=9 / action_chunk=33` vs 数据 30fps → 按帧率设 action 采样 stride。

---

## 5. 数据准备

| 项 | 做法 |
|---|---|
| 关节-14 数据 | `wam_fold_v1`(`visrobot01_train×3 + kairobot01`)**直接用**,排除 kai 7 脏 episode |
| 归一化 | 按 tau0 `meanstd` 约定算 per-embodiment 关节 mean/std(`statistics.json`) |
| VAE latent | **用 tau0 VAE 重抽**(prefer);`visrobot01_val` 也补抽以加速评估 |
| T5 | "Flatten and fold the cloth." 经 tau0 UMT5 编码(或复用,若同 UMT5) |
| held-out | **沿用** `visrobot01_train/val` 的 seed=42 划分(可复现、不重叠、与历史可比) |

---

## 6. 训练计划:两段式(P1 暖启 → P2 专精)

经典「先探针、后微调」,避免新头还是噪声时破坏预训练干。

### P1 — 冻结干,只暖启 joint 头(兼先验 ablation)
- **训练**:仅 `action_proj_in` + `action_head`(~3 万参);**整干冻结**。
- **为何**:随机头的"垃圾梯度"会冲坏预训练干 → 先让小头在固定干上追平。参数少、训得快、近零风险。
- **ablation(go/no-go)**:干冻死 → joint 头精度完全由冻结的预训练干特征决定。
  - vis_val joint MSE **明显优于**「随机干 + 训头」基线 → **先验有效,进 P2**;
  - **持平** → action_blocks 对 joint 帮助有限 → **退回 GigaWorld 关节-14**。
- LR 较高(头小稳),少步数。

### P2 — 解冻 `action_blocks`,专精
- **训练**:joint 头 + `action_blocks×30`(+可选 LoRA);视频主干一般继续冻(或低 LR LoRA)。
- **为何**:让动作动力学层贴合叠衣 + visrobot01 rig,榨出 P1 冻干摸不到的性能。
- **顺序理由**:P1 后头已非随机、梯度有意义,解冻干才安全有益。
- **专精路径**:先 `vis×3 + kai`,再 **vis-only 续训**消域差;LR 较低护干。
- **LoRA 仍可选**,但理由变了:不再是保 eef6d,而是**防 5B 在 ~1898 集上过拟合**;小数据下 LoRA / 只解冻 action_blocks 更稳。

| | P1 暖启 | P2 专精 |
|---|---|---|
| 训练 | 仅 joint 头 | joint 头 + `action_blocks`(+可选 LoRA) |
| 冻结 | 整干 | 视频主干(动作干解冻) |
| LR | 较高 | 较低 |
| 目的 | 头对齐 + **先验 ablation** | 任务/本体深度适配 |
| 产出 | go/no-go + 暖好的头 | 最终可部署模型 |

---

## 7. 离线评估

- 主指标:**vis_val(200 集)joint action MSE / L1**,分维度(尤其夹爪、首末帧),EMA 与 raw 都看。
- 诊断:世界模型 rollout 视频定性看"会不会叠"(非只看 MSE)。
- 对比基线:P1(冻干)vs 随机干;P1 vs P2;(可选)tau0 关节微调 vs GigaWorld 关节-14。

---

## 8. 在线关节空间直控部署(visrobot01)

新建 `TauPolicyJoint`(平行 `TauPolicy.play()`),关节模式比 eef6d **更简单**:

| 步骤 | eef6d 现状 | **joint 模式(目标)** |
|---|---|---|
| state 输入 | eef 四元数→6d | **直接 14 维关节,免 FK** |
| 推理 | in_proj_eef + head_eef | in_proj_joint + head_joint |
| 后处理 | rela_eef6d → `rela_eef_to_abs` | 关节 delta **直接累加**(夹爪按约定),**免 IK** |
| 下发 | EEF → 需 IK | **直接关节指令给 Piper** |

- 全程无运动学转换 → 少误差源、少 IK 失败、少时延。
- 复用 tau0 推理优化(RTC、action cross-attn KV-cache、fused-attn、context-null cache)。
- 对接:相机 `cam_high/left_wrist/right_wrist`,state=14 关节,贴合现 kai0/piper 关节控制栈。
- 分级验证:① 单步动作合理性 + 安全限位 → ② 接管下闭环 → ③ 完整叠衣**成功率**(多布料/初始姿态,统计 N 次)。
- 调 `num_inference_steps`、chunk 执行步数,平衡时延/平滑。

---

## 9. 风险与对策
| 风险 | 对策 |
|---|---|
| **核心赌注**:action_blocks 先验是否跨空间可迁移 | **P1 ablation 廉价定论**;无效则退 GigaWorld 关节-14 |
| 需从零搭 tau0 trainer | 复刻 GigaWorld `forward_step`;先小规模冒烟 |
| joint 头冷启破坏干 | P1 冻干暖启在前 |
| 5B 在 ~1898 集过拟合 | LoRA / 只解冻 action_blocks / vis-only 专精 + 早停 |
| latent 不兼容 | tau0 VAE 重抽(§2.3) |
| 时序 chunk/帧率不对齐 | 核 `chunk/action_chunk` vs 30fps,设 stride |
| kai 脏数据 | 排除 7 个 episode |
| 跨 rig 残差 | per-embodiment norm + vis-only 专精;必要时真机 DAgger 回灌 |

---

## 10. 里程碑
**P0 装配(2–3d:搭 trainer + 14 维模型子类 + 载权重校验)→ P1 暖启 + 先验 ablation(1–2d,go/no-go)→ P2 joint 专精(2–4d)→ 离线 vis_val 选模 → TauPolicyJoint 真机直控验证(2–3d)**。
**P1 ablation 为分水岭**:决定"真·复用 tau0"是否比直接 GigaWorld 关节-14 更值。

---

## 附录 A:被否决/备选方案(决策留痕)
- **eef6d-20 训练(Plan B)**:数据 Piper URDF FK 可转,理论跨 rig 更优,但需改框架 + FK + EEF/IK 部署;仅当关节空间跨 rig 泛化实测不足时启用。
- **dual-head + joint-LoRA 保 eef6d**:仅当需一份 ckpt 同时服务 eef6d 与 joint 时采用;visrobot01 单目标场景不必。
- **从 tau0 权重初始化 GigaWorld**:不同源、命名/维度不符,**不可直接加载** → 已否决。
- **回原始 Task_A 训练**:信息量相同但含未修尖峰、丢可复现划分、且数据不在本节点 → 否决。
- **GigaWorld 关节-14(兜底主线)**:原生、已验证、可直接部署;若 P1 判 tau0 先验无效则采用。

## 附录 B:关键路径
- tau0 推理/模型:`tau0_wm/`(`web_infer_utils/TauPolicy.py`、`models/wan_2_2_models/transformers/model.py` 的 `use_ae` 分支)
- tau0 权重:`tau0_wm/checkpoints/tau-0-wm/`(下载中)
- GigaWorld 框架(参考/兜底):`giga_world_policy/`(`world_action_model/trainer/wa_casual_trainer.py`、`scripts/inference_server.py`、`assets_visrobot01/norm_stats_*.json`)
- 数据:`kai0/data/wam_fold_v1/{visrobot01_train,visrobot01_val,kairobot01}`
- Piper URDF(若走 eef6d):`calib/piper_local.urdf`、`kai0/.../piper_description.urdf`
