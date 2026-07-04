# LMWAM v2 整体规划 — 接 kai0 π0.5 VLA(以终为始,2026-07-04)

> v1(pooled subgoal + milestone 预测 + 3 解码器)已快照备份:git tag `lmwm-v1-pooled-20260704`;模型 tar 于 `/vePFS/tim/lmwm_v1_backup_20260704/`。本文规划 v2:把 LMWM 作为**冻结的 world-model provider**接进 kai0 的 π0.5 VLA,对齐 LaWAM 的 action-expert 注入范式,**最终产出下游 SR**(补上与 LaWAM 唯一不可比的 L2 缺口)。

## 0. 终点(以终为始)

一个基于 **kai0 π0.5** 的 **LMWAM VLA**,其 flow-matching action expert 条件于:
1. 当前观测 = DINOv3-H patch-grid **256 token × 1280**
2. 预测未来子目标 û_T = DINOv3-H patch-grid **256 token × 1280**(我们 LMWM patch 预测器产出)
3. 潜在动作 ẑ(从我们 LAM 蒸馏,**64** 维)
4. 语言 l(走 **kai0 的 PaliGemma**,不用 Qwen3-VL)

产出 50 步动作块(action_dim 32),用**下游 SR** 评估 → 直接和 LaWAM 90–98% 摆一起。

## 1. 参照系(两边确切规格)

**kai0 VLA = openpi π0.5**(`kai0/src/openpi/models_pytorch/pi0_pytorch.py`,`models/pi0.py`,`models/pi0_config.py`):
- backbone PaliGemma `gemma_2b`:width **2048** / 18 层;action expert `gemma_300m`:width **1024** / 18 层,双专家共享注意力(`gemma.py`)。
- prefix(backbone 2048,双向,KV-cache):图像 3 视图 × 256 = 768 token(SigLIP So400m)+ 语言 ≤200 token。
- suffix(action expert 1024,因果):state(π0.5 离散化进 prompt)+ 噪声动作 (b,50,32)→1024 + 时间(π0.5 走 **adaRMSNorm**)。
- action head:rectified flow,`action_dim=32`,`action_horizon=50`,Euler 10 步。
- **已有 LeWM 前端**(`models_pytorch/lewm_vision_encoder.py`):可用 **DINOv3-ViT-L/16 冻结**旁路 SigLIP → 复用此 bypass 模式,但换 **DINOv3-H(1280)全 256 token**(不做 15-token 压缩)。

**LaWAM(arXiv 2606.15768)action expert 注入**:u(256×768 patch)+ û_T(256×768 patch 未来特征图,"compact subgoal")+ ẑ(latent action,蒸馏)+ l(Qwen3-VL,1024);**Alternate-DiT** 交替 VLM 流与 dynamics 流 `[u | û_T]`;**latent-action distillation** `L_distill=‖ẑ−z‖²` + **Knowledge Insulation (KI)** 保护 world model 不被策略梯度覆盖。

## 2. 四路输入映射(LaWAM → 我们的 provider → π0.5 注入点)

| 路 | LaWAM | 我们的 provider(冻结) | 注入 π0.5 的位置 | 维度/投影 |
|---|---|---|---|---|
| 当前观测 u | 256×768 ViT-B patch | DINOv3-H `encode_grid` → 256×1280(复用 LeWM bypass) | **prefix**(替换 SigLIP 图像槽) | 1280→2048 |
| 未来子目标 û_T | 256×768 未来特征图 | patch 预测器 `predict_deploy_patch`(deploy grid-cos 0.653)→ 256×1280 | **prefix**(新增视觉流,KV-cache 一次) | 1280→2048 |
| 潜在动作 ẑ | latent-action query token | policy-prior 头 `predm(obs)`→64;teacher=`inverse(g_t,g_future)`→64 | **suffix**(action expert,新增 1 个 latent-action token) | 64→1024 |
| 语言 l | Qwen3-VL 1024 | —(用 kai0 PaliGemma) | prefix(π0.5 原生语言路) | 原生 2048 |

**为何 u 和 û_T 都进 prefix 而非 LaWAM 的 dynamics 流**:π0.5 是**共享注意力双专家**,prefix 只编码/缓存一次,action expert 通过共享注意力天然"看到"prefix → 用 π0.5 原生机制即可达到 LaWAM"action expert 看到 [u|û_T]"的效果,**无需照搬 Alternate-DiT**(减少踩坑)。u 与 û_T 加**可学习 segment/type embedding** 区分。

## 3. 冻结 provider 模块(M2 产物)

把 v1 LMWM 封成一个**纯前向、全冻结**的 `LMWMProvider`:
```
obs(当前帧 256² + [可选]历史/state) →
  obs_grid   : (256,1280)  = DINOv3-H.encode_grid(frame)         [冻结编码器]
  subgoal    : (256,1280)  = patch_predictor(obs_grid)           [冻结, deploy 无 future peek]
  z_prior    : (64,)       = predm(augin⁺)                        [策略 prior 的蒸馏目标之一/初值]
  z_target   : (64,)       = inverse(g_t, g_future)  [仅训练期, teacher, 需 future]
  milestone  : top-k id+conf  [可选, 折进 prompt]
```
- **全部 stop-grad**(KI):VLA 训练**不回传**到 encoder/predictor/inverse/forward。
- 复用:`crave.encoders.encode_grid`、`train_lawm_patch.py`/`predict_deploy_patch.py`(patch 预测器)、`fwd_from_current`(inverse/forward code)、`patch_dec.pt`(仅可视化)。

## 4. 训练配方(LaWAM 式三段,KI 强制)

- **Stage 0(已完成 v1,已备份)**:预训练 LMWM(冻结 DINOv3-H + milestone 头 + LAM inverse/forward + patch 子目标预测器)。
- **Stage 1(v2 核心)**:**冻结 LMWM**,训练 VLA:
  `L = L_flow + λ_distill·‖ẑ_policy − z_target‖² (+ λ_wm·subgoal 一致性辅助)`,λ≈0.1,**KI**(world model 梯度隔离)。
  只训:三个投影(1280→2048×2、64→1024)+ latent-action token + action expert(可 LoRA)。
- **Stage 2(可选)**:小 LR 联合微调(仍 KI 或部分解冻)。
- 评估:**下游 SR**(kai0 held-out + vis_base 跨数据集)+ 保留 intrinsic latent-cos。

## 5. 关键踩坑清单(减少不必要的坑 — 核心)

1. **KI 必须做**:VLA 训练全程**冻结 LMWM**(encoder/predictor/inverse/forward),否则 world model 被策略梯度污染 → 子目标质量崩(LaWAM 明确结论)。只训投影+expert+蒸馏头。
2. **obs 与 subgoal 必须同空间**(都 DINOv3-H 1280),否则子目标不是有意义的"未来 obs"。我们 patch 预测器已输出 DINOv3-H grid ✓。
3. **新投影零/小初始化**:1280→2048、64→1024 的新 token 以**近零残差**接入,训练中 ramp-up,避免一上来冲击预训练 π0.5(VLA 微调常见坑)。
4. **子目标质量中等**(deploy grid-cos 0.653):(a) 必做 **with/without subgoal 消融**验证它真帮忙;(b) 训练期 **subgoal dropout** 提鲁棒;(c) flow 头能学着对噪声子目标降权。LaWAM û_T 也是单步/不完美但有用(dynamics-aware)。
4. **token 预算**:256(obs)+256(subgoal)+≤200(语言)≈700 token/单视图,尚可;多视图再加要留意注意力开销 → **v2 先单 top-head 视图**,wrist 视图后加。
5. **编码分辨率一致**:obs 与 subgoal 都走 256²→16×16=256 token 的同一 DINOv3-H 路径(别用 LeWM 的 288×384 混入),保证两者可比。
6. **用 π0.5 不用 π0**:adaRMS 时间 + state 离散进 prompt,对齐 kai0 默认;投影目标是 expert 宽 1024。
7. **proprio**:kai0 π0.5 把离散 state 折进 prompt → 先对齐 kai0 保留;但记 LaWAM **刻意不给 proprio**(防过拟合)为一个消融。
8. **不覆盖 v1**:v2 用新 ckpt 目录 + 已 tag/tar 备份 ✓。
9. **milestone 计划**:先不注入(收紧 scope),后续作为 prompt token 试(π0.5 已有 state 折进 prompt 的先例)。

## 6. 以终为始的里程碑(倒推)

- **M5(终点)**:LMWAM VLA 在 kai0 held-out + vis_base 跨数据集测 **SR**,对齐 LaWAM 90–98%。
- **M4**:训练集成 LMWAM(flow + distill + KI)→ **集群提交**(长训练走 submit-training-job,不本地 2 卡)。
- **M3**:把四路接进 π0.5 action expert —— 新 `Pi0Config` 变体(obs-grid/subgoal-grid prefix 注入 + latent-action token + 投影),改 `pi0_pytorch.py embed_prefix/embed_suffix`。
- **M2**:封 `LMWMProvider`(冻结,输出 obs_grid/subgoal/z_target/z_prior),复用 v1 脚本;小 smoke 验证 shape。
- **M1**:定接口契约 + config 变体 + **前向 smoke(不训练)**,验证 token 拼接/维度/KI 生效。
- **M0(已完成)**:备份 v1 code(tag)+ models(tar)✓。

## 7. 复用索引(可直接复用的既有工作)

- kai0 侧:`lewm_vision_encoder.py`(DINOv3 旁路模式)、`pi0_pytorch.py`(`PaliGemmaWithExpertModel`、`embed_prefix/suffix`)、`pi0_config.py`(config 扩展点)、`train_pytorch.py`(训练入口)。
- LMWM 侧:`crave.encoders.encode_grid`、`train_lawm_patch.py`/`predict_deploy_patch.py`(patch 子目标预测器)、`fwd_from_current`(inverse/forward 64 码)、`lever_patch_token.py`、`patch_dec.pt`(可视化)。

## 8. 待定/需拍板的分叉(其余已按最小踩坑默认)

1. **latent-action ẑ 是否真接**:LaWAM 有,我们有原料(64 码)。接=更对齐但多一个蒸馏头+token;不接=先只做 obs+subgoal+language。**默认:接**(这是与 LaWAM 最实质的对齐点)。
2. **多视图**:v2 先单 top-head;是否一开始就上 3 视图?**默认:先单视图**。
3. **action expert 全训 vs LoRA**:**默认 LoRA**(`gemma_300m_lora` rank32,省显存、稳)。
4. **base ckpt**:从哪个 π0.5 base 权重起(pi05_base / kai0 已有折叠 ckpt)?需确认 kai0 现有最佳折叠 π0.5 checkpoint 路径。
