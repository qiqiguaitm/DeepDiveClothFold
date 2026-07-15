# 计划:用 LMWM 替换 LaWM 接入同一 starVLA,DINOv3-base 下对比(2026-07-12)

> 目标:把 LaWAM 里的世界模型 **LaWM 换成我们的 LMWM**,**两者都在 DINOv3-base 特征下训练**,接入**同一个 starVLA(Qwen3-VL-2B)**,训出策略后在同一 benchmark 上对比下游 SR。用 **gf3 8×H20** 加速。
> 基础:LaWAM 已在本地跑通并复现(LIBERO libero_10 98.0% vs 论文 97.0%),见 [[reference_lawam_repro_local]] / `lawam/docs/LAWAM_reproduce_and_kai0_sft_plan_2026-07-12.md`。

---

## 0. 实验设计(为什么这么比)

**控制变量的干净 A/B**:VLA 基座、注入位置、训练数据、DINOv3-base 特征空间**全部相同**,唯一变量 = 世界模型/子目标提供者(LaWM vs LMWM)。→ 直接回答:**我们的 milestone 世界模型作为 latent subgoal,是否比 LaWAM 的 latent-action 世界模型给策略带来更好/相当的下游 SR。**

| 变量 | 固定为 |
|---|---|
| VLA 基座 | starVLA = Qwen3-VL-2B(前16层)+ action expert(Alternate-DiT) |
| 特征空间 | **DINOv3-base**(= LaWM 用的 `dinov3-vitb16`,我们 LMWM 也切到 base) |
| 注入接口 | starVLA 的 LAM 槽(`framework.action_model.lam_*` + `policy_backend.lam.`/`vlm_to_lam.`) |
| 训练数据 / 配方 | 同一份(LIBERO 或 RoboTwin 或 kai0)+ 同 SFT recipe |
| **唯一变量** | 世界模型:**LaWM(baseline)vs LMWM(ours)** |

---

## 1. 待替换的接口(已摸清)

LaWAM 的 LaWM 通过 `train_*.yaml` 的 `framework.action_model` 加载:
- `lam_ckpt_path` + `lam_yaml_path` → LAM 权重/配置(`latent_action_model/`,VQ-VAE + DINOv3 编码器)。
- ckpt 内 `policy_backend.lam.`(660 keys)+ `policy_backend.vlm_to_lam.`(19 keys,VLM→LAM 适配)。
- 关键开关:`future_prediction: true`(LAM 预测未来特征)、`detach_future_feature: true`、`enable_loss_distill: true`、`num_action_queries: 8`、`perceptual_weight: 0.1`。
- **契约**:给定当前观测(→DINOv3 特征),LAM 出一个 latent subgoal(未来特征/code),action expert 经 `vlm_to_lam` 消费它生成动作。

**LMWM 要满足的同一契约**:current DINOv3-base 特征 → 预测 subgoal latent(我们的 milestone+1 生成 grid/code),tensor 形状/空间与 action expert 期望一致。

---

## 2. 我们 LMWM 的现状 vs 需要的改动

| 维度 | LMWM 现状(见 LMWM2_FINAL) | 本计划需要 |
|---|---|---|
| 特征空间 | 主线已切 **SigLIP 同空间**(E2 deploy 0.716);DINOv3-H 作离线 label | **切回 DINOv3-base** 作在线特征(对齐 LaWM);need 重抽特征+重训预测器/生成器 |
| 输出 | milestone+1:预测器(MDN code)+ 生成器(AdaLN grid ĝ) | 映射成 starVLA LAM 槽期望的 subgoal latent(形状/归一化对齐) |
| 训练数据 | kai0 叠衣 | 换到对比 benchmark 的数据(LIBERO/RoboTwin,LaWAM 有 baseline)或双方都用 kai0 |
| 接口 | `lmwm.runtime.UnifiedLMWMPredictor` | 包一层 adapter 冒充 starVLA 的 LAM(实现 lam 的 forward 契约) |

---

## 3. 分阶段计划(gf3 8×H20 加速)

### P0 · 钉死 LAM↔starVLA 契约 ✅ 已完成
产出:[`LAM_starVLA_contract_2026-07-12.md`](LAM_starVLA_contract_2026-07-12.md)。**关键洞察**:LAM 对 backend 只暴露 3 个方法,其中 **`decoder(h_t, action_emb)` 就是世界模型预测**(LaWM 预测 next-frame 特征)。→ **最干净的替换 = 只换 `decoder`**:
- **保持共享**:DINOv3-vitb16 编码器(`extract_vision_features`)、starVLA、注入链路(vlm_to_lam/query/flow)全不动。
- **只换世界模型**:`decoder` 从"预测 next-frame 特征"换成 **LMWM 预测 next-milestone 特征**(DINOv3-vitb16 空间 [B,256,768])。
- **唯一变量 = 未来预测目标:next-frame(LaWM)vs next-milestone(LMWM)**。这正是我们要证的科学问题。

### P1 · LMWM 在 LIBERO(DINOv3-base)重训,产出 milestone decoder(gf3 8卡)
按 P0 洞察,LMWM 要产出的 = **`decoder(h_t[B,256,768], code[B,1,32]) → next-milestone 特征[B,256,768]`**(DINOv3-vitb16 空间)+ 可选 teacher。步骤:
1. **LIBERO demo → DINOv3-vitb16 特征**(用 LaWM 同款 encoder,复用 crave 特征管线换 encoder)。
2. **CRAVE 在 LIBERO(DINOv3-base)上挖 milestone**(零训练聚类+顺序化)→ 每帧 milestone id + milestone+1 目标帧。
3. **构造训练对**:`(frame_t 特征, milestone+1 帧特征)` 替代 LaWM 的 `(t, t+Δ)` 对 —— **这是差异的根源**。
4. **训 LMWM 生成器**(AdaLN,复用 LMWM2_FINAL)在 DINOv3-vitb16 空间预测 next-milestone grid;teacher 编码 (t, milestone+1)→32-dim code(对齐 code_dim=32,喂蒸馏)。
5. gf3 8卡并行 sweep(复用 `run_gf3_sweep.sh`)。SigLIP 版**留接口**(encoder 可切),本轮不训。
- kill:LMWM 在 DINOv3-base 的 next-milestone 重建 cos 太低(<持久基线),先修目标再进 P2。

### P2 · LMWM→starVLA LAM adapter(接线)
- 写 `LMWMasLAM` adapter:实现 P0 的契约,内部调 `UnifiedLMWMPredictor` 出 subgoal,投影到 LAM 槽期望的 latent 形状。
- 冻结 LMWM provider(纯前向),只训 `vlm_to_lam` + action expert(同 LaWAM 的 freeze_policy 策略)。
- 单机 2×A100 冒烟:forward 通、loss 下降、不崩。

### P3 · 训 starVLA-with-LMWM(gf3 8卡)+ baseline
- 用 `train_lawam.sh` 同一 recipe,`framework.action_model` 指向 LMWM adapter;从 `lawam_pretrain` 初始化。
- **同时**跑 LaWM baseline(released `lawam_*_sft` 已有,或用相同数据重训一遍保证公平)。
- gf3 8卡 DDP(`train_lawam_distributed.sh`)。

### P4 · 评测对比
- 同 benchmark(先 LIBERO,轻;再 RoboTwin)跑官方 eval(harness 已就绪)。
- 主指标:**下游 SR**(LMWM-VLA vs LaWM-VLA)+ action-MAE。分层报困难/OOD 子集(见注入分析的评测铁律)。

---

## 4. 决策(2026-07-12 已拍板)

| 决策 | 定案 |
|---|---|
| **对比数据集** | ✅ **先 LIBERO**(公平锚点 + harness 已验证 98.0%);kai0 作第二步 |
| **特征空间** | ✅ **DINOv3-base 为对比空间**(和 LaWM 对齐);**SigLIP 留好接口、未来再测**(adapter 做成 encoder 可切换) |
| **LaWM baseline** | ✅ **用 released** `lawam_libero_sft_release`(已下 + 已验证 libero_10 98.0%),直接作 baseline,不自训 |

**推论**:LMWM 也必须在 **LIBERO demo 数据**上训(学 milestone)→ 需下载 `jialei02/libero_merged_no_noops_20hz`(LeRobot 3.0, eval 时跳过了)。encoder 抽 DINOv3-base 特征。

---

## 5. 风险
1. **契约耦合**:LAM 可能和 action expert 深耦合(VQ code 词表、特定 latent 维度),LMWM 输出未必即插即用 → P0 决定成本。
2. **DINOv3-base 掉点**:我们主线已弃 DINOv3 走 SigLIP;强行 base 可能 intrinsic 变差,影响下游。
3. **gf3 共享**:8卡波动(曾被 GigaWorld 占满,2026-07-12 复查空闲),跑前 `nvidia-smi` 确认。
4. **公平性**:数据/配方/step 数必须严格对齐,否则对比无效(用 experiment-audit 式自查)。

## 6. 一句话
先 P0 读码钉死 LAM 契约 → P1 gf3 8卡把 LMWM 切 DINOv3-base 重训 → P2 写 adapter 塞进 starVLA LAM 槽 → P3 gf3 训 LMWM-VLA + LaWM baseline → P4 LIBERO 先对比 SR。唯一变量=世界模型,干净 A/B。
