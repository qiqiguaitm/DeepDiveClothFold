# Cross-Embodiment 战略与数据分析

> **范围**: 3 异构机器人 (KAI0 / vis / XVLA-Soft-Fold) 数据复用、conditioning 设计、跨数据集 norm-stats / operator / 混训策略实证分析、tri-track (A/C/X) 训练架构、决策点。
> **状态**: 持续更新 (此文档替代原 `cross_embodiment_data_reuse_plan.md`, 抽离了状态表与具体训练计划, 后者迁入 `docs/training/future_plans/` 与 `docs/training/history/`)。
> **最近更新**: 2026-05-25。

---

## 1. Embodiment Gap — 3 异构机器人定义

### 1.1 三机器人对比表

| 维度 | A: KAI0 (官方) | **B: vis (⭐ 部署目标)** | C: XVLA-Soft-Fold (第三方) |
|---|---|---|---|
| 机械臂型号 | piper 双臂 | piper 双臂 (同型) | piper 双臂 (待确认型号) |
| Joint DOF | 14 (7×2) | 同 | 同 (假设) |
| 控制频率 | 30 Hz | 同 | 待确认 |
| **Wrist 相机** | D435 (FOV 69°×42°, rolling, min depth 28cm) | **D405** (FOV 87°×58°, global, min depth 7cm) ⭐ | 待确认 hdf5 元数据 |
| **Wrist 安装** | 旧 flange | 新 flange (高度/角度略差) | 第三方采集, 不同 setup |
| **双臂间距** | 标准 | 略差 (毫米级) | 待确认 |
| **Top 头部相机** | D435 (76cm 高, 30° 角) | 同 (略差 <5cm/5°) | 待确认 |
| Action 语义 | "Flatten and fold the cloth." | 同 | 同 (cloth fold) |
| **Episodes** | 6,512 | **895** (B 数据量少 → 关键瓶颈) | 1,729 |
| **数据格式** | LeRobot v2.1 (14D joint) | LeRobot v2.1 (14D joint) | hdf5 (EE6D? 待确认) |
| **部署?** | ❌ 不部署 | ✅ **唯一部署目标** | ❌ 不部署 |
| domain_id (Track X) | 0 | **1** | 2 |

### 1.2 与 B 部署目标的 gap 量化

A vs B 关键 wrist gap (§2 实测):
- **R 腕 yaw+roll paired shift ~19°** (SE3 复合旋转) — 部署 B 时 wrist 视野 OOD
- L2 mean diff 0.47 rad (跨 robot effect, 剔除 operator confound 后)
- 13/14 dim 在 ±1σ 内 (大部分可被 per-dataset norm 吸收)

C (XVLA-Soft-Fold) vs B gap **待量化** (等格式适配后做 norm_stats 对比, 预期 wrist 相机差异最大)。

### 1.3 实测真机症状

1. **Cloth loop** (复杂场景): mixed_1 baseline (纯 A 训练) 部署 B 出现循环卡死 — D435→D405 视觉 OOD 累积漂移
2. **空桌面抖动**: vis SFT 后 prior 被高 jump 帧拉宽, 空桌面 condition 弱 → 抽到大 action
3. **混训抖动 > 纯 B**: 早期 naive 混训 (`mixed_pure2_1800_6000`) 真机抖 > `pure_1200_new_norm` → naive 混训创造双模式策略, chunk 间切换抖

---

## 2. 实测 Norm-stats / Operator / 混训 — 数据集对比 (2026-05-21)

### 2.1 KAI0 ↔ vis Norm-stats Δmean (joint angles, 14D)

直接从原始 parquet 重算 (kai0_base 102/3055 ep × 114k frames, vis_v2_merged 112/895 ep × 133k frames)。

| dim | label | A.mean | B.mean | Δmean (A−B) | Δ角度 | |Δ|/A.σ |
|---:|---|---:|---:|---:|---:|---:|
| 0 | L_肩 yaw | -0.062 | -0.079 | +0.017 | 1.0° | 0.09σ |
| 1 | L_肩 pit | +1.547 | +1.368 | +0.179 | 10.2° | 0.33σ |
| 2 | L_肘 | -1.301 | -1.200 | -0.100 | 5.7° | 0.22σ |
| 3 | L_腕 yaw | -0.095 | -0.159 | +0.064 | 3.7° | 0.22σ |
| 4 | L_腕 pit | +0.796 | +0.719 | +0.077 | 4.4° | 0.32σ |
| 5 | L_腕 rol | +0.031 | +0.158 | -0.127 | 7.3° | 0.45σ |
| 6 | L_grip | +0.028 | +0.029 | -0.001 | — | 0.03σ |
| 7 | R_肩 yaw | +0.115 | -0.006 | +0.121 | 6.9° | 0.71σ |
| 8 | R_肩 pit | +1.486 | +1.476 | +0.010 | 0.6° | 0.02σ |
| 9 | R_肘 | -1.461 | -1.284 | -0.177 | 10.1° | 0.33σ |
| **10** | **R_腕 yaw** ⚠️ | +0.048 | +0.341 | **-0.293** | **16.8°** | **1.05σ** |
| 11 | R_腕 pit | +0.918 | +0.899 | +0.019 | 1.1° | 0.08σ |
| **12** | **R_腕 rol** ⚠️ | +0.003 | -0.241 | **+0.244** | **14.0°** | **0.99σ** |
| 13 | R_grip | +0.035 | +0.021 | +0.013 | — | 0.41σ |

```
L1 norm: 1.44 rad   L2 norm: 0.51 rad   L∞: 0.293 rad (16.8° @ R_腕yaw)
Within ±1σ: 13/14   median 0.32σ        max 1.05σ
Per-arm L2:  Left 0.26    Right 0.44     → 右臂偏 1.67× 左臂
B/A motion range: median 1.07, mean 1.11, [0.88, 1.48]
```

### 2.2 关键发现 (4 条)

1. **整体分布高度重叠 but not identical**: 13/14 维落在 ±1σ 内, per-dataset norm 大部分可吸收。但 L2 = 0.51 rad 在 50-step chunk 上累积影响显著。
2. **右臂偏移 1.67× 大于左臂** — 双臂间距不同的直接证据。
3. **R 腕 yaw (-16.8°) + R 腕 roll (+14°) 是配对偏移** ⭐ (核心):
   - paired correlated shift, 不是独立
   - SE(3) 表示下合成 **~21° 复合旋转**
   - 物理意义: 右手 wrist 末端在 B 上比 A 整体旋转 21° → D405 wrist 视野下 cloth 出现 21° 旋转 OOD
4. **B 运动幅度比 A 大 10-30%** — 解释 "vis SFT 后 prior 被拉宽" 现象。

### 2.3 vis 内部 Operator 与时间漂移

| Group | Operator | Episodes | 占比 |
|---|---|---:|---:|
| G1 (主操作员, ztm+lym 同一人) | ztm 723 + lym 149 | 872 | 97.4% |
| G2 (助手) | gsy | 23 | 2.6% |

**G1 内时间漂移**: 同一 operator 跨 5 天 (4-24 vs 4-28) L2 drift = 0.47 rad, **与 cross-robot effect 同量级**。
**G2 影响微弱**: gsy 剔除后跨 robot L2 仅降 8.9% (0.51 → 0.47), R 腕 paired shift 仍 ~19°。结论: **不必 per-operator norm**, 真正的 cross-robot geometric effect 真实存在。

### 2.4 混训策略 6 方案对比

| 方案 | 描述 | R 腕 19° | 时间 drift | motion range | 工程量 |
|---|---|:-:|:-:|:-:|:-:|
| A. Naive joint norm | 合并算单一 norm_stats | ❌ | ❌ | ❌ | 0.5 day |
| **B. Per-dataset norm + Single model** | 每数据集 own norm, 同模型 | ⚠️ 90.7% | ⚠️ 90.7% | ✅ | 1 day |
| **C. Soft Prompt + Per-DS norm** | 显式 routing (X-VLA) | ✅ | ✅ | ✅ | 0 (代码就绪) |
| D. Curriculum (A pretrain → B finetune) | 现有 mixed_1 → smooth_800 | ✅ B finetune | ✅ | ✅ | 1 day |
| E. SSL Decoupled | A 进 visual SSL, B 进 policy | 不参与 | 不参与 | 不参与 | 9 week |
| F. EE-based action | delta EE pose | ✅ **天然消除** | ⚠️ | ⚠️ | 3 day |

**MMD 实测**: per-dataset norm 后 MMD(A_norm, B_norm) = 0.00558 vs raw 0.0597 (降 90.7%), 但仍 28× self-baseline (残余高阶矩 + joint synergy 差异不可消)。

**关键 Insight**: 早期 `mixed_pure2_1800_6000` 失败的真因可能不是"混训不能", 而是用了 naive joint norm (方案 A) 而非 per-dataset norm (方案 B)。

**推荐 Layered Combination**:
- L1 Visual: 方案 E (SSL decoupled, A+B+C all in, no action loss)
- L2 Policy: 方案 C (Soft Prompt + Per-DS norm) + 方案 D (curriculum)
- L3 Ablation: 方案 F (EE-based, paper 对照)

---

## 3. 4-层 ROI 战略框架

**判断**: 某 loss / objective 是否依赖 A 和 B 的 action space 对齐?

| Layer | 内容 | 依赖 action 对齐? | A 价值 | 工程复杂度 |
|---|---|---|---|---|
| **L1: Visual SSL / World Model** ⭐ | V-JEPA + point track + flow + dynamics | ❌ 不依赖 | **全功率可用** | 高 |
| **L2: Embodiment-cond Policy** | A+B 共训, embedding 区分 | ⚠️ 弱依赖 (需 conditioning) | 可用, 需对齐 | 中 |
| **L3: Auxiliary tasks** | Inverse dynamics, future frame pred | ⚠️ 部分依赖 | 可用, 不入主 loss | 中 |
| **L4: Data engine / Sim2Real** | Retargeting, replay-augmentation | ✅ 强依赖 | 低 (需高保真 retarget) | 高 |

> **核心原则**: A 的价值不在"直接帮 B 做 task", 而在 **representation / dynamics / prior** 这些更上游的层次。

---

## 4. 假说矩阵 (H1-H4)

| ID | 假说 | 关键实验 | 成功标准 |
|---|---|---|---|
| **H1** | SSL pretrain on A+XVLA+B 提供的 visual repr 在 cloth 任务上 > π0.5 default | E3.1 vs E3.0 | B finetune val MAE 降 ≥10%, 真机抖动减少 |
| **H2** | Multi-objective (V-JEPA + track + flow + xview) > 单 V-JEPA | E1.5 vs E1.1 | Downstream val MAE 进一步降低 |
| **H3** | Embodiment-conditioned dynamics 让 A 的物理 prior 不污染 B policy | E3.3 vs E3.2 | 真机平滑度 + 复杂场景成功率提升 |
| **H4** | Motion-residual: cloth_residual 部分 embodiment-invariant | E2.3 latent 分析 | A vs B cloth_residual latent MMD < 0.1 |

---

## 5. Action 表示与 Conditioning 选项

### 5.1 Action 表示决策

| 模型 | 默认 Action |
|---|---|
| π0 (老) | Delta (relative to chunk start) |
| **π0.5** | **Absolute** (默认), 可选 relative |
| OpenPI Aloha | DeltaActions transform (`use_delta_joint_actions=True`) |
| 本地 `mixed_1` ckpt | **Absolute** (norm_stats 实测: mean[1]=1.48, std=0.63) |
| KAI0 raw 数据 | **Absolute** (joint angles ±π) |
| X-VLA | EE6D absolute pose (20D = xyz+Rot6D+grip per arm) |

**实证 ([Demystifying Action Space Design 2602.23408](https://arxiv.org/abs/2602.23408))**: 单机器人/单任务/long-horizon → absolute 更稳; 多 embodiment/跨设备 → delta 更稳; 混合 mask (joint delta + gripper absolute) 是 pragmatic 选择。

### 5.2 Embodiment Conditioning 选项

| 方式 | 实现复杂度 | 本地代码状态 | 推荐度 |
|---|---|---|---|
| Hard prompt (`"[D405 wrist] ..."`) | 极低 (0 改 model) | ✅ 任意 config 可用 | ⭐⭐ 弱 (信号沿 LLM attention 隐式传播) |
| **Soft prompt (X-VLA 官方原生)** | 低 (Track X 走官方 `SoftPromptedTransformer`) | ✅ `lerobot/xvla-base` ckpt | ⭐⭐⭐⭐ **主线 (Track X)** |
| **Action Head Embedding** (Track C 方案 A) | 极低 (action expert input concat 1 token) | ✅ 已实现 (`pi0.py:action_head_cond_hub`) | ⭐⭐⭐ paper 对照 |

### 5.3 Action Head Cond (Track C, 方案 A) — 设计要点

**动机**: Soft Prompt 在 **VLM 输入端**注入 domain (24 层 PaliGemma + cross-attn); 方案 A 在 **action expert 输入端**直接 concat domain token (paligemma 完全不知 domain)。形成 "VLM 端 vs Action expert 端" 1:1 对照。

```
Soft Prompt (Track X 官方):
  d → SoftPromptedTransformer.soft_prompts[d] (B,32,1024)
    → 拼到 Florence2-VLM input → 24 层 SoftPromptedTransformer → DomainAwareLinear → action

方案 A (Track C):
  d → action_head_cond_hub[d] (B,1,1024) → 拼到 action expert input (与 noise_action_token 同级)
    → action expert self-attn (4-8 层) → action  [paligemma 完全不知 domain]
```

**关键差异**: Soft Prompt 控制 *VLM 如何看世界* (domain-specific perception); 方案 A 控制 *action expert 如何 denoise* (domain-specific motor)。互不竞争, paper E3.7 vs E3.8 验证 "perception vs motor" 注入点选择。

**为什么单阶段而非 curriculum**: Soft Prompt 信号路径 24 层 (改 image 编码), 需 stage 2 freeze-backbone 保护; 方案 A 只影响 4-8 层 action expert (不改 image 编码), stage 2 价值边际低 → 单阶段 joint balanced (vis ×7) 50k step。

**代码改造点**:
- `pi0_config.py`: 加 `action_head_cond_num_domains: int = 0`
- `pi0.py.__init__`: 加 `self.action_head_cond_hub = nnx.Embed(num_domains, action_expert_width)` (init N(0, 0.02))
- `pi0.py` action expert forward: `domain_token = self.action_head_cond_hub(obs.dataset_id)[:, None, :]; action_input = jnp.concat([domain_token, action_input], axis=1)`
- `transforms.py`: dataset_id 透传 ✓ (已修)

---

## 6. RTC / TAC — Action Chunking 实时性方案对比 (2026-05-22)

### 6.1 三论文核心对比

| 论文 | 时间 | 路线 | 改 base 模型? | 推理 latency | 真机验证 |
|---|---|---|:-:|:-:|:-:|
| **Inference RTC** ([2506.07339](https://arxiv.org/abs/2506.07339)) | 2025-06 | 推理时 inpainting + vjp guidance | ❌ | **+28%** (97 vs 76 ms) | ✅ π0.5 6 task × 480 ep |
| **TAC** ⭐ ([2512.05964](https://arxiv.org/abs/2512.05964)) | 2025-12 | **训练时** prefix actions 作 ground-truth | ❌ (改 loss + adaLN per-token) | **0** (与 baseline 持平) | ✅ π0.6 box/espresso |
| **A2C2** ([2509.23224](https://arxiv.org/abs/2509.23224)) | 2025-09 | lightweight correction head, 每步 Δa | ❌ (base frozen, +新 module) | +4.7ms (~6%) | ❌ 仅 sim |

**三者正交可叠加**, 各解决不同子问题:
- Inference RTC = pseudo-inverse 强行约束 (老 ckpt 补救)
- **TAC** = 模型自己学会 chunk overlap (训练时一次, 推理零开销)
- A2C2 = 实时反应模块 (cloth dynamic state 时强相关)

### 6.2 本地实现状态

| 项 | 文件 | 状态 |
|---|---|---|
| **Inference RTC** | `kai0/src/openpi/models/pi0_rtc.py` | ✅ 完整 1:1 复刻 (`get_prefix_weights` 4 schedules, `jax.vjp` guidance, `min(c·inv_r2, max_gw)`) |
| **TAC training** | — | ✅ 已集成 (`pi0_rtc.py::compute_loss`, `Pi0Config.tac_enabled`, adaLN per-token, 0 新参数) — vis_v2_full v4 16gpu (hdv82) 已成功 running |
| **A2C2 correction head** | — | ❌ 未实现 (搁置) |

### 6.3 TAC Algorithm (Algorithm 1, JAX)

```python
def compute_loss(rng, obs, actions, *, max_delay=10):
    b, ah, ad = actions.shape
    noise_rng, time_rng, delay_rng = jax.random.split(rng, 3)
    time  = jax.random.uniform(time_rng, (b,))
    noise = jax.random.normal(noise_rng, (b, ah, ad))

    # TAC 新增 4 行:
    delay        = jax.random.randint(delay_rng, (b,), 0, max_delay)
    prefix_mask  = jnp.arange(ah)[None, :] < delay[:, None]
    time         = jnp.where(prefix_mask, 1.0, time[:, None])   # per-token time
    postfix_mask = jnp.logical_not(prefix_mask)[:, :, None]

    x_t = time[:, :, None] * actions + (1 - time[:, :, None]) * noise
    v_t = model(obs, x_t, time)
    loss = (v_t - (noise - actions)) ** 2
    return jnp.sum(loss * postfix_mask) / (jnp.sum(postfix_mask) + 1e-8)
```

**Architecture**: `Pi0Config.tac_enabled: bool = False` + `tac_max_delay: int = 10`; **adaLN-zero conditioning 改 per-token** (scale/shift/gate 在 sequence 维允许差异); 0 新参数。

### 6.4 训练超参 (论文披露)

| Setting | π0.6 论文值 | 本地 Cloth Task |
|---|---|---|
| Fine-tune steps | 8000 | 同 (~12h on 16 H20) |
| Batch size | 512 | 128-256 (GPU 较少) |
| Delay sampling | uniform [0, 10] | uniform [0, 6] (30Hz × 200ms = d=6) |
| Inference denoising steps | 5 | 同 |
| Init | π0.6 base | pi05_base 或 mixed_1 |

### 6.5 复现难易度

| 维度 | 评分 | 说明 |
|---|:-:|---|
| 算法清晰度 | ⭐⭐⭐⭐⭐ | Algorithm 1 完整 JAX 代码 (论文附录) |
| 代码开源 (full repo) | ❌ | 仅论文 Algorithm 1, 无 GitHub |
| 模型 ckpt 可用 (π0.6) | ❌ | 闭源 |
| 超参完整 | ⭐⭐⭐⭐⭐ | 训练 step / batch / delay 全披露 |
| 架构改动复杂度 | ⭐⭐⭐⭐⭐ | adaLN per-token, 0 新参数 |
| 对 π0.5 可移植性 | ⭐⭐⭐⭐⭐ | adaLN 同架构 |

**总体可复现性**: 不依赖 π0.6 ckpt, 可在 π0.5 + 自有 cloth 数据上复现。

---

## 7. Tri-Track 训练架构 (M2 主线)

> **2026-05-22 战略转向**: 放弃 pi0.5 + 移植 soft prompt (Track B 已废弃), 改用 X-VLA 官方完整架构 (Track X) + Action Head Cond (Track C) + SSL Pretrain (Track A)。

```
┌─────────────────────────────────────┐
│   Track A: SSL Visual Pretrain      │
│   (uc02 + Robot-North-H20)          │
│   ├── Phase 0 Pseudo-labels         │
│   ├── Phase 1 V-JEPA + track + flow │
│   ├── Phase 2 Dynamics + Embodiment │
│   └── Phase 3 Policy + Ablation     │
├─────────────────────────────────────┤
│   Track C: Action Head Cond Emb     │
│   (Action expert 端 cond, paper     │
│    ablation 对照, 16 GPU)           │
│   └── 单阶段 balanced joint training│
├─────────────────────────────────────┤
│   Track X: X-VLA 官方架构 ⭐ 主线    │
│   (Florence2 + SoftPromptedXformer, │
│    EE6D 20D action, 16 GPU)         │
│   ├── X3.A: 3-domain (A+B+C)        │
│   └── X3.B: 2-domain (A+B, no XVLA) │
└─────────────────┬───────────────────┘
                  ↓
       ┌──── Final Merge ────┐
       │ SSL Backbone +      │
       │ X-VLA 官方 ckpt +   │
       │ Dynamics-cond head  │
       └─────────────────────┘
```

**资源分配**:
- Track A: uc02 (Phase 0) + Robot-North-H20 (Phase 1-3)
- Track C: cn-shanghai / cn-beijing 跑 paper ablation 对照
- Track X (主线): uc01+uc02 16 A800 — X3.A / X3.B 顺序执行

**Milestone**:
- **M1 (1-2 周)**: 短期真机修复 — 已 deprioritize
- **M2 (~9 周)**: Multi-track 训练 ⭐ 主线 (本表全部)
- **M3 (M2 之后)**: Merge SSL + X-VLA 官方 ckpt + Dynamics-cond head + 真机大规模 (60-100 ep/abl) + Paper
- **M4 (long-tail)**: ATOM Policy 扩展 (跨任务: Task B 检索, Task C 挂衣)

**具体 plan/状态**: 详见 `docs/training/future_plans/plans/ssl_phase_pretrain_pipeline.md` (Track A) + `xvla_track_x_curriculum.md` (Track X) + `pytorch_native_vis_v2_full.md` (R1/R2)。

---

## 8. 风险预警 + 关键陷阱

| # | 风险 | 应对 |
|---|---|---|
| 1 | CoTracker3 在 heavy occlusion (crumpled cloth) 失败 | Pseudo-track 加 confidence filter; track loss 按 mask 加权 |
| 2 | RAFT 在 fast motion 失败 | Quasi-static 阶段训 flow, dynamic 阶段降权重 |
| 3 | **D435 FOV (69°) < D405 (87°)** Wrist sensor gap | ❌ ~~输入端 D405 crop 到 D435 FOV~~ (不可持续, 训练-推理双向维护, 跨相机不通用, 丢失 D405 周边信息). ✅ **representation-level invariance**: (a) view-conditioned token (data loader 标 `view_id`, E1.4/E1.5 xview head 自学); (b) RandomResizedCrop augmentation (scale 0.6-1.0) |
| 4 | π0.5 PaliGemma backbone continual SSL 易 catastrophic forget | Layer-wise lr decay, peak 5e-5, anchor loss on 1% LAION subset |
| 5 | 叠衣 success criterion 真机评估难自动化 | 设计 IoU / fold count / stage completion 离线 metric |
| 6 | IK 在 delta EE 推理时不连续 | Warm-start with current joints, 或训练同时输出 delta EE + delta joints |
| 7 | EE-relative 丢失绝对工作空间位置信息 | 加 base→top_camera frame 的 anchor token (从 hand-eye calibration 得来) |
| 8 | Multi-objective loss 不收敛 | Phase 1 先单 V-JEPA 5k step 预热, 再逐项加 |
| 9 | Embodiment cond 在 visual 还是 dynamics? | Phase 1 visual 不区分 view 来源 (要 invariant); Phase 2 dynamics 才区分 (要 partition) |
| 10 | Phase 0 (CoTracker) 慢 | Temporal stride 3 + batch size 优化 + 8-GPU 并行, 实测 ~17h |

---

## 9. 决策点

### 决策点 1: 是否引入 dagger?
- L1 (SSL): ✅ 引入 (3457 ep 增加 vision diversity)
- L2 (policy): ❌ 不引入 (抖动 +62%, 污染 action prior)
- L3 (aux): ⚠️ 可选 (作为 inverse dynamics 目标)

### 决策点 2: Embodiment conditioning 实现方式? ✅ (2026-05-22 二次更新)
- ~~Hard prompt only~~ (信号沿 LLM attention 隐式传播, 不显式 gate)
- ✅ **Soft Prompt (X-VLA 官方原生)** — Track X, VLM input 端
- ⭐ **Action Head Cond Token (方案 A)** — Track C, action expert input 端
- ~~方案 B (FiLM) / C (adaLN) / D (Cross-attn)~~ 暂搁置
- ~~终态 E3.9 双端 Soft + Action Cond~~ 暂搁置 — 待单端结果再决定
- 真机评估: 全部 Track C/X 终态在 vis (B 真机) 测试

### 决策点 3: EE-relative action 是否启用? ❌ 已 deprioritize (2026-05-22)
- ~~Phase 0 E0.5 EE-relative preprocessing~~ 取消
- ~~Phase 3 E3.4 / E3.8 delta EE~~ 取消
- 理由: R 腕 21° paired shift 由 Soft Prompt + Action Head Cond 处理, 工程量更低 + 无 IK 不连续风险
- 保留作为远期 backup

### 决策点 4: 是否回看 M1 短期方案?
- 触发条件: Phase 1 SSL + Track C / Track X 首轮 ablation 完成, 如 E3.1 / E3.7 / E3.8 已超过 baseline → M1 不需要
- 否则: 回看 B oversample 修复抖动 (EE-relative 不再回看)

### 决策点 5: 采纳 TAC 作为 Phase 3 ablation 新维度 ✅
- 采纳 TAC (零参数, 几乎零成本, 7-13% improvement 已 paper 验证)
- A2C2 暂搁置 — 等 TAC 结果再决定 dynamic obs response 是否必要
- 保留 Inference RTC (`pi0_rtc.py` 已实现) — 老 ckpt 部署还能用

---

## 10. 相关文档

- **Plan 详情**: `docs/training/future_plans/plans/` (`ssl_phase_pretrain_pipeline.md`, `xvla_track_x_curriculum.md`, `pytorch_native_vis_v2_full.md`)
- **Conditioning 跟踪**: `docs/training/history/experiments/xvla_conditioning_methods_results.md`, `conditioning_vs_action_representation_ablation.md`
- **实时推理**: `docs/deployment/realtime_vla_optimization_analysis.md`, `docs/deployment/rtc_implementation.md`
- **数据集诊断**: `docs/training/history/experiments/dataset_diagnostic_report.md`
- **Norm-stats ablation 实测**: `docs/training/history/experiments/norm_stats_ablation_apr28_450.md`
