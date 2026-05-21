# SSL Pretraining Experiment Plan — Cross-Embodiment Cloth Folding

> **Created**: 2026-05-21
> **Status**: Phase 0 ready to start
> **Owner**: Tim
> **Related**: [cross_embodiment_data_reuse_plan.md](./cross_embodiment_data_reuse_plan.md), [training_servers_knowledge_base.md](./training_servers_knowledge_base.md)

---

## 1. 背景 + 动机

### 1.1 问题
我们有 4 个核心痛点:
1. **B (vis_base D405 + 新双臂) 数据量小** (~800 ep), 真机表现不够稳
2. **A (kai0_base + dagger, D435 + 老双臂) 数据多** (6512 ep), 但 naive 混训 → 真机抖动 (`mixed_pure2_1800_6000` 验证)
3. **XVLA-Soft-Fold** (1729 ep), 另一套软折叠分布, 没法直接用
4. **cross-embodiment gap** (硬件清单见 §1.2) 无法在 supervised policy 层 absorb

### 1.2 Embodiment Gap (按影响从大到小)

| Gap 类型 | A | B | 影响 |
|---|---|---|---|
| **Wrist 相机** | D435 (FOV 69×42°, rolling, min depth 28cm) | D405 (FOV 87×58°, global, min depth 7cm) | 🔴 **最硬** — 近距 RGB-D 行为完全不同 |
| **Wrist extrinsics** | 老 flange | 新 flange | 🟠 第二严重 — wrist view 物体 pos/scale 受影响 |
| **双臂间距** | 老 setup | 新 setup | 🟡 absolute EE 时 systematic bias |
| **Top 相机** | 老高度/角度 | 新高度/角度 | 🟢 < 5cm/5° 时 augmentation cover |

### 1.3 核心 Insight (定义路线)

> **A 的数据价值不在 "直接帮 B 做 task", 而在 representation / dynamics / prior 这些更上游的层次。**

按 ROI 分四层 (越高越值得做):
| Layer | 描述 | 是否依赖 action 对齐? | A 数据可用程度 |
|---|---|---|---|
| **L1: SSL Visual Pretrain** ⭐ | V-JEPA / track / flow / xview | 否 | **Full power** |
| L2: Embodiment-cond Policy | Co-train with embodiment token | 部分 (need conditioning) | 谨慎 |
| L3: Auxiliary tasks | Inverse dyn, future frame pred | 部分 | 谨慎 |
| L4: Data engine | Replay/retarget, sim2real | 否 | Full power 但需 retargeting |

**本文档专注 L1 (+ L2 dynamics, + L3 policy) 端到端实验。**

---

## 2. 假设 + 验证

| ID | 假说 | 关键实验 | 成功标准 |
|---|---|---|---|
| **H1** | SSL pretrain on A+XVLA+B 提供的 visual repr 在 cloth 任务上 > π0.5 default | E3.1 vs E3.0 | B finetune val MAE 降低 ≥10%, 真机抖动减少 |
| **H2** | Multi-objective (V-JEPA + track + flow + xview) > 单 V-JEPA | E1.5 vs E1.1 | Downstream val MAE 进一步降低 |
| **H3** | Embodiment-conditioned dynamics 让 A 的物理 prior 不污染 B policy | E3.3 vs E3.2 | 真机平滑度 + 复杂场景成功率 提升 |
| **H4** | Motion-residual decomposition: cloth_residual 部分 embodiment-invariant | E2.3 latent 分析 | A vs B cloth_residual latent 的 MMD < 0.1 |

---

## 3. 数据 + 资源

### 3.1 训练数据池 (2026-05-21 实测路径)

| 来源 | Episodes | 总帧数 | Avg/ep | 视频路径 | Size |
|---|---:|---:|---:|---|---:|
| **A: Kai0 base** | 3055 | 3.36M | 1101 | `/data/shared/ubuntu/workspace/dataset/Kai0_official/Task_A/base/videos/` | 46G |
| **A: Kai0 dagger** | 3457 | 2.42M | 699 | `/data/shared/ubuntu/workspace/dataset/Kai0_official/Task_A/dagger/videos/` | 39G |
| **B: vis_v2_merged** | 895 | 1.06M | 1188 | `/data/shared/ubuntu/workspace/dataset/Task_A/vis_v2_merged/videos/` | 6.3G |
| **XVLA-Soft-Fold** | 1729 | ~? | — | 见 §3.1.x XVLA 多地副本表 | 444G |
| **合计** | **9136 ep** | **~7M frames** | — | 3 views (top_head, hand_left, hand_right) per ep | **~535G** |

#### 3.1.x XVLA-Soft-Fold 多地副本 (2026-05-21)

| 服务器 | 路径 | 用途 | 状态 |
|---|---|---|---|
| **uc02 本地** | `/data/tim/datasets/xvla_soft_fold/` | 原始下载位置 | ✅ 完整 (1729 files, 444G) |
| **uc01/02/03 NFS** | `/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/xvla_soft_fold/` | uc 集群训练用 (走 NFS 到 uc01 disk) | ✅ 完整 (1729 files, 444G) |
| **gf0 vePFS-cnsh** | `/vePFS/tim/xvla/data/xvla_soft_fold/` | robot-task (cn-shanghai) volc job 共享 | 🔄 下载中 (gf0 ← hf-mirror, ~7h ETA) |
| **gf3 vePFS-cnbj** | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/xvla/data/xvla_soft_fold/` | **Robot-North-H20** (cn-beijing) 集群 job 共享 ⭐ | 🔄 下载中 (gf3 ← hf-mirror, ~8h ETA) |

> gf3 副本用于 **Phase 1 SSL pretrain on Robot-North-H20** — 集群 job 挂载 vePFS-cnbj 后 XVLA 自动可见。

**视角统一命名** (LeRobot v2.1 convention, 也用于 SSL data loader):
- `observation.images.top_head` — top 相机 (A 全是 D435, B 用 D435)
- `observation.images.hand_left` — 左 wrist (A 用 D435, B 用 D405) ⚠️ embodiment gap
- `observation.images.hand_right` — 右 wrist (同上)

> 注: 之前文档草稿引用 `vis_base_clean_v2`, 但实测该路径只剩 metadata (2.8M). 真实 B 数据在 **`vis_v2_merged`** (895 ep, 6.3G)。

### 3.2 GPU 资源 (2026-05-21 实测可用)

| 资源 | GPU | 状态 | 分配 |
|---|---:|---|---|
| **Robot-North-H20** (cn-beijing) | 39 H20 free / 56 total | active | SSL 主战场 (单 exp 16 GPU, 可 2-3 并发) |
| **uc02** | 8 A800 | free (XVLA dl 完成) | 数据预处理 + dev / smoke |
| **gf3** | 1 H20 | active (smoke 用) | 单卡 smoke / debug |
| uc01, uc03 | busy | 其他 exp 占用 | 不动 |

### 3.3 预算

- SSL pretrain (Phase 1): ~80 GPU-day
- Dynamics (Phase 2): ~30 GPU-day
- Policy + ablation (Phase 3): ~30 GPU-day
- **总计**: ~140 GPU-day on Robot-North-H20

---

## 4. 实验阶段 (4 个 Phase, 14 个实验)

### Phase 0: 数据预处理 + 伪标签生成 (Week 1-2)

**目标**: 把 9136 ep 视频 (7M frames × 3 view = 21M frame-views) 跑过 CoTracker3 / RAFT / SAM, 生成离线 pseudo-labels 给 Phase 1 SSL 用。

> **资源分配 (并行)**: uc02 跑 CoTracker + flow (8 A800 80GB), Robot-North-H20 1 节点 (8 H20) 跑 SAM。

**计算预估** (基于 21M frame-views 总规模):

| Exp | Tool | 输入 (有效) | 输出 | Resource | ETA (实测调整) |
|---|---|---|---|---|---|
| **E0.1** Pseudo-track | CoTracker3 (v3.0 windowed) | T=24 window, stride=12 → ~580k windows × 3 view | `tracks/{ep_id}/{view}.npz` shape (T=24, N=32, 3) | uc02 8 A800 | ~3-5 day |
| **E0.2** Optical flow | RAFT-Large | adjacent frame pair, **temporal stride 3** (减 3×) → ~2.3M pairs × 3 view | `flow/{ep_id}/{view}.npz` (H/8, W/8, 2) per pair | uc02 4 GPU 并行 | ~2-3 day |
| **E0.3** Cloth mask | SAM2 (Hiera-L) | 1 frame/sec (~30 frame/ep) × 9136 ep × 3 view = ~820k mask | `mask/{ep_id}/{view}.npz` (T_sub, H, W, K) | Robot-North-H20 8 H20 | ~1 day |
| **E0.4** FOV align | OpenCV (offline) | D405 wrist (B 数据 895 ep × 双 wrist) | `rgb_d405_d435align/{ep_id}/wrist_*.npy` | CPU 多核 | 几小时 |
| **E0.5** EE-relative action | Python (PiperFK + DH) | A + B + XVLA actions | `action_ee_relative/{ep_id}.npz` shape (T, 14) | CPU | 几小时 |

**优化策略** (压缩 Phase 0 时间):
1. **Temporal stride 3**: 7M → 2.3M effective frames, 不损失主信号
2. **CoTracker windowed**: T=24 window stride=12, 不一次跑完整 ep
3. **SAM 稀疏化**: 1 frame/sec 足够 (cloth segmentation 时变化慢)
4. **Multi-process per GPU**: uc02 8 A800 80G 可跑 4 instance/GPU (每个 ~5G VRAM for CoTracker)

**质量检查**: 跑完抽样 50 ep 人工 inspect tracks/flow/mask 是否合理; 不合格的 ep 记入 `skip_list.txt`。

**Phase 0 状态**: ⏳ 待启动

---

### Phase 1: SSL Visual Encoder Pretrain (Week 2-4)

**核心**: 从 π0.5 SigLIP/PaliGemma vision tower **continual-pretrain** (不 from scratch), 输出 cloth-fold-specific encoder。

> **并行调度 (per user 决策)**: 跑 3 jobs in parallel — E1.1 (baseline) / E1.4 (xview) / E1.5 (full multi-objective)。Skip E1.2/E1.3 中间点 (从 E1.5 ablation 反推单项贡献)。

#### E1.1 — V-JEPA Baseline (单目标)
```yaml
backbone:     π0.5 SigLIP (continual-pretrain)
input:        T=16 frames, 3 views, 224×224
objective:    masked latent prediction (V-JEPA 2.1)
mask:         tube 30%, edge-saliency 2× boost on cloth edges (用 E0.3 mask)
lr:           5e-5 layer-wise decay (peak), warmup 2k, cosine to 5e-7
batch:        128 total (16 H20 × 8/gpu)
steps:        50k
data:         A + XVLA + B 全量 (~9000 ep)
embodiment:   不区分 (统一 backbone)
output_path:  /vePFS-North-E/vis_robot/.../ssl_ckpts/E1.1_vjepa_base/
```

#### E1.4 — V-JEPA + Track + Flow + Cross-view (no Phase-2 weight switch)
```yaml
继承 E1.1, 加 3 个 head:
  - track_head: 8M param, predict 32 keypoint tracks over T=16
                loss = L2(xy) + BCE(visibility), w=0.5
  - flow_head:  predict dense flow from latent
                loss = EPE vs RAFT pseudo, masked by cloth mask, w=0.3
  - xview_head: top latent → wrist latent (autoregressive)
                loss = cosine + L2, w=0.2
weights:      固定 (w_vjepa=1.0, w_track=0.5, w_flow=0.3, w_xview=0.2)
```

#### E1.5 — Full Multi-objective + Phase Weights + Saliency + Multi-scale
```yaml
继承 E1.4, 进一步:
  - Phase 1 (step 0-25k):  w_vjepa=1.0, w_track=0.5, w_flow=0.3, w_xview=0.2
  - Phase 2 (step 25k-50k): w_vjepa=0.5, w_track=1.0, w_flow=0.5, w_xview=0.3
  - Saliency mask: edges 2×, interior 0.5×
  - Multi-scale temporal: 一半 batch T=8 (short), 一半 T=48 (long)
  - Anchor loss: 1% batch on LAION subset (防 catastrophic forget)
```

**Phase 1 验收 (downstream micro-eval)**:
- 每个 E1.x 跑完 → 小规模 B-only policy finetune: 3k step, batch 32, 8 GPU on uc02
- 比较 val MAE on B val set
- E1.5 应该明显胜 E1.1 (h2 验证)

**Phase 1 状态**: ⏳ 待 Phase 0 完成

---

### Phase 2: Dynamics Pretrain (Week 5-6)

**核心**: 在 frozen visual encoder 上训 latent dynamics model, 引入 embodiment conditioning + motion-residual decomposition。

#### E2.1 — Latent Dynamics Baseline (无 embodiment)
```yaml
backbone:     E1.5 visual encoder (frozen)
dynamics_net: 6-layer Transformer (512 dim, 50M param)
input:        (z_t, action_t) → predict z_{t+1}
loss:         L2 on latent
data:         A + XVLA + B 全量
embodiment:   不区分 (baseline)
batch:        256, lr 5e-5 cosine 30k step
```

#### E2.2 — + Embodiment Conditioning
```diff
继承 E2.1:
- input: (z_t, action_t)
+ input: (z_t, embodiment_emb, action_t)
+ embedding_emb: 3 个 learnable vec (A_emb, XVLA_emb, B_emb), dim=128
inference: 部署 B 时只用 B_emb
```

#### E2.3 — + Motion-Residual Decomposition ⭐ (paper 原创)
```yaml
分两个 head:
  ego_motion_head:    (action_t, emb_e) → ego_motion_latent
                      (受 embodiment 影响, A/B 各自学)
  cloth_residual_head: (z_t, action_t)   → cloth_residual_latent
                      (NOT 依赖 embodiment, 共享物理 prior)
update: z_{t+1} = z_t + ego_motion_latent + cloth_residual_latent

paper claim: cloth_residual 部分在 A vs B 上分布相同 (用 MMD < 0.1 验证)
```

#### E2.4 — + Inverse Dynamics Auxiliary
```diff
+ Aux head: predict a_t | (z_t, z_{t+1}, emb_e)
  loss: L2 on action, weight 0.2
  作用: 给 visual + dynamics backbone 提供 action-grounded gradient
```

**Phase 2 验收**:
- E2.3 latent 上做 t-SNE / MMD: A vs B 在 cloth_residual 部分应近, 在 ego_motion 部分应远
- 如 motion-residual 分不开 → 回退 E2.2 全联合

**Phase 2 状态**: ⏳ 待 Phase 1 完成

---

### Phase 3: Downstream Policy + Ablation (Week 7-8)

**核心**: 把 Phase 1+2 产出接到 π0.5 policy, B-only finetune, 真机评估。

| Exp | Visual | Dynamics | Action | Data | 用途 |
|---|---|---|---|---|---|
| **E3.0** baseline | π0.5 default | × | absolute | B (smooth_800) | 当前 SOTA, baseline |
| **E3.1** | E1.5 frozen | × | absolute | B | 测 H1 (visual repr 单独 value) |
| **E3.2** | E1.5 LoRA | × | absolute | B | 测 fine-tunable 是否更好 |
| **E3.3** | E1.5 LoRA | E2.3 frozen | absolute | B | 测 H3 (dynamics 额外贡献) |
| **E3.4** Full Stack | E1.5 LoRA | E2.3 frozen | EE-relative | B + A weighted (embodiment cond) | 终态最强 |

**Phase 3 训练设置**:
- 16 H20 × 50k step ≈ 35h/exp
- batch 128, lr 1.5e-5 → 1.5e-6, num-workers 64 (uc cluster style)
- EMA 0.999

**真机测试 protocol**:
- 30 episode per exp, 固定场景 + 3 OOD 场景 (布料/姿态/光线)
- 指标: 抓衣角成功率, 完整折叠成功率, 平均执行时长, 抖动 metric (action diff p99)

**Phase 3 状态**: ⏳ 待 Phase 1+2 完成

---

### Phase 4: Real Machine Evaluation + Paper Tables (Week 9)

- 真机大规模测试 (60-100 episodes/exp)
- Ablation table (见 §6)
- Failure case analysis
- Paper figures (architecture diagram, latent t-SNE, ablation curve)

---

## 5. 时间线 (Gantt)

```
Week 1   ┌────[Phase 0] 数据预处理 (uc02 + Robot-North-H20 并行)
         │       ↓ Phase 0 完成
Week 2-4 │   ┌──[Phase 1] SSL (3 并发: E1.1 + E1.4 + E1.5) on Robot-North-H20
         │   │
Week 5-6 │   │  ┌──[Phase 2] Dynamics (4 串行: E2.1→E2.2→E2.3→E2.4)
         │   │  │
Week 7-8 │   │  │  ┌──[Phase 3] Policy + Ablation (5 jobs)
         │   │  │  │
Week 9   │   │  │  │  ┌──[Phase 4] 真机 + Paper
         └───┴──┴──┴──┘
```

**总 9 周** (并行可压缩到 7-8 周)。

---

## 6. Final Ablation Table (核心 paper 表)

| Variant | Visual | Dynamics | Embodiment cond | Motion-residual | Inverse Dyn | Val MAE | 真机平滑度 | 真机成功率 |
|---|---|---|---|---|---|---:|---:|---:|
| **E3.0** baseline (π0.5 default) | — | — | — | — | — | TBD | TBD | TBD |
| **E3.1** + Visual SSL | E1.5 frozen | — | — | — | — | ? | ? | ? |
| **E3.2** + LoRA tune | E1.5 LoRA | — | — | — | — | ? | ? | ? |
| **E3.3** + Dynamics | E1.5 LoRA | E2.3 | ✓ | ✓ | — | ? | ? | ? |
| **E3.4** Full Stack | E1.5 LoRA | E2.3 | ✓ | ✓ | ✓ | ? | ? | ? |

(待填)

---

## 7. 关键陷阱 + 应对

| 陷阱 | 应对 |
|---|---|
| CoTracker 在 heavy occlusion 失败 | Pseudo-track 加 confidence filter; track loss 按 mask 加权 |
| **D435 FOV (69°) < D405 (87°)** | **把 D405 crop 到 D435 等效 FOV** (而不是反向 — 原方案有 bug) |
| π0.5 SigLIP catastrophic forget | Layer-wise lr decay + anchor loss on 1% LAION batch |
| Multi-objective loss 不收敛 | Phase 1 先单 V-JEPA 5k step 预热, 再逐项加 |
| Phase 0 (CoTracker) 慢 | 9000 ep × 3 view × T=200 frames ≈ 5.4M tracker calls. 用 batch=64 + temporal stride 2 减半 |
| Embodiment cond 在 visual 还是 dynamics? | **Phase 1 visual 不区分 view 来源**, Phase 2 dynamics 才区分。原因: visual repr 想要 invariant, dynamics 必须 partition |

---

## 8. 状态跟踪 (持续更新)

> 每次推进一步, 更新这一节。

### 8.1 Phase 0 — 数据预处理 🔄 in_progress

**启动**: 2026-05-21

| Sub-task | 状态 | 启动 | 完成 | 备注 |
|---|---|---|---|---|
| **环境安装** (uc02 kai0 venv) | ✅ done | 2026-05-21 06:05 | 2026-05-21 06:14 | cotracker3 (local git clone + uv pip -e), decord 0.6.0, einops 0.8.1, opencv 4.11, pyarrow 20.0. CoTracker3 ckpt 从 hf-mirror 下载 (96MB) |
| **真实视频 timing 实测** | ✅ done | 2026-05-21 06:19 | 2 ep 123s = **60s/ep × 3 view** | 8 GPU 并行预估总 ~17h (远好于初版 5-day 估算) |
| **E0.1 Kai0_base** (3055 ep) | 🔄 running | 2026-05-21 06:20 | — | uc02 8 GPU 并行, 每 GPU 382 ep, ETA ~6.4h. 输出 `/data/shared/.../ssl_phase0/tracks/kai0_base/` |
| E0.1 Kai0_dagger (3457 ep) | 待启动 | — | — | 等 Kai0_base 完成或如有空闲 GPU 即启 |
| E0.1 vis_v2_merged (895 ep) | 待启动 | — | — | 同上 |
| E0.1 XVLA-Soft-Fold (1729 ep) | 待启动 | — | — | XVLA 用 hdf5 格式, 需要不同的 dataset adapter |
| E0.2 RAFT optical flow | 待启动 | — | — | 待 E0.1 完成, 复用同一 uc02 GPU |
| E0.3 SAM2 cloth mask | 待启动 | — | — | 待环境装好, Robot-North-H20 1 节点跑 |
| E0.4 FOV alignment script | 待启动 | — | — | 仅 B 数据 (D405 wrist), CPU 任务 |
| E0.5 EE-relative action conversion | 待启动 | — | — | 全数据, CPU + PiperFK |
| **Phase 0 整体** | 🔄 in_progress | 2026-05-21 | — | 修正 ETA: ~5-7 day (CoTracker3 17h + flow 20h + SAM 12h + 其他) |

#### Phase 0 输出位置 (统一约定)
```
/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/ssl_phase0/
├── tracks/
│   ├── kai0_base/ep_XXXXXX/{top_head,hand_left,hand_right}.npz
│   ├── kai0_dagger/...
│   ├── vis_v2_merged/...
│   └── xvla_soft_fold/...
├── flow/    (同结构)
├── masks/   (同结构, 稀疏 1/sec)
├── rgb_d405_d435align/ (仅 B 数据)
├── action_ee_relative/{ep_XXXXXX}.npz
└── logs/    (每 GPU shard 一个 log)
```

#### Phase 0 设计要点 (实测调整后)
- **CoTracker3 模型**: scaled_offline.pth (96MB), 480×640 原始视频, 760ms/window on A800
- **Windowing**: T=24 stride=12 → 149 window per 1800-frame ep
- **Query points**: grid_size=6 → N=36 per window
- **Sharding**: 8 GPU 各处理 1/8 数据, 断点续传 (.npz 已存在 + size > 100 byte 即 skip)
- **RAFT**: temporal stride 3
- **SAM2**: 1 frame/sec sampling
- **存储**: ~1MB/ep × 9136 × 3 view ≈ 27GB 总

#### Phase 0 输出位置 (统一约定)
```
/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/ssl_phase0/
├── tracks/
│   ├── kai0_base/ep_XXXXXX/{top_head,hand_left,hand_right}.npz
│   ├── kai0_dagger/...
│   ├── vis_v2_merged/...
│   └── xvla_soft_fold/...
├── flow/    (同结构)
├── masks/   (同结构, 稀疏 1/sec)
├── rgb_d405_d435align/ (仅 B 数据)
└── action_ee_relative/{ep_XXXXXX}.npz
```

#### Phase 0 设计要点
- **CoTracker3**: 用 windowed (T=24 stride=12), 不是 full-episode (后者 OOM 风险). N=32 query keypoints (auto-grid sampling).
- **RAFT**: temporal stride 3 (从 7M frame pair 减到 2.3M)
- **SAM2**: 1 frame/sec sampling (cloth 形态变化慢, 不需 dense)
- **断点续传**: 每个 .npz 文件存在且 size > 100 byte 即 skip (脚本支持)

### 8.2 Phase 1 — SSL Pretrain ⏳

| Exp | 状态 | Job ID | Val Loss | Downstream MAE | 备注 |
|---|---|---|---|---|---|
| E1.1 V-JEPA baseline | — | — | — | — | 待 Phase 0 |
| E1.4 + track + flow + xview | — | — | — | — | 待 Phase 0 |
| E1.5 Full multi-objective | — | — | — | — | 待 Phase 0 |

### 8.3 Phase 2 — Dynamics ⏳

| Exp | 状态 | Job ID | Val Loss | MMD A↔B | 备注 |
|---|---|---|---|---|---|
| E2.1 Latent dyn baseline | — | — | — | — | 待 Phase 1 |
| E2.2 + Embodiment cond | — | — | — | — | 待 Phase 1 |
| E2.3 + Motion-residual | — | — | — | — | 待 Phase 1 |
| E2.4 + Inverse dyn aux | — | — | — | — | 待 Phase 1 |

### 8.4 Phase 3 — Policy + Ablation ⏳

| Exp | 状态 | Job ID | Val MAE | 真机 | 备注 |
|---|---|---|---|---|---|
| E3.0 baseline | — | — | — | — | 用现有 smooth_800 结果 |
| E3.1 + SSL frozen | — | — | — | — | 待 Phase 1 |
| E3.2 + SSL LoRA | — | — | — | — | 待 Phase 1 |
| E3.3 + Dynamics | — | — | — | — | 待 Phase 2 |
| E3.4 Full Stack | — | — | — | — | 待 Phase 2 |

---

## 9. 修订历史

| 日期 | 内容 |
|---|---|
| 2026-05-21 | 初版: Phase 0-4 (~9 周) 计划; 数据池 (A 6512 + XVLA 1729 + B 800); 资源 (Robot-North-H20 + uc02); 用户决策: 完整 9 周, Phase 0 并行, Phase 1 3-job 并发 |
