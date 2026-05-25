# Track A — SSL Phase 0-3 预训练 Pipeline

> **状态**: Phase 0 部分 done (Kai0 base+dagger CoTracker3 + SAM2), 余下待启动。
> **关联 task**: `#11 SSL Phase 0`, `#12 SSL Phase 1`, `#13 SSL Phase 2`, `#14 SSL Phase 3`。
> **战略上下文**: [cross_embodiment_strategy.md](../../../deployment/cross_embodiment_strategy.md) §3 (4-层 ROI) + §7 (Tri-track) + §8 (风险)。

## Phase 0 — 数据预处理 🔄 in_progress (启动 2026-05-21)

| Sub-task | 状态 | 启动 | 完成 | 备注 |
|---|---|---|---|---|
| 环境安装 (uc02 kai0 venv) | ✅ done | 2026-05-21 06:05 | 2026-05-21 06:14 | cotracker3, decord, einops, opencv, pyarrow |
| 真实视频 timing 实测 | ✅ done | 2026-05-21 06:19 | 60s/ep × 3 view | 8 GPU 并行预估总 ~17h |
| **E0.1 Kai0_base** (3055 ep) | ✅ done | 2026-05-21 06:20 | 2026-05-21 12:22 | uc02 8 GPU, 6h02. 输出 2.0G tracks |
| **E0.1 Kai0_dagger** (3457 ep) | ✅ done | 2026-05-21 12:23 | 2026-05-22 ~04 UTC | uc02 8 GPU |
| E0.1 vis_v2_merged (895 ep) | ⏳ 待启动 | — | — | 同上 |
| E0.1 XVLA-Soft-Fold (1729 ep) | ⏳ 待启动 | — | — | hdf5 格式, 需 dataset adapter |
| E0.2 RAFT optical flow | ⏳ 待启动 | — | — | 待 E0.1 完成, 复用 uc02 GPU |
| E0.3 SAM2 cloth mask | ✅ done | 2026-05-21 17:40 UTC | 2026-05-22 01:13 UTC | Robot-North-H20 1 节点, 6512 ep × 3 view |
| ~~E0.4 FOV alignment~~ | ❌ 取消 | — | — | 由 view-cond token + RandomResizedCrop 替代 |
| ~~E0.5 EE-relative action~~ | ❌ 取消 | — | — | 由 Soft Prompt + Action Head Cond 替代 |
| **Phase 0 整体** | 🔄 in_progress | 2026-05-21 | — | ETA ~3-5 day |

## Phase 1 — SSL Pretrain ⏳ pending Phase 0

| Exp | 状态 | Job ID | Val Loss | Downstream MAE | 备注 |
|---|---|---|---|---|---|
| E1.1 V-JEPA baseline | — | — | — | — | 待 Phase 0 |
| E1.4 + track + flow + xview | — | — | — | — | 待 Phase 0 |
| E1.5 Full multi-objective | — | — | — | — | 待 Phase 0 |

## Phase 2 — Dynamics ⏳ pending Phase 1

| Exp | 状态 | Job ID | Val Loss | MMD A↔B | 备注 |
|---|---|---|---|---|---|
| E2.1 Latent dyn baseline | — | — | — | — | 待 Phase 1 |
| E2.2 + Embodiment cond | — | — | — | — | 待 Phase 1 |
| E2.3 + Motion-residual | — | — | — | — | 待 Phase 1 |
| E2.4 + Inverse dyn aux | — | — | — | — | 待 Phase 1 |

## Phase 3 — Policy + Final Ablation Table ⏳ pending Phase 2

| Variant | Visual | Dynamics | Soft Prompt | Action Head Cond | Motion-residual |
|---|---|---|---|---|---|
| **E3.0** baseline (π0.5 default) | — | — | — | — | — |
| **E3.1** + Visual SSL | E1.5 frozen | — | — | — | — |
| **E3.2** + LoRA tune | E1.5 LoRA | — | — | — | — |
| **E3.3** + Dynamics | E1.5 LoRA | E2.3 | — | — | ✓ |
| **X3.A** Track X 3-domain ⭐ | Florence2 | — | ✓ | — | — |
| **X3.B** Track X 2-domain | Florence2 | — | ✓ | — | — |
| **C3.0** Track C only ⭐ 新 | π0.5 default | — | — | ✓ | — |
| **E3.7** Soft Prompt + SSL | E1.5 LoRA | — | ✓ | — | — |
| **E3.8** Action Cond + SSL ⭐ 新 | E1.5 LoRA | — | — | ✓ | — |
| **E3.9** Dual Cond + SSL ⭐ 新 | E1.5 LoRA | — | ✓ | ✓ | — |
| **E3.4** Full Stack | E1.5 LoRA | E2.3 | ✓ | ✓ | ✓ |

## 相关风险 (`cross_embodiment_strategy.md` §8 摘选)

1. CoTracker3 在 heavy occlusion (crumpled cloth) 失败 → confidence filter + mask 加权 track loss
2. RAFT 在 fast motion 失败 → Quasi-static 阶段训 flow, dynamic 降权
3. π0.5 PaliGemma continual SSL 易 catastrophic forget → layer-wise lr decay, peak 5e-5, 1% LAION anchor loss
4. Multi-objective loss 不收敛 → Phase 1 先单 V-JEPA 5k step 预热, 再逐项加
5. Embodiment cond 位置 → Phase 1 visual 不区分 view, Phase 2 dynamics 才区分
