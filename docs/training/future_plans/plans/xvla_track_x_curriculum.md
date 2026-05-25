# Track X — X-VLA 官方架构 Native 训练 (X3.A + X3.B)

> **状态**: 🟢 RUNNING (2026-05-23 启动) — Stage A 进行中, Stage B 待启动。
> **关联 task**: `#17 Track X X-VLA 官方架构训练`。
> **战略上下文**: [cross_embodiment_strategy.md](../../../deployment/cross_embodiment_strategy.md) §1 (3 robots) + §5.2 (Soft Prompt) + §7 (Tri-track)。

## 1. 核心思路

用 LeRobot's `lerobot/xvla-base` 0.9B ckpt + custom multi-domain wrapper (`xvla_scripts/multi_domain_dataset.py` + `xvla_train.py`) 在 uc01/02 各 8 A800 上跑。EE6D 20D action (kai+vis 用 PiperFK + Rot6D 编码, XVLA-Soft-Fold 用预计算 `observation/eef_6d`)。

与论文 paper-faithful 不同点: 用 lerobot port 不是原 X-VLA repo (LeRobot wrapper 实现更简洁)。

**Curriculum**: continual pretrain (Stage A, multi-domain mixed) → vis-only adaptation (Stage B), 对齐 X-VLA Phase I' + Phase II 框架。

## 2. 数据状态 (全部就绪)

| 数据集 | EE6D 格式 | 路径 |
|---|---|---|
| kai0_base 20D EE6D parquet | 3055 ep / 3.36M frames | uc01/02 NFS |
| kai0_dagger 20D EE6D parquet | 3457 ep / 2.42M frames | 同 |
| vis_v2_merged 20D EE6D parquet | 895 ep / 1.06M frames | 同 |
| xvla_soft_fold action FK cache | 1542 files / 2.85M frames | 同 |

## 3. Prep ✅ 完成

| 项 | 状态 |
|---|---|
| HF ckpt `lerobot/xvla-base` (3.3GB) | ✅ uc01 NFS `/data/shared/ubuntu/workspace/xvla_ckpts/` |
| X-VLA env (lerobot + torch+cu121 + 全依赖) | ✅ uc01 NFS `/data/shared/ubuntu/workspace/X-VLA-env/.venv` |
| EE6D 转换 (kai/vis joint→EE6D 20D, PiperFK + Rot6D) | ✅ |
| XVLA-Soft-Fold action FK 缓存 | ✅ |
| Multi-domain dataset wrapper + DDP training script | ✅ |

## 4. X3.A — 3-domain (A + B + C) 🟢 RUNNING

Balanced sampling: kai+vis×7+xvla×2 (vis ×7 上采样确保部署 prior 占优)。

| 阶段 | 状态 | Job ID | 起 | 终 | Step | 备注 |
|---|---|---|---|---|---|---|
| **X3.A Stage A** Continual Pretrain | 🟢 32% | uc02 PID 429201 | 2026-05-23 08:40 | ETA 4.3h | 6.4k / 20k | uc02 8 A800. 修了 weakref-str bug (save_pretrained → torch.save) |
| **X3.A Stage B** vis-only Adapt | ⏳ pending | — | — | — | / 10k | LR 5e-5 防 overfit, freeze_steps=500, val MAE 选 best |

## 5. X3.B — 2-domain (A + B, no XVLA) 🟢 RUNNING

Balanced sampling: kai+vis×7 (无 XVLA, 用于对照 XVLA 数据贡献)。

| 阶段 | 状态 | Job ID | 起 | 终 | Step | 备注 |
|---|---|---|---|---|---|---|
| **X3.B Stage A** Continual Pretrain | 🟢 32% | uc01 PID 802242 | 2026-05-23 08:40 | ETA 4.3h | 6.4k / 20k | uc01 8 A800 |
| **X3.B Stage B** vis-only Adapt | ⏳ pending | — | — | — | / 10k | 同 X3.A |

## 6. domain_id slot 分配

base ckpt 中未占用 slot:
- 19 = A (KAI0)
- 20 = B (vis) ⭐ 部署目标
- 21 = C (XVLA-Soft-Fold)

推理时 force `domain_id=20` (vis)。

## 7. 决策点

- **D1 (Stage A 完成后)**: X3.A vs X3.B 的 mixed val MAE 对比 — 量化 XVLA 第三方数据贡献是否值得 +2× sample 成本
- **D2 (Stage B 完成后)**: vis B 真机评估 vs X-VLA SoftFold (同硬件) 100% baseline 对照
- **D3**: 若 X3.A/B 都打不过 baseline → Track X 主线降权, Track C (Action Head Cond) 提优先级

## 8. 关联 paper ablation

(完整 Phase 3 ablation 设计见 [`cross_embodiment_strategy.md`](../../../deployment/cross_embodiment_strategy.md) §9 决策点 + §6 RTC/TAC 集成)

Phase 3 table 中:
- **X3.A** Track X (3-domain ⭐) — Florence2 + Soft Prompt, 全数据
- **X3.B** Track X (2-domain) — Florence2 + Soft Prompt, 无 XVLA
- 对照 **C3.0** Track C (Action Head Cond only) — 同 π0.5, 不同 conditioning 注入点
