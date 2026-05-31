# Track X — X-VLA 官方架构 Native 训练 (X3.A + X3.B + X3.C)

> **状态**: 🔄 **原版 X3.A/B/C (vis_v2_merged) 全部作废** (buggy 管线 + 未控制变量, 见 §⚠️/§4/§5/§5.5) → **2026-05-29 换新版控制变量三件套** (全 A_0423_0527 vis + 统一超参, 运行中, 见 §0)。
> **关联 task**: `#17 Track X X-VLA 官方架构训练`。
> **战略上下文**: [cross_embodiment_strategy.md](../../../deployment/strategy/cross_embodiment_strategy.md) §1 (3 robots) + §5.2 (Soft Prompt) + §7 (Tri-track)。

## ⭐ 主要结论 (2026-05-25 update)

**X3.A vs X3.B Stage A eval (vis_v2_merged last 50 ep × 20 q, EE6D 20D MAE)**:

| Metric | X3.A (3-domain A+B+C) | **X3.B (2-domain A+B, no XVLA)** | Δ |
|---|---:|---:|---|
| MAE@1 | 0.0380 | **0.0343** ✓ | -9.7% |
| MAE@10 | 0.0424 | **0.0396** ✓ | -6.6% |
| MAE@25 | 0.0574 | **0.0561** ✓ | -2.3% |

→ **X3.B 全 horizon 完胜 X3.A**, 确认 H1 假设: XVLA-Soft-Fold (第三方 C domain) **dilute vis prior**, 对 vis 部署是 net-negative. **Track X 终态用 X3.B 路线 (kai+vis only, 弃 XVLA)**.

## ⚠️ 数据管线 bug 修复 (2026-05-29) — 上述 X3.A/B/C 结论需重新验证

X3.A/B/C 用的 EE6D 转换器 + dataset wrapper 发现 3 个 bug, 均已修复。脚本同时从 uc `workspace/xvla_scripts/` (repo sibling, 未版本管理) **归位到 `train_scripts/xvla/`** (data/ + launch/)。详见 [`../../../../train_scripts/xvla/data/README.md`](../../../../train_scripts/xvla/data/README.md)。

| Bug | 影响 | 修复 commit |
|---|---|---|
| **Rot6D 排布** `R[:,:2].T.flatten()` (block `[r00,r10,r20,r01,r11,r21]`) ≠ 上游 `quat_to_rotate6d` (interleaved `[r00,r01,r10,r11,r20,r21]`) | 6 个旋转通道 4 个与预训练 base 错位; 部署用上游 `rotation_6d_to_matrix` 解码会 garble 旋转 | `2a01c85` |
| **Gripper 未二值化** (灌原始米值 ~0–0.08) | action_hub 对 gripper(9,19) 用 BCEWithLogitsLoss 要 {0,1}, 原始值近 0 → gripper 永不学闭合 | `5d5d0a4` (`raw*50<1.0→1`, 匹配上游 AIRAgilex) |
| **decode_frame `frame.index`** (当前 PyAV VideoFrame 无此属性) | 每帧解码抛 AttributeError → except 返回全 0 → **所有 vis/parquet 域为黑图** | `9633e2a` (改 pts 推算帧号) |

→ **X3.A/B/C 全部用此 buggy 管线训练**: rot6d 错排 + gripper 失效是**确定**的; 黑图取决于训练时 PyAV 版本 (若与现在同版本, 则 vis/kai parquet 域全黑, 仅 xvla_soft_fold 的 hdf5 cv2 解码不受影响)。**因此 "X3.B 全 horizon 完胜 X3.A" 等结论建立在 buggy 数据上, 必须用修复版重训后重新验证, 暂不作为定论。**

**官方一致性核对** (2026-05-29, 对照实际训练用的 `lerobot.policies.xvla.modeling_xvla.XVLAPolicy`, 非 upstream `xvla/X-VLA` repo): `forward` 内**无任何 Normalize/Unnormalize** (config 的 `ACTION:MEAN_STD`/`VISUAL:IDENTITY` 被自定义 forward 绕过) → **不需要 norm_stats 也不需要 ImageNet 归一**; `chunk_size=n_action_steps=30`; 图像 dataset 出 256/256/224 = `input_features` 声明, policy `resize_imgs_with_padding=[224,224]` 内部统一; EE6D 路径用 **absolute xyz** (upstream real_world handler 同, lerobotv21 的 delta 仅 joint 域)。

## ⭐ §0. 新版控制变量 X3 三件套 (2026-05-29) — 取代原版 §4/§5/§5.5

原版 X3.A/B/C 用 `vis_v2_merged` 作 vis + buggy 管线 + 各异超参 (X3.A/B 20k/lr1e-4, X3.C 30k/5e-5), **既受 bug 污染又未控制变量** (vis 数据 + 超参不一致, 对比不干净) → **全部作废**。

**新版**: 三个实验**统一** vis 数据 = `A_0423_0527`、**统一**超参 (30k / lr 5e-5 / warmup 500 / freeze 1000)、统一 fixed 管线, **唯一变量 = 域组成**。域配比沿用原设计 kai 1:1 / vis(A_0423_0527) ×7 / xvla ×2。

| 新实验 | 域组成 (vis = A_0423_0527) | config | 节点 | output_dir (local_ckpts) | 状态 |
|---|---|---|---|---|---|
| **X3.C** baseline (vis-only) | 仅 A_0423_0527 | `A_0423_0527` | uc01 | `xvla_A_0423_0527` | ⏳ 运行中 (2026-05-29) |
| **X3.B** (+kai) | kai base+dagger + A_0423_0527×7 | `X3B_a0423` | uc02 | `xvla_x3b_a0423` | ⏳ 运行中 |
| **X3.A** (+kai+xvla) | + xvla_soft_fold×2 | `X3A_a0423` | uc03 | `xvla_x3a_a0423` | ⏳ 运行中 |

- 全 30k step / ckpt 每 2k / eff batch 64; 三节点并行 ETA 各 ~4.5h。
- EE6D 数据 (fixed: interleaved rot6d + 二值 gripper) 在 `xvla/data/self_built/{A_0423_0527, kai0_base, kai0_dagger, xvla_soft_fold_action_cache}`。

### §0.1 Eval 结果 (2026-05-31) ✅ — 三件套全部完成

Eval 脚本: `train_scripts/xvla/eval/eval_xvla_ee6d.py` (PyTorch, `XVLAPolicy.predict_action_chunk` vs GT, EE6D 20D MAE)。**统一 val** = A_0423_0527 domain_id=20 末尾 50 ep 的 1000 个 deterministic strided windows (stride 82, 三模型完全相同 + 同 flow-matching init noise seed, 10 denoise steps, chunk=30)。

| 实验 | 域组成 | MAE@1 | MAE@10 | MAE@25 | MAE@30 |
|---|---|---:|---:|---:|---:|
| **X3.C** ⭐ | vis-only (A_0423_0527) | **0.0142** | **0.0194** | **0.0316** | **0.0351** |
| X3.B | kai + vis(×7) | 0.0252 | 0.0296 | 0.0417 | 0.0453 |
| X3.A | kai + vis(×7) + xvla(×2) | 0.0274 | 0.0323 | 0.0442 | 0.0478 |

**结论**: **X3.C (vis-only) 各 horizon 全胜**, 严格序 X3.C < X3.B < X3.A。
- **加 kai 域 (X3.B vs X3.C) 明显 HURT**: MAE@1 +78% (0.0142→0.0252), @30 +29%。
- **再加 xvla 域 (X3.A vs X3.B) 进一步微 HURT**: @1 +9%, @30 +6% — 主要退化来自 kai, xvla 仅小幅追加。
- → 在 A_0423_0527 vis 分布的 action fidelity 上, 跨域 co-training 混合均回退于干净单域 fit, kai 域代价最大。

⚠️ **关键 caveat**: 这是 **fit 不是 generalization** — val windows 来自三模型都训练过的 ep (X3.B/A vis 权重 ×7)。vis-only 自然最 fit vis, 此 MAE **不直接预测真机成功率** (真机域多样性可能仍有助 robustness)。视作 "vis action-fit 保真度" 排名, 非部署裁决。**真机测试待做**才是 X3 域贡献的终判。

---

## 1. 核心思路

用 LeRobot's `lerobot/xvla-base` 0.9B ckpt + custom multi-domain wrapper (`train_scripts/xvla/data/multi_domain_dataset.py` + `train_scripts/xvla/launch/xvla_train.py`, 2026-05-29 从 uc `xvla_scripts/` 归位) 在 uc01/02 各 8 A800 上跑。EE6D 20D action (kai+vis 用 PiperFK + Rot6D 编码, XVLA-Soft-Fold 用预计算 `observation/eef_6d`)。

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

## 4. ~~X3.A — 3-domain (vis_v2_merged)~~ ❌ 作废 (buggy 管线 + 未控制变量, 由 §0 新版 X3A_a0423 取代)

> 以下为原版记录 (保留供溯源)。结果建立在 buggy 数据 (block rot6d + 失效 gripper + 可能黑图) 上, 不作数。

Balanced sampling: kai+vis×7+xvla×2 (vis ×7 上采样确保部署 prior 占优)。

| 阶段 | 状态 | Job ID | 完成 | Step | MAE@1 | @10 | @25 | 备注 |
|---|---|---|---|---|---:|---:|---:|---|
| **X3.A Stage A** Continual Pretrain | ✅ done | uc02 | 2026-05-23 23:08 | 20k step_final | **0.0380** | 0.0424 | 0.0574 | uc02 8 A800, ckpt 3.3GB at `/data/shared/ubuntu/local_ckpts/xvla_x3a_stage_a/step_final/state_dict.pt`. EE6D 20D MAE on vis_v2_merged last 50 ep × 20 q |
| **X3.A Stage B** vis-only Adapt | ❌ skipped | — | — | — | — | — | — | X3.B 已胜出, Stage B 不必走 X3.A |

## 5. ~~X3.B — 2-domain (vis_v2_merged)~~ ❌ 作废 (buggy + 未控制变量, 由 §0 新版 X3B_a0423 取代)

> 原版记录 (溯源用)。"X3.B 完胜 X3.A / 作为 Track X 终态" 结论已作废 — buggy 数据。

Balanced sampling: kai+vis×7 (无 XVLA, 用于对照 XVLA 数据贡献)。

| 阶段 | 状态 | Job ID | 完成 | Step | MAE@1 | @10 | @25 | 备注 |
|---|---|---|---|---|---:|---:|---:|---|
| **X3.B Stage A** Continual Pretrain | ✅ done | uc01 | 2026-05-23 23:09 | 20k step_final | **0.0343** ⭐ | **0.0396** | **0.0561** | uc01 8 A800. ckpt at `/data/shared/ubuntu/local_ckpts/xvla_x3b_stage_a/step_final/state_dict.pt`. **全 horizon 优于 X3.A** |
| **X3.B Stage B** vis-only Adapt | ⏳ pending | — | — | — / 10k | — | — | — | 待启 (LR 5e-5, freeze 500, val MAE 选 best). 或直接用 Stage A 部署 |

## 5.5 ~~X3.C — vis-only direct (vis_v2_merged)~~ ❌ 作废 (buggy, 由 §0/§5.6 新版 X3.C=A_0423_0527 取代)

> 原版记录 (溯源用)。vis 数据是 vis_v2_merged 且 buggy 管线, 结果不作数。

Ablation: 量化 Stage A continual pretrain 的实际增益。从 `lerobot/xvla-base` 官方直接 vis 微调 (无 multi-domain 预训练)。

| 阶段 | 状态 | Job ID | Step | 备注 |
|---|---|---|---|---|
| **X3.C** direct vis-only | ✅ done (2026-05-25) | uc02 PID 2739605 | 20k | 20k step, lr=5e-5, freeze 1000. Output `/data/shared/ubuntu/local_ckpts/xvla_x3c_vis_only_direct/step_final/` |
| **X3.C eval** | ⏳ pending | — | — | 同 X3.A/B 协议 eval, 对比 Stage A 价值 |

## 5.6 X3.C (新版控制集 arm) = A_0423_0527 单数据集 finetune (**fixed pipeline**) — 2026-05-29

新版控制变量三件套的 **baseline arm (vis-only)**, 见 §0。首个用**修复版管线** (rot6d interleaved + gripper 二值化 + decode 修复) 的 X-VLA run。单数据集直接从 `xvla-base` finetune, 也作为 A_0423_0527 在 X-VLA 架构上的 baseline (对照同数据集的 JAX pi05 Run-A/B)。X3.B/A 在此基础上加 kai / kai+xvla 域 (同 vis + 同超参)。

| 项 | 值 |
|---|---|
| 数据集 | `xvla/data/self_built/A_0423_0527` (1085 ep, 1.40M frames, 1.37M chunk-samples, EE6D 20D fixed) |
| 来源 | `kai0/data/Task_A/self_built/A_0423_0527` (Run-A/B 同数据集) joint→EE6D, cnsh→uc TOS 传 8GB deref |
| Config | `A_0423_0527` (`train_scripts/xvla/launch/xvla_train.py`) |
| Steps | **30k** (≈1.40 epoch @ eff batch 64; A_0423_0527 比 vis_v2_merged 大 32%, 30k 匹配/超过 X3.C 1.23-epoch 曝光) |
| LR/freeze | 5e-5, warmup 500, freeze 1000 (同 X3.C) |
| 集群 | uc01 8 GPU, torchrun (port 29534, workers 4) |
| Ckpt | `/data/shared/ubuntu/local_ckpts/xvla_A_0423_0527/` 每 2k step |
| 状态 | ⏳ 运行中 (2026-05-29, step0 loss 102.9, GPU ~96%, ETA ~6h) |

> **数据集存放规范**: 自建 X-VLA EE6D 数据集一律放 `xvla/data/self_built/<name>/` (文件夹经 `self_built/.gitignore` 保留、内容忽略, 不入 git)。转换脚本: `train_scripts/xvla/data/joint_to_ee6d.py` (LeRobot parquet) / `convert_xvla_action.py` (hdf5)。

## 6. domain_id slot 分配

base ckpt 中未占用 slot:
- 19 = A (KAI0)
- 20 = B (vis) ⭐ 部署目标
- 21 = C (XVLA-Soft-Fold)

推理时 force `domain_id=20` (vis)。

## 7. 决策点

- ✅ **D1 (Stage A 完成后)**: X3.A vs X3.B mixed val MAE 对比 — **结果: X3.B 全 horizon 完胜 X3.A (MAE@1 -9.7%)**, XVLA 不值得加。**Track X 终态用 X3.B (kai+vis only)**.
- **D1.5 (X3.C eval 后)**: 量化 Stage A multi-domain pretrain 的价值. 若 X3.C ≈ X3.B, Stage A 是浪费; 若 X3.B < X3.C, Stage A 有效.
- **D2 (X3.B Stage B 后, 可选)**: vis B 真机评估 vs X-VLA SoftFold (同硬件) 100% baseline 对照
- **D3**: 若 X3.B 都打不过 baseline → Track X 主线降权, Track C (Action Head Cond) 提优先级 (但 Track C 已知 collapse, 见 `conditioning_vs_action_representation_ablation.md`)

## 8. 关联 paper ablation

(完整 Phase 3 ablation 设计见 [`cross_embodiment_strategy.md`](../../../deployment/strategy/cross_embodiment_strategy.md) §9 决策点 + §6 RTC/TAC 集成)

Phase 3 table 中:
- **X3.A** Track X (3-domain ⭐) — Florence2 + Soft Prompt, 全数据
- **X3.B** Track X (2-domain) — Florence2 + Soft Prompt, 无 XVLA
- 对照 **C3.0** Track C (Action Head Cond only) — 同 π0.5, 不同 conditioning 注入点
