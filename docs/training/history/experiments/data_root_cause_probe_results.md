# 数据问题排查实验 — 结果记录 (Data Root-Cause Probe Results)

> **作用**: 记录 [`../../future_plans/plans/data_root_cause_probe_experiments.md`](../../future_plans/plans/data_root_cause_probe_experiments.md) 系列实验的训练 / offline MAE / 真机结果。
> **状态**: 🔄 Exp-1 (裁投放) 训练+MAE 完成; Exp-1b (不裁对照) 训练中, MAE 待出; **H1 终判待两者真机对比**。
> **建立**: 2026-06-02
>
> ⚠️ **方法学铁律 (来自 plan §0)**: 本系列**以真机为终判, offline MAE 系统性反指** (慢/停顿轨迹逐帧误差低却真机灾难; gripper/wrist 问题被 12D arm 稀释)。下面的 MAE **只用于** ① 确认训练健康收敛 ② 选真机测试用的 best ckpt ③ Exp-1 vs Exp-1b 同验证集的相对差。**MAE 不能单独判定 H1** —— 走停/犹豫 (症状①) 在 offline 逐帧 MAE 上几乎不可见。

---

## Exp-1 — `A_0522_0526_no_release` (裁投放, 验证 H1) ✅ 训练+MAE 完成

### 1. 训练配置 (实跑)

| 项 | 值 |
|---|---|
| Config | `pi05_flatten_fold_A_0522_0526_no_release` |
| 集群 | **cnsh 16×A100** (Volc robot-task), FSDP effective batch=128 |
| Init | `mixed_1_clean` |
| 数据 | `A_0522_0526_no_release` (5-22+5-26 共 200 ep, 裁投放后 ~313k frames) |
| Prompt | `"Flatten and fold the cloth."` / abs joints (`use_delta_joint_actions=False`) |
| Steps | **50,000** (plan 写 40k, config 实跑 50k); save_interval=2000, keep_period=10000 |
| 速度 | ~46 步/min (2000 步/43min), 全程稳定 |
| ckpt 根 | `/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_A_0522_0526_no_release/A_0522_0526_no_release_cnsh/` (保留 step `10000 20000 30000 40000 49999`) |

### 2. Offline MAE — saved ckpt 逐点重测 (2026-06-02)

验证集 `vis_v2_merged_val` (30 ep, 与训练 inline-eval 同集, **cross-val**: 训练数据 ≠ 验证数据), prompt `"Flatten and fold the cloth."`, 200 frames, gf0 A100。

| step | MAE@1 | @10 | @25 | @50 | |
|---|---|---|---|---|---|
| **20000** ⭐ | **0.0160** | **0.0378** | **0.0686** | **0.1093** | **全 horizon 最优 → 真机首选** |
| 30000 | 0.0160 | 0.0384 | 0.0695 | 0.1101 | @1 平, 长程更差 |
| 49999 | 0.0163 | 0.0393 | 0.0704 | 0.1110 | 最差 (轻微过拟) |

**训练期 inline-eval 曲线** (同验证集, 每 8k 一次, 交叉印证):

| step | 16000 | 24000 | 32000 | 40000 | 48000 |
|---|---|---|---|---|---|
| MAE@1 | 0.0161 | **0.0159** | 0.0161 | 0.0163 | 0.0163 |
| MAE@50 | 0.1090 | 0.1096 | 0.1103 | 0.1107 | 0.1110 |

> 交叉印证: offline `49999`(@1=0.0163 @50=0.1110) ≈ inline `48000`(@1=0.0163 @50=0.1110) 完全吻合 → offline 重测可信。

### 3. 分析

- **曲线在 16k–24k 触底后单调微劣化** (@1 0.0159→0.0163, @50 0.1090→0.1110): 该数据集 (200 ep / 313k frame) **~20k 步即收敛, 之后轻微过拟**。50k 步对这个规模偏多。
- **best 可部署 ckpt = step 20000** (落在甜区、且是保存点)。已按 `checkpoints_layout.md` 扁平拓扑 A 打包 (剥 train_state, norm_stats 烘进 `assets/A_0522_0526_no_release/`):
  - `TAR: /vePFS/tim/pkg/A_0522_0526_no_release_best_step20000.tar` (11.6 GB)
  - ⚠️ 真机 config 需 `AssetsConfig(asset_id="A_0522_0526_no_release")`, 见打包说明。
- **MAE ≈ 后期 baseline 水平、无显著改善** —— 这**符合预期, 不构成 H1 的证据**: 裁投放只删开头 ~7% 静止帧, 而走停/犹豫 (症状①) 是**推理时的时序行为**, offline 逐帧 teacher-forcing MAE 看不见。H1 成不成立**只能靠真机**。

---

## Exp-1b — `A_0522_0526_raw` (不裁对照) 🔄 训练中, MAE 待出

**对照意义** (plan §1.6): 同两天数据、同 config、同 init、同 step, **唯一差别 = 不裁投放**。排除"只是用了 2 天/200ep 规模效应"的混淆, 让 H1 判定干净。

| 项 | 值 / 状态 |
|---|---|
| Config | `pi05_flatten_fold_A_0522_0526_raw` (50k step) |
| 集群 | **uc02 + uc03 2-node 16×A800** (JAX 多机 FSDP) |
| 数据 | `A_0522_0526_raw` (200 ep, 336,917 frames, 不裁) |
| Init / Prompt | `mixed_1_clean` / `"Flatten and fold the cloth."` (与 Exp-1 一致) |
| 状态 | 🔄 **训练中** (2026-06-02 起), MAE/ckpt 待训练完成回填 |
| ckpt 根 | `/data/shared/ubuntu/workspace/multinode_ckpts/pi05_flatten_fold_A_0522_0526_raw/A_0522_0526_raw_uc16/` (共享 NFS) |

> **基建踩坑** (迁 uc 多机时): 首跑崩在 step-2000 orbax 落盘 (ckpt 落节点本地盘), 换节点重跑又连挂 3 次 (JAX 编译缓存不对称致跨主机 clique init 死锁)。根因+修复见 [`../../deployment/training_ops/submission/uc_cluster_jobs.md §12.11 坑 9/10`](../../deployment/training_ops/submission/uc_cluster_jobs.md)。

### MAE (待回填)

| step | MAE@1 | @10 | @25 | @50 |
|---|---|---|---|---|
| _待训练完成_ | | | | |

---

## H1 终判 — ⏳ PENDING (需 Exp-1b MAE + 两者真机)

| 比较 | 状态 |
|---|---|
| Exp-1 vs Exp-1b **同验证集 offline MAE** (apples-to-apples) | ⏳ 待 Exp-1b 训完 |
| Exp-1 (裁) vs Exp-1b (不裁) **真机走停/犹豫** = H1 终判 | ⏳ 待两 ckpt 真机测 |

**判定规则** (plan §1.7):
- 裁后真机走停/犹豫显著改善 → ✅ **H1 成立** (投放静止段是症状①主因)。
- 无改善 → ❌ H1 排除, 转 Exp-2 (H2 整段慢节奏 / H4 wrist)。
- 改善但残留 loop → ⚠️ H1 部分成立, 与 H2/H4 叠加。

> 松手 (症状②) 预期本实验不改善 (没动 gripper)。
> ⚠️ 再次强调: **offline MAE 即便 Exp-1 与 Exp-1b 几乎相同, 也不能据此说"裁投放无效"** —— 必须真机。
