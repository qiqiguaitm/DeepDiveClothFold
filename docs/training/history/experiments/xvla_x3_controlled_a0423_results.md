# X-VLA 控制变量 X3 三件套结果 (X3.C / X3.B / X3.A on A_0423_0527)

> **作用**: 记录 2026-05-29 起跑的**控制变量** X-VLA X3 三件套 (全用 A_0423_0527 作 vis 数据 + 统一超参, 唯一变量=域组成) 的训练 + eval 结果。取代原版 vis_v2_merged X3 (作废, 见 [`xvla_track_x_x3_ablation_results.md`](xvla_track_x_x3_ablation_results.md))。
>
> **关联**: 计划/配置 [`../../future_plans/plans/xvla_track_x_curriculum.md`](../../future_plans/plans/xvla_track_x_curriculum.md) §0; 数据管线修复 [`../../../../train_scripts/xvla/data/README.md`](../../../../train_scripts/xvla/data/README.md)。
>
> **最近更新**: 2026-05-31 (三件套训练 + eval 全部完成)。

## 1. 实验设计 (控制变量)

三个实验**唯一变量 = 域组成**, 其余全部相同:

| 项 | 值 (三者一致) |
|---|---|
| Base ckpt | `lerobot/xvla-base` (`/data/shared/ubuntu/workspace/xvla_ckpts`) |
| Vis 数据 | **A_0423_0527** (1085 ep, EE6D 20D, fixed 管线) — 三者**同一份** |
| Steps / lr / warmup / freeze | 30k / 5e-5 / 500 / 1000 |
| Batch | eff 64 (8/gpu × 8) |
| 数据管线 | fixed: rot6d interleaved + gripper 二值化 + decode 修复 |

| 实验 | 域组成 | config | 节点 | ckpt |
|---|---|---|---|---|
| **X3.C** | vis-only (A_0423_0527) | `A_0423_0527` | uc01 | `local_ckpts/xvla_A_0423_0527/step_final` |
| **X3.B** | kai(base+dagger) + A_0423_0527(vis ×7) | `X3B_a0423` | uc02 | `local_ckpts/xvla_x3b_a0423/step_final` |
| **X3.A** | + xvla_soft_fold(×2) | `X3A_a0423` | uc03 | `local_ckpts/xvla_x3a_a0423/step_final` |

域配比沿用原 X3 设计: kai 1:1 / vis ×7 / xvla ×2。

## 2. Eval 结果 (2026-05-31)

**Eval 脚本**: `train_scripts/xvla/eval/eval_xvla_ee6d.py` (PyTorch, `XVLAPolicy.predict_action_chunk` vs GT action)。
**统一 val** (三模型完全相同): A_0423_0527 domain_id=20 末尾 50 ep 的 **1000 个 deterministic strided windows** (stride 82, 全局 idx 1290288…1372206; 同 flow-matching init noise seed `12345+batch_idx`; num_denoising_steps=10; chunk=30)。
**MAE 定义**: `mean(|pred − gt|)` over 20 EE6D dims, 取前 h 步累积平均 (per-step MAE 的前 h 步均值)。

| 实验 | 域组成 | MAE@1 | MAE@10 | MAE@25 | MAE@30 |
|---|---|---:|---:|---:|---:|
| **X3.C** ⭐ | vis-only | **0.0142** | **0.0194** | **0.0316** | **0.0351** |
| X3.B | kai + vis(×7) | 0.0252 | 0.0296 | 0.0417 | 0.0453 |
| X3.A | kai + vis(×7) + xvla(×2) | 0.0274 | 0.0323 | 0.0442 | 0.0478 |

> json: uc01/02/03 `/tmp/eval_x3{c,b,a}.json` (含完整 per-step MAE 曲线 + window 选择 metadata)。

## 3. 结论

- **X3.C (vis-only) 各 horizon 全胜**, 严格序 **X3.C < X3.B < X3.A**。
- **加 kai 域 (X3.B vs X3.C) 明显 HURT**: MAE@1 +78% (0.0142→0.0252), @30 +29%。
- **再加 xvla 域 (X3.A vs X3.B) 进一步微 HURT**: @1 +9%, @30 +6% — 主退化来自 kai, xvla 仅小幅追加。
- → 在 A_0423_0527 vis 分布的 action fidelity 上, 跨域 co-training 混合均回退于干净单域 fit, **kai 域代价最大**。

### ⚠️ 关键 caveat — fit 非 generalization

val windows 来自三模型**都训练过**的 ep (X3.B/A vis 权重 ×7)。vis-only 自然最 fit vis 分布, 此 MAE 是 **"vis action-fit 保真度" 排名, 不直接预测真机成功率** (真机域多样性可能仍有助 robustness)。**真机测试待做才是 X3 域贡献的终判。**

## 4. ⚠️ 与 pi 系列 (pi0.5/JAX) MAE 的可比性 — **不可直接比数值**

X-VLA 的 MAE 与 pi 系列文档 (如 [`task_a_new_pure_200_new_norm_results.md`](task_a_new_pure_200_new_norm_results.md)、[`A_0423_0527_run_a_b_results.md`](A_0423_0527_run_a_b_results.md)) 里的 MAE@k **聚合公式相同, 但数值不可横向比较**。

**相同点 (聚合方式一致)**: 两者都是 `mean(|pred − gt|)` over 前 h 个 timestep × 全部 action 维, 再对所有 eval window/frame 取均值。所以 "@1/@10/@25 趋势随 horizon 上升" 这种**形状**可类比。

**不可比的根因**:

| 维度 | pi 系列 (`eval_val_action_mse.py`) | X-VLA (`eval_xvla_ee6d.py`) |
|---|---|---|
| **Action 表示** | **14D 关节** (12 joint 弧度 + 2 gripper) | **20D EE6D** (xyz 米 + rot6d 单位向量分量 + 二值 gripper, ×2 臂) |
| **单位/量纲** | 全部弧度 (rad, 同量纲) | **混合**: 米 (~0–0.5) + 无量纲 (~[-1,1]) + 二值 {0,1} |
| **每维尺度** | 关节角 std ~0.2–0.5 rad, 同尺度平均 | 米/旋转/夹爪尺度差异大, MAE 是**混合量纲的平均** |
| **归一化** | 推理内部 norm_stats 归一, 但 MAE 在**原始 14D 关节**算 | 无任何归一, MAE 在**原始 EE6D** 算 |
| **Horizon 上限** | @50 (chunk 长) | **@30** (chunk_size=30, 没有 @50) |
| **Val 性质** | held-out val (真泛化, 如 vis_v2_merged_val) | 训练见过的 ep 子集 (fit) |

**结论**:
- ❌ **跨族绝对值不可比**: pi 的 `@1=0.0073` (关节弧度) 和 X-VLA 的 `@1=0.0142` (混合 EE6D) 度量的是**不同物理量**, 数值大小无意义对比。0.0142 > 0.0073 **不代表** X-VLA 更差。
- ✅ **族内相对比较有效**: pi 内部 (Run-A vs Run-B, 同 14D 关节同 val) 可比; X-VLA 内部 (X3.C/B/A, 同 20D EE6D 同 val 同 window) 可比。本文档 §2 的三件套对比正属此类, **有效**。
- ⚠️ 要跨族比, 需统一到**同一物理空间** (例如都转成末端执行器位姿误差 cm + 角度误差 deg, 或都算真机任务成功率), 当前两套 MAE 做不到。
