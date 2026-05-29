# train_scripts/xvla/data/ — kai/vis → X-VLA EE6D 20D 数据构建

从 uc01 `workspace/xvla_scripts/` 归位 (该目录是 deepdive_kai0 的 sibling, git 不跟踪)。脚本内绝对路径 (`/data/shared/ubuntu/...`) 是 uc 数据位置, 在别处跑需改。

## 脚本

| 脚本 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `joint_to_ee6d.py` | LeRobot v2.1 parquet (14D joint) | LeRobot parquet (20D EE6D) | state+action 整体重写, 更新 info.json 为 20D, video symlink 复用 |
| `convert_xvla_action.py` | XVLA-Soft-Fold hdf5 (14D joint) | `.npy` action cache (T,20) | 仅 action, 供 `XVLAHdf5Dataset` mmap 读 |
| `multi_domain_dataset.py` | 上述两种 | torch Dataset | `LeRobotEE6DDataset` / `XVLAHdf5Dataset` / `MultiDomainDataset`, domain_id 19=kai 20=vis 21=xvla |

launcher 在 `../launch/xvla_train.py` (X3A/X3B/X3C/stage_b configs) + `xvla_train_smoke.py`。

## 14D joint → 20D EE6D 约定

**输入 14D** (per arm 7D, left=[0:7] right=[7:14]): `[6 joints(rad), 1 gripper]`
**FK**: piper `C_PiperForwardKinematics(0x01)` (2° j2/j3 offset), xyz mm→m (/1000), 姿态 rpy(deg)→matrix
**输出 20D** (per arm 10D, left=[0:10] right=[10:20]): `[xyz(3,m), Rot6D(6), gripper(1)]`, **全 absolute** (无 delta)
- **gripper 二值化** {0,1}: `raw*50<1.0 → 1(闭合)` (匹配上游 AIRAgilex + 部署阈值)。action_hub 对该通道用 BCEWithLogitsLoss, 必须 {0,1}, 不能灌原始米值 (否则全≈0, gripper 学不会闭合)。

## 官方一致性核对 (2026-05-29, 提交训练前 gate)

对照实际训练用的 `lerobot.policies.xvla.modeling_xvla.XVLAPolicy` (非 upstream `xvla/X-VLA` repo):
- ✅ 一致: 20D 维度/per-arm 布局/rot6d 顺序(修复后)/action chunk=30 (`config.chunk_size=n_action_steps=30`)/相机数/图像尺寸 (dataset 出 256/256/224 = `input_features` 声明, policy `resize_imgs_with_padding=[224,224]` 内部统一)/RGB/abs xyz (EE6D 路径上游也是 abs)/tokenizer(BART max50)/raw 米制
- ✅ **不需要** norm_stats 或 ImageNet 归一: `modeling_xvla.forward` 没有任何 Normalize/Unnormalize, 直接对原始 proprio/action 算 loss (BCE gripper + MSE pos×500 + rot×10), 图像只 resize。config 的 `ACTION:MEAN_STD`/`VISUAL:IDENTITY` 被自定义 forward 绕过。
- 🚫→✅ 唯一 blocker: gripper 未二值化 → 本次已修。
- ⚠️ 非阻断的训练超参差异 (与 upstream train.py): vlm_lr_scale 0.1 vs 1.0, weight_decay 1e-4 vs 0, warmup 1000 vs 2000, cosine vs constant, freeze 粒度。不影响正确性, 保留我们既有设置。

## ⚠️ 已确认的 Rot6D 排布冲突 (2026-05-29 核定)

本目录脚本编码 Rot6D 用:
```python
rot6d = R[:, :2].T.flatten()        # → [r00, r10, r20, r01, r11, r21]  (block: 整列0, 整列1)
```
但 X-VLA 上游全栈是另一种排布 (interleaved / row-major):
| 处 | 代码 | 排布 |
|---|---|---|
| 上游 canonical `datasets/utils.py::quat_to_rotate6d` | `as_matrix()[...,:,:2].reshape(...,6)` | `[r00,r01,r10,r11,r20,r21]` |
| 部署编码 `SoftFold-Agilex/deploy/utils/rotation.py::rotation_matrix_to_6d` | `concat([R[0,:2],R[1,:2],R[2,:2]])` | `[r00,r01,r10,r11,r20,r21]` |
| 部署解码 同文件 `rotation_6d_to_matrix` | `a1=rot[0:5:2], a2=rot[1:6:2]` | 期望上行排布 |

**根因**: 多了个 `.T`。**已于 2026-05-29 修复** → `rot6d = R[:, :2].flatten()` (两脚本均已改, 对齐上游 interleaved)。

**影响**:
- **训练: 不崩, 自洽**。`models/action_hub.py` 的 `EE6DActionSpace.compute_loss` 是逐元素 MSE (ROT_IDX=(3..8)/(13..18) 当平铺向量), 不解释列结构 → 模型只是回归 target 的排布。但 **fine-tune 自 `xvla-base` 时, 6 个旋转通道有 4 个与预训练表示错位 (仅 idx 0/5 重合)**, 浪费预训练对齐, 可能拖慢旋转收敛。
- **部署: 真冲突**。用上游 `rotation_6d_to_matrix` (interleaved) 解码本脚本 block 排布的输出 → 旋转矩阵拼乱, 机器人姿态错。

**后续**: 修复后**所有**用本目录脚本建的 EE6D 数据 (block 排布) 都已过时, 需用修复版重建并重训 X3 — 既包括 `xvla_soft_fold` action cache, 也包括 parquet 域 (kai base/dagger/vis_v2_merged), 否则混用两种排布。修复前的 X3.A/B/C ckpt 作废。
