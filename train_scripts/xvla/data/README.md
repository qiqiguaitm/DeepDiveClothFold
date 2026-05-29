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

**根因**: 多了个 `.T`。去掉即对齐: `rot6d = R[:, :2].flatten()`。

**影响**:
- **训练: 不崩, 自洽**。`models/action_hub.py` 的 `EE6DActionSpace.compute_loss` 是逐元素 MSE (ROT_IDX=(3..8)/(13..18) 当平铺向量), 不解释列结构 → 模型只是回归 target 的排布。但 **fine-tune 自 `xvla-base` 时, 6 个旋转通道有 4 个与预训练表示错位 (仅 idx 0/5 重合)**, 浪费预训练对齐, 可能拖慢旋转收敛。
- **部署: 真冲突**。用上游 `rotation_6d_to_matrix` (interleaved) 解码本脚本 block 排布的输出 → 旋转矩阵拼乱, 机器人姿态错。

**现状**: **未改** `.T` — 保留产出现有 X3.A/B/C 数据 + 已训 ckpt 的确切逻辑 (改了需重建 `xvla_soft_fold` 数据 + 重训)。修复 = 去 `.T` + 重建数据 + 重训, 或部署侧改用 block 解码。决策待定。
