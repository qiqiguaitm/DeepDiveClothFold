# 我们的 vis EE6D 数据集 vs 官方 X-VLA Agilex 数据 — 对齐审计

> **目的**: 在 P0 (修 R1 图像归一化) 之上, 审计**数据集层面**我们的 vis EE6D (`A_new_smooth_800`) 与官方 X-VLA Agilex handler 的对齐度, 排查除图像外是否还有 action/采样侧问题。
> **建立**: 2026-06-02
> **方法**: 官方 = `xvla/X-VLA/datasets/domain_handler/real_world.py` (AIR-AGILEX) + `action_hub.py` + `base.py`; 我们 = `train_scripts/xvla/data/{joint_to_ee6d,multi_domain_dataset}.py` + 实测 806 ep EE6D 数值。
> **关联**: [`xvla_vs_official_gap_rootcause.md`](xvla_vs_official_gap_rootcause.md) (R1-R4) · [`x3c_realrobot_trace_20260601.md`](x3c_realrobot_trace_20260601.md)。

---

## 0. TL;DR — 数据数值健康, 一个采样差异 (D5) + 两个小瑕疵

| # | 项 | 官方 | 我们 | 判断 |
|---|---|---|---|---|
| ✅ | action 表示 | EE6D 20D absolute | 同 | 对齐 |
| ✅ | xyz 单位/范围 | 米, ±1~2m | 米, max 0.73m | 对齐 (作业空间略小, 正常) |
| ✅ | rot6d 正交归一 | 旋转矩阵前两列 | **col 模长 1.000, 点积 0.000** | **完美** (FK 转换正确) |
| ✅ | gripper | 二值 {0,1}, raw×50<1 | 二值 {0,1}, 同阈值 | 对齐 |
| ✅ | loss scale | XYZ×500 / ROT×10 / grip×1 | 同 (lerobot action_hub) | 对齐 |
| ✅ | action 归一化 | 无 (靠 loss scale 平衡量纲) | 无 | 对齐 |
| ✅ | proprio = action[0] | abs_trajectory[0] | state[0], |Δ|=0.0mm | 对齐 |
| ✅ | action chunk = 未来轨迹 | linspace 插值 | 连续帧 (action[t] vs state[t+k] 平滑递增) | 语义正确 (见 §3) |
| 🟡 **D5** | **action 时间窗口** | **qdur=2.0s, 30 点插值 (67ms/点)** | **连续 30 帧 (33ms/点) = 1.0s** | ⚠️ **我们 chunk 时间跨度仅官方一半** |
| 🟡 | xyz 突跳 | — | 0.20% 帧 >5cm, 3 ep >20cm | 极少, 可忽略/可清 |
| 🟡 | 图像归一化 | ImageNet + ColorJitter | P0 前缺 (R1); **P0 已修** | 见 rootcause R1 |

> **结论**: 数据数值层面**没有致命错误** (rot6d/gripper/xyz/loss-scale 全对齐, 标签语义正确)。**唯一实质差异是 D5 采样时间窗口** (我们 chunk 覆盖 1s, 官方 2s)。图像问题 (R1) 已由 P0 解决。

---

## 1. action 表示 — 完全对齐 ✅

官方 AIR-AGILEX (`real_world.py:27-50`) 与我们 (`joint_to_ee6d.py`):

| | 官方 | 我们 |
|---|---|---|
| 维度 | 20D = L[xyz3+rot6d6+grip1] + R[同] | 同 |
| xyz | 米, 原始 EE 位置 (无 scale) | FK 末端 mm→m (joint_to_ee6d:38) |
| rot6d | quat→6D (旋转矩阵前两列) | FK→R→前两列 interleaved (joint_to_ee6d:43) |
| gripper | `raw×50 < 1.0` → 二值 (real_world:42) | `gripper×50 < 1.0` → 二值 (joint_to_ee6d:48) **同阈值** |
| 归一化 | 无 | 无 |

**实测验证** (806 ep, 924k frames):
- xyz 米级: L_x [−0.24, 0.55], L_z [0.09, 0.73], |max| 0.73m (官方 ±1~2m, 我们作业空间略小, 正常)
- rot6d: 范围 [−1, 1], **col1 模长 mean 1.000, col2 1.000, col1·col2 = 0.000** → FK 输出是合法旋转矩阵前两列, 正交归一完美
- gripper: 干净 {0, 1}, L 闭合 58% / R 44%

→ **action 表示零问题**, 修复版管线 (interleaved rot6d + 二值 gripper) 与官方 Agilex handler 逐项一致。

---

## 2. loss scale — 对齐 ✅

官方 (`action_hub.py:115-117`) 和我们实际训练用的 lerobot port 完全相同:
```
GRIPPER_SCALE = 1.0   XYZ_SCALE = 500.0   ROT_SCALE = 10.0
```
- 官方**不归一化 action**, 而是用 loss scale 平衡量纲: xyz (米, ~0.001 量级 MSE) ×500, rot6d (±1, ~0.01 MSE) ×10, gripper (BCE, ~1) ×1。
- 我们用同一份 lerobot action_hub → scale 一致。**不需要也不应该对 action 做 mean/std 归一化** (与官方一致)。

---

## 3. action chunk 语义 — 正确 (非 bug) ✅

**实测发现**: 源 base 数据 `action[t] ≡ state[t]` (|Δ|=0.00000)。初看像 "action=state 复制" 的标签 bug, 但深查 `action[t] vs state[t+k]`:

| k | |action[t] − state[t+k]| |
|---|---|
| 0 | 0.00000 |
| 1 | 0.00776 |
| 2 | 0.01546 |
| 3 | 0.02309 |
| 4 | 0.03062 |

**平滑单调递增** → action chunk (连续帧) **确实是真实的未来运动轨迹**。`action[t]=state[t]` 只是 Agilex 遥操作录制约定 (action 列 = 当前同步关节读数), action **序列**仍是有效轨迹。X-VLA chunk 训练取 `action[0:30]` = 未来 30 帧轨迹, 语义正确。pi05 也用同份数据 work, 佐证标签无误。

> 注: 官方 handler 取 `proprio=abs_traj[0], action=abs_traj[1:]` (proprio 不进 action); 我们 `multi_domain_dataset` 取 `state=state[f_idx], action=action[f_idx:f_idx+30]`。因 action[t]=state[t], 两者等价。

---

## 4. ⚠️ D5 (唯一实质数据差异) — action 时间窗口只有官方一半

| | 官方 X-VLA Agilex | 我们 |
|---|---|---|
| 采样 | `np.linspace(cur, cur+qdur, num_actions+1)` 插值 (base.py:150) | 连续帧 `action[f_idx : f_idx+30]` (multi_domain_dataset:163) |
| **qdur (未来窗口)** | **2.0 秒** (real_world.py:40) | 隐式 **1.0 秒** (30 帧 @ 30fps) |
| 时间间隔/点 | 2.0/30 = **66.7ms** (≈每 2 帧) | **33.3ms** (逐帧) |
| chunk 覆盖 | 未来 2 秒 | 未来 1 秒 |

**含义**:
- 官方一个 30-step chunk 规划 **2 秒**的动作; 我们只规划 **1 秒**。
- 对叠衣 (慢速、长程任务) , **更短的规划窗口 → 模型每次只看到更近的未来 → 长程一致性差 / 更频繁重规划 → 真机走停**。这与真机 trace 的 "折返/震荡" 部分吻合 (虽主因是 R1, D5 可能加重)。
- 官方用插值 (每 2 帧取 1) 还有**降噪**效果 (跳过逐帧抖动); 我们逐帧把每个 33ms 的小噪声都喂进去。

**修复选项** (P0 之后视真机结果决定):
1. **改采样为 2 秒窗口插值** (对齐官方 qdur=2.0): `multi_domain_dataset` 把 `action[f_idx:f_idx+30]` 改为在 `[f_idx, f_idx+60]` 上 linspace 插值 30 点 (需 scipy interp1d, 同官方)。**这是最对齐官方的改法**。
2. 保持逐帧但加 chunk 长度到 60 (覆盖 2s) — 但 model chunk_size=30 固定, 不可行。
3. 暂不改, 先看 P0 (R1) 真机效果; 若 R1 修后仍有长程走停, 再上 D5。

> ⚠️ **D5 改动需训练+推理一致**: 若改训练采样, 推理侧 `serve_policy_xvla` 的 chunk 执行节奏 (30Hz publish) 与 action 时间间隔 (67ms) 也要对齐, 否则速度错配。**D5 不进 P0** (P0 单变量=R1), 留作 P0 后的独立实验。

---

## 5. 小瑕疵 — xyz 突跳 (可忽略)

- **0.20%** 帧 xyz 位移 >5cm/帧 (1841/924249)。
- **3 个 ep** 有 >20cm/帧 突跳 (ep 54: 284mm, ep 181: 225mm, ep 425: 208mm)。
- 量级极小 (千分之二), 大概率是 FK 在腕部奇异点附近的瞬时跳变或个别坏帧。**可忽略**; 若洁癖可用 §5 脚本定位这 3 ep 截掉突跳段。

---

## 6. 结论 + 对 Track X 的指导

1. **数据集本身健康**: action 表示/单位/rot6d/gripper/loss-scale/标签语义全部与官方 Agilex 对齐, 无致命错误。**pi05 同数据 work 也证明数据可学**。
2. **真机差的数据侧因素排序**: R1 (图像归一化, P0 修复中) ≫ D5 (chunk 时间窗口减半) > xyz 突跳 (可忽略)。
3. **D5 是 P0 之后的下一个候选**: 若 P0 (R1) 真机仍有长程走停/折返, D5 (改 2 秒窗口插值对齐官方) 是优先补救。
4. **不要对 action 做归一化**: 官方靠 loss scale (×500/×10/×1) 平衡, 我们已一致, 加归一化反而偏离。

---

## 附录 — 文件:行

| 项 | 官方 | 我们 |
|---|---|---|
| Agilex action handler | `xvla/X-VLA/datasets/domain_handler/real_world.py:27-50` | `train_scripts/xvla/data/joint_to_ee6d.py:29-49` |
| loss scale | `action_hub.py:115-117` | lerobot `policies/xvla/action_hub.py:119-121` |
| chunk 采样 | `base.py:150-154` (linspace qdur=2.0) | `multi_domain_dataset.py:161-167` (连续帧) |
| gripper 阈值 | `real_world.py:42` (×50<1) | `joint_to_ee6d.py:48` (×50<1) |
| 分析脚本 (一次性) | — | `/tmp/ds_compare.py`, `/tmp/ds_anomaly.py`, `/tmp/sem_check.py` (uc01) |
