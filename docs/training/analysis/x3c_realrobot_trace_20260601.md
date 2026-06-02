# X3.C 真机 trace 分析 (trace_20260601_192213) — 震荡/折返实证

> **数据**: `temp/trace_20260601_192213.tar.gz` — X3.C smooth800 step_final 真机一次执行 (200 推理帧 / 66.4s, ipc01, firmware IK)。
> **ckpt** (meta.json): `xvla_x3c_smooth800_step_final` step=30000, domain_id=20, action_format=ee6d_interleaved, ee_ctrl=firmware。
> **建立**: 2026-06-01
> **关联**: 根因分析 [`xvla_vs_official_gap_rootcause.md`](xvla_vs_official_gap_rootcause.md) (R1 归一化 / R2 IK / R3 欠训 / R4 架构)。本文 = 那 4 个根因的**真机实证**。

---

## 0. TL;DR — 任务失败, 表现为全臂高频震荡 + 严重折返

| 现象 | 量化 | 含义 |
|---|---|---|
| **任务未完成** | 帧10 布平摊 → 帧100 **有人手介入** → 帧199 布**仍摊开未折** | 机械臂 66s 没折成, 需人干预 |
| **EE 轨迹震荡** | EE-L y/z 速度 **lag-1 自相关 −0.34/−0.35** | 一步往一个方向、下一步就往回 = 高频抖 |
| **严重折返** | EE-L 折返比 **9.1×**, EE-R **13.1×** (总路程/净位移) | 走 4.4m 净位移才 0.48m, 反复来回 |
| **全臂关节来回** | 12 关节方向反转率 **0.45~0.69** (右臂全 >0.55) | smooth 训练数据 ~0.1; 真机 5~7× | 
| **夹爪反复开合** | L_grip 200 帧切换 **16 次** | "夹取后松手" 的直接体现 |

→ 与你描述的 "机械臂走走停停/犹豫/来回 + 夹取后松手" **完全吻合**。

---

## 1. 模型输出本身 — 健康但不够稳

| 指标 | 值 | 判断 |
|---|---|---|
| out20 范围 (server) | −1.04 ~ 1.14 | ✅ rot6d ±1 正常, **无塌缩** (与 offline 健康一致) |
| EE-L xyz 帧间位移 | median 1.8cm, p90 4.2cm, max 8.5cm | 🟡 大体连续, 但 **9/200 帧 >5cm 突跳** |
| EE-L rpy 帧间 | median 0.037rad, max 0.228rad | 🟡 偶有姿态跳 |
| infer_ms | median 130ms | ✅ 推理速度正常 |

**关键**: 模型 EE 预测**逐帧位移不大**(median 1.8cm), 但**方向高频反复**(lag1 自相关负) → 不是"塌缩/发散", 是"**缓慢趋势上叠加高频抖动**"。这正是视觉前端弱 (R1) → 对相邻帧观测过敏 → 预测方向不稳的表现。

---

## 2. EE↔joint 矛盾 — IK 是放大器 (R2 实证)

| 指标 | 值 | 含义 |
|---|---|---|
| **EE位移 vs joint位移 相关系数** | **0.472** | 正常 IK 应 >0.7; 偏低 = IK 解不够连续 |
| 纯 IK 跳变帧 (EE<1cm 但 joint>0.05rad) | 1.5% (3/199) | IK 多解跳变**存在但非主因** |
| 实际 joint 帧间 |Δ| | median 0.030rad, max 0.217rad | 关节在动, 偶有大跳 |

**判断**: IK (R2) **放大**了模型的 EE 抖动 (相关性只 0.47 说明笛卡尔→关节映射不连续), 但纯 IK 多解跳变只占 1.5% → **IK 是放大器不是主源**。主源是模型 EE 预测的高频不稳 (R1)。pi05 直出 joint 完全绕开这层放大。

---

## 3. 与根因的对应

| 根因 (rootcause.md) | 真机实证 |
|---|---|
| **R1 缺 ImageNet 归一化** (主因) | 真机图像 /255 后 mean=0.31 (非 0 中心); base ckpt 期望 ImageNet 归一化域 (mean~0, range −2.1~1.6)。视觉前端输入错位 → EE 预测高频抖 (lag1 自相关负) → 震荡。**这是任务失败的上游源头**。 |
| **R2 EE6D→IK 链** | EE↔joint 相关仅 0.47, IK 把 EE 抖动放大传到全臂 (反转率 0.45~0.69)。offline 无此链所以测不到。 |
| R3 欠训 / R4 架构 | 模型 EE 预测虽不塌缩但不够稳, 与 30k 欠训 + 0.9B 容量一致 (§0.NEW.2.5b: pi05 EE6D MAE 5× 优)。 |

> **offline 健康 vs 真机失败的桥**: offline 测的是"单帧预测 vs GT 的数值距离"(被训练分布喂同款图像, 自洽), 测不到**真机新观测下视觉前端的泛化抖动**, 更测不到 IK 放大。真机 trace 第一次让这两层暴露。

---

## 4. 对 P0 的指导

1. **R1 修复优先级再次确认**: 真机震荡的上游是 EE 预测高频抖 (lag1 自相关负), 而非 IK 本身 → 修视觉前端 (ImageNet 归一化) 是治本。
2. **P0 A/B 重训后, 复测真机这几个指标**作为客观判据 (不只看"感觉好点"):
   - EE-L y/z 速度 lag1 自相关 (现 −0.34, 目标 → 接近 0 或正)
   - EE 折返比 (现 9~13×, 目标 → smooth 数据级 ~2×)
   - 关节方向反转率 (现 0.45~0.69, 目标 → ~0.1)
   - 夹爪切换次数 (现 16, 目标 → 个位数)
3. **R2 旁证实验**: 若 R1 修后 EE 变平滑但 joint 仍抖 → IK (R2) 是独立残余, 考虑 IK seed 连续性约束 / 切 host IK 带 max_jump 限制。

---

## 附录 — 分析脚本 (一次性, 数据在 temp/, gitignored)

trace schema: `client_trace.jsonl` (client_infer 200 + poscmd 2000), `server_trace.jsonl` (server_infer 201), `client_arrays/*.npz` (200 帧 state14/ee_chunk/pose14), `client_images/*.jpg` (3 路)。
- pose14 在 EE-firmware 模式 = `[xyzL(3), rpyL(3), gripL(1), xyzR(3), rpyR(3), gripR(1)]` (euler 角, **非关节角**; 见 `policy_inference_node.py:609`)。
- state14 = 真实 14D 关节 (7+7: 6 joint + 1 gripper/臂)。
