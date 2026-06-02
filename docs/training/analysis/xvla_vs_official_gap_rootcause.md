# X3.C (我们的 X-VLA) vs 官方 X-VLA + vs pi05 — 真机差的根因分析

> **问题**: X3.C 用 smooth_800 数据训练, **offline action MAE 健康** (单调收敛, 见 [`../future_plans/plans/xvla_track_x_curriculum.md`](../future_plans/plans/xvla_track_x_curriculum.md) §0.NEW.2.5), 但**真机机械臂完成任务不尽人意**; **同一份数据训练的 pi05 真机 work**。
>
> **建立**: 2026-06-01
> **方法**: 逐行对比 ① 官方 X-VLA repo (`xvla/X-VLA/`, THU-AIR) ② lerobot port (`lerobot/policies/xvla/`, 实际训练用) ③ 我们的训练/推理代码 (`train_scripts/xvla/`, `kai0/scripts/serve_policy_xvla.py`)。
> **关联**: [`xvla_vs_official_gap_rootcause`] · pi05 同标尺对比 (xvla_track_x_curriculum §0.NEW.2.5b: pi05 EE6D MAE ≈ X3.C 的 1/5)

---

## 0. TL;DR — 三层根因 (按置信度)

| # | 根因 | 证据 | 影响 | 置信 |
|---|---|---|---|---|
| **R1** | **训练全程缺 ImageNet 图像归一化** | `xvla_train.py:330` 直接 `model.forward(batch)`, 绕过 lerobot 的 `XVLAImageNetNormalizeProcessorStep`; dataset 只 `/255.0` | base ckpt 在 ImageNet 域预训练, 我们用 [0,1] 域 finetune → 视觉特征**输入分布错位**, 预训练白费 + 泛化差 | 🔴 高 (代码坐实) |
| **R2** | **EE6D→IK 真机执行链 vs pi05 直出 joint** | X3.C 出 20D EE6D → server `_ee6d_to_world8` → ROS node IK→joint; pi05 直接出 14D joint | IK 在奇异点/容差 (pos 0.04m/rot 0.12rad) 处放大误差; offline 不经此链所以测不出 | 🔴 高 (架构性) |
| **R3** | **欠训 + 弱配方**: 30k step/warmup500 vs 官方 50k/warmup2000 | 官方 README finetune 示例 `--iters 50000 --warmup_steps 2000`; 我们 30k/500; 训练曲线末端仍降 | action fidelity 未到位 (§0.NEW.2.5 显示 30k 仅 2.1 epoch, 末段还在降) | 🟡 中 |
| **R4** | **架构/容量**: X-VLA-base 0.9B vs pi05 2.2B + 成熟 flow-matching | §0.NEW.2.5b 同标尺 pi05 EE6D MAE 各 horizon 4~6× 优于 X3.C | action 保真度天花板更低 | 🟡 中 |

> **为什么 pi05 同数据 work 而 X3.C 不**: pi05 (a) 图像归一化正确 (openpi AgilexInputs 标准管线), (b) 直接输出 joint **不经 IK**, (c) 模型更大更成熟, (d) 训练配方匹配 (50k step EMA)。X3.C 在 R1/R2/R3/R4 四处同时吃亏。

---

## 1. 关键差异表 — 官方 X-VLA / lerobot port / 我们的实现

| 维度 | 官方 X-VLA repo | lerobot port (训练用) | 我们的代码 | 一致? |
|---|---|---|---|---|
| **图像归一化** | `transforms.Normalize(ImageNet mean/std)` (dataset.py:81) | `XVLAImageNetNormalizeProcessorStep` (processor_xvla.py:73) — **在 processor, 不在 forward** | ❌ **dataset 只 /255.0** (multi_domain_dataset.py:157), `forward` 不归一化, 训练脚本不调 processor | 🔴 **不一致 (R1)** |
| 图像增强 | ColorJitter(0.2) 训练时 (dataset.py:78) | (processor 可选) | ❌ 无 ColorJitter | 🟡 缺增强 |
| 图像 resize | Resize(224,224) bicubic | `resize_with_pad` in forward (modeling_xvla:318) | resize_pad → 256/256/224 + forward 内再 resize_with_pad | ✅ 近似 |
| 动作表示 | EE6D 20D, gripper BCE | 同 | 同 (joint_to_ee6d.py) | ✅ |
| 动作归一化 | 无 mean/std, gripper mask | 无 | 无 | ✅ |
| Rot6D 排布 | interleaved `[r00,r01,r10,r11,r20,r21]` | 同 | interleaved (fixed, joint_to_ee6d:42) | ✅ (已修 bug) |
| **总步数** | 默认 1M iters; README finetune **50k** | — | **30k** | 🟡 **偏少 (R3)** |
| **warmup** | **2000** | — | **500** | 🟡 偏少 |
| freeze_steps | 1000 | — | 1000 | ✅ |
| lr | 1e-4 | — | 5e-5 | 🟡 (更小, 配 30k 合理) |
| optimizer | AdamW β(0.9,0.95) wd0 | — | AdamW β(0.9,0.95) wd **1e-4** | 🟡 wd 不同 |
| batch | 16 | — | 8/gpu × 8 = **64 eff** | 🟡 更大 |
| 推理 denoise | 10 步 | — | 10 步 | ✅ |
| **推理平滑** | **无** temporal ensembling (官方) | — | ✅ ROS node `StreamActionBuffer` min_jerk 混合 (policy_inference_node) | ✅ 我们反而更好 |
| **EE→joint** | 官方 deploy 出 world EE pose, 下游 IK | — | server `_ee6d_to_world8` → ROS IK (firmware/host) | 🔴 **额外 IK 链 (R2)** |

---

## 2. R1 (主因) — 训练全程缺 ImageNet 归一化 [代码坐实]

### 证据链

1. **lerobot XVLAPolicy 的 forward 不做归一化**: `modeling_xvla.py:_prepare_images` (line 313-320) 只 `resize_with_pad`, **无 mean/std**。
2. **归一化是 processor 层职责**: `processor_xvla.py:73` `XVLAImageNetNormalizeProcessorStep()` — 文档明确 "validates [0,1] range before normalizing, formula (image-mean)/std" (line 350-355)。**必须在调 forward 前手动跑 processor**。
3. **我们的训练脚本绕过 processor**: `xvla_train.py:330` `model.forward(batch)` 直接调, **全文件无 `make_xvla_pre_post_processors` / `Normalize` / processor import**。
4. **dataset 只 /255.0**: `multi_domain_dataset.py:157` `torch.from_numpy(frame).permute(2,0,1).float()/255.0` → 图像停在 [0,1], 再没归一化。
5. **官方 dataset 明确归一化**: `xvla/X-VLA/datasets/dataset.py:81` `transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))`。

### 为什么这是真机杀手 (而 offline 看不出)

- **base ckpt `lerobot/xvla-base` 是在 ImageNet 归一化域预训练的** (Florence2 视觉塔标准做法)。我们用 [0,1] 域 (均值 ~0.5, 非 0 中心) finetune → **预训练视觉特征的输入统计被破坏**, 模型得从头重学视觉前端 → 等效"半随机初始化 + 欠训"。
- **训练/推理自洽**: 因为训练和 eval 都缺归一化, 模型在 [0,1] 域内自洽学习 → **offline MAE 正常** (这就是 §0.NEW.2.5 单调收敛但绝对值差的原因)。
- **真机差**: 视觉特征质量低 → 对光照/视角/布料外观的泛化差 → 真机新场景动作犹豫/不准。pi05 用 openpi 标准图像管线 (归一化正确), 视觉前端泛化好。

### 验证方法 (推荐先做, 成本最低)

```bash
# A/B: 同 smooth_800, 唯一改 = 加 ImageNet 归一化, 重训 X3.C → 对比 offline MAE + 真机
# 实现: multi_domain_dataset.py 图像 /255 后追加 (img-mean)/std (ImageNet),
#       serve_policy_xvla.py 推理侧同步加 (训练/推理必须一致!)
```

---

## 3. R2 — EE6D→IK 真机执行链 (pi05 无此链)

| | pi05 | X3.C XVLA |
|---|---|---|
| 模型输出 | **14D joint** (直接关节角) | 20D EE6D (末端位姿 + 二值 gripper) |
| 真机执行 | joint 直接下发 | EE6D → world pose (`_ee6d_to_world8`) → **IK → joint** |
| 误差传导 | 无额外环节 | IK 在奇异点 / 容差边界放大 (serve 配置 pos 容差 0.04m, rot 0.12rad, max_jump 0.15rad) |
| offline 是否覆盖 | 是 (MAE 直接在 joint) | **否** (eval 在 EE6D 空间, 不经 IK) → 真机才暴露 |

**机理**: 即使 EE6D 预测精确, IK 求解可能 (a) 多解跳变 (b) 接近奇异点时关节剧烈变化 (c) 容差内"够用即停"导致末端漂移。这些**只在真机闭环出现, offline EE6D MAE 完全测不到**。pi05 直出 joint 绕开整条链。

> 注: 我们的 ROS `StreamActionBuffer` (min_jerk 混合) 已经比官方更努力地平滑, 但平滑的是 IK 之后的 joint 轨迹, 救不了 IK 本身的多解/奇异问题。

---

## 4. R3 — 欠训 + 弱配方

- **步数**: 官方 README finetune 示例 `--iters 50000`, 我们 30k。§0.NEW.2.5 训练曲线显示 **30k 末段 MAE 仍在降** (24k→30k @1 -4%), 30k 仅 ≈2.1 epoch → 确实欠训。
- **warmup**: 官方 2000, 我们 500 — 短 warmup + frozen backbone 1000 步, 解冻后 lr 冲击更大。
- **wd**: 官方 0, 我们 1e-4 — 轻微但方向性差异。
- → 已规划 100k 延长训练 (§0.NEW.5) 验证欠训分量。**但注意: 若 R1 (归一化) 不修, 单纯加步数只是在错误输入域上练更久, 收益有限。R1 应优先。**

---

## 5. R4 — 架构/容量天花板

§0.NEW.2.5b 同标尺 (同 val 窗口 + 同 FK) 对比: **pi05 EE6D MAE 各 horizon 4~6× 优于 X3.C**, xyz/rot6d/gripper 三组全胜。即使修了 R1/R2/R3, X-VLA-base (0.9B, lerobot port) 的 action 保真度天花板大概率仍低于 pi05 (2.2B + 成熟 flow-matching + EMA)。**这是 Track X 的结构性劣势**, 决定了它的价值只能来自跨域/跨本体迁移 (soft prompt 多域共享), 而非单任务 action fidelity。

---

## 6. 修复优先级 + 行动

| 优先级 | 行动 | 成本 | 预期 |
|---|---|---|---|
| ⭐⭐⭐ **P0** | **修 R1: 训练+推理都加 ImageNet 归一化**, 重训 X3.C (A/B 对照: 有/无归一化) | 1 训练 (~5h) | 若真机显著改善 → R1 坐实, 这是主因 |
| ⭐⭐ P1 | **R2 诊断**: 真机录 X3.C 一次执行, log EE6D 预测 + IK 解 + 实际 joint, 看 IK 是否跳变/奇异 | 1 真机 session | 定位 IK 是否独立贡献真机差 |
| ⭐⭐ P1 | R3: P0 修好后再跑 100k (§0.NEW.5), 否则白练 | ~15h | 补足欠训分量 |
| ⭐ P2 | R1 修复后重跑 §0.NEW.2.5b pi05 对比, 看 gap 缩小多少 → 估 R4 残余 | 2 eval | 量化架构天花板 |

> **关键判断**: pi05 同数据 work 证明**数据没问题、任务可学**。X3.C 的差是**实现/配方**问题 (R1 主导) + **架构**劣势 (R4)。**先修 R1 (零数据成本, 纯代码) 是性价比最高的一步。**

---

## 附录 — 关键文件:行

| 项 | 位置 |
|---|---|
| 我们训练 forward (绕过 processor) | `train_scripts/xvla/launch/xvla_train.py:330` |
| 我们 dataset 图像 /255 无归一化 | `train_scripts/xvla/data/multi_domain_dataset.py:157` |
| lerobot 归一化在 processor | `lerobot/policies/xvla/processor_xvla.py:73,349` |
| lerobot forward 只 resize | `lerobot/policies/xvla/modeling_xvla.py:313-320` |
| 官方 dataset 归一化 | `xvla/X-VLA/datasets/dataset.py:81` |
| 官方 finetune 50k/warmup2000 | `xvla/X-VLA/README.md:273-275` |
| EE6D→world→IK 真机链 | `kai0/scripts/serve_policy_xvla.py` (`_ee6d_to_world8`) + `ros2_ws/.../policy_inference_node.py` (IK) |
| pi05 同标尺对比 | `xvla_track_x_curriculum.md` §0.NEW.2.5b |
