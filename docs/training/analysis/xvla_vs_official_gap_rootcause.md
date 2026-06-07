# X3.C (我们的 X-VLA) vs 官方 X-VLA + vs pi05 — 真机差的根因分析

> **问题**: X3.C 用 smooth_800 数据训练, **offline action MAE 健康** (单调收敛, 见 [`../future_plans/plans/xvla_track_x_curriculum.md`](../future_plans/plans/xvla_track_x_curriculum.md) §0.NEW.2.5), 但**真机机械臂完成任务不尽人意**; **同一份数据训练的 pi05 真机 work**。
>
> **建立**: 2026-06-01
> **方法**: 逐行对比 ① 官方 X-VLA repo (`xvla/X-VLA/`, THU-AIR) ② lerobot port (`lerobot/policies/xvla/`, 实际训练用) ③ 我们的训练/推理代码 (`train_scripts/xvla/`, `kai0/scripts/serve_policy_xvla.py`)。
> **真机实证**: [`x3c_realrobot_trace_20260601.md`](x3c_realrobot_trace_20260601.md) — 一次真机执行 trace 证实任务失败 = EE 预测高频震荡 (lag1 自相关 −0.34) + IK 放大 (EE↔joint 相关仅 0.47) + 折返比 9~13× + 夹爪 16 次开合, 与 R1/R2 吻合。
> **数据集审计**: [`xvla_dataset_vs_official.md`](xvla_dataset_vs_official.md) — 我们 vis EE6D vs 官方 Agilex 逐项对齐: action 表示/rot6d/gripper/loss-scale/标签语义**全对齐无致命错**; 唯一实质差异 **D5 = action chunk 时间窗口 1s vs 官方 2s** (qdur), 列为 P0 后下一候选。
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
>
> ⚠️ **2026-06-07 重定位(以 §7 为准)**: P0 修了 R1 后真机仍"抓不到衣角",实测复盘 → **R4 容量排除**(官方 0.9B 能 fold)、**新增 D5 = 动作 chunk 表示(我们 1s 稠密 vs 官方 2s intention-abstraction anchor)升为主因**。上表 R1/R4 权重已过时,详见 §7。

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
| ⭐⭐⭐ **P0** 🔄 **执行中** | **修 R1 + 对齐官方配方**, 重训 X3.C (`X3C_smooth800_p0`, uc01, 2026-06-02 起, ETA ~11h) | 1 训练 | 若真机显著改善 → R1 坐实 |
| ⭐⭐ P1 | **R2 诊断**: 已有真机 trace ([`x3c_realrobot_trace_20260601.md`](x3c_realrobot_trace_20260601.md)) 证 EE↔joint 相关仅 0.47; P0 后复测看 IK 是否独立残余 | — | 定位 IK 残余 |
| ⭐ P2 | P0 真机若仍不足, 再评估 R3 (更多步) / R4 (架构) | — | — |

### 6.1 P0 重训配置 (`X3C_smooth800_p0`, 2026-06-02) — 修 R1 + 对齐官方

| 参数 | 官方 X-VLA | 30k 旧版 | **P0 版** | 说明 |
|---|---|---|---|---|
| **图像归一化** | ImageNet | ❌ 只 /255 | ✅ **ImageNet (img-mean)/std** | **R1 修复, 主改动** |
| ColorJitter | 0.2 | ❌ | ✅ 0.2 (brightness/contrast/saturation) | 对齐官方 |
| steps | 50k (finetune 示例) | 30k | **60k** | 适配 eff batch 64 (官方 4×) + R1 后视觉重适应余量 (≈4.3 epoch) |
| lr | 1e-4 | 5e-5 | **1e-4** | 对齐官方 (VLM lr = 1e-5 via scale 0.1) |
| warmup | 2000 | 500 | **2000** | 对齐官方 |
| freeze | 1000 | 1000 | 1000 | 已一致 |
| weight_decay | 0.0 | 1e-4 | **0.0** | 对齐官方 |
| lr schedule | constant (默认) | cosine | cosine (适配) | 定长训练 cosine 更稳, 非照搬 |
| batch | 16 (单卡) | 64 | 64 (适配) | 单机 8 A800 |
| betas / grad clip | (0.9,0.95) / 1.0 | 同 | 同 | ✅ |

- ckpt: `uc01:/data/shared/ubuntu/local_ckpts/xvla_x3c_smooth800_p0/` (每 2k step), 不覆盖 30k 版。
- **train/serve parity 铁律**: `multi_domain_dataset.imagenet_normalize_chw` 与 `serve_policy_xvla` 的 `_IMAGENET_MEAN/STD` 数值完全一致; 推理须加 `--imagenet_norm` (旧 30k ckpt 用 `--no-imagenet_norm`)。ColorJitter 仅训练 (eval/serve image_aug=False)。
- ⚠️ eval: `eval_xvla_ee6d.py` 经 `multi_domain_dataset` 现也归一化 → **P0 ckpt eval 自洽; 旧 30k ckpt 勿再用此 eval** (mismatch)。

### 6.2 P0 验收 — 真机客观判据 (复测 trace 指标)

P0 ckpt 真机录 trace, 对比 [`x3c_realrobot_trace_20260601.md`](x3c_realrobot_trace_20260601.md) 的 30k 基线:

| 指标 | 30k 基线 (差) | P0 目标 |
|---|---|---|
| EE-L y/z 速度 lag1 自相关 | −0.34 / −0.35 (震荡) | → 接近 0 |
| EE 折返比 (路程/净位移) | 9.1× / 13.1× | → ~2× (smooth 数据级) |
| 关节方向反转率 | 0.45~0.69 | → ~0.1 |
| 夹爪开合切换 (200帧) | 16 | → 个位数 |
| 任务完成 | ❌ 需人干预 | → 自主折叠 |
| ⭐ P2 | R1 修复后重跑 §0.NEW.2.5b pi05 对比, 看 gap 缩小多少 → 估 R4 残余 | 2 eval | 量化架构天花板 |

> **关键判断**: pi05 同数据 work 证明**数据没问题、任务可学**。X3.C 的差是**实现/配方**问题 (R1 主导) + **架构**劣势 (R4)。**先修 R1 (零数据成本, 纯代码) 是性价比最高的一步。**

---

## 7. P0 后实测复盘 + 根因重定位 (2026-06-07) — 容量排除,动作表示(D5)是主因

> P0 版 `xvla_x3c_smooth800_p0/step58000` 训完:offline xyz MAE@1 0.0052(P0 前 0.0122→减半,R1 见效),**但真机仍"连衣角都抓不到"**(用户实测)。两个针对性探针(Claude 离线 A100,bart 真 token + 在线 EE6D 转换 + held-out vis;脚本在 gitignored `_xvla_gripper_debug/`)定位真因。

### 7.1 实测诊断

**(a) 运动剖面 — MAE@1 误导,模型没学会"运动幅度"**(按 GT chunk 位移分高/低动量):

| 窗口 | xyz MAE@1 | GT chunk 位移 | pred chunk 位移 | pred/GT |
|---|---:|---:|---:|---:|
| 接近/抓取(高动量) | ~0.005 | ~1.2m | ~0.7–0.9m | **0.60–0.80(欠到位 → 够不到衣角)** |
| 持握(低动量) | ~0.004 | ~0.012m | ~0.31m | **~26×(该静止却乱飘)** |

→ MAE@1 看着好(≈5mm)但只测**第一步 + teacher-forcing**;整条 chunk 的**运动幅度是错的**:接近只走 60-80%(够不到衣角=真机抓不到),持握乱飘 26×。**又一次应验"offline MAE 系统性反指"。**

**(b) denoise 步数无效(零成本旋钮已排除)**:10→20→50 步,欠到位仅 0.74→0.77→0.80(补 6pp,仍够不到),持握更糟 0.33→0.38m,MAE@1 略升。→ 非推理问题。

### 7.2 ⭐ 根因重定位:R4 容量**排除**,D5 动作表示**升为主因**

- **官方 X-VLA-0.9B 能学会 Agilex 叠衣**(Soft-Fold ~100% 成功 / 33 folds·h⁻¹,paper)→ **同架构同 0.9B 容量足够、任务可学 → R4 容量不成立,DEMOTE**。差异必在我们的实现。
- 代码坐实 **D5(动作 chunk 表示)= 真凶**:

| | 官方 X-VLA (Agilex real-world) | 我们 |
|---|---|---|
| 动作 chunk 构造 | `freq=30, qdur=2.0` → `np.linspace(cur, cur+2s, 31)` = **30 anchor 均匀铺在 2 秒**(intention abstraction 插值下采样) | `stack(action[f_idx:f_idx+30])` = **30 连续帧 = 1 秒稠密** |
| 代码 | `base.py:152` + `real_world.py:40`(`qdur=2.0`) | `multi_domain_dataset.py:162-163` |
| 时间窗口 | **2s** | **1s(仅一半)** |
| 采样 | linspace 时间插值 anchor | 连续帧,无下采样 |

- **机理**:`lerobot/xvla-base` 预训练用的是"30 anchor over 2s"的 **intention-abstraction** 动作表示;我们用"30 连续帧 1s 稠密"finetune → **动作的时间窗 + 尺度与预训练先验冲突** → 模型 hedge → (a) 欠到位(走 60-80% → 够不到衣角)、(b) 持握乱飘。与 7.1 实测完全吻合,**且不依赖容量**(官方 0.9B 用对表示就 fold)。
- 这正是数据审计 [`xvla_dataset_vs_official.md`](xvla_dataset_vs_official.md) 早标的 **D5(唯一实质差异:chunk 1s vs 官方 2s)**;P0 后经实测 + 代码坐实,**从"下一候选"升为头号主因**。

### 7.3 根因权重修正(P0 后)

| 根因 | P0 前 | P0 后(本次)修正 |
|---|---|---|
| **D5 动作表示(1s 稠密 vs 官方 2s anchor)** | 未列 | 🔴 **主因**(实测+代码坐实) |
| R1 ImageNet 归一 | 主因 | P0 已修,offline 减半但真机仍废 → **非真机主因** |
| R3 欠训 | 中 | 次(58k 近 plateau) |
| **R4 容量 0.9B** | 中 | ❌ **排除**(官方 0.9B 能 fold) |
| R2 EE6D→IK 链 | 高 | 放大器,非源头 |
| gripper 部署映射(SoftFold -0.0055) | — | 叠加项 |

### 7.4 验证 / 修复(下一步)

- **重建 EE6D 数据用官方 intention abstraction**:每个 query time 取 `linspace(cur, cur+2.0s, num_actions+1)` 时间插值 anchor(对齐 `real_world.py qdur=2.0`),替换"30 连续帧"。重训 X3.C → 看欠到位/乱飘是否消失、真机能否抓到衣角。**这是数据/表示重建,与"官方能 fold"自洽,不是容量。**
- 次要嫌疑:官方 Soft-Fold 可能 **co-train 在 290K 多域语料**(享跨本体先验),我们是**单域 finetune-from-base on 811ep** → 配方差异。D5 修复后若仍不足再查。

---

## 附录 — 关键文件:行

| 项 | 位置 |
|---|---|
| **官方动作 anchor(intention abstraction)** | `xvla/X-VLA/datasets/domain_handler/base.py:152` (`linspace(cur,cur+qdur,N+1)`) + `real_world.py:40` (`qdur=2.0`) |
| **我们动作 chunk(30 连续帧 1s)** | `train_scripts/xvla/data/multi_domain_dataset.py:162-163` |
| 我们训练 forward (绕过 processor) | `train_scripts/xvla/launch/xvla_train.py:330` |
| 我们 dataset 图像 /255 无归一化 | `train_scripts/xvla/data/multi_domain_dataset.py:157` |
| lerobot 归一化在 processor | `lerobot/policies/xvla/processor_xvla.py:73,349` |
| lerobot forward 只 resize | `lerobot/policies/xvla/modeling_xvla.py:313-320` |
| 官方 dataset 归一化 | `xvla/X-VLA/datasets/dataset.py:81` |
| 官方 finetune 50k/warmup2000 | `xvla/X-VLA/README.md:273-275` |
| EE6D→world→IK 真机链 | `kai0/scripts/serve_policy_xvla.py` (`_ee6d_to_world8`) + `ros2_ws/.../policy_inference_node.py` (IK) |
| pi05 同标尺对比 | `xvla_track_x_curriculum.md` §0.NEW.2.5b |
