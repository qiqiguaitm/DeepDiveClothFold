# Track X — X-VLA 官方架构 Native 训练 (X3.A + X3.B + X3.C)

> **状态**: 🔄 vis_v2_merged 版作废 → A_0423_0527 控制变量 → **smooth_800 三件套训练完成 + eval (§0.NEW)** → X3.C 真机**失败** → 根因 R1 (缺 ImageNet 归一化) → **P0 重训 `X3C_smooth800_p0` (修 R1 + 对齐官方) **uc→volc cnsh 8卡 60k ✅ 训完 + offline eval done (best=step_058000, @1=0.0341), 见 §0.NEW.6.7; 🔴 真机终判待做**, 见 §0.NEW.6 ⭐⭐**。§0/§0.1 (A_0423_0527) 降为对照。
> **关联 task**: `#17 Track X X-VLA 官方架构训练`。
> **战略上下文**: [cross_embodiment_strategy.md](../../../deployment/strategy/cross_embodiment_strategy.md) §1 (3 robots) + §5.2 (Soft Prompt) + §7 (Tri-track)。

## ⚠️ 数据管线 bug 修复 (2026-05-29) — 上述 X3.A/B/C 结论需重新验证

X3.A/B/C 用的 EE6D 转换器 + dataset wrapper 发现 3 个 bug, 均已修复。脚本同时从 uc `workspace/xvla_scripts/` (repo sibling, 未版本管理) **归位到 `train_scripts/xvla/`** (data/ + launch/)。详见 [`../../../../train_scripts/xvla/data/README.md`](../../../../train_scripts/xvla/data/README.md)。

| Bug | 影响 | 修复 commit |
|---|---|---|
| **Rot6D 排布** `R[:,:2].T.flatten()` (block `[r00,r10,r20,r01,r11,r21]`) ≠ 上游 `quat_to_rotate6d` (interleaved `[r00,r01,r10,r11,r20,r21]`) | 6 个旋转通道 4 个与预训练 base 错位; 部署用上游 `rotation_6d_to_matrix` 解码会 garble 旋转 | `2a01c85` |
| **Gripper 未二值化** (灌原始米值 ~0–0.08) | action_hub 对 gripper(9,19) 用 BCEWithLogitsLoss 要 {0,1}, 原始值近 0 → gripper 永不学闭合 | `5d5d0a4` (`raw*50<1.0→1`, 匹配上游 AIRAgilex) |
| **decode_frame `frame.index`** (当前 PyAV VideoFrame 无此属性) | 每帧解码抛 AttributeError → except 返回全 0 → **所有 vis/parquet 域为黑图** | `9633e2a` (改 pts 推算帧号) |

→ **X3.A/B/C 全部用此 buggy 管线训练**: rot6d 错排 + gripper 失效是**确定**的; 黑图取决于训练时 PyAV 版本 (若与现在同版本, 则 vis/kai parquet 域全黑, 仅 xvla_soft_fold 的 hdf5 cv2 解码不受影响)。**因此 "X3.B 全 horizon 完胜 X3.A" 等结论建立在 buggy 数据上, 必须用修复版重训后重新验证, 暂不作为定论。**

**官方一致性核对** (2026-05-29, 对照实际训练用的 `lerobot.policies.xvla.modeling_xvla.XVLAPolicy`, 非 upstream `xvla/X-VLA` repo): `forward` 内**无任何 Normalize/Unnormalize** (config 的 `ACTION:MEAN_STD`/`VISUAL:IDENTITY` 被自定义 forward 绕过) → **不需要 norm_stats 也不需要 ImageNet 归一**; `chunk_size=n_action_steps=30`; 图像 dataset 出 256/256/224 = `input_features` 声明, policy `resize_imgs_with_padding=[224,224]` 内部统一; EE6D 路径用 **absolute xyz** (upstream real_world handler 同, lerobotv21 的 delta 仅 joint 域)。

## ⭐⭐ §0.NEW — X3 三件套换 vis = A_new_smooth_800 重训 (2026-05-31, 当前主线)

> **动机**: §0 (2026-05-29) 用 `A_0423_0527` 作 vis 跑了控制变量三件套, 但 §0.1 eval 是 **fit 非泛化** (val 来自训练过的 ep), 且 A_0423_0527 真机未验证。改用 **`A_new_smooth_800`** 作 vis 重训 —— 这个数据集 **811 ep / X1 cleaned, 真机已验证 work** (见 [`../../history/experiments/task_a_vis_curated_subset_experiments.md`](../../history/experiments/task_a_vis_curated_subset_experiments.md) + [`task_a_new_smooth_800_new_norm_results.md`](../../history/experiments/task_a_new_smooth_800_new_norm_results.md)), 作为 X-VLA vis domain 比 A_0423_0527 更可靠的部署锚点。

### §0.NEW.1 数据集 — A_new_smooth_800

| 项 | 值 |
|---|---|
| 路径 (cnsh) | `kai0/data/Task_A/self_built/A_new_smooth_800/base` (811 ep, ~930K frames) |
| Val | `kai0/data/Task_A/self_built/A_new_smooth_800/val` |
| 来源 | vis_base 全期 → X1 自动化清洗 (排高频抖动 / Class C 跳变) |
| 真机 | ✅ 已验证 (闭合稳定 / 不 oscillate / 不松开), pi05 JAX MAE@1=0.0089 |
| 为何换 | A_0423_0527 eval 是 fit 非泛化 + 真机未验证; smooth_800 是已验证的 work 锚点 |

> ⚠️ **EE6D 转换前置**: smooth_800 是 14D joint parquet, X-VLA 要 EE6D 20D。需先用 `train_scripts/xvla/data/` 的 **fixed 转换器** (interleaved Rot6D + 二值 gripper, 见 §⚠️ 3 bug 修复) 把 smooth_800 转成 EE6D, 落到 `xvla/data/self_built/A_new_smooth_800/`。**不要复用旧 buggy 转换缓存。**

### §0.NEW.2 三件套配置 (唯一变量 = 域组成)

统一: vis = A_new_smooth_800 (EE6D), 统一超参 (30k / lr 5e-5 / warmup 500 / freeze 1000 / eff batch 64 / ckpt 每 2k), 统一 fixed 管线。域配比沿用 kai 1:1 / vis ×7 / xvla ×2。

| 实验 | 域组成 (vis = A_new_smooth_800 EE6D) | config | 节点 | output_dir | 状态 |
|---|---|---|---|---|---|
| **X3.C** baseline (vis-only) | 仅 A_new_smooth_800 | `X3C_smooth800` | uc01 | `xvla_x3c_smooth800` | ✅ 训练完成 (30k) + eval (§0.NEW.2.5) |
| **X3.B** (+kai) | kai base+dagger + smooth800 ×7 | `X3B_smooth800` | uc02 | `xvla_x3b_smooth800` | ✅ 训练完成 (30k), eval 待做 |
| **X3.A** (+kai+xvla) | + xvla_soft_fold ×2 | `X3A_smooth800` | uc03 | `xvla_x3a_smooth800` | ✅ 训练完成 (30k), eval 待做 |

> ckpt: `uc0{1,2,3}:/data/shared/ubuntu/local_ckpts/xvla_x3{c,b,a}_smooth800/step_final/state_dict.pt` (各 3.3G, step=30000)。三个 run 2026-05-31 17:54 启动, 6-1 03:40~03:44 跑完, 日志均 `=== DONE ===`。

### §0.NEW.2.5 X3.C Eval — 训练健康 ✅ + 与 pi05 跨架构对比 (2026-06-01)

Eval 脚本: `train_scripts/xvla/eval/eval_xvla_ee6d.py` (EE6D 20D MAE) + `eval_pi05_fk_ee6d.py` (新增, pi05 joint→FK→EE6D 同标尺)。统一 val = A_new_smooth_800 EE6D domain_id=20 末 50 ep (episode_index ≥ 756) 的 1000 个 deterministic strided windows (stride 87), 10 denoise steps, chunk=30。

#### (a) X3.C 训练健康检查 — 全部正常 ✅

| 检查项 | 结果 |
|---|---|
| 权重 NaN/Inf | ✅ 904 个 float tensor 全干净, global L2=1050 (非退化/爆炸) |
| 训练完整性 | ✅ step=30000 满额, pred chunk=30 对齐 GT |
| **per-step MAE 曲线** | ✅ 单调爬升 t=1→30: 0.0146→0.0596, 平滑无 flat/跳变 → 学到真实 action 动态 |
| **逐维无死通道** | ✅ 20 维 pred std 全贴合 GT std, 无塌缩 |
| **gripper (历史 BCE bug)** | ✅ 二值准确率 L=99.4% / R=99.7%, pred_closed% ≈ gt_closed% |
| **rot6d (历史排布 bug)** | ✅ 6 旋转维 pred std 匹配 GT, 修复版管线 (interleaved + 二值化) 生效 |

→ **X3.C smooth800 训练完全正常, 20 个 action 维度无一异常。**

X3.C 自身 MAE (EE6D 20D, final, 1000 窗口):

| | MAE@1 | MAE@10 | MAE@25 | MAE@30 |
|---|---:|---:|---:|---:|
| X3.C (smooth800) | 0.0146 | 0.0206 | 0.0333 | 0.0372 |
| X3.C (§0.1 a0423 版, 参考) | 0.0142 | 0.0194 | 0.0316 | 0.0351 |

两数据集上 X3.C fit 质量高度一致 (@1 差 3%) → 管线在 smooth800 上行为稳定。

#### (a.2) 训练曲线 — 全 15 个 ckpt 的 MAE 收敛 (脚本 `eval_xvla_ee6d.py` × 各 ckpt, 500 窗口同 val)

| step | MAE@1 | MAE@10 | MAE@30 | bucket loss |
|---:|---:|---:|---:|---:|
| 2000 | 0.0774 | 0.0783 | 0.0873 | 2.18 |
| 4000 | 0.0441 | 0.0468 | 0.0613 | 1.54 |
| 6000 | 0.0347 | 0.0382 | 0.0524 | 1.18 |
| 8000 | 0.0313 | 0.0347 | 0.0485 | 1.18 |
| 10000 | 0.0282 | 0.0328 | 0.0501 | 0.90 |
| 12000 | 0.0280 | 0.0327 | 0.0494 | 1.12 |
| 14000 | 0.0235 | 0.0296 | 0.0475 | 0.72 |
| 16000 | 0.0199 | 0.0257 | 0.0425 | 0.85 |
| 18000 | 0.0196 | 0.0253 | 0.0426 | 0.81 |
| 20000 | 0.0184 | 0.0241 | 0.0412 | 0.60 |
| 22000 | 0.0169 | 0.0227 | 0.0396 | 0.64 |
| 24000 | 0.0155 | 0.0213 | 0.0376 | 0.55 |
| 26000 | 0.0152 | 0.0209 | 0.0375 | 0.61 |
| 28000 | 0.0150 | 0.0205 | 0.0364 | 0.67 |
| **30000** (final) | **0.0149** | **0.0205** | **0.0367** | — |

> loss = log 中每 2k step 桶均值 (8.0→0.55 单调降, 见下)。MAE/loss 双口径交叉验证。

**收敛分析**:
- **MAE@1 全程严格单调下降** 0.0774 → 0.0149 (无回弹/无过拟合), final=best。**没有 plateau 前的过拟合回升**, 说明 30k step 数据曝光充分但未过训。
- **降速分段**: 早期陡降 (2k→8k: 0.0774→0.0313, -60%), 中期稳降 (8k→20k: -41%), 后期放缓 (20k→30k: 0.0184→0.0149, -19%)。**末段 10k 仍在降但收益递减** → 与 loss 曲线 (20k 后 0.55~0.67 plateau) 吻合。
- **long-horizon (@30) 同步单调降** 0.0873→0.0367, 无发散。
- loss 曲线 (2k 桶均值): `8.0(0) → 2.18(2k) → 1.18(8k) → 0.90(10k) → 0.60(20k) → 0.55(24k) → 0.67(28k)`, 单调降无发散, 20k 后 plateau。

→ **训练动力学完全健康**: 单调收敛、final=best、loss/MAE 双口径一致、无过拟合回弹。若想省算力, **24k step 起 MAE 已近收敛** (24k→30k @1 仅 -4%), 下次可考虑 24-26k 截断。

#### (b) ⭐ 跨架构对比: pi05 (FK→EE6D) vs X3.C (XVLA) — 同标尺

把 **pi05 smooth_800** (uc03 JAX ckpt, joint-space MAE@1=0.0089) 拉到 EE6D 同标尺: 在**完全相同的 1000 val 窗口** (同 cutoff 756 / stride 87) 上, pi05 推理出 14D joint → 用**同一 `joint_to_ee6d.py` FK** 转 20D EE6D, 与 X3.C 同 EE6D GT 比。

**总体 MAE (1000 窗口, EE6D 20D, 可直接比)**:

| Horizon | **pi05 (FK→EE6D)** | X3.C (XVLA) | pi05 优势 |
|---|---:|---:|---:|
| MAE@1 | **0.0025** | 0.0146 | **5.8×** |
| MAE@10 | **0.0053** | 0.0206 | 3.9× |
| MAE@25 | **0.0065** | 0.0333 | 5.1× |
| MAE@30 | **0.0067** | 0.0372 | 5.5× |

**分组 MAE (320 窗口, 定位差距) — 三组全胜**:

| 维度组 | **pi05** | X3.C | pi05 优势 |
|---|---:|---:|---:|
| xyz (末端位置 6 维) | **0.0027** | 0.0122 | 4.5× |
| rot6d (末端朝向 12 维) | **0.0082** | 0.0451 | 5.5× |
| gripper (二值 2 维) | **0.0055** | 0.0523 | 9.5× |
| gripper 二值准确率 | 99.4/99.5% | 99.4/99.7% | 持平 (都好) |

**结论**: 在 smooth_800 同源数据、同 1000 val 窗口、同 EE6D 20D 标尺下, **pi05 的动作预测保真度约为 X3.C XVLA 的 5 倍 (@1 0.0025 vs 0.0146), 且位置/朝向/夹爪三组维度全面领先**。X3.C 自身训练健康 (无塌缩), 但 action fidelity 显著不及成熟 pi0.5。

> 🔴 **真机差的根因分析 (2026-06-01)**: 见 [`../../analysis/xvla_vs_official_gap_rootcause.md`](../../analysis/xvla_vs_official_gap_rootcause.md) — 逐行对比官方 X-VLA / lerobot port / 我们的代码, 定位 4 层根因: **R1 训练全程缺 ImageNet 图像归一化** (主因, 代码坐实: `xvla_train.py:330` 绕过 lerobot processor, dataset 只 /255) → base ckpt 输入域错位; **R2 EE6D→IK 真机执行链** (pi05 直出 joint 无此链, offline 测不到); R3 欠训 (30k vs 官方 50k); R4 架构容量 (0.9B vs pi05 2.2B)。**P0: 先修 R1 (纯代码零数据成本) 重训对照。**

**Caveat (诚实标注)**:
1. **action-fit 保真度, 非真机成功率** — 两者 held-out 均来自训练分布 (fit 非泛化)。MAE 低 ≠ 真机一定好, 但证明 pi05 动作精度远高于 X3.C。
2. **FK 对 pi05 中性偏不利** — pi05 在 joint 空间训练, FK 转 EE6D 会**放大** joint 误差 (几何链式传导), 按理 pi05 应吃亏; 它仍大幅领先 → 真实优势比表面数字更大。
3. **差距主因 = 架构/预训练** — 同 smooth_800 同 vis 域; 差异来自 pi0.5 (成熟 flow-matching VLA) vs X-VLA-base (0.9B lerobot port)。
4. 唯一不严格处: pi05 horizon=50 取前 30 对齐 X3.C chunk=30 (已处理); gripper 量纲在 FK 后两者均二值化, 公平。

#### (c) ⭐ P0 修复后实测 + 真机 gripper 部署 bug (2026-06-05)

P0 版 ckpt `xvla_x3c_smooth800_p0/step_058000` 训练完成后实测 (Claude 离线诊断, A100; bart 真 token + 14D→EE6D 在线转换, held-out 末50ep strided 256 窗口, 同 `joint_to_ee6d.py` FK 标尺):

**(c1) P0 把位置 MAE 砍掉一半 (R1 ImageNet 修复见效)**:

| | xyz MAE@1 | @10 | @25 | @30 |
|---|---:|---:|---:|---:|
| pi05 (FK→EE6D, 参考) | **0.0027** | — | — | — |
| X3.C **P0 前** (§(b)) | 0.0122 | — | — | — |
| **X3.C P0 (step58000) 实测** | **0.0052** | 0.0113 | 0.0251 | 0.0295 |

- P0 (ImageNet 归一化) 把 xyz MAE@1 从 **0.0122 → 0.0052 (~2.3× 改善)**, 坐实 R1 是主因。但仍 **~2× pi05 (0.0027)** —— 剩余差距 = R4 容量 (0.9B vs 2.2B) + R3 欠训, **继续训不划算** (MAE ~30k 后 plateau)。
- gripper 二值: 稳定帧 **>99% 正确** (与 §(b) 一致); 仅抓取过渡帧约 77% (难帧, 正常)。**模型 gripper 信号无塌缩。**

**(c2) 🔴 真机"抓不到 + 夹爪状态异常"的根因 = 部署端 gripper 米值映射错配 (配置 bug, 非模型)**:

| | vis Piper 真实行程 (A_0423_0527 实测) | 部署常量 (`serve_policy_xvla.py:410-411`) |
|---|---|---|
| 闭 (closed) | ~**0.0003 m** (正, 近 0) | `g_close = **-0.0055 m**` (负!) |
| 开 (open) | ~**0.079 m** | `g_open = 0.0656 m` |

- 部署用的是 **SoftFold 域默认值**, 不是 vis Piper 行程。模型输出二值方向正确 (sig>0.5→闭), 但 serve 把"闭"映射成 **负米值 -0.0055** → 夹爪命令异常/闭合不到位 → **抓不到布**。
- ⚠️ bringup `xvla_inference_bringup.md` **R3 早预警过** ("SoftFold 默认值, 负 close 可能被硬件 clip, 需 `--gripper_open_value/--gripper_close_value` 覆盖"), 部署时未覆盖。
- **修复**: 部署加 `--gripper_close_value 0.0`(或 0.0003)`--gripper_open_value 0.08`, 对齐 vis Piper; 并核对机器人 gripper 命令单位/方向。**先改这个再上真机。**

> **小结**: P0 训练健康 (xyz 0.0052, gripper 无塌缩); 真机失败主因是**部署 gripper 映射 bug** (与 MAE 高低无关)。XVLA 位置 MAE 比 pi05 高 ~2× 是 0.9B 容量固有差距, 不是 bug。诊断脚本在 gitignored `_xvla_gripper_debug/`。

### §0.NEW.3 实施步骤

| Step | 内容 | ETA | 状态 |
|---|---|---:|---|
| T1 | smooth_800 → EE6D 20D (fixed 转换器, interleaved Rot6D + 二值 gripper) → `xvla/data/self_built/A_new_smooth_800/` | 1-2h | ✅ |
| T2 | 写 3 个 config (X3A/B/C `_smooth800`), datasets_yaml 指向新 EE6D | 0.5h | ✅ |
| T3 | 三节点并行训练 (uc01/02/03, 各 30k step) | 各 ~4.5h | ✅ 完成 |
| T4 | 统一 val eval (smooth_800 val EE6D, deterministic windows, 同 seed) | 1h | 🔄 X3.C 完成 (§0.NEW.2.5), X3.B/A 待做 |
| T5 | ⭐ **真机测试** (X3 三件套终判, 非 offline MAE) | 1 day | ⏳ 待做 |

### §0.NEW.4 与 §0 (A_0423_0527) 的关系

- §0 / §0.1 (A_0423_0527) **降为对照**, 不删 (保留 fit 排名 X3.C < X3.B < X3.A 作参考)。
- §0.NEW 用 smooth_800 重训, 重点是 **真机可部署 + 真机终判**, 不只看 offline fit。
- 两版可对比 "vis 数据选择 (A_0423_0527 vs smooth_800) 对 X3 域贡献结论是否稳健"。

### §0.NEW.5 ⭐ X3.C 100K 步延长训练 — ⚠️ 被 P0 取代 (2026-06-02)

> 🔄 **状态更新 (2026-06-02)**: 真机 trace ([`../../analysis/x3c_realrobot_trace_20260601.md`](../../analysis/x3c_realrobot_trace_20260601.md)) + 官方对比 ([`../../analysis/xvla_vs_official_gap_rootcause.md`](../../analysis/xvla_vs_official_gap_rootcause.md)) 发现真机差**主因是 R1 (训练缺 ImageNet 归一化), 不是欠训**。**单纯加步数 = 在错误输入域上白练**。改走 **P0 重训** (`X3C_smooth800_p0`): 修 R1 + 对齐官方配方 (60k / lr1e-4 / warmup2000 / wd0 / ColorJitter), uc01 运行中。本 100K 纯延长计划**作废** — 若 P0 后仍欠训再议。配置见 rootcause.md §6.1。
>
> **(以下为原 100K 纯延长计划, 已被 P0 取代)**
>
> **动机**: §0.NEW.2.5 (a.2) 训练曲线显示 X3.C 30k **MAE 全程严格单调下降、final=best、无过拟合回弹** —— 末段仍在降 (24k→30k @1 还有 -4%)。30k step 仅 ≈ **2.1 epoch** (eff batch 64 × 30k / ~901k windows)。对比: pi05 smooth_800 跑 50k step (≈ 3.4 epoch 等效) 才 plateau。**X-VLA 0.9B 可能只是欠训, 而非架构上限**。延长到 100K (≈ 7 epoch) 验证: action fidelity 能否继续逼近 pi05 (§0.NEW.2.5 b 的 5× gap 是否部分来自欠训)。

**配置 (与 30k 版完全相同, 仅改 steps)**:

| 项 | 值 | 说明 |
|---|---|---|
| config | `X3C_smooth800_100k` (新增, 复制 `X3C_smooth800` 仅改 steps) | `train_scripts/xvla/launch/xvla_train.py` |
| 数据 | 仅 `A_new_smooth_800` (vis-only, domain_id=20) | 同 30k 版, 不变 |
| **steps** | **100_000** (原 30_000) | **唯一改动** |
| lr | 5e-5 | 同 (cosine schedule, decay 自动拉到 100k → lr 衰减更慢) |
| warmup_steps | 500 | 同 |
| freeze_steps | 1000 | 同 |
| batch_size_per_gpu | 8 (eff batch 64 @ 8 GPU) | 同 |
| vlm_lr_scale | 0.1 | 同 |
| ckpt save | 每 2000 step (与 30k 版一致, 共 50 个) | 便于画 100k 曲线 |
| 节点 | uc01 8× A800 (或空闲节点) | torchrun |
| output_dir | `/data/shared/ubuntu/local_ckpts/xvla_x3c_smooth800_100k` | 不覆盖 30k 版 |
| ETA | ≈ 15h (30k 用 ~4.5h, 线性外推 100k ≈ 15h) | 单节点 |
| epoch | ≈ 7.0 (100k × 64 / 901k) | 30k 版 ≈ 2.1 |

> ⚠️ **cosine schedule 注意**: launch script 用 `get_cosine_schedule_with_warmup(num_training_steps=steps)`。改 steps=100k 后 lr 从 5e-5 余弦衰减到 0 的长度自动拉到 100k → **后期 lr 比 30k 版同 step 处更高**, 这是延长训练的预期行为 (更慢退火)。不需额外改 lr。

**实施步骤**:

| Step | 内容 | ETA |
|---|---|---:|
| L1 | launch script 加 `X3C_smooth800_100k` dict (steps=100_000, 余同 X3C_smooth800) | 5min |
| L2 | uc01 起训 (torchrun 8 GPU, ckpt 每 2k) | ~15h |
| L3 | 全 ckpt MAE 曲线 eval (同 `eval_xvla_ee6d.py`, 同 1000 val 窗口, 与 30k 曲线对齐画图) | ~1h |
| L4 | 判定 (见下) | — |

**判定标准**:

| 100k 结果 | 结论 |
|---|---|
| @1 继续降 (如 30k 0.0149 → 100k ≤ 0.010) | ✅ **X-VLA 欠训确认**, 5× gap 部分是 step 数不足。后续 X3.A/B 也应跑 100k |
| @1 在 ~30-50k 后 plateau (≈ 0.014) | ❌ 30k 已够, gap 是架构/容量上限, 非欠训 → 加 step 无用, Track X 降权 (D3/D4) |
| @1 先降后 **回弹** (过拟合) | ⚠️ 901k windows / 7 epoch 触及过拟合点, 记录回弹起始 step 作为该数据集上限 |

> **为何只延 X3.C**: X3.C 是 vis-only 干净 baseline, 步数-性能关系最干净 (无跨域混合干扰)。先用它定位 "X-VLA 欠训 vs 架构上限", 再决定 X3.A/B 是否值得 100k (省算力)。

---

## §0.NEW.6 ⭐⭐ X3.C P0 重训 (修 R1 + 对齐官方) — 🔄 运行中 (2026-06-02)

> **由来**: 真机测试 X3.C 30k **任务失败** (走停/震荡/夹后松手)。逐层根因分析 ([`../../analysis/xvla_vs_official_gap_rootcause.md`](../../analysis/xvla_vs_official_gap_rootcause.md)) + 真机 trace 实证 ([`../../analysis/x3c_realrobot_trace_20260601.md`](../../analysis/x3c_realrobot_trace_20260601.md)) 定位 **主因 R1 = 训练全程缺 ImageNet 图像归一化** (lerobot XVLAPolicy.forward 不归一, 归一在被绕过的 processor; 我们 dataset 只 /255 → base ckpt 视觉前端输入域错位)。pi05 同数据真机 work (图像管线正确)。

### §0.NEW.6.1 实验设计 (单一主变量 = R1 归一化, 顺带对齐官方配方)

| 项 | 30k 旧版 (失败) | **P0 版 `X3C_smooth800_p0`** | 性质 |
|---|---|---|---|
| **图像归一化** | ❌ 只 /255 [0,1] | ✅ **ImageNet (img-mean)/std** | 🔴 **主改动 (修 R1)** |
| ColorJitter | ❌ | ✅ 0.2 (brightness/contrast/saturation) | 对齐官方 |
| steps | 30k | **60k** | 适配官方 50k 量级 (eff batch 64=官方4×, ≈4.3 epoch) |
| lr | 5e-5 | **1e-4** (VLM 1e-5 via scale 0.1) | 对齐官方 |
| warmup / freeze / wd | 500 / 1000 / 1e-4 | **2000 / 1000 / 0.0** | 对齐官方 |
| lr schedule / batch | cosine / 64 | cosine / 64 | 适配 (非照搬官方 constant/16) |
| vis 数据 | A_new_smooth_800 | 同 (不变) | 控制 |

**Batch 决策 (2026-06-02)**: 保持 **eff batch 64 (per_gpu 8)**, 不上 128。实测 batch64 时 GPU 已 **99-100% 利用 + 显存仅 38/80GB** → 上 128 不提速 (算力已饱和, 单步耗时翻倍 wall-time 不变)、偏离官方更远 (已 4×)、削弱泛化 (大 batch 减梯度噪声不利真机泛化)、小数据集 (901k windows) 过拟合风险↑。64 是单机 8 A800 对齐官方的合理上限。

### §0.NEW.6.2 实施 + 状态

| Step | 内容 | 状态 |
|---|---|---|
| P0.1 | 代码: `multi_domain_dataset.imagenet_normalize_chw` (train) + `serve_policy_xvla --imagenet_norm` (serve, 数值完全一致) + ColorJitter | ✅ commit `f8f7c79`+`0e0775b` |
| P0.2 | config `X3C_smooth800_p0` (官方对齐) | ✅ |
| P0.3 | uc01 卡死 → uc 回退 → **volc cnsh 8卡单节点重训 60k** (§0.NEW.6.6) | ✅ 训完 2026-06-04 (`xvla/ckpts/xvla_x3c_smooth800_p0/`) |
| P0.4 | offline eval (ckpt MAE 扫描, val=`A_new_smooth_800_xvla_val`) | ✅ **best=step_058000, @1=0.0341 (§0.NEW.6.7)** |
| P0.5 | ⭐ **真机测试 + 复测 trace 客观判据** (X3C_p0 终判) | ⏳ 待做 |

### §0.NEW.6.3 验收判据 — 真机 trace 客观指标 (对比 30k 基线)

| 指标 | 30k 基线 (失败) | P0 目标 |
|---|---|---|
| EE-L y/z 速度 lag1 自相关 | −0.34 / −0.35 (震荡) | → 接近 0 |
| EE 折返比 (路程/净位移) | 9.1× / 13.1× | → ~2× (smooth 数据级) |
| 关节方向反转率 | 0.45~0.69 | → ~0.1 |
| 夹爪开合切换 (200帧) | 16 | → 个位数 |
| 任务完成 | ❌ 需人干预 | → 自主折叠 |

### §0.NEW.6.4 判定 + 后续

| P0 真机结果 | 结论 + 行动 |
|---|---|
| ✅ 显著改善 (判据达标 + 任务自主完成) | **R1 坐实为主因**。推广: X3.A/B 也加归一化重训; 评估 Track X 真机价值 |
| ⚠️ 部分改善 (震荡减但仍不完成) | R1 是主因之一, 残余查 R2 (IK, EE↔joint 相关仅 0.47) / R3 (步数) |
| ❌ 无改善 | R1 假说弱化, 重审 R2 (IK 链) / R4 (架构 0.9B 天花板, §0.NEW.2.5b pi05 5× 优) |

> ⚠️ **使用铁律**: P0 ckpt 真机推理**必须** `--imagenet_norm` (train/serve parity); 旧 30k ckpt 用 `--no-imagenet_norm` 且勿再用现 eval (已归一化 → mismatch)。详见 rootcause.md §6.1。

### §0.NEW.6.5 数据集审计 + D5 候选 (P0 之后)

数据集对齐审计 ([`../../analysis/xvla_dataset_vs_official.md`](../../analysis/xvla_dataset_vs_official.md)): 我们 vis EE6D (`A_new_smooth_800`) vs 官方 X-VLA Agilex handler 逐项对比, 实测 806 ep:
- ✅ **全对齐无致命错**: action 20D EE6D absolute、xyz 米级、rot6d 正交归一 (col 模长 1.000/点积 0.000)、gripper 二值同阈值、loss scale (XYZ×500/ROT×10/grip×1)、无归一化 (靠 loss scale)、proprio=action[0]、action chunk = 真实未来轨迹 (标签语义正确, pi05 同数据 work 佐证)。
- 🟡 **D5 (唯一实质差异) — action chunk 时间窗口**: 我们连续 30 帧 = **1.0s** (33ms/点); 官方 qdur=2.0s 插值 30 点 = **2.0s** (67ms/点)。对叠衣慢长程任务, 短规划窗口可能加重长程走停 (但主因仍 R1)。
- **D5 候选** (P0 后): 若 P0 (R1) 真机仍长程走停, 改 `multi_domain_dataset` 采样为 2 秒窗口 linspace 插值 (对齐官方); 需训练+推理节奏一致, **不进 P0** (单变量), 留作独立实验。

### §0.NEW.6.6 ⭐ X3C_p0 迁 volc 8 卡提交 (2026-06-03) — uc 回退后执行路径

**由来**: uc01 上的 X3C_p0 重训历经 NFS I/O 争用卡死(用户 NFS 删除任务期间,全进程 D 态 `nfs_wait_bit_killable`,根因见 [`../../deployment/training_ops/submission/uc_cluster_jobs.md`](../../deployment/training_ops/submission/uc_cluster_jobs.md))→ 用户决定 **uc 集群回退、不再提交**。X3C_p0 改走 **volc 8 卡单节点**。数据(`A_new_smooth_800_xvla` 189M)+ 基座(`xvla_ckpts` 3.3G)已迁本地作备份单卡路径;volc 提交另需上目标 vePFS。

**8 卡单节点提交规格**:
| 项 | 值 |
|---|---|
| 集群 | ✅ **cnsh robot-task (A100)** — 8 GPU 单节点 (queue `q-20251204185107-fvnpx`, vepfs-cnsh, zone cn-shanghai-a, 无 SubPath) |
| 镜像 | X-VLA torch 镜像 `visincept-cn-shanghai.cr.volces.com/grasp/h2r:1.0`(cnsh 区,X-VLA Stage1 实测可跑) |
| Flavor | 单节点 8 GPU(`ml.hpcpni3ln.45xlarge` 类) |
| Framework | PyTorch(单节点 torchrun 用 `--standalone`,不依赖 MLP 多机 env) |
| 启动 | `torchrun --standalone --nproc_per_node=8 xvla_train.py --config X3C_smooth800_p0 --output_dir <vePFS>/xvla_x3c_smooth800_p0 --workers 4` |
| env | `XVLA_SB=<vePFS>/.../xvla/data/self_built` `XVLA_CKPT_INIT=<vePFS>/xvla_ckpts`(代码已 env 可覆盖) |
| 配置 | 不变(60k / eff batch 64 / lr1e-4 / imagenet_norm / ColorJitter,§0.NEW.6.1) |

**前置(数据/模型上 vePFS)**:
1. 数据 `A_new_smooth_800_xvla`(189M, 已改名,见 §0.NEW.1) → 目标 vePFS `.../xvla/data/self_built/`;
2. 基座 `xvla_ckpts`(3.3G) → 目标 vePFS;
3. vePFS 的 git checkout `pull` 到含 env-override + 数据改名 commit。

**⚠️ 注意**:
- output_dir 落 **vePFS 大盘**(非节点本地;X-VLA 每 2k 存 ~3.3G,60k≈100G);
- X-VLA ckpt **只存 model_state、无 optimizer、无 resume**(step 硬编码 0)→ 中断只能从头跑或加 warm-restart(`--init-from`,目前未实现)。volc 上要一气呵成;
- 真机推理务必 `--imagenet_norm`(train/serve parity,§0.NEW.6.4 铁律)。

**状态**: ✅ **训练完成 + offline eval done** (2026-06-04~05)。实际执行: 镜像用 `visincept-cn-shanghai.../grasp/h2r:1.0`(cnsh 已缓存,秒级部署);venv 从 uc 迁 cnsh vePFS(`xvla/X-VLA-env/.venv` + repoint cp3.10.20);bart-large tokenizer 离线缓存到 `xvla/assets/bart-large-tokenizer`(`XVLA_BART_TOK` env);YAML `train_scripts/kai/volc/xvla_x3c_p0_cnsh_8gpu.yaml`。60k step 训完(0.85 it/s ≈ 20h),每 2k 存 → `xvla/ckpts/xvla_x3c_smooth800_p0/`。

### §0.NEW.6.7 ⭐ X3C_p0 offline eval — best ckpt = step_058000 (2026-06-05)

**协议**: `eval_xvla_ee6d.py`(EE6D 20D action-chunk MAE),val = **`A_new_smooth_800_xvla_val`**(独立 26 held-out ep,domain_id=20,10 denoise steps,chunk=30)。先 7-ckpt 粗扫(300 win)→ 58k/final 精扫(1000 win)。

**ckpt MAE 收敛曲线**(300 窗口):

| step | MAE@1 | MAE@10 |
|---|---|---|
| 10k | 0.0607 | 0.0692 |
| 20k | 0.0522 | 0.0628 |
| 30k | 0.0426 | 0.0507 |
| 40k | 0.0349 | 0.0455 |
| 50k | 0.0326 | 0.0441 |
| **58k** | **0.0305** | **0.0422** |
| final (60k) | 0.0305 | 0.0423 |

→ **MAE@1 全程严格单调下降 0.0607→0.0305,无过拟合回弹,末段收敛**(50k→58k 仍 -6%,58k≈final)。与旧 30k 版同样 "final≈best" 的健康动力学。

**最佳 ckpt = `step_058000`**(1000 窗口精扫,各 horizon 微胜 final):

| | MAE@1 | MAE@10 | MAE@25 | MAE@30 |
|---|---|---|---|---|
| **step_058000** ✅ | **0.0341** | **0.0467** | **0.0687** | **0.0747** |
| step_final (60k) | 0.0342 | 0.0468 | 0.0689 | 0.0748 |

**最佳 ckpt 路径**:
```
/vePFS/tim/workspace/deepdive_kai0/xvla/ckpts/xvla_x3c_smooth800_p0/step_058000/state_dict.pt
```

> ⚠️ **口径注意**: 本次 val 用**独立的 `A_new_smooth_800_xvla_val`(26 ep)**, 与 §0.NEW.2.5 旧 30k 版用的 "A_new_smooth_800 训练集末 50 ep(idx≥756)slice" **不是同一个 val** → 两版 MAE **不可直接数值对比**(要可比须同 val 重跑)。
> ⚠️ **铁律**: P0 ckpt(已修 R1 imagenet_norm)真机推理**必须** `--imagenet_norm`(§0.NEW.6.4)。**offline MAE 低 ≠ 真机 work** —— X3.C 终判仍是真机测试(P0.5 §0.NEW.6.5,⏳ 待做)。

---

## §0.NEW.7 ⭐⭐ 数据隔离实验 — 官方 Soft-Fold vs 我们数据 (同 pipeline) — 规划 (2026-06-07)

> **目的**: 在 X-VLA-base 上,用**同参数 + 同(D5 修复后)pipeline + 同部署**,分别训**官方 Soft-Fold** 和**我们自己的数据**,**隔离"我们的数据集本身是否有问题"**。
> **前置已成立**: 数据/处理已用真实官方 HDF5 核验正确(EE 帧 0.1mm / gripper 米 / 相机 3×640×480 全对齐,见 [`../../analysis/xvla_dataset_vs_official.md`](../../analysis/xvla_dataset_vs_official.md) §5)。故本实验把矛头指向 **demonstration 数据内容本身**(质量/覆盖/多样性),而非帧/单位/管线。
> **数据已就位**: 官方 Soft-Fold 1532ep/441G 已迁到 gf0 `xvla/data/xvla_soft_fold`。

### 两实验(单变量 = 数据内容)
| | **Exp-O 官方数据** | **Exp-S 我们数据** |
|---|---|---|
| 数据 | 官方 Soft-Fold (1532ep, `xvla/data/xvla_soft_fold`) | 我们 A_new_smooth_800 (811ep) |
| init | X-VLA-base (`xvla/xvla_ckpts`) | 同 |
| 参数 | 60k / lr1e-4 / warmup2k / freeze1k / wd0 / bs64 / ImageNet 归一 / ColorJitter | **完全相同** |
| 动作表示 | **2s anchor (action_qdur=2.0, D5 修复)** | **2s anchor (D5 修复)** |
| 部署 | serve EE6D→IK link6;gripper 米值用**各自实测行程**(官方 open~0.062 / 我们 ~0.08,**不用 SoftFold 默认 -0.0055**) | 同 |
| config | `X3_official_softfold_d5anchor`(**待建**) | `X3C_smooth800_d5anchor`(**已建 + 已提交** task `t-20260607152340-4j7q5`,直接复用作 Exp-S) |

### 判据(真机为终判)
| 结果 | 结论 |
|---|---|
| Exp-O fold ✅ + Exp-S 不 fold ❌ | 🎯 **我们的数据本身有问题**(demonstration 质量/覆盖/多样性),非 pipeline/arch/容量 |
| 两者都 fold | 我们数据没问题;之前真机失败 = D5 表示(已修)/ 部署 gripper 映射 |
| 两者都不 fold | 我们 **pipeline 有 bug**(连官方数据都训不出)→ 回头查 pipeline |

### 实现要点
1. **同 loader(推荐,最干净单变量)**: 官方 Soft-Fold 是 hdf5(eef_6d/qpos/3 cam)。为走和 Exp-S **完全相同的 `LeRobotEE6DDataset`(+qdur=2.0)**,**把官方 hdf5 转成 lerobot parquet EE6D** `A_official_softfold_xvla`(直接取 `eef_6d`,或 `qpos→joint_to_ee6d`,二者已验等价 0.1mm)。这样两边同 loader + 同 anchor + 同 ImageNet,真正单变量。
   - 备选:给 `XVLAHdf5Dataset` 也加 `action_qdur`(现仅 LeRobotEE6DDataset 有),直接喂 hdf5 —— 省转换但需对齐两 loader 的图像/anchor 处理。**优先转 lerobot。**
2. **gripper 部署米值**: 官方 raw 0~0.062 / 我们 0~0.08 → 各自 serve 用各自 `--gripper_open/close_value`(避免又踩 SoftFold 默认 -0.0055 坑,见 §0.NEW.2.5(c2))。
3. **数据量差异**: 官方 1532ep vs 我们 811ep(属"数据"一部分,可接受);若要更严格,可下采样官方到 ~811ep 做同量对照。
4. **资源**: cnsh 8 A100(同 X3C),Exp-O 一个 job;Exp-S 复用已提交的 d5anchor。

### 执行 checklist
- [x] **Exp-O 数据**(2026-06-07,⚠️改走 hdf5-direct 而非 lerobot 转换):官方 hdf5 本地已就位(cnsh `xvla/data/xvla_soft_fold` 441G),给 `XVLAHdf5Dataset` 加 `action_qdur`(D5 2s anchor)+ `image_aug`(ColorJitter)对齐 `LeRobotEE6DDataset`;ee6d action cache 生成(1532ep/2.83M帧)。**省掉 441G→lerobot 转换**(无传输,直接喂 hdf5)。
- [x] 新 config **`X3_official_softfold_d5anchor`**(克隆 `X3C_smooth800_d5anchor`,type=hdf5 + softfold root,gripper 部署值留到 serve)。
- [x] 提交 **Exp-O**(cnsh 8 A100,60k,`t-20260607182015-2th5p`,Running);**Exp-S** = `t-20260607152340-4j7q5` ✅ **训完(60k)** + offline MAE 已扫(见下)。
- [ ] 两者 **offline 运动剖面** + **真机 fold** 对比 → 落判据(Exp-O 训完后同样扫 MAE 并排)。
- [ ] 回填结论到本节 + `xvla_vs_official_gap_rootcause.md` §7

### Exp-S offline MAE (2026-06-08) + 与既往 XVLA 对比
> `eval_xvla_ee6d.py`,val=`A_new_smooth_800_xvla` 末50ep(ep≥756)1000 strided 窗口,10 denoise,chunk30。**X-VLA 训练不产 MAE(只噪声 flow-matching loss),事后单独扫。**

| ckpt | MAE@1 | @10 | @25 | @30 |
|---|---|---|---|---|
| 50000 | 0.0393 | **0.0611** | **0.0918** | **0.0996** |
| 54000 | 0.0391 | 0.0614 | 0.0926 | 0.1003 |
| 58000 | 0.0388 | 0.0619 | 0.0934 | 0.1012 |
| **final (60k)** | **0.0387** | 0.0616 | 0.0929 | 0.1006 |

- **完全平台**(各 ckpt 差 ~1–2%):@1 单调微降到 final=0.0387;长 horizon 50k 略优。**最佳 ckpt ≈ `step_final`**(@1 最低),`/vePFS/tim/.../xvla/ckpts/xvla_x3c_smooth800_d5anchor/step_final/state_dict.pt`。

**对比既往 XVLA（同 eval 协议）:**
| | Exp-S（**D5 2s anchor**） | X3C_p0（**dense 1s**,§0.NEW.6.7) | Δ |
|---|---|---|---|
| @1 | 0.0387 | 0.0341 | **+13%** |
| @10 | 0.0616 | 0.0467 | +32% |
| @25 | 0.0929 | 0.0687 | +35% |
| @30 | 0.1006 | 0.0747 | **+35%** |

**分析(关键)**:Exp-S 的 offline MAE **每个 horizon 都高于 X3C_p0,且 horizon 越长差距越大(@1 +13% → @30 +35%)**。这**不是质量退步**,而是 **metric 失配 + 铁律"offline 反指"的教科书案例**:
1. **eval 用 dense 逐帧 GT**(`action_qdur=None`),但 Exp-S 是 **D5 模型**——预测的是 30 个 anchor 铺在 2s(60帧)上,不是 dense 连续 30 帧。anchor 输出 vs dense GT 逐帧比,**越往后(horizon↑)anchor 与 dense 偏离越大 → MAE 被系统性放大**,正好解释"差距随 horizon 张开"。
2. 而 **D5 anchor 恰恰是为真机修的**(X3.C 真机失败 → 动作表示根因 → 2s anchor),offline dense-MAE 看着更差但真机才是它要解决的。
3. 横向:两者都 **~4× 高于 pi05 smooth800**(joint MAE@1=0.0089 / FK-EE6D 0.0025),= R4 容量(0.9B vs 2.2B)+ R3 欠训的固有差距(P0 §0.NEW.6.5 已述),非本次变量。

**结论**:offline MAE **无法公平排序 D5-anchor vs dense 模型**(GT 标尺不同),Exp-S 的"更高 MAE"不可解读为更差。**真机对比 Exp-O(官方数据)vs Exp-S(我们数据)才是数据隔离实验的终判**;offline 仅确认 Exp-S 已收敛 + 选 ckpt(平台,取 final)。

---

## §0. 控制变量 X3 三件套 (2026-05-29, A_0423_0527) — ⬇️ 降为对照, 由 §0.NEW 取代

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
| A_new_smooth_800 20D EE6D (vis, **待转换**) | 811 ep / ~930K frames | T1 转换后落 `xvla/data/self_built/A_new_smooth_800/` |
| xvla_soft_fold action FK cache | 1542 files / 2.85M frames | 同 |

## 3. Prep ✅ 完成

| 项 | 状态 |
|---|---|
| HF ckpt `lerobot/xvla-base` (3.3GB) | ✅ uc01 NFS `/data/shared/ubuntu/workspace/xvla_ckpts/` |
| X-VLA env (lerobot + torch+cu121 + 全依赖) | ✅ uc01 NFS `/data/shared/ubuntu/workspace/X-VLA-env/.venv` |
| EE6D 转换 (kai/vis joint→EE6D 20D, PiperFK + Rot6D) | ✅ |
| XVLA-Soft-Fold action FK 缓存 | ✅ |
| Multi-domain dataset wrapper + DDP training script | ✅ |

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

- ⚠️ **D1 (域贡献)**: 原 vis_v2_merged "X3.B 完胜 X3.A" 结论已作废 (buggy 管线)。A_0423_0527 fit 排名 X3.C<X3.B<X3.A (§0.1, 但是 fit 非泛化)。**最终域贡献以 §0.NEW (A_new_smooth_800) 真机测试为准。**
- **D1.5 (X3.C eval 后)**: 量化 Stage A multi-domain pretrain 的价值. 若 X3.C ≈ X3.B, Stage A 是浪费; 若 X3.B < X3.C, Stage A 有效.
- **D2 (X3.B Stage B 后, 可选)**: vis B 真机评估 vs X-VLA SoftFold (同硬件) 100% baseline 对照
- **D3**: 若 X3.B 都打不过 baseline → Track X 主线降权, Track C (Action Head Cond) 提优先级 (但 Track C 已知 collapse, 见 `conditioning_vs_action_representation_ablation.md`)

## 8. 关联 paper ablation

(完整 Phase 3 ablation 设计见 [`cross_embodiment_strategy.md`](../../../deployment/strategy/cross_embodiment_strategy.md) §9 决策点 + §6 RTC/TAC 集成)

Phase 3 table 中:
- **X3.A** Track X (3-domain ⭐) — Florence2 + Soft Prompt, 全数据
- **X3.B** Track X (2-domain) — Florence2 + Soft Prompt, 无 XVLA
- 对照 **C3.0** Track C (Action Head Cond only) — 同 π0.5, 不同 conditioning 注入点

> ⚠️ **D4 (跨架构, 2026-06-01)**: §0.NEW.2.5 同标尺对比显示 **pi05 EE6D action fidelity ≈ X3.C XVLA 的 5×** (同 smooth_800 同 val 窗口)。X-VLA 路线在 offline action 保真度上明显不及成熟 pi0.5。Track X 的价值若存在, 须来自**真机鲁棒性 / 跨域泛化**而非 action fit (T5 真机终判)。若真机也不及 pi05 → Track X 主线降权 (D3)。
