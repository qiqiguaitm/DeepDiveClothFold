# Track X — X-VLA 官方架构 Native 训练 (X3.A + X3.B + X3.C)

> **状态**: 🔄 **原版 X3.A/B/C (vis_v2_merged) 全部作废** (buggy 管线 + 未控制变量) → 2026-05-29 控制变量三件套 (A_0423_0527 vis) → **2026-05-31 再换 vis = `A_new_smooth_800` (811 ep, X1 cleaned, 真机已验证 work) 重训**, 见 §0.NEW (⭐ 当前)。§0/§0.1 (A_0423_0527) 降为对照; 原 vis_v2_merged 版 (含旧 §主要结论 + §4/§5/§5.5) 已删除。
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

**Caveat (诚实标注)**:
1. **action-fit 保真度, 非真机成功率** — 两者 held-out 均来自训练分布 (fit 非泛化)。MAE 低 ≠ 真机一定好, 但证明 pi05 动作精度远高于 X3.C。
2. **FK 对 pi05 中性偏不利** — pi05 在 joint 空间训练, FK 转 EE6D 会**放大** joint 误差 (几何链式传导), 按理 pi05 应吃亏; 它仍大幅领先 → 真实优势比表面数字更大。
3. **差距主因 = 架构/预训练** — 同 smooth_800 同 vis 域; 差异来自 pi0.5 (成熟 flow-matching VLA) vs X-VLA-base (0.9B lerobot port)。
4. 唯一不严格处: pi05 horizon=50 取前 30 对齐 X3.C chunk=30 (已处理); gripper 量纲在 FK 后两者均二值化, 公平。

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
