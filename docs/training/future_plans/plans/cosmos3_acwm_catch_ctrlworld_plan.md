# Cosmos3 AC-WM 追平 Ctrl-World 方案(叠衣服)

> 目标:把 Cosmos3-Nano FD 世界模型从"动作被忽略(ΔPSNR≈0)"提升到 **Ctrl-World 级别的动作可控性**,在同一份 wam_fold_v3 数据上。
> 立项 2026-06-26。前置诊断见 [[cosmos3-acwm-fold-v3-plan]];与 `ctrl_wm/`(SVD,已验证有效)同口径对照。

---

## 0. 核心判断(为什么追不上 + 怎么追)

诊断结论:**能否学会动作跟随 ≈ 归纳偏置 × 信号强度,要过阈值。** Cosmos3 joint-attn **偏置弱**,叠衣 **信号弱**(刚体性低/运动占比小/单任务),我们的配置(30fps 细分辨率 / 单条件帧 / 高-σ 加权)又把信号压到阈值下。Ctrl-World 用**强偏置 + 强信号设计**把同样的弱信号抠了出来。

**策略 = 把 Ctrl-World 五个制胜轴逐个移植到 Cosmos3,按"信号→偏置→架构"分层,便宜的先上、A/B 验证后再放大。** 不追求一步到位,追求**每层都用 IDM/ΔPSNR 量到增益**。

| Ctrl-World 制胜轴 | Cosmos3 现状 | 对应改动(层) |
|---|---|---|
| ~5–10hz 预测、帧间大运动 | 30fps、帧间近静止 | L1 抽帧/随机 skip |
| 只预测 5 帧(4 未来步)+ AR | 一次预测 32 帧 | L1 缩短窗 + AR 评测 |
| 6 帧历史条件 | 1 条件帧 | L2 多历史(原生支持) |
| 低-σ/细节加权 | shift=5 高-σ | L1 改 σ 加权 |
| cross-attn 特权 + 逐帧动作 | 动作 ~32 token 混进 ~19500(600:1) | L3 强化动作通路 |
| 外观/运动解耦(channel-wise 当前帧) | 外观运动揉一起 | L3 外观锚 |
| DROID 95k 预训练→finetune | base 多机型动作训练(主干继承,16/17 头是空槽) | L4 暖启动/运动预训练 |

---

## 1. 成功判据(可测,对标 Ctrl-World)

先把"追上"量化,且**先升级评测**(L0),否则无法判断每层是否有效。

| 指标 | 当前 iter_3000 | 目标(追平 Ctrl-World) |
|---|---|---|
| **ΔPSNR(GT−错误动作)** 运动加权 | +0.079(WEAK) | **> 1.0 dB** |
| **IDM-MAE**(从生成视频反推动作 vs 注入) | 未测 | 显著低于"零动作/视频"基线 |
| **AR rollout 可控性**(policy/keyboard 在环,长程) | 未测 | 指令跟随可见、不崩 |
| 视频保真 PSNR/SSIM/LPIPS(分机型) | GT 17.5 | 不退化 |
| 闭环对齐 Pearson r(终极) | — | ≥ 0.8 |

**关键:所有指标先在 Ctrl-World ckpt 上同口径跑一遍,作为"目标线"。**

---

## L0. 评测升级(最先做,1–2 天,本机)

无论走哪层,这些都得有,否则盲跑。

1. **IDM-MAE 探针** —— `mode="inverse_dynamics"`:从生成视频反推动作,与注入动作算 MAE。直接量"动作有没有进画面",绕开 PSNR 弱感知。数据类已支持该 mode;eval 加一个分支。
2. **运动加权 ΔPSNR** —— 按帧间运动量(光流/像素差)加权,或只在高运动窗口算。破解"静止窗口稀释"假阴性。
3. **AR rollout 评测** —— 自回归拼到 10–30s,replay GT 动作 / 扰动动作,逐相机退化曲线(对标 Ctrl-World 的 replay/keyboard rollout)。
4. **分机型**(visrobot vs kairobot)+ **Ctrl-World 目标线**同口径。

落点:`wam_fold_wm/eval/fd_infer.py`(加 IDM 分支 + 运动加权)、新增 `eval/ar_rollout_v3.py`、`eval/ctrlworld_baseline.py`。

---

## L1. 信号增强(便宜、数据/配置,最高 ROI)—— 先打这层

全部是数据/配置级,单节点 8 卡 2–3k 步即可 A/B。

| 改动 | 旋钮 / 位置 | 默认→新 |
|---|---|---|
| **① 抽帧到 ~10hz** | `WamFoldLeRobotDataset` 新增 `frame_stride`;窗口/动作/timestamp/cache-key 按 stride 取 | stride 1→**3** |
| **② 随机 skip 增强** | 同上,`frame_stride` 每样本随机 ∈{2,3,4} | 固定→**随机** |
| **③ 缩短预测窗** | `chunk_length`(须 4k:VAE 4k+1) | 32→**16**(17 帧=4·4+1→5 latent,预测 4 未来步,贴 Ctrl-World) |
| **④ 低-σ 加权** | `rectified_flow_training_config.shift` 480、`train_time_video_distribution` | 5→**1~2**、'waver'→偏低噪 |
| **⑤ 运动窗口筛选** | 用现成 `idle_frames` 丢/降采样近静止窗口 | 全收→**滤静止** |

> ⚠️ 工程约束:抽帧在**送 Wan VAE 之前**对源帧做(取每 stride 帧凑够 4k+1 个采样),不改 VAE。动作在抽样位置取 delta(每步运动放大 ~stride 倍 → 更可见)。cache-key 加 `_k{stride}` 避免与现有 latent 撞键。

**预期**:① 是与 Ctrl-World 最大且可控的差异,**最可能单独见效**;②③④⑤ 叠加。L1 跑完用 L0 评测,看 ΔPSNR/IDM 是否动。

---

## L2. 偏置增强(中等、训练框架)—— L1 不够再叠

| 改动 | 旋钮 / 位置 | 说明 |
|---|---|---|
| **⑥ K 帧历史条件** | `build_sequence_plan_from_mode` FD 分支 `[0]`→`range(K)`;`num_history_vision` 透传(opt-in,默认 1=现状) | **已验证原生支持、权重兼容、不破坏预训练**(IDM 模式本就多帧 clean);评测侧放开 `omni_mot_model.py:2562` 的 `0:1` 硬编码 |
| **⑦ 联合 IDM 损失** | `mode="forward_dynamics"`→`"joint"`、`lambda_action`/`action_loss_weight`>0 | 强制"视频可反推动作"→ 逼动作 token 携带信息。**逼模型用动作最硬的一招** |
| **⑧ 提高 action CFG** | `ActionDataPacker.cfg_dropout_rate` 0.1→0.2 + 推理 guidance↑ | 仅与①⑥⑦叠加才有意义(放大已有依赖,造不出依赖) |

---

## L3. 架构(贵、风险高)—— 仅当 L1+L2 仍不达标

| 改动 | 说明 |
|---|---|
| **⑨ 强化动作注入** | 给动作更多 token / 专门的动作 cross-attn 分支 / FiLM 调制,打破 600:1 稀释。改 `cosmos3_vfm_network.py` |
| **⑩ 外观/运动解耦** | Ctrl-World 式:当前帧 channel-wise 喂每一帧供外观,使 loss 只能靠运动降 → 逼动作。改 packer/model 条件构造 |

---

## L4. 初始化/预训练(按需)

| 改动 | 说明 |
|---|---|
| **⑪ 暖启动动作头** | 不 fresh-init;从训练过的**双臂域** `robomind-franka-dual`(dom12)拷 action2llm/llm2action 到 16/17(语义不完全匹配,hack,低优先) |
| **⑫ 运动丰富预训练** | 先在 DROID-like 高运动多任务上 FD 后训练 → 再叠衣 finetune(复刻 Ctrl-World 配方,贵) |

---

## 2. 执行节奏 + 决策门

**纪律:每层单节点便宜 A/B(2–3k 步,改一个/一组旋钮),L0 评测过门后才放大到 5n8g。当前 M1(fresh-init/32帧/高-σ)是对照组。**

```
Sprint 0 (1-2d): L0 评测升级 + 在 Ctrl-World ckpt 上跑出目标线
Sprint 1 (2-3d): L1-① 抽帧10hz+随机skip (单变量A/B)  ── 门:ΔPSNR/IDM 是否动?
Sprint 2 (2-3d): L1 全量(①②③④⑤)              ── 门:是否接近目标线一半?
Sprint 3 (3-4d): + L2-⑥⑦ (历史+联合IDM损失)      ── 门:ΔPSNR>1 或 IDM 明显改善?
   ├─ 达标 → 放大 5n8g 收敛 + AR/闭环评测
   └─ 不达标 → L3 架构(⑨⑩) 或 走战略 fallback
```

**决策树**:L1 见效 → 大概率是"信号"主导,继续叠 L2 即可;L1 完全不动 → "偏置"主导,优先 L2-⑦(联合 IDM 损失)/ L3-⑨;两者都不动 → 接受"Cosmos3 通用机制在此弱信号任务上需要 L3 级架构改动",评估性价比。

---

## 3. 战略 fallback(诚实)

- **若目标是"现在要一个能用的叠衣 AC-WM"**:Ctrl-World 在同数据已验证有效,**直接用/继续 finetune 它**最省;Cosmos3 这条按 L1+L2 推进,定位为"验证通用骨干能否被补强"的研究线。
- **若目标是"Cosmos3-native 统一世界模型"**:L1+L2 是必经的性价比最高路径;L3 只在确有战略价值时投入。
- 任何一层都**不破坏当前 M1**(全部 opt-in / 新配置),M1 继续跑作对照与"纯 FD 基线"。

---

## 4. 立即可做(本 sprint)
1. L0:`fd_infer.py` 加 IDM-MAE + 运动加权 ΔPSNR;在 Ctrl-World ckpt 上出目标线。
2. L1-①②③:`WamFoldLeRobotDataset` 加 `frame_stride`(随机)+ `chunk_length=16`;新配置 `wam_fold_wm_nano_t1.py`;单节点 smoke→2–3k 步 A/B。
3. 用 L0 评测对照当前 M1(iter_3000 ΔPSNR=+0.079)与 Ctrl-World 目标线,决定是否叠 L2。

> 关联:[[cosmos3-acwm-fold-v3-plan]](v3 主线/M1 对照)、[[cosmos3-wam-fold-world-model-plan]](v1 起点)、`ctrl_wm/`(目标参照)。
</content>
