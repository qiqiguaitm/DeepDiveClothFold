# LMWM 最终报告:子目标预测器(milestone prediction)

面向 kai0 π0.5 VLA 的冻结 world-model 子目标模块。本文汇总最终网络结构、各指标含义(本科生级讲解)、
与 LaWM(arXiv 2606.15768)的对比,以及 V1→V3.1 目标构建的演进结论。

---

## 1. 各指标是什么、有什么用(本科生级)

我们预测的是"未来某个 milestone 的 DINOv3 特征网格"。用一堆指标从不同角度衡量"预测得好不好":

| 指标 | 定义(直白版) | 作用 / 表明什么 |
|---|---|---|
| **grid-cos** | 预测的特征网格 与 真实目标网格 的**余弦相似度**(1=完全一致,0=无关) | 核心保真度。越高=预测越像真实未来。 |
| **oracle grid-cos** | 用"作弊码"(inverse 偷看了真实未来算出的转移码)重建目标的 cos | **天花板**:即使知道真实动作,forward 解码器能重建到多好。衡量"模型容量/表达上限"。 |
| **deploy grid-cos** | 真实部署:只看当前帧、用预测码(predm)重建目标的 cos | **真实可用性**:实战中(没有未来)能做多好。这是我们最关心的数。 |
| **persistence** | 直接把"当前帧"当作预测(不动)的 cos | **平庸基线**:"啥也不预测"能得多少分。用来判断预测器有没有真本事。 |
| **lift = deploy − persistence** | 预测器比"不动"多赚了多少 | **真实技巧**:扣掉"目标本来就离当前近"的便宜,看模型净贡献。比绝对 deploy 更公平。 |
| **model lag(时间滞后)** | 把预测的网格匹配到最像的真实帧,那帧比当前晚多少秒 | 模型"敢预测多远"。越接近目标 lag=越敢 commit。 |
| **dataset lag** | 真实目标帧比当前晚多少秒 | 目标本身的时间跨度(horizon)。 |
| **ratio = model/dataset lag** | 模型达成度 | <1 表示**欠射**(模型不敢预测那么远,退回当前附近)。越接近 1 越好。 |
| **best-of-8** | VAE 采样 8 个预测,取和真实最像的那个的 cos | 衡量**未来是不是多模态**:若 best-of-8 ≫ 均值,说明未来有多个分支、采样能覆盖到。 |
| **corr(value, time)** | CRAVE 给每帧的 progress 值 与 归一化时间 的相关系数 | CRAVE 进度标签质量。0.94=进度标签几乎和真实时间同步上升,很可靠。 |
| **monotonicity** | 预测/进度序列中"非递减"的比例 | 进度是否单调不倒退。0.98=基本只前进。 |

**一句话**:oracle=上限,deploy=实战,persistence=白给基线,lift=净技巧,lag/ratio=敢不敢预测远,best-of-8=未来多不多模态。

---

## 2. 最终网络结构

```
                    ┌─ 冻结编码器 DINOv3-H (ViT-H+/16, ~840M) ─┐
   256²RGB 帧 ──────┤  256²→ 16×16×1280 patch grid            │  (纯torch standalone loader,
                    └──────────────────────────────────────────┘   任何 env 可用,cosine 0.9999)

   子目标预测器(forward-from-current,冻结后喂 VLA):
     inverse(g_t, g_target) ──► 转移码 z        [teacher, 训练用, CNN 2层, 6.5M]
     predm(g_t)             ──► 预测码 ẑ         [deploy, CNN, 3.6M]  (+ VAE 头 kl≈1e-2, 可选)
     forward(g_t, code)     ──► ĝ_target        [CNN 3层, 17M, 空间基底=当前 grid]
     部署推理 predm+forward ≈ 21M  (inverse 仅训练)

   目标构建 = V3.1 (milestone_viterbi):
     CRAVE 37 milestone + Viterbi 单调分配 → 每 stage 干净 medoid
     目标 = 进度前进的下一个 milestone 的 medoid (时间前向 + 进度单调 + 跨集一致)

   patch 解码器(仅可视化,不喂 VLA): make_decoder big + GDL, 5M, L1 0.0206 / sharp 574
```

**注入 VLA(π0.5)**:obs_grid(256×1280) + 子目标 û_T(256×1280) + 转移码 ẑ(64/128) → action expert,
全 stop-grad + KI 梯度隔离 + 蒸馏 ‖ẑ−z‖²(对齐 LaWM 接口)。

---

## 3. 与 LaWM 的对比

**LaWM(官方 ckpt 在我们数据上实测)** vs **我们最终 V3.1**:

### 3a. 平均未来预测(核心指标)
| | LaWM | 我们 V3.1 |
|---|---|---|
| 目标 horizon | **固定物理时间 1.6s** | **进度前进一个 milestone**(时间~2.4s 变长但进度固定) |
| oracle grid-cos(我们数据) | **0.770**(kai0, ViT-B 空间) | **0.789**(ViT-H+ 空间) |
| deploy grid-cos | 需策略预测码(未单独测) | **0.694** |
| lift over persistence | +0.143 | +0.128 |
| ⚠️ 可比性 | 编码器空间不同(ViT-B 768 vs ViT-H+ 1280)+ 目标定义不同 → **绝对数不能硬比,趋势可比** |

**结论**:同数据上 LaWM 230M transformer **没有碾压**我们 ~27M CNN(lift 相当)。我们的**进度目标(V3.1)是 LaWM 没有的设计**——LaWM 只有固定时间 horizon,我们额外做了"跨 episode 进度一致"的目标。

### 3b. 其他指标
| 维度 | LaWM | 我们 |
|---|---|---|
| 世界模型参数 | 230M(24enc/12dec transformer)| **~27M CNN**(轻 8×)|
| 编码器 | DINOv3 ViT-B/16 ~86M | DINOv3 ViT-H+/16 ~840M(强 10×)|
| 潜在动作码 | 32-d **VAE** | 64/128,det 或 VAE |
| horizon 类型 | 固定时间 | 固定时间(near-future)/ 事件(milestone)/ **进度(V3.1,新)** |
| 下游 SR(extrinsic) | LIBERO 98.6% / RoboTwin 91% / 真机 90% | **未接 VLA 测 SR(最大缺口)** |
| 跨数据集泛化 | 定性(3000h 多源数据) | **定量**(vis_base cos 0.68 零微调)|

**最诚实的差距**:LaWM 有**下游成功率(SR)**,我们只有 intrinsic 预测指标——接 VLA 测 SR 是必须补的实验。

---

## 4. 目标构建演进(V1→V3.1)的核心结论

| 版本 | 目标 | deploy | lift | 欠射ratio | 时间一致 | 跨集一致 | 单调 |
|---|---|---|---|---|---|---|---|
| V1 temporal-next | 时序下一段 | 0.726 | 0.108 | 0.29 | 前向但欠射 | ❌ | ❌ |
| V2 progress-argmax | value 更高 milestone | 0.594 | 0.136 | — | ❌错乱(负滞后)| ✅ | ✅ |
| V3 progress-Δ | 连续进度+Δ | 0.62 | 0.20 | — | 前向但远(6s)| ✅ | ✅ |
| **V3.1 milestone-viterbi** | **Viterbi 进度下一段** | 0.694 | 0.128 | **0.347** | ✅**前向** | ✅ | ✅ |
| **V3.1+VAE(最终定版)** | 同上+VAE(kl1e-2)头 | **0.700**(best8 **0.705**)| 0.135 | 0.34 | ✅前向 | ✅ | ✅ |

**最终定版 = V3.1 + VAE(kl 1e-2)**:deploy 0.700 / best-of-8 0.705(比 V3.1-det 微升 +0.006/+0.011),VAE 头对齐 LaWM 潜在动作接口。欠射 ratio 0.34(VAE 不改善欠射,印证 gap 是本质信息损失)。

**结论**:**V3.1 = 最佳数据集构建**——用 CRAVE Viterbi 单调分配,唯一同时满足"时间前向+跨集一致+进度单调+干净标准 medoid"。模型欠射从 V1 的 0.29 改善到 0.35(预测 lag 0.28s→0.845s,敢 commit 3× 远)。

**三大天花板(实测)**:①容量(transformer)无用;②多模态(VAE best-of-8)只回收 +0.02;③感知端(多视角+时序)只 +0.005。oracle→deploy 的 gap 大部分是**"从当前预测未来"的真实信息损失**,是硬的。

---

## 5. 产物索引
- 预测器:`optimize_subgoal.py`(--mode {nearfuture, milestone, milestone_value, progress_delta, milestone_viterbi}, --code_head vae)
- 目标术语:`milestone_target_terminology.md`
- 诊断:`measure_milestone_lag.py`(欠射)、`crave_dataset_effect.py`(CRAVE 处理效果)
- 官方 LaWM 评测:`eval_lawm_lam.py` + vendored `lmwm/vendor/LaWAM`
- 渲染:`render_milestone_predict_video.py`(--viterbi, --raw_video 跨数据集)
- 架构:`lmwm_architecture_current.md`
