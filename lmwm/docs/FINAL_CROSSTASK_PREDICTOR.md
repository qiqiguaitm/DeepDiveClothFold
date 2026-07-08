# 最优跨任务 milestone 预测器(交付)

> 目标:一个在多任务上联合训练的、最优的 milestone+1 预测器。以下是**实验收敛出的最优配置 + 交付模型 + 依据 + 用法**。
> 日期 2026-07-07。模型:`lmwm/checkpoints/final_crosstask_3task.pt`(kai0+coffee+xvla)、`final_crosstask_4task.pt`(+vis,训练中)。

---

## 1. 最优配置(实验收敛结论)

| 组件 | 最优选择 | 依据 |
|---|---|---|
| **teacher** | **proto(簇中心)** ← 更新 | 与 inv 打平(deploy .7532 vs .754,id3 .710 vs .702),但**去掉 InverseEnc + 去掉 CE 锚、码=连续中心=开放词表、轻 5.9M**;xvla 身份反升(见 §3b) |
| (teacher 备选) | inv(逆向动力学 + 蒸馏) | 与 proto 同级;去掉任何 teacher(none)deploy 掉 0.07–0.13 |
| **锚 anchor** | proto 下**无需锚**(码即中心=身份);inv 下用 union_ce | in-dist union_ce 最强(见 §3a) |
| 编码器 | π0.5 SigLIP 同塔(冻结,共享归一化) | 与策略同空间 |
| 预测器 Predictor | 逆向 teacher 出码 z → MDN(K=4)部署 | — |
| 生成器 Generator | AdaLN(当前 grid 画布,zero-gate) | 已有定论 |
| 训练 | 4 任务并集、状态条件(无语言)、per-task 各自 milestone、全局 id 偏移 | — |

**一句话**:`train_multitask.py --datasets kai0,coffee,xvla[,vis] --anchor union_ce --teacher inv`。

---

## 2. 交付模型指标(12k step)

> **推荐交付 = proto teacher**:`teach_proto_3task.pt`(135MB)/ `teach_proto_4task.pt`(135MB)。与下方 inv 版打平但更简/更轻/更开放词表(见 §3b)。以下 inv 版作为等效对照保留。

**inv 版 3 任务** `final_crosstask_3task.pt`:
| task(簇数) | deploy | id_top3 |
|---|---|---|
| kai0 叠衣(37) | 0.699 | 0.483 |
| coffee 咖啡(15) | 0.788 | 0.985 |
| xvla 叠衣变体(47) | 0.774 | 0.638 |
| **mean** | **0.754** | **0.702** |

**备选 = 4 任务(+vis)** `final_crosstask_4task.pt`:mean deploy **0.786**(vis 简单拉高),但 mean id3 降到 0.563 —— vis id3 仅 0.11(与 kai0 冗余同为叠衣 + bank 质量偏低,主要拖累身份)。**推荐 3 任务版**:三个真正不同的任务、每个 id3 都干净(≥.48);vis 换来"名义 4 任务"但身份下降。两个 ckpt 都在 `lmwm/checkpoints/`,按需选。

---

## 3. 支撑证据(全实验链)

**in-dist 联合训练(所有任务都训过)—— union_ce 最强**
| 规模 | union_ce | progress(标量连续) | progress_id(连续+身份) |
|---|---|---|---|
| 2 任务(52类) deploy | **0.7525** | 0.7418 | 0.7523 |
| 3 任务(99类) deploy | **0.770** | 0.758 | 0.725 |

**teacher 有效性(3 任务,deploy)—— inv 明显优于 none**
| anchor | teacher=inv | teacher=none | Δ |
|---|---|---|---|
| union_ce | **0.770** | 0.698 | +0.072 |
| progress | 0.758 | 0.630 | +0.128 |

### 3b. teacher = proto(簇中心)vs inv —— 打平但 proto 更优(推荐)
| | inv | **proto** | 说明 |
|---|---|---|---|
| 3任务 mean deploy / id3 | 0.754 / 0.702 | **0.7532 / 0.7097** | 打平(噪声内) |
| 3任务 xvla id3 | 0.638 | **0.679** | 最难身份反升 +0.04 |
| 4任务 mean deploy / id3 | 0.786 / 0.563 | 0.783 / 0.546 | 打平 |
| ckpt / 结构 | 158MB,含 InverseEnc + CE 锚 | **135MB,无 InverseEnc、无 CE 锚**(码=连续中心=身份=开放词表) | proto 更简/更轻/更跨任务 |
→ **proto 同性能、更简洁、天然开放词表(对齐 LOO 连续锚在 unseen 身份反超)→ 采纳为最优。** teacher 码 = 下一 milestone SigLIP 中心的固定投影;predm 蒸馏它;生成器把它渲染到当前画布。

**open-vocab / LOO(留一任务当 unseen,train 其余)—— 连续锚在身份上小胜,但整体弱**
| unseen | union_ce id3 | progress id3 | deploy vs persist |
|---|---|---|---|
| kai0 | 0.210 | 0.191(pid 0.237) | ≈persist(弱) |
| coffee | 0.323 | **0.353** | <persist(负) |
| xvla | 0.304 | **0.344** | <persist(负) |
→ deploy 走生成器不过头,open-vocab 上所有锚都弱;**身份 id3 上连续锚稳定小胜**,但没有锚能做强零样本跨任务。

---

## 4. 诚实边界 + 何时该换

- **本模型 = "训练时见过的任务"上的最优**(现实场景:用你全部任务数据联合训)。**union_ce 在 ≤~100 类完全够用且最强。**
- **词表随任务数线性增长**(每任务 +~30 类):100 任务≈3000 类。到**远大规模**或**要零样本上没见过的新任务**时,union 头才成瓶颈 —— 那时应换**连续价值锚 / 共享 codebook**(LOO 已显示连续在 unseen 身份上更好)。当前规模不需要。
- **未验证**:下游 SR(世界模型是否帮策略)—— 这是唯一没测、也是最终裁决价值的量。锚/teacher 的选择对 deploy 只有 ~0.01–0.13 级影响,**接 π0.5 测 action-MAE/SR 才是下一步的重心**。

---

## 5. 加载 / 使用

```python
import torch
d = torch.load("lmwm/checkpoints/final_crosstask_3task.pt", map_location="cpu", weights_only=False)
# d: fwd/predm/inv/anchor_head state_dicts + idproj/gmu/gsd/din/code_dim/K/anchor/teacher/total_M/tasks
from train_twomodel_v2 import MilestoneGenerator, MilestonePredictor
from train_lawm_patch import InverseEnc
fwd  = MilestoneGenerator(d["din"], d["code_dim"]); fwd.load_state_dict(d["fwd"]); fwd.eval()
predm= MilestonePredictor(d["din"], d["code_dim"], d["K"]); predm.load_state_dict(d["predm"]); predm.eval()
# 部署:SigLIP grid G_t (norm 用 d["gmu"]/d["gsd"]) → gist → ẑ=predm.deploy_mean(gist) → Ĝ=fwd(G_t, ẑ)
```

- 训练/评测:`lmwm/scripts/train_multitask.py`(`--anchor/--teacher/--heldout/--val_cap/--save_ckpt`)。
- 数据 bank:`temp/{kai0→crave_full_dinov3h,coffee_dinov3h,xvla_dinov3h,vis_dinov3h}` + `lmwm/data/recurrence_graphs/*`。
- bank 构建:`make_visbase_dinov3h_index.py`(通用)、xreb 缓存捷径(coffee/xvla)。
