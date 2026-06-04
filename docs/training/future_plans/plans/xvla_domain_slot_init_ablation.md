# XVLA domain 槽位 warm-init ablation — vis 借用上游已训 agilex 域先验

> **目的**: 验证把 vis 的 domain 槽 (domain_id=20, 在 xvla-base 里是**冷启动随机**) 用一个**已预训练的 agilex 域槽** (AIR-AGILEX=10 / AIR-AGILEX-HQ=5 / robomind-agilex=16) 做 warm-init, 能否在小数据上**加速收敛 / 提升 vis 真机表现**。
> **状态**: 📋 规划 (2026-06-04)
> **来源**: [`../../../deployment/inference/xvla_upstream_vs_local_consistency.md`](../../../deployment/inference/xvla_upstream_vs_local_consistency.md) §2 I3 (上游 vs 本地一致性分析)
> **关联**:
> - Track X curriculum: [`xvla_track_x_curriculum.md`](xvla_track_x_curriculum.md)
> - 部署 bring-up: [`../../../deployment/inference/xvla_inference_bringup.md`](../../../deployment/inference/xvla_inference_bringup.md)
> - 上游 domain 表: `DeepDive-XVLA/datasets/domain_config.py` (本地 clone `/vePFS/tim/workspace/DeepDive-XVLA`)

---

## 0. 背景与动机 (I3)

X-VLA 用 **domain-conditioned 参数** 区分本体: 每个 `domain_id` 索引一组专属 soft-prompt + per-domain 线性层权重。

- **上游 xvla-base** 预训练时, agilex 双臂落在**已训练**的槽位:
  - `AIR-AGILEX = 10`, `AIR-AGILEX-HQ = 5`, `robomind-agilex = 16` (`domain_config.py`)。
- **本地** (`xvla_train.py:37-39`) 给 kai=19 / **vis=20** / xvla_sf=21 —— 这些槽在 base 里**从未被训练** (`soft_prompt_hub` 是 `nn.init.normal_(std=0.02)` 冷启, soft_transformer.py:345)。
- vis 是部署目标且**数据少** (A_0423_0527 ~1085 ep)。让它的 domain 先验从零学 = 浪费了 base 里现成的 agilex 物理先验。

> **假说 H**: vis = 双臂 AgileX/Piper 叠衣, 与上游 AIR-AGILEX 域**本体高度同构**。用 agilex 槽 warm-init vis 槽 → 借 pretrain 先验 → 收敛更快、小数据泛化更好、真机更稳。

---

## 1. 机制 — domain 身份的 5 个载体 (已从 ckpt 实测确认)

本地 lerobot XVLAPolicy 里, **所有** `shape[0]==num_domains(30)` 的参数 (= 按 domain_id 索引的行) 共 **5 个**:

| 参数名 | shape | 含义 |
|---|---|---|
| `model.transformer.soft_prompt_hub.weight` | (30, 32768) | 32 soft-prompt × 1024 hidden / 域 |
| `model.transformer.action_encoder.fc.weight` | (30, 73728) | DomainLinear 输入投影权重 / 域 |
| `model.transformer.action_encoder.bias.weight` | (30, 1024) | 同上 bias |
| `model.transformer.action_decoder.fc.weight` | (30, 20480) | DomainLinear 输出投影权重 / 域 |
| `model.transformer.action_decoder.bias.weight` | (30, 20) | 同上 bias |

> ✅ 这 5 个都是真 per-domain (按 `domain_id` 取行)。`num_actions` 也=30 但不与这些冲突 (action 相关参数 shape 不是 `[30, …]` 按域索引)。**warm-init = 对这 5 个参数做行拷贝, 别的不碰。**

---

## 2. Warm-init 配方 (state_dict surgery, 不改模型代码)

对 **base ckpt** 做一次性手术: 把源 agilex 槽的行拷到 vis(20) 槽, 存成新的 warm-init base, 训练 `weight_loader` 指向它。

```python
# build_xvla_warm_init_base.py  (在 .venv_xvla / X-VLA-env 跑)
import torch
SRC = 5      # AIR-AGILEX-HQ (主选; 备选 10=AIR-AGILEX / 16=robomind-agilex)
DST_VIS = 20
DST_KAI = 19      # 可选: 也 warm kai
NUM_DOMAINS = 30
WARM_KAI = False  # A1 只 warm vis; A2 同时 warm kai

base = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
ms = base["model_state"] if "model_state" in base else base
for k, v in ms.items():
    if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == NUM_DOMAINS:
        v[DST_VIS] = v[SRC].clone()
        if WARM_KAI:
            v[DST_KAI] = v[SRC].clone()
torch.save(base, WARM_BASE_CKPT)   # 同结构, 仅改了 5 个参数的 row 20(/19)
```

- **源槽选择**: 主选 **AIR-AGILEX-HQ=5** (HQ=高质量, 双臂 AgileX 最匹配 vis); 子 ablation 扫 {5, 10, 16}。
- ⚠️ **必须在 base 上做** (base 的 5/10/16 是 pretrain 权重)。也可从任一 vis-only x3c ckpt 抽 5/10/16 (vis-only 训练只更新了 row 20, 其余 == base), 但优先用真 base。

---

## 3. 实验矩阵

| 组 | config / init | warm 源→目标 | 数据 | 作用 |
|---|---|---|---|---|
| **B0 baseline** | 现 X3.C (cold base) | — (vis=20 冷启) | A_0423_0527 (vis-only) | 对照 (= 已有运行) |
| **A1 warm-vis** | warm-base (5→20) | AIR-AGILEX-HQ → vis | A_0423_0527 | 主实验 (H) |
| **A2 warm-kai+vis** (可选) | warm-base (5→19,20) | agilex → kai+vis | kai+A_0423×7 (= X3.B 域组成) | 测混训时也 warm kai |
| **A3 源槽扫** (可选) | warm-base (10→20), (16→20) | AIR-AGILEX / robomind-agilex → vis | A_0423_0527 | 选最佳源槽 |

> 最小闭环 = **B0 + A1** (一次性回答 H)。A2/A3 视 A1 结果再开。所有组**同超参** (30k / lr5e-5 / warmup500 / freeze1000) + 同数据, 唯一变量 = init。

---

## 4. 训练配置

- 新 config 复制 `A_0423_0527` (X3.C),仅改 `weight_loader` 指向 `WARM_BASE_CKPT`,其余完全相同。
- `num_workers = 16 × 节点数` (勿默认 64)。
- 部署/eval 仍 force `domain_id=20` (vis)。
- freeze_steps=1000 期间 backbone 冻结 — warm-init 的 domain 参数是否参与 freeze 需对齐 B0 (保持唯一变量是 init)。

---

## 5. 验证清单

| ID | 项 | 方法 |
|---|---|---|
| **V1** | base 的 5/10/16 确是"已训练"非冷启 | dump base, 比 row{5,10,16} 与 row20 的 norm/分布; 冷启 row 应 ~N(0,0.02), 已训 row 应不同 scale/结构 |
| **V2** | 槽位编号匹配 | 确认本地 xvla-base 的 domain 编号 = `domain_config.py` (即 base 确实按此 registry 预训练; 若 base 是 lerobot/xvla-base port, 核对其 domain map) |
| **V3** | surgery 正确 | warm-base 里 `row20 == row_src` (5 个参数逐一), 其余 row 不变 |
| **V4** | 训练健康 | step2000 mu PASS + loss 正常 (沿用 smoke 习惯) |

> ⚠️ **V1/V2 是前置闸门**: 若 base 槽 5/10/16 其实也是冷启 (base 没用这个 registry 预训练), 则整个 warm-init 无意义 —— **先验证再训**。

---

## 6. 评估协议

**Offline (健康 + 收敛, 非终判)**:
- val MAE@1/10/25/50 **曲线** (warm vs cold): 重点看 **早期收敛速度** (warm 应在更少 step 达到同 MAE) 与 **终值**。
- 同 val set (vis_v3_val 或 A_0423_0527 held-out), 同协议。

**真机 (终判, vis 机器)**:
| metric | 对比 |
|---|---|
| 抓衣角 / 完整折叠成功率 | A1 vs B0 |
| 抖动 (action diff p99 / 空桌面) | A1 vs B0 |
| OOD 场景成功率 | A1 vs B0 (测泛化是否被 agilex 先验提升) |

---

## 7. 决策树

```
V1/V2 通过 (base 槽 5/10/16 确为已训 + 编号匹配)?
  否 → warm-init 前提不成立, 放弃本 ablation (记录: base 未用该 registry)
  是 → 训 A1, offline 健康闸门 (vis MAE 同量级, 非 0.47)
        A1 收敛更快 / 真机成功率 ≥ B0?
          是, 明显 → ✅ warm-init 有效 → 推广: 所有新本体都从最近 agilex 槽 warm-init; 跑 A3 选最佳源槽
          是, 微弱 → 边际收益, 记录; 真机 A/B 定夺是否纳入主线
          否 / 更差 → cold 启动即可, agilex 先验对这任务无迁移价值 (记录, 关闭本线)
```

---

## 8. 执行 checklist

- [ ] **S1** 取 base ckpt (cluster `XVLA_CKPT_INIT`), dump 验证 V1 (槽 5/10/16 已训) + V2 (编号匹配)
- [ ] **S2** 写 `build_xvla_warm_init_base.py` (§2), 产 `WARM_BASE_CKPT` (5→20), 验 V3
- [ ] **S3** 新增 config (复制 X3.C, 改 weight_loader → warm base)
- [ ] **S4** smoke (mu PASS) + 提交 A1 (submit-training-job, num_workers=16×节点)
- [ ] **S5** offline 收敛曲线对比 (A1 vs B0 同 val 同协议, fresh 测)
- [ ] **S6** 选 ckpt → vis 真机 A/B (抖动 + 成功率)
- [ ] **S7** (条件触发) A3 源槽扫 / A2 warm-kai
- [ ] **S8** 回填结果到 `xvla_conditioning_methods_results.md` + 更新一致性文档 I3 结论

---

## 铁律

1. **V1/V2 前置**: 没确认 base 槽是"已训 + 编号匹配"前不要训 —— 否则白跑。
2. **唯一变量 = init**: A1 与 B0 数据/超参/freeze 完全一致。
3. **真机为终判**: offline 只看收敛速度 + 健康, agilex 先验是否真帮泛化要真机定。
4. **跨 session 数字 fresh 复测**: B0 的历史 MAE 仅作量级参考, 正式对比重测。
5. **surgery 只碰 5 个 per-domain 参数**, 别动其它权重。
