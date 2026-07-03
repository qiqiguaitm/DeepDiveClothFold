# LMWM 最终架构(收紧版,2026-07-03)

> 经 milestone+1 方向的全量实验(P0 gf3 8卡 + 七大本地快验 + 跨外观泛化 + 解码横评)收敛后的**定稿架构**。
> 本文是架构与指标的**单一事实源**;试错教训见 [`pitfalls_and_lessons_20260703.md`](pitfalls_and_lessons_20260703.md),调研见 [`architecture_research_20260703.md`](architecture_research_20260703.md)。

## 1. 总体 pipeline

```
当前帧 (RGB 256²)
  │  ① 感知(冻结)
  ▼
[DINOv3-H ViT-H/16+  ~840M ❄]  →  pooled latent 1280
  │        (patch-grid 已否决:空间 token 无增益 + 130GB)
  ├───────────────────────────────────────────────┐
  ▼  ② milestone 预测器 (Part A)                    │
[augin⁺ 3854 → MLP 512×2 → milestone头+subgoal头]   │
  → 下一 milestone 分布 top-k + 置信度/熵            │
  → 融合图先验 (λ=0.2)                               │
  │                                                 ▼  ③ VLA subgoal 接口 (Part B)
  │                              [forward(当前latent, code) → 下一状态latent]
  │                               code = predictor(当前);外观由当前观测带入
  ▼                                                 ▼
  离散 milestone 计划        +      外观正确的 on-manifold subgoal latent
  └────────────── 一起喂 VLA(planning prior)──────────────┘

  ④ 解码(仅可视化,不在 VLA 路径):检索最近真实帧(再编码 cos 0.84);量化用 latent-cos 不解码
```

## 2. 模块 · 参数量 · 输入输出

| 模块 | 参数量 | 输入 | 输出 | 训练 |
|---|---|---|---|---|
| ① DINOv3-H | ~840M | 帧 256² | pooled 1280 | ❄ 冻结 |
| ② milestone 预测器(5-ens) | **15.9M**(单成员 3.18M) | augin⁺ **3854** | milestone logits 37 + subgoal proto 1280 | ✅ |
| ③ VLA subgoal 接口 | **5.5M**(inverse 1.61 + forward 1.61 + predictor 2.27) | inverse(2560)/forward(1344)/pred(3854) | code 64 → 下一 latent 1280 | ✅ |
| **合计可训练** | **~21.4M** | | | |

**augin⁺ (3854) = pooled 1280 + prev-milestone latent 1280 (H1) + current-milestone latent 1280 (H3) + proprio 14**
(H1: latent>one-hot;H3: +current-milestone latent 是最大单项增益 +1.9pt)

**输出**:①离散 milestone 计划(top-k + 置信度);②on-manifold subgoal latent(经 forward-from-current,外观正确)。

## 3. 损失 / 推理配方

- milestone 头:**CVaR-CE**(H4,均值+方差同降)
- subgoal 头:**特征空间 cos**(on-manifold,给 VLA)
- VLA subgoal:**forward(当前观测, code)** —— 外观由当前带入,泛化到未见衣物
- 推理:温度校准 → **图先验融合 λ=0.2**(H5,输入更强→更少先验)→ 5 成员集成

## 4. 最终指标 vs baseline

| 指标 | 旧 baseline(augin one-hot 6-ens+fuse) | **最终收紧版** | Δ |
|---|---|---|---|
| milestone top1 | 0.459 | **0.465** | +0.6pt |
| milestone top5 | ~0.85 | **0.875** | +2.5pt |
| NLL | 1.74 | **1.68** | −0.06 |
| subgoal cos(in-dist) | 0.874 | **0.882** | +0.8pt |
| subgoal 未见外观(vis_base) | absolute 漂到 0.82 | **forward oracle 0.935 / 部署 ~0.82–0.90** | 机制可达 0.93 |
| 解码保真(检索,再编码 cos) | — | **0.843** | 清晰真帧 |
| 可训练参数 | ~18M | ~21.4M | 相当 |

**参考基线(LaWM 相邻帧)**:forward+oracle 恒 ~0.97(所有 horizon),forward+**predicted** 封顶 **~0.90**(未来欠定的普适上限)—— 我们 milestone 级已追平(in-dist 0.90),milestone 大跳并不比帧级差。

## 5. 跨外观泛化(vis_base,真·未见衣物)

| | forward+oracle | absolute | forward+predicted |
|---|---|---|---|
| kai0 in-dist | 0.971 | 0.897 | 0.90 |
| **vis_base 未见外观** | **0.935** | 0.820 | ~0.82 |

- **forward-from-current 机制外观无关**(oracle 未见 0.935 vs absolute 0.82)—— 子目标继承**当前观测**的外观(红衣→红衣),而非漂到 demo 色。
- 部署瓶颈 = code 预测(未来欠定 + 外观敏感),~0.82–0.90;**非 milestone 特有**(LaWM 帧级也 0.90)。

## 6. "收紧" = 砍掉的死重(全部实测否决)

| 否决项 | 理由 |
|---|---|
| LaWM 大 backbone / grid 空间 token | top1 0.382<0.465;H7 证空间无增益(A≈B);100× 成本 |
| patch-grid 编码(130GB) | pooled 足矣 |
| 扩散 / 多假设 subgoal 头 | best-of-N < 回归(未来欠定,采样加噪) |
| 大 trunk 1024×3 | 生产过拟合 |
| milestone-pair 转移码 | 太粗(0.83≈absolute) |
| **color-aug 强制颜色无关** | **毁颜色相关任务(红块放蓝块);forward-from-current 无需它** |

## 7. 设计原则(任务无关,勿过拟合叠衣服)

- subgoal 表示**不做"外观/颜色无关"硬假设**(那是叠衣服特性,非通用);机制保持任务无关。
- 外观由当前观测带入;泛化靠**多任务/多外观数据**,而非硬编不变性。
- 颜色相关任务(红块放蓝块):颜色∈当前观测,code 自然学成颜色引用 —— forward-from-current **不加修改**即正确。

## 8. 产物

- 生产 milestone 模型:`scripts/train_prod_milestone.py` → `checkpoints/prod_milestone/member_*.pt`
- VLA subgoal 泛化:`scripts/verify_appearance_visbase.py` / `verify_transition_code.py` / `verify_color_aug.py` → `outputs/appearance_gen/`
- LaWM 相邻帧参考:`scripts/lawm_adjacent_baseline.py`
- 解码横评:`scripts/viz_lmwm_decode_compare.py`(检索 vs 合成)
