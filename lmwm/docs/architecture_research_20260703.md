# milestone+1 预测质量:架构调研与起步方案(2026-07-03)

> 唯一目标:**最大化 milestone+1 预测质量**(下一 milestone 的 latent/subgoal + 离散 id),其余次优先。
> 本文汇总文献调研 + 我们自己的实验证据,给出可直接起步的参考架构与第一步验证。

## 1. 先定位瓶颈(据我们的实验,不是拍脑袋)

| 证据 | 结论 |
|---|---|
| kNN ≈ MLP;L3 大 trunk 过拟合;7B 编码器更差;LaWM 54M 也只打平 | **backbone 容量不是瓶颈**;frame-only 已到信息天花板 |
| L2 多假设头 oracle:best-of-K → **0.90**,但 deploy(gate 选)→ 0.85 | **未来是多模态(~13 分支);模态可分,但确定性头把它们平均掉** |
| 回归/JEPA-mean subgoal cos 封顶 **0.874** | **求均值 = 结构性上限** |

→ **提升 milestone+1 质量的核心杠杆 = 建模 milestone+1 的分布(多模态),而非更大 backbone。**

## 2. 研究地图(4 条线,按相关度)

**① JEPA / 隐空间预测(我们的天然底座:在表示空间预测,不碰像素)**
- EB-JEPA 开源库(2026),支持 image-SSL → video → action-conditioned world model:https://hf.co/papers/2602.03604
- Image World Models / IWM(Garrido, Assran, LeCun 2024):https://hf.co/papers/2403.00504
- V-JEPA 2(CRAVE §7.1 已标注)属此线。

**② 层级 / 语义世界模型(milestone = 时间/语义抽象层,概念最贴)**
- Hierarchical Planning with Latent World Models(2026, Zhang/Assran/Bar/Balestriero),多时间尺度隐世界模型:https://hf.co/papers/2604.03208
- Semantic World Models(2025),预测任务语义而非像素:https://hf.co/papers/2510.19818

**③ Latent Action(我们已站的 LaWM 这条)**
- Learning Latent Action World Models In The Wild(2026, Garrido/Nagarajan/LeCun/Rabbat)= LaWM 进化版,连续受约束表示 + 空间局部化 + VQ:https://hf.co/papers/2601.05230
- Why Latent Actions Fail, and How to Prevent It(2026),辅助目标缓解外生干扰 —— 印证我们"多头辅助=正则、帮 milestone":https://hf.co/papers/2605.20223

**④ 扩散 / 生成式头(直击多模态,最高杠杆)**
- Diffusion World Model(2024):https://hf.co/papers/2402.03570
- ForeDiff / Consistent World Models via Foresight Diffusion(2025),解耦条件理解与目标去噪:https://hf.co/papers/2505.16474
- VideoWorld 2 / Latent Dynamics Model(2026),动作动态与外观解耦:https://hf.co/papers/2602.10102
- 综述:World Models for Embodied AI(2025):https://hf.co/papers/2510.16732

## 3. 起步方案(结合证据 + 已站的 LaWM)

**底座沿用已 vendored 的 LaWM 框架(lineage ③,已验证可接我们数据),两处关键升级:**

1. **数据流(已定)**:`当前 DINO latent + milestone-1 latent + state → milestone+1 latent(+ 离散 id 作粗锚)`。已是层级/语义世界模型形态(线②)。
2. **确定性 subgoal 头 → 条件隐扩散/flow 头**(线④):在 milestone+1 latent 上采样多个候选,而非回归一个均值。评估用 **best-of-N cos + 覆盖率**(对齐 VLA 用 top-k)。**唯一能捅破 0.874 的方向。**
3. **保留多头辅助**(我们实验 + "Why Latent Actions Fail" 双重支持):离散 CE + 未来特征重建正则。

**一句话**:起步 = LaWM 框架 + milestone-latent 数据流 + **条件隐扩散的 milestone+1 头**;参照 Hierarchical Planning 的层级、EB-JEPA 的工程实现。

## 4. 诚实边界

扩散头**不增加帧里的信息**,不会把"给定当前帧、唯一确定下一分支"变可能;它的收益是**不再平均模态** → best-of-N / 覆盖率 / 采样质量提升(正是 VLA 需要的),而非单点 top1 必然提升。若 best-of-N 逼近 oracle 0.90,则多模态假设成立、路线确认。

## 5. 第一步验证(本文档配套实验)

在现有 pooled augin 数据上:**条件 flow-matching subgoal 头 vs 回归头**,量 best-of-8 cos / 单样本 cos / 覆盖率。
脚本 `scripts/lever_diffusion_subgoal.py` → `outputs/diffusion_subgoal/`。判据:best-of-8 cos 是否显著 > 回归 0.874 且逼近 0.90。
