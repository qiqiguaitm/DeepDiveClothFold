# 提议分析:用 episode-local medoid latent 作为 milestone 目标(2026-07-02)

> 用户提议:不再用全局簇中心表达 milestone,而是把每个 episode 切成阶段后,用"该阶段内与簇中心最相似帧的 latent"(episode-local medoid)作为该 milestone 的表示;输入和预测的 milestone 都来自当前 episode。分析其是否更优。

## 提议本质

- 阶段切分仍来自全局聚类(最近簇中心 → 压缩)。
- 变的是**阶段的表示**:全局簇中心(固定 37 个向量)→ **episode-local medoid**(当前 episode 该阶段内最接近簇中心的真实帧 latent)。
- 目标从"预测固定表中的一个向量"变为"预测当前 episode 下一阶段的真实帧 latent"。

这正是天花板分析里最深的杠杆(A2:连续、episode-真实的目标,脱离 37 向量固定表)的一个具体实例。

## 测量(全量 DINOv3-H,`scripts/probe_episode_medoid_target.py` + kNN 探针)

| 量 | 值 | 含义 |
|---|---|---|
| M1 cos(medoid, 全局簇心) | 0.837 ± 0.079 | medoid 明显偏离簇心 → 目标确实变了(非等价) |
| M2 同 milestone 跨 episode medoid 一致性 | 0.729 | medoid 高度 episode-特异(簇心是跨 episode 平均,抹掉了这部分) |
| M3 簇心预测器 → 真实下一 medoid | 0.836 | 预测簇心对 medoid 目标的得分(有 oracle next_m) |
| M3 持续性(当前 medoid → 下一 medoid) | 0.806 | 朴素持续 |
| **kNN(当前帧)→ 真实下一 medoid** | **0.877 / 0.879**(k=10/50) | **帧条件预测器** |

## 关键结论:提议确实更优(已测)

**决定性对比**:从当前帧预测下一阶段 episode-medoid,kNN 达 **cos 0.877**,而预测全局簇心(即使有 oracle next_m)只有 **0.836**。帧条件预测器 **+0.04 超过簇心** —— 而且 kNN 连 next_m 都不需要 oracle,全从当前帧推断。

**这证明当前帧携带了簇心没有的、可预测的 episode-特异信息。** 回归到均值(regression-to-mean)并没有完全主导 —— 存在真实、可学的 episode-特异结构。

**机制正是用户的直觉**:M2=0.73 说明同一 milestone 的 medoid 跨 episode 差异很大;但**在同一 episode 内**,当前帧透露了布料实例/光照/已走到的位姿,这些延续到下一阶段的 medoid → 所以当前帧能预测到 0.877。簇心是跨 episode 平均,天然拿不到这部分。**"输入和目标都来自当前 episode"带来的 within-episode 一致性是真实增益来源。**

## 诚实的注意点

1. **别被绝对数字误导**:medoid 目标下最优 ~0.877,低于我们现在 proto 头对**簇心目标**的 0.94 —— 但这是不同目标。对**同一个 medoid 目标**,帧条件(0.877)> 簇心(0.836)。数字"看起来低"只是因为 medoid 目标方差更大、更难,模型相对天花板其实做得更好。
2. **不修复离散 milestone 混叠**:阶段切分仍来自全局簇(重访混叠仍在),所以离散 next-milestone top1 不因此改变。但它改善的是**连续 subgoal 表示**——正是 VLA 真正消费的隐变量目标,也是更有用的输出。
3. **评估要换**:应对 medoid 目标 + 用**检索指标**(预测 latent 的最近真实帧是否落在正确的下一阶段),而非拿去和簇心 cosine 混比。

## 建议的落地版本(在用户提议上加两点)

1. **改 proto 头目标**:`greedy_proto_target` 从 `proto[future_m]`(簇心)换成"当前 episode 下一阶段的 medoid latent"。小改动,端到端可测(预期 proto 头对 medoid 目标能从 0.836 逼近 kNN 上界 0.877)。
2. **用 delta/残差目标**:预测 `next_medoid − current_frame`(LaWM 有 `delta` loss 类型),显式利用 within-episode 一致性,可能超过 0.877。
3. **检索评估**:预测 latent → 最近真实帧,判定是否命中正确下一阶段(方差鲁棒,即使预测条件均值也算命中就成功)。

## 结论一句话

**用户的提议在方向和机制上都对,并被测量证实:** 从当前帧预测 episode-local 下一阶段 medoid(0.877)优于预测全局簇心(0.836),增益来自 within-episode 一致性。它是"连续 episode-真实目标"这一天花板突破杠杆的正确实例化,值得作为 proto/subgoal 头的新目标落地(配 delta 目标 + 检索评估)。它不改变离散 milestone 的混叠,但改善的正是对 VLA 更重要的连续 subgoal。

## 落地结果(已实现并端到端训练验证)

实现:`export_episode_medoid_pairs.py` 给 pair 增加 `next_medoid` 字段(行对齐,split/real-future 目标不变);`data.py` 加 `proto_target_source: centroid | episode_medoid`;trainer 读取该开关。以 `proto_target_source: episode_medoid` 训一版,与 centroid 版做 A/B。

**proto/subgoal 头 vs 真实下一阶段 medoid(held-out,`eval_proto_subgoal.py`):**

| 指标 | centroid 训练 | **medoid 训练** |
|---|---|---|
| cos 到真实下一 medoid | 0.832 | **0.864** |
| cos 到下一簇心 | 0.944 | 0.912 |
| 检索 top1(最近簇心 == future_m) | 0.289 | **0.300** |
| (kNN 上界参考) | — | 0.877 |

**结论证实**:换成 episode-medoid 目标后,subgoal 头对真实下一帧的 cosine 从 **0.832 → 0.864**(+0.031,逼近 kNN 上界 0.877),检索 top1 还略升(不损失)。离散 milestone top1 不变(0.383,符合预期 —— 只改 proto 头目标)。**这是对 VLA 更有用的 subgoal(更接近真实可达的下一帧),已作为可选目标落地。**

剩余头空间(0.864→0.877+)可用 delta 目标(预测 `next_medoid − current_frame`,需 proto 头去归一/残差化)或更长训练进一步逼近 —— 边际较小,列为可选后续。

## 产物
- `scripts/probe_episode_medoid_target.py` → `outputs/ceiling_diag/medoid_probe.json`;kNN 帧→下一 medoid = 0.877/0.879 vs 簇心 0.836
- `scripts/export_episode_medoid_pairs.py` → `data/.../pairs_next_unique_medoid.npz`(加 `next_medoid`)
- `data.py` 新增 `proto_target_source`(默认 centroid,向后兼容);config `kai0base_dinov3h_stage3_realfuture_medoid.yaml`
- 训练:`checkpoints/stage3_realfuture_medoid/`;评估:`scripts/eval_proto_subgoal.py` → `outputs/proto_subgoal_eval/{centroid,medoid}_trained.json`
