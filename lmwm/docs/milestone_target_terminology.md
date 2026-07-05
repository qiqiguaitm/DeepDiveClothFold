# Milestone 预测目标术语定义(2026-07-05 敲定,消除"milestone+1"歧义)

之前文档里 **"milestone+1" / "next-stage medoid" 是有歧义的**(既可指时序后继,也被当成进度后继),这里统一定义并**废弃裸用 "milestone+1"**。

## 基础名词
- **milestone(里程碑基元)**:CRAVE 聚类原型,kai0_base 共 **37 个**,每个带一个 CRAVE **value/progress** `pord ∈ [0,1]`(见 `recurrence_graph.npz["pord"]`)。
- **stage(阶段段)**:一个 episode 内,逐帧对原型 argmax 后的连续同标签片段。一个 episode 会切成多段(raw 中位 51 段,Viterbi 23 段)。
- **stage medoid(阶段代表帧)**:该段内与其原型中心最相似的帧(老办法取法不变)。
- **episode milestone library(M_e,本集里程碑库)**:该 episode 访问过的**去重** milestone 集合,每个用其 medoid 帧代表,**按 value(pord)升序排列**。

## 两种预测目标(核心区别)

| 术语 | 定义 | 单调性 | 版本 |
|---|---|---|---|
| **时序后继 milestone(temporal-next)** | 当前 stage 的**时间上下一段**的 medoid | ❌ value 可倒退(m25→m21 这种)| **V1(旧,`--mode milestone`)** |
| **进度后继 milestone(progress-next / value-next)** | 在 M_e 中,value **严格大于当前 stage value 的最小者**的 medoid | ✅ value 永远前进 | **V2(新,`--mode milestone_value`)** |

- **V1 = temporal-next**:跟着**时间顺序**走,episode 若回退/重做/argmax 抖动,目标会在 value 上倒退 → 目标"忽远忽近、忽进忽退"。
- **V2 = progress-next**:跟着 **CRAVE value 顺序**走,永远指向"进度上更进一步的那个 milestone"(在本集库里选),**单调、目标一致**。

## 为什么要 V2(动机,来自诊断)
- lag 诊断显示 **V1 temporal-next 目标 horizon 极不稳定**(std 0.91,0-4s),模型**欠射**(effective lag ratio 0.42,中位退回当前帧)。
- 固定/单调的目标(near-future ratio 1.0;V2 预期同理)更可学,模型敢 commit。
- V2 用 **value 排序**取代 **时间排序**,消除目标 value 倒退 → 期望减少欠射、提升 deploy。

## 构建流程(V2)
```
episode → CRAVE 分 stage → 每 stage 取 medoid(老办法)→ 按 pord(value)升序 = M_e 本集库
for 每个 stage 的帧:  预测目标 = M_e 中 value 比当前 stage 更进一步的那个 milestone(medoid)
```
stage 分割 + medoid 取法与 V1 **完全相同**,只有**目标选择**从"时序下一段"改为"value 下一个" → 干净对比。

## 命名规范(今后统一用)
- ❌ 不再裸用 "milestone+1"。
- ✅ V1 目标 = **temporal-next milestone(时序后继)**;V2 目标 = **progress-next milestone(进度后继/value-next)**。
- near-future 流仍叫 **near-future(固定绝对时间 horizon)**,与上面两个 milestone 目标区分。

代码:`optimize_subgoal.py build_pairs(mode=milestone|milestone_value|nearfuture, pord=...)`。
