# CRAVE v2 · 复现密度场 r(o) 架构(普适方向当前权威)

> **日期**: 2026-07-20(归档;方案成型于 2026-07-14~16,详见 lmwm roadmap)
> **取代关系**: 在**普适方向**(跨 LIBERO/kai0/robotwin)上取代 v1 离散 milestone 聚类管线
> ([final_architecture.md](final_architecture.md))。**v1 在 kai0 任务 A 的 value 读出 scope 内仍有效**
> (corr 0.943 收口未被推翻);但其三大件(BGMM 视觉聚类 / per-mode coverage≥0.5 / 双锚 Viterbi)
> 已被证不普适——LIBERO 低视觉方差下 BGMM 塌成 M=1(lmwm roadmap §5)。
> **唯一详源**: `lmvla/lmwm/docs/RECURRENCE_UNIVERSAL_goals_and_roadmap.md`(§2 方法/§4 实验日志)。本文只做 CRAVE 侧归档索引。

## 0. 一句话

**把示范集当冻结 DINOv3 空间里的非参数知识库,逐帧算跨-episode 复现密度 r(o)——无聚类·无阈值·无K·无锚;
谷=子任务边界,脊=canonical 阶段代表帧(=新"milestone"),幅值=流形监控;一套超参跨本体普适。**

## 1. 定义

```
r(o_t) = 1/(N_ep−1) · Σ_{j≠ep(t)} exp( −dmin(o_t, E_j)² / 2σ² )
  dmin(o_t,E_j) = o_t 到 episode j 最近帧距离(冻结 DINOv3-base pooled, L2)
  σ = 所有跨-ep dmin 的中位数(median-heuristic, 尺度无关)
```

三个正交读法:**幅值**=on-manifold/OOD 监控(用途C)+训练加权(用途A);**低谷**=跨-ep 分歧点=子任务边界(用途B);**高脊**=canonical 收敛态=阶段代表帧/世界模型目标(用途B)。

## 2. 阶段代表帧提取(替代 v1 的"BGMM 簇")

实现:`lmvla/lmwm/scripts/p1_libero_rvalley_pairs.py`(robotwin 版 `p1_robotwin_rvalley_pairs.py`)
```
① 每任务全 ep 拼特征 → r(o) 场
② 分段: find_peaks(−gaussian_filter1d(r,1.4), prominence=0.03(全局唯一阈值), distance=n/12) → r-谷=段边界
③ 代表帧: 每段 argmax r = r-脊(canonical 态)
④ 世界模型目标: 帧 t → 下一段的 r-脊; 末段锚末帧(不丢帧)
   (⭐ 目标必须是"脊"非"谷": 谷=分歧点,用边界当目标=重蹈 milestone+1 覆辙)
产物: lmwm/data/libero_rvalley/pairs.npz(137154 对=100% 帧覆盖) + target_compact.npz
```

## 3. 关键证据(为什么换掉 v1 聚类)

| 判据 | 结果 | 出处(lmwm roadmap) |
|---|---|---|
| 聚类塌处 r 仍连续 | task6 BGMM 塌 M=1,r 场 std=0.10、range 0.2-0.8,跨-ep 竖带一致 | §4.1 V0 |
| 边界普适涌现 | 单一全局阈值跨 40 LIBERO+kai0;真 recall 0.67 vs 随机 0.29(100% 任务真>随机) | §4.2 V1 |
| OOD 判别 | 跨任务 AUROC 中位 0.999 | §4.3 V3c |
| 脊>边界>固定(WM 目标) | r-脊 persist 0.891(更远)下 recon gain +0.079=旧 milestone 的 **2.1×** | §4.7 V5(a) |
| 下游 SR | r-脊 93.8% > 旧 milestone 92.2%(弥散 task6 +12);hintdrop 后 94.4% = baseline 打平(per-task 交易见 §4.13/4.14) | §4.9/4.10/4.13 |

## 4. 与 v1 的分工(勿混用)

| | v1 BGMM 管线(final_architecture.md) | v2 r 场(本文) |
|---|---|---|
| scope | kai0 任务A **value/AWBC 读出**(离散 milestone→进度值) | **普适信号**:边界/代表帧/WM目标/OOD,跨本体 |
| 阶段代表 | 簇(coverage 筛选的 mode) | **r-脊帧**(无聚类) |
| 已知失效 | LIBERO 塌 M=1;coverage 门槛塌 M=2 | task8 型"同物二次操作"别名(V7/V8 在解) |
| 后续 | 维持现状,不再迭代 | 主线,见 lmwm roadmap §3 路线图 |
