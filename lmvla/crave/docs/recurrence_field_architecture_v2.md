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

## 5. ⚠️ 直接对标:UR-VC(2026-07-14 先发,必读)

**UR-VC**(arXiv 2607.12892,HKU/OpenDriveLab,Ping Luo 组):免训练跨-ep 检索(SigLIP-2,**per-episode 1-NN**,时间带 τ=0.3,ρ=0.90)→ **平均匹配帧的归一化时间** = 修正 progress → advantage 喂 π0.5,真机双臂布料折叠。
- **对 v1 的冲击最大**:它 ≈ "无聚类版的 v1 value 读出"(v1 的 milestone median-T ≈ 它的匹配帧时间均值),且"零训练+跨-ep+无标签"叙事已被先发。
- **v2 的差异仍在**:密度场(时间无关)vs 时间标签去噪;三读法结构 vs 标量;无时间带 vs τ-hack;跨本体 + 蒸馏 LMWM。
- **行动**:UR-VC = 必做 baseline(6 公式可复现)+ 前置引用。positioning 详见 lmwm roadmap §6.2/6.4/6.5(2026-07-20 更新)。

### 5.1 基线复现 + 两场仗定稿(2026-07-20,对齐特征版)

> ⚠️ 初版(dino_sub20 特征)存在**特征↔GT 错对齐**(该缓存非本数据 stride-20,已弃用),数字作废。
> 定稿用严格对齐特征:直接解 kai0_base top_head 视频 stride-20(帧数=parquet 行数逐 ep 核对),
> 双编码器各 110 ep / 8131 帧。脚本 `lmwm/scripts/{kai0_aligned_extract,urvc_aligned_analysis}.py`,
> 特征 `lmwm/data/kai0_aligned_urvc/{dino,siglip2}/`。旧 `urvc_baseline_kai0.py` 弃用勿跑。

**A) 基线(vs `stage_progress_gt`,DINOv3 / SigLIP-2 两列)**

| 方法 | corr(med) | mono | end | 覆盖率@ρ=0.90 |
|---|---|---|---|---|
| 时间标签 g(式1) | **0.977 / 0.977** | 1.000 | 1.000 | — |
| UR-VC(ρ 三档) | 0.958–0.966 / 0.960–0.967 | 0.73 / 0.68 | 0.98 / 0.99 | 98.9% / 98.5% |

覆盖率 98.9/98.5% ≈ 论文自报 98% → 机制复现可信。**读数:成功-only 数据上 UR-VC 是净负修正**(corr −0.01~−0.02、mono −0.27~−0.32):式(3) 的 ε_e 在成功示教里本来就小,检索平均反而注入噪声。(v1 0.943 为 3055ep/30Hz 另一口径,不直接同表比。)

**B) 时序之战(τ-band 靶点,DINOv3 列;SigLIP-2 同构)**

| | S1 变速 corr | S2 长回退 corr | S2 回退捕获率 |
|---|---|---|---|
| 时间标签(wall) | **0.867** | 0.674 | 0.0% |
| UR-VC τ=0.3 | 0.834 | 0.711 | **7.4%**(siglip2: 5.7%) |
| 无带检索 τ=∞ | 0.780 | **0.862** | **87.7%**(siglip2: 83.4%) |

**三句话**:①常规成功-only 数据:UR-VC 严格劣于裸时间标签;②它的主场(长回退):**τ-band 把捕获率杀到 <8%**(无带 >83%)——为抗别名装的带,废掉了方法的存在理由;③变速场景证明带确实在抗别名(0.834>0.780)——**τ-band = 拿回退表达力换别名鲁棒**;我们的密度场不做这笔交换(来源多样性加权抗 dwelling,别名走 belief/V7)。

**C) 编码器之战:平局(SigLIP-2 vs DINOv3 各指标 Δ≤0.005,S2 同样瘫痪)→ 病灶是 τ-band+时间标签聚合这两个架构选择,与编码器无关**;同时堵死"编码器用错"的辩护。
