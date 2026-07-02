# Symmetric Adaptive Vote — Viterbi-free 在线 progress 读出

> **日期**: 2026-07-01  
> **定位**: 替代 CRAVE Viterbi-DP 的在线推理方法  
> **实验 ep**: 2302 (kai0_base, 30Hz, 39 milestones)  
> **数据**: `temp/coverage_gated_jump/`, `temp/crave_interp_ep2302_30hz_decoded/`

---

## 核心思想

Viterbi 做三件事,我们逐层替换:

| Viterbi | 我们的替代 |
|---|---|
| 全局 DP 最优路径 | 逐帧 distance correction (EMA 进度先验) |
| 转移惩罚 λ\|Δb\| | 对称累积投票 (同一个 wd 对上升/下降) |
| 中值滤波 / smooth_monotone | 同款 boxcar 卷积 (w=15, 0.5s) |

算法:

```
for each frame t:
  1. raw_k = argmin_k || feat[t] - C_k ||           # 最近簇
  2. ema[t] = 0.1 * Pord[raw_k] + 0.9 * ema[t-1]    # 进度估计
  3. cd[t,k] = dist[t,k] * (1 + α * |ema[t] - Pord[k]|)  # 校正距离
  4. best_k = argmin_k cd[t,k]                       # 最佳候选
  5. if best_k == cur_k: stay; vote reset
     else:
       d_ratio = dist[t,cur_k] / median(dist[t,:])
       wd_eff = wd * clamp(d_ratio^t, 0.5, 1.8)     # 自适应阈值
       if vote_count[best_k] >= wd_eff: switch       # 累积投票
       else: stay
  6. v[t] = Pord[cur_k]
  7. (post) v = boxcar_smooth(v, w=15)               # 斜坡过渡
```

**参数**: `α=2.0, wd=10, t=0.3, smooth_w=15` (30Hz 原生)

---

## 与 Viterbi 的对比

| 维度 | Viterbi-DP | Sym Adaptive Vote |
|---|---|---|
| 推理模式 | 离线 (需完整序列) | **在线** (因果,仅用过去帧) |
| 优化目标 | min Σ(emission + transition) | 逐帧贪心 + 累积共识 |
| 参数数量 | 5 (lam, bins, medw, end_bonus, start_anchor) | **3** (α, wd, smooth_w) |
| 转移模型 | 几何惩罚 `λ\|Δb\|` | 对称累积投票 + 自适应阈值 |
| 方向处理 | 对称 (前进/后退同价) | **对称** (同一 wd,无方向偏置) |
| 后处理 | median + smooth_monotone | smooth_monotone (同款) |
| 复杂度 | O(NF·NB²) per sequence | **O(NF·NM)** per frame |
| 延迟 | 需等 episode 结束 | **0 延迟** (因果) |
| 可部署性 | 离线 only | **真机实时可用** |

## 实验数据 (ep2302, 39 milestones, with state, 30Hz)

| 方法 | mono | end | jumps | bounces | corr vs Viterbi |
|---|---|---|---|---|---|
| raw argmin | 0.941 | 0.988 | 213 | 156 | 0.941 |
| DC only (α=2.0) | 0.962 | 0.953 | 224 | 89 | — |
| DC + asym vote (fwd=1,bwd=20) | 0.990 | 0.953 | 93 | 38 | — |
| DC + sym vote (wd=10) | 0.970 | 0.988 | — | 5 | 0.970 |
| DC + sym vote + smooth (wd=10,w=15) | — | 0.981 | — | 0 | **0.974** |
| Viterbi (3Hz→upsample) | 0.998 | 0.950 | 22 | 0 | 1.000 |

**最终方案**: DC(α=2.0) + sym adaptive vote(wd=10,t=0.3) + smooth_monotone(w=15)。corr=0.974,末值=0.981,0 尖峰,完全在线。

## 消融实验关键结论

1. **30Hz 原生远优于 3Hz+upsample** — raw argmin mono 0.96 vs 0.78,高频采样本身就是最强平滑器
2. **State 信息必须包含** — 796D (arm+raw+state) 比 768D (arm+raw only) 的簇质量更好
3. **Coverage gate 无效** — KMeans+coverage filter 已把低 cov 簇排除在 argmin 范围外
4. **EM-HMM 无法替代 KMeans** — 768D 高维空间下 EM 塌缩 (96→12 states),方差惩罚主导
5. **累积投票的 wd 控制尖峰** — wd=10 (0.33s) 可把尖峰从 10→5;wd=16 可到 0 但末值稍降
6. **boxcar smooth 优于 median** — 在 plateau 间生成真正斜坡,提升 corr

## 实验路径

```
temp/coverage_gated_jump/
  ablation_30hz.png              — 30Hz 消融全景
  sym_adaptive_vote.png          — 对称自适应投票最终效果
  spike_free.png                 — wd=16 零尖峰版本
  smooth_monotone.png            — boxcar 平滑对比
  ep2302_sym_adaptive.mp4        — 对齐视频 (vs raw+baseline)
  ep2302_vs_decoded_target.mp4   — 对齐视频 (vs Viterbi target)
  ep2302_spike_free.mp4          — 对齐视频 (wd=16 零尖峰)
```

## 脚本

- `crave/experiments/argmin_jump_analysis.py` — 跳变机制深度分析
- `crave/experiments/viterbi_vs_ema_ep2302.py` — EMA vs DP 对比
- `crave/experiments/em_hmm_vs_kmeans.py` — EM-HMM 尝试(已否决)
- `crave/experiments/crave_ep2302_covgate_video.py` — coverage gate 视频

---

*2026-07-01 · 4 小时密集实验 · 从 Viterbi 到 3 参数在线方案*
