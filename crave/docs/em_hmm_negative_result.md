# EM-HMM 统一概率框架 → 已否决

> **结论**: EM-HMM 在 768D 特征空间上系统性地弱于 KMeans,不建议再投入。
> **实验日期**: 2026-07-01
> **脚本**: `crave/experiments/em_hmm_vs_kmeans.py`
> **数据**: `temp/em_hmm_vs_kmeans/comparison_results.npz`

---

## 动机

CRAVE 的 KMeans + Viterbi + smooth_monotone 管道缺乏统一数学框架。尝试用一个标准 HMM(高斯发射 + 从数据学转移)替代,通过 EM (Baum-Welch) 端到端拟合。

## 对比实验

| 配置 | 相同条件 |
|---|---|
| 数据 | 200 episode kai0_base (DINOv2 armmask+raw, 768D, 3Hz) |
| 初始 K | 96 (与 CRAVE 相同) |
| 初始化 | KMeans 中心 (fair start) |
| 防塌缩 | Dirichlet prior on A (α=2.0), banded init |

对比 KMeans K=96 (CRAVE 现状) vs KMeans K=12 (有效簇数 fair comparison) vs EM-HMM K=96。

## 结果

### 簇内特征方差 (lower = tighter = better)

| Method | K | Active | MeanVar | WtMeanVar | MeanPStd |
|---|---|---|---|---|---|
| **KMeans (CRAVE)** | 96 | **96** | **435.7** | **426.9** | **0.1780** |
| KMeans K=12 | 12 | 12 | 567.0 | 550.2 | 0.2243 |
| EM-HMM K=96 | 96 | 12 | 595.3 | 704.9 | 0.2109 |

EM-HMM 在所有簇质量指标上均差于 KMeans。

### HMM 塌缩详情

96 个初始状态中 84 个死亡 (<10 frames), 仅剩 12 个。其中 3 个支配了 81% 的数据:

```
state 34: 12351 frames (56%)  var=760  p_std=0.26
state 38:  3211 frames (15%)  var=677
state 17:  2338 frames (11%)  var=685
```

即使加了 Dirichlet 先验(α=2.0) 和 banded 初始化,塌缩仍然发生。

## 根因分析

高维 isotropic Gaussian 的 log-likelihood:

$$\log p(x \mid z=k) = -\frac{\|x-\mu_k\|^2}{2\sigma^2_k} - \frac{d}{2}\log(2\pi\sigma^2_k)$$

第二项 $-d \cdot \log(\sigma^2_k)$ 在 768D 中被放大 768 倍。方差大 5% 的簇,log-lik 优势 ×768。EM 自然倾向于用少数大方差簇覆盖数据。**这不是实现 bug,是 isotropic Gaussian 在高维中的根本性缺陷。**

KMeans 之所以工作,是因为它的硬分配 + 等权优化目标没有这个维度惩罚——均匀划分的偏置恰好匹配了 CRAVE 需要细粒度覆盖 progress 轴的需求。

## 如果要复活这个方向

需要至少解决以下之一:

1. **降维** — 先 PCA/UMAP 到 64-128D,再跑 EM
2. **Tied covariance** — 所有簇共享 σ²,消除方差竞争
3. **Cosine emission** — $p(x \mid z) \propto \exp(\cos(x, \mu_k) / \tau)$, 对高维友好
4. **Mixture of von Mises-Fisher** — 球面分布,天然适合 L2 归一化后的特征

但即使解决了塌缩,EM 能否在实际 milestone 质量上超越 KMeans 仍是 open question。

## 其他记录

- 图: `temp/em_hmm_vs_kmeans/em_hmm_vs_kmeans_quality.png`, `variance_comparison.png`
- 预测能力 demo (仅供记录): HMM 可以前向采样预测未来帧特征,但因塌缩太粗,无实用价值
- 相关讨论: `crave/docs/viterbi_computation.md` §10 (milestone-HMM, 同方向但用预设转移矩阵而非学出来的)

---

*2026-07-01 · 一次诚实的负面结果 —— KMeans 在 CRAVE 场景下就是更好的选择。*
