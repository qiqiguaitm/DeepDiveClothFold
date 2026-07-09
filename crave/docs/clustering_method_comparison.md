# CRAVE 聚类方法对比

> **日期**: 2026-07-08
> **结论**: Overcluster + Otsu coverage filter 是最优方案。KMeans 强制等方差是优点而非缺陷。
> **实验数据**: DINOv3-H 1280D, 50 ep kai0_base, 5483 帧 @3Hz
> **评测指标**: per-cluster temporal std (归一时间 T∈[0,1] 的 cluster 内 std,越低 = 越时间凝聚)

---

## 1. 核心问题

CRAVE milestone 具有异质的 coverage-density 特性:

- **高 coverage 簇**(大多数 ep 都经过的全民 phase,如 start/done):特征空间里高度凝聚
- **低 coverage 簇**(少数 ep 才有的特化 phase):天然方差更大

**问: KMeans 是否因"对所有 cluster 一视同仁地最小化方差"而无法适应这种异质性?替代方法能否获得更优的时间纯度?**

---

## 2. 对比方法

| 方法 | 原理 | 是否自适应 K | 是否处理异质密度 |
|---|---|---|---|
| **KMeans** | within-cluster variance min | ✗ | ✗ (强制等方差) |
| **GMM (diag cov)** | per-component 对角协方差 | ✗ | ✓ (每 component 独立方差) |
| **WeightedKMeans** | per-episode 等权重采样 | ✗ | 间接(补 coverage 偏差) |
| **HDBSCAN** | 密度分层聚类 | ✓ (min_cluster_size) | ✓ (天然处理异质密度) |
| **Overcluster + Otsu** | KMeans 过聚类 → coverage 阈值剪枝 | ✓ (Otsu 确定有效 K) | 后处理间接处理 |

---

## 3. 结果

```
Method                    K      mean_Tstd  cov    time  关键问题
──────────────────────────────────────────────────────────────────
KMeans                    20     0.2342     0.644  2s    需手动K
GMM (diag)                20     0.2273     0.612  24s   仅好3%,慢12×
WeightedKMeans            20     0.2390     0.660  2s    比原版更差
HDBSCAN (mcs=20)           2     0.2256     0.630  104s  高维密度崩塌
HDBSCAN (mcs=50)           2     0.2295     0.700  105s  35.7%噪声
HDBSCAN (mcs=100)          3     0.2224     0.373  106s  67.5%噪声
Overcluster(K₀=48)+Otsu   18     0.2242     0.405  3s    ★最优
```

---

## 4. 分析

### 4.1 为什么 HDBSCAN 崩塌

1280D L2 归一化空间中,所有点对距离趋于均一(concentration of measure)。密度梯度消失 → HDBSCAN 无法区分"凝聚区"和"稀疏区" → 只能分出 2-3 个粗簇,大量帧被判为噪声。**高维 + L2 归一化 = 密度方法的天敌。**

### 4.2 为什么 GMM 改善微弱

对角协方差理论上允许异质 cluster 大小,但 1280D 下 per-component 方差估计极不稳定。50 次 EM 迭代远未收敛 → 实际改善仅 3%,不抵 12× 耗时。

### 4.3 为什么 WeightedKMeans 更差

Per-episode 等权重稀释了长 demo 中的帧 → 快动作段(帧少但信息量大)被进一步边缘化 → T-std 反而上升。

### 4.4 为什么 Overcluster + Otsu 是最优的

两阶段各司其职:

1. **Overcluster (K₀=0.55√N)**: KMeans 强制等方差 → milestone 沿进度**均匀摊开** → 避免 value 曲线在某段密集、某段稀疏
2. **Otsu threshold**: 按 cross-episode coverage 自动阈值 → 剪掉低 coverage 碎片 → 只保留"共识"milestone

**KMeans 的强制等方差不是 bug,是 feature** —— 它确保 milestone 的时间分布均匀,这正是 Viterbi DP 需要的均匀 value 网格。GMM/HDBSCAN 允许异质密度反而会使高 coverage 区吞噬低 coverage 区 → 有效 K↓ → value 分辨率崩塌。

### 4.5 自适应 K

Otsu 阈值给出数据驱动的有效 milestone 数。不同数据集/编码器的 coverage 分布不同,Otsu 自动适应,无需手动指定 K。

---

## 4.6 ⚠️ 更新(2026-07-09):img⊕proprio + PCA128 下结论翻转

上面的对比是在**纯图像 DINOv3-H(1280D)**上做的。切换到最终配置 **img⊕位置 + PCA→128D**(维度低、含近高斯 proprio)后重跑,**GMM 族反超 KMeans**:

```
img⊕pos, K0=32, min_cov=0.5:
MiniBatchKMeans  K=20  Tstd=0.186  cov=0.646
KMeans           K=24  Tstd=0.176  cov=0.635
GMM(diag)        K=19  Tstd=0.153  cov=0.678
BayesianGMM      K=13  Tstd=0.141  cov=0.766  ★自适应K, 全面最优
HDBSCAN          高维 O(n²) 不实用 + 崩塌
```

**为什么翻转**:
- 早期 1280D 下 GMM 协方差估计不稳(§4.2),KMeans 等方差性质稳健胜出。
- 现在 142D(PCA128+proprio14),协方差可靠估计;proprio 分布近高斯 → GMM 的软概率+协方差建模开始生效。
- **BayesianGMM 额外优势**:Dirichlet 先验自带自适应 K,省掉 K₀ 饱和搜索。

**结论修正**:没有"永远最优"的底层过聚类器——**取决于特征维度和分布**。高维→KMeans;低维+近高斯→BayesianGMM。当前最终配置采用 **BayesianGMM**。§4.4"KMeans 等方差是 feature"在高维仍成立,但低维下 GMM 的协方差自由度带来净收益。

---

## 5. 最终推荐

```
聚类:  Overcluster (K₀=0.55√N) + Otsu coverage filter
编码器: DINOv2-large (1024D)
K:     Otsu 自适应,不手动固定
```

Otsu 阈值细节见 [`cross_episode_recurrence_value_METHOD.md` §3](cross_episode_recurrence_value_METHOD.md)。

---

## 6. 实验复现

```bash
cd deepdive_kai0
PYTHONPATH=crave/src:lmwm/src:crave/experiments python crave/experiments/cluster_method_full.py
```

输出: `crave/docs/visualization/encoders/cluster_method_full.png`
