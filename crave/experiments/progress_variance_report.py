#!/usr/bin/env python
"""统一汇报:每种聚类方法×每个编码器 的 per-cluster progress 方差(std) 对比。
数据:聚类方法对比(DINOv3-H 50ep) + 跨编码器对比(Overcluster+Otsu 100ep)。
"""
import numpy as np
from pathlib import Path
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0")

# ====== Data from cluster_method_full.py (DINOv3-H, 50ep, K=20 or adaptive) ======
method_data = {
    "KMeans K=20":         [0.2342, 0.644],   # mean_Tstd, mean_cov
    "GMM(diag) K=20":      [0.2273, 0.612],
    "WeightedKMeans K=20": [0.2390, 0.660],
    "HDBSCAN mcs=50 K=2":  [0.2295, 0.700],
    "Overcluster+Otsu K=18": [0.2242, 0.405],
}

# Per-cluster progress std distribution (from raw data — approximated from cluster_method_full.py output)
# These need the actual per-cluster values. Let me recompute from scratch.
# For the figure, I'll use the approximated distributions based on the summary stats.

# ====== Data from cross_encoder_otsu.py (100ep, Overcluster+Otsu) ======
encoder_data = {
    "DINOv2-L":    [0.2101, 0.614, 19, 0.476],
    "DINOv3-H":    [0.2192, 0.560, 18, 0.484],
    "Wan-VAE":     [0.2684, 0.368, 17, 0.264],
    "SigLIP2":     [0.2805, 0.478, 18, 0.359],
}
# [mean_Tstd, mean_cov, K_eff, tau]

# ====== Per-cluster progress profile (from cluster_progress_profile.py) ======
profile_data = {
    "DINOv2-L":  {"K": 19, "mean_T_range": [0.14, 0.91], "mean_std": 0.210, "mean_cov": 0.614, "gap_CV": 1.455},
    "DINOv3-H":  {"K": 18, "mean_T_range": [0.22, 0.91], "mean_std": 0.219, "mean_cov": 0.560, "gap_CV": 0.895},
    "Wan-VAE":   {"K": 17, "mean_T_range": [0.11, 0.80], "mean_std": 0.268, "mean_cov": 0.368, "gap_CV": 1.000},
    "SigLIP2":   {"K": 18, "mean_T_range": [0.19, 0.70], "mean_std": 0.261, "mean_cov": 0.484, "gap_CV": 0.656},
}

# ====== Figure: 3-panel comparison ======
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: 聚类方法 vs progress std (bar)
ax = axs[0]
names_m = list(method_data.keys()); cols_m = plt.cm.viridis(np.linspace(0, 0.85, len(names_m)))
vals_m = [method_data[n][0] for n in names_m]
bars = ax.bar(range(len(names_m)), vals_m, color=cols_m, alpha=.85)
ax.set_xticks(range(len(names_m))); ax.set_xticklabels([n[:20] for n in names_m], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("mean per-cluster progress std (↓好)"); ax.set_title(f"聚类方法 · DINOv3-H 50ep")
ax.grid(alpha=.25, axis="y")
for b, v in zip(bars, vals_m): ax.text(b.get_x() + b.get_width() / 2, v + .002, f"{v:.4f}", ha="center", fontsize=8)

# Panel 2: 编码器 vs progress std (bar, Overcluster+Otsu)
ax = axs[1]
names_e = list(encoder_data.keys()); cols_e = {"DINOv2-L": "#1f77b4", "DINOv3-H": "#2ca02c", "Wan-VAE": "#ff7f0e", "SigLIP2": "#d62728"}
vals_e = [encoder_data[n][0] for n in names_e]
covs_e = [encoder_data[n][1] for n in names_e]
ks_e = [encoder_data[n][2] for n in names_e]
bars = ax.bar(range(len(names_e)), vals_e, color=[cols_e[n] for n in names_e], alpha=.85)
ax.set_xticks(range(len(names_e))); ax.set_xticklabels([f"{n}\nK={ks_e[i]}" for i, n in enumerate(names_e)], fontsize=8)
ax.set_ylabel("mean per-cluster progress std (↓好)"); ax.set_title(f"编码器 · Overcluster+Otsu 100ep")
ax.grid(alpha=.25, axis="y")
for i, (b, v) in enumerate(zip(bars, vals_e)):
    ax.text(b.get_x() + b.get_width() / 2, v + .002, f"{v:.4f}\ncov={covs_e[i]:.3f}", ha="center", fontsize=7)

# Panel 3: Combined — encoder × method matrix (where available)
# Show the gap: best method + best encoder
ax = axs[2]
# Horizontal lines: encoder baselines
for i, (name, val) in enumerate([(n, encoder_data[n][0]) for n in names_e]):
    ax.axhline(val, color=cols_e[name], ls="--", lw=1, alpha=.5, label=f"{name} (Otsu)")
# Points: clustering methods
for i, (name, val) in enumerate([(n, method_data[n][0]) for n in names_m]):
    ax.scatter(i, val, s=120, c=[cols_m[i]], edgecolors="k", linewidth=.5, zorder=3)
    ax.annotate(name[:15], (i, val), fontsize=7, xytext=(0, -12), textcoords="offset points", ha="center")
ax.set_xticks(range(len(names_m))); ax.set_xticklabels([])
ax.set_ylabel("mean per-cluster progress std"); ax.set_title("交叉参照: 方法×编码器")
ax.legend(fontsize=6, loc="upper left"); ax.grid(alpha=.25)
# gold standard
best_method = min(vals_m); best_encoder = min(vals_e)
ax.text(0.5, 0.05, f"最优组合: Overcluster+Otsu × DINOv2-L\nprogress std = {min(best_method, best_encoder):.4f}",
        transform=ax.transAxes, ha="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=.6))

fig.suptitle("Per-Cluster Progress 方差(std) 统一对比 · 聚类方法 × 编码器", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .93])
out = "crave/docs/visualization/encoders/progress_variance_report.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=130)
print("SAVED", out, flush=True)

# Text summary
print("""
============= Per-Cluster Progress 方差 统一汇报 =============

一、聚类方法 (DINOv3-H, 50ep)
----------------------------------------------------------
方法                    mean progress_std   mean_cov
""", flush=True)
for n in names_m:
    print(f"  {n:<25s}  {method_data[n][0]:.4f}            {method_data[n][1]:.3f}", flush=True)

print("""
二、编码器 (Overcluster+Otsu, 100ep)
----------------------------------------------------------
编码器            K_eff   mean progress_std   mean_cov   τ
""", flush=True)
for n in names_e:
    d = encoder_data[n]
    print(f"  {n:<15s}  {d[2]:<5d}  {d[0]:.4f}            {d[1]:.3f}       {d[3]:.3f}", flush=True)

print("""
三、结论
----------------------------------------------------------
1. 最优聚类方法: Overcluster+Otsu (progress_std=0.2242)
2. 最优编码器:    DINOv2-L         (progress_std=0.2101)
3. 改变聚类方法的收益: 0.2390(KMeans)→0.2242(Otsu) = 下降 6.2%
4. 改变编码器的收益:   0.2805(SigLIP)→0.2101(DINOv2) = 下降 25.1%
5. 编码器的影响是聚类方法的 4× —— 选对编码器比选对聚类方法重要得多
""", flush=True)
