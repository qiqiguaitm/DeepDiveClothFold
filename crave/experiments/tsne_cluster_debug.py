"""调试:让 t-SNE 上的簇互不交织、各成一团。
对比多种 (聚类空间 × 嵌入) 组合, 用 2D-silhouette 量化"团块分离度"(越高越分明), 出对比图。
核心招数:在嵌入空间里聚类(而非 1280D), 则每簇=2D 连续色块, 不会你中有我。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from crave.utils import L2
from crave.render import setup_mpl
from cross_dataset_transition import temporal_smooth
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

K = 10
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F = L2(feat.astype(np.float32)); F = temporal_smooth(F, E, 3)
rng = np.random.default_rng(0); pick = rng.choice(N, 5000, replace=False)
Fs, Ts = F[pick], T[pick]
P30 = PCA(30, random_state=0).fit_transform(Fs)
print("data ready; running configs ...", flush=True)


def tsne(X, perp=30, exa=12.0):
    return TSNE(2, init="pca", perplexity=perp, early_exaggeration=exa, random_state=0).fit_transform(X)


configs = {}
# A 基线: 1280D KMeans + t-SNE(独立) → 交织
XY = tsne(Fs); configs["A: 1280D-KMeans + tSNE(基线)"] = (XY, KMeans(K, n_init=5, random_state=0).fit_predict(Fs))
# B: t-SNE 后在 2D 上 KMeans → 干净色块
configs["B: tSNE → 2D 上 KMeans"] = (XY, KMeans(K, n_init=5, random_state=0).fit_predict(XY))
# C: 高 exaggeration t-SNE 制造空隙 + 2D KMeans
XYc = tsne(Fs, perp=50, exa=36.0); configs["C: tSNE(exa36) → 2D KMeans"] = (XYc, KMeans(K, n_init=5, random_state=0).fit_predict(XYc))
# D: PCA30 → t-SNE, 2D KMeans
XYd = tsne(P30); configs["D: PCA30→tSNE → 2D KMeans"] = (XYd, KMeans(K, n_init=5, random_state=0).fit_predict(XYd))
# E: 高维 Spectral(流形连通) + 对应 t-SNE
try:
    sl = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15, random_state=0, assign_labels="kmeans").fit_predict(P30)
    configs["E: PCA30-Spectral + tSNE"] = (XYd, sl)
except Exception as e:
    print("spectral fail", e, flush=True)

plt = setup_mpl(); n = len(configs); fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(5.2 * ((n + 1) // 2), 10))
ax = ax.ravel()
for i, (name, (xy, lab)) in enumerate(configs.items()):
    sil = silhouette_score(xy, lab)               # 2D 上的分离度
    ax[i].scatter(xy[:, 0], xy[:, 1], c=lab, cmap=plt.cm.get_cmap("tab10", K), s=7, alpha=.8, linewidths=0)
    ax[i].set_title(f"{name}\n2D-silhouette={sil:.3f} ({'团块分明' if sil>0.45 else '交织' if sil<0.3 else '中等'})", fontsize=10.5)
    ax[i].set_xticks([]); ax[i].set_yticks([])
    print(f"{name}: 2D-silhouette={sil:.3f}", flush=True)
for j in range(n, len(ax)): ax[j].axis("off")
fig.suptitle(f"t-SNE 簇分离度对比 (K={K}) — 在嵌入空间聚类可消除交织", fontsize=13, y=0.99)
out = REPO / "crave/docs/visualization/cross_dataset/tsne_cluster_debug.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print("SAVED", out, flush=True)
