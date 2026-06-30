"""用 UMAP(比 t-SNE 更能把流形拆成分离岛)+ 在嵌入上聚类, 追求"团块分明、互不交织"。
扫多组 UMAP 参数 × 聚类, 用 2D-silhouette 选最佳; 出对比图。"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from crave.utils import L2
from crave.render import setup_mpl
from cross_dataset_transition import temporal_smooth
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap, hdbscan

K = 10
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F = L2(feat.astype(np.float32)); F = temporal_smooth(F, E, 3)
rng = np.random.default_rng(0); pick = rng.choice(N, 5000, replace=False)
P = PCA(40, random_state=0).fit_transform(F[pick]); Ts = T[pick]
print("data ready; UMAP sweep ...", flush=True)


def um(nn, md):
    return umap.UMAP(n_neighbors=nn, min_dist=md, n_components=2, random_state=0, metric="euclidean").fit_transform(P)


configs = {}
for nn, md in [(15, 0.1), (30, 0.0), (10, 0.0), (50, 0.0)]:
    XY = um(nn, md)
    lab = KMeans(K, n_init=5, random_state=0).fit_predict(XY)
    configs[f"UMAP nn={nn} md={md} +KMeans{K}"] = (XY, lab)
# HDBSCAN 自动找密度岛(真分离), 在最聚拢的 UMAP(nn=10,md=0) 上
XYh = um(10, 0.0)
hl = hdbscan.HDBSCAN(min_cluster_size=80, min_samples=10).fit_predict(XYh)
configs[f"UMAP nn=10 +HDBSCAN ({hl.max()+1}簇,{np.mean(hl<0)*100:.0f}%噪声)"] = (XYh, hl)

plt = setup_mpl(); n = len(configs); fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(5.2 * ((n + 1) // 2), 10)); ax = ax.ravel()
best = (None, -1)
for i, (name, (xy, lab)) in enumerate(configs.items()):
    m = lab >= 0
    sil = silhouette_score(xy[m], lab[m]) if len(set(lab[m])) > 1 else 0
    ax[i].scatter(xy[~m, 0], xy[~m, 1], c="#ddd", s=5, alpha=.4, linewidths=0) if (~m).any() else None
    ax[i].scatter(xy[m, 0], xy[m, 1], c=lab[m], cmap=plt.cm.get_cmap("tab20", max(K, lab.max() + 1)), s=7, alpha=.85, linewidths=0)
    ax[i].set_title(f"{name}\n2D-sil={sil:.3f} ({'团块分明' if sil>0.5 else '中等' if sil>0.35 else '交织'})", fontsize=10)
    ax[i].set_xticks([]); ax[i].set_yticks([])
    print(f"{name}: sil={sil:.3f}", flush=True)
    if sil > best[1]: best = (name, sil)
for j in range(n, len(ax)): ax[j].axis("off")
fig.suptitle(f"UMAP 簇分离度扫描 (K={K}) — 最佳: {best[0]} (sil={best[1]:.2f})", fontsize=12.5, y=0.99)
out = REPO / "crave/docs/visualization/cross_dataset/umap_cluster_debug.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print(f"BEST={best}", flush=True); print("SAVED", out, flush=True)
