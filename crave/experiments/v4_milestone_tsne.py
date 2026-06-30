"""最终图: UMAP(nn=10,md=0) 把 DINOv3-H 流形拆开 + 在 2D 上 KMeans 聚 K 簇(默认10) →
每簇=连续色块、互不交织。左=按簇 value(进度)着色, 右=按 cluster label。
(说明: HDBSCAN 只找到~2 个真密度岛 → 数据本质是一条连续流形, 这 K 簇是对它的干净切分, 按进度排序。)"""
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
import umap

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NPTS = 8000
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F = L2(feat.astype(np.float32)); F = temporal_smooth(F, E, 3)
rng = np.random.default_rng(0); pick = rng.choice(N, NPTS, replace=False)
P = PCA(40, random_state=0).fit_transform(F[pick]); Ts = T[pick]
print(f"UMAP nn=10 md=0 on {NPTS} 帧 ...", flush=True)
XY = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=2, random_state=0).fit_transform(P)
# 在 2D 嵌入上聚类 → 干净色块
lab = KMeans(K, n_init=8, random_state=0).fit_predict(XY)
cval = np.array([Ts[lab == c].mean() for c in range(K)])        # 每簇平均进度=value
rank = {c: r for r, c in enumerate(np.argsort(cval))}
olab = np.array([rank[c] for c in lab])
sil = silhouette_score(XY, lab)
print(f"K={K} 2D-silhouette={sil:.3f}; 簇 value(排序)={np.round(np.sort(cval),2).tolist()}", flush=True)

plt = setup_mpl(); fig, ax = plt.subplots(1, 2, figsize=(17, 7.8))
sc = ax[0].scatter(XY[:, 0], XY[:, 1], c=Ts, cmap="viridis", s=11, alpha=.85, linewidths=0, vmin=0, vmax=1)
cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.02); cb.set_label("每帧任务进度 0→1", fontsize=10)
pur = np.mean([Ts[lab == c].std() for c in range(K)])
ax[0].set_title(f"(A) 每帧按自身进度着色 — 簇内进度 std 均={pur:.2f}\n(blob 内颜色越杂=该视觉态在不同进度复现)", fontsize=12); ax[0].set_xticks([]); ax[0].set_yticks([])
cmap = plt.cm.get_cmap("tab10" if K <= 10 else "tab20", K)
sc2 = ax[1].scatter(XY[:, 0], XY[:, 1], c=olab, cmap=cmap, vmin=0, vmax=K - 1, s=11, alpha=.85, linewidths=0)
cb2 = fig.colorbar(sc2, ax=ax[1], fraction=0.046, pad=0.02, ticks=range(K)); cb2.set_label("cluster label(按进度序)", fontsize=10)
ax[1].set_title(f"(B) {K} 簇按 cluster label 着色 — 每簇一团一色", fontsize=12); ax[1].set_xticks([]); ax[1].set_yticks([])
fig.suptitle(f"kai0 DINOv3-H · UMAP(nn=10,md=0)+2D-KMeans 聚 {K} 簇 · 团块分明 2D-silhouette={sil:.2f}"
             f"\n(HDBSCAN 仅检出~2 真密度岛 → 数据本质连续流形, 这 {K} 簇是对其的干净切分, 按进度排序)", fontsize=12, y=1.0)
out = REPO / f"crave/docs/visualization/cross_dataset/kai0_umap_k{K}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("SAVED", out, flush=True)
