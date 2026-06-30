"""按 tsne_debug A 配方的独立图: 1280D KMeans(K簇) → 忠实 t-SNE, 按 cluster label 着色。
后续统一配色方案: 簇用颜色, 不用 value 着色。"""
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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NPTS = 5000
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F = L2(feat.astype(np.float32)); F = temporal_smooth(F, E, 3)
rng = np.random.default_rng(0); pick = rng.choice(N, NPTS, replace=False)
Fs = F[pick]

lab = KMeans(K, n_init=5, random_state=0).fit_predict(Fs)
cval = np.array([T[pick][lab == c].mean() for c in range(K)])
rank = np.array([{c: r for r, c in enumerate(np.argsort(cval))}[c] for c in lab])
XY = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(Fs)
sil2 = silhouette_score(XY, lab)

plt = setup_mpl(); fig, ax = plt.subplots(figsize=(8.5, 7))
cmap = plt.cm.get_cmap("tab10" if K <= 10 else "tab20", K)
sc = ax.scatter(XY[:, 0], XY[:, 1], c=rank, cmap=cmap, vmin=0, vmax=K - 1, s=10, alpha=.85, linewidths=0)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, ticks=range(K))
cb.set_label("cluster label (按进度排序 0→1)", fontsize=10)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"DINOv3-H 1280D KMeans(K={K}) → 忠实 t-SNE (2D-sil={sil2:.2f})", fontsize=12)
out = REPO / f"crave/docs/visualization/cross_dataset/tsne_1280D_k{K}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("SAVED", out, flush=True)
