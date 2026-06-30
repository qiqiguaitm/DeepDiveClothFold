"""在 1280D 里聚类(CRAVE 真实做法), 再找忠实展示这些 1280D 簇的 2D 嵌入。
对比: 1280D-KMeans 标签 投到 (a)t-SNE (b)UMAP无监督 (c)UMAP有监督(y=标签)。
报告 1280D silhouette(簇本身分离度) + 各嵌入 2D-silhouette(显示分离度)。"""
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
Fs, Ts = F[pick], T[pick]

# === 1280D 真实聚类 ===
lab = KMeans(K, n_init=10, random_state=0).fit_predict(Fs)
sil1280 = silhouette_score(Fs, lab)
cval = np.array([Ts[lab == c].mean() for c in range(K)])
olab = np.array([{c: r for r, c in enumerate(np.argsort(cval))}[c] for c in lab])
print(f"1280D KMeans K={K}: 1280D-silhouette={sil1280:.3f} (簇本身分离度;<0.1=连续流形重叠)", flush=True)

# === 三种嵌入, 都按 1280D 标签上色 ===
emb = {}
emb["(a) t-SNE\n(1280D标签)"] = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(Fs)
emb["(b) UMAP 无监督\n(1280D标签)"] = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(Fs)
emb["(c) UMAP 有监督 y=1280D标签\n(忠实展示该聚类)"] = umap.UMAP(n_neighbors=30, min_dist=0.0, random_state=0).fit_transform(Fs, y=lab)

plt = setup_mpl(); fig, ax = plt.subplots(1, 4, figsize=(22, 6)); cmap = plt.cm.get_cmap("tab10" if K <= 10 else "tab20", K)
for i, (name, XY) in enumerate(emb.items()):
    s2 = silhouette_score(XY, lab)
    ax[i].scatter(XY[:, 0], XY[:, 1], c=olab, cmap=cmap, vmin=0, vmax=K - 1, s=8, alpha=.85, linewidths=0)
    ax[i].set_title(f"{name}\n2D-sil={s2:.3f}", fontsize=11); ax[i].set_xticks([]); ax[i].set_yticks([])
    print(f"{name.splitlines()[0]}: 2D-sil={s2:.3f}", flush=True)
# 第4图: 有监督UMAP 按进度着色
XY = emb["(c) UMAP 有监督 y=1280D标签\n(忠实展示该聚类)"]
sc = ax[3].scatter(XY[:, 0], XY[:, 1], c=Ts, cmap="viridis", s=8, alpha=.85, vmin=0, vmax=1, linewidths=0)
fig.colorbar(sc, ax=ax[3], fraction=0.046, pad=0.02).set_label("每帧进度0→1", fontsize=9)
pur = np.mean([Ts[lab == c].std() for c in range(K)])
ax[3].set_title(f"(d) 同(c)布局·按进度着色\n簇内进度std均={pur:.2f}", fontsize=11); ax[3].set_xticks([]); ax[3].set_yticks([])
fig.suptitle(f"在 1280D 聚类(K={K}, 1280D-silhouette={sil1280:.2f}=连续流形)→ 三种 2D 嵌入忠实展示该聚类", fontsize=13, y=1.02)
out = REPO / f"crave/docs/visualization/cross_dataset/kai0_cluster1280_k{K}.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print("SAVED", out, flush=True)
