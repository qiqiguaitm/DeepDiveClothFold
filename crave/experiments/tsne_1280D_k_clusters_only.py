"""1280D KMeans → t-SNE。不是每点都归簇: 离质心太远的点(簇内离群)灰色, 簇颜色避开灰色。
参数: K=簇数, sig=标准差倍数(离质心>mean+sig*std→灰色,默认1.5)。"""
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
SIG = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2
NPTS = 5000
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F = L2(feat.astype(np.float32)); F = temporal_smooth(F, E, 3)
rng = np.random.default_rng(0); pick = rng.choice(N, NPTS, replace=False)
Fs, Ts = F[pick], T[pick]

km = KMeans(K, n_init=5, random_state=0).fit(Fs); lab = km.labels_
# 每簇: 点->质心距离, 剔除离群
cent = km.cluster_centers_
d2c = np.linalg.norm(Fs - cent[lab], axis=1)
keep = np.ones(NPTS, bool)
cluster_info = []
for c in range(K):
    m = lab == c
    dd = d2c[m]; mu, sd = dd.mean(), dd.std()
    th = mu + SIG * sd
    good = dd <= th
    # 标记被剔除的点
    keep_idx = np.where(m)[0][good]
    drop_idx = np.where(m)[0][~good]
    keep[drop_idx] = False
    cluster_info.append(f"  c{c}: {m.sum():4d}帧 → {good.sum():4d}在簇 {len(drop_idx):4d}灰色")
XY = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(Fs)
# 剩余: 要上色的点
ism = keep
klab = lab[ism]  # 剩余点的簇标签
cval = np.array([Ts[lab == c].mean() for c in range(K)])
rank = np.array([{c: r for r, c in enumerate(np.argsort(cval))}[c] for c in klab])

print(f"K={K} σ={SIG}: {ism.sum()}/{NPTS} 帧在簇, {(~ism).sum()} 灰色", flush=True)
for l in cluster_info: print(l, flush=True)

plt = setup_mpl(); fig, ax = plt.subplots(figsize=(8.5, 7))
# 灰色底
ax.scatter(XY[~ism, 0], XY[~ism, 1], color="#bbbbbb", s=7, alpha=.35, linewidths=0)
# 显式色板: 10 种 Material Design 干净色(无灰色系)
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
          "#fdb415","#e377c2","#17becf","#3b5998","#8c564b"]
if K > len(COLORS): COLORS = COLORS * (K // len(COLORS) + 1)
cmap = plt.cm.colors.ListedColormap(COLORS[:K])
sc = ax.scatter(XY[ism, 0], XY[ism, 1], c=rank, cmap=cmap, vmin=0, vmax=K - 1, s=13, alpha=.9, linewidths=0)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb = fig.colorbar(sc, cax=cbar_ax, ticks=range(K))
cb.set_label("cluster label (按进度排序)", fontsize=10)
if K <= 15: cb.ax.set_yticklabels([str(i) for i in range(K)])
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"DINOv3-H 1280D KMeans K={K} · σ={SIG} 去离群\n"
             f"{ism.sum()} 帧在簇 ({(~ism).sum()} 灰色) · 忠实 t-SNE", fontsize=12)
out = REPO / f"crave/docs/visualization/cross_dataset/tsne_1280D_k{K}_σ{SIG}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("SAVED", out, flush=True)
