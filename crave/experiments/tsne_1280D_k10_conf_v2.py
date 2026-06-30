"""1280D KMeans K=10 → 每簇算置信度(到质心距离累计分布) → 保留置信分位内的点。
仅用置信点训练 t-SNE; 灰色低置信点用 transform 被动投影。
参数: conf_quantile=置信分位(默认0.85=保留离质心最近85%)。"""
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

K = 10; CQ = 0.85; NPTS = 5000
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F = L2(temporal_smooth(L2(feat.astype(np.float32)), E, 3))
rng = np.random.default_rng(0); pick = rng.choice(N, NPTS, replace=False)
Fs, Ts = F[pick], T[pick]

# 1) KMeans
km = KMeans(K, n_init=5, random_state=0).fit(Fs); lab = km.labels_
cent = km.cluster_centers_
d2c = np.linalg.norm(Fs - cent[lab], axis=1)

# 2) 每簇: 距离累积分布 → 保留置信分位内的点
keep = np.zeros(NPTS, bool)
conf_info = []
for c in range(K):
    m = lab == c
    dd = d2c[m]
    th = np.quantile(dd, CQ)  # 距离越小越置信, 取CQ分位的距离阈值
    good = dd <= th
    keep[np.where(m)[0][good]] = True
    conf_info.append(f"  c{c}: {m.sum():4d}帧, 阈值={th:.4f}, 保留{good.sum():4d}({int(good.sum()/m.sum()*100):2d}%)")
    print(conf_info[-1], flush=True)

ism = keep; klab = lab[ism]; n_conf = ism.sum(); n_low = NPTS - n_conf
print(f"总计: {n_low}/{NPTS} ({n_low/NPTS*100:.0f}%) 低置信→灰; {n_conf} 高置信→训练t-SNE", flush=True)

# 3) 仅用置信点训练 t-SNE
XY_conf = TSNE(2, init="pca", perplexity=min(30, n_conf // 3), random_state=0).fit_transform(Fs[ism])

# 4) 低置信点: 用 sklearn TSNE 不支持 transform, 所以取到各置信点质心的距离加权平均
# 作为近似: 找到最近的高置信点的 t-SNE 坐标加点噪声
from scipy.spatial import cKDTree
tree = cKDTree(Fs[ism])
dists, idxs = tree.query(Fs[~ism], k=3)
# 用 3NN 加权平均(距离倒数加权)
EPS = 1e-10
ws = 1.0 / (dists + EPS)
ws /= ws.sum(1, keepdims=True)
XY_low = (ws[:, :, None] * XY_conf[idxs]).sum(1)

# 5) 拼接成完整坐标
XY_full = np.zeros((NPTS, 2))
XY_full[ism] = XY_conf
XY_full[~ism] = XY_low

# 6) 颜色
cval = np.array([Ts[lab == c].mean() for c in range(K)])
sorted_c = np.argsort(cval)
order = {c: i for i, c in enumerate(sorted_c)}
rank = np.array([order[c] for c in klab])

COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
          "#fdb415","#e377c2","#17becf","#3b5998","#8c564b"]
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
cmap_B = ListedColormap(COLORS[:K])

plt = setup_mpl(); fig, ax = plt.subplots(figsize=(8.5, 7))
ax.scatter(XY_full[~ism, 0], XY_full[~ism, 1], color="#bbbbbb", s=7, alpha=.35, linewidths=0)
sc = ax.scatter(XY_full[ism, 0], XY_full[ism, 1], c=rank, cmap=cmap_B,
                vmin=0, vmax=K - 1, s=13, alpha=.9, linewidths=0)
cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
sm = ScalarMappable(cmap=cmap_B, norm=Normalize(0, K - 1))
cb = fig.colorbar(sm, cax=cax, ticks=range(K))
cb.set_label("cluster label (按进度排序)", fontsize=10)
cb.ax.set_yticklabels([str(i) for i in range(K)])
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"DINOv3-H 1280D KMeans K={K}\n"
             f"仅置信点(CQ={CQ}, {n_conf}帧)训练 t-SNE, 低置信(灰, {n_low}帧)投影\n"
             f"→ 簇更紧凑, 离群不干扰布局", fontsize=12)
out = REPO / f"crave/docs/visualization/cross_dataset/tsne_1280D_k{K}_conf{CQ}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("SAVED", out, flush=True)
