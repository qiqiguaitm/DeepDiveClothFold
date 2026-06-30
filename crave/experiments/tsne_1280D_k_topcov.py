"""1280D KMeans K=20 → t-SNE, 只取覆盖率最高的 K_top 个簇上色(其余簇全灰)。
叠 σ 离群剔除: 上色簇内离质心远者灰度(非簇内离群保留原色)。
参数: K=总簇数, K_top=上色簇数。"""
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

K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
Ktop = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SIG = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
NPTS = 5000
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F = L2(temporal_smooth(L2(feat.astype(np.float32)), E, 3))
ne = len(np.unique(E))
rng = np.random.default_rng(0); pick = rng.choice(N, NPTS, replace=False)
Fs, Ts = F[pick], T[pick]

km = KMeans(K, n_init=5, random_state=0).fit(Fs); lab_sub = km.labels_
# 全量预测后算覆盖率
lab_full = km.predict(F)
cov = np.array([len(set(E[lab_full == c])) / ne for c in range(K)])

# 选覆盖率最高的 Ktop 个簇
top_c = np.argsort(-cov)[:Ktop]
is_milestone = np.isin(lab_sub, top_c)
print(f"覆盖率最高的 {Ktop} 簇: {np.round(cov[top_c],2).tolist()}", flush=True)
print(f"  = {is_milestone.sum()}/{NPTS} 帧在上色簇, {(~is_milestone).sum()} 非上色簇→灰", flush=True)

# 在 milestone 簇内按 σ 剔除离群
cent = km.cluster_centers_
d2c = np.linalg.norm(Fs - cent[lab_sub], axis=1)
keep = np.ones(NPTS, bool)
for c in top_c:
    m = lab_sub == c
    dd = d2c[m]; mu, sd = dd.mean(), dd.std()
    th = mu + SIG * sd
    good = dd <= th
    drop = np.where(m)[0][~good]
    keep[drop] = False
    print(f"  c{c}: {m.sum():4d}帧 → 上色{good.sum():4d} 离群灰色{(~good).sum():4d}", flush=True)

# 组装上色: 既是 milestone 簇又在 σ 内
colored = is_milestone & keep
rest = ~colored  # 所有灰色: 非milestone簇 + milestone簇内的离群

# 只对上色簇按进度排序(从低→高)
sorted_by_progress = sorted(top_c, key=lambda c: Ts[lab_sub == c].mean())
rank = np.full(NPTS, -1)
for i, c in enumerate(sorted_by_progress):
    rank[lab_sub == c] = i

XY = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(Fs)

plt = setup_mpl(); fig, ax = plt.subplots(figsize=(8.5, 7))
ax.scatter(XY[rest, 0], XY[rest, 1], color="#bbbbbb", s=7, alpha=.35, linewidths=0)
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
          "#fdb415","#e377c2","#17becf","#3b5998","#8c564b"]
cmap = plt.cm.colors.ListedColormap(COLORS[:Ktop])
sc = ax.scatter(XY[colored, 0], XY[colored, 1], c=rank[colored], cmap=cmap,
                vmin=0, vmax=Ktop - 1, s=13, alpha=.9, linewidths=0)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb = fig.colorbar(sc, cax=cbar_ax, ticks=range(Ktop))
cb.set_label("milestone 簇 (按进度排序)", fontsize=10)
cb.ax.set_yticklabels([str(i) for i in range(Ktop)])
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"1280D KMeans K={K} · 高覆盖 {Ktop} 簇上色 · σ={SIG} 去离群\n"
             f"{colored.sum()} 帧上色 / {rest.sum()} 灰色", fontsize=12)
out = REPO / f"crave/docs/visualization/cross_dataset/tsne_1280D_k{K}_top{Ktop}_σ{SIG}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("SAVED", out, flush=True)
