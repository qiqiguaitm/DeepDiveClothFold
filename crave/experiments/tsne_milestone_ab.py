"""两面板:
A: t-SNE, 按 cluster label 着色(K=10 σ1.2 离群剔除)
B: 横轴=每帧自身 value(进度), 纵轴=该帧所在簇的 value 均值, 同色标。
→ B 展示"簇内 value 一致性": 水平带窄 = 纯, 宽 = 不纯"""
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

K = 10; SIG = 1.2; NPTS = 5000
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F = L2(temporal_smooth(L2(feat.astype(np.float32)), E, 3))
rng = np.random.default_rng(0); pick = rng.choice(N, NPTS, replace=False)
Fs, Ts = F[pick], T[pick]

km = KMeans(K, n_init=5, random_state=0).fit(Fs); lab = km.labels_
cent = km.cluster_centers_
d2c = np.linalg.norm(Fs - cent[lab], axis=1)
keep = np.ones(NPTS, bool)
for c in range(K):
    m = lab == c; dd = d2c[m]; th = dd.mean() + SIG * dd.std()
    keep[np.where(m)[0][dd > th]] = False
ism = keep; klab = lab[ism]
cval = np.array([Ts[lab == c].mean() for c in range(K)])   # 每簇平均 value
sorted_c = np.argsort(cval)  # 按 value 升序排列的簇号
order = {c: i for i, c in enumerate(sorted_c)}
rank = np.array([order[c] for c in klab])

XY = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(Fs)
print(f"K={K} σ={SIG}: {ism.sum()}/{NPTS} 上色, {(~ism).sum()} 灰色")

COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
          "#fdb415","#e377c2","#17becf","#3b5998","#8c564b"]
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
cmap_B = ListedColormap(COLORS[:K])

plt = setup_mpl()
fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[5, 0.08, 5], wspace=0.3)

# A: t-SNE
axA = fig.add_subplot(gs[0])
axA.scatter(XY[~ism, 0], XY[~ism, 1], color="#bbbbbb", s=7, alpha=.35, linewidths=0)
axA.scatter(XY[ism, 0], XY[ism, 1], c=rank, cmap=cmap_B,
            vmin=0, vmax=K - 1, s=13, alpha=.9, linewidths=0)
axA.set_xticks([]); axA.set_yticks([])
axA.set_title("图A: 特征空间 t-SNE\n按 cluster label 着色", fontsize=12)
# colorbar
cbA_ax = fig.add_subplot(gs[1])
sm = ScalarMappable(cmap=cmap_B, norm=Normalize(0, K - 1))
cbA = fig.colorbar(sm, cax=cbA_ax, ticks=range(K))
cbA.set_label("cluster label (按进度排序)", fontsize=9)
cbA.ax.set_yticklabels([str(i) for i in range(K)])

# B: 每帧 value vs 簇均值 value
axB = fig.add_subplot(gs[2])
# 灰色: 全部点数(非上色簇+离群) + 上色点
axB.scatter(Ts[~ism], cval[lab[~ism]], color="#bbbbbb", s=7, alpha=.25, linewidths=0)
axB.scatter(Ts[ism], cval[klab], c=rank, cmap=cmap_B,
            vmin=0, vmax=K - 1, s=11, alpha=.6, linewidths=0)
# 各簇水平线
for i, c in enumerate(sorted_c):
    y_line = cval[c]; sel = lab == c
    if sel.sum():
        x_min, x_max = Ts[sel].min(), Ts[sel].max()
        axB.hlines(y_line, x_min, x_max, color=COLORS[i], lw=1.8, alpha=.8, zorder=3)
        axB.annotate(f"C{i}", xy=(min(x_max + 0.015, 1.02), y_line + 0.01),
                     fontsize=8, color=COLORS[i], ha="left", va="bottom")
# 对角参考
axB.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="对角(y=x,完美一致)")
axB.set_xlim(-0.02, 1.02); axB.set_ylim(-0.02, 1.02)
axB.set_xlabel("每帧自身 value (任务进度)", fontsize=11)
axB.set_ylabel("该帧所在簇的 value 均值", fontsize=11)
axB.set_title("图B: 簇内 value 一致性\n横轴=帧 value, 纵轴=簇均值", fontsize=12)
axB.grid(alpha=.15); axB.legend(fontsize=9, loc="upper left")

fig.suptitle("CRAVE milestone 特征空间 (DINOv3-H 1280D KMeans K=10)", fontsize=14, y=1.02)
web = REPO / "web/showcase/content/img/tsne_milestone_ab.png"
doc = REPO / "crave/docs/visualization/cross_dataset/tsne_milestone_ab.png"
fig.savefig(web, dpi=140, bbox_inches="tight"); print("web", web, flush=True)
fig.savefig(doc, dpi=140, bbox_inches="tight"); print("doc", doc, flush=True)
