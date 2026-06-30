"""两面板 A/B v2:
K=20 + 置信度过滤(CQ=0.85) + 仅置信点训练 t-SNE。
A: t-SNE 特征空间, 按 cluster label 着色(高覆盖 top10 上色, 其余全灰)
B: 横轴=每帧自身 value, 纵轴=该帧所在簇的 value 均值, 同色标。
→ B 每条水平带宽=簇内 value 分散度, 容=纯(好), 宽=复现(需Viterbi)"""
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
from scipy.spatial import cKDTree

K = 20; Ktop = 10; CQ = 0.85; NPTS = 5000
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F_full = L2(temporal_smooth(L2(feat.astype(np.float32)), E, 3))  # 全量(用于覆盖率)
rng = np.random.default_rng(0); pick = rng.choice(N, NPTS, replace=False)
Fs, Ts = F_full[pick], T[pick]

# 1) KMeans K=20
km = KMeans(K, n_init=5, random_state=0).fit(Fs); lab = km.labels_
cent = km.cluster_centers_

# 2) 覆盖率(全量) + 选 top10
lab_full = km.predict(F_full)
cov = np.array([len(set(E[lab_full == c])) / len(np.unique(E)) for c in range(K)])
top_c = np.argsort(-cov)[:Ktop]
is_top = np.isin(lab, top_c)

# 3) 置信度过滤(仅对 top 簇做)
d2c = np.linalg.norm(Fs - cent[lab], axis=1)
keep = np.zeros(NPTS, bool)
status = np.full(NPTS, -1)  # -1=非milestone簇, 0=低置信, 1=高置信
for c in top_c:
    m = lab == c; dd = d2c[m]; th = np.quantile(dd, CQ)
    good = dd <= th
    ii = np.where(m)[0]; keep[ii[good]] = True; status[ii] = 0
    status[ii[good]] = 1
# 其他簇一律灰
status[~np.isin(lab, top_c)] = -1

ism = keep; klab = lab[ism]
cval = np.array([Ts[lab == c].mean() for c in range(K)])
sorted_top = sorted(top_c, key=lambda c: cval[c])
order = {c: i for i, c in enumerate(sorted_top)}
rank = np.array([order[c] for c in klab])
n_conf = ism.sum(); n_low = NPTS - n_conf
print(f"K={K} top{Ktop} CQ={CQ}: {n_conf}高置信, {n_low}低置信/非milestone", flush=True)

# 4) t-SNE 只由高置信点训练
XY_conf = TSNE(2, init="pca", perplexity=min(30, n_conf // 3), random_state=0).fit_transform(Fs[ism])
# 低置信点用 3NN 投影
tree = cKDTree(Fs[ism]); dists, idxs = tree.query(Fs[~ism], k=3)
EPS = 1e-10; ws = 1.0 / (dists + EPS); ws /= ws.sum(1, keepdims=True)
XY_low = (ws[:, :, None] * XY_conf[idxs]).sum(1)
XY = np.zeros((NPTS, 2)); XY[ism] = XY_conf; XY[~ism] = XY_low

# 5) 颜色
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
          "#fdb415","#e377c2","#17becf","#3b5998","#8c564b"]
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
cmap_B = ListedColormap(COLORS[:Ktop])

# ============ 绘图 ============
plt = setup_mpl(); fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[5, 0.08, 5], wspace=0.3)
axA = fig.add_subplot(gs[0])
# 灰色分两层: 非top簇(更浅) / top但低置信
gray = status == -1; lowc = status == 0
for mask, col, sz in [(gray, "#d0d0d0", 6), (lowc, "#aaaaaa", 8)]:
    if mask.any(): axA.scatter(XY[mask, 0], XY[mask, 1], color=col, s=sz, alpha=.35, linewidths=0)
axA.scatter(XY[ism, 0], XY[ism, 1], c=rank, cmap=cmap_B, vmin=0, vmax=Ktop-1, s=13, alpha=.9, linewidths=0)
axA.set_xticks([]); axA.set_yticks([])
axA.set_title(f"图A: t-SNE (K={K}, top{Ktop}上路)\n灰=其余10簇+低置信 | 仅高置信训练", fontsize=11)

cax_A = fig.add_subplot(gs[1])
sm = ScalarMappable(cmap=cmap_B, norm=Normalize(0, Ktop - 1))
cbA = fig.colorbar(sm, cax=cax_A, ticks=range(Ktop))
cbA.set_label("milestone cluster (按进度排序)", fontsize=9); cbA.ax.set_yticklabels([str(i) for i in range(Ktop)])

axB = fig.add_subplot(gs[2])
# 灰色点
for mask, col in [(gray, "#d0d0d0"), (lowc, "#aaaaaa")]:
    if mask.any(): axB.scatter(Ts[mask], cval[lab[mask]], color=col, s=7, alpha=.2, linewidths=0)
# 高置信彩点
scB = axB.scatter(Ts[ism], cval[klab], c=rank, cmap=cmap_B, vmin=0, vmax=Ktop-1, s=11, alpha=.6, linewidths=0)
# 水平线 + 标注
for i, c in enumerate(sorted_top):
    y_v = cval[c]; sel = lab == c
    if sel.sum():
        x_mi, x_ma = Ts[sel].min(), Ts[sel].max()
        axB.hlines(y_v, x_mi, x_ma, color=COLORS[i], lw=2.0, alpha=.8, zorder=3)
        axB.annotate(f"C{i}", xy=(min(x_ma+0.015,1.02), y_v+0.01), fontsize=8, color=COLORS[i])
axB.plot([0,1],[0,1],"k--",lw=1,alpha=.5,label="对角(y=x)")
axB.set_xlim(-0.02,1.02); axB.set_ylim(-0.02,1.02)
axB.set_xlabel("每帧自身 value (任务进度)", fontsize=11); axB.set_ylabel("该帧所在簇的 value 均值", fontsize=11)
axB.set_title("图B: 簇内 value 一致性 (K=20 top10 + CQ)\n水平带越窄=簇越纯", fontsize=11)
axB.grid(alpha=.15); axB.legend(fontsize=9, loc="upper left")

fig.suptitle("CRAVE milestone 特征空间 (DINOv3-H · 置信度过滤· K=20 top10)", fontsize=13, y=1.02)
web = REPO / "web/showcase/content/img/tsne_milestone_ab_v2.png"
doc = REPO / "crave/docs/visualization/cross_dataset/tsne_milestone_ab_v2.png"
fig.savefig(web, dpi=140, bbox_inches="tight"); print("web", web, flush=True)
fig.savefig(doc, dpi=140, bbox_inches="tight"); print("doc", doc, flush=True)
