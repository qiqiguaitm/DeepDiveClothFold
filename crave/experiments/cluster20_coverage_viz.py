"""1280D 聚 K0=20, 只取高覆盖率(跨 episode 反复出现)的簇作 milestone 上色, 其余灰色。
测试假设: 高覆盖簇是否比全体更分离(silhouette)。嵌入用忠实 t-SNE(不喂标签)。"""
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

K0 = 20
NPTS = 8000
OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, T = idx["E"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
F = L2(temporal_smooth(L2(feat.astype(np.float32)), E, 3))
ne = len(np.unique(E))

# 在全量 1280D 聚 20 簇 + 算覆盖率
km = KMeans(K0, n_init=8, random_state=0).fit(F); lab_all = km.labels_
cov = np.array([len(set(E[lab_all == c].tolist())) / ne for c in range(K0)])
tpos = np.array([T[lab_all == c].mean() for c in range(K0)])
order_cov = np.argsort(-cov)
print(f"K0={K0} 各簇覆盖率(降序)={np.round(cov[order_cov],2).tolist()}", flush=True)
# 高覆盖选择: otsu 阈值
from cross_dataset_transition import temporal_smooth as _ts  # noqa
hist, edges = np.histogram(cov, 20)
def otsu(x):
    th = np.unique(x); best, bt = -1, x.mean()
    for t in th:
        a, b = x[x <= t], x[x > t]
        if len(a) and len(b):
            v = len(a) * len(b) * (a.mean() - b.mean()) ** 2
            if v > best: best, bt = v, t
    return bt
thr = otsu(cov); sel = np.where(cov > thr)[0]
print(f"otsu 覆盖阈={thr:.2f} → 选 {len(sel)} 簇作 milestone: 覆盖={np.round(cov[sel],2).tolist()}", flush=True)

# 子采样 + 忠实 t-SNE
rng = np.random.default_rng(0); pick = rng.choice(N, NPTS, replace=False)
XY = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(F[pick])
labp, Tp = lab_all[pick], T[pick]
ism = np.isin(labp, sel)
# 高覆盖簇 vs 全体 的分离度
sil_all = silhouette_score(F[pick], labp)
sil_ms = silhouette_score(F[pick][ism], labp[ism]) if len(set(labp[ism])) > 1 else 0
sil_ms2d = silhouette_score(XY[ism], labp[ism]) if len(set(labp[ism])) > 1 else 0
print(f"1280D silhouette: 全{K0}簇={sil_all:.3f}  仅高覆盖{len(sel)}簇={sil_ms:.3f}  (高覆盖2D-sil={sil_ms2d:.3f})", flush=True)
ms_rank = {c: r for r, c in enumerate(sel[np.argsort(tpos[sel])])}

plt = setup_mpl(); fig, ax = plt.subplots(1, 2, figsize=(17, 7.8))
for a in ax: a.scatter(XY[~ism, 0], XY[~ism, 1], color="#d5d5d5", s=6, alpha=.25, linewidths=0)
sc = ax[0].scatter(XY[ism, 0], XY[ism, 1], c=Tp[ism], cmap="viridis", s=13, alpha=.9, vmin=0, vmax=1, linewidths=0)
fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.02).set_label("每帧进度0→1", fontsize=10)
ax[0].set_title(f"(A) 高覆盖 milestone 簇按进度着色·其余灰\n{len(sel)}/{K0} 簇 (覆盖>{thr:.2f})", fontsize=12); ax[0].set_xticks([]); ax[0].set_yticks([])
cmap = plt.cm.get_cmap("tab10" if len(sel) <= 10 else "tab20", len(sel))
col = np.array([ms_rank[c] for c in labp[ism]])
ax[1].scatter(XY[ism, 0], XY[ism, 1], c=col, cmap=cmap, vmin=0, vmax=len(sel) - 1, s=13, alpha=.9, linewidths=0)
ax[1].set_title(f"(B) 高覆盖 milestone 簇按 label 着色·其余灰\n忠实 t-SNE; 仅高覆盖簇 1280D-sil={sil_ms:.2f} (全{K0}簇{sil_all:.2f})", fontsize=12); ax[1].set_xticks([]); ax[1].set_yticks([])
fig.suptitle(f"kai0 1280D 聚 {K0} 簇 → 取 {len(sel)} 个高覆盖(跨episode复现)簇作 milestone, 其余灰色 · 忠实 t-SNE", fontsize=12.5, y=1.0)
out = REPO / "crave/docs/visualization/cross_dataset/kai0_cov_milestone_k20.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("SAVED", out, flush=True)
