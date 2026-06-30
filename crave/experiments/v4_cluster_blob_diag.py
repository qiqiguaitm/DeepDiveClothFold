"""定量解释:为何 KMeans 簇在 2D 不成团块。
(1) 2D 只能承载 1280d 方差的多少(PCA explained var);
(2) 簇分离度 silhouette(本就低=连续流形,非团块);
(3) 特征是否被"时间/进度"连续组织(邻接帧相似 → 轨迹流形)。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crave.config import REPO
from crave.utils import L2

OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, FR, T = idx["E"], idx["FR"], idx["T"].astype(float)
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
rng = np.random.default_rng(0); pick = rng.choice(N, 6000, replace=False)
F = L2(feat[pick].astype(np.float32)); Tp = T[pick]

# (1) PCA: 2D 能承载多少方差
from sklearn.decomposition import PCA
p = PCA(50).fit(F); ev = p.explained_variance_ratio_
n90 = int(np.argmax(np.cumsum(ev) >= 0.90) + 1)
print(f"(1) PCA: 前2维方差占比={ev[:2].sum()*100:.1f}%  前10维={ev[:10].sum()*100:.1f}%  达90%需 {n90} 维", flush=True)
print(f"    → 1280d 里 {n90} 维才装下 90% 结构, 投到 2D 必然把 ~{(1-ev[:2].sum())*100:.0f}% 的簇间分离压扁", flush=True)

# (2) silhouette: 簇是否"团块状"(>0.5 团块; ~0 连续无间隙)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
for k in [37, 96]:
    lab = KMeans(k, n_init=3, random_state=0).fit_predict(F)
    sil = silhouette_score(F, lab, sample_size=3000, random_state=0)
    print(f"(2) KMeans k={k}: silhouette={sil:.3f}  ({'团块分明' if sil>0.5 else '弱/连续流形——簇间无密度间隙'})", flush=True)

# (3) 连续流形证据:同 episode 时间相邻帧 vs 随机帧 的特征距离
d_adj, d_rnd = [], []
for e in rng.choice(np.unique(E), 60, replace=False):
    gi = np.where(E == e)[0]; gi = gi[np.argsort(FR[gi])]
    if len(gi) < 10: continue
    Fe = L2(feat[gi].astype(np.float32))
    d_adj.append(np.linalg.norm(np.diff(Fe, axis=0), axis=1).mean())
    ii = rng.choice(len(Fe), (20, 2)); d_rnd.append(np.linalg.norm(Fe[ii[:, 0]] - Fe[ii[:, 1]], axis=1).mean())
print(f"(3) 时间相邻帧距离={np.mean(d_adj):.3f}  vs  同ep随机帧距离={np.mean(d_rnd):.3f}  比值={np.mean(d_adj)/np.mean(d_rnd):.2f}", flush=True)
print(f"    → 相邻帧远近 {np.mean(d_adj)/np.mean(d_rnd):.2f}× 于随机帧 = 特征沿时间平滑变化 = 连续轨迹流形, 非孤立团块", flush=True)
