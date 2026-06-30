"""1280D 聚类分离度能否提高? 扫 特征预处理 × K, 报告 silhouette 上限 + 自然 K。
silhouette 随 K 变、且对预处理(标准化/白化)敏感; 找最大可达分离度。"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from crave.utils import L2
from cross_dataset_transition import temporal_smooth
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E = idx["E"]
N = len(E); feat = np.zeros((N, 1280), np.float16)
for sh in ["shard_0.npz", "shard_1.npz"]:
    s = np.load(OUTD / sh); feat[s["gidx"]] = s["feat"]
raw = feat.astype(np.float32)
rng = np.random.default_rng(0); pick = rng.choice(N, 10000, replace=False)

# 方差集中度诊断
pc = PCA(50, random_state=0).fit(L2(raw[pick]))
print(f"方差集中度: top1={pc.explained_variance_ratio_[0]*100:.1f}% top5={pc.explained_variance_ratio_[:5].sum()*100:.1f}% top20={pc.explained_variance_ratio_[:20].sum()*100:.1f}%", flush=True)

Fsm = L2(temporal_smooth(L2(raw), E, 3)[pick])
Fraw = L2(raw[pick])
variants = {
    "L2+smooth(当前)": Fsm,
    "L2 原始(无平滑)": Fraw,
    "zscore 每维标准化": StandardScaler().fit_transform(raw[pick]),
    "PCA50 白化": PCA(50, whiten=True, random_state=0).fit_transform(L2(raw[pick])),
    "PCA50+L2": L2(PCA(50, random_state=0).fit_transform(L2(raw[pick]))),
}
Ks = [2, 3, 4, 5, 6, 8, 10, 15, 20]
print(f"\n{'变体':<18}" + "".join(f"K{k:<5}" for k in Ks) + "  最佳", flush=True)
for name, X in variants.items():
    row = []
    for k in Ks:
        lab = KMeans(k, n_init=5, random_state=0).fit_predict(X)
        row.append(silhouette_score(X, lab, sample_size=5000, random_state=0))
    best = max(range(len(Ks)), key=lambda i: row[i])
    print(f"{name:<16}" + "".join(f"{v:.3f} " for v in row) + f"  → K={Ks[best]} sil={row[best]:.3f}", flush=True)
