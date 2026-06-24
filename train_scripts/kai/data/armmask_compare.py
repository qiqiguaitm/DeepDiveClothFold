#!/usr/bin/env python
"""臂掩膜前后对比(§5.7 缓解a 验证):
同 50 ep、同 KMeans(k=48,seed0)、同 top-10 milestone 协议:
  ① GT 验证: V_milestone τ vs stage_progress_gt(基线=未掩膜 0.812)
  ② 偏置复查: cov-spread 可疑簇(基线=c4 92% 臂簇)是否消失
  ③ 最高覆盖簇代表帧(肉眼验证不再是臂)
用法: python armmask_compare.py --probe temp/recurrence_v0_kai0/embeddings.npz \
        --cache temp/tcc_kai0_armmask/feat_cache --dataset kai0/data/Task_A/kai0_advantage --out temp/armmask
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.cluster import KMeans

ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True, help="V0 probe embeddings.npz(取同 50 ep 列表)")
ap.add_argument("--cache", required=True, help="armmask feat_cache")
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", default="temp/armmask")
ap.add_argument("--tag", default="kai0")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
ds = Path(args.dataset)
chunks_size = json.load(open(ds / "meta/info.json")).get("chunks_size", 1000)

zp = np.load(args.probe)
eps = sorted(set(zp["ep_ids"].tolist()))
print(f"[cmp] {len(eps)} eps (同 V0 探针)")

F, E, FR, T = [], [], [], []
for ep in eps:
    f = np.load(Path(args.cache) / f"ep{ep}.npz")["f"]
    n = len(f)
    F.append(f); E.append(np.full(n, ep))
    FR.append(np.arange(n) * 10); T.append(np.arange(n) / max(1, n - 1))
F = np.concatenate(F); E = np.concatenate(E); FR = np.concatenate(FR); T = np.concatenate(T)
print(f"[cmp] frames {F.shape}")

km = KMeans(n_clusters=48, n_init=4, random_state=0).fit(F)
lab = km.labels_
n_ep = len(eps)
cov = np.array([len(set(E[lab == c].tolist())) / n_ep for c in range(48)])
tpos = np.array([T[lab == c].mean() for c in range(48)])
spread = np.array([np.linalg.norm(F[lab == c] - km.cluster_centers_[c], axis=1).mean() for c in range(48)])
ms = sorted(np.argsort(cov)[-10:].tolist(), key=lambda c: tpos[c])
print("[cmp] milestones:", [(int(c), f"{cov[c]:.0%}", f"t={tpos[c]:.2f}", f"spr={spread[c]:.3f}") for c in ms])

# ② 偏置复查
from scipy.stats import pearsonr
r, _ = pearsonr(cov, spread)
sus = [c for c in np.argsort(cov)[-10:] if spread[c] < np.median(spread)]
print(f"[cmp] corr(cov,spread)={r:.3f}  可疑(高覆盖+低离散)簇: "
      f"{[(int(c), f'{cov[c]:.0%}', f't={tpos[c]:.2f}') for c in sus]}")

# ① GT 验证(τ)
taus = []
idx = {c: i + 1 for i, c in enumerate(ms)}
for ep in eps:
    m = np.where(E == ep)[0]
    pq = ds / "data" / f"chunk-{ep // chunks_size:03d}" / f"episode_{ep:06d}.parquet"
    try:
        gt = pd.read_parquet(pq, columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy()
    except Exception:
        continue
    fri = FR[m]; fri = fri[fri < len(gt)]
    g = gt[fri]
    v = np.zeros(len(fri)); passed = set()
    for j, i in enumerate(m[: len(fri)]):
        if lab[i] in idx:
            passed.add(lab[i])
        v[j] = len(passed) / len(ms)
    if g.std() < 1e-6:
        continue
    taus.append(kendalltau(v, g)[0])
print(f"\n========== ARMMASK GT VALIDATION ({args.tag}) ==========")
print(f"V_milestone tau: mean={np.nanmean(taus):.3f} median={np.nanmedian(taus):.3f}  (未掩膜基线 0.812/0.854)")

# ③ 最高覆盖簇代表帧
top = int(np.argmax(cov))
m = np.where(lab == top)[0]
d = np.linalg.norm(F[m] - km.cluster_centers_[top], axis=1)
seen, picks = set(), []
for i in m[np.argsort(d)]:
    if E[i] not in seen:
        seen.add(E[i]); picks.append((int(E[i]), int(FR[i])))
    if len(picks) == 4:
        break
print(f"[cmp] 最高覆盖簇 c{top}({cov[top]:.0%}, t={tpos[top]:.2f}) 代表帧: {picks}")
np.savez_compressed(out / f"compare_{args.tag}.npz", cov=cov, tpos=tpos, spread=spread,
                    milestones=np.array(ms), top_frames=np.array(picks))
