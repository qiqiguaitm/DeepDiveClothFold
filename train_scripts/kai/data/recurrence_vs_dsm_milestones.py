#!/usr/bin/env python
"""自动挖掘 milestone vs ViVa-DSM 手标 milestone — 同 episode 直接对比 (task_a_0509v2, 30 eps).
手标: /vePFS/zundong/ViVa/data/milestones_task_a_0509v2.jsonl (每 ep 5 个 milestone start_frame)
自动: V0 探针協議 (DINOv2+KMeans 覆盖率) 在同 30 eps 上挖 top-M 簇, 取每 ep 首入帧。
指标: 每个手标边界 与 最近自动 milestone 首入帧 的 |Δt|(归一);可视化 overlay。
前置: 先跑 probe --dataset kai0/data/Task_A/vis_base/v2/2026-05-09-v2 --n-episodes 30 --out temp/recurrence_v0_0509
"""
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
PROBE = REPO / "temp/recurrence_v0_0509"
DSM = Path("/vePFS/zundong/ViVa/data/milestones_task_a_0509v2.jsonl")
K, TOP_M, SEED = 48, 10, 0

z = np.load(PROBE / "embeddings.npz")
feats, ep_ids, fr_idx, tnorm = z["feats"], z["ep_ids"], z["fr_idx"], z["tnorm"]
km = KMeans(n_clusters=K, n_init=4, random_state=SEED).fit(feats)
lab = km.labels_
n_ep = len(set(ep_ids.tolist()))
cov = np.array([len(set(ep_ids[lab == c].tolist())) / n_ep for c in range(K)])
tpos = np.array([tnorm[lab == c].mean() for c in range(K)])
ms = sorted(np.argsort(cov)[-TOP_M:], key=lambda c: tpos[c])
print("auto milestones:", [(int(c), f"{cov[c]:.0%}", f"t={tpos[c]:.2f}") for c in ms])

hand = {json.loads(l)["episode_idx"]: json.loads(l)["milestones"] for l in open(DSM)}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
deltas = []
fig, ax = plt.subplots(figsize=(13, 7))
eps_sorted = sorted(set(ep_ids.tolist()))
for row, ep in enumerate(eps_sorted):
    m = np.where(ep_ids == ep)[0]
    ep_len = fr_idx[m].max() + 1
    # 自动: 每个 milestone 簇的首入帧
    auto_t = []
    for c in ms:
        hits = m[lab[m] == c]
        if len(hits):
            auto_t.append(fr_idx[hits].min() / ep_len)
    # 手标: start_frame (跳过第0个=episode开头)
    hd = hand.get(ep, [])
    hand_t = [mm["start_frame"] / ep_len for mm in hd[1:]] if hd else []
    ax.scatter(auto_t, [row] * len(auto_t), c="tab:purple", s=14, marker="o",
               label="auto (recurrence)" if row == 0 else None)
    ax.scatter(hand_t, [row] * len(hand_t), c="k", s=40, marker="|",
               label="hand (DSM)" if row == 0 else None)
    for ht in hand_t:
        if auto_t:
            deltas.append(min(abs(ht - a) for a in auto_t))
ax.set_xlabel("normalized time"); ax.set_ylabel("episode")
ax.set_title("auto-mined milestones (purple) vs hand-annotated DSM boundaries (black) — task_a_0509v2")
ax.legend(loc="upper right"); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(PROBE / "auto_vs_hand_milestones.png", dpi=120)
deltas = np.array(deltas)
print(f"\nhand boundaries matched: n={len(deltas)}")
print(f"|Δt| to nearest auto milestone: mean={deltas.mean():.3f} median={np.median(deltas):.3f} "
      f"p90={np.percentile(deltas,90):.3f} (单位=episode 长度比例)")
print(f"≤0.05 (≈3s/min) 命中率: {(deltas<=0.05).mean():.0%}   ≤0.10: {(deltas<=0.10).mean():.0%}")
print("plot ->", PROBE / "auto_vs_hand_milestones.png")
