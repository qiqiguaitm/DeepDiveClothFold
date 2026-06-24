#!/usr/bin/env python
"""V0 GT 验证: 在 kai0_advantage(有 stage_progress_gt)上,用 V0 探针产物构造两种
零训练 value,并与 GT 进度对比(Kendall tau / Pearson):
  V_tpos(t)      = 帧所属簇的平均时间位置(soft 相位 proxy)
  V_milestone(t) = 已通过的 milestone 数 / M(milestone = top-M 覆盖率簇按时间排序;
                   "通过"=本 episode 中首次进入该簇)
对照: 现有 pi0-AE absolute_value 的 GT corr ≈0.896(文档口径), 差分 advantage ≈0.3-0.4。
用法: python recurrence_v0_gt_validation.py --probe-out temp/recurrence_v0_kai0 \
        --dataset kai0/data/Task_A/kai0_advantage --top-m 10
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr
from sklearn.cluster import KMeans

ap = argparse.ArgumentParser()
ap.add_argument("--probe-out", default="temp/recurrence_v0_kai0")
ap.add_argument("--dataset", default="kai0/data/Task_A/kai0_advantage")
ap.add_argument("--k", type=int, default=48)
ap.add_argument("--top-m", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

out, ds = Path(args.probe_out), Path(args.dataset)
z = np.load(out / "embeddings.npz")
feats, ep_ids, fr_idx, tnorm = z["feats"], z["ep_ids"], z["fr_idx"], z["tnorm"]
print(f"[gtval] feats {feats.shape}, eps {len(set(ep_ids.tolist()))}")

km = KMeans(n_clusters=args.k, n_init=4, random_state=args.seed).fit(feats)
lab = km.labels_
n_ep = len(set(ep_ids.tolist()))
cov = np.array([len(set(ep_ids[lab == c].tolist())) / n_ep for c in range(args.k)])
tpos = np.array([tnorm[lab == c].mean() for c in range(args.k)])

# milestones: top-M coverage, 按时间排序
ms = sorted(np.argsort(cov)[-args.top_m:], key=lambda c: tpos[c])
print(f"[gtval] milestones (cov,t): " + ", ".join(f"c{c}({cov[c]:.0%},{tpos[c]:.2f})" for c in ms))

chunks_size = json.load(open(ds / "meta" / "info.json")).get("chunks_size", 1000)

taus_tpos, taus_ms, rs_tpos, rs_ms = [], [], [], []
taus_lin = []  # 对照: 纯线性时间 (frame_idx 归一) — value 的 trivial 上界参考
for ep in sorted(set(ep_ids.tolist())):
    m = np.where(ep_ids == ep)[0]
    pq = ds / "data" / f"chunk-{ep // chunks_size:03d}" / f"episode_{ep:06d}.parquet"
    gt_all = pd.read_parquet(pq, columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy()
    gt = gt_all[fr_idx[m]]
    # V_tpos
    v1 = tpos[lab[m]]
    # V_milestone: 首次进入各 milestone 后 +1
    v2 = np.zeros(len(m))
    passed = set()
    for j, i in enumerate(m):
        if lab[i] in ms:
            passed.add(lab[i])
        v2[j] = len(passed) / len(ms)
    # 对照: 线性时间
    v3 = tnorm[m]
    if gt.std() < 1e-6:
        continue
    taus_tpos.append(kendalltau(v1, gt)[0]); rs_tpos.append(pearsonr(v1, gt)[0])
    taus_ms.append(kendalltau(v2, gt)[0]);   rs_ms.append(pearsonr(v2, gt)[0])
    taus_lin.append(kendalltau(v3, gt)[0])

def s(a): return f"mean={np.nanmean(a):.3f} median={np.nanmedian(a):.3f} min={np.nanmin(a):.3f}"
print("\n========== GT VALIDATION (vs stage_progress_gt) ==========")
print(f"eps evaluated: {len(taus_tpos)}")
print(f"V_tpos      Kendall tau: {s(taus_tpos)} | Pearson r: {s(rs_tpos)}")
print(f"V_milestone Kendall tau: {s(taus_ms)} | Pearson r: {s(rs_ms)}")
print(f"[对照] 线性时间 tau: {s(taus_lin)}  (gt 本身近似线性, 此为 trivial 上界)")
print("[对照] pi0-AE absolute_value GT corr≈0.896 (监督训练, 文档口径)")

# 图: 3 个随机 episode 的 V_milestone vs GT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
sample = random.Random(0).sample(sorted(set(ep_ids.tolist())), min(3, n_ep))
fig, axes = plt.subplots(1, len(sample), figsize=(5 * len(sample), 3.4))
for ax, ep in zip(np.atleast_1d(axes), sample):
    m = np.where(ep_ids == ep)[0]
    pq = ds / "data" / f"chunk-{ep // chunks_size:03d}" / f"episode_{ep:06d}.parquet"
    gt = pd.read_parquet(pq, columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy()[fr_idx[m]]
    v2 = np.zeros(len(m)); passed = set()
    for j, i in enumerate(m):
        if lab[i] in ms: passed.add(lab[i])
        v2[j] = len(passed) / len(ms)
    x = tnorm[m]
    ax.plot(x, gt, "k-", lw=1.5, label="stage_progress_gt")
    ax.step(x, v2, "r-", lw=1.5, where="post", label="V_milestone (zero-train)")
    ax.plot(x, tpos[lab[m]], "b.", ms=2, alpha=.5, label="V_tpos")
    ax.set_title(f"ep{ep}", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.suptitle("zero-train recurrence value vs GT (kai0_advantage)", fontsize=11)
fig.tight_layout(); fig.savefig(out / "gt_validation.png", dpi=120)
print(f"plot -> {out}/gt_validation.png")
