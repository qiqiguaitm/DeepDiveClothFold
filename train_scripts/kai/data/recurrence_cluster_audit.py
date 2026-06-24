#!/usr/bin/env python
"""聚类-审计可视化: 把覆盖率计算透明化, 供人工核对。
① occupancy raster: 每行一个 episode, x=归一时间, 着色=该帧命中的 milestone 簇
   → 右侧标注从图上直接重算的覆盖率, 与报告数字对账
② first-entry 帧网格: 选定簇在 12 个不同 episode 的"首入帧"原图
   → 肉眼核对: 是否同一语义状态(覆盖率的物理含义)
③ 单 episode 簇序列: 3 条 episode 的逐帧簇分配(全部 48 簇灰点 + milestone 高亮)
   → ep ↔ 簇 ↔ 时间 三者对应关系
用法: python recurrence_cluster_audit.py --tag smooth800   (或 kai0)
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="smooth800", choices=["smooth800", "kai0"])
ap.add_argument("--audit-cluster-rank", type=int, default=1, help="审计第几高覆盖簇(1=最高)")
args = ap.parse_args()

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
CFG = {
    "smooth800": dict(probe="temp/recurrence_v0/embeddings.npz",
                      cache="temp/tcc_smooth800_armmask/feat_cache",
                      ds="kai0/data/Task_A/self_built/A_new_smooth_800/base"),
    "kai0": dict(probe="temp/recurrence_v0_kai0/embeddings.npz",
                 cache="temp/tcc_kai0_armmask/feat_cache",
                 ds="kai0/data/Task_A/kai0_advantage"),
}[args.tag]
ds = REPO / CFG["ds"]
cache = REPO / CFG["cache"]
chunks_size = json.load(open(ds / "meta/info.json")).get("chunks_size", 1000)

# ---- 挖掘(armmask 特征, V0 同 50 ep, 确定性) ----
zp = np.load(REPO / CFG["probe"])
mined = sorted(set(zp["ep_ids"].tolist()))
F, E, T, FR = [], [], [], []
for ep in mined:
    f = np.load(cache / f"ep{ep}.npz")["f"]
    n = len(f)
    F.append(f); E.append(np.full(n, ep)); T.append(np.arange(n) / max(1, n - 1)); FR.append(np.arange(n) * 10)
F = np.concatenate(F); E = np.concatenate(E); T = np.concatenate(T); FR = np.concatenate(FR)
km = KMeans(n_clusters=48, n_init=4, random_state=0).fit(F)
lab = km.labels_
n_ep = len(mined)
cov = np.array([len(set(E[lab == c].tolist())) / n_ep for c in range(48)])
tpos = np.array([T[lab == c].mean() for c in range(48)])
ms = sorted(np.argsort(cov)[-10:].tolist(), key=lambda c: tpos[c])
print(f"[{args.tag}] milestones:", [(int(c), f"{cov[c]:.0%}", f"t={tpos[c]:.2f}") for c in ms])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
OUT = REPO / "temp/recurrence_v0"
mscolors = cm.get_cmap("tab10")

# ====== ① occupancy raster (50 ep × 时间, milestone 着色) ======
fig, ax = plt.subplots(figsize=(14, 0.22 * n_ep + 2.4))
for row, ep in enumerate(mined):
    m = np.where(E == ep)[0]
    ax.plot([0, 1], [row, row], "-", color="#eee", lw=3, zorder=0)        # 背景轨
    for k, c in enumerate(ms):
        hit = m[lab[m] == c]
        if len(hit):
            ax.scatter(T[hit], np.full(len(hit), row), s=10, marker="s",
                       color=mscolors(k % 10), zorder=2)
# 右侧: 图上重算覆盖率 vs 报告值
for k, c in enumerate(ms):
    eps_hit = len(set(E[lab == c].tolist()))
    ax.text(1.02, n_ep - 2.6 * k - 1, f"M{k+1}=c{c}: {eps_hit}/{n_ep} ep = {eps_hit/n_ep:.0%}",
            color=mscolors(k % 10), fontsize=8, transform=ax.get_yaxis_transform())
ax.set_xlim(0, 1.0); ax.set_ylim(-1, n_ep)
ax.set_xlabel("normalized time"); ax.set_ylabel("episode (row, 50 mined eps)")
ax.set_title(f"{args.tag} (armmask): milestone occupancy raster - each row = 1 episode, colored = frame hits a milestone cluster\n"
             f"right margin = coverage recomputed FROM THIS PLOT (should match reported numbers)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / f"audit_raster_{args.tag}.png", dpi=120)
print("saved", OUT / f"audit_raster_{args.tag}.png")

# ====== ② first-entry 帧网格(选定簇, 12 个不同 episode) ======
import av
rank = args.audit_cluster_rank
c_audit = sorted(ms, key=lambda c: -cov[c])[rank - 1]
k_audit = ms.index(c_audit)
firsts = []   # (ep, raw_frame, tnorm)
for ep in mined:
    m = np.where(E == ep)[0]
    hit = m[lab[m] == c_audit]
    if len(hit):
        i = hit[0]
        firsts.append((int(ep), int(FR[i]), float(T[i])))
print(f"audit cluster c{c_audit} (M{k_audit+1}, cov={cov[c_audit]:.0%}): {len(firsts)}/{n_ep} eps 首入")
rng = np.random.RandomState(0)
sel = [firsts[i] for i in rng.choice(len(firsts), min(12, len(firsts)), replace=False)]
sel = sorted(sel)
def cam_path(ep):
    for cam in ("observation.images.top_head", "top_head"):
        p = ds / "videos" / f"chunk-{ep // chunks_size:03d}" / cam / f"episode_{ep:06d}.mp4"
        if p.is_file():
            return p
fig, axes = plt.subplots(3, 4, figsize=(13, 7.4))
for ax_, (ep, fr, tn) in zip(axes.flat, sel):
    cont = av.open(str(cam_path(ep)))
    for i, f in enumerate(cont.decode(video=0)):
        if i == fr:
            ax_.imshow(f.to_ndarray(format="rgb24")); break
    cont.close()
    ax_.set_title(f"ep{ep} f{fr} (t={tn:.2f})", fontsize=8)
    ax_.axis("off")
fig.suptitle(f"{args.tag}: cluster c{c_audit} (M{k_audit+1}, coverage {cov[c_audit]:.0%}, t={tpos[c_audit]:.2f}) first-entry frames in 12 DIFFERENT episodes\n"
             f"visual check: same semantic state? (= physical meaning of coverage)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / f"audit_firstentry_{args.tag}_c{c_audit}.png", dpi=115)
print("saved", OUT / f"audit_firstentry_{args.tag}_c{c_audit}.png")

# ====== ③ 单 episode 簇序列(3 条) ======
sel_eps = [mined[i] for i in (5, 20, 40)]
fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
for ax_, ep in zip(axes, sel_eps):
    m = np.where(E == ep)[0]
    ax_.scatter(T[m], lab[m], s=6, c="#bbb", label="other clusters")
    for k, c in enumerate(ms):
        hit = m[lab[m] == c]
        if len(hit):
            ax_.scatter(T[hit], lab[hit], s=26, color=mscolors(k % 10), label=f"M{k+1}=c{c}")
    ax_.set_ylabel(f"ep{ep}\ncluster id"); ax_.grid(alpha=.2)
    ax_.legend(fontsize=6, ncol=6, loc="upper left")
axes[-1].set_xlabel("normalized time")
fig.suptitle(f"{args.tag}: per-frame cluster assignment of single episodes (gray = all 48 clusters, colored = milestones) - ep / cluster / time correspondence", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / f"audit_epsequence_{args.tag}.png", dpi=120)
print("saved", OUT / f"audit_epsequence_{args.tag}.png")
