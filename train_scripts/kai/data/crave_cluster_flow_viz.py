#!/usr/bin/env python
"""可视化单个 episode 在 CRAVE 特征空间里如何从一个簇流转到另一个簇。
- PCA(2D) 投影:背景=挖矿帧(按所属簇 progress 着色,显示特征空间的进度梯度);
  milestone 簇中心=星标(按 P_k 着色+标进度);目标 episode 轨迹=按帧时间着色的连线+方向箭头。
- 右侧"milestone 流转"阶梯:逐帧最近 milestone 的 P_k 随时间爬升。
- --video: 相机帧(左)+ PCA 空间彗星(当前点+拖尾)+ 当前所在 milestone 高亮(右)同步视频。

用法: python crave_cluster_flow_viz.py [--ep 2302] [--video]
CRAVE 挖矿核心与 crave_vs_ae_kai0base / smooth800_v24_full 逐字一致(kai0_base+dagger 缓存)。
"""
import argparse, json, os
import numpy as np, pandas as pd, matplotlib, cv2
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
FC = REPO / "temp/crave_kai0bd/feat_cache"
BASE = REPO / "kai0/data/Task_A/kai0_base"
csB = json.load(open(BASE / "meta/info.json"))["chunks_size"]
ap = argparse.ArgumentParser(); ap.add_argument("--ep", type=int, default=2302); ap.add_argument("--video", action="store_true")
ARGS = ap.parse_args(); EP = ARGS.ep


def loadep(e):
    d = np.load(FC / f"ep{e}.npz"); a, r, s = d["armmask"], d["raw"], d["state"]
    n = min(len(a), len(r), len(s)); return a[:n], r[:n], s[:n], n


def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
mined = eps
Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st):
    an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
    Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
    return np.concatenate([rn, an, Pn], 1)


A, R, S, T, E = [], [], [], [], []
for e in mined:
    aa, rr, st, n = loadep(e); A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
G = emb(A, R, S)
km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
N = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
Pstart = {}
for e in sorted(set(E.tolist())):
    m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Pstart[e] = float(np.median(tpos[nn]))
cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / N) for c in range(96)])
bk = np.linspace(0, 1, 11); sel = []
for b in range(10):
    inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
    if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:2]
sel = sorted(set(sel), key=lambda c: tpos[c])


def gr(idx):
    o = []; s = None; pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s is not None: o.append((s, pv))
            s = i
        pv = i
    if s is not None: o.append((s, pv))
    return [x for x in o if x[1] - x[0] >= 1]


Pk = {}
for c in sel:
    fe = []
    for e in sorted(set(E.tolist())):
        m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
        if rs: fe.append(T[rs[0][0]])
    Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]; Pord = np.array([Pk[c] for c in order])
print(f"milestones {len(order)}", flush=True)

# ---- PCA(2D): fit on milestone 中心(progress 有序) → PC1≈进度轴, 布局干净 ----
rs = np.random.RandomState(0); idx = rs.permutation(len(G))[:6000]
pca = PCA(2, random_state=0).fit(C)
if np.corrcoef(pca.transform(C)[:, 0], Pord)[0, 1] < 0: pca.components_[0] *= -1
G2 = pca.transform(G[idx]); allC2 = pca.transform(allC); C2 = pca.transform(C)

# ---- 目标 episode 轨迹 ----
aa, rr, st, n = loadep(EP); Fq = emb(aa, rr, st); Fq2raw = pca.transform(Fq)
d2C = np.linalg.norm(Fq[:, None] - C[None], axis=2); near_ms_raw = d2C.argmin(1)


def smooth(a, w=5):
    h = w // 2; return np.array([a[max(0, j - h):j + h + 1].mean(0) for j in range(len(a))])


def medf(a, w=9):
    h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


Fq2 = smooth(Fq2raw, 5)                                   # 显示轨迹平滑(降视觉缠绕)
near_ms = np.round(medf(near_ms_raw.astype(float), 9)).astype(int)   # 流转时间线去噪(中值滤波)
tcol = np.arange(n) / max(1, n - 1)

# ---- 静态图 ----
fig = plt.figure(figsize=(16, 7.5)); gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1, 0.04], wspace=0.18)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(G2[:, 0], G2[:, 1], s=4, c=tpos[lab[idx]], cmap="Greys", alpha=.25, vmin=0, vmax=1)
sc_ms = ax.scatter(allC2[:, 0], allC2[:, 1], s=20, c=tpos, cmap="viridis", alpha=.35, marker="o")
ax.scatter(C2[:, 0], C2[:, 1], s=320, c=Pord, cmap="viridis", marker="*", edgecolor="k", linewidth=1.2, zorder=5)
for i, (xy, p) in enumerate(zip(C2, Pord)):
    ax.annotate(f"{p:.0%}", xy, fontsize=7.5, ha="center", va="center", zorder=6, color="k")
# 轨迹: 时间着色连线 + 箭头
ln = ax.scatter(Fq2[:, 0], Fq2[:, 1], s=14, c=tcol, cmap="plasma", zorder=4, alpha=.9)
ax.plot(Fq2[:, 0], Fq2[:, 1], color="#444", lw=0.6, alpha=.5, zorder=3)
for j in range(0, n - 1, max(1, n // 18)):
    ax.annotate("", Fq2[j + 1], Fq2[j], arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.3, alpha=.8), zorder=7)
ax.scatter(*Fq2[0], s=180, c="lime", edgecolor="k", marker="o", zorder=8, label="起点")
ax.scatter(*Fq2[-1], s=180, c="red", edgecolor="k", marker="s", zorder=8, label="终点")
ax.set_xlabel("PC1 (≈任务进度轴)"); ax.set_ylabel("PC2"); ax.legend(fontsize=9, loc="upper left")
ax.set_title(f"ep{EP} 在 CRAVE 特征空间的簇间流转\n★=milestone簇(色/标=进度P_k) · 轨迹按帧时间着色(紫→黄) · 红箭头=方向", fontsize=11)
cax = fig.add_subplot(gs[0, 2]); plt.colorbar(ln, cax=cax, label="帧时间(归一)")
# milestone 流转阶梯
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(np.arange(n), Pord[near_ms], drawstyle="steps-post", color="#1f77b4", lw=1.0, alpha=.5, label="逐帧最近 milestone 的 P_k")
ax2.scatter(np.arange(n), Pord[near_ms], s=10, c=tcol, cmap="plasma", zorder=4)
ax2.set_ylim(-0.03, 1.03); ax2.set_xlabel("frame"); ax2.set_ylabel("最近 milestone 的进度 P_k")
ax2.set_title("milestone 流转(逐帧最近簇的进度)\n随时间从低进度簇爬到高进度簇", fontsize=10); ax2.grid(alpha=.25); ax2.legend(fontsize=8, loc="lower right")
fig.suptitle(f"CRAVE 特征空间簇间流转可视化 — kai0_base ep{EP}", fontsize=13, y=0.99)
out = REPO / f"docs/visualization/cross_episode_recurrence_value/crave_cluster_flow_ep{EP}.png"
fig.savefig(out, dpi=125, bbox_inches="tight"); print("SAVED", out, flush=True)
plt.close(fig)

np.savez(REPO / f"temp/_crave_flow_ep{EP}.npz", Fq2=Fq2, G2=G2, allC2=allC2, C2=C2, Pord=Pord,
         tpos_lab=tpos[lab[idx]], near_ms=near_ms, tcol=tcol)
# 高维数组(供 3D 脚本复用, 免重新挖矿)
np.savez(REPO / f"temp/_crave_flow_hd_ep{EP}.npz", Fq=Fq, Gs=G[idx], allC=allC, C=C, Pord=Pord,
         lab_idx=lab[idx], tpos=tpos, near_ms=near_ms)
print("FLOW_STATIC_DONE", flush=True)

if not ARGS.video:
    raise SystemExit

# ---- 彗星视频: 相机 + PCA 空间移动点+拖尾 ----
import av
vid = BASE / "videos" / f"chunk-{EP//csB:03d}" / "observation.images.top_head" / f"episode_{EP:06d}.mp4"
NFv = len(pd.read_parquet(BASE / "data" / f"chunk-{EP//csB:03d}" / f"episode_{EP:06d}.parquet", columns=["frame_index"]))
# 帧→3Hz 索引映射(特征 3Hz)
def fidx(t): return min(int(t / 10), n - 1)
# 背景面板画一次
PFIG = plt.figure(figsize=(8, 7.5), dpi=100); axp = PFIG.add_subplot(111)
axp.scatter(G2[:, 0], G2[:, 1], s=4, c=tpos[lab[idx]], cmap="Greys", alpha=.22, vmin=0, vmax=1)
axp.scatter(C2[:, 0], C2[:, 1], s=300, c=Pord, cmap="viridis", marker="*", edgecolor="k", lw=1.1, zorder=5)
for xy, p in zip(C2, Pord): axp.annotate(f"{p:.0%}", xy, fontsize=7, ha="center", va="center", zorder=6)
axp.plot(Fq2[:, 0], Fq2[:, 1], color="#bbb", lw=0.6, alpha=.5, zorder=2)
axp.set_xlabel("PC1 (≈进度)"); axp.set_ylabel("PC2"); axp.set_title(f"ep{EP} CRAVE 特征空间簇间流转(彗星=当前帧)", fontsize=11)
xl, yl = axp.get_xlim(), axp.get_ylim()
PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]
bb = axp.get_position()


def to_px(p2):
    fx = (p2[0] - xl[0]) / (xl[1] - xl[0]); fy = (p2[1] - yl[0]) / (yl[1] - yl[0])
    return int(round((bb.x0 + fx * (bb.x1 - bb.x0)) * Wp)), int(round((1 - (bb.y0 + fy * (bb.y1 - bb.y0))) * Hp))
plt.close(PFIG)
TRJ = [to_px(p) for p in Fq2]; MS = [to_px(p) for p in C2]
c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2
Wtot = (cw2 + Wp) // 2 * 2; Htot = Hp // 2 * 2
out_mp4 = str(REPO / f"temp/crave_cluster_flow_ep{EP}.mp4")
oc = av.open(out_mp4, mode="w"); stv = oc.add_stream("libx264", rate=30)
stv.width, stv.height, stv.pix_fmt = Wtot, Htot, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
cmap = matplotlib.colormaps["plasma"]
c = av.open(str(vid)); i = 0; mid = NFv // 2
for fr in c.decode(video=0):
    if i >= NFv: break
    fi = fidx(i); panel = PANEL.copy()
    # 拖尾(最近100特征帧)
    for k in range(max(0, fi - 100), fi):
        col = tuple(int(255 * x) for x in cmap(tcol[k])[:3]); cv2.circle(panel, TRJ[k], 3, col, -1)
    cv2.circle(panel, TRJ[fi], 9, (255, 30, 30), -1); cv2.circle(panel, TRJ[fi], 9, (0, 0, 0), 1)
    # 当前最近 milestone 高亮
    cv2.circle(panel, MS[near_ms[fi]], 16, (30, 120, 255), 2)
    cam = fr.to_ndarray(format="rgb24"); cam2 = cv2.resize(np.ascontiguousarray(cam), (cw2, Hp))
    cv2.rectangle(cam2, (6, 6), (300, 56), (0, 0, 0), -1)
    cv2.putText(cam2, f"frame {i}/{NFv}  ->milestone {Pord[near_ms[fi]]:.0%}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel
    frame = np.ascontiguousarray(canv[:Htot, :Wtot])
    if i == mid: cv2.imwrite(out_mp4.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for pkt in stv.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")): oc.mux(pkt)
    i += 1
    if i % 1500 == 0: print(f"  {i}/{NFv}", flush=True)
c.close()
for pkt in stv.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {out_mp4} {i}f", flush=True); print("FLOW_VIDEO_DONE", flush=True)
