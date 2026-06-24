#!/usr/bin/env python
"""单个 episode 在 CRAVE 特征空间(3D PCA)的逐帧簇间流转 —— 静态图 + 旋转彗星视频。
复用 crave_cluster_flow_viz.py 落盘的高维数组(temp/_crave_flow_hd_ep{EP}.npz),
PCA(3) fit 在 milestone 中心(PC1≈进度轴)。视频:相机帧(左) + 3D 特征空间(右,缓慢旋转,
彗星=当前帧+拖尾,★=milestone簇按进度着色,蓝圈=当前最近 milestone)。

用法: python crave/experiments/crave_cluster_flow_3d.py [--ep 2302] [--no-video]

Thin entrypoint over `crave`: paths/kai0_base from crave.config; mpl from crave.render.
Reads the high-dim arrays dumped by crave_cluster_flow_viz.py (no re-mining); the PCA(3)
+ rotating-comet video composition stays verbatim.
"""
import argparse
from pathlib import Path

import numpy as np, pandas as pd, matplotlib, cv2
from sklearn.decomposition import PCA

from crave.config import REPO, resolve_dataset
from crave.data import kai0
from crave.render import setup_mpl

plt = setup_mpl()

BASE_CFG = resolve_dataset("kai0_base")
BASE = Path(BASE_CFG.root)
csB = kai0.chunks_size(str(BASE))
ap = argparse.ArgumentParser(); ap.add_argument("--ep", type=int, default=2302); ap.add_argument("--no-video", action="store_true")
A = ap.parse_args(); EP = A.ep

z = np.load(REPO / f"temp/_crave_flow_hd_ep{EP}.npz")
Fq, Gs, allC, C, Pord, lab_idx, tpos, near_ms = (z["Fq"], z["Gs"], z["allC"], z["C"], z["Pord"],
                                                 z["lab_idx"], z["tpos"], z["near_ms"])
n = len(Fq); tcol = np.arange(n) / max(1, n - 1)
pca = PCA(3, random_state=0).fit(C)
if np.corrcoef(pca.transform(C)[:, 0], Pord)[0, 1] < 0: pca.components_[0] *= -1
G3 = pca.transform(Gs); C3 = pca.transform(C); F3raw = pca.transform(Fq)


def smooth(a, w=5):
    h = w // 2; return np.array([a[max(0, j - h):j + h + 1].mean(0) for j in range(len(a))])


F3 = smooth(F3raw, 5)
vir = matplotlib.colormaps["viridis"]; plas = matplotlib.colormaps["plasma"]


def setup_ax(ax):
    ax.scatter(G3[:, 0], G3[:, 1], G3[:, 2], s=3, c=tpos[lab_idx], cmap="Greys", alpha=.18, vmin=0, vmax=1)
    ax.scatter(C3[:, 0], C3[:, 1], C3[:, 2], s=240, c=Pord, cmap="viridis", marker="*",
               edgecolor="k", linewidth=1.0, depthshade=False, zorder=5)
    ax.set_xlabel("PC1 (≈进度)", fontsize=8); ax.set_ylabel("PC2", fontsize=8); ax.set_zlabel("PC3", fontsize=8)
    ax.tick_params(labelsize=6)


# ---- 静态图: 两个角度 + 轨迹 ----
fig = plt.figure(figsize=(15, 7))
for k, (el, az) in enumerate([(18, -60), (22, 40)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d"); setup_ax(ax)
    ax.plot(F3[:, 0], F3[:, 1], F3[:, 2], color="#555", lw=0.7, alpha=.6)
    ax.scatter(F3[:, 0], F3[:, 1], F3[:, 2], s=10, c=tcol, cmap="plasma", depthshade=False, zorder=4)
    ax.scatter(*F3[0], s=160, c="lime", edgecolor="k", marker="o", depthshade=False, zorder=8)
    ax.scatter(*F3[-1], s=160, c="red", edgecolor="k", marker="s", depthshade=False, zorder=8)
    ax.view_init(elev=el, azim=az); ax.set_title(f"视角 {k+1} (elev{el},azim{az})", fontsize=9)
fig.suptitle(f"CRAVE 3D 特征空间(PCA)簇间流转 — kai0_base ep{EP} · ★=milestone(色=进度) · 轨迹紫→黄", fontsize=12)
out = REPO / f"crave/docs/visualization/crave_cluster_flow_3d_ep{EP}.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print("SAVED", out, flush=True); plt.close(fig)
print("FLOW3D_STATIC_DONE", flush=True)
if A.no_video:
    raise SystemExit

# ---- 旋转彗星视频 ----
import av
vid = BASE / "videos" / f"chunk-{EP//csB:03d}" / "observation.images.top_head" / f"episode_{EP:06d}.mp4"
NFv = len(pd.read_parquet(BASE / "data" / f"chunk-{EP//csB:03d}" / f"episode_{EP:06d}.parquet", columns=["frame_index"]))


def fidx(i): return min(i // 10, n - 1)


# 视频背景下采样(加速 + 更干净)
_sub = np.random.RandomState(1).permutation(len(G3))[:800]
Gbg = G3[_sub]; Cbg = tpos[lab_idx][_sub]
PFIG = plt.figure(figsize=(7.2, 7.0), dpi=100); axp = PFIG.add_subplot(111, projection="3d")
xl = (G3[:, 0].min(), G3[:, 0].max()); yl = (G3[:, 1].min(), G3[:, 1].max()); zl = (G3[:, 2].min(), G3[:, 2].max())
TRAIL = 90
c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
# 先估面板尺寸
axp.scatter([0], [0], [0]); PFIG.canvas.draw()
PH, PW = np.asarray(PFIG.canvas.buffer_rgba()).shape[:2]
csc = PH / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2
Wtot = (cw2 + PW) // 2 * 2; Htot = PH // 2 * 2
out_mp4 = str(REPO / f"temp/crave_cluster_flow_3d_ep{EP}.mp4")
oc = av.open(out_mp4, mode="w"); stv = oc.add_stream("libx264", rate=30)
stv.width, stv.height, stv.pix_fmt = Wtot, Htot, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
print(f"canvas {Wtot}x{Htot} frames~{NFv}", flush=True)


def render_panel(fi, azim):
    axp.clear()
    axp.scatter(Gbg[:, 0], Gbg[:, 1], Gbg[:, 2], s=4, c=Cbg, cmap="Greys", alpha=.18, vmin=0, vmax=1)
    axp.scatter(C3[:, 0], C3[:, 1], C3[:, 2], s=230, c=Pord, cmap="viridis", marker="*",
                edgecolor="k", linewidth=1.0, depthshade=False, zorder=5)
    s0 = max(0, fi - TRAIL)
    if fi > s0:
        seg = F3[s0:fi + 1]
        axp.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#d62728", lw=1.6, alpha=.8, zorder=6)
        axp.scatter(seg[:, 0], seg[:, 1], seg[:, 2], s=12, c=tcol[s0:fi + 1], cmap="plasma", depthshade=False, zorder=6)
    axp.scatter(*F3[fi], s=160, c="red", edgecolor="k", depthshade=False, zorder=9)
    ms = near_ms[fi]
    axp.scatter(*C3[ms], s=520, facecolor="none", edgecolor="#1f77ff", linewidth=2.4, depthshade=False, zorder=10)
    axp.set_xlim(xl); axp.set_ylim(yl); axp.set_zlim(zl)
    axp.set_xlabel("PC1 (≈进度)", fontsize=8); axp.set_ylabel("PC2", fontsize=8); axp.set_zlabel("PC3", fontsize=8)
    axp.tick_params(labelsize=6); axp.view_init(elev=18, azim=azim)
    axp.set_title(f"ep{EP} 3D 特征空间流转 · 当前→milestone {Pord[ms]:.0%}", fontsize=10)
    PFIG.canvas.draw()
    return np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy()


import time as _t; t0 = _t.time()
c = av.open(str(vid)); i = 0; mid = NFv // 2
for fr in c.decode(video=0):
    if i >= NFv: break
    az = -60 + i * (120.0 / NFv)                    # 整段缓慢转 120°
    panel = render_panel(fidx(i), az)
    if panel.shape[:2] != (PH, PW):
        panel = cv2.resize(panel, (PW, PH))
    cam = fr.to_ndarray(format="rgb24"); cam2 = cv2.resize(np.ascontiguousarray(cam), (cw2, PH))
    cv2.rectangle(cam2, (6, 6), (320, 56), (0, 0, 0), -1)
    cv2.putText(cam2, f"frame {i}/{NFv}  ->milestone {Pord[near_ms[fidx(i)]]:.0%}", (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    canv = np.zeros((PH, cw2 + PW, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel
    frame = np.ascontiguousarray(canv[:Htot, :Wtot])
    if i == mid: cv2.imwrite(out_mp4.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for pkt in stv.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")): oc.mux(pkt)
    i += 1
    if i % 500 == 0: print(f"  {i}/{NFv} ({i/(_t.time()-t0):.0f} fps)", flush=True)
c.close()
for pkt in stv.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {out_mp4} {i}f 用时{_t.time()-t0:.0f}s", flush=True); print("FLOW3D_VIDEO_DONE", flush=True)
