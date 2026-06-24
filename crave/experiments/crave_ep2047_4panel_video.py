#!/usr/bin/env python
"""ep2047 同步视频: 相机帧(左) + 四条 value 信号各自独立面板(右, 4 行不叠), 游标同步。
四条: 3Hz离散(AWBC用) / 30Hz离散·同4s窗 / 30Hz离散·轻平滑 / 30Hz连续soft(跟细过程·仅监控)。
复用 crave_30hz_test.py 的挖矿+信号(exec 其绘图前部分), cv2-overlay 渲染。
"""
import os
import numpy as np, pandas as pd, cv2, av

from crave.config import REPO
from crave.render import setup_mpl

plt = setup_mpl()

# 复用挖矿核心 + emission/dp_fixedlag/soft_progress/loadnpz/FC/FC30 (不跑它的绘图)
# TODO(crave-lib): the 3Hz/30Hz time-binned 96-cluster mining core in crave_30hz_test.py
#   (loadnpz/FC/FC30/emb/emission/dp_fixedlag/med_causal/soft_progress/C/Pord) is a bespoke
#   pipeline distinct from crave.clustering.build_clusters / crave.value.readout_* and has no
#   library equivalent; reuse it verbatim via exec (same as the legacy script). It defines its own
#   REPO (== crave.config.REPO).
_SRC = REPO / "train_scripts/kai/data/crave_30hz_test.py"
src = open(_SRC).read().split("fig, axs")[0]
exec(src)

EP = 2047
BASE = REPO / "kai0/data/Task_A/kai0_base"; cs = 1000
vid = BASE / f"videos/chunk-{EP//cs:03d}/observation.images.top_head/episode_{EP:06d}.mp4"

aa3, rr3, st3, n3 = loadnpz(FC, EP); em3, _ = emission(aa3, rr3, st3, 1); v3 = dp_fixedlag(em3, 12)
aa30, rr30, st30, n30 = loadnpz(FC30, EP); em30, d30 = emission(aa30, rr30, st30, 10)
v30t = dp_fixedlag(em30, 120); v30l = dp_fixedlag(em30, 24); soft30 = med_causal(soft_progress(d30), 30)
NF = len(pd.read_parquet(BASE / f"data/chunk-{EP//cs:03d}/episode_{EP:06d}.parquet", columns=["frame_index"]))
NF = min(NF, n30)
v3_30 = np.repeat(v3, 10)
if len(v3_30) < NF: v3_30 = np.concatenate([v3_30, np.full(NF - len(v3_30), v3_30[-1])])
sigs = [(v3_30[:NF], "3Hz 离散 (AWBC 用的 value 形状)", "#2ca02c"),
        (v30t[:NF], "30Hz 离散 · 同 4s 窗(解 3Hz 混叠 → 多走几阶)", "#1f77ff"),
        (v30l[:NF], "30Hz 离散 · 轻平滑(更敏感)", "#ff7f0e"),
        (soft30[:NF], "30Hz 连续 soft · 跟细过程(仅监控, 非 AWBC 排序)", "#d62728")]
FPS = 30.0; t = np.arange(NF) / FPS
print(f"ep{EP} NF={NF}", flush=True)

# ---- 背景 4 面板画一次 ----
PFIG = plt.figure(figsize=(9, 8), dpi=100); gs = PFIG.add_gridspec(4, 1, hspace=0.55)
AX = []
for i, (sig, title, col) in enumerate(sigs):
    ax = PFIG.add_subplot(gs[i]); ax.plot(t, sig, color=col, lw=1.4)
    ax.set_xlim(0, NF / FPS); ax.set_ylim(-.05, 1.07); ax.set_ylabel("value", fontsize=8)
    ax.set_title(title, fontsize=9.5, color=col); ax.grid(alpha=.25); ax.tick_params(labelsize=7)
    if i == 3: ax.set_xlabel("秒", fontsize=8)
    AX.append((ax, col, sig))
PFIG.suptitle(f"kai0_base ep{EP} — CRAVE 四种 value 读出 (3Hz 离散 / 30Hz 离散×2 / 30Hz 连续)", fontsize=11)
PFIG.canvas.draw()
PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]


def pmap(ax):
    bb = ax.get_position(); xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
    return bb.x0, bb.x1, bb.y0, bb.y1, xlo, xhi, ylo, yhi


MAPS = [pmap(ax) for ax, _, _ in AX]


def px_xy(m, sec, val):
    x0, x1, y0, y1, xlo, xhi, ylo, yhi = m
    return (int(round((x0 + (sec - xlo) / (xhi - xlo) * (x1 - x0)) * Wp)),
            int(round((1 - (y0 + (val - ylo) / (yhi - ylo) * (y1 - y0))) * Hp)))


def yspan(m):
    _, _, y0, y1, _, _, _, _ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))


SPANS = [yspan(m) for m in MAPS]
plt.close(PFIG)

c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2
Wtot = (cw2 + Wp) // 2 * 2; Htot = Hp // 2 * 2
out_mp4 = str(REPO / f"temp/crave_ep{EP}_4panel.mp4")
oc = av.open(out_mp4, mode="w"); stv = oc.add_stream("libx264", rate=30)
stv.width, stv.height, stv.pix_fmt = Wtot, Htot, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}


def b2rgb(hexc): h = hexc.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


COLS = [b2rgb(c) for _, c, _ in AX]
print(f"canvas {Wtot}x{Htot}", flush=True)
c = av.open(str(vid)); i = 0; mid = NF // 2
for fr in c.decode(video=0):
    if i >= NF: break
    panel = PANEL.copy(); sec = i / FPS
    for (m, (yt, yb), (_, _, sig), col) in zip(MAPS, SPANS, AX, COLS):
        px = px_xy(m, sec, 0)[0]
        cv2.line(panel, (px, yt), (px, yb), (120, 120, 120), 1)
        vx, vy = px_xy(m, sec, float(sig[min(i, NF - 1)]))
        cv2.circle(panel, (vx, vy), 6, col, -1); cv2.circle(panel, (vx, vy), 6, (0, 0, 0), 1)
    cam2 = cv2.resize(np.ascontiguousarray(fr.to_ndarray(format="rgb24")), (cw2, Hp))
    cv2.rectangle(cam2, (6, 6), (210, 40), (0, 0, 0), -1)
    cv2.putText(cam2, f"ep{EP}  {i}/{NF}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel
    frame = np.ascontiguousarray(canv[:Htot, :Wtot])
    if i == mid: cv2.imwrite(out_mp4.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for pkt in stv.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")): oc.mux(pkt)
    i += 1
    if i % 600 == 0: print(f"  {i}/{NF}", flush=True)
c.close()
for pkt in stv.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {out_mp4} {i}f", flush=True); print("EP2047_4PANEL_DONE", flush=True)
