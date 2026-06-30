"""§3.2 痛点② 视频:CRAVE value 升降可解释(ep2302,0:06–1:17 段)+ KAI0-AE 对照。
左=相机帧 + 实时分类标签;右=CRAVE value(上)/ KAI0-AE value(下),各按三档着色 + 分类条 + 游标。
直观看 CRAVE 的升/降与动作对齐(可解释)、KAI0-AE 抖动正负横跳(不可解释)。
CRAVE 数据 _cache.npz(cv/ccls);AE 数据 advantage_q5(absolute_value + relative_advantage,P/N 按 relative_value)。纯 CPU。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_value_interp_video_ep2302.py
输出: temp/crave_kai0base_videos/crave_value_interp_ep2302.mp4
"""
from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from crave.config import REPO
from crave.render import setup_mpl

plt = setup_mpl()

CACHE = REPO / "temp/crave_interp_ep2302_30hz_decoded/_cache.npz"
BASE = REPO / "kai0/data/Task_A/kai0_base"
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"
OUT = REPO / "temp/crave_kai0base_videos"; OUT.mkdir(parents=True, exist_ok=True)
EP = 2302; F0, F1 = 180, 2310; EPS = 0.02
RGB = {1: (0.17, 0.63, 0.17), 0: (0.6, 0.6, 0.6), -1: (0.84, 0.15, 0.16)}
BGR = {1: (44, 160, 44), 0: (150, 150, 150), -1: (214, 39, 40)}
CNAME = {1: "POSITIVE  spread -> value UP", 0: "NORMAL  steady", -1: "NEGATIVE  lift -> value DOWN"}
ANAME = {1: "POSITIVE", 0: "NORMAL", -1: "NEGATIVE"}


def three(a): return np.where(a > EPS, 1, np.where(a < -EPS, -1, 0))


def main():
    z = np.load(CACHE, allow_pickle=True)
    cv = z["cv"].astype(float)[F0:F1]; cl = z["ccls"].astype(int)[F0:F1]
    csQ = json.load(open(Q5 / "meta/info.json"))["chunks_size"]
    dQ = pd.read_parquet(Q5 / "data" / f"chunk-{EP//csQ:03d}" / f"episode_{EP:06d}.parquet", columns=["absolute_value", "relative_advantage"])
    ae = dQ["absolute_value"].to_numpy().astype(float); ra = np.clip(dQ["relative_advantage"].to_numpy().astype(float), -1, 1)
    ae, al = ae[F0:F1], three(ra[F0:F1])
    m = min(len(cv), len(ae)); cv, cl, ae, al = cv[:m], cl[:m], ae[:m], al[:m]
    n = len(cv); t = np.arange(n) / 30.0
    cstrip = np.array([RGB[int(c)] for c in cl])[None]; astrip = np.array([RGB[int(c)] for c in al])[None]

    PFIG = plt.figure(figsize=(7.2, 6.6), dpi=100)
    gs = PFIG.add_gridspec(4, 1, height_ratios=[1, 0.2, 1, 0.2], hspace=0.42)
    axcv = PFIG.add_subplot(gs[0]); axcv.plot(t, cv, color="#333", lw=0.7, alpha=.35)
    for c in (-1, 0, 1): mm = cl == c; axcv.scatter(t[mm], cv[mm], s=6, c=[RGB[c]])
    axcv.set_xlim(0, t[-1]); axcv.set_ylim(cv.min() - .03, cv.max() + .03); axcv.set_ylabel("CRAVE\nvalue", fontsize=9); axcv.grid(alpha=.2); axcv.tick_params(labelsize=7)
    axcv.set_title("CRAVE(零训练):铺开→上升(绿)/ 拎起→下降(红)/ 平稳(灰)—— 升降与动作对齐", fontsize=10)
    axcs = PFIG.add_subplot(gs[1]); axcs.imshow(cstrip, aspect="auto", extent=[0, t[-1], 0, 1]); axcs.set_yticks([]); axcs.set_xticks([]); axcs.set_ylabel("分类条", fontsize=7, rotation=0, ha="right", va="center")
    axav = PFIG.add_subplot(gs[2]); axav.plot(t, ae, color="#333", lw=0.7, alpha=.35)
    for c in (-1, 0, 1): mm = al == c; axav.scatter(t[mm], ae[mm], s=6, c=[RGB[c]])
    axav.set_xlim(0, t[-1]); axav.set_ylim(ae.min() - .03, ae.max() + .03); axav.set_ylabel("KAI0-AE\nvalue", fontsize=9); axav.grid(alpha=.2); axav.tick_params(labelsize=7)
    axav.set_title("KAI0-AE(监督,P/N 按 relative_value):抖动 → 正负逐拍横跳,看不出为什么升降", fontsize=10)
    axas = PFIG.add_subplot(gs[3]); axas.imshow(astrip, aspect="auto", extent=[0, t[-1], 0, 1]); axas.set_yticks([]); axas.set_xlabel("秒(相对段首)", fontsize=9); axas.set_ylabel("分类条", fontsize=7, rotation=0, ha="right", va="center"); axas.tick_params(labelsize=7)
    PFIG.suptitle("CRAVE value 升降可解释 · ep2302(0:06–1:17) · vs KAI0-AE", fontsize=11.5, y=0.995)
    PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]

    def pm(a):
        bb = a.get_position(); xl, xh = a.get_xlim(); yl, yh = a.get_ylim(); return bb.x0, bb.x1, bb.y0, bb.y1, xl, xh, yl, yh
    def xpx(mm, s): x0, x1, _, _, xl, xh, _, _ = mm; return int(round((x0 + (s - xl) / (xh - xl) * (x1 - x0)) * Wp))
    def yp(mm, v): x0, x1, y0, y1, xl, xh, yl, yh = mm; return int(round((1 - (y0 + (v - yl) / (yh - yl) * (y1 - y0))) * Hp))
    def ysp(mm): _, _, y0, y1, *_ = mm; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))
    MCV, MAV, MAS = pm(axcv), pm(axav), pm(axas); plt.close(PFIG); span = (ysp(MCV)[0], ysp(MAS)[1])

    cs = json.load(open(BASE / "meta/info.json"))["chunks_size"]
    vid = BASE / "videos" / f"chunk-{EP//cs:03d}" / "observation.images.top_head" / f"episode_{EP:06d}.mp4"
    c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
    csc = Hp / f0.shape[0]; cw = int(round(f0.shape[1] * csc)) // 2 * 2; Wt = (cw + Wp) // 2 * 2; Ht = Hp // 2 * 2
    omp4 = str(OUT / "crave_value_interp_ep2302.mp4")
    oc = av.open(omp4, mode="w"); st = oc.add_stream("libx264", rate=30); st.width, st.height, st.pix_fmt = Wt, Ht, "yuv420p"; st.options = {"preset": "veryfast", "crf": "23"}
    cobj = av.open(str(vid)); i = 0
    for gi, fr in enumerate(cobj.decode(video=0)):
        if gi < F0: continue
        if gi >= F0 + n: break
        panel = PANEL.copy(); s = i / 30.0
        px = xpx(MCV, s); cv2.line(panel, (px, span[0]), (px, span[1]), (40, 40, 40), 1)
        cv2.circle(panel, (xpx(MCV, s), yp(MCV, float(cv[i]))), 6, BGR[int(cl[i])], -1); cv2.circle(panel, (xpx(MCV, s), yp(MCV, float(cv[i]))), 6, (0, 0, 0), 1)
        cv2.circle(panel, (xpx(MAV, s), yp(MAV, float(ae[i]))), 6, BGR[int(al[i])], -1); cv2.circle(panel, (xpx(MAV, s), yp(MAV, float(ae[i]))), 6, (0, 0, 0), 1)
        cam = cv2.resize(np.ascontiguousarray(fr.to_ndarray(format="rgb24")), (cw, Hp))
        cv2.rectangle(cam, (6, 6), (490, 78), (0, 0, 0), -1)
        cv2.putText(cam, f"ep{EP} {gi/30:.1f}s", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cam, f"CRAVE: {CNAME[int(cl[i])]}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR[int(cl[i])][::-1], 2, cv2.LINE_AA)
        cv2.putText(cam, f"KAI0-AE: {ANAME[int(al[i])]}", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR[int(al[i])][::-1], 2, cv2.LINE_AA)
        canv = np.zeros((Hp, cw + Wp, 3), np.uint8); canv[:, :cw] = cam; canv[:, cw:] = panel; frame = np.ascontiguousarray(canv[:Ht, :Wt])
        for pkt in st.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")): oc.mux(pkt)
        i += 1
        if i % 600 == 0: print(f"  {i}/{n}", flush=True)
    cobj.close()
    for pkt in st.encode(): oc.mux(pkt)
    oc.close()
    print(f"SAVED {omp4} {i}f", flush=True)


if __name__ == "__main__":
    main()
