"""为几个【较长 kai0_base episode】渲染 KAI0-AE value+分类条对齐视频(给用户主观裁剪痛点示意短片)。
每条视频:左=相机帧 + 实时分类标签;右=KAI0-AE value(pi0-AE absolute_value)+ "分类时间条"
(绿=POSITIVE 灰=NORMAL 红=NEGATIVE,按 Δvalue 的 advantage 符号)。

用途:痛点② 传统 AE 的 pos/neg 不可解释(value 抖→标签横跳);痛点③ 叠衣末端假跌(看末段)。
只渲染 AE(传统 value 模型),不含 CRAVE。数据全现成、纯 CPU:AE 值 = advantage_q5 的 absolute_value
(pi0-AE 预算输出);相机 = kai0_base 视频。无需 GPU 推理 / 无需挖矿。

Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_kai0base_value_videos.py [--eps 2302,23,2047,2238,2939]
输出: temp/crave_kai0base_videos/kai0base_ae_value_ep{E}.mp4 (+ _preview.png)
"""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import advantage

plt = setup_mpl()

BASE = REPO / "kai0/data/Task_A/kai0_base"
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"   # pi0-AE Stage-2 输出(absolute_value)
OUT = REPO / "temp/crave_kai0base_videos"; OUT.mkdir(parents=True, exist_ok=True)
csB = json.load(open(BASE / "meta/info.json"))["chunks_size"]
csQ = json.load(open(Q5 / "meta/info.json"))["chunks_size"]
W = 50; EPS = 0.02
RGB = {1: (0.17, 0.63, 0.17), 0: (0.6, 0.6, 0.6), -1: (0.84, 0.15, 0.16)}
BGR = {1: (44, 160, 44), 0: (150, 150, 150), -1: (214, 39, 40)}; NAME = {1: "POSITIVE", 0: "NORMAL", -1: "NEGATIVE"}


def adv(v, w=W): return np.clip(advantage(v, w), -1, 1)
def three(a): return np.where(a > EPS, 1, np.where(a < -EPS, -1, 0))


def render_ep(EP):
    dQ = pd.read_parquet(Q5 / "data" / f"chunk-{EP//csQ:03d}" / f"episode_{EP:06d}.parquet", columns=["absolute_value", "relative_advantage"])
    ae = dQ["absolute_value"].to_numpy().astype(float)
    ra = np.clip(dQ["relative_advantage"].to_numpy().astype(float), -1, 1)   # AE 直出 relative value(P/N 依据)
    n = len(ae); acls = three(ra)                                            # 分类按 relative_value(非 Δabsolute_value)
    fa = {c: (acls == c).mean() for c in (1, 0, -1)}
    print(f"ep{EP} {n}f≈{n/30:.0f}s | KAI0-AE(按relative_value) pos{fa[1]:.0%}/norm{fa[0]:.0%}/neg{fa[-1]:.0%}", flush=True)
    x = np.arange(n) / 30.0
    astrip = np.array([RGB[c] for c in acls])[None]

    # ---- 背景面板:AE value(着色散点 + 细线) + 分类时间条 ----
    PFIG = plt.figure(figsize=(9.5, 4.2), dpi=100)
    gs = PFIG.add_gridspec(2, 1, height_ratios=[1, 0.26], hspace=0.32)
    axav = PFIG.add_subplot(gs[0]); axav.plot(x, ae, color="#333", lw=0.6, alpha=.35)
    for c in (-1, 0, 1): m = acls == c; axav.scatter(x[m], ae[m], s=5, c=[RGB[c]])
    axav.set_ylim(ae.min() - .05, ae.max() + .05); axav.set_xlim(0, n / 30); axav.set_ylabel("KAI0-AE value", fontsize=9); axav.tick_params(labelsize=7)
    axav.set_title(f"KAI0-AE value(分类按 relative_value): pos{fa[1]:.0%} / normal{fa[0]:.0%} / neg{fa[-1]:.0%}  (成功 episode 却 {fa[-1]:.0%} 红 = 抖动/正负横跳)", fontsize=10); axav.grid(alpha=.2)
    axas = PFIG.add_subplot(gs[1]); axas.imshow(astrip, aspect="auto", extent=[0, n / 30, 0, 1]); axas.set_yticks([]); axas.set_xlim(0, n / 30); axas.set_xlabel("秒", fontsize=8); axas.tick_params(labelsize=7); axas.set_ylabel("分类条", fontsize=7)
    PFIG.suptitle(f"kai0_base ep{EP} — KAI0-AE value + 分类条(POS 绿 / NORMAL 灰 / NEG 红)", fontsize=11)
    PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]

    def pm(a):
        bb = a.get_position(); xl, xh = a.get_xlim(); yl, yh = a.get_ylim(); return bb.x0, bb.x1, bb.y0, bb.y1, xl, xh, yl, yh

    def xpx(m, sec):
        x0, x1, _, _, xl, xh, _, _ = m; return int(round((x0 + (sec - xl) / (xh - xl) * (x1 - x0)) * Wp))

    def yp(m, val):
        x0, x1, y0, y1, xl, xh, yl, yh = m; return int(round((1 - (y0 + (val - yl) / (yh - yl) * (y1 - y0))) * Hp))

    def ysp(m): _, _, y0, y1, *_ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))
    MA_, MAS = pm(axav), pm(axas); plt.close(PFIG)
    allspan = (ysp(MA_)[0], ysp(MAS)[1])
    vid = BASE / "videos" / f"chunk-{EP//csB:03d}" / "observation.images.top_head" / f"episode_{EP:06d}.mp4"
    c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
    csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2; Wt = (cw2 + Wp) // 2 * 2; Ht = Hp // 2 * 2
    omp4 = str(OUT / f"kai0base_ae_value_ep{EP}.mp4")
    oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
    cobj = av.open(str(vid)); i = 0; mid = n // 2
    for fr in cobj.decode(video=0):
        if i >= n: break
        panel = PANEL.copy(); sec = i / 30.0
        px = xpx(MA_, sec); cv2.line(panel, (px, allspan[0]), (px, allspan[1]), (40, 40, 40), 1)
        cv2.circle(panel, (xpx(MA_, sec), yp(MA_, float(ae[i]))), 7, BGR[acls[i]], -1); cv2.circle(panel, (xpx(MA_, sec), yp(MA_, float(ae[i]))), 7, (0, 0, 0), 1)
        cam2 = cv2.resize(np.ascontiguousarray(fr.to_ndarray(format="rgb24")), (cw2, Hp))
        cv2.rectangle(cam2, (6, 6), (300, 58), (0, 0, 0), -1)
        cv2.putText(cam2, f"ep{EP} {i}/{n}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cam2, f"KAI0-AE: {NAME[acls[i]]}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR[acls[i]][::-1], 2, cv2.LINE_AA)
        canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel; frame = np.ascontiguousarray(canv[:Ht, :Wt])
        if i == mid: cv2.imwrite(omp4.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        for pkt in stv.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")): oc.mux(pkt)
        i += 1
        if i % 1500 == 0: print(f"  ep{EP} vid {i}/{n}", flush=True)
    cobj.close()
    for pkt in stv.encode(): oc.mux(pkt)
    oc.close()
    print(f"SAVED {omp4} {i}f", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="2302,23,2047,2238,2939")
    a = ap.parse_args()
    eps_render = [int(x) for x in a.eps.split(",")]
    print(f"渲染 KAI0-AE value 视频(无 CRAVE): {eps_render}", flush=True)
    for EP in eps_render:
        try:
            render_ep(EP)
        except Exception as ex:
            print(f"ep{EP} FAILED: {ex}", flush=True)
    print("KAI0BASE_AE_VALUE_VIDEOS_DONE", flush=True)


if __name__ == "__main__":
    main()
