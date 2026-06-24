#!/usr/bin/env python
"""CRAVE vs pi0-AE 同步对齐视频: 相机帧(左) + value/advantage 面板(右, 双游标)。
绿=CRAVE 零训练, 红=pi0-AE 监督; 每帧 cursor 竖线 + 两个圆点 + 相机上实时读数,
让视频帧与对应帧的 value 数据一一对齐。背景面板 matplotlib 画一次, 逐帧 cv2 叠加(快)。

用法: python crave_vs_ae_sync_video.py --cam <mp4> --arrays temp/_crave_ae_X.npz \
        --title "..." --out temp/crave_vs_ae_X_sync.mp4 [--rounds 3]

Rewrite onto the `crave` library: only the matplotlib SimHei/Agg boilerplate is replaced
by crave.render.setup_mpl. The bespoke PyAV writer is kept verbatim because it relies on
specific encode options (libx264 preset=veryfast crf=23) and a mid-frame preview dump that
crave.render.VideoWriter does not expose — preserving byte-for-byte output.
"""
import argparse

import numpy as np, cv2
import av

from crave.render import setup_mpl

plt = setup_mpl()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True)
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--title", default="CRAVE vs pi0-AE 同步对齐")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=0, help=">0 时画轮次边界(均分)")
    a = ap.parse_args()
    z = np.load(a.arrays); crave = z["crave"]; ae = z["ae"]; cadv = z["crave_adv"]; aadv = z["ae_adv"]
    FPS = float(z["fps"]); NF = len(crave); t = np.arange(NF) / FPS
    print(f"arrays NF={NF} fps={FPS}", flush=True)

    # ---- 背景面板(画一次) ----
    PFIG = plt.figure(figsize=(10, 7), dpi=100); gs = PFIG.add_gridspec(2, 1, hspace=0.30)
    axv = PFIG.add_subplot(gs[0])
    axv.plot(t, crave, color="#2ca02c", lw=2.0, label=f"CRAVE 零训练 (end {crave[-1]:.2f})")
    axv.plot(t, ae, color="#d62728", lw=1.5, alpha=.85, label=f"pi0-AE 监督 (end {ae[-1]:.2f}, max {ae.max():.2f})")
    axv.axhline(1, color="#2ca02c", ls=":", lw=1, alpha=.5); axv.axhline(0, color="k", lw=.5)
    axv.set_ylim(-.08, 1.13); axv.set_xlim(0, NF / FPS); axv.set_ylabel("value"); axv.grid(alpha=.25)
    axv.legend(fontsize=8.5, loc="center left"); axv.set_title("value: CRAVE 干净 0→1 vs 监督 AE 欠读/噪声", fontsize=9.5)
    axa = PFIG.add_subplot(gs[1], sharex=axv)
    axa.plot(t, cadv, color="#2ca02c", lw=1.3, label=f"CRAVE adv (neg {np.mean(cadv<0)*100:.0f}%)")
    axa.plot(t, aadv, color="#d62728", lw=1.1, alpha=.8, label=f"AE adv (neg {np.mean(aadv<0)*100:.0f}%)")
    axa.axhline(0, color="k", lw=.7); axa.fill_between(t, 0, cadv, where=cadv < 0, color="#2ca02c", alpha=.15)
    axa.set_xlabel("seconds"); axa.set_ylabel("advantage"); axa.grid(alpha=.25); axa.legend(fontsize=8.5, loc="lower left")
    axa.set_title("advantage(退步信号): CRAVE 稀疏对齐真退步 vs AE 弥散", fontsize=9.5)
    if a.rounds > 1:
        for k in range(1, a.rounds):
            b = NF * k / a.rounds / FPS
            for ax in (axv, axa): ax.axvline(b, color="orange", ls=":", lw=1.3)
        for k in range(a.rounds):
            axv.text((NF * k / a.rounds + 60) / FPS, 1.05, f"round{k+1}", fontsize=8.5, color="gray")
    PFIG.suptitle(a.title, fontsize=12)
    PFIG.canvas.draw()
    PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]

    def pmap(ax):
        bb = ax.get_position(); xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
        return bb.x0, bb.x1, bb.y0, bb.y1, xlo, xhi, ylo, yhi
    MV = pmap(axv); MA = pmap(axa)

    def px_xy(m, sec, val):
        x0, x1, y0, y1, xlo, xhi, ylo, yhi = m
        return (int(round((x0 + (sec - xlo) / (xhi - xlo) * (x1 - x0)) * Wp)),
                int(round((1 - (y0 + (val - ylo) / (yhi - ylo) * (y1 - y0))) * Hp)))

    def yspan(m):
        _, _, y0, y1, _, _, _, _ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))
    VT, VB = yspan(MV); AT, AB = yspan(MA)
    plt.close(PFIG)

    # ---- 相机尺寸 ----
    c0 = av.open(a.cam); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
    csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2
    Wtot = (cw2 + Wp) // 2 * 2; Htot = Hp // 2 * 2
    oc = av.open(a.out, mode="w"); stv = oc.add_stream("libx264", rate=int(round(FPS)))
    stv.width, stv.height, stv.pix_fmt = Wtot, Htot, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
    print(f"canvas {Wtot}x{Htot} frames~{NF}", flush=True)

    def compose(cam, i):
        panel = PANEL.copy(); sec = i / FPS
        cvx, _ = px_xy(MV, sec, 0)
        cv2.line(panel, (cvx, VT), (cvx, VB), (120, 120, 120), 2)
        cv2.line(panel, (cvx, AT), (cvx, AB), (120, 120, 120), 2)
        for m, gv, rv in ((MV, crave[i], ae[i]), (MA, cadv[i], aadv[i])):
            gx, gy = px_xy(m, sec, gv); rx, ry = px_xy(m, sec, rv)
            cv2.circle(panel, (rx, ry), 7, (214, 39, 40), -1); cv2.circle(panel, (rx, ry), 7, (0, 0, 0), 1)
            cv2.circle(panel, (gx, gy), 7, (44, 160, 44), -1); cv2.circle(panel, (gx, gy), 7, (0, 0, 0), 1)
        cam2 = cv2.resize(np.ascontiguousarray(cam), (cw2, Hp))
        # 相机上实时读数
        cv2.rectangle(cam2, (6, 6), (250, 76), (0, 0, 0), -1)
        cv2.putText(cam2, f"frame {i}/{NF}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cam2, f"CRAVE {crave[i]:.2f}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (44, 200, 44), 2, cv2.LINE_AA)
        cv2.putText(cam2, f"pi0-AE {ae[i]:.2f}", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 70, 70), 2, cv2.LINE_AA)
        canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel
        return np.ascontiguousarray(canv[:Htot, :Wtot])

    c = av.open(a.cam); i = 0; mid = NF // 2
    for fr in c.decode(video=0):
        if i >= NF: break
        cam = fr.to_ndarray(format="rgb24")
        frame = compose(cam, i)
        if i == mid:
            cv2.imwrite(a.out.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for pkt in stv.encode(vf): oc.mux(pkt)
        i += 1
        if i % 1500 == 0: print(f"  {i}/{NF}", flush=True)
    c.close()
    for pkt in stv.encode(): oc.mux(pkt)
    oc.close()
    print(f"SAVED {a.out} {Wtot}x{Htot} {i}f", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
