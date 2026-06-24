"""可视化 CRAVE 三档 vs AWBC-AE 三档(都 pos/normal/neg)—— ep808。
两者都用同一 advantage 定义(value[n+50]-value[n])+ 同一 ε 三分, 公平对比:
CRAVE 的 normal 是结构性平台(主体), AE 的 normal 薄且 pos/neg 噪声散布。
静态图(整段)+ 同步视频(相机 + 两条 value 按三档着色 + 当前帧标签读数)。
"""
import json, os
import numpy as np, pandas as pd, matplotlib, cv2, av
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path

_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
MV = REPO / "temp/mv_value_full"; AW = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all_awbc"
DS = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"
csAW = json.load(open(AW / "meta/info.json"))["chunks_size"]; csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]
EP = 808; W = 50; EPS = 0.02
COL3 = {1: "#2ca02c", 0: "#999999", -1: "#d62728"}; NAME3 = {1: "POSITIVE", 0: "NORMAL", -1: "NEGATIVE"}
BGR3 = {1: (44, 160, 44), 0: (150, 150, 150), -1: (214, 39, 40)}


def adv(v, w=W):
    n = len(v); a = np.zeros(n)
    for i in range(n): a[i] = np.clip(v[min(i + w, n - 1)] - v[i], -1, 1)
    return a


def three(a, eps=EPS): return np.where(a > eps, 1, np.where(a < -eps, -1, 0))


cv = np.load(MV / f"ep{EP}.npy").astype(float)
d = pd.read_parquet(AW / "data" / f"chunk-{EP//csAW:03d}" / f"episode_{EP:06d}.parquet", columns=["absolute_value", "task_index"])
ae_v = d["absolute_value"].to_numpy().astype(float)
n = min(len(cv), len(ae_v)); cv, ae_v = cv[:n], ae_v[:n]
ccls = three(adv(cv)); acls = three(adv(ae_v))   # 两者都三档(同 advantage 定义 + 同 ε)
fc = {c: (ccls == c).mean() for c in (1, 0, -1)}; fa = {c: (acls == c).mean() for c in (1, 0, -1)}
print(f"ep{EP} {n}帧  CRAVE pos{fc[1]:.0%}/normal{fc[0]:.0%}/neg{fc[-1]:.0%}  |  AE-3档 pos{fa[1]:.0%}/normal{fa[0]:.0%}/neg{fa[-1]:.0%}", flush=True)
x = np.arange(n) / 30.0

fig, ax = plt.subplots(2, 1, figsize=(14, 6.6), sharex=True)
for k, (v, cls, fr, ttl) in enumerate([(cv, ccls, fc, "CRAVE 三档 (normal=结构性平台, 主体)"),
                                       (ae_v, acls, fa, "AWBC-AE 三档 (同 ε 三分; normal 薄, pos/neg 噪声散布)")]):
    ax[k].plot(x, v, color="#333", lw=0.7, alpha=.45)
    for c in (-1, 0, 1):
        m = cls == c; ax[k].scatter(x[m], v[m], s=7, c=COL3[c], label=f"{NAME3[c]} {fr[c]:.0%}")
    ax[k].set_ylim(min(-.05, v.min() - .05), max(1.05, v.max() + .05)); ax[k].set_ylabel("value"); ax[k].grid(alpha=.2)
    ax[k].legend(fontsize=8, loc="upper left", ncol=3); ax[k].set_title(ttl, fontsize=10)
ax[1].set_xlabel("秒")
fig.suptitle(f"ep{EP}(dagger, 末帧叠好=成功): CRAVE 三档 vs AWBC-AE 三档 打标", fontsize=12)
fig.tight_layout(); out = REPO / "docs/visualization/cross_episode_recurrence_value/crave_3level_vs_ae3_labels_ep808.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True); plt.close(fig)

# ---- 同步视频(两条都三档着色)----
vid = DS / "videos" / f"chunk-{EP//csDS:03d}" / "observation.images.top_head" / f"episode_{EP:06d}.mp4"
PFIG = plt.figure(figsize=(9, 5.2), dpi=100); gs = PFIG.add_gridspec(2, 1, hspace=0.5)
A0 = PFIG.add_subplot(gs[0]); A0.plot(x, cv, color="#333", lw=0.6, alpha=.4)
for c in (-1, 0, 1): m = ccls == c; A0.scatter(x[m], cv[m], s=4, c=COL3[c])
A0.set_ylim(-.05, 1.05); A0.set_xlim(0, n / 30); A0.set_ylabel("CRAVE", fontsize=8); A0.tick_params(labelsize=7)
A0.set_title(f"CRAVE 三档: pos{fc[1]:.0%}/normal{fc[0]:.0%}/neg{fc[-1]:.0%} (绿/灰/红)", fontsize=9); A0.grid(alpha=.2)
A1 = PFIG.add_subplot(gs[1], sharex=A0); A1.plot(x, ae_v, color="#333", lw=0.6, alpha=.4)
for c in (-1, 0, 1): m = acls == c; A1.scatter(x[m], ae_v[m], s=4, c=COL3[c])
A1.set_ylim(ae_v.min() - .05, ae_v.max() + .05); A1.set_ylabel("AE", fontsize=8); A1.set_xlabel("秒", fontsize=8); A1.tick_params(labelsize=7)
A1.set_title(f"AWBC-AE 三档: pos{fa[1]:.0%}/normal{fa[0]:.0%}/neg{fa[-1]:.0%} (噪声→normal 薄)", fontsize=9); A1.grid(alpha=.2)
PFIG.suptitle(f"ep{EP} — CRAVE 三档 vs AWBC-AE 三档(都 pos/normal/neg)", fontsize=11)
PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]


def pm(a):
    bb = a.get_position(); xl, xh = a.get_xlim(); yl, yh = a.get_ylim(); return bb.x0, bb.x1, bb.y0, bb.y1, xl, xh, yl, yh


def pxy(m, sec, val):
    x0, x1, y0, y1, xl, xh, yl, yh = m
    return (int(round((x0 + (sec - xl) / (xh - xl) * (x1 - x0)) * Wp)), int(round((1 - (y0 + (val - yl) / (yh - yl) * (y1 - y0))) * Hp)))


def ysp(m): _, _, y0, y1, *_ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))
M0, M1 = pm(A0), pm(A1); S0, S1 = ysp(M0), ysp(M1); plt.close(PFIG)
c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2; Wt = (cw2 + Wp) // 2 * 2; Ht = Hp // 2 * 2
omp4 = str(REPO / f"temp/crave_3level_vs_ae3_ep{EP}.mp4")
oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
cobj = av.open(str(vid)); i = 0; mid = n // 2
for fr in cobj.decode(video=0):
    if i >= n: break
    panel = PANEL.copy(); sec = i / 30.0
    for m, (yt, yb) in ((M0, S0), (M1, S1)):
        px = pxy(m, sec, 0)[0]; cv2.line(panel, (px, yt), (px, yb), (110, 110, 110), 1)
    gx, gy = pxy(M0, sec, float(cv[i])); cv2.circle(panel, (gx, gy), 7, BGR3[ccls[i]], -1); cv2.circle(panel, (gx, gy), 7, (0, 0, 0), 1)
    rx, ry = pxy(M1, sec, float(ae_v[i])); cv2.circle(panel, (rx, ry), 7, BGR3[acls[i]], -1); cv2.circle(panel, (rx, ry), 7, (0, 0, 0), 1)
    cam2 = cv2.resize(np.ascontiguousarray(fr.to_ndarray(format="rgb24")), (cw2, Hp))
    cv2.rectangle(cam2, (6, 6), (290, 78), (0, 0, 0), -1)
    cv2.putText(cam2, f"ep{EP} {i}/{n}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cam2, f"CRAVE: {NAME3[ccls[i]]}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BGR3[ccls[i]][::-1], 2, cv2.LINE_AA)
    cv2.putText(cam2, f"AE-3: {NAME3[acls[i]]}", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BGR3[acls[i]][::-1], 2, cv2.LINE_AA)
    canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel; frame = np.ascontiguousarray(canv[:Ht, :Wt])
    if i == mid: cv2.imwrite(omp4.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for pkt in stv.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")): oc.mux(pkt)
    i += 1
    if i % 1500 == 0: print(f"  vid {i}/{n}", flush=True)
cobj.close()
for pkt in stv.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {omp4} {i}f", flush=True); print("THREELEVEL3_DONE", flush=True)
