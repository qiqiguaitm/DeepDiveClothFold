"""对齐视频: 相机 + CRAVE vs KAI0-AE 把数据分类为 pos/normal/neg 的能力对比。
每条 value 下加一条"分类时间条"(绿=pos 灰=normal 红=neg), 直观看 CRAVE 干净结构化 vs AE 噪声散布。
ep808(dagger, 末帧叠好=成功): AE 在成功 episode 上仍标 35% neg = 噪声。
"""
import json
import numpy as np, pandas as pd, cv2, av

from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import smooth_monotone

plt = setup_mpl()
# TODO(crave-lib): mv_value_full + *_awbc datasets are not in the dataset registry.
MV = REPO / "temp/mv_value_full"; AW = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all_awbc"
DS = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"
csAW = json.load(open(AW / "meta/info.json"))["chunks_size"]; csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]
EP = 808; W = 50; EPS = 0.02
RGB = {1: (0.17, 0.63, 0.17), 0: (0.6, 0.6, 0.6), -1: (0.84, 0.15, 0.16)}      # 0-1 for matplotlib
BGR = {1: (44, 160, 44), 0: (150, 150, 150), -1: (214, 39, 40)}; NAME = {1: "POSITIVE", 0: "NORMAL", -1: "NEGATIVE"}


def adv(v, w=W):
    nn = len(v); a = np.zeros(nn)
    for i in range(nn): a[i] = np.clip(v[min(i + w, nn - 1)] - v[i], -1, 1)
    return a


def three(a): return np.where(a > EPS, 1, np.where(a < -EPS, -1, 0))


# build_ds_A_from_mv.py 同款连续化(阶梯→连续 ramp): crave.utils.smooth_monotone, fps=30 → w=41
cv = smooth_monotone(np.load(MV / f"ep{EP}.npy").astype(float), fps=30.0)
d = pd.read_parquet(AW / "data" / f"chunk-{EP//csAW:03d}" / f"episode_{EP:06d}.parquet", columns=["absolute_value"])
ae_v = d["absolute_value"].to_numpy().astype(float); n = min(len(cv), len(ae_v)); cv, ae_v = cv[:n], ae_v[:n]
ccls = three(adv(cv)); acls = three(adv(ae_v))
fc = {c: (ccls == c).mean() for c in (1, 0, -1)}; fa = {c: (acls == c).mean() for c in (1, 0, -1)}
print(f"ep{EP} {n}帧 CRAVE pos{fc[1]:.0%}/normal{fc[0]:.0%}/neg{fc[-1]:.0%} | AE pos{fa[1]:.0%}/normal{fa[0]:.0%}/neg{fa[-1]:.0%}", flush=True)
x = np.arange(n) / 30.0
cstrip = np.array([RGB[c] for c in ccls])[None]; astrip = np.array([RGB[c] for c in acls])[None]

# ---- 背景面板: value(着色) + 分类时间条, CRAVE 上 / AE 下 ----
PFIG = plt.figure(figsize=(9.5, 6.2), dpi=100)
gs = PFIG.add_gridspec(4, 1, height_ratios=[1, 0.22, 1, 0.22], hspace=0.45)
axcv = PFIG.add_subplot(gs[0]); axcv.plot(x, cv, color="#333", lw=0.6, alpha=.35)
for c in (-1, 0, 1): m = ccls == c; axcv.scatter(x[m], cv[m], s=4, c=[RGB[c]])
axcv.set_ylim(-.05, 1.05); axcv.set_xlim(0, n / 30); axcv.set_ylabel("CRAVE\nvalue", fontsize=8); axcv.tick_params(labelsize=7)
axcv.set_title(f"CRAVE 三档分类: pos{fc[1]:.0%} / normal{fc[0]:.0%} / neg{fc[-1]:.0%}  (绿/灰/红)", fontsize=10); axcv.grid(alpha=.2)
axcs = PFIG.add_subplot(gs[1]); axcs.imshow(cstrip, aspect="auto", extent=[0, n / 30, 0, 1]); axcs.set_yticks([]); axcs.set_xlim(0, n / 30); axcs.tick_params(labelsize=7); axcs.set_ylabel("分类条", fontsize=7)
axav = PFIG.add_subplot(gs[2]); axav.plot(x, ae_v, color="#333", lw=0.6, alpha=.35)
for c in (-1, 0, 1): m = acls == c; axav.scatter(x[m], ae_v[m], s=4, c=[RGB[c]])
axav.set_ylim(ae_v.min() - .05, ae_v.max() + .05); axav.set_xlim(0, n / 30); axav.set_ylabel("AE\nvalue", fontsize=8); axav.tick_params(labelsize=7)
axav.set_title(f"KAI0-AE 三档分类: pos{fa[1]:.0%} / normal{fa[0]:.0%} / neg{fa[-1]:.0%}  (成功episode却{fa[-1]:.0%}红=噪声)", fontsize=10); axav.grid(alpha=.2)
axas = PFIG.add_subplot(gs[3]); axas.imshow(astrip, aspect="auto", extent=[0, n / 30, 0, 1]); axas.set_yticks([]); axas.set_xlim(0, n / 30); axas.set_xlabel("秒", fontsize=8); axas.tick_params(labelsize=7); axas.set_ylabel("分类条", fontsize=7)
PFIG.suptitle(f"ep{EP} — CRAVE vs KAI0-AE 数据分类能力 (pos/normal/neg) 对齐对比", fontsize=11)
PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]


def pm(a):
    bb = a.get_position(); xl, xh = a.get_xlim(); yl, yh = a.get_ylim(); return bb.x0, bb.x1, bb.y0, bb.y1, xl, xh, yl, yh


def xpx(m, sec):
    x0, x1, _, _, xl, xh, _, _ = m; return int(round((x0 + (sec - xl) / (xh - xl) * (x1 - x0)) * Wp))


def yp(m, val):
    x0, x1, y0, y1, xl, xh, yl, yh = m; return int(round((1 - (y0 + (val - yl) / (yh - yl) * (y1 - y0))) * Hp))


def ysp(m): _, _, y0, y1, *_ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))
MV_, MCS, MA_, MAS = pm(axcv), pm(axcs), pm(axav), pm(axas)
plt.close(PFIG)
allspan = (ysp(MV_)[0], ysp(MAS)[1])
vid = DS / "videos" / f"chunk-{EP//csDS:03d}" / "observation.images.top_head" / f"episode_{EP:06d}.mp4"
c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2; Wt = (cw2 + Wp) // 2 * 2; Ht = Hp // 2 * 2
omp4 = str(REPO / f"temp/crave_classify_ability_ep{EP}.mp4")
oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
cobj = av.open(str(vid)); i = 0; mid = n // 2
for fr in cobj.decode(video=0):
    if i >= n: break
    panel = PANEL.copy(); sec = i / 30.0
    px = xpx(MV_, sec); cv2.line(panel, (px, allspan[0]), (px, allspan[1]), (40, 40, 40), 1)
    cv2.circle(panel, (xpx(MV_, sec), yp(MV_, float(cv[i]))), 7, BGR[ccls[i]], -1); cv2.circle(panel, (xpx(MV_, sec), yp(MV_, float(cv[i]))), 7, (0, 0, 0), 1)
    cv2.circle(panel, (xpx(MA_, sec), yp(MA_, float(ae_v[i]))), 7, BGR[acls[i]], -1); cv2.circle(panel, (xpx(MA_, sec), yp(MA_, float(ae_v[i]))), 7, (0, 0, 0), 1)
    cam2 = cv2.resize(np.ascontiguousarray(fr.to_ndarray(format="rgb24")), (cw2, Hp))
    cv2.rectangle(cam2, (6, 6), (300, 80), (0, 0, 0), -1)
    cv2.putText(cam2, f"ep{EP} {i}/{n}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cam2, f"CRAVE: {NAME[ccls[i]]}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR[ccls[i]][::-1], 2, cv2.LINE_AA)
    cv2.putText(cam2, f"AE:    {NAME[acls[i]]}", (12, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR[acls[i]][::-1], 2, cv2.LINE_AA)
    canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel; frame = np.ascontiguousarray(canv[:Ht, :Wt])
    if i == mid: cv2.imwrite(omp4.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for pkt in stv.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")): oc.mux(pkt)
    i += 1
    if i % 1500 == 0: print(f"  vid {i}/{n}", flush=True)
cobj.close()
for pkt in stv.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {omp4} {i}f", flush=True); print("CLASSIFY_VIDEO_DONE", flush=True)
