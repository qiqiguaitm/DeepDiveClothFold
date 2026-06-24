#!/usr/bin/env python
"""ep2047 最终版: 30Hz 下 离线(全局Viterbi) vs 在线(固定滞后) 对比, 窗按频率重标定。
窗标定(3Hz→30Hz): 转移惩罚 lam 8→80(×10, 平衡10×emission累积, 保同等平滑度);
中值窗 9→45(1.5s); 在线固定滞后 L 12→90(3s)。并对照"未标定 lam=8/L=12"证明必须随频率调窗。
产物: 对比图 + 2面板(离线|在线)同步视频。复用 crave_30hz_test 挖矿+emission。
"""
import os
import numpy as np, pandas as pd, cv2, av
from scipy.stats import pearsonr

from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import med

plt = setup_mpl()

# TODO(crave-lib): crave_30hz_test.py mining core (loadnpz/FC30/emission/bins/NB/med_causal/C/Pord)
#   is bespoke (3Hz/30Hz time-binned 96-cluster) and not exposed via crave.clustering/crave.value;
#   reuse verbatim via exec (same as legacy). It defines its own REPO (== crave.config.REPO).
_SRC = REPO / "train_scripts/kai/data/crave_30hz_test.py"
src = open(_SRC).read().split("fig, axs")[0]
exec(src)   # loadnpz, FC30, emission, bins, NB, med_causal, C, Pord, REPO ...

EP = 2047; BASE = REPO / "kai0/data/Task_A/kai0_base"; Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"; cs = 1000


def dp_offline(em, lam, medw):  # 全局 Viterbi(离线/非因果): 前向+末端奖励+反向backtrace
    pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(em); cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return med(bins[path], medw)


def dp_online(em, lam, L, medw):  # 固定滞后 Viterbi(在线/因果)
    pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(em); cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((NF, NB), int); es = np.zeros(NF, int); es[0] = int(cost.argmin())
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k; es[j] = int(cost.argmin())
    out = np.zeros(NF, int)
    for j in range(NF):
        t = min(j + L, NF - 1); s = es[t]
        for jj in range(t, j, -1): s = bp[jj][s]
        out[j] = s
    return med_causal(bins[out], medw)


aa, rr, st, n = loadnpz(FC30, EP); em, d = emission(aa, rr, st, 10)
# 30Hz 标定窗(最终版)
v_off = dp_offline(em, 80, 45)
v_on = dp_online(em, 80, 90, 45)
# 未标定(3Hz原参直接用在30Hz, 错误对照)
v_on_bad = dp_online(em, 8, 12, 9)
dQ = pd.read_parquet(Q5 / f"data/chunk-{EP//cs:03d}/episode_{EP:06d}.parquet"); ae = dQ["absolute_value"].to_numpy().astype(float)
L0 = min(len(v_off), len(ae)); t = np.arange(L0) / 30.0


def jit(v): return float(np.abs(np.diff(v)).mean())


def lag(a, b, ml=90):
    best, bl = -2, 0
    for Lg in range(0, ml):
        c = np.corrcoef(a[Lg:], b[:len(b) - Lg])[0, 1] if len(a) - Lg > 5 else -2
        if c > best: best, bl = c, Lg
    return bl


cr_on = pearsonr(v_on[:L0], v_off[:L0])[0]; cr_bad = pearsonr(v_on_bad[:L0], v_off[:L0])[0]
print(f"30Hz 标定窗(lam80/L90/med45): corr(在线,离线)={cr_on:.3f} 末值 离线{v_off[-1]:.2f}/在线{v_on[-1]:.2f} "
      f"jitter 离线{jit(v_off):.4f}/在线{jit(v_on):.4f} 滞后{lag(v_off,v_on)}帧({lag(v_off,v_on)/30:.1f}s)", flush=True)
print(f"未标定(lam8/L12/med9): corr={cr_bad:.3f} jitter={jit(v_on_bad):.4f}  → 证明随频率调窗的必要", flush=True)

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(t, ae[:L0], color="#999", lw=1.3, ls="--", label="pi0-AE absolute_value(参考)")
ax.plot(t, v_off[:L0], color="#2ca02c", lw=2.4, label=f"离线 全局DP(lam80/med45): 末{v_off[-1]:.2f} 抖{jit(v_off):.4f}")
ax.plot(t, v_on[:L0], color="#1f77ff", lw=1.6, alpha=.9, label=f"在线 固定滞后(lam80/L90/med45): corr{cr_on:.2f} 末{v_on[-1]:.2f} 抖{jit(v_on):.4f} 滞后{lag(v_off,v_on)/30:.1f}s")
ax.plot(t, v_on_bad[:L0], color="#ff7f0e", lw=1.0, ls=":", alpha=.7, label=f"在线·未标定窗(lam8/L12/med9): 抖{jit(v_on_bad):.4f}(更吵→须随频率调窗)")
ax.set_title(f"ep2047 最终版 @30Hz: 离线 vs 在线(窗按频率标定 lam×10 / 窗按秒)", fontsize=11)
ax.set_xlabel("秒"); ax.set_ylabel("value"); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.25); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); out = REPO / "crave/docs/visualization/crave_ep2047_final_offline_vs_online.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True); plt.close(fig)

# ---- 2 面板(离线|在线)同步视频 @30Hz ----
vid = BASE / f"videos/chunk-{EP//cs:03d}/observation.images.top_head/episode_{EP:06d}.mp4"
NF = min(L0, len(pd.read_parquet(BASE / f"data/chunk-{EP//cs:03d}/episode_{EP:06d}.parquet", columns=["frame_index"])))
sigs = [(v_off[:NF], "离线 全局Viterbi-DP(非因果, 最优)", "#2ca02c"), (v_on[:NF], "在线 固定滞后(因果, 延迟3s)", "#1f77ff")]
FPS = 30.0; tt = np.arange(NF) / FPS
PFIG = plt.figure(figsize=(9, 5), dpi=100); gs = PFIG.add_gridspec(2, 1, hspace=0.5); AX = []
for i, (sig, title, col) in enumerate(sigs):
    a2 = PFIG.add_subplot(gs[i]); a2.plot(tt, sig, color=col, lw=1.5); a2.plot(tt, ae[:NF], color="#bbb", lw=0.8, ls="--")
    a2.set_xlim(0, NF / FPS); a2.set_ylim(-.05, 1.05); a2.set_title(title, fontsize=10, color=col); a2.grid(alpha=.25); a2.tick_params(labelsize=7)
    if i == 1: a2.set_xlabel("秒", fontsize=8)
    AX.append((a2, col, sig))
PFIG.suptitle(f"kai0_base ep{EP} @30Hz — CRAVE 离线 vs 在线 value(窗已按频率标定)", fontsize=11)
PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]


def pm(a2):
    bb = a2.get_position(); xl, xh = a2.get_xlim(); yl, yh = a2.get_ylim(); return bb.x0, bb.x1, bb.y0, bb.y1, xl, xh, yl, yh


def pxy(m, sec, val):
    x0, x1, y0, y1, xl, xh, yl, yh = m
    return (int(round((x0 + (sec - xl) / (xh - xl) * (x1 - x0)) * Wp)), int(round((1 - (y0 + (val - yl) / (yh - yl) * (y1 - y0))) * Hp)))


def ysp(m): _, _, y0, y1, *_ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))
MAPS = [pm(a) for a, _, _ in AX]; SP = [ysp(m) for m in MAPS]; plt.close(PFIG)
COLS = [tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for _, c, _ in AX]
c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2; Wt = (cw2 + Wp) // 2 * 2; Ht = Hp // 2 * 2
omp4 = str(REPO / f"temp/crave_ep{EP}_final_offline_vs_online.mp4")
oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
cobj = av.open(str(vid)); i = 0; mid = NF // 2
for fr in cobj.decode(video=0):
    if i >= NF: break
    panel = PANEL.copy(); sec = i / FPS
    for (m, (yt, yb), (_, _, sig), col) in zip(MAPS, SP, AX, COLS):
        px = pxy(m, sec, 0)[0]; cv2.line(panel, (px, yt), (px, yb), (120, 120, 120), 1)
        vx, vy = pxy(m, sec, float(sig[min(i, NF - 1)])); cv2.circle(panel, (vx, vy), 6, col, -1); cv2.circle(panel, (vx, vy), 6, (0, 0, 0), 1)
    cam2 = cv2.resize(np.ascontiguousarray(fr.to_ndarray(format="rgb24")), (cw2, Hp))
    cv2.rectangle(cam2, (6, 6), (210, 40), (0, 0, 0), -1); cv2.putText(cam2, f"ep{EP} {i}/{NF}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel; frame = np.ascontiguousarray(canv[:Ht, :Wt])
    if i == mid: cv2.imwrite(omp4.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for pkt in stv.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")): oc.mux(pkt)
    i += 1
    if i % 600 == 0: print(f"  vid {i}/{NF}", flush=True)
cobj.close()
for pkt in stv.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {omp4} {i}f", flush=True); print("EP2047_FINAL_DONE", flush=True)
