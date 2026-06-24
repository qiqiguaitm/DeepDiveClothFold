#!/usr/bin/env python
"""对比: [30Hz挖矿+30Hz输出] vs [3Hz挖矿+3Hz输出], 同一批 kai0_base episode(同集公平),
最终都在 ep2047 上算 value 并对照。窗按频率标定(lam ∝ fps, proprio Δ帧距 ∝ fps)。
3Hz 特征 = kai0bd 缓存; 30Hz 特征 = crave_30hz_mine(挖矿集) + crave_30hz(ep2047)。

Thin entrypoint over `crave`: REPO from crave.config, 3-path cache reader via crave.data.loadep,
med/viterbi from crave.utils, Agg+SimHei via crave.render. The mine pipeline + the 2-panel
synced ffmpeg/av video stay inline. The 30Hz/advantage_q5 caches are not in the registry, so
their paths stay literal.
"""
import glob, os
import numpy as np, pandas as pd, cv2, av
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

from crave.config import REPO
from crave.data import loadep as loadnpz
from crave.render import setup_mpl
from crave.utils import med, viterbi

plt = setup_mpl()

FC3 = REPO / "temp/crave_kai0bd/feat_cache"           # 3Hz 全集
FC30M = REPO / "temp/crave_30hz_mine/feat_cache"      # 30Hz 挖矿集
FC30E = REPO / "temp/crave_30hz/feat_cache"           # 30Hz ep2047 测试
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"; cs = 1000; EP = 2047
NB = 21; bins = np.linspace(0, 1, NB)
rs = np.random.RandomState(0)
base3 = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(str(FC3 / "ep*.npz")) if int(os.path.basename(p)[2:-4]) < 100000)
MINE = sorted(set([2047, 2238, 2302]) | set(rs.permutation(base3)[:80].tolist()))


def mkp_dt(s, dt):
    d = np.zeros_like(s); d[dt:] = s[dt:] - s[:-dt]; return np.concatenate([s, d], 1)


def run(cache, mine_eps, dt, lam, medw, test_cache, test_ep):
    eps = [e for e in mine_eps if (cache / f"ep{e}.npz").exists()]
    Sall = [loadnpz(cache, e)[2] for e in eps]; Pm = mkp_dt(np.concatenate(Sall), dt); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st):
        rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True); an = a_ / np.linalg.norm(a_, axis=1, keepdims=True)
        Pn = (mkp_dt(st, dt) - PMU) / PSD; Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    A, R, S, T, E, SP, EP_ = [], [], [], [], [], [], []
    for e in eps:
        aa, rr, st, n = loadnpz(cache, e); g = emb(aa, rr, st)
        A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); SP.append(g[:2]); EP_.append(g[-2:])
    A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E); G = emb(A, R, S)
    km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
    Nn = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
    Ps = {}
    for e in sorted(set(E.tolist())):
        m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Ps[e] = float(np.median(tpos[nn]))
    cov = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Ps if Ps[e] > tpos[c] + 0.1)) / Nn) for c in range(96)])
    bk = np.linspace(0, 1, 11); sel = []
    for b in range(10):
        inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
        if inb: sel += sorted(inb, key=lambda c: -cov[c])[:2]
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
            m = np.where(E == e)[0]; rsg = gr(m[lab[m] == c].tolist())
            if rsg: fe.append(T[rsg[0][0]])
        Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
    order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]; Pord = np.array([Pk[c] for c in order])
    startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
    endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP_)).cluster_centers_
    cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]

    aa, rr, st, n = loadnpz(test_cache, test_ep); Fq = emb(aa, rr, st); nq = len(Fq)
    d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
    for ci in range(len(order)):
        for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
    ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
    tn = np.arange(nq) / nq
    em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
    em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    path = viterbi(em, bins, lam=lam, end_bonus=2)[1]
    v = med(bins[path], medw)
    return v, len(order), len(eps), sorted(Pord.round(2).tolist())


print(f"挖矿集 {len(MINE)} base eps(同集)", flush=True)
v3, m3, n3, pk3 = run(FC3, MINE, dt=1, lam=8, medw=9, test_cache=FC3, test_ep=EP)        # 3Hz挖+3Hz出
v30, m30, n30, pk30 = run(FC30M, MINE, dt=10, lam=80, medw=45, test_cache=FC30E, test_ep=EP)  # 30Hz挖+30Hz出
print(f"3Hz挖+3Hz出:  milestones {m3} (挖矿{n3}ep), ep2047 value {len(v3)}点 末{v3[-1]:.2f}", flush=True)
print(f"30Hz挖+30Hz出: milestones {m30} (挖矿{n30}ep), ep2047 value {len(v30)}点 末{v30[-1]:.2f}", flush=True)
# 对齐到秒 + 互相关
t3 = np.arange(len(v3)) / 3.0; t30 = np.arange(len(v30)) / 30.0
v3_up = np.interp(t30, t3, v3)  # 3Hz 插值到 30Hz 时间轴比对
cr = pearsonr(v3_up, v30)[0]
dQ = pd.read_parquet(Q5 / f"data/chunk-{EP//cs:03d}/episode_{EP:06d}.parquet"); ae = dQ["absolute_value"].to_numpy().astype(float)
La = min(len(ae), len(v30)); cr3_ae = pearsonr(np.interp(np.arange(La) / 30.0, t3, v3), ae[:La])[0]; cr30_ae = pearsonr(v30[:La], ae[:La])[0]
print(f"corr(3Hz挖 vs 30Hz挖, 对齐时间)={cr:.3f} | corr-with-AE 3Hz={cr3_ae:.2f} 30Hz={cr30_ae:.2f}", flush=True)
print(f"milestone P_k: 3Hz={pk3}", flush=True)
print(f"milestone P_k: 30Hz={pk30}", flush=True)

fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(np.arange(len(ae)) / 30.0, ae, color="#999", lw=1.1, ls="--", label="pi0-AE absolute_value(参考)")
ax.plot(t3, v3, color="#2ca02c", lw=2.6, drawstyle="steps-post", label=f"3Hz挖矿+3Hz输出: {m3}个milestone, {len(v3)}点, 末{v3[-1]:.2f}")
ax.plot(t30, v30, color="#1f77ff", lw=1.5, alpha=.9, label=f"30Hz挖矿+30Hz输出: {m30}个milestone, {len(v30)}点, 末{v30[-1]:.2f}")
ax.set_title(f"ep2047: [30Hz挖矿+30Hz输出] vs [3Hz挖矿+3Hz输出] (同{len(MINE)}个kai0_base; corr={cr:.2f})", fontsize=11)
ax.set_xlabel("秒"); ax.set_ylabel("value"); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.25); ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); out = REPO / "crave/docs/visualization/crave_mine_freq_compare.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True); plt.close(fig)

# ---- 2 面板同步视频(3Hz挖+3Hz出 | 30Hz挖+30Hz出)----
BASE = REPO / "kai0/data/Task_A/kai0_base"
vid = BASE / f"videos/chunk-{EP//cs:03d}/observation.images.top_head/episode_{EP:06d}.mp4"
NFv = len(pd.read_parquet(BASE / f"data/chunk-{EP//cs:03d}/episode_{EP:06d}.parquet", columns=["frame_index"]))
NF = min(NFv, len(v30)); FPS = 30.0; tt = np.arange(NF) / FPS
v3_30 = np.repeat(v3, 10)
if len(v3_30) < NF: v3_30 = np.concatenate([v3_30, np.full(NF - len(v3_30), v3_30[-1])])
sigs = [(v3_30[:NF], f"3Hz 挖矿 + 3Hz 输出  ({m3} milestone, 263点)", "#2ca02c"),
        (v30[:NF], f"30Hz 挖矿 + 30Hz 输出  ({m30} milestone, 2629点)", "#1f77ff")]
PFIG = plt.figure(figsize=(9, 5), dpi=100); gs = PFIG.add_gridspec(2, 1, hspace=0.5); AX = []
for i, (sig, title, col) in enumerate(sigs):
    a2 = PFIG.add_subplot(gs[i]); a2.plot(tt, sig, color=col, lw=1.5); a2.plot(tt, ae[:NF], color="#bbb", lw=0.8, ls="--")
    a2.set_xlim(0, NF / FPS); a2.set_ylim(-.05, 1.05); a2.set_title(title, fontsize=10, color=col); a2.grid(alpha=.25); a2.tick_params(labelsize=7)
    if i == 1: a2.set_xlabel("秒", fontsize=8)
    AX.append((a2, col, sig))
PFIG.suptitle(f"kai0_base ep{EP} — 挖矿频率对比(灰=pi0-AE 参考)", fontsize=11)
PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]


def pm(a2):
    bb = a2.get_position(); xl, xh = a2.get_xlim(); yl, yh = a2.get_ylim(); return bb.x0, bb.x1, bb.y0, bb.y1, xl, xh, yl, yh


def pxy(m, sec, val):
    x0, x1, y0, y1, xl, xh, yl, yh = m
    return (int(round((x0 + (sec - xl) / (xh - xl) * (x1 - x0)) * Wp)), int(round((1 - (y0 + (val - yl) / (yh - yl) * (y1 - y0))) * Hp)))


def ysp(m): _, _, y0, y1, *_ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))
MAPS = [pm(a) for a, _, _ in AX]; SPN = [ysp(m) for m in MAPS]; plt.close(PFIG)
COLS = [tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for _, c, _ in AX]
c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2; Wt = (cw2 + Wp) // 2 * 2; Ht = Hp // 2 * 2
omp4 = str(REPO / f"temp/crave_mine_freq_ep{EP}.mp4")
oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
cobj = av.open(str(vid)); i = 0; mid = NF // 2
for fr in cobj.decode(video=0):
    if i >= NF: break
    panel = PANEL.copy(); sec = i / FPS
    for (m, (yt, yb), (_, _, sig), col) in zip(MAPS, SPN, AX, COLS):
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
print(f"SAVED {omp4} {i}f", flush=True); print("MINE_FREQ_DONE", flush=True)
