#!/usr/bin/env python
"""三配置对比(挖矿频率 × 输出频率 解耦), 同一批 kai0_base, 最终 ep2047 算 value + 同步视频:
  ① 3Hz挖 + 3Hz出  (基线)
  ② 30Hz挖 + 3Hz出 (隔离: 只变挖矿频率, 输出固定3Hz → 看挖矿频率是否影响)
  ③ 3Hz挖 + 30Hz出 (隔离: 只变输出频率, 挖矿固定3Hz → 看输出分辨率影响)
milestone 模型在挖矿频率的 emb 空间找到, 应用到输出频率的特征(proprio Δ 用 dt 对齐同 Δt=1/3s, 故兼容)。
窗按输出频率标定(3Hz: lam8/med9; 30Hz: lam80/med45)。
"""
import os, glob
import numpy as np, pandas as pd, matplotlib, cv2, av
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
FC3 = REPO / "temp/crave_kai0bd/feat_cache"; FC30M = REPO / "temp/crave_30hz_mine/feat_cache"; FC30E = REPO / "temp/crave_30hz/feat_cache"
Q5 = REPO / "kai0/data/Task_A/self_built/advantage_q5"; BASE = REPO / "kai0/data/Task_A/kai0_base"; cs = 1000; EP = 2047
NB = 21; bins = np.linspace(0, 1, NB)
rs = np.random.RandomState(0)
base3 = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(str(FC3 / "ep*.npz")) if int(os.path.basename(p)[2:-4]) < 100000)
MINE = sorted(set([2047, 2238, 2302]) | set(rs.permutation(base3)[:80].tolist()))


def loadnpz(fc, e):
    d = np.load(fc / f"ep{e}.npz"); a, r, s = d["armmask"], d["raw"], d["state"]
    n = min(len(a), len(r), len(s)); return a[:n], r[:n], s[:n], n


def mkp_dt(s, dt):
    d = np.zeros_like(s); d[dt:] = s[dt:] - s[:-dt]; return np.concatenate([s, d], 1)


def med(a, w): h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


def mine(cache, mine_eps, mine_dt):
    eps = [e for e in mine_eps if (cache / f"ep{e}.npz").exists()]
    Sall = [loadnpz(cache, e)[2] for e in eps]; Pm = mkp_dt(np.concatenate(Sall), mine_dt); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st, dt):
        rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True); an = a_ / np.linalg.norm(a_, axis=1, keepdims=True)
        Pn = (mkp_dt(st, dt) - PMU) / PSD; Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    A, R, S, T, E, SP, EP_ = [], [], [], [], [], [], []
    for e in eps:
        aa, rr, st, n = loadnpz(cache, e); g = emb(aa, rr, st, mine_dt)
        A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e)); SP.append(g[:2]); EP_.append(g[-2:])
    A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E); G = emb(A, R, S, mine_dt)
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
    order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]
    startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
    endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP_)).cluster_centers_
    cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]

    def value(cache_t, ep_t, out_dt, lam, medw):
        aa, rr, st, n = loadnpz(cache_t, ep_t); Fq = emb(aa, rr, st, out_dt); nq = len(Fq)
        d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
        for ci in range(len(order)):
            for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
        ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
        tn = np.arange(nq) / nq
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        pen = lam * np.abs(bins[:, None] - bins[None]); cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((nq, NB), int)
        for j in range(1, nq):
            tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k
        cost[NB - 1] -= 2; path = np.zeros(nq, int); path[-1] = cost.argmin()
        for j in range(nq - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
        return med(bins[path], medw)
    return value, len(order)


print(f"挖矿集 {len([e for e in MINE if (FC3/f'ep{e}.npz').exists()])} eps", flush=True)
m3, nm3 = mine(FC3, MINE, 1)        # 3Hz 挖矿模型
m30, nm30 = mine(FC30M, MINE, 10)   # 30Hz 挖矿模型
v_3m3o = m3(FC3, EP, 1, 8, 9)        # ① 3Hz挖+3Hz出
v_30m3o = m30(FC3, EP, 1, 8, 9)      # ② 30Hz挖+3Hz出
v_3m30o = m3(FC30E, EP, 10, 80, 45)   # ③ 3Hz挖+30Hz出
v_30m30o = m30(FC30E, EP, 10, 80, 45)  # ④ 30Hz挖+30Hz出
dQ = pd.read_parquet(Q5 / f"data/chunk-{EP//cs:03d}/episode_{EP:06d}.parquet"); ae = dQ["absolute_value"].to_numpy().astype(float)


def cae(v, fps):
    t = np.arange(len(v)) / fps; La = min(len(ae), int(t[-1] * 30) + 1); return pearsonr(np.interp(np.arange(La) / 30.0, t, v), ae[:La])[0]


c12 = pearsonr(v_3m3o, v_30m3o)[0]      # ① vs ②(同3Hz输出, 只挖矿不同)
c34 = pearsonr(v_3m30o, v_30m30o)[0]    # ③ vs ④(同30Hz输出, 只挖矿不同)
print(f"① 3Hz挖+3Hz出:   {nm3}ms 末{v_3m3o[-1]:.2f} corrAE{cae(v_3m3o,3):.2f}", flush=True)
print(f"② 30Hz挖+3Hz出:  {nm30}ms 末{v_30m3o[-1]:.2f} corrAE{cae(v_30m3o,3):.2f}", flush=True)
print(f"③ 3Hz挖+30Hz出:  {nm3}ms 末{v_3m30o[-1]:.2f} corrAE{cae(v_3m30o,30):.2f}", flush=True)
print(f"④ 30Hz挖+30Hz出: {nm30}ms 末{v_30m30o[-1]:.2f} corrAE{cae(v_30m30o,30):.2f}", flush=True)
print(f"只变挖矿: corr(①,②)={c12:.3f}(3Hz输出) / corr(③,④)={c34:.3f}(30Hz输出)", flush=True)

# ---- 3 面板同步视频 ----
vid = BASE / f"videos/chunk-{EP//cs:03d}/observation.images.top_head/episode_{EP:06d}.mp4"
NFv = len(pd.read_parquet(BASE / f"data/chunk-{EP//cs:03d}/episode_{EP:06d}.parquet", columns=["frame_index"]))
NF = min(NFv, len(v_3m30o)); FPS = 30.0; tt = np.arange(NF) / FPS


def up(v):
    w = np.repeat(v, 10); return np.concatenate([w, np.full(NF - len(w), w[-1])])[:NF] if len(w) < NF else w[:NF]


sigs = [(up(v_3m3o), f"① 3Hz挖 + 3Hz出  (基线, {nm3}ms 263点)", "#2ca02c"),
        (up(v_30m3o), f"② 30Hz挖 + 3Hz出  (只变挖矿: corr与①={c12:.2f})", "#9467bd"),
        (v_3m30o[:NF], f"③ 3Hz挖 + 30Hz出  (只变输出: 2629点 更细)", "#1f77ff"),
        (v_30m30o[:NF], f"④ 30Hz挖 + 30Hz出  (两者都变: corr与③={c34:.2f})", "#d62728")]
PFIG = plt.figure(figsize=(9, 8.5), dpi=100); gs = PFIG.add_gridspec(4, 1, hspace=0.6); AX = []
for i, (sig, title, col) in enumerate(sigs):
    a2 = PFIG.add_subplot(gs[i]); a2.plot(tt, sig, color=col, lw=1.5); a2.plot(tt, ae[:NF], color="#bbb", lw=0.8, ls="--")
    a2.set_xlim(0, NF / FPS); a2.set_ylim(-.05, 1.05); a2.set_title(title, fontsize=9.5, color=col); a2.grid(alpha=.25); a2.tick_params(labelsize=7)
    if i == 3: a2.set_xlabel("秒", fontsize=8)
    AX.append((a2, col, sig))
PFIG.suptitle(f"kai0_base ep{EP} — 挖矿频率 × 输出频率 2×2 解耦对比(灰=pi0-AE)", fontsize=11)
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
omp4 = str(REPO / f"temp/crave_freq_4way_ep{EP}.mp4")
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
    if i % 700 == 0: print(f"  vid {i}/{NF}", flush=True)
cobj.close()
for pkt in stv.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {omp4} {i}f", flush=True); print("FREQ3WAY_DONE", flush=True)
