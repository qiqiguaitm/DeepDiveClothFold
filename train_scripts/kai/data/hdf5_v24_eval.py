#!/usr/bin/env python
"""V2.4 零训练 milestone-value 评估 + 同步对齐视频, 用于 HDF5 数据集泛化验证.
特征前端 = hdf5_extract_features.py 产物 (ep*.npz: raw/armmask/state).
V2.4 核心(KMeans96 + coverage修正 + 进度分桶 + 端点锚 + Viterbi-DP)与
smooth800_v24_full.py 逐字一致 —— 不为任何数据集特化(泛化前提)。

用法: python hdf5_v24_eval.py --feat <cache> --hdf5dir <dir> --out <outdir> [--nvideos 3] [--mine-n 0]
产物: <out>/metrics.json  +  value_eval_overview.png  +  sync_ep{N}.mp4 ×nvideos  + 预览帧
"""
import argparse, glob, json, os, re
from pathlib import Path
from crave_readout import smooth_monotone
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

_simhei = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_simhei):
    fm.fontManager.addfont(_simhei)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
STRIDE = 10


def loadep(fc, e):
    d = np.load(fc / f"ep{e}.npz")
    a, r, s = d["armmask"], d["raw"], d["state"]
    n = min(len(a), len(r), len(s))
    return a[:n], r[:n], s[:n], n


def mkp(s):
    return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def build_model(fc, eps, mine_eps, log=print):
    Sall = [loadep(fc, e)[2] for e in mine_eps]
    Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(a_, r_, st):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = ((mkp(st) - PMU) / PSD); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    A, R, S, T, E, SP, EP = [], [], [], [], [], [], []
    for e in mine_eps:
        aa, rr, st, n = loadep(fc, e); g = emb(aa, rr, st)
        A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
        SP.append(g[:2]); EP.append(g[-2:])
    A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
    G = emb(A, R, S)
    km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
    N = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
    Pstart = {}
    for e in sorted(set(E.tolist())):
        m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1)
        Pstart[e] = float(np.median(tpos[nn]))
    cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / N)
                      for c in range(96)])
    bk = np.linspace(0, 1, 11); sel = []
    for b in range(10):
        inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
        if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:2]
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
            m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
            if rs: fe.append(T[rs[0][0]])
        Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
    order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]; Pord = np.array([Pk[c] for c in order])
    log(f"V2.4 milestones: {len(order)}  前段(P<0.5): {sum(1 for c in order if Pk[c] < 0.5)}")
    startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
    endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_
    NB = 21; bins = np.linspace(0, 1, NB); cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]

    def dpHB(emit, lam=8.0):
        pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
        cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
        for j in range(1, NF):
            tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
        cost[NB - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
        for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
        return bins[path]

    def med(arr, w):
        h = w // 2; return np.array([np.median(arr[max(0, j - h):j + h + 1]) for j in range(len(arr))])

    def value(aa, rr, st, ret_lab=False):
        Fq = emb(aa, rr, st); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
        em = np.full((nq, NB), 1e3)
        for ci in range(len(order)):
            for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
        ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1)
        de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
        tn = np.arange(nq) / nq
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        v = med(dpHB(em), 9)
        if ret_lab:
            dsrt = np.sort(d, axis=1); marg = dsrt[:, 0] / np.clip(dsrt[:, 1], 1e-9, None)
            return v, d.argmin(1), marg
        return v

    return value, Pord


def render_video(hdf5dir, e, v3hz, lab3, marg3, Pord, out_mp4, preview_png):
    import h5py, av
    fp = os.path.join(hdf5dir, f"episode_{e}.hdf5")
    with h5py.File(fp, "r") as h:
        T = h["observations/qpos"].shape[0]
        frames = [h["observations/images/cam_high"][i] for i in range(T)]
        lang = h["language_instruction"][()]
        lang = lang.decode() if isinstance(lang, bytes) else str(lang)
    NF = T
    V = np.repeat(v3hz, STRIDE)[:NF]
    if len(V) < NF: V = np.concatenate([V, np.full(NF - len(V), V[-1])])
    V = smooth_monotone(V, fps=30.0)  # 连续读出
    lab30 = np.repeat(lab3, STRIDE)[:NF]; marg30 = np.repeat(marg3, STRIDE)[:NF]
    if len(lab30) < NF:
        lab30 = np.concatenate([lab30, np.full(NF - len(lab30), lab30[-1])])
        marg30 = np.concatenate([marg30, np.full(NF - len(marg30), marg30[-1])])
    FPS = 30.0; L = NF; x = np.arange(L) / FPS
    mscol = matplotlib.colormaps["tab20"]
    PFIG = plt.figure(figsize=(10, 7), dpi=100); gs = PFIG.add_gridspec(2, 1, hspace=0.30)
    ax_l = PFIG.add_subplot(gs[0])
    for k in range(len(Pord)):
        hit = np.where((lab30 == k) & (marg30 <= 0.8))[0]
        if len(hit): ax_l.scatter(hit / FPS, np.full(len(hit), Pord[k]), s=6, color=mscol(k % 20), alpha=.6)
    ax_l.set_ylim(-0.03, 1.03); ax_l.set_xlim(0, L / FPS); ax_l.set_ylabel("milestone P_k")
    ax_l.set_title("milestone 命中 (置信 margin<=0.8, 颜色=阶段)", fontsize=9); ax_l.grid(alpha=.2)
    ax_v = PFIG.add_subplot(gs[1], sharex=ax_l)
    ax_v.plot(x, V[:L], color="#2ca02c", lw=2, label="V2.4 零训练 milestone-value")
    ax_v.set_xlabel("seconds"); ax_v.set_ylabel("V"); ax_v.set_ylim(-.05, 1.08)
    ax_v.legend(fontsize=8, loc="lower right"); ax_v.grid(alpha=.3)
    ax_v.set_title(f"V2.4 value (零训练, 跨本体泛化): {lang}", fontsize=9)
    PFIG.suptitle(f"xvla_soft_fold episode_{e} — V2.4 零训练 milestone-value 同步对齐", fontsize=12)
    PFIG.canvas.draw()
    PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]
    PFIG.savefig(preview_png, dpi=110)

    def pmap(ax):
        bb = ax.get_position(); xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
        return bb.x0, bb.x1, bb.y0, bb.y1, xlo, xhi, ylo, yhi
    MV = pmap(ax_v); ML = pmap(ax_l)

    def xpx(sec):
        x0, x1, _, _, xlo, xhi, _, _ = MV; return int(round((x0 + (sec - xlo) / (xhi - xlo) * (x1 - x0)) * Wp))

    def yspan(m):
        _, _, y0, y1, _, _, _, _ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))
    LT, LB = yspan(ML); VT, VB = yspan(MV)

    def valpx(sec, val):
        x0, x1, y0, y1, xlo, xhi, ylo, yhi = MV
        return (int(round((x0 + (sec - xlo) / (xhi - xlo) * (x1 - x0)) * Wp)),
                int(round((1 - (y0 + (val - ylo) / (yhi - ylo) * (y1 - y0))) * Hp)))
    plt.close(PFIG)
    f0 = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    MAXSIDE = 460; s = min(1.0, MAXSIDE / max(f0.shape[:2]))
    cw2 = int(f0.shape[1] * s * (Hp / (f0.shape[0] * s))) // 2 * 2
    csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2
    Wtot = (cw2 + Wp) // 2 * 2; Htot = Hp // 2 * 2
    oc = av.open(out_mp4, mode="w"); stv = oc.add_stream("libx264", rate=30)
    stv.width, stv.height, stv.pix_fmt = Wtot, Htot, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
    for t in range(L):
        cam = cv2.imdecode(np.frombuffer(frames[t], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        panel = PANEL.copy(); px = xpx(t / FPS)
        cv2.line(panel, (px, LT), (px, LB), (110, 110, 110), 2)
        cv2.line(panel, (px, VT), (px, VB), (110, 110, 110), 2)
        vx, vy = valpx(t / FPS, float(V[min(t, L - 1)]))
        cv2.circle(panel, (vx, vy), 7, (44, 160, 46), -1); cv2.circle(panel, (vx, vy), 7, (0, 0, 0), 1)
        cam2 = cv2.resize(np.ascontiguousarray(cam), (cw2, Hp))
        canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(canv[:Htot, :Wtot]), format="rgb24")
        for pkt in stv.encode(vf): oc.mux(pkt)
    for pkt in stv.encode(): oc.mux(pkt)
    oc.close()
    return Wtot, Htot, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--hdf5dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nvideos", type=int, default=3)
    ap.add_argument("--mine-n", type=int, default=0, help="0=use all eps for mining")
    ap.add_argument("--tag", default="xvla")
    a = ap.parse_args()
    fc = Path(a.feat); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps = sorted(int(p.stem[2:]) for p in fc.glob("ep*.npz"))
    print(f"[v24-eval] {len(eps)} eps in {fc}", flush=True)
    mine_eps = eps if not a.mine_n else sorted(
        np.random.RandomState(0).permutation(eps)[:min(a.mine_n, len(eps))].tolist())
    value, Pord = build_model(fc, eps, mine_eps)

    # per-ep value + corr(value, normalized time)
    corr, mono, store = {}, {}, {}
    for e in eps:
        aa, rr, st, n = loadep(fc, e); v, lab, marg = value(aa, rr, st, ret_lab=True)
        store[e] = (v, lab, marg, n)
        t = np.arange(n) / max(1, n - 1)
        corr[e] = float(np.corrcoef(v, t)[0, 1]) if n > 2 and v.std() > 1e-6 else 0.0
        mono[e] = float((np.diff(v) >= -1e-6).mean()) if n > 1 else 1.0
    cc = np.array([corr[e] for e in eps]); mo = np.array([mono[e] for e in eps])
    metrics = {"tag": a.tag, "n_eps": len(eps), "n_mine": len(mine_eps), "n_milestones": int(len(Pord)),
               "corr_mean": float(cc.mean()), "corr_median": float(np.median(cc)),
               "corr_p25": float(np.percentile(cc, 25)), "frac_corr_ge_0.7": float((cc >= 0.7).mean()),
               "frac_corr_ge_0.5": float((cc >= 0.5).mean()), "mono_mean": float(mo.mean()),
               "bad_lt0.5": sorted(int(e) for e in eps if corr[e] < 0.5)}
    json.dump({**metrics, "corr": corr, "mono": mono}, open(out / "metrics.json", "w"), indent=1)
    print(f"[v24-eval] corr mean={metrics['corr_mean']:.3f} median={metrics['corr_median']:.3f} "
          f"frac>=0.7={metrics['frac_corr_ge_0.7']:.2%} mono={metrics['mono_mean']:.2%}", flush=True)

    # overview figure: corr 分布 + 几条样例曲线
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].hist(cc, bins=20, color="#2ca02c", alpha=.8); ax[0].axvline(0.7, color="r", ls="--", label="0.7 阈值")
    ax[0].set_title(f"{a.tag}: corr(value, 归一化时间) 分布 (n={len(eps)})\n"
                    f"mean={cc.mean():.3f} median={np.median(cc):.3f} 占比>=0.7: {(cc>=0.7).mean():.0%}",
                    fontsize=10)
    ax[0].set_xlabel("Pearson r"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.2)
    for e in eps[::max(1, len(eps) // 12)]:
        v = store[e][0]; ax[1].plot(np.arange(len(v)) / max(1, len(v) - 1), v, alpha=.5, lw=1)
    ax[1].plot([0, 1], [0, 1], "k--", lw=1.5, label="理想 0→1")
    ax[1].set_title(f"{a.tag}: 样例 value 曲线 (零训练 V2.4)", fontsize=10)
    ax[1].set_xlabel("归一化时间"); ax[1].set_ylabel("value"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.2)
    fig.suptitle(f"V2.4 零训练 milestone-value 泛化评估 — {a.tag}", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "value_eval_overview.png", dpi=120); plt.close(fig)

    # 3 random videos (seed-fixed)
    rng = np.random.RandomState(42); vids = sorted(rng.permutation(eps)[:a.nvideos].tolist())
    print(f"[v24-eval] 渲染视频 eps={vids}", flush=True)
    vid_info = []
    for e in vids:
        v, lab, marg, n = store[e]
        mp4 = str(out / f"sync_ep{e}.mp4"); png = str(out / f"sync_ep{e}_preview.png")
        W, H, L = render_video(a.hdf5dir, e, v, lab, marg, Pord, mp4, png)
        print(f"  ep{e}: {W}x{H} {L}f corr={corr[e]:.3f} mono={mono[e]:.2%} → {mp4}", flush=True)
        vid_info.append({"ep": int(e), "corr": corr[e], "mono": mono[e], "frames": int(L)})
    json.dump(vid_info, open(out / "videos.json", "w"), indent=1)
    print("HDF5_V24_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
