#!/usr/bin/env python
"""V2.4 零训练 milestone-value 评估 + 同步视频, LeRobot v3.0 (concatenated mp4) 版本.
复用 hdf5_v24_eval.py 的 V2.4 核心 (build_model/loadep) —— 配方逐字一致, 不为数据集特化。
render 从单 mp4 按 episode 帧区间解码。

用法: python lerobot_v24_eval.py --feat <cache> --repo-dir <dl> --cam observation.images.cam_high \
        --out <outdir> [--stride 16] [--nvideos 3] [--tag coffee]
"""
import argparse, glob, json, os
from pathlib import Path
from crave_readout import smooth_monotone
import numpy as np, cv2, matplotlib, pandas as pd
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdf5_v24_eval import build_model, loadep   # 复用 V2.4 核心

_simhei = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_simhei):
    fm.fontManager.addfont(_simhei)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def render_video(mp4, f0, t0, fps, e, v_s, lab_s, marg_s, Pord, stride, lang, out_mp4, preview_png):
    import av
    frames = []
    c = av.open(mp4); gi = 0
    for fr in c.decode(video=0):
        if gi >= t0:
            break
        if gi >= f0:
            frames.append(fr.to_ndarray(format="rgb24"))
        gi += 1
    c.close()
    NF = len(frames)
    V = np.repeat(v_s, stride)[:NF]
    if len(V) < NF: V = np.concatenate([V, np.full(NF - len(V), V[-1])])
    V = smooth_monotone(V, fps=fps)  # 连续读出
    lab30 = np.repeat(lab_s, stride)[:NF]; marg30 = np.repeat(marg_s, stride)[:NF]
    if len(lab30) < NF: lab30 = np.concatenate([lab30, np.full(NF - len(lab30), lab30[-1])])
    if len(marg30) < NF: marg30 = np.concatenate([marg30, np.full(NF - len(marg30), marg30[-1])])
    L = NF; x = np.arange(L) / fps
    mscol = matplotlib.colormaps["tab20"]
    PFIG = plt.figure(figsize=(10, 7), dpi=100); gs = PFIG.add_gridspec(2, 1, hspace=0.30)
    ax_l = PFIG.add_subplot(gs[0])
    for k in range(len(Pord)):
        hit = np.where((lab30 == k) & (marg30 <= 0.8))[0]
        if len(hit): ax_l.scatter(hit / fps, np.full(len(hit), Pord[k]), s=6, color=mscol(k % 20), alpha=.6)
    ax_l.set_ylim(-0.03, 1.03); ax_l.set_xlim(0, L / fps); ax_l.set_ylabel("milestone P_k")
    ax_l.set_title("milestone 命中 (置信 margin<=0.8, 颜色=阶段)", fontsize=9); ax_l.grid(alpha=.2)
    ax_v = PFIG.add_subplot(gs[1], sharex=ax_l)
    ax_v.plot(x, V[:L], color="#2ca02c", lw=2, label="V2.4 零训练 milestone-value")
    ax_v.set_xlabel("seconds"); ax_v.set_ylabel("V"); ax_v.set_ylim(-.05, 1.08)
    ax_v.legend(fontsize=8, loc="lower right"); ax_v.grid(alpha=.3)
    ax_v.set_title(f"V2.4 value (零训练, 跨数据集泛化): {lang[:70]}", fontsize=8)
    PFIG.suptitle(f"aloha_static_coffee episode_{e} — V2.4 零训练 milestone-value 同步对齐", fontsize=12)
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
    csc = Hp / frames[0].shape[0]; cw2 = int(round(frames[0].shape[1] * csc)) // 2 * 2
    Wtot = (cw2 + Wp) // 2 * 2; Htot = Hp // 2 * 2
    oc = av.open(out_mp4, mode="w"); stv = oc.add_stream("libx264", rate=int(round(fps)))
    stv.width, stv.height, stv.pix_fmt = Wtot, Htot, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
    for t in range(L):
        panel = PANEL.copy(); px = xpx(t / fps)
        cv2.line(panel, (px, LT), (px, LB), (110, 110, 110), 2)
        cv2.line(panel, (px, VT), (px, VB), (110, 110, 110), 2)
        vx, vy = valpx(t / fps, float(V[min(t, L - 1)]))
        cv2.circle(panel, (vx, vy), 7, (44, 160, 46), -1); cv2.circle(panel, (vx, vy), 7, (0, 0, 0), 1)
        cam2 = cv2.resize(np.ascontiguousarray(frames[t]), (cw2, Hp))
        canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(canv[:Htot, :Wtot]), format="rgb24")
        for pkt in stv.encode(vf): oc.mux(pkt)
    for pkt in stv.encode(): oc.mux(pkt)
    oc.close()
    return Wtot, Htot, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--cam", default="observation.images.cam_high")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--nvideos", type=int, default=3)
    ap.add_argument("--tag", default="coffee")
    a = ap.parse_args()
    fc = Path(a.feat); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rd = Path(a.repo_dir)
    eps = sorted(int(p.stem[2:]) for p in fc.glob("ep*.npz"))
    print(f"[lr-v24] {len(eps)} eps in {fc}", flush=True)
    info = json.load(open(rd / "meta/info.json")); fps = float(info["fps"])
    epm = pd.read_parquet(glob.glob(str(rd / "meta/episodes/**/*.parquet"), recursive=True)[0])
    rng = {int(r.episode_index): (int(r.dataset_from_index), int(r.dataset_to_index)) for r in epm.itertuples()}
    langs = {}
    for r in epm.itertuples():
        tk = getattr(r, "tasks"); langs[int(r.episode_index)] = (tk[0] if hasattr(tk, "__len__") and not isinstance(tk, str) else str(tk))
    mp4 = glob.glob(str(rd / "videos" / a.cam / "**/*.mp4"), recursive=True)[0]

    value, Pord = build_model(fc, eps, eps)
    corr, mono, store = {}, {}, {}
    for e in eps:
        aa, rr, st, n = loadep(fc, e); v, lab, marg = value(aa, rr, st, ret_lab=True)
        store[e] = (v, lab, marg, n); t = np.arange(n) / max(1, n - 1)
        corr[e] = float(np.corrcoef(v, t)[0, 1]) if n > 2 and v.std() > 1e-6 else 0.0
        mono[e] = float((np.diff(v) >= -1e-6).mean()) if n > 1 else 1.0
    cc = np.array([corr[e] for e in eps]); mo = np.array([mono[e] for e in eps])
    metrics = {"tag": a.tag, "n_eps": len(eps), "n_milestones": int(len(Pord)),
               "corr_mean": float(cc.mean()), "corr_median": float(np.median(cc)),
               "corr_p25": float(np.percentile(cc, 25)), "frac_corr_ge_0.7": float((cc >= 0.7).mean()),
               "frac_corr_ge_0.5": float((cc >= 0.5).mean()), "mono_mean": float(mo.mean()),
               "bad_lt0.5": sorted(int(e) for e in eps if corr[e] < 0.5)}
    json.dump({**metrics, "corr": corr, "mono": mono}, open(out / "metrics.json", "w"), indent=1)
    print(f"[lr-v24] corr mean={metrics['corr_mean']:.3f} median={metrics['corr_median']:.3f} "
          f"frac>=0.7={metrics['frac_corr_ge_0.7']:.2%} mono={metrics['mono_mean']:.2%} "
          f"milestones={len(Pord)}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].hist(cc, bins=20, color="#8c564b", alpha=.85); ax[0].axvline(0.7, color="r", ls="--", label="0.7 阈值")
    ax[0].set_title(f"{a.tag}: corr(value, 归一化时间) 分布 (n={len(eps)})\n"
                    f"mean={cc.mean():.3f} median={np.median(cc):.3f} 占比>=0.7: {(cc>=0.7).mean():.0%}", fontsize=10)
    ax[0].set_xlabel("Pearson r"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.2)
    for e in eps[::max(1, len(eps) // 12)]:
        v = store[e][0]; ax[1].plot(np.arange(len(v)) / max(1, len(v) - 1), v, alpha=.5, lw=1)
    ax[1].plot([0, 1], [0, 1], "k--", lw=1.5, label="理想 0→1")
    ax[1].set_title(f"{a.tag}: 样例 value 曲线 (零训练 V2.4)", fontsize=10)
    ax[1].set_xlabel("归一化时间"); ax[1].set_ylabel("value"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.2)
    fig.suptitle(f"V2.4 零训练 milestone-value 跨数据集泛化 — {a.tag} (真实 ALOHA, 互联网)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "value_eval_overview.png", dpi=120); plt.close(fig)

    rngsel = np.random.RandomState(42); vids = sorted(rngsel.permutation(eps)[:a.nvideos].tolist())
    print(f"[lr-v24] 渲染视频 eps={vids}", flush=True)
    vid_info = []
    for e in vids:
        v, lab, marg, n = store[e]; f0, t0 = rng[e]
        mp4o = str(out / f"sync_ep{e}.mp4"); png = str(out / f"sync_ep{e}_preview.png")
        W, H, L = render_video(mp4, f0, t0, fps, e, v, lab, marg, Pord, a.stride, langs.get(e, ""), mp4o, png)
        print(f"  ep{e}: {W}x{H} {L}f corr={corr[e]:.3f} mono={mono[e]:.2%} → {mp4o}", flush=True)
        vid_info.append({"ep": int(e), "corr": corr[e], "mono": mono[e], "frames": int(L)})
    json.dump(vid_info, open(out / "videos.json", "w"), indent=1)
    print("LEROBOT_V24_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
