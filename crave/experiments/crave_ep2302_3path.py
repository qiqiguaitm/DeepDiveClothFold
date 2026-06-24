"""新架构重做 ep2302 30Hz 分析:
全量 kai0_base 聚类 = DINOv2-large 图像 ⊕ proprio(GPU KMeans)→ precedence 定序 + isotonic value;
ep2302 用其全量 milestone 读出 value(3Hz, 复用 shard 特征)→ 升采样到 30Hz 出图 + 完整过程视频(Wan medoid 渲染)。
跑法: HF_HUB_OFFLINE=1 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_ep2302_3path.py
"""
from __future__ import annotations

import glob
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from crave.clustering import gpu_kmeans
from crave.config import REPO, resolve_dataset
from crave.data import kai0
from crave.encoders import load_encoder
from crave.render import setup_mpl
from crave.utils import med, mkp, otsu, smooth_monotone, viterbi

from crave_ep2302_30hz_decoded import decode_all_frames, render_video

OUTV = REPO / "crave/docs/visualization/centroid_decoder"
OUTD = REPO / "temp/crave_full"; ENC = "dino"; EP = 2302


def main():
    t0 = time.time()
    cfg = resolve_dataset("kai0_base")
    cs = kai0.chunks_size(cfg.root)
    DS = Path(cfg.root)
    # TODO(crave-lib): legacy full-scale dino shard layout (index_{ENC}.npz E/FR/T/n +
    #   {OUTD}/{ENC}/shard_*.npz gidx/feat/valid) is incompatible with crave.data.load_dino_shards;
    #   re-inlined verbatim.
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; img = feat[vi].astype(np.float32); img /= (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    Ev, FRv, Tv = E[vi], FR[vi], T[vi]; ep_list = sorted(set(Ev.tolist()))
    print("proprio ...", flush=True); P = np.zeros((len(vi), 28), np.float32)
    for e in ep_list:
        loc = np.where(Ev == e)[0]; o = np.argsort(FRv[loc]); loc = loc[o]
        st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
        P[loc] = mkp(st[np.minimum(FRv[loc], len(st) - 1)])
    PMU, PSD = P.mean(0), P.std(0) + 1e-8; Pn = (P - PMU) / PSD; Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9)
    F = np.concatenate([img, Pn], 1)
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    print(f"GPU KMeans K0={K0} ...", flush=True); cen, lab = gpu_kmeans(F, K0)
    ne = len(ep_list)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel)
    fe = np.full((ne, M), np.nan)
    for ei, e in enumerate(ep_list):
        fi = np.where(Ev == e)[0]; labe = lab[fi]; te = Tv[fi]
        for m in range(M):
            hit = te[labe == sel[m]]
            if len(hit): fe[ei, m] = hit.min()
    Pk = np.array([np.nanmedian(fe[:, m]) for m in range(M)])
    Pbef = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i == j: continue
            both = np.isfinite(fe[:, i]) & np.isfinite(fe[:, j])
            if both.sum() >= 5: Pbef[i, j] = float(np.mean(fe[both, i] < fe[both, j]))
    soft = np.nansum(np.where(np.isnan(Pbef), 0.0, Pbef), 1); prec = list(np.argsort(-soft))
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec])
    order = [sel[p] for p in prec]; Pord = np.asarray(iso, float); C = cen[order]   # 排序后 milestone 中心 + isotonic value
    SP = np.concatenate([F[np.where(Ev == e)[0][np.argsort(Tv[np.where(Ev == e)[0]])][:2]] for e in ep_list])
    EPp = np.concatenate([F[np.where(Ev == e)[0][np.argsort(Tv[np.where(Ev == e)[0]])][-2:]] for e in ep_list])
    from sklearn.cluster import KMeans
    startK = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_; endK = KMeans(8, n_init=2, random_state=0).fit(EPp).cluster_centers_
    print(f"全量 {ne}ep → {M} milestones (precedence+isotonic)", flush=True)

    # ---- ep2302 3Hz 读出(复用 shard 特征)----
    fi = np.where(Ev == EP)[0]; oo = np.argsort(FRv[fi]); fi = fi[oo]; Fq = F[fi]; n3 = len(fi)
    bins = np.linspace(0, 1, 41); d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
    cb = [int(np.argmin(abs(bins - Pord[m]))) for m in range(M)]; emit = np.full((n3, 41), 1e3)
    for m in range(M): emit[:, cb[m]] = np.minimum(emit[:, cb[m]], d[:, m])
    ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
    tnn = np.arange(n3) / n3
    emit[:, 0] = np.minimum(emit[:, 0], np.where(tnn < 0.3, ds, ds + (tnn - 0.3) * 6)); emit[:, 40] = np.minimum(emit[:, 40], np.where(tnn > 0.6, de, de + (0.6 - tnn) * 6))
    vraw, _ = viterbi(emit, bins, 8.0); v3 = smooth_monotone(med(vraw, 5), fps=3.0)
    ms3 = np.array([int(np.argmin(np.abs(Pord - v3[t]))) for t in range(n3)])           # 每帧 milestone(按 value 最近)
    print(f"ep2302: 3Hz {n3} 帧, value {v3.min():.2f}→{v3.max():.2f}, 访问 milestone {sorted(set(ms3.tolist()))}", flush=True)

    # ---- milestone medoid 图(Wan 解码真实帧)----
    print("Wan 渲 milestone medoid ...", flush=True)
    wan = load_encoder("wan-vae")
    proto = {}
    for mi, c in enumerate(order):
        loc = np.where(lab == c)[0]; dd = np.linalg.norm(F[loc] - cen[c], axis=1); gi = vi[loc[int(np.argmin(dd))]]
        fm = kai0.grab_ep(cfg, int(E[gi]), [int(FR[gi])]); im = fm.get(int(FR[gi]))
        if im is None: proto[mi] = np.zeros((128, 128, 3), np.uint8); continue
        zz = wan.encode_latents([cv2.resize(im, (256, 256))])     # (1,48,16,16) Wan latent (mode)
        proto[mi] = cv2.resize(wan.decode(zz)[0], (128, 128))

    # ---- 30Hz 相机帧(显示)+ 升采样 value/milestone ----
    print("解 ep2302 30Hz 相机帧 ...", flush=True)
    frames = decode_all_frames(cfg, EP); n30 = len(frames)
    xi = np.linspace(0, 1, n3); xo = np.linspace(0, 1, n30)
    v30 = np.interp(xo, xi, v3); ms30 = ms3[np.clip((xo * (n3 - 1)).round().astype(int), 0, n3 - 1)]

    # ---- 图(value 曲线 + milestone 随时间 + medoid 条)----
    plt = setup_mpl()
    fig = plt.figure(figsize=(15, 7)); gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.3, 2])
    ax1 = fig.add_subplot(gs[0]); ax1.plot(np.arange(n3), v3, color="#1a7f37", lw=2, label="CRAVE value (img⊕proprio, precedence+isotonic)")
    for p in Pord: ax1.axhline(p, color="0.9", lw=.5)
    ax1.set_ylim(-.02, 1.02); ax1.set_ylabel("value"); ax1.legend(fontsize=9); ax1.set_title(f"ep2302 30Hz — NEW arch (DINOv2-large⊕proprio cluster, {M} milestones) — value {v3.min():.2f}→{v3.max():.2f}")
    ax2 = fig.add_subplot(gs[1]); ax2.step(np.arange(n3), ms3, where="post", color="#9c27b0"); ax2.set_ylabel("milestone idx"); ax2.set_xlabel("frame (3Hz)"); ax2.set_title("milestone over time", fontsize=9)
    nsh = min(16, M); shi = [int(x) for x in np.linspace(0, M - 1, nsh).round()]
    for k, mi in enumerate(shi):
        axp = fig.add_subplot(gs[2].subgridspec(1, nsh)[0, k]); axp.imshow(proto[mi]); axp.axis("off"); axp.set_title(f"m{mi}\nv={Pord[mi]:.2f}", fontsize=6)
    fig.suptitle("ep2302 NEW-arch analysis: image⊕proprio cluster + precedence/isotonic value + Wan medoid", fontsize=12)
    fig.tight_layout(); o1 = OUTV / "crave_ep2302_3path.png"; fig.savefig(o1, dpi=120, bbox_inches="tight"); plt.close(fig); print(f"SAVED {o1.name}", flush=True)

    # ---- 视频(复用 render_video)----
    print("渲染视频 ...", flush=True)
    render_video(frames, v30, ms30, proto, Pord, OUTV / "crave_ep2302_3path.mp4", fps=30)
    print(f"SAVED crave_ep2302_3path.mp4  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
