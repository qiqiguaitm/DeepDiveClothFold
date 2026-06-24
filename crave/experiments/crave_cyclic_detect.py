"""步骤1: 量化循环簇(成员时间多峰=重复动作)。全量 image⊕proprio 聚类, 每簇成员时间 GMM 1vs2 峰检测。
统计: 循环簇数 / 其中多少被纯度闸(tstd≤P60)滤掉 / 典型动作。出图: top 循环簇 medoid + 时间双峰直方。
跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_cyclic_detect.py
"""
from __future__ import annotations

import glob
import json
import time

import av
import cv2
import numpy as np
import pandas as pd

from crave.clustering import gpu_kmeans
from crave.config import REPO
from crave.utils import mkp, otsu

OUTV = REPO / "crave/docs/visualization/centroid_decoder"
OUTD = REPO / "temp/crave_full"; ENC = "dino"

# TODO(crave-lib): kai0_base raw-video frame grabber (DS/cs/camp/crop224/grab_ep) should move into crave.data
#                  (a "kai0_base" DatasetConfig + a frame-grab loader). Re-inlined here verbatim.
DS = REPO / "kai0/data/Task_A/kai0_base"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]


def camp(e):
    return DS / "videos" / f"chunk-{e//cs:03d}" / "observation.images.top_head" / f"episode_{e:06d}.mp4"


def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])


def grab_ep(e, frames30):
    want = set(int(f) for f in frames30); out = {}
    try:
        c = av.open(str(camp(e)))
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = crop224(f.to_ndarray(format="rgb24"))
                if len(out) == len(want): break
        c.close()
    except Exception:
        pass
    return out


def main():
    t0 = time.time()
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; img = feat[vi].astype(np.float32); img /= (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    Ev, FRv, Tv = E[vi], FR[vi], T[vi]; ep_list = sorted(set(Ev.tolist()))
    P = np.zeros((len(vi), 28), np.float32)
    for e in ep_list:
        loc = np.where(Ev == e)[0]; o = np.argsort(FRv[loc]); loc = loc[o]
        st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
        P[loc] = mkp(st[np.minimum(FRv[loc], len(st) - 1)])
    Pn = (P - P.mean(0)) / (P.std(0) + 1e-8); Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9)
    F = np.concatenate([img, Pn], 1)
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    print(f"GPU KMeans K0={K0} ...", flush=True); cen, lab = gpu_kmeans(F, K0)
    ne = len(ep_list)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    sel_set = set(c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur)

    # ---- 真循环检测: 同一 ep 内被访问 ≥2 次(分离段, FR gap>35=>3帧@3Hz)----
    rec = []   # (cluster, rec_rate, avg_runs, n_eps, cov, in_sel)
    for c in range(K0):
        memc = lab == c
        if memc.sum() < 50: continue
        eps_c = sorted(set(Ev[memc].tolist())); multi = 0; runs = []
        for e in eps_c:
            fr = np.sort(FRv[(Ev == e) & memc])
            nr = 1 + int((np.diff(fr) > 35).sum())   # ep 内分离段数
            runs.append(nr); multi += (nr >= 2)
        rr = multi / len(eps_c); ar = float(np.mean(runs))
        if len(eps_c) >= 20 and rr > 0.25:           # ≥25% 的 ep 在一个 ep 内访问该态 ≥2 次 = 真循环
            rec.append((c, rr, ar, len(eps_c), float(cov[c]), c in sel_set))
    rec.sort(key=lambda r: -r[1])
    n_rec = len(rec); n_rec_sel = sum(1 for r in rec if r[5]); n_rec_drop = n_rec - n_rec_sel
    print(f"K0={K0} | 真循环簇(ep内≥2次访问, rate>25%): {n_rec} 个 | 入选 milestone: {n_rec_sel}, 被滤: {n_rec_drop}", flush=True)
    for r in rec[:14]:
        print(f"  clu{r[0]}: rec_rate={r[1]:.2f} avg_runs={r[2]:.2f} n_eps={r[3]} cov={r[4]:.2f} {'SELECTED' if r[5] else 'DROPPED'}", flush=True)

    # ---- 出图: top 循环簇 medoid + 时间直方 ----
    from crave.render import setup_mpl
    plt = setup_mpl()
    NS = min(10, n_rec)
    if NS == 0:
        print("无真循环簇 —— 步骤1 验证: 问题规模小, 不必加多模态", flush=True); return
    fig, axes = plt.subplots(2, NS, figsize=(1.7 * NS, 4.2))
    for k in range(NS):
        c, rr, ar, neps, cvg, insel = rec[k]
        loc = np.where(lab == c)[0]; d = np.linalg.norm(F[loc] - cen[c], axis=1); gi = vi[loc[int(np.argmin(d))]]
        fm = grab_ep(int(E[gi]), [int(FR[gi])]); im = fm.get(int(FR[gi]), np.zeros((224, 224, 3), np.uint8))
        axes[0, k].imshow(cv2.resize(im, (224, 224))); axes[0, k].axis("off")
        axes[0, k].set_title(f"clu{c} {'SEL' if insel else 'DROP'}\nrec={rr:.2f}", fontsize=7, color=("green" if insel else "red"))
        # 每 ep 内的访问段数分布
        nrs = [1 + int((np.diff(np.sort(FRv[(Ev == e) & (lab == c)])) > 35).sum()) for e in sorted(set(Ev[lab == c].tolist()))]
        axes[1, k].hist(nrs, bins=range(1, 7), color="#9c27b0", align="left"); axes[1, k].set_xlabel("visits/ep", fontsize=6); axes[1, k].tick_params(labelsize=5)
    fig.suptitle(f"真循环簇(ep内≥2次访问): {n_rec} 个 (入选{n_rec_sel}/被滤{n_rec_drop}) — 上=medoid 下=每ep访问段数分布", fontsize=11)
    fig.tight_layout(); out = OUTV / "crave_cyclic_detect.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
