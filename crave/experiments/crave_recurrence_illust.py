"""§2.1 灵感来源示意图:挑一个高覆盖的 kai0 簇,展示「簇中心解码图 + 若干来自不同 episode 的最近帧」,
直观体现"同一 milestone 在不同 episode 反复出现 = 任务必经结构"。

复用 crave_full_7b_centroid 的 DINOv3-H 缓存特征(temp/crave_full_dinov3h/)+ _grids_for + train_dec。
Run: CUDA_VISIBLE_DEVICES=0 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_recurrence_illust.py [--k0 120] [--ncol 6]
输出: crave/docs/visualization/crave_recurrence_milestone.png
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crave_full_7b_centroid as C            # 默认 ENC=dinov3-h, OUTD=temp/crave_full_dinov3h, cfg=kai0_base
from crave.decoding.decoder import train_dec  # noqa: E402
from crave.encoders import load_encoder        # noqa: E402
from crave.render import setup_mpl             # noqa: E402

plt = setup_mpl()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k0", type=int, default=120)
    ap.add_argument("--ncol", type=int, default=6)        # 展示几张不同 episode 的最近帧
    ap.add_argument("--nn", type=int, default=24)         # 簇中心解码用的近邻数(平均 grid)
    ap.add_argument("--train-imgs", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=45)
    a = ap.parse_args()
    OUTD = C.OUTD; E, FR = None, None
    z = np.load(OUTD / "index.npz"); E, FR, T, N = z["E"], z["FR"], z["T"], int(z["n"])
    feat = np.zeros((N, C.DIM), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / "shard_*.npz"))):
        s = np.load(f); feat[s["gidx"]] = s["feat"]; valid[s["gidx"]] = s["valid"]
    vi = np.where(valid)[0]; F = C.L2(feat[vi].astype(np.float32))
    Tv, Ev = T[vi], E[vi]; ne = len(set(E.tolist()))
    print(f"[illust] {len(vi)} frames / {ne} ep; KMeans K0={a.k0} ...", flush=True)
    fit = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(a.k0, random_state=0, batch_size=4096, n_init=3).fit(F[fit])
    cen = km.cluster_centers_; lab = km.predict(F)
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(a.k0)])
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(a.k0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(a.k0)])
    # 合适的簇:中段(避开起止别名)+ 时序紧(coherent)+ 覆盖最大
    cand = [c for c in range(a.k0) if 0.2 <= tpos[c] <= 0.8 and tstd[c] < np.median(tstd[tstd < 9])]
    cstar = max(cand, key=lambda c: cov[c]) if cand else int(np.argmax(cov))
    print(f"[illust] 选中簇 c={cstar}: 覆盖 {cov[cstar]:.1%} episodes, 进度≈{tpos[cstar]:.2f}, tstd={tstd[cstar]:.3f}", flush=True)

    loc = np.where(lab == cstar)[0]; d = np.linalg.norm(F[loc] - cen[cstar], axis=1); order = loc[np.argsort(d)]
    # 每个不同 episode 取最近一帧 → 展示"跨 episode 复现"
    seen, picks = set(), []
    for li in order:
        e = int(Ev[li])
        if e in seen: continue
        seen.add(e); picks.append(li)
        if len(picks) >= a.ncol: break
    near_global = vi[picks]; near_eps = [int(Ev[li]) for li in picks]
    nn_global = vi[order[:a.nn]]
    print(f"[illust] 最近帧来自 {a.ncol} 条不同 episode: {near_eps}", flush=True)

    enc = load_encoder(C.ENC)
    rs = np.random.RandomState(1); tr = vi[rs.choice(len(vi), min(a.train_imgs, len(vi)), replace=False)]
    g_tr, im_tr, ok_tr = C._grids_for(enc, tr, E, FR); keep = np.where(ok_tr)[0]
    gg_nn, _, ok_nn = C._grids_for(enc, nn_global, E, FR)
    _, th_near, ok_near = C._grids_for(enc, near_global, E, FR)
    cen_grid = gg_nn[np.where(ok_nn)[0]].mean(0)
    enc.unload()
    import torch; torch.cuda.empty_cache()
    print(f"[illust] training 16->128 decoder ({len(keep)} pairs, {a.epochs}ep) ...", flush=True)
    decode = train_dec(g_tr[keep], im_tr[keep], C.DIM, dec="small", epochs=a.epochs)
    dec_cen = decode(cen_grid[None].astype(np.float32))[0]

    # 构图:左=簇中心解码图(大),右=N 张不同 episode 的最近真实帧
    ncol = a.ncol
    fig = plt.figure(figsize=(2.0 + 1.6 * ncol, 3.4))
    gs = fig.add_gridspec(1, ncol + 1, width_ratios=[1.5] + [1] * ncol, wspace=0.08)
    a0 = fig.add_subplot(gs[0]); a0.imshow(dec_cen); a0.axis("off")
    a0.set_title(f"簇中心解码图\n(覆盖 {cov[cstar]:.0%} episodes)", fontsize=10, color="#7c3aed")
    for j in range(ncol):
        axj = fig.add_subplot(gs[j + 1]); axj.imshow(th_near[j]); axj.axis("off")
        axj.set_title(f"ep{near_eps[j]}", fontsize=9)
    fig.suptitle(f"同一 milestone(任务进度≈{tpos[cstar]:.2f})在 {ncol} 条不同 episode 反复出现 —— 反复出现 = 任务必经结构(CRAVE 的灵感来源)",
                 fontsize=11.5, y=1.04)
    out = C.REPO / "crave/docs/visualization/crave_recurrence_milestone.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
