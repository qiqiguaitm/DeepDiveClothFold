"""诊断: 为何某些 ep 最终 value 不到 1.0。看末帧真实图 + 末帧最近 milestone 的 value + 全程达到的最高 milestone。
判定: 末帧已折好但 value 低 = 读出/锚点问题; 末帧本就没折完 = value 正确(短/未完成 demo)。
跑法: HF_HUB_OFFLINE=1 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_ep_value_diag.py
"""
from __future__ import annotations

import glob
import json
import time

import cv2
import numpy as np

from crave.config import REPO, resolve_dataset
from crave.data import kai0
from crave.utils import otsu

OUTV = REPO / "crave/docs/visualization/centroid_decoder"
OUTD = REPO / "temp/crave_full"; ENC = "dino"
PICK = [2302, 0, 763, 1527, 2291]


def main():
    t0 = time.time()
    cfg = resolve_dataset("kai0_base")
    # TODO(crave-lib): full-scale dino shard cache here uses the legacy layout
    #   index_{ENC}.npz (keys E/FR/T/n) + {OUTD}/{ENC}/shard_*.npz (keys gidx/feat/valid).
    #   crave.data.load_dino_shards expects index_dino.npz (keys ep/fr/tpos) + shard_*.npz key "f",
    #   so it cannot read this cache; re-inlined verbatim (same as crave_full_cluster.stage_aggregate).
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; Fn = feat.astype(np.float32); Fn /= (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-9)
    F = Fn[vi]; Ev, Tv = E[vi], T[vi]
    from sklearn.cluster import MiniBatchKMeans
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    fit_idx = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit_idx]); cen = km.cluster_centers_; lab = km.predict(F)
    ne = len(set(E.tolist()))
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel); cl = np.array(sel)
    ep_list = sorted(set(Ev.tolist())); fe = np.full((len(ep_list), M), np.nan)
    for ei, e in enumerate(ep_list):
        fi = np.where(Ev == e)[0]; labe = lab[fi]; te = Tv[fi]
        for m in range(M):
            hit = te[labe == cl[m]]
            if len(hit): fe[ei, m] = hit.min()
    Pk = np.array([np.nanmedian(fe[:, m]) for m in range(M)]); C = cen[cl]
    # milestone value(用 Pk 即可, 诊断够)
    mval = Pk

    from crave.render import setup_mpl
    plt = setup_mpl()
    fig, axes = plt.subplots(len(PICK), 3, figsize=(7.5, 2.2 * len(PICK)))
    for r, e in enumerate(PICK):
        fi = np.where(Ev == e)[0]; o = np.argsort(Tv[fi]); fi = fi[o]
        Fq = F[fi]; d = np.linalg.norm(Fq[:, None] - C[None], axis=2); nm = d.argmin(1)  # nearest milestone per frame
        nv = mval[nm]                                                   # nearest-milestone value per frame
        maxv = float(nv.max()); lastv = float(nv[-3:].mean()); n = len(fi)
        # 末帧 raw 图(真实 30fps 帧号 = FR)
        fr0 = int(FR[fi[0]]); frL = int(FR[fi[-1]])
        fm = kai0.grab_ep(cfg, e, [fr0, frL])
        img0 = cv2.resize(fm.get(fr0, np.zeros((256, 256, 3), np.uint8)), (256, 256))
        imgL = cv2.resize(fm.get(frL, np.zeros((256, 256, 3), np.uint8)), (256, 256))
        print(f"ep{e}: n={n} maxV={maxv:.2f} lastV={lastv:.2f} last_nearest_milestone_cov={cov[cl[nm[-1]]]:.2f}", flush=True)
        axes[r, 0].plot(Tv[fi], nv, ".-", ms=3); axes[r, 0].set_ylim(-0.02, 1.02); axes[r, 0].set_title(f"ep{e} nearest-milestone value\nmaxV={maxv:.2f} lastV={lastv:.2f}", fontsize=8); axes[r, 0].grid(alpha=.3)
        axes[r, 1].imshow(img0); axes[r, 1].axis("off"); axes[r, 1].set_title("first frame", fontsize=8)
        axes[r, 2].imshow(imgL); axes[r, 2].axis("off"); axes[r, 2].set_title(f"LAST frame (f{frL})", fontsize=8)
    fig.suptitle("Why value < 1.0?  nearest-milestone value over time + first/last REAL frame  (folded last frame but low V = readout/anchor; unfolded = correct)", fontsize=10)
    fig.tight_layout(); out = OUTV / "crave_ep_value_diag.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
