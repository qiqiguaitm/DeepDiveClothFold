"""逐帧诊断: ep763/ep1527 为何 value 不到 1.0(用户说是标准完整折)。
出接触印张: 全程均匀抽 ~18 帧 top_head 真实图, 每帧标注 [帧号 / 最近milestone value / 该簇coverage / 特征是否有效]。
看: 布料是否在干净折叠 + value 在哪一步开始跟丢 + 是否特征无效/域不匹配。
跑法: HF_HUB_OFFLINE=1 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_ep_contact.py
"""
from __future__ import annotations

import glob
import time

import cv2
import numpy as np

from crave.config import REPO, resolve_dataset
from crave.data import kai0
from crave.utils import otsu

OUTV = REPO / "crave/docs/visualization/centroid_decoder"
OUTD = REPO / "temp/crave_full"; ENC = "dino"; PICK = [763, 1527, 2302]


def main():
    t0 = time.time()
    cfg = resolve_dataset("kai0_base")
    # TODO(crave-lib): legacy full-scale dino shard layout (index_{ENC}.npz E/FR/T/n +
    #   {OUTD}/{ENC}/shard_*.npz gidx/feat/valid) is not what crave.data.load_dino_shards reads;
    #   re-inlined verbatim. Here we also keep the raw (pre-norm) feature norm to flag invalid frames.
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    rawnorm = np.linalg.norm(feat.astype(np.float32), axis=1)                # 原始范数(0=解码失败/无效)
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

    from crave.render import setup_mpl
    plt = setup_mpl()
    NC = 9
    for e in PICK:
        # 全局帧(含无效)
        gall = np.where(E == e)[0]; gall = gall[np.argsort(FR[gall])]
        ntot = len(gall); vfrac = float(valid[gall].mean())
        # 有效帧 → 逐帧最近 milestone value
        fiv = np.where(Ev == e)[0]; fiv = fiv[np.argsort(Tv[fiv])]
        Fq = F[fiv]; dd = np.linalg.norm(Fq[:, None] - C[None], axis=2); nm = dd.argmin(1); nv = Pk[nm]
        nearcov = cov[cl[nm]]
        # 均匀抽 2*NC 帧(全局, 看真实内容 + 有效性)
        idxs = gall[np.linspace(0, ntot - 1, 2 * NC).round().astype(int)]
        # 这些帧对应的有效帧位置(找同 FR)
        fr_of = FR[idxs]
        # 取真实图
        fm = kai0.grab_ep(cfg, e, [int(x) for x in fr_of])
        print(f"ep{e}: 全局帧{ntot} 有效率{vfrac:.2f} | 逐帧最近milestone value: min{nv.min():.2f} max{nv.max():.2f} 末3均{nv[-3:].mean():.2f} | 末帧最近簇cov{nearcov[-1]:.2f}", flush=True)
        fig, axes = plt.subplots(2, NC, figsize=(1.7 * NC, 4.0))
        for k, gi in enumerate(idxs):
            ax = axes[k // NC, k % NC]; fr = int(FR[gi])
            img = fm.get(fr, np.zeros((224, 224, 3), np.uint8)); ax.imshow(cv2.resize(img, (224, 224))); ax.axis("off")
            ok = "OK" if valid[gi] else "INVALID"
            # 该全局帧的逐帧 value(若有效)
            pos = np.where((Ev == e) & (FR[vi] == fr))[0]
            vv = float(Pk[np.argmin(np.linalg.norm(F[pos[0]][None] - C, axis=1))]) if len(pos) else float("nan")
            ax.set_title(f"f{fr} t={FR[gi]/max(1,FR[gall[-1]]):.2f}\nv={vv:.2f} {ok}", fontsize=7, color=("red" if not valid[gi] else "k"))
        fig.suptitle(f"ep{e}  top_head  全程 ({ntot} 帧, 有效率 {vfrac*100:.0f}%)  逐帧 v=最近milestone value  — 看折叠是否干净 + value 在哪跟丢", fontsize=11)
        fig.tight_layout(); out = OUTV / f"crave_ep{e}_contact.png"; fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"SAVED crave_ep{e}_contact.png", flush=True)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
