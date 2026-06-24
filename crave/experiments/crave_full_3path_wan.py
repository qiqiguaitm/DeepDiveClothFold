"""全量 3055ep: 聚类用 Wan2.2-VAE latent ⊕ proprio(对照 DINOv2-large⊕proprio), 渲染解 medoid 的 Wan latent。
看 "Wan 编码 + proprio 消歧" 在同架构(自适应K0+precedence+isotonic)下效果如何 + ep763/1527 是否到 1.0。
复用 8卡 all-Wan 任务存的 wan shard(12288维) + parquet proprio。
跑法: HF_HUB_OFFLINE=1 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_full_3path_wan.py

Thin entrypoint over `crave`: REPO/viz_dir + kai0_base via crave.config, GPU KMeans via
crave.clustering.gpu_kmeans, Wan VAE decode via crave.encoders.load_encoder, otsu/med/mkp/
viterbi/smooth_monotone from crave.utils. The temp/crave_full shard cache (gidx/feat/valid +
index E/FR/T/n) uses a different on-disk layout than crave.data.load_dino_shards, so its loading
stays inline — see TODO.
"""
import glob, time
import numpy as np, pandas as pd
from pathlib import Path

from crave.clustering import gpu_kmeans
from crave.config import REPO, resolve_dataset, viz_dir
from crave.data import kai0
from crave.encoders import load_encoder
from crave.render import setup_mpl
from crave.utils import med, mkp, otsu, smooth_monotone, viterbi

_cfg = resolve_dataset("kai0_base")
DS = Path(_cfg.root); cs = kai0.chunks_size(_cfg.root)
OUTV = viz_dir("centroid_decoder")
OUTD = REPO / "temp/crave_full"; dev = "cuda"; ENC = "wan"; DIM = 12288
TEST = [763, 1527, 2302]


def main():
    t0 = time.time()
    # TODO(crave-lib): temp/crave_full shard cache (gidx/feat/valid + index E/FR/T/n) ->
    # crave.data.load_dino_shards expects f-shards + ep/fr/tpos index; loading stays inline.
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, DIM), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]
    raw = feat[vi].astype(np.float32)                                  # 原始 Wan latent(留作解码)
    img = (raw - raw.mean(0)) / (raw.std(0) + 1e-6)                    # Wan latent 标准化
    img /= (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    Ev, FRv, Tv = E[vi], FR[vi], T[vi]; ep_list = sorted(set(Ev.tolist()))
    print("提取 proprio ...", flush=True)
    P = np.zeros((len(vi), 28), np.float32)
    for e in ep_list:
        loc = np.where(Ev == e)[0]; o = np.argsort(FRv[loc]); loc = loc[o]
        st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
        P[loc] = mkp(st[np.minimum(FRv[loc], len(st) - 1)])
    Pn = (P - P.mean(0)) / (P.std(0) + 1e-8); Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9)
    F = np.concatenate([img, Pn], 1); del feat, img
    print(f"Wan-latent⊕proprio 特征 {F.shape}; 聚类 ...", flush=True)

    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    print(f"GPU KMeans K0={K0} on {F.shape} ...", flush=True)
    cen, lab = gpu_kmeans(F, K0, iters=25)
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
    M = len(sel); cl = np.array(sel)
    fe = np.full((len(ep_list), M), np.nan)
    for ei, e in enumerate(ep_list):
        fi = np.where(Ev == e)[0]; labe = lab[fi]; te = Tv[fi]
        for m in range(M):
            hit = te[labe == cl[m]]
            if len(hit): fe[ei, m] = hit.min()
    Pk = np.array([np.nanmedian(fe[:, m]) for m in range(M)])
    Pbef = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i == j: continue
            both = ~np.isnan(fe[:, i]) & ~np.isnan(fe[:, j])
            if both.sum() >= 5: Pbef[i, j] = float(np.mean(fe[both, i] < fe[both, j]))
    soft = np.array([np.nansum(Pbef[i, :]) for i in range(M)]); prec = list(np.argsort(-soft))
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec]); iso_val = np.empty(M); iso_val[np.array(prec)] = iso
    C = cen[cl]
    SP, EP = [], []
    for e in ep_list:
        fi = np.where(Ev == e)[0]; oo = np.argsort(Tv[fi]); fi = fi[oo]
        if len(fi) >= 2: SP.append(F[fi[:2]]); EP.append(F[fi[-2:]])
    from sklearn.cluster import KMeans
    startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
    endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_

    def inv(order):
        n = 0
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                i, j = order[a], order[b]
                if not np.isnan(Pbef[i, j]) and Pbef[i, j] < 0.5: n += 1
        return n
    print(f"K0={K0} → {M} milestones; tpos序逆序={inv(list(np.argsort([tpos[cl[m]] for m in range(M)])))} prec序逆序={inv(prec)}; 簇内时间std中位={np.median(tstd[tstd<9]):.3f}", flush=True)

    def readout(Fq, Pval, NB=41):
        bins = np.linspace(0, 1, NB); n = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
        cb = [int(np.argmin(abs(bins - Pval[m]))) for m in range(M)]; em = np.full((n, NB), 1e3)
        for m in range(M): em[:, cb[m]] = np.minimum(em[:, cb[m]], d[:, m])
        ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
        tn = np.arange(n) / n
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        return smooth_monotone(med(viterbi(em, bins, lam=8.0)[0], 5), fps=3.0)

    plt = setup_mpl()
    fig, axes = plt.subplots(1, len(TEST), figsize=(5.2 * len(TEST), 4))
    for k, e in enumerate(TEST):
        fi = np.where(Ev == e)[0]; oo = np.argsort(Tv[fi]); fi = fi[oo]
        vc = readout(F[fi], iso_val); tn = Tv[fi]
        print(f"ep{e}: n={len(fi)} Wan⊕proprio value max={vc.max():.2f} last={vc[-3:].mean():.2f} mono={np.mean(np.diff(vc)>=-1e-6):.2f}", flush=True)
        ax = axes[k]; ax.plot(tn, vc, color="#9c27b0", lw=2)
        ax.set_title(f"ep{e} ({len(fi)}fr) Wan⊕proprio  max={vc.max():.2f} last={vc[-3:].mean():.2f}", fontsize=9)
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("progress"); ax.set_ylabel("value"); ax.grid(alpha=.3)
    fig.suptitle("Wan2.2-latent ⊕ proprio (same arch) value — vs DINOv2-large⊕proprio", fontsize=11)
    fig.tight_layout(); o1 = OUTV / "crave_3path_wan_value.png"; fig.savefig(o1, dpi=120, bbox_inches="tight"); plt.close(fig); print(f"SAVED {o1.name}", flush=True)

    print("Wan 解码 medoid latent ...", flush=True)
    enc = load_encoder("wan-vae")
    meds = []
    for m in prec:
        loc = np.where(lab == cl[m])[0]; d = np.linalg.norm(F[loc] - cen[cl[m]], axis=1); gl = loc[int(np.argmin(d))]
        meds.append(enc.decode(raw[gl].reshape(48, 16, 16)[None])[0])  # 解 medoid 原始 Wan latent
    ncol = min(14, M); nrow = int(np.ceil(M / ncol))
    fg, ax2 = plt.subplots(nrow, ncol, figsize=(1.3 * ncol, 1.5 * nrow)); ax2 = np.atleast_2d(ax2)
    for idx in range(nrow * ncol):
        r2, c2 = divmod(idx, ncol); a = ax2[r2, c2]; a.axis("off")
        if idx < M: a.imshow(meds[idx]); a.set_title(f"v={iso_val[prec[idx]]:.2f}", fontsize=6)
    fg.suptitle(f"Wan2.2-latent⊕proprio milestones (Wan medoid-latent decode) — {M} milestones, precedence order", fontsize=11)
    fg.tight_layout(); o2 = OUTV / "crave_3path_wan_gallery.png"; fg.savefig(o2, dpi=125, bbox_inches="tight"); plt.close(fg)
    print(f"SAVED {o2.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
