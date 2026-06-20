"""全量 3055ep: 聚类用 DINOv2-large 图像 ⊕ proprio(消起末别名), 渲染只取 medoid 真实图像 → Wan 解码。
验证 ep763/1527 是否到 1.0(纯图像卡 0.15/0.32)。复用 image shard 特征 + 从 parquet 取 proprio。
跑法: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_full_3path.py
"""
import sys, glob, json, time
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
import numpy as np, cv2, torch, pandas as pd
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from crave_decoder_scale_ablation import REPO, DS, cs, grab_ep
from crave_readout import smooth_monotone
from diffusers import AutoencoderKLWan
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"
OUTD = REPO / "temp/crave_full"; WAN = "checkpoints/Wan2.2-TI2V-5B-Diffusers"; dev = "cuda"; ENC = "dino"
TEST = [763, 1527, 2302]


def otsu(xs):
    s = np.unique(np.sort(xs)); bt, bv = s[0], -1
    for t in s:
        lo, hi = xs[xs < t], xs[xs >= t]
        if len(lo) and len(hi):
            v = (len(lo) / len(xs)) * (len(hi) / len(xs)) * (lo.mean() - hi.mean()) ** 2
            if v > bv: bv, bt = v, t
    return bt


def med(a, w):
    h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


def viterbi(emit, bins, lam, eb=2.0):
    NB = len(bins); pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
    cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= eb; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return bins[path]


def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def main():
    t0 = time.time()
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]
    img = feat[vi].astype(np.float32); img /= (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    Ev, FRv, Tv = E[vi], FR[vi], T[vi]
    # ---- proprio: 每 ep 读 parquet, 取 state[FR], mkp ----
    print("提取 proprio(读 parquet)...", flush=True)
    ep_list = sorted(set(Ev.tolist())); P = np.zeros((len(vi), 28), np.float32)
    for ix, e in enumerate(ep_list):
        loc = np.where(Ev == e)[0]; o = np.argsort(FRv[loc]); loc = loc[o]
        pq = DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet"
        st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
        frs = np.minimum(FRv[loc], len(st) - 1); P[loc] = mkp(st[frs])
        if ix % 500 == 0: print(f"  proprio {ix}/{len(ep_list)} ({time.time()-t0:.0f}s)", flush=True)
    PMU, PSD = P.mean(0), P.std(0) + 1e-8; Pn = (P - PMU) / PSD; Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9)
    F = np.concatenate([img, Pn], 1)                       # 2-path: image ⊕ proprio(各 L2-norm, 等权)
    print(f"3path-lite 特征 {F.shape}; 聚类 ...", flush=True)

    from sklearn.cluster import MiniBatchKMeans
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    fit_idx = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit_idx]); cen = km.cluster_centers_; lab = km.predict(F)
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
    # precedence + isotonic value
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
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec])
    iso_val = np.empty(M); iso_val[np.array(prec)] = iso
    C = cen[cl]
    # start/end 锚点
    SP, EP = [], []
    for e in ep_list:
        fi = np.where(Ev == e)[0]; oo = np.argsort(Tv[fi]); fi = fi[oo]
        if len(fi) >= 2: SP.append(F[fi[:2]]); EP.append(F[fi[-2:]])
    from sklearn.cluster import KMeans
    startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
    endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_
    print(f"K0={K0} → {M} milestones; 测试读出 ...", flush=True)

    def readout(Fq, Pval, NB=41):
        bins = np.linspace(0, 1, NB); n = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
        cb = [int(np.argmin(abs(bins - Pval[m]))) for m in range(M)]; em = np.full((n, NB), 1e3)
        for m in range(M): em[:, cb[m]] = np.minimum(em[:, cb[m]], d[:, m])
        ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
        tn = np.arange(n) / n
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        return smooth_monotone(med(viterbi(em, bins, lam=8.0), 5), fps=3.0)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(TEST), figsize=(5.2 * len(TEST), 4))
    for k, e in enumerate(TEST):
        fi = np.where(Ev == e)[0]; oo = np.argsort(Tv[fi]); fi = fi[oo]
        vc = readout(F[fi], iso_val); tn = Tv[fi]
        print(f"ep{e}: n={len(fi)} image+proprio value max={vc.max():.2f} last={vc[-3:].mean():.2f}", flush=True)
        ax = axes[k]; ax.plot(tn, vc, color="#1a7f37", lw=2)
        ax.set_title(f"ep{e} ({len(fi)}fr) image⊕proprio  max={vc.max():.2f} last={vc[-3:].mean():.2f}", fontsize=9)
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("progress"); ax.set_ylabel("value"); ax.grid(alpha=.3)
    fig.suptitle("3path-lite (DINOv2-large image ⊕ proprio) value — does ep763/1527 reach 1.0 now? (image-only stuck 0.15/0.32)", fontsize=11)
    fig.tight_layout(); o1 = OUTV / "crave_3path_value_test.png"; fig.savefig(o1, dpi=120, bbox_inches="tight"); plt.close(fig); print(f"SAVED {o1.name}", flush=True)

    # ---- medoid 画廊(渲染只取真实图像 → Wan)----
    print("Wan 渲 medoid 画廊(precedence 序)...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    def wan_dec(z):
        with torch.no_grad(): o = vae.decode(torch.from_numpy(z[None, :, None]).to(dev)).sample
        return np.clip((o[0, :, 0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
    meds = []
    for m in prec:
        loc = np.where(lab == cl[m])[0]; d = np.linalg.norm(F[loc] - cen[cl[m]], axis=1); gi = vi[loc[int(np.argmin(d))]]
        fm = grab_ep(int(E[gi]), [int(FR[gi])])
        if int(FR[gi]) not in fm: meds.append(np.zeros((256, 256, 3), np.uint8)); continue
        im = cv2.resize(fm[int(FR[gi])], (256, 256), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(im.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None, :, None].to(dev)
        with torch.no_grad():
            ee = vae.encode(x); zz = (ee.latent_dist.mode() if hasattr(ee, "latent_dist") else ee.latent)[0, :, 0].cpu().numpy()
        meds.append(wan_dec(zz))
    ncol = min(14, M); nrow = int(np.ceil(M / ncol))
    fg, ax2 = plt.subplots(nrow, ncol, figsize=(1.3 * ncol, 1.5 * nrow)); ax2 = np.atleast_2d(ax2)
    for idx in range(nrow * ncol):
        r2, c2 = divmod(idx, ncol); a = ax2[r2, c2]; a.axis("off")
        if idx < M: a.imshow(meds[idx]); a.set_title(f"v={iso_val[prec[idx]]:.2f}", fontsize=6)
    fg.suptitle(f"3path-lite milestones (image⊕proprio cluster, image-only Wan render) — {M} milestones, precedence order", fontsize=11)
    fg.tight_layout(); o2 = OUTV / "crave_3path_gallery.png"; fg.savefig(o2, dpi=125, bbox_inches="tight"); plt.close(fg)
    print(f"SAVED {o2.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
