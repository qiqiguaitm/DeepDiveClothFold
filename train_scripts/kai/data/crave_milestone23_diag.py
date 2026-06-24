"""剖析某 milestone(默认 precedence rank 23, value≈0.6)为何 value 偏低(用户觉得应在 ~0.86)。
3path-lite(image⊕proprio)聚类, 看该簇:成员进度 Tv 分布(是否双峰/宽)+ 每 ep 首达分布 + 抽样成员真实帧。
跑法: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_milestone23_diag.py [rank]
"""
import sys, glob, time
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
import numpy as np, cv2, pandas as pd
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from crave_decoder_scale_ablation import REPO, DS, cs, grab_ep
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"
OUTD = REPO / "temp/crave_full"; ENC = "dino"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 23


def otsu(xs):
    s = np.unique(np.sort(xs)); bt, bv = s[0], -1
    for t in s:
        lo, hi = xs[xs < t], xs[xs >= t]
        if len(lo) and len(hi):
            v = (len(lo) / len(xs)) * (len(hi) / len(xs)) * (lo.mean() - hi.mean()) ** 2
            if v > bv: bv, bt = v, t
    return bt


def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


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
    from sklearn.cluster import MiniBatchKMeans
    K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    fit = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit]); cen = km.cluster_centers_; lab = km.predict(F)
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

    print(f"M={M}. rank→(iso_val, Pk, tpos, cov):", flush=True)
    for r in range(M):
        m = prec[r]; print(f"  r{r:2d} iso={iso_val[m]:.2f} Pk={Pk[m]:.2f} tpos={tpos[cl[m]]:.2f} cov={cov[cl[m]]:.2f} clu{cl[m]}", flush=True)

    m = prec[TARGET]; c = cl[m]; mem = np.where(lab == c)[0]; tm = Tv[mem]; em = Ev[mem]
    pctl = {p: round(float(np.percentile(tm, p)), 3) for p in [5, 10, 25, 50, 75, 90, 95]}
    # 每 ep 内: 该簇出现的时间(全部, 不只首达)
    fe_first = fe[:, TARGET]; fe_first = fe_first[~np.isnan(fe_first)]
    print(f"\n=== milestone rank{TARGET} (clu{c}) 内部分析 ===", flush=True)
    print(f"iso_val={iso_val[m]:.3f} Pk(首达中位)={Pk[m]:.3f} tpos(成员均)={tpos[c]:.3f} cov={cov[c]:.2f} tstd={tstd[c]:.3f}", flush=True)
    print(f"成员 {len(mem)} 帧, 跨 {len(set(em.tolist()))} ep", flush=True)
    print(f"成员进度 Tv 分位: {pctl}", flush=True)
    print(f"首达时间分位: p25={np.percentile(fe_first,25):.2f} p50={np.percentile(fe_first,50):.2f} p75={np.percentile(fe_first,75):.2f}", flush=True)
    # 双峰检测: 直方
    hist, edges = np.histogram(tm, bins=20, range=(0, 1))
    print(f"Tv 直方(20 bins):{hist.tolist()}", flush=True)

    # 抽样成员帧(按 Tv 排序均匀取 12)
    order = mem[np.argsort(tm)]; pick = order[np.linspace(0, len(order) - 1, 12).round().astype(int)]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5)); gs = fig.add_gridspec(2, 12)
    axh = fig.add_subplot(gs[0, :6]); axh.hist(tm, bins=25, range=(0, 1), color="#4c78a8"); axh.axvline(Pk[m], color="r", label=f"Pk={Pk[m]:.2f}"); axh.axvline(np.median(tm), color="g", ls="--", label=f"median Tv={np.median(tm):.2f}")
    axh.set_title(f"rank{TARGET} clu{c}: member progress Tv hist (cov={cov[c]:.2f} tstd={tstd[c]:.2f})", fontsize=9); axh.legend(fontsize=8); axh.set_xlabel("progress Tv")
    axfe = fig.add_subplot(gs[0, 6:]); axfe.hist(fe_first, bins=25, range=(0, 1), color="#e45756"); axfe.set_title(f"per-ep FIRST-ENTRY time (Pk=median={Pk[m]:.2f})", fontsize=9); axfe.set_xlabel("first-entry Tv")
    for k, gi in enumerate(pick):
        g = vi[gi]; fm = grab_ep(int(E[g]), [int(FR[g])]); ax = fig.add_subplot(gs[1, k])
        im = fm.get(int(FR[g]), np.zeros((224, 224, 3), np.uint8)); ax.imshow(cv2.resize(im, (224, 224))); ax.axis("off")
        ax.set_title(f"ep{int(E[g])}\nTv={Tv[gi]:.2f}", fontsize=6)
    fig.suptitle(f"WHY rank{TARGET} value={iso_val[m]:.2f} (低)? 成员进度分布 + 抽样帧 — 看是否双峰/复现/混相位", fontsize=11)
    fig.tight_layout(); out = OUTV / f"crave_milestone{TARGET}_diag.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
