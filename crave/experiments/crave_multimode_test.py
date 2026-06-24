"""步骤2: 多模态 milestone 放置(循环簇=相对value)。对循环簇按成员时间 GMM 取多峰, 每峰放一个 emit 锚点。
ep2302 对比: 单模态(Pord 一个bin) vs 多模态(循环簇多bin)。验证 value 是否更顺 + advantage 密度是否升 + 仍 0→1。

Thin entrypoint over `crave`: `mkp`/`med`/`otsu`/`smooth_monotone` come from `crave.utils`,
`viterbi` from `crave.utils` (returns `(values, path)` → use `[0]`), `gpu_kmeans` from
`crave.clustering`, REPO from `crave.config`. The kai0_base dataset (DS/cs) used to map
features back to parquet proprio is re-inlined — see TODO.

跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_multimode_test.py
"""
import glob
import json
import time

import numpy as np
import pandas as pd

from crave.clustering import gpu_kmeans
from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import med, mkp, otsu, smooth_monotone, viterbi

# TODO(crave-lib): the kai0_base dataset (DS) + its chunks_size (cs) used to pull
# observation.state parquet for the precomputed dino feature shards should move into
# crave.config.datasets / crave.data — same kai0_base cache as crave_value_prod_test.py.
DS = REPO / "kai0/data/Task_A/kai0_base"
cs = json.load(open(DS / "meta/info.json"))["chunks_size"]

OUTV = REPO / "crave/docs/visualization/centroid_decoder"; OUTD = REPO / "temp/crave_full"; ENC = "dino"; EP = 2302


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
    F = np.concatenate([img, Pn], 1); K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    print(f"GPU KMeans K0={K0} ...", flush=True); cen, lab = gpu_kmeans(F, K0); ne = len(ep_list)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    rec_rate = np.zeros(K0)                                   # 每簇 ep内≥2次访问 比例
    for c in range(K0):
        memc = lab == c; epc = sorted(set(Ev[memc].tolist()))
        if len(epc) < 10: continue
        rec_rate[c] = np.mean([1 + int((np.diff(np.sort(FRv[(Ev == e) & memc])) > 35).sum()) >= 2 for e in epc])
    tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    # 选择: 原纯度闸 OR 高复现循环簇(rec_rate>0.4 & cov≥tau_cov) —— 把循环簇放回来
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and (tstd[c] <= tau_pur or rec_rate[c] > 0.4)], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel); n_rec_sel = int(sum(rec_rate[c] > 0.4 for c in sel))
    print(f"{M} milestones (含循环簇 {n_rec_sel})", flush=True)
    # precedence + isotonic(用首达中位定 Pk)
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
            if i != j:
                both = np.isfinite(fe[:, i]) & np.isfinite(fe[:, j])
                if both.sum() >= 5: Pbef[i, j] = float(np.mean(fe[both, i] < fe[both, j]))
    soft = np.nansum(np.where(np.isnan(Pbef), 0.0, Pbef), 1); prec = list(np.argsort(-soft))
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec])
    order = [sel[p] for p in prec]; Pord = np.asarray(iso, float); C = cen[order]

    # ---- 每 milestone 的 bins: 单模态(Pord) vs 多模态(循环簇 GMM 多峰)----
    from sklearn.mixture import GaussianMixture
    bins = np.linspace(0, 1, 41)
    cb_single = [[int(np.argmin(abs(bins - Pord[m])))] for m in range(M)]
    cb_multi = []
    for m in range(M):
        c = order[m]
        if rec_rate[c] > 0.4:                                 # 循环簇 → 多峰放置
            t = Tv[lab == c].reshape(-1, 1)
            best = None
            for k in (2, 3):
                try:
                    gm = GaussianMixture(k, random_state=0, n_init=2).fit(t)
                    if best is None or gm.bic(t) < best[1]: best = (gm, gm.bic(t))
                except Exception: pass
            if best:
                mus = sorted(best[0].means_.ravel().tolist()); ws = best[0].weights_
                modes = [mu for mu, w in zip(sorted(best[0].means_.ravel()), best[0].weights_[np.argsort(best[0].means_.ravel())]) if w > 0.15]
                cb_multi.append(sorted(set(int(np.argmin(abs(bins - mu))) for mu in modes)) or cb_single[m])
            else: cb_multi.append(cb_single[m])
        else: cb_multi.append(cb_single[m])

    nmb = sum(1 for m in range(M) if len(cb_multi[m]) > 1)
    print(f"多bin的 milestone 数: {nmb} (循环簇应得多bin); 例: " + str([(m, cb_single[m], cb_multi[m]) for m in range(M) if len(cb_multi[m]) > 1][:4]), flush=True)
    # ---- 多条 ep 读出对比 ----
    fi = np.where(Ev == EP)[0]; oo = np.argsort(FRv[fi]); fi = fi[oo]; Fq = F[fi]; n3 = len(fi)
    d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
    SP = np.concatenate([F[np.where(Ev == e)[0][np.argsort(Tv[np.where(Ev == e)[0]])][:2]] for e in ep_list])
    EPp = np.concatenate([F[np.where(Ev == e)[0][np.argsort(Tv[np.where(Ev == e)[0]])][-2:]] for e in ep_list])
    from sklearn.cluster import KMeans
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_; ek = KMeans(8, n_init=2, random_state=0).fit(EPp).cluster_centers_
    ds = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - ek[None], axis=2).min(1); tnn = np.arange(n3) / n3

    def readout(cb):
        em = np.full((n3, 41), 1e3)
        for m in range(M):
            for b in cb[m]: em[:, b] = np.minimum(em[:, b], d[:, m])
        em[:, 0] = np.minimum(em[:, 0], np.where(tnn < 0.3, ds, ds + (tnn - 0.3) * 6)); em[:, 40] = np.minimum(em[:, 40], np.where(tnn > 0.6, de, de + (0.6 - tnn) * 6))
        return smooth_monotone(med(viterbi(em, bins, 8.0)[0], 5), fps=3.0)
    v_s = readout(cb_single); v_m = readout(cb_multi)
    def advdens(v, W=15): return float(np.mean(np.abs(np.array([v[min(i + W, len(v) - 1)] - v[i] for i in range(len(v))])) > 1e-3))
    print(f"single: max={v_s.max():.2f} last={v_s[-3:].mean():.2f} advDensity={advdens(v_s):.2f}", flush=True)
    print(f"multi : max={v_m.max():.2f} last={v_m[-3:].mean():.2f} advDensity={advdens(v_m):.2f}", flush=True)
    # 多条 ep 对比(advDensity)
    for te in [0, 763, 1527, 2291, 3054]:
        ff = np.where(Ev == te)[0]; ff = ff[np.argsort(FRv[ff])]; dd = np.linalg.norm(F[ff][:, None] - C[None], axis=2); nn = len(ff)
        dsx = np.linalg.norm(F[ff][:, None] - sk[None], axis=2).min(1); dex = np.linalg.norm(F[ff][:, None] - ek[None], axis=2).min(1); tx = np.arange(nn) / nn
        def ro(cb):
            em = np.full((nn, 41), 1e3)
            for m in range(M):
                for b in cb[m]: em[:, b] = np.minimum(em[:, b], dd[:, m])
            em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6)); em[:, 40] = np.minimum(em[:, 40], np.where(tx > 0.6, dex, dex + (0.6 - tx) * 6))
            return smooth_monotone(med(viterbi(em, bins, 8.0)[0], 5), fps=3.0)
        a_s, a_m = ro(cb_single), ro(cb_multi)
        print(f"  ep{te}: single advD={advdens(a_s):.2f} max={a_s.max():.2f} | multi advD={advdens(a_m):.2f} max={a_m.max():.2f} | Δ={advdens(a_m)-advdens(a_s):+.2f}", flush=True)

    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(13, 4.5)); x = np.arange(n3)
    ax.plot(x, v_s, color="#999", lw=2, label=f"single-mode (advDensity {advdens(v_s):.2f})")
    ax.plot(x, v_m, color="#1a7f37", lw=2, label=f"multi-mode 循环簇 (advDensity {advdens(v_m):.2f})")
    ax.set_ylim(-.02, 1.02); ax.set_xlabel("frame (3Hz)"); ax.set_ylabel("value"); ax.grid(alpha=.3); ax.legend(fontsize=10)
    ax.set_title(f"ep2302: single vs MULTI-MODE milestone placement (循环簇 {n_rec_sel}/{M}) — 多模态让循环态在当前进度匹配=相对value")
    fig.tight_layout(); out = OUTV / "crave_multimode_test.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out.name}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
