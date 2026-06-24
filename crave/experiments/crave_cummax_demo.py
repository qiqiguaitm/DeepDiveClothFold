"""演示: cummax 单调(防崩/不谎报) + 完成残差flag, 对比 no-anchor。3Hz cache, 不重渲全帧。

Thin entrypoint over `crave`: `L2`/`med`/`otsu`/`smooth_monotone` come from `crave.utils`,
`gpu_kmeans` from `crave.clustering`, REPO/out paths from `crave.config`. `build_clusters`
+ `readout_viterbi_ms` are the legacy `crave_align_analyze` helpers (no library equivalent
yet) re-inlined below — see TODOs.

跑法: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_cummax_demo.py
"""
import numpy as np
from sklearn.cluster import KMeans

from crave.clustering import gpu_kmeans
from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import L2, med, otsu, smooth_monotone

OUT = REPO / "temp/crave_align"

BINS = np.linspace(0, 1, 41)


# TODO(crave-lib): build_clusters (full-frame cluster → precedence/isotonic milestone
# selection, returns the `cl` dict consumed by the readout_* variants) should move into
# crave.clustering — it is the crave_align_analyze replica of crave_generalize.main.
def build_clusters(F, E, Tv, ne, seed=0):
    N = len(F)
    K0 = int(np.clip(round(0.55 * np.sqrt(N)), 64, 320))
    cen, lab = gpu_kmeans(F, K0, seed=seed)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(E[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov)
    tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel); eps_sorted = sorted(set(E.tolist()))
    fe = np.full((len(eps_sorted), M), np.nan)
    for ei, e in enumerate(eps_sorted):
        fi = np.where(E == e)[0]; labe = lab[fi]; te = Tv[fi]
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
    Pord = np.asarray(iso, float)
    order = [sel[p] for p in prec]; C = cen[order]  # milestone centroids in value order
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    cb = [int(np.argmin(abs(BINS - Pord[m]))) for m in range(M)]
    return dict(K0=K0, cen=cen, lab=lab, sel=sel, order=order, M=M, C=C, Pord=Pord,
                cb=cb, sk=sk, tstd=tstd, cov=cov, tpos=tpos, tau_pur=tau_pur, tau_cov=tau_cov)


# TODO(crave-lib): readout_viterbi_ms (forward-aware hard-start Viterbi DP directly over M
# milestone centers, value = Pord[assigned]) should move into crave (e.g.
# crave.value / crave.utils) — it is the crave_align_analyze readout variant.
def readout_viterbi_ms(Fq, cl, lam=8.0, fps=3.0):
    """Fix (ii): Viterbi DP directly over M milestones; emit=dist, transition=lam*|Pord[i]-Pord[j]|.
    value = Pord[assigned] (no bins, no value->bin->ms indirection)."""
    C, sk, Pord, M = cl["C"], cl["sk"], cl["Pord"], cl["M"]
    nn = len(Fq)
    emit = np.linalg.norm(Fq[:, None] - C[None], axis=2)  # (nn,M) dist to each milestone
    dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
    emit[:, 0] = np.minimum(emit[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))  # start anchor -> milestone 0
    pen = lam * np.abs(Pord[:, None] - Pord[None])  # (M,M)
    cost = np.full(M, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((nn, M), int)  # HARD start: frame0=milestone0(value0), prevents start aliasing
    for j in range(1, nn):
        tr = cost[None, :] + pen          # tr[i,k] = cost_prev[k] + pen[i,k]
        k = tr.argmin(1); cost = emit[j] + tr[np.arange(M), k]; bp[j] = k
    ms = np.zeros(nn, int); ms[-1] = int(cost.argmin())
    for j in range(nn - 2, -1, -1): ms[j] = bp[j + 1, ms[j + 1]]
    mw = max(5, int(round(5 * fps / 3))) | 1
    v = smooth_monotone(med(Pord[ms], mw), fps=fps)
    return v.astype(np.float32), ms


DS_EPS = {"coffee": [0, 1], "xvla": [7, 39], "vis": [2, 121]}


def main():
    plt = setup_mpl()
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for row, ds in enumerate(["coffee", "xvla", "vis"]):
        c = np.load(OUT / f"{ds}_cache.npz")
        img = L2(c["img"].astype(np.float32)); st = c["state"].astype(np.float32)
        stn = L2((st - st.mean(0)) / (st.std(0) + 1e-8))
        F = np.concatenate([img, stn], 1); E = c["ep"]; T = c["tpos"]; eps_s = sorted(set(E.tolist()))
        cl = build_clusters(F, E, T, len(eps_s))
        EPe = np.concatenate([F[np.where(E == e)[0][np.argsort(T[np.where(E == e)[0]])][-2:]] for e in eps_s])
        ek = KMeans(8, n_init=2, random_state=0).fit(EPe).cluster_centers_
        de_tr = np.array([float(np.linalg.norm(F[np.where(E == e)[0][np.argmax(T[np.where(E == e)[0]])]][None] - ek, axis=1).min()) for e in eps_s])
        de_thr = float(np.quantile(de_tr, 0.9)) * 1.3
        for col, e in enumerate(DS_EPS[ds]):
            fi = np.where(E == e)[0]; fi = fi[np.argsort(T[fi])]; Fq = F[fi]; tt = T[fi]
            v, ms = readout_viterbi_ms(Fq, cl, lam=8.0, fps=3.0)   # 原始 value(保留震荡, 不cummax)
            de_end = float(np.linalg.norm(Fq[-3:][:, None] - ek[None], axis=2).min())
            comp = de_end <= de_thr
            ax = axes[row, col]
            ax.plot(np.linspace(0, 1, len(v)), color="0.7", ls="--", lw=1, label="norm time")
            ax.plot(v, color="#1a7f37", lw=2, label=f"value (corr {np.corrcoef(v,tt)[0,1]:.2f})")
            ax2 = ax.twinx(); ax2.step(range(len(ms)), ms, where="post", color="#9c27b0", lw=1, alpha=.55); ax2.set_ylabel("milestone", color="#9c27b0", fontsize=7); ax2.tick_params(labelsize=6)
            ax.set_ylim(-.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
            flag = "COMPLETE" if comp else "INCOMPLETE(半完成)"
            ax.set_title(f"[{ds}] ep{e}  M={cl['M']}  末value={v[-1]:.2f}  | flag: {flag}(resid {de_end:.2f}/thr {de_thr:.2f})", fontsize=9, color=("#1a7f37" if comp else "#d62728"))
    fig.suptitle("保留震荡(循环milestone重访) + 完成残差flag  (不 cummax, 不 end-anchor)", fontsize=13)
    fig.tight_layout(); out = OUT / "osc_flag_demo.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig); print("SAVED", out)


if __name__ == "__main__":
    main()
