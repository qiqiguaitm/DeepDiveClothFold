"""CRAVE frame->cluster alignment: metric definition, root-cause quantification, fixes, measurement.

Loads temp/crave_align/<ds>_cache.npz, replicates production clustering + precedence/isotonic +
milestone selection (from crave_generalize.py), then:
  - defines an alignment-accuracy metric (margin, contiguity, value-vs-time corr, indirection disagreement)
  - quantifies dominant cause per ds
  - evaluates fixes (direct nearest-ms, Viterbi-over-milestones, proprio upweight, temporal context)
  - keeps best, writes PNGs + JSON to temp/crave_align/

Run: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_align_analyze.py <vis|xvla|coffee>
"""
import sys, os, json, time
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from pathlib import Path
import crave_generalize as G  # reuse otsu, viterbi, gpu_kmeans, smooth_monotone via G.smooth? no -> import
from crave_readout import smooth_monotone

OUT = Path("/vePFS/tim/workspace/deepdive_kai0/temp/crave_align")
BINS = np.linspace(0, 1, 41)


# ---------------- clustering + milestone selection (replica of crave_generalize.main) ----------------
def build_clusters(F, E, Tv, ne, seed=0):
    N = len(F)
    K0 = int(np.clip(round(0.55 * np.sqrt(N)), 64, 320))
    cen, lab = G.gpu_kmeans(F, K0, seed=seed)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(E[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = G.otsu(cov)
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
    # start-anchor km (sk) as production
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    cb = [int(np.argmin(abs(BINS - Pord[m]))) for m in range(M)]
    return dict(K0=K0, cen=cen, lab=lab, sel=sel, order=order, M=M, C=C, Pord=Pord,
                cb=cb, sk=sk, tstd=tstd, cov=cov, tpos=tpos, tau_pur=tau_pur, tau_cov=tau_cov)


def med(a, w):
    h = w // 2
    return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


# ---------------- readout variants (per episode) ----------------
def readout_production(Fq, cl, fps=3.0):
    """Current production: Viterbi over 41 value-BINS, then displayed ms = argmin|Pord - value|."""
    C, sk, M, cb, Pord = cl["C"], cl["sk"], cl["M"], cl["cb"], cl["Pord"]
    nn = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nn, 41), 1e3)
    for m in range(M): em[:, cb[m]] = np.minimum(em[:, cb[m]], d[:, m])
    dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
    em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
    mw = max(5, int(round(5 * fps / 3))) | 1
    v = smooth_monotone(med(G.viterbi(em, BINS, 8.0), mw), fps=fps)
    ms = np.array([int(np.argmin(np.abs(Pord - v[t]))) for t in range(nn)])
    return v, ms


def readout_direct(Fq, cl, fps=3.0):
    """Fix (i): direct nearest-milestone assignment; value = Pord[assigned]."""
    C, Pord = cl["C"], cl["Pord"]
    d = np.linalg.norm(Fq[:, None] - C[None], axis=2)
    ms = d.argmin(1)
    v = Pord[ms]
    return v.astype(np.float32), ms


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


# ---------------- metric ----------------
def alignment_metrics(allFq, allms, allv, allTv, eps_list, cl):
    """Returns dict of alignment-accuracy components, aggregated over episodes.
    allFq/allms/allv/allTv: lists per episode."""
    C = cl["C"]; M = cl["M"]
    margins, frag_runs, flips, corr_list = [], [], [], []
    n_total = 0
    for Fq, ms, v, Tt in zip(allFq, allms, allv, allTv):
        nn = len(Fq); n_total += nn
        d = np.linalg.norm(Fq[:, None] - C[None], axis=2)  # (nn,M)
        ds = np.sort(d, axis=1)
        margin = (ds[:, 1] - ds[:, 0]) / (ds[:, 1] + 1e-9)  # normalized d1/d2 margin in [0,1]
        margins.append(margin)
        # contiguity: number of runs of constant ms / ideal (M-ish). report runs per episode normalized
        runs = 1 + int(np.sum(ms[1:] != ms[:-1]))
        n_unique = len(np.unique(ms))
        frag_runs.append(runs / max(1, n_unique))  # 1.0 = perfectly contiguous
        # per-frame flip rate
        flips.append(np.mean(ms[1:] != ms[:-1]))
        # value vs normalized-time spearman-ish (pearson on ranks)
        if nn > 3 and np.std(v) > 1e-6:
            corr_list.append(np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(Tt)))[0, 1])
    margin_all = np.concatenate(margins)
    # indirection disagreement of THIS readout: displayed ms vs true nearest milestone
    C = cl["C"]; dis = 0; tot = 0
    for Fq, ms in zip(allFq, allms):
        true_ms = np.linalg.norm(Fq[:, None] - C[None], axis=2).argmin(1)
        dis += int(np.sum(ms != true_ms)); tot += len(ms)
    return dict(
        margin_mean=float(margin_all.mean()),
        margin_p10=float(np.percentile(margin_all, 10)),
        frag_ratio=float(np.mean(frag_runs)),   # runs / unique-ms; lower(->1) = more contiguous
        flip_rate=float(np.mean(flips)),         # fraction of adjacent frames changing ms
        time_corr=float(np.mean(corr_list)),     # value rank vs time rank corr (higher=better)
        indir_disagree=float(dis / max(1, tot)), # displayed-ms vs true-nearest-ms disagreement
        n_frames=n_total,
    )


def disagreement_indirection(allFq, allms_prod, cl):
    """% frames where production displayed ms != true nearest milestone (cause 1 magnitude)."""
    C = cl["C"]; dis = 0; tot = 0
    for Fq, ms in zip(allFq, allms_prod):
        nn = len(Fq); true_ms = np.linalg.norm(Fq[:, None] - C[None], axis=2).argmin(1)
        dis += int(np.sum(ms != true_ms)); tot += nn
    return dis / max(1, tot)


def aliasing_timespread(cl, Tv):
    """Within-milestone temporal spread of assigned frames (cause 2/3): mean tstd of kept milestones,
    plus bimodality count (frames split across early & late)."""
    lab, sel = cl["lab"], cl["sel"]
    spreads = []; bimod = 0
    for c in sel:
        t = Tv[lab == c]
        if len(t) > 2:
            spreads.append(float(t.std()))
            hist, _ = np.histogram(t, bins=np.linspace(0, 1, 11))
            # bimodal if mass in both first-third and last-third
            if hist[:3].sum() > 0.15 * len(t) and hist[-3:].sum() > 0.15 * len(t):
                bimod += 1
    return float(np.mean(spreads)) if spreads else 0.0, bimod, len(sel)


def dropped_transient(cl, Tv):
    """Cause 5: fraction of clusters dropped by purity gate that are high-time-variance & well-covered."""
    lab, cov, tstd, tau_pur, tau_cov, K0 = cl["lab"], cl["cov"], cl["tstd"], cl["tau_pur"], cl["tau_cov"], cl["K0"]
    sel = set(cl["sel"])
    dropped_by_purity = [c for c in range(K0) if cov[c] >= tau_cov and tstd[c] > tau_pur and tstd[c] < 9]
    n_frames_dropped = int(sum((lab == c).sum() for c in dropped_by_purity))
    tot = len(lab)
    return len(dropped_by_purity), n_frames_dropped / max(1, tot)


# ---------------- main ----------------
def run(ds, wproprio=1.0, tempctx=False):
    """wproprio scales proprio block; tempctx appends short-window mean+delta of pooled img feat."""
    z = np.load(OUT / f"{ds}_cache.npz")
    img, Pm, E, Tv, thumb = z["img"], z["state"], z["ep"], z["tpos"], z["thumb"]
    ne = len(np.unique(E))
    # normalize proprio as production
    PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
    Pn = (Pm - PMU) / PSD; Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9)
    blocks = [img, Pn * wproprio]
    if tempctx:
        # temporal context: per-episode short-window mean and delta of pooled img feature
        ctx_mean = np.zeros_like(img); ctx_d = np.zeros_like(img)
        for e in np.unique(E):
            fi = np.where(E == e)[0]; x = img[fi]
            k = 3; cm = np.zeros_like(x)
            for j in range(len(x)):
                cm[j] = x[max(0, j - k):j + 1].mean(0)
            d = np.zeros_like(x); d[1:] = x[1:] - x[:-1]
            ctx_mean[fi] = cm; ctx_d[fi] = d
        # small weight so it adds temporal smoothness without dominating
        blocks += [ctx_mean * 0.3, ctx_d * 0.3]
    F = np.concatenate(blocks, 1).astype(np.float32)
    cl = build_clusters(F, E, Tv, ne)
    return F, E, Tv, thumb, cl


def eval_config(F, E, Tv, cl, readout_fn, **kw):
    eps_list = sorted(set(E.tolist()))
    allFq, allms, allv, allTv = [], [], [], []
    for e in eps_list:
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        Fq = F[fi]; Tt = Tv[fi]
        v, ms = readout_fn(Fq, cl, **kw)
        allFq.append(Fq); allms.append(ms); allv.append(v); allTv.append(Tt)
    m = alignment_metrics(allFq, allms, allv, allTv, eps_list, cl)
    return m, allFq, allms, allv, allTv, eps_list


def main(ds):
    t0 = time.time()
    res = {"ds": ds}
    # ---------- baseline build (production feature: img + proprio*1) ----------
    F, E, Tv, thumb, cl = run(ds, wproprio=1.0)
    res["M"] = cl["M"]; res["K0"] = cl["K0"]; res["n_eps"] = len(np.unique(E))
    print(f"[{ds}] M={cl['M']} K0={cl['K0']} eps={res['n_eps']} ({time.time()-t0:.0f}s)", flush=True)

    # baseline = production readout
    mb, Fq_b, ms_b, v_b, Tv_b, eps_list = eval_config(F, E, Tv, cl, readout_production)
    res["baseline"] = mb
    # causes
    res["cause1_indirection_disagree"] = disagreement_indirection(Fq_b, ms_b, cl)
    sp, bimod, nsel = aliasing_timespread(cl, Tv); res["cause23_aliasing"] = dict(mean_tstd=sp, bimodal_ms=bimod, n_ms=nsel)
    res["cause4_jitter_fliprate"] = mb["flip_rate"]
    ndrop, fdrop = dropped_transient(cl, Tv); res["cause5_dropped"] = dict(n_clusters=ndrop, frac_frames=fdrop)
    print(f"[{ds}] baseline {mb}", flush=True)
    print(f"[{ds}] causes: indir={res['cause1_indirection_disagree']:.3f} alias_tstd={sp:.3f} "
          f"bimod={bimod}/{nsel} flip={mb['flip_rate']:.3f} dropped={ndrop}({fdrop:.2%})", flush=True)

    # ---------- proprio ablation ----------
    res["ablation"] = {}
    for tag, w in [("img_only", 0.0), ("w1", 1.0), ("w2", 2.0), ("w3", 3.0)]:
        Fa, Ea, Tva, _, cla = run(ds, wproprio=w)
        ma, _, msa, _, _, _ = eval_config(Fa, Ea, Tva, cla, readout_viterbi_ms)
        # aliasing under this weight
        spa, ba, na = aliasing_timespread(cla, Tva)
        res["ablation"][tag] = dict(metrics=ma, M=cla["M"], alias_tstd=spa, bimodal=ba)
        print(f"[{ds}] ablation {tag}: M={cla['M']} flip={ma['flip_rate']:.3f} frag={ma['frag_ratio']:.2f} "
              f"corr={ma['time_corr']:.3f} alias_tstd={spa:.3f}", flush=True)

    # ---------- fixes on baseline feature ----------
    res["fixes"] = {}
    md, _, ms_d, v_d, _, _ = eval_config(F, E, Tv, cl, readout_direct)
    res["fixes"]["direct_nearest"] = md
    for lam in [4.0, 8.0, 16.0]:
        mv, Fq_v, ms_v, v_v, _, _ = eval_config(F, E, Tv, cl, readout_viterbi_ms, lam=lam)
        res["fixes"][f"viterbi_ms_lam{int(lam)}"] = mv
    print(f"[{ds}] fix direct {md}", flush=True)

    # ---------- temporal-context feature + best viterbi ----------
    Ft, Et, Tvt, _, clt = run(ds, wproprio=1.0, tempctx=True)
    mt, _, ms_t, v_t, _, _ = eval_config(Ft, Et, Tvt, clt, readout_viterbi_ms, lam=8.0)
    res["fixes"]["tempctx_viterbi_lam8"] = dict(metrics=mt, M=clt["M"])

    json.dump(res, open(OUT / f"{ds}_metrics.json", "w"), indent=2, default=float)
    print(f"[{ds}] DONE {time.time()-t0:.0f}s → {OUT/f'{ds}_metrics.json'}", flush=True)

    # save the per-episode arrays needed for plotting best vs baseline
    np.savez(OUT / f"{ds}_curves.npz",
             eps=np.array(eps_list),
             allow_pickle=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1])
