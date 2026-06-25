"""Option-2 fix for the cyclic-cluster spurious value-drop: fold the empirical milestone
transition prior P(next|cur) into the readout DP, and compare vs the geometric forward-Viterbi
on coffee (where ep46 etc. drop to 0 mid-episode due to late states aliasing the start milestone).

Mechanism: forward moves cost `lam_geo·ΔP + beta·(−log Pf)` (data orders/skip-levels cheaply);
backward moves cost `lam_geo·|ΔP| + back_barrier`. Since successful demos almost never go
mid→start, a higher back_barrier blocks the alias while the forward −logP keeps the true track cheap.

Run: CUDA_VISIBLE_DEVICES=0 PY crave/experiments/transition_prior_fix.py [--ds coffee]
Out: crave/docs/visualization/cross_dataset/<ds>_transition_fix.png (+ _transition.json)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crave.config import REPO, resolve_dataset
from crave.data import list_eps, load_ep
from crave.encoders import load_encoder
from crave.render import setup_mpl
from crave.utils import L2, med, mkp_gap, smooth_monotone
from generalize import build_milestones, make_readout
OUTV = REPO / "crave/docs/visualization/cross_dataset"


def visited_sequence(am, w=5, min_run=2):
    a = med(am.astype(float), w).round().astype(int); seq = []; i = 0; n = len(a)
    while i < n:
        j = i
        while j < n and a[j] == a[i]: j += 1
        if j - i >= min_run: seq.append(int(a[i]))
        i = j
    out = [seq[0]] if seq else []
    for s in seq[1:]:
        if s != out[-1]: out.append(s)
    return out


def build_pen(counts, Pord, lam_geo, beta, back_barrier, alpha=0.05):
    fwd = (Pord[None] > Pord[:, None] + 1e-9)
    Pf = np.where(fwd, counts, 0.0)
    Pf = (Pf + alpha * fwd) / (Pf.sum(1, keepdims=True) + alpha * fwd.sum(1, keepdims=True) + 1e-9)
    fcost = -np.log(Pf + 1e-12); fcost = fcost - np.where(fwd, fcost, np.inf).min(1, keepdims=True)
    geo = lam_geo * np.abs(Pord[:, None] - Pord[None]); back = (Pord[None] < Pord[:, None] - 1e-9)
    return geo + np.where(fwd, beta * fcost, 0.0) + np.where(back, back_barrier, 0.0)


def viterbi_pen(emit, pen):
    nn, Mn = emit.shape; cost = np.full(Mn, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((nn, Mn), int)
    for t in range(1, nn):
        tr = cost[:, None] + pen; k = tr.argmin(0); cost = emit[t] + tr[k, np.arange(Mn)]; bp[t] = k
    ms = np.zeros(nn, int); ms[-1] = int(cost.argmin())
    for t in range(nn - 2, -1, -1): ms[t] = bp[t + 1, ms[t + 1]]
    return ms


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="coffee"); ap.add_argument("--encoder", default="dinov3-h")
    ap.add_argument("--betas", type=float, nargs="+", default=[2.0]); ap.add_argument("--lam-geo", type=float, default=3.0)
    ap.add_argument("--barriers", type=float, nargs="+", default=[4.0, 10.0, 20.0, 40.0])
    a = ap.parse_args(); t0 = time.time(); cfg = resolve_dataset(a.ds); enc = load_encoder(a.encoder)
    eps = list_eps(cfg); print(f"[{a.ds}] {len(eps)} eps, encoding...", flush=True)
    POOL, STATE, EPID, TPOS, eplen = [], [], [], [], {}
    for k, e in enumerate(eps):
        try: f224, state, _t, _ = load_ep(cfg, e, strd=1)
        except Exception: continue
        if len(f224) < 5: continue
        POOL.append(L2(enc.encode_pooled(f224))); STATE.append(mkp_gap(state, cfg.stride))
        n = len(f224); EPID.append(np.full(n, e)); TPOS.append(np.arange(n) / max(1, n - 1)); eplen[e] = n
        if (k + 1) % 25 == 0: print(f"  {k+1}/{len(eps)}", flush=True)
    img = np.concatenate(POOL); Pm = np.concatenate(STATE); E = np.concatenate(EPID); Tv = np.concatenate(TPOS)
    F = np.concatenate([img, L2((Pm - Pm.mean(0)) / (Pm.std(0) + 1e-8))], 1); ne = len(eps)
    cen, lab, order, Pord, M = build_milestones(F, E, Tv, ne)
    C = cen[order]; eps_sorted = sorted(set(E.tolist()))
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    print(f"[{a.ds}] {M} milestones; building transition matrix...", flush=True)

    am_all = np.linalg.norm(F[:, None] - C[None], axis=2).argmin(1)            # per-frame nearest milestone
    counts = np.zeros((M, M))
    for e in eps_sorted:
        seq = visited_sequence(am_all[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])]])
        for i, j in zip(seq[:-1], seq[1:]): counts[i, j] += 1

    def emit_of(Fq):
        em = np.linalg.norm(Fq[:, None] - C[None], axis=2); nn = len(Fq)
        dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
        em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6)); return em

    def corr_set(readout_ms):
        cs, curves = [], {}
        for e in eps_sorted:
            fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
            if len(fi) < 5: continue
            ms = readout_ms(F[fi]); v = smooth_monotone(med(Pord[ms], 5), fps=3.0); tq = Tv[fi]
            cs.append(float(np.corrcoef(v, tq)[0, 1]) if (v.std() > 1e-6) else np.nan)
            curves[e] = (tq, v)
        return np.array(cs), curves

    geo_ro = make_readout(C, sk, Pord)
    geo_c, geo_cur = corr_set(lambda Fq: geo_ro(Fq)[1])
    res = {"geometric(down25)": (geo_c, geo_cur)}
    for beta in a.betas:
        for bb in a.barriers:
            pen = build_pen(counts, Pord, a.lam_geo, beta, bb)
            tc, tcur = corr_set(lambda Fq, pen=pen: viterbi_pen(emit_of(Fq), pen))
            res[f"trans β{beta} back{bb:.0f}"] = (tc, tcur)
    for name, (c, _) in res.items():
        ok = np.isfinite(c)
        print(f"  {name:24s}: mean={c[ok].mean():.3f} med={np.median(c[ok]):.3f} %>=0.7={np.mean(c[ok]>=0.7):.0%}", flush=True)

    best = max([k for k in res if k.startswith("trans")], key=lambda k: np.mean(np.nan_to_num(res[k][0]) >= 0.7))
    EP46 = 46 if 46 in eplen else eps_sorted[np.argmin(geo_c)]
    plt = setup_mpl(); fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    gc = res["geometric(down25)"][0]; bc = res[best][0]
    ax[0].hist(gc[np.isfinite(gc)], bins=22, alpha=0.6, color="#888", label=f"几何(down25) %≥0.7={np.mean(gc[np.isfinite(gc)]>=0.7):.0%}")
    ax[0].hist(bc[np.isfinite(bc)], bins=22, alpha=0.6, color="#1a7f37", label=f"{best} %≥0.7={np.mean(bc[np.isfinite(bc)]>=0.7):.0%}")
    ax[0].axvline(0.7, color="r", ls="--"); ax[0].set_title(f"[{a.ds}] corr 分布: 几何 vs 转移先验"); ax[0].set_xlabel("corr(value, 进度)"); ax[0].legend(fontsize=9)
    tqg, vg = res["geometric(down25)"][1][EP46]; tqt, vt = res[best][1][EP46]
    ax[1].plot(tqg, vg, color="#d62728", lw=2, label="几何(down25): 塌到0")
    ax[1].plot(tqt, vt, color="#1a7f37", lw=2, label=f"{best}: 修复")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=.4); ax[1].set_title(f"[{a.ds}] ep{EP46} value: 转移先验消除循环态假跌"); ax[1].set_xlabel("进度"); ax[1].set_ylabel("value"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3); ax[1].set_ylim(-.02, 1.02)
    fig.suptitle(f"循环簇假跌 — 转移先验(选②)修复 — {a.ds}, {a.encoder}, {M} milestones", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); OUTV.mkdir(parents=True, exist_ok=True)
    out = OUTV / f"{a.ds}_transition_fix.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
    json.dump({name: {"mean": float(np.nanmean(c)), "median": float(np.nanmedian(c)), "frac_ge_0.7": float(np.mean(np.nan_to_num(c) >= 0.7))}
               for name, (c, _) in res.items()}, open(OUTV / f"{a.ds}_transition.json", "w"), indent=2)
    print(f"SAVED {out}  best={best} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
