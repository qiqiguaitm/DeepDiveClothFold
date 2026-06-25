"""Rebalance the transition prior: similarity(emission) PRIMARY, transition prior AUXILIARY.
Process at native 30Hz. Sweep transition weight (emission-only → heavy) and measure milestone
JUMP rate + corr, so we can see how a too-strong prior freezes milestone transitions (e.g. XVLA's
repeated cloth-flattening should re-trigger jumps once the prior is light).

Caches the encode so re-sweeps are cheap.
Run: CUDA_VISIBLE_DEVICES=0 PY crave/experiments/transition_rebalance.py --ds xvla --n-eps 80
Out: crave/docs/visualization/cross_dataset/<ds>_rebalance.png (+ _rebalance.json)
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
from generalize import build_milestones
from transition_prior_fix import build_pen, viterbi_pen, visited_sequence
OUTV = REPO / "crave/docs/visualization/cross_dataset"


def emit_of(Fq, C, sk):
    em = np.linalg.norm(Fq[:, None] - C[None], axis=2); nn = len(Fq)
    dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
    em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6)); return em


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="xvla"); ap.add_argument("--encoder", default="dinov3-h")
    ap.add_argument("--n-eps", type=int, default=80); ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(); t0 = time.time(); cfg = resolve_dataset(a.ds)
    cache = REPO / f"temp/xreb_cache_{a.ds}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        F, E, Tv, lab, Pord = z["F"].astype(np.float32), z["E"], z["Tv"], z["lab"], z["Pord"]
        order, cen, counts = z["order"], z["cen"], z["counts"]; eplen = dict(zip(z["ek"].tolist(), z["ev"].tolist()))
        print(f"[{a.ds}] loaded cache: N={len(F)} M={len(order)}", flush=True)
    else:
        enc = load_encoder(a.encoder); eps = list_eps(cfg)[: a.n_eps]
        print(f"[{a.ds}] {len(eps)} eps @30Hz(strd=1), encoding...", flush=True)
        POOL, STATE, EPID, TPOS, eplen = [], [], [], [], {}
        for k, e in enumerate(eps):
            try: f224, state, _t, _ = load_ep(cfg, e, strd=1)
            except Exception: continue
            if len(f224) < 5: continue
            POOL.append(L2(enc.encode_pooled(f224))); STATE.append(mkp_gap(state, cfg.stride))
            n = len(f224); EPID.append(np.full(n, e)); TPOS.append(np.arange(n) / max(1, n - 1)); eplen[e] = n
            if (k + 1) % 25 == 0: print(f"  {k+1}/{len(eps)} ({time.time()-t0:.0f}s)", flush=True)
        img = np.concatenate(POOL); Pm = np.concatenate(STATE); E = np.concatenate(EPID); Tv = np.concatenate(TPOS)
        F = np.concatenate([img, L2((Pm - Pm.mean(0)) / (Pm.std(0) + 1e-8))], 1)
        cen, lab, order, Pord, M = build_milestones(F, E, Tv, len(eps))
        C = cen[order]; am = np.linalg.norm(F[:, None] - C[None], axis=2).argmin(1); counts = np.zeros((M, M))
        for e in sorted(set(E.tolist())):
            seq = visited_sequence(am[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])]])
            for i, j in zip(seq[:-1], seq[1:]): counts[i, j] += 1
        np.savez(cache, F=F.astype(np.float16), E=E, Tv=Tv, lab=lab, Pord=Pord, order=order, cen=cen, counts=counts,
                 ek=np.array(list(eplen)), ev=np.array(list(eplen.values())))
        print(f"[{a.ds}] encoded+cached: N={len(F)} M={M} ({time.time()-t0:.0f}s)", flush=True)

    C = cen[order]; M = len(order); eps_sorted = sorted(set(E.tolist()))
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_

    # configs: emission PRIMARY → transition AUXILIARY (low) → heavy (old)
    cfgs = {"emission-only": build_pen(counts, Pord, 0.3, 0.0, 0.0),
            "light β0.4 back1": build_pen(counts, Pord, 1.0, 0.4, 1.0),
            "aux β0.6 back3": build_pen(counts, Pord, 1.5, 0.6, 3.0),
            "heavy β2 back40": build_pen(counts, Pord, 3.0, 2.0, 40.0)}
    mw = max(5, int(round(5 * a.fps / 3))) | 1
    stats = {}; ms_long = {}
    elong = max(eplen, key=lambda k: eplen[k])
    for name, pen in cfgs.items():
        njump, nuniq, corrs = [], [], []
        for e in eps_sorted:
            fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
            if len(fi) < 5: continue
            ms = viterbi_pen(emit_of(F[fi], C, sk), pen)
            njump.append(int(np.sum(np.diff(ms) != 0))); nuniq.append(len(set(ms.tolist())))
            v = smooth_monotone(med(Pord[ms], mw), fps=a.fps); tq = Tv[fi]
            corrs.append(float(np.corrcoef(v, tq)[0, 1]) if v.std() > 1e-6 else np.nan)
            if e == elong: ms_long[name] = (Tv[fi], ms, v)
        c = np.array(corrs); ok = np.isfinite(c)
        stats[name] = {"jump_per_ep": float(np.mean(njump)), "uniq_per_ep": float(np.mean(nuniq)),
                       "corr_mean": float(c[ok].mean()), "frac_ge_0.7": float(np.mean(c[ok] >= 0.7))}
        print(f"  {name:18s}: jumps/ep={np.mean(njump):.0f} uniq/ep={np.mean(nuniq):.1f} corr={c[ok].mean():.3f} %>=0.7={np.mean(c[ok]>=0.7):.0%}", flush=True)

    plt = setup_mpl(); fig, ax = plt.subplots(2, 1, figsize=(13, 7), height_ratios=[1.4, 1])
    names = list(cfgs); jr = [stats[n]["jump_per_ep"] for n in names]; cr = [stats[n]["corr_mean"] for n in names]
    axb = ax[0]; x = np.arange(len(names)); axb.bar(x - 0.2, jr, 0.4, color="#d62728", label="milestone 跳变次数/ep")
    axb.set_xticks(x); axb.set_xticklabels(names, fontsize=9); axb.set_ylabel("跳变次数/ep", color="#d62728")
    ax2 = axb.twinx(); ax2.plot(x, cr, "-o", color="#1a7f37", label="corr(value,进度)"); ax2.set_ylabel("corr", color="#1a7f37"); ax2.set_ylim(0, 1)
    axb.set_title(f"[{a.ds} @{a.fps:.0f}Hz] 转移先验权重 ↑ → milestone 跳变率 ↓(过强=冻结);相似度为主才保留跳变")
    for c_ in ["#9467bd", "#1f77b4", "#ff7f0e", "#888"]:
        pass
    cols = {"emission-only": "#1f77b4", "light β0.4 back1": "#2ca02c", "aux β0.6 back3": "#ff7f0e", "heavy β2 back40": "#888"}
    for name in names:
        if name in ms_long:
            tq, ms, v = ms_long[name]; ax[1].step(tq, ms, where="post", lw=1.4, color=cols[name], label=name, alpha=0.85)
    ax[1].set_title(f"[{a.ds}] 最长 ep{elong} 的 milestone 时间线:重复动作处 emission 主导会跳变,过强先验则冻结成阶梯")
    ax[1].set_xlabel("归一化进度"); ax[1].set_ylabel("milestone idx"); ax[1].legend(fontsize=8, ncol=2)
    fig.tight_layout(); out = OUTV / f"{a.ds}_rebalance.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
    json.dump(stats, open(OUTV / f"{a.ds}_rebalance.json", "w"), indent=2)
    print(f"SAVED {out} ({time.time()-t0:.0f}s)\n{json.dumps(stats, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
