"""Prototype: confidence-gated milestone transition (hold value when no confident cluster match)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import med, smooth_monotone
from transition_prior_fix import build_pen, viterbi_pen, visited_sequence
from milestone_select import build_milestones_uniform


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="xvla"); ap.add_argument("--ep", type=int, default=58)
    ap.add_argument("--native-fps", type=float, default=30.0); ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--tau-lo", type=float, default=60); ap.add_argument("--tau-hi", type=float, default=92)
    ap.add_argument("--gamma-fwd", type=float, default=0.8); ap.add_argument("--margin-fwd", type=float, default=0.35)
    ap.add_argument("--stats-n", type=int, default=60)
    a = ap.parse_args()
    z = np.load(REPO / f"temp/xreb_cache_{a.ds}.npz", allow_pickle=True)
    F = z["F"].astype(np.float32); E, Tv = z["E"], z["Tv"]; eps = sorted(set(E.tolist()))
    cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, len(eps), max_gap=0.12)
    C = cen[order]
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    am = np.linalg.norm(F[:, None] - C[None], axis=2).argmin(1); counts = np.zeros((M, M))
    for e in eps:
        seq = visited_sequence(am[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])]])
        for i, j in zip(seq[:-1], seq[1:]): counts[i, j] += 1
    pen = build_pen(counts, Pord, 1.0, 0.4, 1.0)
    dM = np.linalg.norm(F[:, None] - C[None], axis=2).min(1)
    tau_lo, tau_hi = np.percentile(dM, a.tau_lo), np.percentile(dM, a.tau_hi)
    print(f"[{a.ds}] dM median={np.median(dM):.3f} tau_lo={tau_lo:.3f} tau_hi={tau_hi:.3f} M={M}", flush=True)

    def readout(Fq, gate, hold_thresh=0.5):
        base = np.linalg.norm(Fq[:, None] - C[None], axis=2); nn = len(Fq); tx = np.arange(nn) / nn
        em = base.copy(); dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1)
        em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
        em = em + a.gamma * np.maximum(0.0, tx[:, None] - Pord[None, :] - 0.15)            # value << progress (drop-to-0)
        em = em + a.gamma_fwd * np.maximum(0.0, Pord[None, :] - tx[:, None] - a.margin_fwd)  # value >> progress (confident forward-alias spike)
        ms = viterbi_pen(em, pen); conf = np.ones(nn)
        if gate:   # HARD HOLD: low-confidence frame (weak match to all clusters) inherits previous milestone instead of jumping
            dmin = base.min(1); conf = np.clip((tau_hi - dmin) / (tau_hi - tau_lo + 1e-6), 0.0, 1.0)
            held = ms.copy()
            for t in range(1, nn):
                if conf[t] < hold_thresh: held[t] = held[t - 1]
            ms = held
        return ms, conf

    mw = max(5, int(round(5 * a.native_fps / 3))) | 1
    sub = eps[:a.stats_n]
    for gate in (False, True):
        J, Cr = [], []
        for e in sub:
            fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
            if len(fi) < 5: continue
            ms, _ = readout(F[fi], gate); v = smooth_monotone(med(Pord[ms], mw), fps=a.native_fps); tq = Tv[fi]
            J.append(int(np.sum(np.diff(ms) != 0))); Cr.append(np.corrcoef(v, tq)[0, 1] if v.std() > 1e-6 else np.nan)
        Cr = np.array(Cr); ok = np.isfinite(Cr)
        print(f"[{a.ds} {a.stats_n}ep] {'GATE' if gate else 'raw '}: jumps/ep={np.mean(J):.0f} corr={Cr[ok].mean():.3f} %>=0.7={np.mean(Cr[ok]>=0.7):.0%}", flush=True)

    fi = np.where(E == a.ep)[0]; fi = fi[np.argsort(Tv[fi])]; tq = Tv[fi]
    plt = setup_mpl(); fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    for gate, col in [(False, "#d62728"), (True, "#1a7f37")]:
        ms, conf = readout(F[fi], gate); v = smooth_monotone(med(Pord[ms], mw), fps=a.native_fps)
        nj = int(np.sum(np.diff(ms) != 0)); c = np.corrcoef(v, tq)[0, 1]
        ax[0].plot(tq, v, lw=1.7, color=col, label=f"{'gated' if gate else 'raw'}: 跳变={nj} corr={c:.2f}")
        if gate:
            ax[1].plot(tq, conf, lw=1.2, color="#1f77b4"); ax[1].fill_between(tq, 0, 1, where=conf < 0.5, color="#d62728", alpha=.15)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=.4); ax[0].set_ylim(-.02, 1.02); ax[0].legend(fontsize=9)
    ax[0].set_title(f"[{a.ds}] ep{a.ep} value:置信度门控抑制稀有场景误跳"); ax[0].set_xlabel("进度"); ax[0].set_ylabel("value"); ax[0].grid(alpha=.3)
    ax[1].set_title(f"ep{a.ep} 相似度置信度(红=低置信→保持)"); ax[1].set_xlabel("进度"); ax[1].set_ylim(-.02, 1.02); ax[1].grid(alpha=.3)
    fig.tight_layout(); out = REPO / f"crave/docs/visualization/cross_dataset/{a.ds}_confgate_ep{a.ep}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
