"""Regenerate cross-dataset validation figures with: light transition prior (emission-primary)
+ a gentle ONE-SIDED progress prior on emission that kills the "late frame → value-0 start
milestone" drop WITHOUT a transition barrier (so XVLA same-progress repeats still jump).

progress prior: emit[t,m] += gamma * relu(tx[t] - Pord[m] - margin)
  → only penalizes a milestone whose value is far BELOW the elapsed-time fraction (the drop
    direction); value-ahead-of-time (fast progress) is never penalized.

Uses the cached encode (temp/xreb_cache_<ds>.npz) → fast, no re-encode.
Run: PY crave/experiments/regen_validation.py --ds coffee [--gamma 2.0 --margin 0.15]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crave.config import REPO
from crave.render import setup_mpl
from crave.utils import L2, med, smooth_monotone
from transition_prior_fix import build_pen, viterbi_pen, visited_sequence
from milestone_select import build_milestones_uniform
OUTV = REPO / "crave/docs/visualization/cross_dataset"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="coffee")
    ap.add_argument("--gamma", type=float, default=0.8); ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--native-fps", type=float, default=30.0)   # xvla 30Hz, coffee 50Hz — rate-scaled value smoothing
    ap.add_argument("--max-gap", type=float, default=0.0)   # >0 → re-cluster with uniformity-aware gap-fill (caps milestone value gaps)
    a = ap.parse_args()
    z = np.load(REPO / f"temp/xreb_cache_{a.ds}.npz", allow_pickle=True)
    F = z["F"].astype(np.float32); E, Tv = z["E"], z["Tv"]
    eps_sorted = sorted(set(E.tolist()))
    if a.max_gap > 0:   # re-cluster with uniform selection, rebuild transition counts
        cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, len(eps_sorted), max_gap=a.max_gap)
        C = cen[order]; am = np.linalg.norm(F[:, None] - C[None], axis=2).argmin(1); counts = np.zeros((M, M))
        for e in eps_sorted:
            seq = visited_sequence(am[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])]])
            for i, j in zip(seq[:-1], seq[1:]): counts[i, j] += 1
        print(f"[{a.ds}] uniform re-cluster: M={M} (+{nf} gap-filled)", flush=True)
    else:
        Pord, order, cen, counts = z["Pord"], z["order"], z["cen"], z["counts"]; C = cen[order]; M = len(order)
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    pen = build_pen(counts, Pord, 1.0, 0.4, 1.0)   # light transition prior (emission-primary)

    def readout(Fq):
        nn = len(Fq); emit = np.linalg.norm(Fq[:, None] - C[None], axis=2)
        dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
        emit[:, 0] = np.minimum(emit[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
        emit = emit + a.gamma * np.maximum(0.0, tx[:, None] - Pord[None, :] - a.margin)   # progress prior
        return viterbi_pen(emit, pen)

    mw = max(5, int(round(5 * a.native_fps / 3))) | 1   # rate-scaled smoothing window (ms keeps jumps; v smoothed)
    rows = []; ndrop = 0; njump = []
    for e in eps_sorted:
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        if len(fi) < 5: continue
        ms = readout(F[fi]); njump.append(int(np.sum(np.diff(ms) != 0)))
        v = smooth_monotone(med(Pord[ms], mw), fps=a.native_fps); tq = Tv[fi]
        c = float(np.corrcoef(v, tq)[0, 1]) if v.std() > 1e-6 else np.nan
        drop = bool(np.any((tq > 0.4) & (v < 0.05)))   # value collapses to ~0 after 40% progress
        ndrop += int(drop); rows.append((e, c, v, tq, drop))
    corrs = np.array([r[1] for r in rows]); ok = np.isfinite(corrs)
    ev = {"ds": a.ds, "readout": f"light+progress-prior(γ{a.gamma},m{a.margin})", "n_eps": int(ok.sum()), "M": int(M),
          "corr_mean": float(corrs[ok].mean()), "corr_median": float(np.median(corrs[ok])),
          "frac_corr_ge_0.7": float(np.mean(corrs[ok] >= 0.7)), "n_drop_to_0": int(ndrop),
          "jumps_per_ep": float(np.mean(njump)), "native_fps": a.native_fps}
    json.dump(ev, open(OUTV / f"{a.ds}_evalall.json", "w"), indent=2); np.save(OUTV / f"{a.ds}_corrs.npy", corrs)
    print(f"[{a.ds}] {ev}", flush=True)

    plt = setup_mpl(); fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].hist(corrs[ok], bins=24, color="#1a7f37", alpha=0.8); ax[0].axvline(0.7, color="r", ls="--", lw=1.5, label="阈值 0.7")
    ax[0].set_title(f"[{a.ds} @{a.native_fps:.0f}Hz] 逐 episode corr(value,进度) — 相似度为主+轻转移先验+进度先验\nmean={ev['corr_mean']:.3f} median={ev['corr_median']:.3f} %≥0.7={ev['frac_corr_ge_0.7']:.0%} · 跌零ep={ndrop} · 跳变/ep={ev['jumps_per_ep']:.0f}(结构保留) (n={ev['n_eps']})")
    ax[0].set_xlabel("corr(value, normalized-time)"); ax[0].set_ylabel("#episodes"); ax[0].legend(fontsize=9)
    mi = sorted([r for r in rows if np.isfinite(r[1])], key=lambda r: r[1])
    for r in [mi[len(mi)//10], mi[len(mi)//2], mi[-1]]:
        ax[1].plot(r[3], r[2], lw=1.8, label=f"ep{r[0]} corr={r[1]:.2f}")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=.4, label="参照 value=进度"); ax[1].set_xlim(0, 1); ax[1].set_ylim(-.02, 1.02)
    ax[1].set_title(f"[{a.ds}] 示例 value 曲线(进度先验消除跌零)"); ax[1].set_xlabel("归一化进度"); ax[1].set_ylabel("CRAVE value"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle(f"跨数据集验证 — {a.ds} — DINOv3-H 零训练, 相似度为主 + 轻先验 + 进度先验 — {M} milestones", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUTV / f"{a.ds}_validation.png", dpi=130, bbox_inches="tight")
    print(f"SAVED {OUTV / (a.ds + '_validation.png')}", flush=True)


if __name__ == "__main__":
    main()
