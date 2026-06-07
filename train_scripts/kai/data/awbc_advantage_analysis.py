"""AWBC Stage-2.5 analysis: inspect the Advantage Estimator's labels over the whole
labeled dataset BEFORE discretizing (Stage 3). Answers two questions the user wants:

  1) Direction / health: per-episode corr(absolute_value, frame_progress). The estimator's
     value should INCREASE with task progress (pi0-AE value = increasing-progress). High
     positive corr = healthy episode; low/negative = suspect.
  2) Discretize threshold: distribution of `relative_advantage` (value(t+K)-value(t)) — the
     signal Stage-3 binarizes into positive/negative. Show percentiles + what positive-fraction
     each candidate threshold yields, so the threshold can be chosen deliberately.

Reads the labeled parquets written by eval.py (cols: absolute_value, relative_advantage,
absolute_advantage). Outputs a text summary + histograms PNG. No GPU / model needed.

Usage: kai0/.venv/bin/python train_scripts/kai/data/awbc_advantage_analysis.py \
         [--labeled <dir>] [--out <prefix>]
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd

DEF_LABELED = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_smooth800_dagger_all/data_KAI0_100000/chunk-000"


def pct(a, ps):
    return {p: float(np.percentile(a, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=DEF_LABELED, help="dir of labeled episode_*.parquet")
    ap.add_argument("--out", default="/vePFS/tim/workspace/deepdive_kai0/logs/awbc_advantage_analysis")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.labeled, "episode_*.parquet")))
    if not files:
        raise SystemExit(f"no labeled parquets under {a.labeled}")
    print(f"[load] {len(files)} labeled episodes from {a.labeled}", flush=True)

    all_av, all_ra, all_aa = [], [], []     # absolute_value / relative_advantage / absolute_advantage
    ep_corr = []                            # (ep_idx, n, corr(av, progress))
    for fp in files:
        ep = int(Path(fp).stem.split("_")[1])
        t = pd.read_parquet(fp, columns=["absolute_value", "relative_advantage", "absolute_advantage"])
        av = t["absolute_value"].to_numpy(dtype=np.float64)
        ra = t["relative_advantage"].to_numpy(dtype=np.float64)
        aa = t["absolute_advantage"].to_numpy(dtype=np.float64)
        all_av.append(av); all_ra.append(ra); all_aa.append(aa)
        n = len(av)
        if n >= 3 and np.std(av) > 1e-9:
            prog = np.arange(n) / (n - 1)
            ep_corr.append((ep, n, float(np.corrcoef(av, prog)[0, 1])))
        else:
            ep_corr.append((ep, n, float("nan")))

    av = np.concatenate(all_av); ra = np.concatenate(all_ra); aa = np.concatenate(all_aa)
    corrs = np.array([c for _, _, c in ep_corr], dtype=np.float64)
    corrs_v = corrs[~np.isnan(corrs)]
    ps = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

    lines = []
    def P(s=""): lines.append(s); print(s, flush=True)

    P(f"===== AWBC advantage analysis ({len(files)} eps, {len(av):,} frames) =====")
    P("\n--- 1) per-episode corr(absolute_value, frame_progress)  [want >0: value tracks progress] ---")
    P(f"  episodes with valid corr: {len(corrs_v)}/{len(ep_corr)}")
    P(f"  mean={np.mean(corrs_v):+.3f}  median={np.median(corrs_v):+.3f}  std={np.std(corrs_v):.3f}")
    P(f"  corr percentiles: " + "  ".join(f"p{p}={np.percentile(corrs_v,p):+.3f}" for p in [1,5,10,25,50,75,90]))
    for thr in (0.0, 0.3, 0.5, 0.7):
        frac = float(np.mean(corrs_v > thr))
        P(f"  frac eps corr> {thr:+.1f}: {frac:.3f}  ({int(frac*len(corrs_v))} eps)")
    worst = sorted([t for t in ep_corr if not np.isnan(t[2])], key=lambda x: x[2])[:12]
    P("  12 lowest-corr episodes (suspect / candidate to drop): " +
      ", ".join(f"ep{e}({c:+.2f},n={n})" for e, n, c in worst))

    P("\n--- 2) relative_advantage distribution  [Stage-3 binarizes this → positive/negative] ---")
    P(f"  mean={np.mean(ra):+.5f}  median={np.median(ra):+.5f}  std={np.std(ra):.5f}")
    P("  percentiles: " + "  ".join(f"p{p}={ra_p:+.4f}" for p, ra_p in pct(ra, ps).items()))
    P(f"  frac frames ra>0 (natural sign split): {float(np.mean(ra>0)):.3f}")
    P("  positive-fraction at candidate thresholds (top-X% = positive):")
    for topx in (10, 20, 30, 40, 50):
        thr = float(np.percentile(ra, 100 - topx))
        P(f"    top {topx:>2}%  → ra threshold {thr:+.4f}")

    P("\n--- 3) absolute_value + absolute_advantage (context) ---")
    P("  absolute_value percentiles: " + "  ".join(f"p{p}={v:+.3f}" for p, v in pct(av, ps).items()))
    P("  absolute_advantage: " + "  ".join(f"p{p}={v:+.4f}" for p, v in pct(aa, ps).items()))

    Path(a.out + ".txt").write_text("\n".join(lines))
    json.dump({"n_eps": len(files), "n_frames": int(len(av)),
               "corr_median": float(np.median(corrs_v)), "corr_mean": float(np.mean(corrs_v)),
               "ra_median": float(np.median(ra)), "ra_pcts": pct(ra, ps),
               "worst_eps": [{"ep": e, "n": n, "corr": c} for e, n, c in worst]},
              open(a.out + ".json", "w"), indent=2)

    # histograms
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].hist(corrs_v, bins=40, color="tab:blue"); ax[0].axvline(0, color="k", lw=.8)
        ax[0].set_title("per-ep corr(value, progress)"); ax[0].set_xlabel("corr")
        ax[1].hist(np.clip(ra, np.percentile(ra,0.5), np.percentile(ra,99.5)), bins=60, color="tab:green")
        ax[1].axvline(0, color="k", lw=.8); ax[1].set_title("relative_advantage (clipped 0.5-99.5%)")
        ax[2].hist(np.clip(av, np.percentile(av,0.5), np.percentile(av,99.5)), bins=60, color="tab:orange")
        ax[2].set_title("absolute_value (clipped)")
        fig.tight_layout(); fig.savefig(a.out + ".png", dpi=120)
        P(f"\n[out] {a.out}.png / .txt / .json")
    except Exception as e:
        P(f"[warn] plot skipped: {e}")


if __name__ == "__main__":
    main()
