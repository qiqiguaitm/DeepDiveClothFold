"""Cross-dataset generalization validation: run the SAME zero-train CRAVE recipe (no per-dataset
tuning) on a new dataset and measure how well the read-out value tracks task progress.

Metric (GT-free, same as the legacy generalization test): per-episode corr(value, normalized-time)
on successful demos — value should rise monotonically with progress, so high corr ⇒ the auto-
discovered milestone-value transfers. Reports mean/median/p25 corr, %corr≥0.7, monotonicity.

Run: CUDA_VISIBLE_DEVICES=0 PY crave/experiments/cross_dataset_validate.py xvla --encoder dinov3-h
Out: crave/docs/visualization/cross_dataset/<ds>_validation.png (+ <ds>_evalall.json, <ds>_corrs.npy)
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
from crave.utils import L2, mkp_gap
from generalize import build_milestones, make_readout   # clean reference functions

OUTV = REPO / "crave/docs/visualization/cross_dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--encoder", default="dinov3-h")
    a = ap.parse_args()
    t0 = time.time(); cfg = resolve_dataset(a.dataset); enc = load_encoder(a.encoder)
    eps = list_eps(cfg)
    print(f"[{a.dataset}] {len(eps)} eps, encoding ({a.encoder})...", flush=True)
    POOL, STATE, EPID, TPOS, eplen = [], [], [], [], {}
    for k, e in enumerate(eps):
        try:
            f224, state, _th, _ = load_ep(cfg, e, strd=1)
        except Exception as ex:
            print(f"  ep{e} skip ({ex})", flush=True); continue
        if len(f224) < 5: continue
        POOL.append(L2(enc.encode_pooled(f224))); STATE.append(mkp_gap(state, cfg.stride))
        n = len(f224); EPID.append(np.full(n, e)); TPOS.append(np.arange(n) / max(1, n - 1)); eplen[e] = n
        if (k + 1) % 25 == 0: print(f"  {k+1}/{len(eps)} ({time.time()-t0:.0f}s)", flush=True)
    img = np.concatenate(POOL); Pm = np.concatenate(STATE)
    E = np.concatenate(EPID); Tv = np.concatenate(TPOS)
    Pn = L2((Pm - Pm.mean(0)) / (Pm.std(0) + 1e-8))
    F = np.concatenate([img, Pn], 1); ne = len(eps)
    print(f"[{a.dataset}] N={len(F)} clustering...", flush=True)
    cen, lab, order, Pord, M = build_milestones(F, E, Tv, ne)
    eps_sorted = sorted(set(E.tolist()))
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    readout = make_readout(cen[order], sk, Pord)

    rows = []   # (ep, corr, mono, value-curve, time)
    for e in eps_sorted:
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        if len(fi) < 5: continue
        v, _ = readout(F[fi]); tq = Tv[fi]
        c = float(np.corrcoef(v, tq)[0, 1]) if (v.std() > 1e-6 and tq.std() > 1e-6) else np.nan
        rows.append((e, c, float(np.mean(np.diff(v) >= -1e-6)), v, tq))
    corrs = np.array([r[1] for r in rows]); monos = np.array([r[2] for r in rows]); ok = np.isfinite(corrs)
    ev = {"ds": a.dataset, "encoder": a.encoder, "n_eps": int(ok.sum()), "M": int(M),
          "corr_mean": float(corrs[ok].mean()), "corr_median": float(np.median(corrs[ok])),
          "corr_p25": float(np.percentile(corrs[ok], 25)), "frac_corr_ge_0.7": float(np.mean(corrs[ok] >= 0.7)),
          "mono_mean": float(monos.mean())}
    OUTV.mkdir(parents=True, exist_ok=True)
    json.dump(ev, open(OUTV / f"{a.dataset}_evalall.json", "w"), indent=2); np.save(OUTV / f"{a.dataset}_corrs.npy", corrs)
    print(f"[{a.dataset}] {ev}", flush=True)

    # figure: corr histogram + 3 example value curves
    plt = setup_mpl(); fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].hist(corrs[ok], bins=24, color="#1a7f37", alpha=0.8); ax[0].axvline(0.7, color="r", ls="--", lw=1.5, label="阈值 0.7")
    ax[0].set_title(f"[{a.dataset}] 逐 episode corr(value, 进度) 分布\nmean={ev['corr_mean']:.3f} median={ev['corr_median']:.3f} "
                    f"%≥0.7={ev['frac_corr_ge_0.7']:.0%} (n={ev['n_eps']})"); ax[0].set_xlabel("corr(value, normalized-time)"); ax[0].set_ylabel("#episodes"); ax[0].legend(fontsize=9)
    med_i = sorted([r for r in rows if np.isfinite(r[1])], key=lambda r: r[1])
    picks = [med_i[len(med_i)//10], med_i[len(med_i)//2], med_i[-1]]   # p10 / median / best
    for r in picks:
        e, c, _mo, v, tq = r; ax[1].plot(tq, v, lw=1.8, label=f"ep{e}  corr={c:.2f}")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=.4, label="参照 value=进度"); ax[1].set_xlim(0, 1); ax[1].set_ylim(-.02, 1.02)
    ax[1].set_title(f"[{a.dataset}] 示例 value 曲线(贴合进度对角线)"); ax[1].set_xlabel("归一化进度(时间)"); ax[1].set_ylabel("CRAVE value"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle(f"跨数据集验证 — {a.dataset}({ne}ep)— {a.encoder} 零训练 CRAVE, 配方逐字不改 — {M} milestones", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUTV / f"{a.dataset}_validation.png", dpi=130, bbox_inches="tight")
    print(f"SAVED {OUTV / (a.dataset + '_validation.png')} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
