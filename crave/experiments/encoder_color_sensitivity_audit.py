"""Is DINOv3-H's color-splitting good or bad for CRAVE value — on kai0_base (cloth fold)?

Uses the frame-aligned FULL pooled features already on disk (H+ in temp/crave_full_dinov3h,
DINOv2-large in temp/crave_full). For each encoder, frame-for-frame:
  1. cluster fragmentation: per-cluster coverage + #clusters per progress bin
  2. per-episode value quality (build_milestones + forward-Viterbi readout): max value reached
     + monotonicity — a worse encoder for this task leaves more episodes whose value undershoots
  3. color attribution: on ~6k sampled frames, is H+ clustering more hue-pure than DINOv2?
     (lower within-cluster hue dispersion ⇒ splitting by garment color)
  4. does value-undershoot correlate with garment-hue rarity? (the rare-color tail risk)

Run: CUDA_VISIBLE_DEVICES=0 /home/tim/miniconda3/envs/srpo/bin/python \
       crave/experiments/encoder_color_sensitivity_audit.py
Out:  crave/docs/visualization/encoders/enc_color_sensitivity_audit.png  (+ ..._audit.json)
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crave.config import REPO, resolve_dataset, viz_dir          # noqa: E402
from crave.data import kai0                                       # noqa: E402
from crave.utils import L2, otsu                                  # noqa: E402
from generalize import build_milestones, make_readout            # noqa: E402

cfg = resolve_dataset("kai0_base")
# tag -> (shard_dir, dim, index_filename). DINOv2-large uses the legacy crave_full layout
# (index_dino.npz + dino/ shard subdir); DINOv3-* use crave_full_7b_centroid layout (index.npz, flat).
SRC = {
    "dinov2-large":   (REPO / "temp/crave_full/dino", 1024, REPO / "temp/crave_full/index_dino.npz"),
    "dinov3-h":       (REPO / "temp/crave_full_dinov3h", 1280, REPO / "temp/crave_full_dinov3h/index.npz"),
    "dinov3-7b-int8": (REPO / "temp/crave_full_dinov37bint8", 4096, REPO / "temp/crave_full_dinov37bint8/index.npz"),
}


def load_full(tag):
    shard_dir, dim, idxp = SRC[tag]
    idx = np.load(idxp)
    E, FR, T, N = idx["E"], idx["FR"], idx["T"], int(idx["n"])
    feat = np.zeros((N, dim), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(shard_dir / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    return E, FR, T, feat, valid


def cluster_value(tag, E, T, F, valid):
    """build milestones + per-episode value on valid frames. Returns dict of stats + labels."""
    vi = np.where(valid)[0]
    Fv = L2(F[vi].astype(np.float32)); Ev, Tv = E[vi], T[vi]
    ne = len(set(Ev.tolist()))
    cen, lab, order, Pord, M = build_milestones(Fv, Ev, Tv, ne)
    K0 = len(cen)
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    C = cen[order]
    from sklearn.cluster import KMeans
    sk = KMeans(8, n_init=2, random_state=0).fit(Fv[Tv < 0.1]).cluster_centers_
    readout = make_readout(C, sk, Pord)
    # per-episode value
    maxv, mono = [], []
    eps = sorted(set(Ev.tolist()))
    for e in eps:
        fq = Fv[Ev == e]                       # already temporal (ascending global idx = ascending FR)
        if len(fq) < 4:
            continue
        v, _ms = readout(fq, fps=3.0)
        maxv.append(float(v.max())); mono.append(float(np.mean(np.diff(v) >= -1e-6)))
    return {"tag": tag, "M": int(M), "K0": int(K0),
            "cov": cov, "tpos": tpos, "lab_full": lab, "vi": vi,
            "maxv": np.array(maxv), "mono": np.array(mono),
            "tau_cov": float(otsu(cov))}


def cloth_hue(rgb):
    """Median hue (deg) of non-white, non-skin pixels in the 224 crop — a coarse garment-color proxy."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(float) * 2, hsv[..., 1], hsv[..., 2]
    mask = (s > 50) & (v > 40) & (v < 245)        # drop white table / dark / low-sat
    return float(np.median(h[mask])) if mask.sum() > 200 else np.nan


def hue_dispersion(hues, labels):
    """Mean within-cluster circular std of hue (deg). Lower ⇒ clusters are more color-pure."""
    out = []
    for c in np.unique(labels):
        hh = hues[labels == c]; hh = hh[np.isfinite(hh)]
        if len(hh) >= 8:
            a = np.deg2rad(hh); R = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
            out.append(np.rad2deg(np.sqrt(-2 * np.log(max(R, 1e-6)))))
    return float(np.mean(out)) if out else np.nan


def main():
    t0 = time.time()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs=2, default=["dinov2-large", "dinov3-h"],
                    help="two encoder tags to compare (baseline first)")
    a = ap.parse_args()
    A, B = a.encoders
    res = {}
    Eref = None
    for tag in [A, B]:
        E, FR, T, F, valid = load_full(tag)
        Eref = (E, FR, T) if Eref is None else Eref
        print(f"[{tag}] loaded {valid.sum()}/{len(E)} valid; clustering + per-ep value ...", flush=True)
        res[tag] = cluster_value(tag, E, T, F, valid)
        res[tag]["_FR"] = FR
        print(f"[{tag}] M={res[tag]['M']} K0={res[tag]['K0']} "
              f"med-maxv={np.median(res[tag]['maxv']):.3f} "
              f"frac maxv<0.8={np.mean(res[tag]['maxv']<0.8):.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- color attribution: sample ~6k frames, grab raw, compute hue ----
    E, FR, T = Eref
    common = np.intersect1d(res[A]["vi"], res[B]["vi"])
    rng = np.random.RandomState(0); samp = np.sort(rng.choice(common, min(6000, len(common)), replace=False))
    print(f"[color] grabbing {len(samp)} raw frames for hue ...", flush=True)
    imgs, ok = kai0.decode_images(cfg, samp, E, FR)
    hues = np.array([cloth_hue(imgs[i]) if ok[i] else np.nan for i in range(len(samp))])
    # map sampled global idx -> each encoder's full-label (lab_full is indexed over vi order)
    def labels_for(tag):
        vi = res[tag]["vi"]; pos = {g: k for k, g in enumerate(vi)}
        return np.array([res[tag]["lab_full"][pos[g]] for g in samp])
    disp = {tag: hue_dispersion(hues, labels_for(tag)) for tag in (A, B)}
    print(f"[color] within-cluster hue dispersion (deg): {disp}", flush=True)

    # ---- figure ----
    from crave.render import setup_mpl
    plt = setup_mpl()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    col = {A: "#1f77b4", B: "#d62728"}
    # (1) coverage distribution
    for tag in res:
        ax[0].hist(res[tag]["cov"], bins=30, alpha=0.55, label=f"{tag} (M={res[tag]['M']})", color=col[tag])
        ax[0].axvline(res[tag]["tau_cov"], color=col[tag], ls="--", lw=1)
    ax[0].set_title("簇覆盖率分布 (虚线=Otsu τ_cov)\n左移=更多低覆盖簇=更碎"); ax[0].set_xlabel("coverage"); ax[0].legend(fontsize=8)
    # (2) clusters per progress bin
    bins = np.linspace(0, 1, 21)
    for tag in res:
        cnt, _ = np.histogram(res[tag]["tpos"], bins=bins)
        ax[1].plot((bins[:-1] + bins[1:]) / 2, cnt, "-o", ms=3, label=tag, color=col[tag])
    ax[1].set_title("每进度段的簇数\n高=同相位被拆得多"); ax[1].set_xlabel("progress (tpos)"); ax[1].set_ylabel("#clusters"); ax[1].legend(fontsize=8)
    # (3) per-episode max value CDF
    for tag in res:
        mv = np.sort(res[tag]["maxv"]); ax[2].plot(mv, np.linspace(0, 1, len(mv)), label=f"{tag} med={np.median(mv):.3f}", color=col[tag])
    ax[2].axvline(0.8, color="0.5", ls=":"); ax[2].set_title("逐 episode 最大 value 的 CDF\n左尾=value 没到顶(相位没匹配上)"); ax[2].set_xlabel("max value per episode"); ax[2].set_ylabel("CDF"); ax[2].legend(fontsize=8)
    fig.suptitle(f"颜色敏感性核对 — kai0_base 折叠任务 @3Hz ({valid.sum()}fr) — {B} vs {A}  |  "
                 f"hue簇内离散: {B}={disp[B]:.1f}° vs {A}={disp[A]:.1f}° (越低=越按颜色分簇)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    tagB = B.replace("-", "")
    out = viz_dir("encoders") / f"enc_color_audit_{tagB}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")

    summary = {tag: {"M": res[tag]["M"], "K0": res[tag]["K0"], "tau_cov": res[tag]["tau_cov"],
                     "frac_clusters_below_tau": float(np.mean(res[tag]["cov"] < res[tag]["tau_cov"])),
                     "median_cov": float(np.median(res[tag]["cov"])),
                     "median_maxv": float(np.median(res[tag]["maxv"])),
                     "frac_ep_maxv_below_0.8": float(np.mean(res[tag]["maxv"] < 0.8)),
                     "median_monotonicity": float(np.median(res[tag]["mono"])),
                     "hue_dispersion_deg": disp[tag]} for tag in res}
    json.dump(summary, open(viz_dir("encoders") / f"enc_color_audit_{tagB}.json", "w"), indent=2)
    print(f"SAVED {out}\n{json.dumps(summary, indent=2)}\n({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
