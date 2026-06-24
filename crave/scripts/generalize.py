"""Generalization eval: full-frame cluster → precedence/isotonic milestones →
forward-biased Viterbi value, on VIS / XVLA / coffee, with any encoder.

This is the reference *thin entrypoint*: every heavy operation is a `crave` library
call; only orchestration + plotting live here.

    python -m crave.scripts.generalize coffee --encoder dinov3-h
    CRAVE_ENC=dinov3-h python crave/scripts/generalize.py vis --novideo
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from crave.clustering import gpu_kmeans
from crave.config import out_dir, resolve_dataset
from crave.data import list_eps, load_ep, load_ep_native
from crave.decoding import train_dec
from crave.encoders import load_encoder
from crave.render import setup_mpl
from crave.utils import L2, med, mkp_gap, otsu, smooth_monotone, viterbi_forward


def build_milestones(F, E, Tv, ne):
    """Full-frame cluster → coverage/purity/spacing select → precedence+isotonic order."""
    from sklearn.isotonic import IsotonicRegression
    N = len(F)
    K0 = int(np.clip(round(0.55 * np.sqrt(N)), 64, 320))
    cen, lab = gpu_kmeans(F, K0)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(E[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov, tau_pur = otsu(cov), float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel); eps_sorted = sorted(set(E.tolist()))
    fe = np.full((len(eps_sorted), M), np.nan)
    for ei, e in enumerate(eps_sorted):
        fi = np.where(E == e)[0]; labe, te = lab[fi], Tv[fi]
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
    Pord = np.asarray(IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec]), float)
    order = [sel[p] for p in prec]
    return cen, lab, np.array(order), Pord, M


def make_readout(C, sk, Pord):
    """Forward-biased Viterbi readout over milestone centers (start-anchored, hard-start)."""
    def readout(Fq, fps=3.0):
        nn = len(Fq)
        emit = np.linalg.norm(Fq[:, None] - C[None], axis=2)
        dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
        emit[:, 0] = np.minimum(emit[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
        ms = viterbi_forward(emit, Pord, up=3.0, down=25.0, hard_start=True)
        mw = max(5, int(round(5 * fps / 3))) | 1
        return smooth_monotone(med(Pord[ms], mw), fps=fps), ms
    return readout


def main(ds_name: str, encoder: str | None = None, novideo: bool = False):
    t0 = time.time()
    cfg = resolve_dataset(ds_name)
    OUT = out_dir(f"crave_generalize_{(encoder or 'dinov2-large')}" if encoder else "crave_generalize") / ds_name
    OUT.mkdir(parents=True, exist_ok=True)
    enc = load_encoder(encoder)
    eps = list_eps(cfg)
    print(f"[{ds_name}] {len(eps)} eps, full-frame features ({enc.spec.name})...", flush=True)

    POOL, STATE, EPID, TPOS, THUMB, eplen = [], [], [], [], [], {}
    for k, e in enumerate(eps):
        try:
            f224, state, th, _ = load_ep(cfg, e, strd=1)
        except Exception as ex:
            print(f"  ep{e} skip ({ex})", flush=True); continue
        if len(f224) < 5: continue
        pooled = L2(enc.encode_pooled(f224)); n = len(f224)
        POOL.append(pooled); STATE.append(mkp_gap(state, cfg.stride))
        EPID.append(np.full(n, e)); TPOS.append(np.arange(n) / max(1, n - 1)); THUMB += th; eplen[e] = n
        if (k + 1) % 25 == 0: print(f"  {k+1}/{len(eps)} ({time.time()-t0:.0f}s)", flush=True)

    img = np.concatenate(POOL); Pm = np.concatenate(STATE)
    E = np.concatenate(EPID); Tv = np.concatenate(TPOS); THUMB = np.stack(THUMB)
    PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
    Pn = L2((Pm - PMU) / PSD)
    F = np.concatenate([img, Pn], 1); ne = len(eps)
    print(f"[{ds_name}] N={len(F)} clustering...", flush=True)
    cen, lab, order, Pord, M = build_milestones(F, E, Tv, ne)
    C = cen[order]
    print(f"[{ds_name}] {M} milestones (precedence+isotonic)", flush=True)

    from sklearn.cluster import KMeans
    eps_sorted = sorted(set(E.tolist()))
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    EPp = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][-2:]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    ek = KMeans(8, n_init=2, random_state=0).fit(EPp).cluster_centers_
    de_tr = np.array([float(np.linalg.norm(F[np.where(E == e)[0][np.argmax(Tv[np.where(E == e)[0]])]][None] - ek, axis=1).min()) for e in eps_sorted])
    de_thr = float(np.quantile(de_tr, 0.9)) * 1.3
    readout = make_readout(C, sk, Pord)

    # centroid decode + gallery
    print(f"[{ds_name}] centroid decode...", flush=True)
    NS = 20; samp = []; rng = []
    for c in order:
        loc = np.where(lab == c)[0]
        if len(loc) > NS: loc = loc[np.linspace(0, len(loc) - 1, NS).astype(int)]
        s0 = len(samp)
        for gi in loc: samp.append(cv2.resize(THUMB[gi], (224, 224)))
        rng.append((s0, len(samp)))
    samp = np.stack(samp); grids = enc.encode_grid(samp)
    imgs128 = np.stack([cv2.resize(s, (128, 128)) for s in samp])
    decf = train_dec(grids, imgs128, enc.dim, "small", 55)
    proto, medoid = {}, {}
    for mi, c in enumerate(order):
        s0, e0 = rng[mi]
        proto[mi] = cv2.resize(decf(grids[s0:e0].mean(0)[None])[0], (96, 96)) if e0 > s0 else np.zeros((96, 96, 3), np.uint8)
        loc = np.where(lab == c)[0]; gi = loc[int(np.argmin(np.linalg.norm(F[loc] - cen[c], axis=1)))]
        medoid[mi] = cv2.resize(THUMB[gi], (96, 96))

    plt = setup_mpl()
    PR = 18; NBd = (M + PR - 1) // PR
    fig, axes = plt.subplots(2 * NBd, PR, figsize=(PR * 0.82, 2 * NBd * 1.18)); axes = np.atleast_2d(axes)
    for ax in axes.ravel(): ax.axis("off")
    for mi in range(M):
        b, col = mi // PR, mi % PR
        axes[2 * b, col].imshow(proto[mi]); axes[2 * b, col].set_title(f"m{mi} v={Pord[mi]:.2f}", fontsize=6)
        axes[2 * b + 1, col].imshow(medoid[mi])
    fig.suptitle(f"[{ds_name}] {M} milestones — top: decoded centroid | bottom: medoid", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"{ds_name}_centroid_gallery.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # 2 long episodes: native-rate forward value
    long2 = sorted(eplen, key=lambda e: -eplen[e])[:2]
    print(f"[{ds_name}] long eps {long2}; native-rate readout...", flush=True)
    vlast = np.array([0.0, 1.0])
    for e in long2:
        f224, staten, fpsd = load_ep_native(cfg, e)
        imgn = L2(enc.encode_pooled(f224))
        Pg = L2((mkp_gap(staten, cfg.stride) - PMU) / PSD)
        Fq = np.concatenate([imgn, Pg], 1); nn = len(Fq)
        v, ms = readout(Fq, fps=fpsd); vlast = v
        de_end = float(np.linalg.norm(Fq[-3:][:, None] - ek[None], axis=2).min()); comp = de_end <= de_thr
        print(f"  ep{e}: {nn}f@{fpsd}Hz value {v.min():.2f}->{v.max():.2f}", flush=True)
        fig = plt.figure(figsize=(13, 5.5)); gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.4])
        ax0 = fig.add_subplot(gs[0]); ax0.plot(v, color="#1a7f37", lw=2)
        for p in Pord: ax0.axhline(p, color="0.93", lw=.5)
        ax0.set_ylim(-.02, 1.02); ax0.set_ylabel("forward value")
        ax0.set_title(f"[{ds_name}] ep{e} forward value ({nn}@{fpsd}Hz) {M}ms — {'COMPLETE' if comp else 'INCOMPLETE'} (resid {de_end:.2f}/thr {de_thr:.2f})",
                      color=("#1a7f37" if comp else "#d62728"))
        ax1 = fig.add_subplot(gs[1]); ax1.step(range(nn), ms, where="post", color="#9c27b0")
        ax1.set_ylabel("milestone"); ax1.set_xlabel(f"frame({fpsd}Hz)")
        fig.tight_layout(); fig.savefig(OUT / f"{ds_name}_ep{e}_value.png", dpi=120, bbox_inches="tight"); plt.close(fig)
        if novideo:
            print(f"  SAVED {ds_name}_ep{e}_value.png (novideo)", flush=True)

    json.dump({"ds": ds_name, "encoder": enc.spec.name, "n_eps": len(eps), "N": int(len(F)), "M": M,
               "long2": long2, "value_range": [float(vlast.min()), float(vlast.max())]},
              open(OUT / f"{ds_name}_summary.json", "w"), indent=2)
    print(f"[{ds_name}] DONE {time.time()-t0:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["vis", "xvla", "coffee"])
    ap.add_argument("--encoder", default=None, help="encoder name (default: $CRAVE_ENC or dinov2-large)")
    ap.add_argument("--novideo", action="store_true")
    a = ap.parse_args()
    main(a.dataset, a.encoder, a.novideo)
