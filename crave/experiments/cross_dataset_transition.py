"""Transition-prior cross-dataset: rebuild milestones + empirical transition prior on a dataset,
then (A) regenerate the validation figure with the transition-prior readout, and (B) render a
long-episode video (real frame | value curve w/ cursor | aligned milestone centroid-decode).

Run: CUDA_VISIBLE_DEVICES=0 PY crave/experiments/cross_dataset_transition.py --ds coffee
Out: visualization/cross_dataset/<ds>_validation.png (overwrite, now transition-prior)
     visualization/cross_dataset/<ds>_transition_video.mp4
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crave.config import REPO, resolve_dataset
from crave.data import list_eps, load_ep, load_ep_native
from crave.decoding import train_dec
from crave.encoders import load_encoder
from crave.render import setup_mpl
from crave.utils import L2, med, mkp_gap, smooth_monotone
from generalize import build_milestones
from transition_prior_fix import build_pen, viterbi_pen, visited_sequence
from milestone_select import build_milestones_uniform
OUTV = REPO / "crave/docs/visualization/cross_dataset"


def emit_of(Fq, C, sk, Pord=None, gamma=0.8, margin=0.15, gamma_fwd=1.3, margin_fwd=0.25):
    em = np.linalg.norm(Fq[:, None] - C[None], axis=2); nn = len(Fq)
    dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
    em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))
    if Pord is not None and gamma > 0:        # value << progress: penalize late frame matching low-value milestone (cyclic drop-to-0)
        em = em + gamma * np.maximum(0.0, tx[:, None] - Pord[None, :] - margin)
    if Pord is not None and gamma_fwd > 0:    # value >> progress: penalize far-ahead alias (rare scene matching end-state cluster). real repeats go backward, so safe
        em = em + gamma_fwd * np.maximum(0.0, Pord[None, :] - tx[:, None] - margin_fwd)
    return em


def conf_hold(ms, base, tau_lo, tau_hi, thresh=0.5):
    """HARD HOLD: a low-confidence frame (weak match to ALL clusters → dmin near outlier) inherits the
    previous milestone instead of jumping to a wrong cluster. Confident frames (real repeats) untouched."""
    if tau_lo is None: return ms
    dmin = base.min(1); conf = np.clip((tau_hi - dmin) / (tau_hi - tau_lo + 1e-6), 0.0, 1.0)
    held = ms.copy()
    for t in range(1, len(ms)):
        if conf[t] < thresh: held[t] = held[t - 1]
    return held


def _smooth_block(B, k):
    """temporal moving-average over k frames within one episode (edge-padded; preserves length)."""
    if k <= 1 or len(B) <= 1: return B
    pad = np.pad(B, ((k - 1 - (k - 1) // 2, (k - 1) // 2), (0, 0)), mode="edge")
    cs = np.concatenate([np.zeros((1, B.shape[1]), B.dtype), np.cumsum(pad, 0)])
    return ((cs[k:] - cs[:-k]) / k).astype(B.dtype)


def temporal_smooth(F, E, k):
    """smooth features per-episode (cut noise-driven cluster flips; XVLA emission is highly ambiguous)."""
    if k <= 1: return F
    out = F.copy()
    for e in np.unique(E):
        idx = np.where(E == e)[0]; out[idx] = _smooth_block(F[idx], k)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="coffee"); ap.add_argument("--encoder", default="dinov3-h")
    ap.add_argument("--back-barrier", type=float, default=1.0); ap.add_argument("--beta", type=float, default=0.4); ap.add_argument("--lam-geo", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.8); ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--gamma-fwd", type=float, default=1.3); ap.add_argument("--margin-fwd", type=float, default=0.25)   # far-ahead cap (value>>progress alias)
    ap.add_argument("--conf-lo", type=float, default=50.0); ap.add_argument("--conf-hi", type=float, default=82.0); ap.add_argument("--hold-thresh", type=float, default=0.5)   # confidence hold percentiles (conf-hi<=0 disables)
    ap.add_argument("--video-ep", type=int, default=-1); ap.add_argument("--video-corr", type=float, default=0.9)   # video episode: forced id, else longest with corr>=video_corr
    ap.add_argument("--native-fps", type=float, default=0.0)   # 0 = auto (coffee 50Hz, xvla 30Hz) — cluster AND value-output at native rate
    ap.add_argument("--proprio-weight", type=float, default=0.2)   # weight on the 28d proprio block in F=[vis ⊕ w·prop]; 1.0=equal(proprio dominates XVLA), 0.2=vision-primary
    ap.add_argument("--feat-smooth", type=int, default=9)   # temporal moving-average window (frames) on features → cut noise-driven cluster flips (XVLA value 突变)
    ap.add_argument("--max-gap", type=float, default=0.12)   # uniformity-aware milestone selection: cap value gaps (0 = stock build_milestones)
    ap.add_argument("--decode-workers", type=int, default=24)   # parallel video-decode threads (56 cores avail)
    ap.add_argument("--enc-bs", type=int, default=448)   # DINOv3 encode batch (128 default → 4GB/80GB; 448 ≈ 14GB)
    ap.add_argument("--multi-gpu", action="store_true")   # split encode across cuda:0+cuda:1 (launch with CUDA_VISIBLE_DEVICES=0,1)
    a = ap.parse_args(); t0 = time.time(); cfg = resolve_dataset(a.ds); enc = load_encoder(a.encoder)
    eps = list_eps(cfg); print(f"[{a.ds}] {len(eps)} eps — parallel-decode (CPU) + big-batch encode (GPU)...", flush=True)
    from concurrent.futures import ThreadPoolExecutor   # cv2/pyav decode releases the GIL → threads parallelize across cores

    def _dec(e):
        try:
            f224, state, th, _ = load_ep(cfg, e, strd=1)
            return (e, f224, state, th) if len(f224) >= 5 else None
        except Exception:
            return None
    td = time.time()
    with ThreadPoolExecutor(max_workers=min(a.decode_workers, len(eps))) as ex:
        loaded = [r for r in ex.map(_dec, eps) if r is not None]   # map preserves eps order → deterministic milestones
    print(f"[{a.ds}] decoded {len(loaded)}/{len(eps)} eps in parallel ({time.time()-td:.0f}s)", flush=True)
    # encode: optionally split episodes across cuda:0 + cuda:1 (encode is GPU-compute-bound → 2nd A100 ~doubles throughput)
    encoders = [enc]
    if a.multi_gpu:
        import torch
        if torch.cuda.device_count() >= 2:
            encoders.append(load_encoder(a.encoder, device="cuda:1")); print(f"[{a.ds}] 2-GPU encode: cuda:0 + cuda:1", flush=True)
        else:
            print(f"[{a.ds}] --multi-gpu set but only {torch.cuda.device_count()} GPU visible (set CUDA_VISIBLE_DEVICES=0,1)", flush=True)
    for eo in encoders:   # force model load on each device SEQUENTIALLY first — concurrent lazy-load races on the transformers import (ImportError AutoImageProcessor)
        eo.encode_pooled(np.zeros((2, 224, 224, 3), np.uint8), bs=2)
    te = time.time(); groups = [[] for _ in encoders]
    for i, item in enumerate(loaded): groups[i % len(encoders)].append((i, item))

    def _enc_group(gi):
        eo = encoders[gi]; out = []
        for i, (e, f224, state, th) in groups[gi]:
            out.append((i, e, L2(eo.encode_pooled(f224, bs=a.enc_bs)), state, th, len(f224)))
        return out
    with ThreadPoolExecutor(max_workers=len(encoders)) as ex:
        merged = sorted([r for fut in [ex.submit(_enc_group, gi) for gi in range(len(encoders))] for r in fut.result()], key=lambda r: r[0])
    print(f"[{a.ds}] encoded {len(merged)} eps on {len(encoders)} GPU(s) ({time.time()-te:.0f}s)", flush=True)
    POOL, STATE, EPID, TPOS, THUMB, eplen = [], [], [], [], [], {}
    for i, e, pooled, state, th, n in merged:
        POOL.append(pooled); STATE.append(mkp_gap(state, cfg.stride))
        EPID.append(np.full(n, e)); TPOS.append(np.arange(n) / max(1, n - 1)); THUMB += th; eplen[e] = n
    img = np.concatenate(POOL); Pm = np.concatenate(STATE); E = np.concatenate(EPID); Tv = np.concatenate(TPOS); THUMB = np.stack(THUMB)
    PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
    F = np.concatenate([img, a.proprio_weight * L2((Pm - PMU) / PSD)], 1); ne = len(eps)   # proprio down-weighted → vision-primary assignment (fixes XVLA proprio-driven mis-match)
    F = temporal_smooth(F, E, a.feat_smooth)   # temporal smoothing → fewer noise-driven flips (XVLA emission margin tiny)
    if a.feat_smooth > 1: print(f"[{a.ds}] temporal feature smoothing: window={a.feat_smooth} frames", flush=True)
    if a.max_gap > 0:   # uniformity-aware selection: cap milestone value gaps (fills coffee's v=0.56→0.94 void)
        cen, lab, order, Pord, M, nf = build_milestones_uniform(F, E, Tv, ne, max_gap=a.max_gap)
        print(f"[{a.ds}] uniform milestone selection: M={M} (+{nf} gap-filled, max_gap={a.max_gap})", flush=True)
    else:
        cen, lab, order, Pord, M = build_milestones(F, E, Tv, ne)
    C = cen[order]
    eps_sorted = sorted(set(E.tolist()))
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_
    am = np.linalg.norm(F[:, None] - C[None], axis=2).argmin(1); counts = np.zeros((M, M))
    for e in eps_sorted:
        seq = visited_sequence(am[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])]])
        for i, j in zip(seq[:-1], seq[1:]): counts[i, j] += 1
    pen = build_pen(counts, Pord, a.lam_geo, a.beta, a.back_barrier)
    tau_lo = tau_hi = None   # confidence scale from mining frames: nearest-milestone distance = clusters' natural radius
    if a.conf_hi > 0:
        dM = np.linalg.norm(F[:, None] - C[None], axis=2).min(1)
        tau_lo, tau_hi = float(np.percentile(dM, a.conf_lo)), float(np.percentile(dM, a.conf_hi))
        print(f"[{a.ds}] confidence hold: tau_lo(p{a.conf_lo:.0f})={tau_lo:.3f} tau_hi(p{a.conf_hi:.0f})={tau_hi:.3f}", flush=True)
    print(f"[{a.ds}] {M} milestones, transition prior built (back={a.back_barrier:.0f})", flush=True)

    def ms_of(Fq):
        base = np.linalg.norm(Fq[:, None] - C[None], axis=2)
        ms = viterbi_pen(emit_of(Fq, C, sk, Pord, a.gamma, a.margin, a.gamma_fwd, a.margin_fwd), pen)
        return conf_hold(ms, base, tau_lo, tau_hi, a.hold_thresh)

    # ---- (A) validation figure with transition-prior readout (value smoothed at NATIVE rate) ----
    nfps = a.native_fps or {"coffee": 50.0, "xvla": 30.0}.get(a.ds, 30.0)
    mwv = max(5, int(round(5 * nfps / 3))) | 1   # native-rate value-smoothing window (3Hz med-5 → jittery at 50Hz)
    rows = []
    for e in eps_sorted:
        fi = np.where(E == e)[0]; fi = fi[np.argsort(Tv[fi])]
        if len(fi) < 5: continue
        v = smooth_monotone(med(Pord[ms_of(F[fi])], mwv), fps=nfps); tq = Tv[fi]
        rows.append((e, float(np.corrcoef(v, tq)[0, 1]) if v.std() > 1e-6 else np.nan, v, tq))
    corrs = np.array([r[1] for r in rows]); ok = np.isfinite(corrs)
    ep_corr = {r[0]: r[1] for r in rows}
    ev = {"ds": a.ds, "readout": f"transition back{a.back_barrier:.0f}", "n_eps": int(ok.sum()), "M": int(M),
          "corr_mean": float(corrs[ok].mean()), "corr_median": float(np.median(corrs[ok])),
          "frac_corr_ge_0.7": float(np.mean(corrs[ok] >= 0.7))}
    json.dump(ev, open(OUTV / f"{a.ds}_evalall.json", "w"), indent=2); np.save(OUTV / f"{a.ds}_corrs.npy", corrs)
    print(f"[{a.ds}] {ev}", flush=True)
    plt = setup_mpl(); fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].hist(corrs[ok], bins=24, color="#1a7f37", alpha=0.8); ax[0].axvline(0.7, color="r", ls="--", lw=1.5, label="阈值 0.7")
    ax[0].set_title(f"[{a.ds}] 逐 episode corr(value,进度) — 转移先验读出\nmean={ev['corr_mean']:.3f} median={ev['corr_median']:.3f} %≥0.7={ev['frac_corr_ge_0.7']:.0%} (n={ev['n_eps']})")
    ax[0].set_xlabel("corr(value, normalized-time)"); ax[0].set_ylabel("#episodes"); ax[0].legend(fontsize=9)
    mi = sorted([r for r in rows if np.isfinite(r[1])], key=lambda r: r[1])
    for r in [mi[len(mi)//10], mi[len(mi)//2], mi[-1]]:
        ax[1].plot(r[3], r[2], lw=1.8, label=f"ep{r[0]} corr={r[1]:.2f}")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=.4, label="参照 value=进度"); ax[1].set_xlim(0, 1); ax[1].set_ylim(-.02, 1.02)
    ax[1].set_title(f"[{a.ds}] 示例 value 曲线(转移先验,无循环态假跌)"); ax[1].set_xlabel("归一化进度"); ax[1].set_ylabel("CRAVE value"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle(f"跨数据集验证(转移先验读出)— {a.ds}({ne}ep)— {a.encoder} 零训练 — {M} milestones", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUTV / f"{a.ds}_validation.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[{a.ds}] SAVED {a.ds}_validation.png", flush=True)

    # ---- centroid-decode protos ----
    NS = 20; samp = []; rng = []
    for c in order:
        loc = np.where(lab == c)[0]
        if len(loc) > NS: loc = loc[np.linspace(0, len(loc) - 1, NS).astype(int)]
        s0 = len(samp); [samp.append(cv2.resize(THUMB[gi], (224, 224))) for gi in loc]; rng.append((s0, len(samp)))
    samp = np.stack(samp); grids = enc.encode_grid(samp); imgs128 = np.stack([cv2.resize(s, (128, 128)) for s in samp])
    decf = train_dec(grids, imgs128, enc.dim, "small", 55)
    proto = {mi: (cv2.resize(decf(grids[s0:e0].mean(0)[None])[0], (128, 128)) if e0 > s0 else np.zeros((128, 128, 3), np.uint8)) for mi, (s0, e0) in enumerate(rng)}

    # ---- milestone gallery (decoded centroid | medoid), uniform-selection milestones ----
    medoid = {}
    for mi, c in enumerate(order):
        loc = np.where(lab == c)[0]; gi = loc[int(np.argmin(np.linalg.norm(F[loc] - cen[c], axis=1)))]
        medoid[mi] = cv2.resize(THUMB[gi], (96, 96))
    PR = 18; NBd = (M + PR - 1) // PR
    figg, axg = plt.subplots(2 * NBd, PR, figsize=(PR * 0.82, 2 * NBd * 1.18)); axg = np.atleast_2d(axg)
    for ax in axg.ravel(): ax.axis("off")
    for mi in range(M):
        b, col = mi // PR, mi % PR
        axg[2 * b, col].imshow(cv2.resize(proto[mi], (96, 96))); axg[2 * b, col].set_title(f"m{mi} v={Pord[mi]:.2f}", fontsize=6)
        axg[2 * b + 1, col].imshow(medoid[mi])
    figg.suptitle(f"[{a.ds}] {M} milestones(均匀化选簇,封堵 value 空洞)— top: decoded centroid | bottom: medoid", fontsize=12)
    figg.tight_layout(); figg.savefig(OUTV / f"{a.ds}_milestone_gallery.png", dpi=120, bbox_inches="tight"); plt.close(figg)
    print(f"[{a.ds}] SAVED {a.ds}_milestone_gallery.png", flush=True)

    # ---- (B) long-episode video: forced --video-ep, else longest among well-tracked eps (corr>=video_corr) ----
    if a.video_ep >= 0 and a.video_ep in eplen:
        e = a.video_ep
    else:
        good = [x for x in eplen if ep_corr.get(x, -1) >= a.video_corr]
        e = max(good or list(eplen), key=lambda k: eplen[k])
    print(f"[{a.ds}] video ep {e} ({eplen[e]} stride-frames, corr={ep_corr.get(e, float('nan')):.2f}); native readout+video...", flush=True)
    f224, staten, fpsd = load_ep_native(cfg, e)
    Fq = np.concatenate([L2(enc.encode_pooled(f224, bs=a.enc_bs)), a.proprio_weight * L2((mkp_gap(staten, cfg.stride) - PMU) / PSD)], 1)
    Fq = _smooth_block(Fq, a.feat_smooth)   # same proprio-weight + temporal smoothing as clustering F
    ms = ms_of(Fq)   # full readout (progress prior + far-ahead cap + confidence hold), consistent with validation
    mw = max(5, int(round(5 * fpsd / 3))) | 1
    v = smooth_monotone(med(Pord[ms], mw), fps=fpsd); n = len(v)
    H = 360; CW = int(round(f224[0].shape[1] / f224[0].shape[0] * H)); PW, CardW = 540, 300; L, Rr, Tt, B = 56, 16, 16, 38
    base = np.full((H, PW, 3), 255, np.uint8)
    cv2.line(base, (L, H - B), (PW - Rr, Tt), (210, 210, 210), 1)
    def px(t): return int(L + t / max(1, n - 1) * (PW - Rr - L))
    def py(val): return int((H - B) - val * (H - B - Tt))
    pts = np.array([[px(t), py(v[t])] for t in range(n)], np.int32)
    cv2.polylines(base, [pts], False, (55, 127, 26), 2)
    cv2.putText(base, "CRAVE value (transition prior)", (L, Tt + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    Wt = (CW + PW + CardW) // 2 * 2; Ht = H // 2 * 2
    with tempfile.TemporaryDirectory() as td:
        oc = __import__("av").open(str(OUTV / f"{a.ds}_transition_video.mp4"), mode="w")
        stv = oc.add_stream("libx264", rate=int(round(fpsd))); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"; stv.options = {"crf": "23", "preset": "veryfast"}
        for t in range(n):
            cam = cv2.resize(f224[t], (CW, H)); panel = base.copy()
            cv2.line(panel, (px(t), Tt), (px(t), H - B), (150, 150, 150), 1); cv2.circle(panel, (px(t), py(v[t])), 5, (214, 39, 40), -1)
            cv2.putText(panel, f"v={v[t]:.2f}  m{ms[t]}  P={Pord[ms[t]]:.2f}", (L, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
            card = np.full((H, CardW, 3), 245, np.uint8); pr = cv2.resize(proto[ms[t]], (CardW - 24, CardW - 24)); card[40:40 + pr.shape[0], 12:12 + pr.shape[1]] = pr
            cv2.putText(card, "aligned milestone", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
            cv2.putText(card, f"#{ms[t]}  (centroid decode)", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (59, 111, 176), 1, cv2.LINE_AA)
            canv = np.concatenate([cam, panel, card], 1)[:Ht, :Wt]
            if canv.shape[:2] != (Ht, Wt): canv = cv2.resize(canv, (Wt, Ht))
            for pkt in stv.encode(__import__("av").VideoFrame.from_ndarray(np.ascontiguousarray(canv), format="rgb24")): oc.mux(pkt)
        for pkt in stv.encode(): oc.mux(pkt)
        oc.close()
    print(f"[{a.ds}] SAVED {a.ds}_transition_video.mp4  ep{e} {n}f@{fpsd:.0f}fps ({n/fpsd:.0f}s) ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
