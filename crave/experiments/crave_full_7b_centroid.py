"""FULL kai0_base @3Hz with DINOv3-7B(int8): cluster the whole 3055-ep set, then
decode the cluster centroids with a trained 16->128 decoder ("簇中心解码图").

Memory note: 334k frames × 4096-d grids can't be held (≈700GB), so the encode stage
shards only POOLED features to disk; the decode stage re-encodes a training subsample
+ each milestone's top-N nearest frames to get grids (bounded), trains the decoder, and
decodes the per-cluster mean grid.

Stages (run encode sharded across the 2 GPUs, then decode once):
  --stage encode --rank 0 --world 2   (GPU0)   } pooled feat shards ->
  --stage encode --rank 1 --world 2   (GPU1)   } temp/crave_full_d3b7/shard_R.npz
  --stage decode                                -> centroid-decode gallery PNG
Small smoke: --stage encode --mine-n 8 --world 1 ; then --stage decode --mine-n 8
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crave.config import REPO, resolve_dataset, viz_dir          # noqa: E402
from crave.config.encoders import resolve as resolve_enc         # noqa: E402
from crave.data import kai0                                       # noqa: E402
from crave.decoding.decoder import train_dec                      # noqa: E402
from crave.encoders import load_encoder                           # noqa: E402
from crave.utils import otsu                                      # noqa: E402

# Set per-encoder in main(); defaults keep module importable.
ENC = "dinov3-h"
DIM = 1280
TAG = "dinov3h"
OUTD = REPO / "temp/crave_full_dinov3h"
cfg = resolve_dataset("kai0_base")
CS = kai0.chunks_size(cfg.root)


def n30(e: int) -> int:
    pq = Path(cfg.root) / "data" / f"chunk-{e // CS:03d}" / f"episode_{e:06d}.parquet"
    return len(pd.read_parquet(pq, columns=["timestamp"]))


def build_index(mine_n: int):
    eps = sorted(int(p.stem.split("_")[1]) for p in (Path(cfg.root) / "data").glob("chunk-*/episode_*.parquet"))
    if mine_n and mine_n < len(eps):
        eps = sorted(np.random.RandomState(0).permutation(eps)[:mine_n].tolist())
    E, FR, T = [], [], []
    for e in eps:
        n = max(1, n30(e) // 10)                       # 3Hz from 30Hz
        for i in range(n):
            E.append(e); FR.append(i * 10); T.append(i / max(1, n - 1))
    return np.array(E), np.array(FR), np.array(T, np.float32)


def L2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


# ----------------------------------------------------------------------------- encode
def stage_encode(a):
    t0 = time.time(); E, FR, T = build_index(a.mine_n); N = len(E)
    OUTD.mkdir(parents=True, exist_ok=True)
    if a.rank == 0:
        np.savez(OUTD / "index.npz", E=E, FR=FR, T=T, n=N)
    shard = np.arange(N)[a.rank::a.world]
    print(f"[rank {a.rank}/{a.world}] {ENC} shard {len(shard)}/{N} frames @3Hz", flush=True)
    enc = load_encoder(ENC)
    feat = np.zeros((len(shard), DIM), np.float16); valid = np.zeros(len(shard), bool)
    for c in range(0, len(shard), a.chunk):
        sub = shard[c:c + a.chunk]
        imgs, val = kai0.decode_images(cfg, sub, E, FR)
        vi = np.where(val)[0]
        for b in range(0, len(vi), 16):
            bb = vi[b:b + 16]
            f = enc.encode_pooled([imgs[i] for i in bb], bs=16)
            feat[c + bb] = f.astype(np.float16); valid[c + bb] = True
        print(f"[rank {a.rank}] {min(c+a.chunk,len(shard))}/{len(shard)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(OUTD / f"shard_{a.rank}.npz", gidx=shard, feat=feat, valid=valid)
    print(f"[rank {a.rank}] SAVED shard_{a.rank}.npz ({time.time()-t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- decode
def _grids_for(enc, idxs, E, FR):
    """Re-encode a set of global frame indices -> (grids(n,DIM,16,16) fp16, imgs128 uint8, ok mask)."""
    imgs, ok = kai0.decode_images(cfg, idxs, E, FR)
    vi = np.where(ok)[0]
    grids = np.zeros((len(idxs), DIM, 16, 16), np.float16)
    for b in range(0, len(vi), 16):
        bb = vi[b:b + 16]
        grids[bb] = enc.encode_grid([imgs[i] for i in bb], bs=16)
    th = np.stack([cv2.resize(imgs[i], (128, 128), interpolation=cv2.INTER_AREA) for i in range(len(idxs))])
    return grids, th, ok


def stage_decode(a):
    t0 = time.time()
    z = np.load(OUTD / "index.npz"); E, FR, T, N = z["E"], z["FR"], z["T"], int(z["n"])
    feat = np.zeros((N, DIM), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / "shard_*.npz"))):
        s = np.load(f); feat[s["gidx"]] = s["feat"]; valid[s["gidx"]] = s["valid"]
    vi = np.where(valid)[0]; F = L2(feat[vi].astype(np.float32))
    print(f"[decode] {len(vi)}/{N} valid frames; KMeans ...", flush=True)

    from sklearn.cluster import MiniBatchKMeans
    K0 = a.k0 if a.k0 else int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    fit_idx = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(K0, random_state=0, batch_size=4096, n_init=3).fit(F[fit_idx])
    cen = km.cluster_centers_; lab = km.predict(F)
    Tv, Ev = T[vi], E[vi]; ne = len(set(E.tolist()))
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); vt = tstd[tstd < 9]; tau_pur = float(np.percentile(vt, 60)) if len(vt) else 9.0
    if a.cov_above_mean:
        sel = sorted([c for c in range(K0) if cov[c] >= cov.mean()], key=lambda c: tpos[c])  # drop below-mean coverage
    elif a.all_milestones:
        sel = sorted(range(K0), key=lambda c: tpos[c])          # coarse K: every cluster is a high-coverage phase
    else:
        cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
        gap = max(0.006, 0.5 / max(len(cand), 1)); sel = []
        for c in cand:
            if not sel or tpos[c] - tpos[sel[-1]] >= gap: sel.append(c)
            elif cov[c] > cov[sel[-1]]: sel[-1] = c
    print(f"[decode] K0={K0} cov-τ={tau_cov:.3f} pur-τ={tau_pur:.3f} → {len(sel)} milestones "
          f"(cov range {cov[sel].min():.2f}-{cov[sel].max():.2f}); re-encoding for decoder ...", flush=True)

    enc = load_encoder(ENC)
    # decoder training pairs: random valid frames
    rs = np.random.RandomState(1)
    tr = vi[rs.choice(len(vi), min(a.train_imgs, len(vi)), replace=False)]
    g_tr, im_tr, ok_tr = _grids_for(enc, tr, E, FR)
    keep = np.where(ok_tr)[0]; g_tr, im_tr = g_tr[keep], im_tr[keep]
    print(f"[decode] decoder pairs {len(g_tr)} ({time.time()-t0:.0f}s); re-encoding milestone centroids ...", flush=True)
    # per milestone: top-NN nearest frames -> mean grid (faithful centroid), + nearest real frame
    NN = a.topk; cen_grids = []; near_imgs = []
    for c in sel:
        loc = np.where(lab == c)[0]; d = np.linalg.norm(F[loc] - cen[c], axis=1)
        g = vi[loc[np.argsort(d)][:NN]]
        gg, th, ok = _grids_for(enc, g, E, FR)
        kk = np.where(ok)[0]
        cen_grids.append(gg[kk].mean(0) if len(kk) else np.zeros((DIM, 16, 16), np.float16))
        near_imgs.append(th[kk[0]] if len(kk) else np.zeros((128, 128, 3), np.uint8))
    cen_grids = np.stack(cen_grids)
    enc.unload()
    import torch; torch.cuda.empty_cache()

    print(f"[decode] training 16->128 decoder ({len(g_tr)} pairs, {a.epochs}ep) ...", flush=True)
    decode = train_dec(g_tr, im_tr, DIM, dec="small", epochs=a.epochs)
    dec_cen = decode(cen_grids.astype(np.float32))

    from crave.render import setup_mpl
    plt = setup_mpl()
    M = len(sel); ncol = min(16, M); nrow = int(np.ceil(M / ncol))
    fig, ax = plt.subplots(2 * nrow, ncol, figsize=(1.25 * ncol, 2.6 * nrow)); ax = np.atleast_2d(ax)
    for i in range(M):
        rr, cc = divmod(i, ncol)
        a0 = ax[2 * rr, cc]; a0.imshow(dec_cen[i]); a0.axis("off"); a0.set_title(f"m{i} P={tpos[sel[i]]:.2f}", fontsize=6)
        a1 = ax[2 * rr + 1, cc]; a1.imshow(near_imgs[i]); a1.axis("off")
    for j in range(M, nrow * ncol):
        rr, cc = divmod(j, ncol); ax[2 * rr, cc].axis("off"); ax[2 * rr + 1, cc].axis("off")
    ax[0, 0].set_ylabel("decoded\ncentroid", fontsize=7)
    fig.suptitle(f"FULL kai0_base @3Hz ({ne}ep/{len(vi)}fr) — {ENC} cluster + trained 16→128 decoder — {M} milestones\n"
                 f"(每个 milestone: 上=簇中心解码图(top-{NN}近邻平均grid解码), 下=最近真实帧)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    suffix = (f"_k{a.k0}" if a.k0 else "") + ("_covmean" if a.cov_above_mean else "")
    out = viz_dir("encoders") / f"enc_full_{TAG}_kai0_centroid_decode{suffix}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    json.dump({"encoder": ENC, "ep": ne, "frames": int(len(vi)), "K0": int(K0), "milestones": M,
               "tau_cov": float(tau_cov), "tau_pur": float(tau_pur)}, open(OUTD / "summary.json", "w"), indent=2)
    print(f"SAVED {out}  ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov3-h", help="any DINO encoder name (e.g. dinov3-h, dinov3-7b-int8)")
    ap.add_argument("--stage", choices=["encode", "decode"], required=True)
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--mine-n", type=int, default=0); ap.add_argument("--chunk", type=int, default=6000)
    ap.add_argument("--train-imgs", type=int, default=4000); ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--k0", type=int, default=0, help="force cluster count (0=adaptive 0.55√N)")
    ap.add_argument("--all-milestones", action="store_true", help="use every cluster as a milestone (skip coverage/purity filter) — for coarse K")
    ap.add_argument("--cov-above-mean", action="store_true", help="keep only clusters with coverage >= mean coverage (drop below-average)")
    a = ap.parse_args()
    global ENC, DIM, TAG, OUTD
    ENC = a.encoder; DIM = resolve_enc(ENC).dim
    TAG = ENC.replace("dinov3-", "dinov3").replace("-", "")
    OUTD = REPO / f"temp/crave_full_{TAG}"
    print(f"[cfg] encoder={ENC} dim={DIM} tag={TAG} out={OUTD}", flush=True)
    (stage_encode if a.stage == "encode" else stage_decode)(a)


if __name__ == "__main__":
    main()
