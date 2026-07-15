#!/usr/bin/env python
"""Generate a NEW-CRAVE-method milestone_file per dataset, in the shared DINOv3-H space.

New method (see crave/docs/final_architecture.md, 2026-07-09 收口):
  img : DINOv3-H 1280D pooled -> L2 -> PCA128 -> L2
  pos : proprio 14D -> zero-mean/unit-var -> L2
  joint = concat[img128, pos14]           # each side L2 => energy 1:1
  cluster: BayesianGaussianMixture(diag, adaptive K) over l2(joint)
  milestone = surviving cluster mode with per-mode cross-episode coverage >= 0.50
  value = median normalized-time T of the mode's member frames (order-preserving)

Output milestone_file {C(M,1280), Pord(M)}:
  C[m]    = mean of L2(DINOv3-H feature) over frames of milestone m  (shared 1280D space,
            consumed unchanged by build_recurrence_graph.py -> train_multitask.py)
  Pord[m] = median T of milestone m  (progress ordering)

Milestones are DISCOVERED in the img128⊕pos joint space (the better segmentation from the
new CRAVE ablations) but REPRESENTED as DINOv3-H 1280D centroids so the multitask predictor
keeps a single shared input/prototype space across kai0/coffee/vis/xvla.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

REPO = Path(os.environ.get("CRAVE_REPO", "/home/tim/workspace/deepdive_kai0"))  # gf3: export CRAVE_REPO=<repo root>
MIN_COV = 0.50
PCA_DIM = 128

# fdir = cached DINOv3-base bank (user-prepared, 768D full-rate); proprio = per-format loader; root = raw dataset
TASKS = {
    "kai0":   dict(fdir="lmvla/crave/data/kai_dinov3base",         proprio="kai0",      root="kai0/data/Task_A/kai0_base"),
    "coffee": dict(fdir="lmvla/crave/data/coffee_dinov3base",      proprio="lerobotv3", root="temp/aloha_static_coffee"),
    "libero10": dict(fdir="lmvla/crave/data/libero10_dinov3base",  proprio="kai0",
                     root="/vePFS/tim/workspace/LIBERO_fastwam/libero_10_no_noops_lerobot"),  # LaWM in-dist data
    "libero_spatial": dict(fdir="lmvla/crave/data/liberospatial_dinov3base", proprio="kai0",
                     root="/vePFS/tim/workspace/LIBERO_fastwam/libero_spatial_no_noops_lerobot"),
    "libero_goal": dict(fdir="lmvla/crave/data/liberogoal_dinov3base", proprio="kai0",
                     root="/vePFS/tim/workspace/LIBERO_fastwam/libero_goal_no_noops_lerobot"),
    **{f"aloha_{k}": dict(fdir=f"lmvla/crave/data/aloha_static_{v}_dinov3base", proprio="lerobotv3",
                     root=f"temp/aloha_tasks/aloha_static_{v}")
       for k, v in {"candy": "candy", "cups": "cups_open", "ziploc": "ziploc_slide", "screw": "screw_driver"}.items()},
    "vis":    dict(fdir="lmvla/crave/data/vis_dinov3base",         proprio="kai0",      root="kai0/data/Task_A/vis_base/v1/2026-04-24"),
    "xvla":   dict(fdir="lmvla/crave/data/xvla_dinov3base_full",   proprio="hdf5",      root="xvla/data/xvla_soft_fold/0707_11pm_stage_1_stage2new_new_cam_very_slow"),
}


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def load_feats(fdir: Path):
    idx = np.load(fdir / "index.npz")
    e = idx["E"].astype(np.int64)
    fr = idx["FR"].astype(np.int64)
    tnorm = idx["T"].astype(np.float32)
    n = int(idx["n"])
    shards = sorted(s for s in fdir.glob("shard_*.npz") if "_bak" not in s.name)
    dim = int(np.load(shards[0])["feat"].shape[1])                       # infer feat dim (768 base / 1280 H)
    feat = np.zeros((n, dim), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for sh in shards:
        z = np.load(sh)
        g = z["gidx"].astype(np.int64)
        feat[g] = z["feat"]
        valid[g] = z["valid"].astype(bool)
    return e[valid], fr[valid], tnorm[valid], feat[valid].astype(np.float32)


# ---------- per-format proprio: return {ep: state_array (n_native, 14)} ----------
def _proprio_kai0(root: Path, eps):
    import pandas as pd
    info = json.loads((root / "meta/info.json").read_text())
    cs = int(info.get("chunks_size", 1000))
    out = {}
    for e in eps:
        p = root / f"data/chunk-{e // cs:03d}/episode_{e:06d}.parquet"
        st = np.stack(pd.read_parquet(p, columns=["observation.state"])["observation.state"].to_numpy())
        out[int(e)] = st.astype(np.float32)
    return out


def _proprio_lerobotv3(root: Path, eps):
    import glob

    import pandas as pd
    frames = []
    for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)):
        frames.append(pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"]))
    df = pd.concat(frames, ignore_index=True)
    out = {}
    for e in eps:
        sub = df[df["episode_index"] == int(e)].sort_values("frame_index")
        out[int(e)] = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)
    return out


def _proprio_hdf5(root: Path, eps):
    import h5py
    out = {}
    for e in eps:
        fp = root / f"episode_{int(e)}.hdf5"
        if not fp.exists():
            continue
        with h5py.File(fp, "r") as h:
            out[int(e)] = np.asarray(h["observations/qpos"][:], dtype=np.float32)
    return out


def load_proprio(kind: str, root: Path, eps):
    if kind == "kai0":
        return _proprio_kai0(root, eps)
    if kind == "lerobotv3":
        return _proprio_lerobotv3(root, eps)
    if kind == "hdf5":
        return _proprio_hdf5(root, eps)
    raise NotImplementedError(kind)


def mode_split(Tc: np.ndarray, nbins: int = 30):
    """Valley-split a cluster's time histogram into modes; return [(median, submask_over_Tc), ...]."""
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1))
    h = h.astype(float) / (h.sum() + 1e-9)
    hs = gaussian_filter1d(h, 1.2)
    c = (ed[:-1] + ed[1:]) / 2
    peaks = [i for i in range(nbins)
             if hs[i] >= hs[max(0, i - 1)] and hs[i] >= hs[min(nbins - 1, i + 1)] and hs[i] >= 0.10 * hs.max()]
    merged = []
    for p in peaks:
        if merged and abs(c[p] - c[merged[-1]]) < 0.10:
            if hs[p] > hs[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    final = [merged[0]] if merged else [int(np.argmax(hs))]
    for p in merged[1:]:
        valley = hs[final[-1]:p + 1].min()
        if valley < 0.6 * min(hs[final[-1]], hs[p]):
            final.append(p)
        elif hs[p] > hs[final[-1]]:
            final[-1] = p
    if len(final) <= 1:
        return [(float(np.median(Tc)), np.ones(len(Tc), bool))]
    cuts = [c[a + int(np.argmin(hs[a:b + 1]))] for a, b in zip(final[:-1], final[1:])]
    edges = [0.0] + cuts + [1.0]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (Tc >= lo) & (Tc < hi)
        if msk.sum() >= 5:
            out.append((float(np.median(Tc[msk])), msk))
    return out or [(float(np.median(Tc)), np.ones(len(Tc), bool))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(TASKS))
    ap.add_argument("--n_components", type=int, default=40)
    ap.add_argument("--wcp", type=float, default=1e-2, help="Dirichlet weight_concentration_prior")
    ap.add_argument("--min_cov", type=float, default=MIN_COV)
    ap.add_argument("--max_frames", type=int, default=500000,
                    help="subsample frames for PCA/BGMM fit + assignment (0=all); keeps every episode represented")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = TASKS[args.dataset]
    fdir = REPO / cfg["fdir"]
    root = REPO / cfg["root"]
    out = args.out or (REPO / f"temp/newcrave_specs/{args.dataset}_milestones_newmethod.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{args.dataset}] loading DINOv3-base features from {fdir} ...", flush=True)
    E, FR, T, F = load_feats(fdir)
    eps = np.unique(E)
    NC = len(eps)
    print(f"  {len(F)} frames, {NC} episodes", flush=True)

    # subsample frames FIRST (stratified per-episode so every episode stays represented) for tractable fit
    if args.max_frames and len(F) > args.max_frames:
        rng = np.random.default_rng(args.seed)
        keep_per = max(1, args.max_frames // NC)
        order = np.argsort(E, kind="stable")                            # group frames by episode
        Es = E[order]; bounds = np.searchsorted(Es, eps, side="left")
        ends = np.append(bounds[1:], len(Es))
        sel = []
        for b, e_ in zip(bounds, ends):
            idx = order[b:e_]
            sel.append(idx if len(idx) <= keep_per else rng.choice(idx, keep_per, replace=False))
        sel = np.sort(np.concatenate(sel))
        print(f"  subsample {len(F)} -> {len(sel)} frames ({keep_per}/ep, {NC} eps kept)", flush=True)
        E, FR, T, F = E[sel], FR[sel], T[sel], F[sel]

    print("  loading proprio ...", flush=True)
    pro = load_proprio(cfg["proprio"], root, np.unique(E))
    pdim = next((v.shape[1] for v in pro.values() if v is not None and len(v)), 14)  # 14 kai0 / 8 LIBERO
    ST = np.zeros((len(F), pdim), np.float32)
    miss = 0
    for e in np.unique(E):                                              # vectorized per-episode fill
        st = pro.get(int(e))
        m = E == e
        if st is None or len(st) == 0:
            miss += int(m.sum())
            continue
        ST[m] = st[np.minimum(FR[m], len(st) - 1)]
    if miss:
        print(f"  WARN: {miss} frames missing proprio (episodes absent)", flush=True)

    # normalize T to [0,1] within each episode (index["T"] is raw seconds = FR/fps, not normalized)
    Tn = np.zeros_like(T)
    for e in np.unique(E):
        m = E == e
        t = T[m]; lo, hi = t.min(), t.max()
        Tn[m] = (t - lo) / (hi - lo) if hi > lo else 0.0
    T = Tn

    # img: L2 -> PCA128 -> L2 ; pos: standardize -> L2 ; joint 1:1
    print("  PCA(D->128) on img ...", flush=True)
    imgL = l2(F)
    pca = PCA(n_components=PCA_DIM, random_state=args.seed).fit(imgL)
    img128 = l2(pca.transform(imgL).astype(np.float32))
    SMU, SSD = ST.mean(0), ST.std(0) + 1e-8
    pos = l2((ST - SMU) / SSD)
    joint = np.concatenate([img128, pos], axis=1)
    Jn = l2(joint)

    print(f"  BayesianGMM(n={args.n_components}, diag) fitting ...", flush=True)
    t0 = time.time()
    bgmm = BayesianGaussianMixture(n_components=args.n_components, covariance_type="diag",
                                   weight_concentration_prior=args.wcp, n_init=1, max_iter=150,
                                   random_state=args.seed).fit(Jn)
    labs = bgmm.predict(Jn)
    print(f"    eff_components={(bgmm.weights_ > 0.01).sum()} ({time.time() - t0:.0f}s)", flush=True)

    # candidate centroids = bgmm components with >=20 frames, nearest-centroid re-assign
    cand = []
    for k in range(args.n_components):
        mk = labs == k
        if mk.sum() < 20:
            continue
        cand.append(Jn[mk].mean(0))
    cand = np.asarray(cand, np.float32)
    assign = np.empty(len(Jn), int)
    for i in range(0, len(Jn), 20000):
        assign[i:i + 20000] = np.linalg.norm(Jn[i:i + 20000, None] - cand[None], axis=2).argmin(1)

    # per-candidate mode-split + per-mode coverage/median -> milestones (global member idx)
    milestones = []
    n_drop = 0
    for ki in range(len(cand)):
        gm = np.where(assign == ki)[0]
        if len(gm) < 20:
            continue
        Tc = T[gm]
        for _mv, sub in mode_split(Tc):
            subg = gm[sub]
            cov = len(set(E[subg].tolist())) / NC
            if cov >= args.min_cov:
                milestones.append((float(np.median(T[subg])), subg))
            else:
                n_drop += 1
    milestones.sort(key=lambda x: x[0])
    M = len(milestones)
    C = np.stack([l2(F[g]).mean(0) for _v, g in milestones]).astype(np.float32)   # (M,1280) DINOv3-H centroids
    Pord = np.asarray([v for v, _g in milestones], np.float32)
    covs = np.asarray([len(set(E[g].tolist())) / NC for _v, g in milestones], np.float32)
    sizes = np.asarray([len(g) for _v, g in milestones], np.int64)
    print(f"  kept {M} milestones (dropped {n_drop} weak modes)", flush=True)
    print(f"  Pord(median T): {[round(float(v), 2) for v in Pord]}", flush=True)
    print(f"  coverage: {[round(float(c), 2) for c in covs]}", flush=True)

    # sanity: monotone ordering has no inversions by construction (sorted); check duplicate collapse
    inv = int((np.diff(Pord) < 0).sum())
    np.savez_compressed(
        out,
        C=C, Pord=Pord, covs=covs, sizes=sizes, M=np.int64(M),
        min_cov=np.float32(args.min_cov), pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32), SMU=SMU.astype(np.float32),
        SSD=SSD.astype(np.float32), method=np.str_("dinov3h_pca128_pos_bgmm_median"),
    )
    meta = {
        "dataset": args.dataset, "n_frames": int(len(F)), "n_episodes": int(NC),
        "n_milestones": int(M), "dropped_weak_modes": int(n_drop),
        "pord_inversions": inv, "min_cov": args.min_cov,
        "n_components": args.n_components, "wcp": args.wcp,
        "output": str(out), "feature_dir": str(fdir), "proprio_kind": cfg["proprio"],
        "prototype_space": "DINOv3-base 768D (cluster mean of L2 feat)",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
