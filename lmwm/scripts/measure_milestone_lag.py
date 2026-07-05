#!/usr/bin/env python
"""Compare the time-lag of milestone+1 vs the current frame:
  (A) DATASET lag  = time(real next-stage medoid) - time(current)         [ground-truth horizon]
  (B) MODEL  lag   = time(frame whose grid is nearest the PREDICTED m+1) - time(current)
                     [effective horizon the model's forward-from-current prediction actually reaches]

If MODEL lag << DATASET lag, the deterministic predictor under-shoots (hedges toward nearer, more
certain futures) instead of committing to the full milestone jump.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_lawm_patch import load_index, read_imgs, ForwardDec  # noqa: E402
from optimize_subgoal import PredM  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="lmwm/outputs/subgoal_opt/milestone_cd128.pt")
    ap.add_argument("--mode", choices=["milestone", "nearfuture"], default="milestone")
    ap.add_argument("--horizon", type=int, default=3, help="nearfuture: fixed steps ahead in the 3Hz index")
    ap.add_argument("--graph", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--n_eps", type=int, default=120)
    ap.add_argument("--future_only", action="store_true", help="restrict model-match to frames >= current")
    ap.add_argument("--out", default="lmwm/outputs/milestone_lag", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device; args.out.mkdir(parents=True, exist_ok=True)

    proto = np.load(args.graph)["prototype_table"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    E, FR, Fn = load_index(args.feature_dir)
    ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
    cd, gmu, gsd, din = ck["code_dim"], ck["gmu"], ck["gsd"], ck.get("din", 1280)
    fwd = ForwardDec(din, cd).to(dev); fwd.load_state_dict(ck["fwd"]); fwd.eval()
    predm = PredM(din, cd).to(dev); predm.load_state_dict(ck["predm"]); predm.eval()
    vae_pm = None
    if "vae_pm" in ck:
        import torch.nn as nn
        vae_pm = nn.Linear(cd, 2 * cd).to(dev); vae_pm.load_state_dict(ck["vae_pm"]); vae_pm.eval()
    enc = load_encoder("dinov3-h", device=dev)
    print(f"predictor {Path(args.predictor).name} (head={ck.get('code_head','det')}); {len(proto)} milestones", flush=True)

    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps); eps = eps[:args.n_eps]
    ds_lags, md_lags = [], []
    for ei, e in enumerate(eps):
        fi = np.where(E == e)[0]; fi = fi[np.argsort(FR[fi])]
        if len(fi) < 12:
            continue
        fr = FR[fi]; Fq = Fn[fi]
        seq = (Fq @ protoL.T).argmax(1)
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        seg_med = [s + int((Fq[s:e2] @ protoL[int(seq[s])]).argmax()) for s, e2 in zip(st, en)]
        seg_of = np.zeros(len(seq), int)
        for i, (s, e2) in enumerate(zip(st, en)):
            seg_of[s:e2] = i
        enc_imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, fi, 256, 128)
        G = enc.encode_grid(enc_imgs).astype(np.float32)                     # (n,1280,16,16)
        Gz = torch.from_numpy(((G - gmu) / gsd).astype(np.float32))
        with torch.no_grad():
            pred = np.zeros_like(G)
            for b in range(0, len(fi), 128):
                gt = Gz[b:b + 128].to(dev); code = predm(gt)
                if vae_pm is not None:
                    code = vae_pm(code).chunk(2, -1)[0]
                pred[b:b + 128] = (fwd(gt, code).cpu().numpy() * gsd + gmu)
        Gf = G.reshape(len(fi), -1); Gf /= (np.linalg.norm(Gf, axis=1, keepdims=True) + 1e-8)
        Pf = pred.reshape(len(fi), -1); Pf /= (np.linalg.norm(Pf, axis=1, keepdims=True) + 1e-8)
        sims = Pf @ Gf.T                                                     # (n,n) predicted-vs-frame cos
        for j in range(len(fi)):
            if args.mode == "milestone":
                ni = seg_of[j] + 1
                if ni >= len(seg_med):
                    continue
                tgt = seg_med[ni]
            else:                                                # nearfuture: fixed horizon h steps ahead
                tgt = j + args.horizon
                if tgt >= len(fi):
                    continue
            ds_lags.append((fr[tgt] - fr[j]) / args.fps)
            row = sims[j].copy()
            if args.future_only:
                row[:j] = -1
            md_lags.append((fr[int(row.argmax())] - fr[j]) / args.fps)
        if (ei + 1) % 30 == 0:
            print(f"  {ei+1}/{len(eps)} eps", flush=True)
    ds = np.array(ds_lags); md = np.array(md_lags)
    res = {"predictor": Path(args.predictor).name, "mode": args.mode, "future_only": args.future_only, "n_frames": len(ds),
           "dataset_lag_s_mean": round(float(ds.mean()), 3), "dataset_lag_s_median": round(float(np.median(ds)), 3),
           "dataset_lag_s_std": round(float(ds.std()), 3),
           "model_lag_s_mean": round(float(md.mean()), 3), "model_lag_s_median": round(float(np.median(md)), 3),
           "model_lag_s_std": round(float(md.std()), 3),
           "model/dataset_ratio_mean": round(float(md.mean() / (ds.mean() + 1e-9)), 3)}
    (args.out / "lag.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    b = np.linspace(0, np.percentile(np.concatenate([ds, md]), 98), 40)
    ax[0].hist(ds, bins=b, alpha=0.6, label=f"dataset (real m+1) med {np.median(ds):.2f}s", color="#4477aa")
    ax[0].hist(md, bins=b, alpha=0.6, label=f"model (pred m+1) med {np.median(md):.2f}s", color="#ee6677")
    ax[0].set_xlabel("time lag to milestone+1 (s)"); ax[0].set_title("lag distribution"); ax[0].legend(fontsize=8)
    m = min(ds.max(), np.percentile(ds, 99))
    ax[1].plot([0, m], [0, m], "k:", lw=1); ax[1].scatter(ds, md, s=3, alpha=0.15, color="#aa3377")
    ax[1].set_xlabel("dataset lag (s)"); ax[1].set_ylabel("model lag (s)"); ax[1].set_title("per-frame: model vs dataset lag")
    ax[1].set_xlim(0, m); ax[1].set_ylim(0, m)
    fig.tight_layout(); fig.savefig(args.out / "lag.png", dpi=120); plt.close(fig)
    print(f"wrote {args.out}/lag.png", flush=True)


if __name__ == "__main__":
    main()
