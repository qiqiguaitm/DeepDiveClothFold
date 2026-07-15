#!/usr/bin/env python
"""Prediction time-lag (reach) of the UNIFIED DINOv3-base LMWM (train_multitask --encoder dinov3base).

Same protocol as measure_twomodel_v2_lag.py, but everything in DINOv3-base grid space:
  DATASET lag = time(Viterbi next-milestone medoid) - time(current)                 [GT horizon]
  MODEL   lag = time(frame whose DINOv3-base grid is nearest the PREDICTED m+1) - time(current)
  undershoot ratio = MODEL / DATASET

Predicted m+1 = generator fwd(G_t, predm.deploy_mean(gist_t)). Grids are per-patch L2 then
standardized by (gmu,gsd) (as trained). Run in the CRAVE env:
    /home/tim/miniconda3/envs/srpo/bin/python
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from crave.utils.dp import viterbi_forward  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from train_twomodel_v2 import MilestonePredictor, MilestoneGenerator  # noqa: E402


def l2c(x):  # per-patch L2 over channel axis=1 of (N,768,16,16)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lmwm/checkpoints/dinov3base_lmwm_kaicoffee.pt")
    ap.add_argument("--graph", default="lmwm/data/recurrence_graphs/kai0base_dinov3base/recurrence_graph.npz")
    ap.add_argument("--feature_dir", default="crave/data/kai_dinov3base", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--n_eps", type=int, default=120)
    ap.add_argument("--future_only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    g = np.load(REPO / args.graph)
    proto = g["prototype_table"].astype(np.float32); pord = g["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    E, FR, Fn = load_index(REPO / args.feature_dir if not Path(args.feature_dir).is_absolute() else args.feature_dir)
    ck = torch.load(REPO / args.ckpt if not Path(args.ckpt).is_absolute() else args.ckpt, map_location="cpu", weights_only=False)
    din, cd, gmu, gsd = ck["din"], ck["code_dim"], ck["gmu"], ck["gsd"]
    predm = MilestonePredictor(din, cd, ck["K"]).to(dev); predm.load_state_dict(ck["predm"]); predm.eval()
    fwd = MilestoneGenerator(din, cd).to(dev); fwd.load_state_dict(ck["fwd"]); fwd.eval()
    enc = load_encoder("dinov3-base", device=dev)
    print(f"unified DINOv3-base LMWM K={ck['K']} din={din}; measuring reach over {args.n_eps} eps", flush=True)

    droot = REPO / args.dataset_root if not Path(args.dataset_root).is_absolute() else args.dataset_root
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps); eps = eps[:args.n_eps]
    ds_lags, md_lags, pred_smooth, real_smooth = [], [], [], []
    for ei, e in enumerate(eps):
        fi = np.where(E == e)[0]; fi = fi[np.argsort(FR[fi])]
        if len(fi) < 12:
            continue
        fr = FR[fi]; Fq = Fn[fi]
        ms_v = viterbi_forward(np.linalg.norm(Fq[:, None] - protoL[None], axis=2), pord, up=3.0, down=25.0, hard_start=True)
        chv = np.where(np.diff(ms_v) != 0)[0] + 1
        stv = np.concatenate([[0], chv]); env = np.concatenate([chv, [len(ms_v)]])
        vseg_med = [s + int((Fq[s:e2] @ protoL[int(ms_v[s])]).argmax()) for s, e2 in zip(stv, env)]
        vseg_of = np.zeros(len(ms_v), int)
        for i, (s, e2) in enumerate(zip(stv, env)):
            vseg_of[s:e2] = i
        enc_imgs, _ = read_imgs(droot, args.camera, E, FR, fi, 224, 128)
        G = l2c(enc.encode_grid(enc_imgs, bs=64).astype(np.float32))          # (n,768,16,16) DINOv3-base, per-patch L2
        Gz = torch.from_numpy(((G - gmu) / gsd).astype(np.float32)).to(dev)
        with torch.no_grad():
            gist = Gz.mean((2, 3)); z = predm.deploy_mean(gist)
            pred = np.concatenate([l2c(fwd(Gz[b:b + 128], z[b:b + 128]).cpu().numpy() * gsd + gmu)
                                   for b in range(0, len(fi), 128)])
        Gf = G.reshape(len(fi), -1); Gf /= (np.linalg.norm(Gf, axis=1, keepdims=True) + 1e-8)
        Pf = pred.reshape(len(fi), -1); Pf /= (np.linalg.norm(Pf, axis=1, keepdims=True) + 1e-8)
        if len(Pf) > 1:
            pred_smooth.extend((Pf[:-1] * Pf[1:]).sum(1).tolist())
            real_smooth.extend((Gf[:-1] * Gf[1:]).sum(1).tolist())
        sims = Pf @ Gf.T
        for j in range(len(fi)):
            ni = vseg_of[j] + 1
            if ni >= len(vseg_med):
                continue
            ds_lags.append((fr[vseg_med[ni]] - fr[j]) / args.fps)
            row = sims[j].copy()
            if args.future_only:
                row[:j] = -1
            md_lags.append((fr[int(row.argmax())] - fr[j]) / args.fps)
        if (ei + 1) % 30 == 0:
            print(f"  {ei+1}/{len(eps)} eps", flush=True)
    enc.unload()
    ds = np.array(ds_lags); md = np.array(md_lags)
    res = {"model": "unified DINOv3-base LMWM", "n_frames": int(len(ds)), "n_eps": int(args.n_eps),
           "future_only": bool(args.future_only),
           "dataset_lag_s_mean": round(float(ds.mean()), 3), "dataset_lag_s_median": round(float(np.median(ds)), 3),
           "model_lag_s_mean": round(float(md.mean()), 3), "model_lag_s_median": round(float(np.median(md)), 3),
           "frac_lag_negative(<0)": round(float((md < 0).mean()), 3),
           "frac_lag_forward(>0)": round(float((md > 0).mean()), 3),
           "undershoot_ratio_mean": round(float(md.mean() / (ds.mean() + 1e-9)), 3),
           "pred_smoothness_adj_cos": round(float(np.mean(pred_smooth)), 4),
           "real_frame_smoothness_adj_cos": round(float(np.mean(real_smooth)), 4)}
    out = REPO / "lmwm/outputs/dinov3base_lag.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
