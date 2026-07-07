#!/usr/bin/env python
"""Prediction time-lag of the NEW two-model milestone predictor (Stage-1 MDN + Stage-2 grounding,
π0.5 SigLIP space). Same protocol as measure_milestone_lag.py:

  DATASET lag = time(V3.1 Viterbi next-milestone medoid) - time(current)      [ground-truth horizon]
  MODEL   lag = time(frame whose SigLIP grid is nearest the PREDICTED m+1) - time(current)
  undershoot ratio = MODEL / DATASET   (<1 = hedges toward nearer futures)

Segmentation/target stay DINOv3-H + CRAVE Viterbi (offline label factory); prediction + matching are
in π0.5 SigLIP grid space. Predicted m+1 = Stage-2(G_t, Stage-1.deploy_mean(gist_t)).
Compare to the OLD DINOv3-H predictor: model lag 0.845s / dataset 2.43s / ratio 0.347.
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
from _siglip_bigvision import SiglipBigVision  # noqa: E402
from train_twomodel_v2 import MilestonePredictor, MilestoneGenerator, PI05_NPZ, PI05_NPZ_GF3
from train_lawm_patch import InverseEnc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm_ckpt", default="lmwm/checkpoints/twomodel_v2/milestone_viterbi_K4.pt")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--n_eps", type=int, default=120)
    ap.add_argument("--future_only", action="store_true", help="restrict model-match to frames >= current")
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)

    g = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = g["prototype_table"].astype(np.float32); pord = g["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    E, FR, Fn = load_index(args.feature_dir)
    tm = torch.load(args.tm_ckpt, map_location="cpu", weights_only=False)
    din, cd, gmu, gsd = tm["din"], tm["code_dim"], tm["gmu"], tm["gsd"]
    predm = MilestonePredictor(din, cd, tm["K"]).to(dev); predm.load_state_dict(tm["predm"]); predm.eval()
    if tm.get("fwd_arch") == "concat":                                     # ablation concat variant
        from train_lawm_patch import ForwardDec
        fwd = ForwardDec(din, cd).to(dev)
    else:
        fwd = MilestoneGenerator(din, cd).to(dev)
    fwd.load_state_dict(tm["fwd"]); fwd.eval()
    enc = SiglipBigVision(npz, device=dev)
    print(f"two-model V2 K={tm['K']}; measuring lag over {args.n_eps} eps", flush=True)

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
        enc_imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, fi, 224, 128)
        G = enc.encode_grid(enc_imgs, bs=32).astype(np.float32)              # (n,1152,16,16) SigLIP
        Gz = torch.from_numpy(((G - gmu) / gsd).astype(np.float32)).to(dev)
        with torch.no_grad():
            gist = Gz.mean((2, 3)); z = predm.deploy_mean(gist)
            pred = np.concatenate([(fwd(Gz[b:b + 128], z[b:b + 128]).cpu().numpy() * gsd + gmu)
                                   for b in range(0, len(fi), 128)])
        Gf = G.reshape(len(fi), -1); Gf /= (np.linalg.norm(Gf, axis=1, keepdims=True) + 1e-8)
        Pf = pred.reshape(len(fi), -1); Pf /= (np.linalg.norm(Pf, axis=1, keepdims=True) + 1e-8)
        if len(Pf) > 1:                                                       # temporal smoothness of the prediction
            pred_smooth.extend((Pf[:-1] * Pf[1:]).sum(1).tolist())           # adjacent-frame pred cos (high=smooth)
            real_smooth.extend((Gf[:-1] * Gf[1:]).sum(1).tolist())           # baseline: adjacent real-frame cos
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

    ds = np.array(ds_lags); md = np.array(md_lags)
    res = {"n_frames": int(len(ds)), "future_only": bool(args.future_only),
           "dataset_lag_s_mean": round(float(ds.mean()), 3), "dataset_lag_s_median": round(float(np.median(ds)), 3),
           "model_lag_s_mean": round(float(md.mean()), 3), "model_lag_s_median": round(float(np.median(md)), 3),
           "model_lag_p25": round(float(np.percentile(md, 25)), 3), "model_lag_p75": round(float(np.percentile(md, 75)), 3),
           "frac_lag_negative(<0)": round(float((md < 0).mean()), 3),      # predicts a PAST frame
           "frac_lag_zero_or_back(<=0)": round(float((md <= 0).mean()), 3),
           "frac_lag_forward(>0)": round(float((md > 0).mean()), 3),
           "undershoot_ratio_mean": round(float(md.mean() / (ds.mean() + 1e-9)), 3),
           "pred_smoothness_adj_cos": round(float(np.mean(pred_smooth)), 4),   # high = temporally smooth (no jumping)
           "real_frame_smoothness_adj_cos": round(float(np.mean(real_smooth)), 4),  # reference
           "old_dinov3_ref": {"model_lag_s": 0.845, "dataset_lag_s": 2.43, "ratio": 0.347}}
    (REPO / "lmwm/outputs/twomodel_v2_lag.json").write_text(json.dumps(res, indent=2))
    np.save(REPO / f"lmwm/outputs/lag_raw_{args.tag}.npy", md)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
