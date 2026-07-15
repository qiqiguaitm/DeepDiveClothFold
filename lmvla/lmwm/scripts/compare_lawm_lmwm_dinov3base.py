#!/usr/bin/env python
"""Same-space, same-data, same-metric head-to-head: LaWM-style vs LMWM in DINOv3-base.

The official LaWM LAM and our unified LMWM BOTH encode with facebook/dinov3-vitb16-pretrain-lvd1689m
(768D) -- so the old "different encoder space" caveat is GONE. This trains a faithful LaWM-style
baseline (inverse-dynamics latent-action code + FIXED-HORIZON future target) in the SAME DINOv3-base
space on kai0, and evaluates it with the IDENTICAL metric formulas as train_multitask.py (the LMWM
generator eval). Only the METHOD differs:

    LaWM-style : code = inverse(g_t, g_future) ; target = frame at fixed +1.6s
    LMWM (ours): code = proto teacher (milestone center) ; target = value-monotone milestone+1

Metrics (per train_multitask.py lines 308-330), each model vs ITS OWN native target:
    oracle       = cos(fwd(Gc, inverse(Gc,Gf)), Gf)     teacher-forced (uses future)
    deploy       = cos(fwd(Gc, predm(g_t)),     Gf)     deployment (predm predicts code, no future)
    persistence  = cos(Gc,                      Gf)     copy-current baseline
    lift         = deploy - persistence                 <-- the fair, horizon-normalized number
    value_fwd    = frac(prog(nearest-ms of pred subgoal) > prog(nearest-ms of current))
    reach        = model_lag / dataset_lag (undershoot); dataset_lag = 1.6s by LaWM design

Run in the CRAVE env:  /home/tim/miniconda3/envs/srpo/bin/python
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import InverseEnc, ForwardDec, load_index, read_imgs  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


class PredM(torch.nn.Module):
    """gist g_t (768) -> code (code_dim). Deploy-time predictor (no future frame)."""
    def __init__(self, d, code_dim, hid=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, hid), torch.nn.GELU(),
            torch.nn.Linear(hid, hid), torch.nn.GELU(),
            torch.nn.Linear(hid, code_dim), torch.nn.LayerNorm(code_dim))

    def forward(self, g):
        return self.net(g)


def l2c(x):  # per-patch L2 over channel axis=1 of (N,768,16,16)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="crave/data/kai_dinov3base", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--graph", default="lmwm/data/recurrence_graphs/kai0base_dinov3base/recurrence_graph.npz")
    ap.add_argument("--horizon_s", type=float, default=1.6)   # LaWM fixed design horizon
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--code_dim", type=int, default=32)       # official LaWM code_dim
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--lift_w", type=float, default=1.0)      # same generator training as LMWM
    ap.add_argument("--reach_eps", type=int, default=30)      # episodes for reach pass
    ap.add_argument("--reach_stride", type=int, default=3)    # frame subsample for reach (10fps)
    ap.add_argument("--tag", default="lawm_style_kai0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="lmwm/outputs/compare_lawm_lmwm_kai0.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_val, args.steps, args.reach_eps = 800, 400, 800, 6
    dev = "cuda"
    H = int(round(args.horizon_s * args.fps))

    E, FR, Fn = load_index(REPO / args.feature_dir)             # Fn = pooled L2 feats (gist), per frame
    g = np.load(REPO / args.graph)
    proto = g["prototype_table"].astype(np.float32); pord = g["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    # ---- fixed-horizon pairs (LaWM native target): (frame t, frame t+H clamped to ep end) ----
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    tr_pairs, va_pairs, reach_by_ep = [], [], {}
    for ep in eps:
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        if len(order) < 2:
            continue
        for i in range(len(order)):
            j = min(i + H, len(order) - 1)
            if j == i:
                continue
            (va_pairs if ep in val_eps else tr_pairs).append((int(order[i]), int(order[j])))
        if ep in val_eps and len(reach_by_ep) < args.reach_eps:
            reach_by_ep[int(ep)] = order[::args.reach_stride]      # dense frames for reach
    rng.shuffle(tr_pairs); rng.shuffle(va_pairs)
    tr_pairs = tr_pairs[:args.n_train]; va_pairs = va_pairs[:args.n_val]
    print(f"H={H} frames ({args.horizon_s}s) | {len(tr_pairs)} train + {len(va_pairs)} val pairs "
          f"| {len(reach_by_ep)} reach eps", flush=True)

    # ---- encode grids for all needed frames (dinov3-base, per-patch L2 + standardize) ----
    reach_gidx = sorted({int(x) for arr in reach_by_ep.values() for x in arr})
    uniq = sorted(set([gi for p in tr_pairs + va_pairs for gi in p]) | set(reach_gidx))
    u2k = {gi: k for k, gi in enumerate(uniq)}
    print(f"encoding {len(uniq)} unique frames ...", flush=True)
    enc_imgs, _ = read_imgs(REPO / args.dataset_root, args.camera, E, FR, np.array(uniq), 256, 128)
    enc = load_encoder("dinov3-base", device=dev)
    grids = enc.encode_grid(enc_imgs).astype(np.float32)          # (U,768,16,16)
    grids = l2c(grids)                                            # per-patch L2 (match train_multitask)
    din = grids.shape[1]
    gmu, gsd = float(grids.mean(dtype=np.float64)), float(grids.std(dtype=np.float64) + 1e-6)
    gz = ((grids - gmu) / gsd).astype(np.float32)
    gist = grids.mean((2, 3)); gist = gist / (np.linalg.norm(gist, axis=1, keepdims=True) + 1e-8)  # pooled
    GZ = torch.from_numpy(gz)                                     # CPU, move batches
    GI = torch.from_numpy(gist.astype(np.float32))

    def idx_ab(pairs):
        return (np.array([u2k[c] for c, _ in pairs]), np.array([u2k[n] for _, n in pairs]))
    tra, trb = idx_ab(tr_pairs); vaa, vab = idx_ab(va_pairs)
    tra_t, trb_t = torch.from_numpy(tra), torch.from_numpy(trb)

    # ---- train LaWM-style inverse/forward (+ lift reg, same as LMWM generator) ----
    inv = InverseEnc(din, args.code_dim).to(dev); fwd = ForwardDec(din, args.code_dim).to(dev)
    opt = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=2e-4, weight_decay=1e-5)
    cosr = lambda a, b: F.cosine_similarity(a, b, dim=1)

    def batch(a, b, bs):
        sel = torch.randint(0, len(a), (bs,))
        return GZ[a[sel]].to(dev), GZ[b[sel]].to(dev)

    print("training LaWM-style inverse/forward ...", flush=True)
    for step in range(args.steps):
        gt, gf = batch(tra_t, trb_t, 32)
        u = inv(gt, gf); gh = fwd(gt, u)
        lift = torch.relu(cosr(gh.flatten(1), gt.flatten(1)) - cosr(gh.flatten(1), gf.flatten(1))).mean()
        loss = F.smooth_l1_loss(gh, gf) + args.lift_w * lift
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  step {step+1}/{args.steps} loss {loss.item():.4f}", flush=True)

    # ---- train deploy-predm: predm(gist_t) -> code ; smooth_l1(fwd(gt, code), gf) with inv/fwd frozen ----
    for p in list(inv.parameters()) + list(fwd.parameters()):
        p.requires_grad_(False)
    predm = PredM(din, args.code_dim).to(dev)
    optp = torch.optim.AdamW(predm.parameters(), lr=3e-4, weight_decay=1e-5)
    print("training deploy-predm ...", flush=True)
    for step in range(args.steps):
        sel = torch.randint(0, len(tra_t), (64,))
        gt = GZ[tra_t[sel]].to(dev); gf = GZ[trb_t[sel]].to(dev); gi = GI[tra_t[sel]].to(dev)
        gh = fwd(gt, predm(gi))
        loss = F.smooth_l1_loss(gh, gf) + args.lift_w * torch.relu(
            cosr(gh.flatten(1), gt.flatten(1)) - cosr(gh.flatten(1), gf.flatten(1))).mean()
        optp.zero_grad(); loss.backward(); optp.step()
        if (step + 1) % 2000 == 0:
            print(f"  predm step {step+1}/{args.steps} loss {loss.item():.4f}", flush=True)

    # ---- eval on val (IDENTICAL formulas to train_multitask.py) ----
    inv.eval(); fwd.eval(); predm.eval()
    f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    cn = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    co, cd_, cp, idpred, curpool = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, len(vaa), 256):
            Gc = GZ[vaa[s:s+256]].float().to(dev); Gf = GZ[vab[s:s+256]].float().to(dev)
            gi = GI[vaa[s:s+256]].to(dev); gtr = f(Gf)
            zdep = predm(gi)
            co.append(cn(f(fwd(Gc, inv(Gc, Gf))), gtr))       # oracle (teacher-forced)
            ghd = fwd(Gc, zdep)
            cd_.append(cn(f(ghd), gtr)); cp.append(cn(f(Gc), gtr))
            idpred.append(ghd.mean((2, 3)).cpu().numpy())
            curpool.append(Gc.mean((2, 3)).cpu().numpy())
    idpred = np.concatenate(idpred); idpred /= (np.linalg.norm(idpred, axis=1, keepdims=True) + 1e-8)
    curpool = np.concatenate(curpool); curpool /= (np.linalg.norm(curpool, axis=1, keepdims=True) + 1e-8)
    pred_ms = (idpred @ protoL.T).argmax(1); cur_ms = (curpool @ protoL.T).argmax(1)
    value_fwd = float((pord[pred_ms] > pord[cur_ms]).mean())
    deploy = float(np.concatenate(cd_).mean()); persist = float(np.concatenate(cp).mean())
    oracle = float(np.concatenate(co).mean())

    # ---- reach / undershoot (model_lag vs fixed 1.6s dataset_lag) ----
    lags = []
    with torch.no_grad():
        for ep, arr in reach_by_ep.items():
            ks = [u2k[int(x)] for x in arr]
            Gall = GZ[ks].float().to(dev)                       # (T,768,16,16) standardized
            Graw = (Gall.cpu().numpy() * gsd + gmu).reshape(len(ks), -1)   # un-std, flat
            Graw = Graw / (np.linalg.norm(Graw, axis=1, keepdims=True) + 1e-8)
            gi = GI[ks].to(dev)
            pred = f(fwd(Gall, predm(gi)))                      # (T, flat) un-std
            pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
            frn = FR[np.array([int(x) for x in arr])].astype(np.float64)
            for i in range(len(ks)):
                j = int((pred[i:i+1] @ Graw.T).argmax())        # nearest real frame to prediction
                lags.append((frn[j] - frn[i]) / args.fps)
    lags = np.array(lags)
    model_lag = float(np.median(lags)); model_lag_mean = float(lags.mean())
    dataset_lag = args.horizon_s
    undershoot = float(model_lag_mean / dataset_lag)

    res = {
        "tag": args.tag, "space": "dinov3-base (facebook/dinov3-vitb16-pretrain-lvd1689m, 768D)",
        "method": "LaWM-style: inverse-dynamics code + FIXED-HORIZON future target",
        "horizon_s": args.horizon_s, "code_dim": args.code_dim, "H_frames": H,
        "n_train": len(tr_pairs), "n_val": len(va_pairs),
        "params_M": round(sum(p.numel() for p in list(inv.parameters()) + list(fwd.parameters())
                              + list(predm.parameters())) / 1e6, 2),
        "kai0": {
            "oracle": round(oracle, 4), "deploy": round(deploy, 4),
            "persistence": round(persist, 4), "lift": round(deploy - persist, 4),
            "value_forward_frac": round(value_fwd, 4),
            "reach_model_lag_median_s": round(model_lag, 4),
            "reach_model_lag_mean_s": round(model_lag_mean, 4),
            "reach_dataset_lag_s": dataset_lag, "reach_undershoot": round(undershoot, 4),
        },
        "lmwm_reference_same_space": {   # from lmwm/outputs/multitask/dinov3base_lmwm_sharedpca_kaicoffee.json
            "note": "LMWM (proto teacher + milestone+1), SAME dinov3-base space, SAME metric code",
            "deploy": 0.8653, "persistence": 0.781, "lift": 0.0843,
            "id_top3": 0.886, "value_forward_frac": 0.744,
            "reach_model_lag_s": 0.811, "reach_undershoot": 0.26, "params_M": 34,
        },
    }
    outp = REPO / args.out; outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
