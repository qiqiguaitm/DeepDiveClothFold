#!/usr/bin/env python
"""LaWM baseline prediction time-lag: build a deploy-predm on the FROZEN official LaWM LAM (RLinf/LaWAM
jialei02/lawam_lam, ViT-B, code_dim 32 VAE), then measure the same lag protocol as our v2.

LaWM LAM has no native predm (deploy code offloaded to VLA). We add one, mirroring our optimize_subgoal
l2: predm(dec_in) -> z ; train smooth_l1(decoder(dec_in, z), tgt) with LAM frozen. Then:
  MODEL lag = time(frame whose ViT-B feat is nearest the predicted future feat) - time(current)
  DATASET lag = LaWM's FIXED horizon (1.6s by design)  -> undershoot ratio = model/dataset
⚠️ absolute seconds NOT comparable to our milestone horizon (2.8s); compare the RATIO (达成度).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
sys.path.insert(0, str(REPO / "lmwm/vendor/LaWAM"))
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from eval_lawm_lam import _stub_heavy_deps, _patch_automodel, load_lam, imagenet  # noqa: E402


class LawmPredM(nn.Module):
    """dec_in (B,1,K,D) -> code (B,1,code_dim). CNN over the K=16x16 patch grid."""
    def __init__(self, D, code_dim, hid=256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(D, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(),
                                  nn.Conv2d(hid, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU())
        self.head = nn.Linear(hid, code_dim); self.ln = nn.LayerNorm(code_dim); self.D = D

    def forward(self, dec_in):
        B, _, K, D = dec_in.shape; P = int(round(K ** 0.5))
        x = dec_in.squeeze(1).transpose(1, 2).reshape(B, D, P, P)
        return self.ln(self.head(self.conv(x).mean((2, 3)))).unsqueeze(1)     # (B,1,code_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lmwm/vendor/LaWAM/ckpts_dl/checkpoints/pytorch_model.pt")
    ap.add_argument("--yaml", default="lmwm/vendor/LaWAM/ckpts_dl/dino_large_vae.yaml")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--horizon_s", type=float, default=1.6)                   # LaWM design horizon
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--n_train_pairs", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n_eps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    _stub_heavy_deps(); _patch_automodel()
    lam = load_lam(args.ckpt, args.yaml, dev)
    for p in lam.parameters():
        p.requires_grad_(False)
    gap = int(round(args.horizon_s * args.fps))
    E, FR, _ = load_index(args.feature_dir)

    @torch.no_grad()
    def encode(frames_u8):                                                   # (N,256,256,3)u8 -> dec_in (N,1,K,D)
        X = imagenet(frames_u8, dev)
        vid = torch.stack([X, X], dim=1)                                     # [f,f]
        return lam.get_latent_action(videos=vid, states=None, dec_videos=vid, predict_future_frame=False)["dec_in"]

    # ---- collect training pairs (current, future@1.6s) across episodes ----
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    cur_g, fut_g = [], []
    for ep in eps:
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]; fr = FR[order]
        for i in range(len(order)):
            j = int(np.argmin(np.abs(fr - (fr[i] + gap))))
            if j > i and abs(fr[j] - fr[i] - gap) <= gap // 2:
                cur_g.append(int(order[i])); fut_g.append(int(order[j]))
        if len(cur_g) > args.n_train_pairs * 3:
            break
    idx = rng.permutation(len(cur_g))[:args.n_train_pairs]
    cur_g = np.array(cur_g)[idx]; fut_g = np.array(fut_g)[idx]
    uniq = np.array(sorted(set(cur_g.tolist() + fut_g.tolist()))); u2k = {g: k for k, g in enumerate(uniq)}
    imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, uniq, 256, 128)
    print(f"encoding {len(uniq)} frames for {len(cur_g)} train pairs ...", flush=True)
    feats = []
    for s in range(0, len(uniq), 64):
        feats.append(encode(imgs[s:s + 64]).cpu())
    feats = torch.cat(feats)                                                 # (U,1,K,D)
    D = feats.shape[-1]; cdim = lam.vq.code_dim if hasattr(lam, "vq") and hasattr(lam.vq, "code_dim") else 32
    ca = np.array([u2k[c] for c in cur_g]); fa = np.array([u2k[f] for f in fut_g])

    # ---- train deploy-predm (LAM frozen), reconstruction like our l2 ----
    predm = LawmPredM(D, cdim).to(dev)
    opt = torch.optim.AdamW(predm.parameters(), lr=2e-4, weight_decay=1e-5)
    print(f"training deploy-predm (code_dim={cdim}) ...", flush=True)
    for step in range(args.steps):
        sel = np.random.randint(0, len(ca), 16)
        din = feats[ca[sel]].to(dev); tgt = feats[fa[sel]].to(dev)
        z = predm(din)
        recon = lam.decoder(features=din, actions=z)
        loss = F.smooth_l1_loss(recon, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    predm.eval()

    # ---- lag over episodes: predicted feat -> nearest frame feat ----
    def cflat(t): x = t.reshape(t.shape[0], -1); return x / (x.norm(dim=1, keepdim=True) + 1e-8)
    ds_lags, md_lags = [], []
    for ep in eps[:args.n_eps]:
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]; fr = FR[order]
        if len(order) < 12:
            continue
        fimgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, order, 256, 128)
        df = []
        for s in range(0, len(order), 64):
            df.append(encode(fimgs[s:s + 64]))
        df = torch.cat(df)                                                   # (n,1,K,D)
        with torch.no_grad():
            pred = lam.decoder(features=df, actions=predm(df))              # (n,1,K,D)
        Ff = cflat(df).cpu().numpy(); Pf = cflat(pred).cpu().numpy()
        sims = Pf @ Ff.T
        for i in range(len(order)):
            j = int(np.argmin(np.abs(fr - (fr[i] + gap))))
            if not (j > i and abs(fr[j] - fr[i] - gap) <= gap // 2):
                continue
            ds_lags.append((fr[j] - fr[i]) / args.fps)
            md_lags.append((fr[int(sims[i].argmax())] - fr[i]) / args.fps)
    ds = np.array(ds_lags); md = np.array(md_lags)
    res = {"model": "LaWM (jialei02/lawam_lam, ViT-B, code32 VAE) + our deploy-predm",
           "horizon_design_s": args.horizon_s, "n_frames": int(len(ds)),
           "dataset_lag_s_mean": round(float(ds.mean()), 3),
           "model_lag_s_mean": round(float(md.mean()), 3), "model_lag_s_median": round(float(np.median(md)), 3),
           "frac_lag_forward(>0)": round(float((md > 0).mean()), 3),
           "frac_lag_negative(<0)": round(float((md < 0).mean()), 3),
           "undershoot_ratio_mean": round(float(md.mean() / (ds.mean() + 1e-9)), 3),
           "ours_v2_ccenter_ref": {"model_lag_s": 1.277, "dataset_lag_s": 2.787, "ratio": 0.458}}
    (REPO / "lmwm/outputs/lawm_lag.json").write_text(json.dumps(res, indent=2))
    np.save(REPO / "lmwm/outputs/lag_raw_lawm.npy", md)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
