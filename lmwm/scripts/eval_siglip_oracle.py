#!/usr/bin/env python
"""Experiment A: oracle/deploy grid-cos of the milestone_viterbi predictor when the TARGET SPACE is
SigLIP (VLA-family) instead of DINOv3-H. Decides "decision point A": is predicting in the VLA's own
feature space (trivial fusion, KV-native) good enough vs DINOv3-H (richer, but cross-space bridge)?

Same pairs (milestone_viterbi), same LAM (build_lam), ONLY the grid features change. Directly
comparable to optimize_subgoal's DINOv3-H numbers (oracle ~0.789 / persistence).

PROXY CAVEAT: uses on-disk google/siglip2-so400m-patch14-384 (SigLIP2 family), NOT the exact
SigLIP-so400m@224 tower π0.5 fine-tunes. Faithful tower needs the pt_224.npz JAX->PyTorch conversion.
This is a directional read of "SigLIP-family features as a milestone-prediction target space".
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
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from lam_arch import build_lam, nparams  # noqa: E402
from optimize_subgoal import build_pairs, PredM  # noqa: E402  (build_pairs is the target constructor)

SIGLIP_DIRS = {
    "siglip2-so400m": "/vePFS/tim/workspace/hf_cache/hub_default/models--google--siglip2-so400m-patch14-384",
}


def find_snapshot(hub_dir):
    snaps = sorted((Path(hub_dir) / "snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"no snapshot under {hub_dir}")
    return str(snaps[-1])


class SiglipGrid:
    """SigLIP vision tower -> (N, dim, P, P) patch grid, L2-safe. res controls token count
    (224 -> 16x16 like paligemma; 384 -> 27x27 native)."""
    def __init__(self, hub_dir, device, res=224):
        from transformers import SiglipVisionModel
        path = find_snapshot(hub_dir)
        self.m = SiglipVisionModel.from_pretrained(path, torch_dtype=torch.float32).to(device).eval()
        self.dev = device; self.res = res
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_grid(self, imgs_u8, bs=32):
        out = []
        for s in range(0, len(imgs_u8), bs):
            x = torch.from_numpy(imgs_u8[s:s + bs]).to(self.dev).float().permute(0, 3, 1, 2) / 255.0
            if x.shape[-1] != self.res:
                x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            h = self.m(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state  # (B, P, D)
            P = int(round(h.shape[1] ** 0.5))
            g = h.permute(0, 2, 1).reshape(h.shape[0], h.shape[2], P, P)
            out.append(g.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="siglip2-so400m", choices=list(SIGLIP_DIRS))
    ap.add_argument("--res", type=int, default=224, help="224->16x16 (paligemma-like), 384->27x27 native")
    ap.add_argument("--code_dim", type=int, default=128)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_val", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    E, FR, Fn = load_index(args.feature_dir)
    rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = rg["prototype_table"].astype(np.float32); pord = rg["pord"].astype(np.float32)
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())

    tr, va = build_pairs(E, FR, Fn, proto, "milestone_viterbi", 5, val_eps, args.seed, pord=pord)
    tr = tr[:args.n_train]; va = va[:args.n_val]
    uniq = sorted(set([g for p in tr + va for g in p])); u2k = {g: k for k, g in enumerate(uniq)}
    print(f"[siglip:{args.encoder}@{args.res}] {len(tr)} train + {len(va)} val, {len(uniq)} frames", flush=True)

    enc_imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, np.array(uniq), args.res, 128)
    enc = SiglipGrid(SIGLIP_DIRS[args.encoder], dev, res=args.res)
    grids = enc.encode_grid(enc_imgs); din = grids.shape[1]
    print(f"[siglip] grid {grids.shape} din={din}", flush=True)
    gmu, gsd = grids.mean(), grids.std() + 1e-6
    GZ = torch.from_numpy(((grids - gmu) / gsd).astype(np.float32))

    tra = np.array([u2k[c] for c, _ in tr]); trb = np.array([u2k[n] for _, n in tr])
    vaa = np.array([u2k[c] for c, _ in va]); vab = np.array([u2k[n] for _, n in va])

    inv, fwd, predm = build_lam("cnn", din, args.code_dim, 512, 8)
    inv, fwd, predm = inv.to(dev), fwd.to(dev), predm.to(dev)
    o1 = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(list(predm.parameters()), lr=2e-4, weight_decay=1e-5)
    print(f"[siglip] deploy params {(nparams(predm)+nparams(fwd))/1e6:.1f}M; training ...", flush=True)
    for step in range(args.steps):
        sel = torch.randint(0, len(tra), (32,))
        gt = GZ[tra[sel]].to(dev); gf = GZ[trb[sel]].to(dev)
        l1 = F.smooth_l1_loss(fwd(gt, inv(gt, gf)), gf, beta=1.0)
        o1.zero_grad(); l1.backward(); o1.step()
        l2 = F.smooth_l1_loss(fwd(gt, predm(gt)), gf, beta=1.0)
        o2.zero_grad(); l2.backward(); o2.step()
    inv.eval(); fwd.eval(); predm.eval()

    def cos(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    co, cd_, cp = [], [], []
    with torch.no_grad():
        for s in range(0, len(vaa), 256):
            gt = GZ[vaa[s:s + 256]].to(dev); gf = GZ[vab[s:s + 256]].to(dev); gtr = f(gf)
            co.append(cos(f(fwd(gt, inv(gt, gf))), gtr))
            cd_.append(cos(f(fwd(gt, predm(gt))), gtr))
            cp.append(cos(f(gt), gtr))
    res = {"encoder": args.encoder, "res": args.res, "grid_dim": int(din),
           "tokens": int(grids.shape[2] * grids.shape[3]), "code_dim": args.code_dim,
           "oracle_grid_cos": round(float(np.concatenate(co).mean()), 4),
           "deploy_grid_cos": round(float(np.concatenate(cd_).mean()), 4),
           "persistence_grid_cos": round(float(np.concatenate(cp).mean()), 4),
           "lift": round(float(np.concatenate(cd_).mean() - np.concatenate(cp).mean()), 4),
           "note": "PROXY siglip2 (not pi0.5 tuned tower); compare to DINOv3-H oracle 0.789/deploy 0.694"}
    outp = REPO / f"lmwm/outputs/siglip_oracle_{args.encoder}_{args.res}.json"
    outp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
