#!/usr/bin/env python
"""Color-augmented code predictor for appearance-invariant subgoal.

The predictor eats a color-JITTERED current frame (same state, randomized color) but
must output a code that -- via forward(ORIGINAL current, code) -- reconstructs the true
next. Randomizing the current color forces the code to ignore appearance -> generalizes
to unseen vis_base garment colors. Compared against the plain (no-aug) predictor.

forward is trained on full kai0 pooled; predictors on a frame subset (orig + jitters).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lmwm.data import split_indices  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from lever_patch_token import read_enc  # noqa: E402
from verify_transition_code import MLP, build_visbase  # noqa: E402


def color_jitter(imgs, rng):
    """imgs (N,256,256,3) uint8 RGB -> hue/sat/val jittered."""
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.integers(-40, 40)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.5, 1.5), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.8, 1.2), 0, 255)
        out[i] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--kai0_camera", default="observation.images.top_head")
    ap.add_argument("--visbase_root", default="kai0/data/Task_A/vis_base/v1", type=Path)
    ap.add_argument("--visbase_camera", default="observation.images.top_head")
    ap.add_argument("--n_frames", type=int, default=30000)
    ap.add_argument("--n_jit", type=int, default=2)
    ap.add_argument("--n_eps", type=int, default=120)
    ap.add_argument("--code_dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="lmwm/outputs/appearance_gen/color_aug.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    proto = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    z = np.load(args.pairs)
    pooled = z["current"][:, :1280].astype(np.float32)
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    n = len(z["current_milestone"]); tri, _ = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    tri = tri.numpy(); tri = tri[ok[tri]]
    Xp = torch.from_numpy(pooled); Md = torch.from_numpy(med)

    # ---- forward/inverse on full kai0 pooled ----
    inv = MLP(2560, args.code_dim).to(dev); fwd = MLP(1280 + args.code_dim, 1280, l2=True).to(dev)
    o1 = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=5e-4, weight_decay=1e-5)
    print("forward/inverse on full kai0 ...", flush=True)
    for s in range(args.steps):
        bi = tri[np.random.randint(0, len(tri), 1024)]
        cur = Xp[bi].to(dev); nxt = Md[bi].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = (1 - (fwd(torch.cat([cur, inv(torch.cat([cur, nxt], -1))], -1)) * nxt).sum(-1)).mean()
        o1.zero_grad(); l.backward(); o1.step()
    inv.eval(); fwd.eval()
    for p in fwd.parameters():
        p.requires_grad_(False)

    # ---- read a frame subset, encode original + color-jittered latents ----
    rng = np.random.default_rng(0)
    sub = rng.choice(tri, min(args.n_frames, len(tri)), replace=False)
    print(f"reading {len(sub)} current frames ...", flush=True)
    imgs = read_enc(args.dataset_root, args.kai0_camera, z["episode_id"][sub], z["t"][sub], 256)
    enc = load_encoder("dinov3-h", device=str(dev))
    def enc_l2(a): x = enc.encode_pooled(a).astype(np.float32); return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    lat_orig = enc_l2(imgs)
    lat_jit = [enc_l2(color_jitter(imgs, rng)) for _ in range(args.n_jit)]
    med_sub = med[sub]
    LO = torch.from_numpy(lat_orig); LJ = [torch.from_numpy(x) for x in lat_jit]; MS = torch.from_numpy(med_sub)
    ns = len(sub)

    def train_pred(augment):
        p = MLP(1280, args.code_dim).to(dev)
        opt = torch.optim.AdamW(p.parameters(), lr=1e-3, weight_decay=1e-5)
        for s in range(args.steps):
            bi = np.random.randint(0, ns, 1024)
            orig = LO[bi].to(dev); nxt = MS[bi].to(dev)
            inp = (LJ[np.random.randint(0, args.n_jit)][bi].to(dev) if augment else orig)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = fwd(torch.cat([orig, p(inp)], -1)); loss = (1 - (rec * nxt).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        p.eval(); return p

    print("training predictors (plain vs color-aug) ...", flush=True)
    p_plain = train_pred(False)
    p_aug = train_pred(True)

    print("building vis_base ...", flush=True)
    vc, vm, vcm, vnm = build_visbase(args.visbase_root, args.visbase_camera, proto, enc, args.n_eps, 10)
    print(f"vis_base: {len(vc)} pairs", flush=True)

    def ev(cur_np, med_np, tag):
        cur = torch.from_numpy(cur_np).to(dev); nxt = med_np
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            orc = fwd(torch.cat([cur, inv(torch.cat([cur, torch.from_numpy(med_np).to(dev)], -1))], -1)).float().cpu().numpy()
            pl = fwd(torch.cat([cur, p_plain(cur)], -1)).float().cpu().numpy()
            au = fwd(torch.cat([cur, p_aug(cur)], -1)).float().cpu().numpy()
        return {tag: {"forward_oracle": round(float((orc * nxt).sum(1).mean()), 4),
                      "predicted_plain": round(float((pl * nxt).sum(1).mean()), 4),
                      "predicted_color_aug": round(float((au * nxt).sum(1).mean()), 4), "n": len(cur_np)}}

    res = {}
    res.update(ev(pooled[sub[:6000]], med_sub[:6000], "kai0_in_distribution"))
    res.update(ev(vc, vm, "visbase_UNSEEN_appearance"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
