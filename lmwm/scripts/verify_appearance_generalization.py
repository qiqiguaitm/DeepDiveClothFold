#!/usr/bin/env python
"""Cross-appearance generalization: does forward(current, code) beat absolute-medoid
prediction on HELD-OUT garment appearances?

Hypothesis: predicting an ABSOLUTE next-medoid latent bakes in the training garments'
appearance -> on an unseen-color garment the predicted subgoal drifts to a training
color (low cos). A transition CODE (appearance-invariant) + forward(current, code)
anchors the subgoal to the CURRENT observation -> inherits the unseen color -> stays high.

Split episodes by garment color (KMeans on per-episode non-white mean color); hold out
one color cluster entirely from training; evaluate subgoal cos on it.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lmwm.data import split_indices  # noqa: E402
from lever_patch_token import read_enc  # noqa: E402


def episode_colors(z, dataset_root, camera, eps):
    """One frame per episode -> mean color of non-near-white pixels (garment tint)."""
    rows = []
    for e in eps:
        idx = np.where(z["episode_id"] == e)[0]
        rows.append(idx[len(idx) // 2])
    rows = np.array(rows)
    imgs = read_enc(dataset_root, camera, z["episode_id"][rows], z["t"][rows], 128)
    cols = np.zeros((len(eps), 3), np.float32)
    for i, im in enumerate(imgs):
        flat = im.reshape(-1, 3).astype(np.float32)
        nonwhite = flat[flat.min(1) < 190]                       # drop near-white background
        cols[i] = nonwhite.mean(0) if len(nonwhite) else flat.mean(0)
    return cols


class MLP(nn.Module):
    def __init__(self, din, dout, hid=512, l2=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, dout))
        self.l2 = l2

    def forward(self, x):
        o = self.net(x)
        return F.normalize(o, dim=-1) if self.l2 else o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--k_color", type=int, default=3)
    ap.add_argument("--code_dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default="lmwm/outputs/appearance_gen/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    pooled = z["current"][:, :1280].astype(np.float32)
    state = z["current"][:, -14:].astype(np.float32)
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    ep = z["episode_id"].astype(np.int64)
    uep = np.unique(ep)

    print(f"computing garment colors for {len(uep)} episodes ...", flush=True)
    cols = episode_colors(z, args.dataset_root, args.camera, uep)
    # KMeans (numpy, few iters) on colors
    rng = np.random.default_rng(0)
    C = cols[rng.choice(len(cols), args.k_color, replace=False)]
    for _ in range(30):
        a = ((cols[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(args.k_color):
            if (a == j).any():
                C[j] = cols[a == j].mean(0)
    sizes = [int((a == j).sum()) for j in range(args.k_color)]
    # held-out = the color cluster most distant from the others' centroid, with enough episodes
    inter = np.array([np.linalg.norm(C[j] - np.delete(C, j, 0).mean(0)) for j in range(args.k_color)])
    order = np.argsort(-inter)
    hold = next(j for j in order if sizes[j] >= 0.12 * len(uep))
    ho_eps = set(uep[a == hold].tolist())
    print(f"cluster sizes={sizes} colors={C.astype(int).tolist()} -> HELD-OUT cluster {hold} "
          f"(color {C[hold].astype(int).tolist()}, {sizes[hold]} eps)", flush=True)

    is_ho = np.array([e in ho_eps for e in ep])
    # train on non-held-out appearance episodes (further split off a small val within them)
    tr = np.where(~is_ho & ok)[0]
    te = np.where(is_ho & ok)[0]                                  # held-out APPEARANCE test
    rng.shuffle(tr)
    Xp = torch.from_numpy(pooled); St = torch.from_numpy(state); Md = torch.from_numpy(med)
    feat = np.concatenate([pooled, state], 1)                     # predictor input (appearance in pooled, but predicts code)
    Ff = torch.from_numpy(feat.astype(np.float32))

    # ---- train inverse + forward (transition autoencoder) on train appearances ----
    inv = MLP(1280 + 1280, args.code_dim).to(dev)
    fwd = MLP(1280 + args.code_dim, 1280, l2=True).to(dev)
    opt = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=5e-4, weight_decay=1e-5)
    for s in range(args.steps):
        bi = tr[np.random.randint(0, len(tr), 1024)]
        cur = Xp[bi].to(dev); nxt = Md[bi].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            code = inv(torch.cat([cur, nxt], -1))
            rec = fwd(torch.cat([cur, code], -1))
            loss = (1 - (rec * nxt).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    inv.eval(); fwd.eval()

    # ---- predictor: current feat -> code (so forward(current, code) ~ next) ----
    pred = MLP(feat.shape[1], args.code_dim).to(dev)
    opt2 = torch.optim.AdamW(pred.parameters(), lr=5e-4, weight_decay=1e-5)
    for s in range(args.steps):
        bi = tr[np.random.randint(0, len(tr), 1024)]
        cur = Xp[bi].to(dev); nxt = Md[bi].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            code = pred(Ff[bi].to(dev))
            rec = fwd(torch.cat([cur, code], -1))
            loss = (1 - (rec * nxt).sum(-1)).mean()
        opt2.zero_grad(); loss.backward(); opt2.step()
    pred.eval()

    # ---- absolute baseline: feat -> next_medoid directly ----
    absm = MLP(feat.shape[1], 1280, l2=True).to(dev)
    opt3 = torch.optim.AdamW(absm.parameters(), lr=5e-4, weight_decay=1e-5)
    for s in range(args.steps):
        bi = tr[np.random.randint(0, len(tr), 1024)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = absm(Ff[bi].to(dev)); loss = (1 - (p * Md[bi].to(dev)).sum(-1)).mean()
        opt3.zero_grad(); loss.backward(); opt3.step()
    absm.eval()

    def evalset(rows, tag):
        with torch.no_grad():
            cur = Xp[rows].to(dev); nxt = med[rows]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                abs_cos = (absm(Ff[rows].to(dev)).float().cpu().numpy() * nxt).sum(1)
                orc = fwd(torch.cat([cur, inv(torch.cat([cur, Md[rows].to(dev)], -1))], -1)).float().cpu().numpy()
                fpred = fwd(torch.cat([cur, pred(Ff[rows].to(dev))], -1)).float().cpu().numpy()
        return {tag: {"absolute": round(float(abs_cos.mean()), 4),
                      "forward_oracle_code": round(float((orc * nxt).sum(1).mean()), 4),
                      "forward_predicted_code": round(float((fpred * nxt).sum(1).mean()), 4), "n": len(rows)}}

    res = {"held_out_color": C[hold].astype(int).tolist(), "cluster_sizes": sizes}
    res.update(evalset(tr[:8000], "in_distribution_appearance"))
    res.update(evalset(te, "HELD_OUT_appearance"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
