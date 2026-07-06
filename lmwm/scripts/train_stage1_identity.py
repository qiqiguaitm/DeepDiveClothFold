#!/usr/bin/env python
"""E1 / reframed-B: does a DISTRIBUTIONAL Stage-1 (predict next-milestone identity from the current
frame) capture the identity multimodality that best-of-8-on-code missed?

Two heads, both from the current pooled DINOv3-H frame feature:
  classifier  : MLP -> 37-way categorical over milestones (a distribution over "which milestone next")
  proto-reg   : MLP -> 1280 predicted next-milestone PROTOTYPE embedding; retrieve top-N milestones by
                cos (the CONTINUOUS-latent form our two-model Stage-1 would use -> generalizes, no hard 37)

Metric = top-N identity hit-rate on held-out episodes (N=1,2,3,5). top-1 = deterministic Stage-1;
top-N = best-of-N coverage. If top-3 >> top-1, sampling the identity distribution genuinely pays off
(unlike best-of-8-on-code, +0.02) -> confirms Stage-1 must be distributional AND that it works.

Runs BOTH candidate target constructions (V2 milestone_value, V3.1 milestone_viterbi).
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
from train_lawm_patch import load_index  # noqa: E402
from analyze_identity_conditioning import collect  # noqa: E402  (faithful V2 argmax / V3.1 viterbi pairs)


def mlp(din, dout, hid=512):
    return nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, dout))


def topn_hit(scores, target, Ns=(1, 2, 3, 5)):
    """scores (V, M) higher=better; target (V,) -> dict N->hit-rate."""
    order = np.argsort(-scores, axis=1)
    return {f"top{n}": round(float(np.mean([t in order[i, :n] for i, t in enumerate(target)])), 4) for n in Ns}


def run_mode(pairs, Fn, protoL, dev, steps, seed):
    cur = np.array([p[0] for p in pairs]); nxt = np.array([p[2] for p in pairs])
    X = Fn[cur].astype(np.float32); M = protoL.shape[0]
    # episode-agnostic split by pair index (episodes already shuffled into pairs); hold out 20%
    rng = np.random.default_rng(seed); perm = rng.permutation(len(cur))
    nval = int(0.2 * len(cur)); vi, ti = perm[:nval], perm[nval:]
    Xt = torch.from_numpy(X[ti]).to(dev); yt = torch.from_numpy(nxt[ti]).long().to(dev)
    Xv = torch.from_numpy(X[vi]).to(dev); yv = nxt[vi]
    PT = torch.from_numpy(protoL.astype(np.float32)).to(dev)

    clf = mlp(X.shape[1], M).to(dev); reg = mlp(X.shape[1], X.shape[1]).to(dev)
    oc = torch.optim.AdamW(clf.parameters(), lr=2e-4, weight_decay=1e-5)
    orr = torch.optim.AdamW(reg.parameters(), lr=2e-4, weight_decay=1e-5)
    for step in range(steps):
        sel = torch.randint(0, len(ti), (256,), device=dev)
        xb, yb = Xt[sel], yt[sel]
        lc = F.cross_entropy(clf(xb), yb)
        oc.zero_grad(); lc.backward(); oc.step()
        pr = reg(xb); pr = pr / (pr.norm(dim=1, keepdim=True) + 1e-8)
        lr_ = F.smooth_l1_loss(pr, PT[yb])                 # regress to next prototype (unit)
        orr.zero_grad(); lr_.backward(); orr.step()
    clf.eval(); reg.eval()
    with torch.no_grad():
        clf_scores = clf(Xv).float().cpu().numpy()
        pv = reg(Xv); pv = pv / (pv.norm(dim=1, keepdim=True) + 1e-8)
        reg_scores = (pv @ PT.T).float().cpu().numpy()     # cos to every prototype -> retrieval scores
    # marginal baseline: always predict the most frequent next milestones (ignores current frame)
    freq = np.bincount(nxt[ti], minlength=M).astype(float)
    marg = topn_hit(np.tile(freq, (len(yv), 1)), yv)
    return {"n_pairs": len(cur), "n_val": len(yv),
            "classifier_topN": topn_hit(clf_scores, yv),
            "proto_reg_topN": topn_hit(reg_scores, yv),
            "marginal_baseline_topN": marg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    E, FR, Fn = load_index(REPO / "temp/crave_full_dinov3h")
    rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = rg["prototype_table"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    pm = collect(E, FR, Fn, proto, protoL, rg["pord"].astype(np.float32))

    out = {}
    for mode in ["milestone_value", "milestone_viterbi"]:
        out[mode] = run_mode(pm[mode], Fn, protoL, args.device, args.steps, args.seed)
        print(mode, json.dumps(out[mode], indent=2), flush=True)
    (REPO / "lmwm/outputs/stage1_identity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
