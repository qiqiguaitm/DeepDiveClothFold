#!/usr/bin/env python
"""Reuse LaWM's LAMEncoder (Q-Former over the DINO token grid) for next-milestone
prediction, vs our pooled-MLP. Same held-out subset, CE on real_future, +graph fuse.

Four models isolate the two questions:
  A) MLP  over pooled  (frame-only)          -- our frame-only analog
  B) MLP  over augin   (pooled+prev+state)   -- our production analog
  C) LaWM LAMEncoder over grid (frame-only)  -- LaWM framework, frame-only
  D) LaWM LAMEncoder over grid + prev+state  -- LaWM framework, augin-equivalent
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "crave/src"))
from lmwm.data import split_indices  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from lever_patch_token import read_enc  # noqa: E402  (frame reader + resize)

# import LaWM's LAMEncoder without triggering core/__init__ (which pulls lightning)
_BASE = (HERE.parents[1] / "vendor/LaWAM/latent_action_model").resolve()
sys.path.insert(0, str(_BASE.parent))
for _n, _s in [("latent_action_model", _BASE), ("latent_action_model.core", _BASE / "core"),
               ("latent_action_model.core.utils", _BASE / "core" / "utils")]:
    _m = types.ModuleType(_n); _m.__path__ = [str(_s)]; sys.modules[_n] = _m
from latent_action_model.core.utils.lam_encoder import LAMEncoder  # noqa: E402


class MLP(nn.Module):
    def __init__(self, din, num_m, hid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, num_m))

    def forward(self, pooled, extra=None):
        x = pooled if extra is None else torch.cat([pooled, extra], -1)
        return self.net(x)


class LaWMHead(nn.Module):
    def __init__(self, num_m, ctx=768, layers=6, heads=12, extra_dim=0):
        super().__init__()
        self.enc = LAMEncoder(context_dim=ctx, input_dim=1280, num_layers=layers, num_heads=heads,
                              num_frames=1, num_queries=1, grid_hw=(16, 16), code_dim=ctx)
        self.head = nn.Sequential(nn.LayerNorm(ctx + extra_dim), nn.Linear(ctx + extra_dim, num_m))

    def forward(self, grid, extra=None):        # grid [B,256,1280]
        code = self.enc(grid.unsqueeze(1))[:, 0]  # [B,ctx]
        if extra is not None:
            code = torch.cat([code, extra], -1)
        return self.head(code)


def stats(probs, y):
    per = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))
    rank = (np.argsort(-probs, 1) == y[:, None]).argmax(1)
    return {"top1": round(float((rank == 0).mean()), 4), "top3": round(float((rank < 3).mean()), 4),
            "top5": round(float((rank < 5).mean()), 4), "nll": round(float(per.mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_val", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--ctx", type=int, default=768)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--out", default="lmwm/outputs/lawm_milestone/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = vi.numpy(); itr = np.setdiff1d(np.arange(n), vi)
    rng = np.random.default_rng(0)
    tr = rng.choice(itr, min(args.n_train, len(itr)), replace=False)
    va = rng.choice(vi, min(args.n_val, len(vi)), replace=False)
    sel = np.concatenate([tr, va]); ntr = len(tr)

    cur = z["current"][sel].astype(np.float32)
    prev, state = cur[:, 1280:1280 + 38], cur[:, -14:]
    extra = np.concatenate([prev, state], 1).astype(np.float32)   # 52-D side info
    y = z["future_milestone"][sel].astype(np.int64); num_m = int(z["future_milestone"].max()) + 1
    eps = z["episode_id"][sel].astype(np.int64); ts = z["t"][sel].astype(np.int64)
    cur_m = z["current_milestone"][sel].astype(np.int64)

    print(f"encoding {len(sel)} grids ...", flush=True)
    imgs = read_enc(args.dataset_root, args.camera, eps, ts, 256)
    enc = load_encoder("dinov3-h", device=str(dev))
    grid = enc.encode_grid(imgs).astype(np.float32).reshape(len(sel), 1280, 256).transpose(0, 2, 1)  # [N,256,1280]
    pooled = grid.mean(1)                                          # [N,1280]

    trans = np.load(args.graph_npz)["transition_probs"].astype(np.float64)
    trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior = trans[cur_m[ntr:]]
    def fuse(p):
        lp = (1 - args.lam) * np.log(np.clip(p, 1e-12, 1)) + args.lam * np.log(np.clip(prior, 1e-12, 1))
        lp -= lp.max(1, keepdims=True); e = np.exp(lp); return e / e.sum(1, keepdims=True)

    G = torch.from_numpy(grid); Pp = torch.from_numpy(pooled); Ex = torch.from_numpy(extra); Y = torch.from_numpy(y)
    yv = y[ntr:]

    def run(model, kind):
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
        bs = 128 if kind.startswith("lawm") else 512
        for s in range(args.steps):
            bi = torch.randint(0, ntr, (bs,))
            if kind == "mlp_pooled":
                lg = model(Pp[bi].to(dev))
            elif kind == "mlp_augin":
                lg = model(Pp[bi].to(dev), Ex[bi].to(dev))
            elif kind == "lawm_grid":
                lg = model(G[bi].to(dev))
            else:  # lawm_augin
                lg = model(G[bi].to(dev), Ex[bi].to(dev))
            loss = F.cross_entropy(lg, Y[bi].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval()
        ps = []
        with torch.no_grad():
            for s in range(ntr, len(sel), 256):
                idx = slice(s, min(s + 256, len(sel)))
                if kind == "mlp_pooled":
                    lg = model(Pp[idx].to(dev))
                elif kind == "mlp_augin":
                    lg = model(Pp[idx].to(dev), Ex[idx].to(dev))
                elif kind == "lawm_grid":
                    lg = model(G[idx].to(dev))
                else:
                    lg = model(G[idx].to(dev), Ex[idx].to(dev))
                ps.append(F.softmax(lg, -1).cpu().numpy())
        return np.concatenate(ps)

    configs = [
        ("A_mlp_pooled", MLP(1280, num_m), "mlp_pooled"),
        ("B_mlp_augin", MLP(1280 + 52, num_m), "mlp_augin"),
        ("C_lawm_grid", LaWMHead(num_m, args.ctx, args.layers), "lawm_grid"),
        ("D_lawm_augin", LaWMHead(num_m, args.ctx, args.layers, extra_dim=52), "lawm_augin"),
    ]
    res = {}
    for name, model, kind in configs:
        npar = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        p = run(model, kind)
        res[name] = {"params_M": npar, "raw": stats(p, yv), "fused": stats(fuse(p), yv)}
        r = res[name]
        print(f"{name:16s} {npar:6.1f}M | raw top1={r['raw']['top1']:.4f} nll={r['raw']['nll']:.2f} "
              f"| +fuse top1={r['fused']['top1']:.4f} top5={r['fused']['top5']:.4f} nll={r['fused']['nll']:.2f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
