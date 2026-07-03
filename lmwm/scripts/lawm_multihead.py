#!/usr/bin/env python
"""LaWM backbone with the LMWM data flow:
   current latent (DINO grid) + prev-milestone + state  ->  next milestone + subgoal.

Multi-head (milestone CE + episode-medoid subgoal cosine), the aux subgoal acting as
a regularizer (shown to help milestone). A/B: LaWM LAMEncoder backbone vs our pooled-MLP
backbone, same held-out subset. Reports milestone top1/5/NLL (raw + graph-fused) and
subgoal cos.
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
from lever_patch_token import read_enc  # noqa: E402

_BASE = (HERE.parents[1] / "vendor/LaWAM/latent_action_model").resolve()
sys.path.insert(0, str(_BASE.parent))
for _n, _s in [("latent_action_model", _BASE), ("latent_action_model.core", _BASE / "core"),
               ("latent_action_model.core.utils", _BASE / "core" / "utils")]:
    _m = types.ModuleType(_n); _m.__path__ = [str(_s)]; sys.modules[_n] = _m
from latent_action_model.core.utils.lam_encoder import LAMEncoder  # noqa: E402

# LaWM-faithful milestone-latent flow: replace the prev-milestone ONE-HOT with the
# prev-milestone LATENT (milestone-1 latent, prototype table lookup, 1280-D), like
# LaWM's frame latent(t-1); the subgoal target is the milestone+1 latent (next_medoid).
EXTRA = 1280 + 14  # milestone-1 latent(1280) + state(14)


class MLPBackbone(nn.Module):
    def __init__(self, hid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1280 + EXTRA, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid))
        self.out_dim = hid

    def forward(self, pooled, extra):
        return self.net(torch.cat([pooled, extra], -1))


class LaWMBackbone(nn.Module):
    def __init__(self, ctx=768, layers=6, heads=12):
        super().__init__()
        self.enc = LAMEncoder(context_dim=ctx, input_dim=1280, num_layers=layers, num_heads=heads,
                              num_frames=1, num_queries=1, grid_hw=(16, 16), code_dim=ctx)
        self.out_dim = ctx + EXTRA

    def forward(self, grid, extra):
        return torch.cat([self.enc(grid.unsqueeze(1))[:, 0], extra], -1)


class MultiHead(nn.Module):
    def __init__(self, backbone, num_m, ld=1280):
        super().__init__()
        self.backbone = backbone
        d = backbone.out_dim
        self.milestone = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_m))
        self.proto = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, ld))

    def forward(self, main, extra):
        h = self.backbone(main, extra)
        return self.milestone(h), F.normalize(self.proto(h), dim=-1)


def mstats(probs, y):
    per = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))
    rank = (np.argsort(-probs, 1) == y[:, None]).argmax(1)
    return {"top1": round(float((rank == 0).mean()), 4), "top5": round(float((rank < 5).mean()), 4),
            "nll": round(float(per.mean()), 4)}


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
    ap.add_argument("--out", default="lmwm/outputs/lawm_multihead/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = vi.numpy(); itr = np.setdiff1d(np.arange(n), vi)
    med_ok = np.linalg.norm(z["next_medoid"], axis=1) > 1e-6
    rng = np.random.default_rng(0)
    tr = rng.choice(itr[med_ok[itr]], min(args.n_train, int(med_ok[itr].sum())), replace=False)
    va = rng.choice(vi[med_ok[vi]], min(args.n_val, int(med_ok[vi].sum())), replace=False)
    sel = np.concatenate([tr, va]); ntr = len(tr)

    proto_tbl = np.load(args.graph_npz)["prototype_table"].astype(np.float32)  # [37,1280] milestone latents
    cur = z["current"][sel].astype(np.float32)
    prev_id = cur[:, 1280:1280 + 38].argmax(1)                                  # 0..37 (37=START)
    prev_latent = np.where(prev_id[:, None] < len(proto_tbl),
                           proto_tbl[np.clip(prev_id, 0, len(proto_tbl) - 1)], 0.0).astype(np.float32)
    extra = np.concatenate([prev_latent, cur[:, -14:]], 1).astype(np.float32)   # milestone-1 latent + state
    y = z["future_milestone"][sel].astype(np.int64); num_m = int(z["future_milestone"].max()) + 1
    med = z["next_medoid"][sel].astype(np.float32); med /= np.linalg.norm(med, axis=1, keepdims=True) + 1e-8
    eps = z["episode_id"][sel].astype(np.int64); ts = z["t"][sel].astype(np.int64)
    cur_m = z["current_milestone"][sel].astype(np.int64)

    print(f"encoding {len(sel)} grids ...", flush=True)
    imgs = read_enc(args.dataset_root, args.camera, eps, ts, 256)
    enc = load_encoder("dinov3-h", device=str(dev))
    grid = enc.encode_grid(imgs).astype(np.float32).reshape(len(sel), 1280, 256).transpose(0, 2, 1)
    pooled = grid.mean(1)

    trans = np.load(args.graph_npz)["transition_probs"].astype(np.float64)
    trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior = trans[cur_m[ntr:]]
    def fuse(p):
        lp = (1 - args.lam) * np.log(np.clip(p, 1e-12, 1)) + args.lam * np.log(np.clip(prior, 1e-12, 1))
        lp -= lp.max(1, keepdims=True); e = np.exp(lp); return e / e.sum(1, keepdims=True)

    G = torch.from_numpy(grid); Pp = torch.from_numpy(pooled); Ex = torch.from_numpy(extra)
    Y = torch.from_numpy(y); Md = torch.from_numpy(med)
    yv, mv = y[ntr:], med[ntr:]

    def run(model, is_lawm):
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
        bs = 128 if is_lawm else 512
        for s in range(args.steps):
            bi = torch.randint(0, ntr, (bs,))
            main = (G[bi] if is_lawm else Pp[bi]).to(dev)
            lg, pr = model(main, Ex[bi].to(dev))
            loss = F.cross_entropy(lg, Y[bi].to(dev)) + 5.0 * (1.0 - (pr * Md[bi].to(dev)).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval()
        ps, cs = [], []
        with torch.no_grad():
            for s in range(ntr, len(sel), 256):
                idx = slice(s, min(s + 256, len(sel)))
                main = (G[idx] if is_lawm else Pp[idx]).to(dev)
                lg, pr = model(main, Ex[idx].to(dev))
                ps.append(F.softmax(lg, -1).cpu().numpy()); cs.append(pr.cpu().numpy())
        return np.concatenate(ps), np.concatenate(cs)

    res = {}
    for name, model, is_lawm in [
        ("MLP_multihead(augin)", MultiHead(MLPBackbone(), num_m), False),
        ("LaWM_multihead(augin)", MultiHead(LaWMBackbone(args.ctx, args.layers), num_m), True),
    ]:
        npar = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        p, pr = run(model, is_lawm)
        cos = float((pr * mv).sum(1).mean())
        res[name] = {"params_M": npar, "raw": mstats(p, yv), "fused": mstats(fuse(p), yv), "subgoal_cos": round(cos, 4)}
        r = res[name]
        print(f"{name:24s} {npar:6.1f}M | raw top1={r['raw']['top1']:.4f} | +fuse top1={r['fused']['top1']:.4f} "
              f"top5={r['fused']['top5']:.4f} nll={r['fused']['nll']:.2f} | subgoal cos={cos:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
