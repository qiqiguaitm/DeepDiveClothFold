#!/usr/bin/env python
"""P0 full-data LaWM-backbone milestone training (gf3, 8xH20).

Reads precomputed DINOv3-H grid shards (.npy, memmap -> shared across the 8 parallel
member processes via OS page cache). Data flow: current DINO grid + milestone-1 latent
+ state -> next milestone (CE) + milestone+1 subgoal latent (cos). LaWM LAMEncoder
(Q-Former over the token grid) backbone.

Ensemble = launch one process per GPU (different --init_seed); then --eval averages
the members' probs, fuses with the graph prior, and reports top1/5/NLL + subgoal cos.
"""

from __future__ import annotations

import argparse
import glob
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
from lmwm.data import split_indices  # noqa: E402

_BASE = (HERE.parents[1] / "vendor/LaWAM/latent_action_model").resolve()
sys.path.insert(0, str(_BASE.parent))
for _n, _s in [("latent_action_model", _BASE), ("latent_action_model.core", _BASE / "core"),
               ("latent_action_model.core.utils", _BASE / "core" / "utils")]:
    _m = types.ModuleType(_n); _m.__path__ = [str(_s)]; sys.modules[_n] = _m
from latent_action_model.core.utils.lam_encoder import LAMEncoder  # noqa: E402

EXTRA = 1280 + 14  # milestone-1 latent + state


class GridStore:
    """Memmapped contiguous .npy shards grid_{lo}_{hi}.npy -> gather(rows)->[B,256,1280]."""

    def __init__(self, grid_dir):
        self.shards = []
        for f in sorted(Path(grid_dir).glob("grid_*.npy")):
            lo, hi = int(f.stem.split("_")[1]), int(f.stem.split("_")[2])
            self.shards.append((lo, hi, np.load(f, mmap_mode="r")))
        assert self.shards, f"no grid shards in {grid_dir}"
        self.lo = np.array([s[0] for s in self.shards]); self.hi = np.array([s[1] for s in self.shards])

    def gather(self, rows):
        out = np.empty((len(rows), 256, 1280), np.float16)
        si = np.searchsorted(self.hi, rows, side="right")
        for i, (r, s) in enumerate(zip(rows, si)):
            lo, _, arr = self.shards[s]
            out[i] = arr[r - lo]
        return out


class LaWMMulti(nn.Module):
    def __init__(self, num_m, ctx=768, layers=6, heads=12, ld=1280):
        super().__init__()
        self.enc = LAMEncoder(context_dim=ctx, input_dim=1280, num_layers=layers, num_heads=heads,
                              num_frames=1, num_queries=1, grid_hw=(16, 16), code_dim=ctx)
        d = ctx + EXTRA
        self.milestone = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_m))
        self.proto = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, ld))

    def forward(self, grid, extra):
        h = torch.cat([self.enc(grid.unsqueeze(1))[:, 0], extra], -1)
        return self.milestone(h), F.normalize(self.proto(h), dim=-1)


def build_extra(z, proto_tbl):
    cur = z["current"].astype(np.float32)
    prev_id = cur[:, 1280:1280 + 38].argmax(1)
    prev_latent = np.where(prev_id[:, None] < len(proto_tbl),
                           proto_tbl[np.clip(prev_id, 0, len(proto_tbl) - 1)], 0.0).astype(np.float32)
    return np.concatenate([prev_latent, cur[:, -14:]], 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--graph_npz", required=True)
    ap.add_argument("--grid_dir", required=True)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--init_seed", type=int, default=100)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=768)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--save_dir", default="temp/lmwm_p0/lawm_members", type=Path)
    ap.add_argument("--members_glob", default="temp/lmwm_p0/lawm_members/member_*.pt")
    ap.add_argument("--out", default="temp/lmwm_p0/lawm_eval.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    proto_tbl = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    num_m = int(z["future_milestone"].max()) + 1
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    extra = build_extra(z, proto_tbl)
    y = z["future_milestone"].astype(np.int64)
    store = GridStore(args.grid_dir)
    ti = ti[ok[ti]]; vi_all = vi.copy(); vi = vi[ok[vi]]

    if args.mode == "train":
        torch.manual_seed(args.init_seed)
        model = LaWMMulti(num_m, args.ctx, args.layers).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
        Ex = torch.from_numpy(extra); Md = torch.from_numpy(med); Y = torch.from_numpy(y)
        for s in range(args.steps):
            bi = ti[np.random.randint(0, len(ti), args.bs)]
            g = torch.from_numpy(store.gather(bi).astype(np.float32)).to(dev)
            lg, pr = model(g, Ex[bi].to(dev))
            loss = F.cross_entropy(lg, Y[bi].to(dev)) + 5.0 * (1 - (pr * Md[bi].to(dev)).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            if s % 500 == 0:
                print(f"seed{args.init_seed} step {s} loss {loss.item():.3f}", flush=True)
        args.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "ctx": args.ctx, "layers": args.layers, "num_m": num_m,
                    "init_seed": args.init_seed}, args.save_dir / f"member_{args.init_seed}.pt")
        print(f"saved member_{args.init_seed}.pt", flush=True)
        return

    # ---- eval: ensemble members + graph fuse ----
    paths = sorted(glob.glob(args.members_glob))
    print(f"eval {len(paths)} members", flush=True)
    Ex = torch.from_numpy(extra)
    probs = None; protos = None
    for p in paths:
        ck = torch.load(p, map_location="cpu")
        m = LaWMMulti(ck["num_m"], ck["ctx"], ck["layers"]).to(dev); m.load_state_dict(ck["model"]); m.eval()
        pp, gg = [], []
        with torch.no_grad():
            for s in range(0, len(vi), 512):
                b = vi[s:s + 512]
                g = torch.from_numpy(store.gather(b).astype(np.float32)).to(dev)
                lg, pr = m(g, Ex[b].to(dev))
                pp.append(F.softmax(lg, -1).cpu().numpy()); gg.append(pr.cpu().numpy())
        pp = np.concatenate(pp); gg = np.concatenate(gg)
        probs = pp if probs is None else probs + pp
        protos = gg if protos is None else protos + gg
    probs /= len(paths); protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)

    trans = np.load(args.graph_npz)["transition_probs"].astype(np.float64)
    trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior = trans[z["current_milestone"][vi].astype(np.int64)]
    lp = (1 - args.lam) * np.log(np.clip(probs, 1e-12, 1)) + args.lam * np.log(np.clip(prior, 1e-12, 1))
    lp -= lp.max(1, keepdims=True); fused = np.exp(lp); fused /= fused.sum(1, keepdims=True)
    yv = y[vi]; mv = med[vi]

    def st(p):
        per = -np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, 1)); rank = (np.argsort(-p, 1) == yv[:, None]).argmax(1)
        return {"top1": round(float((rank == 0).mean()), 4), "top5": round(float((rank < 5).mean()), 4),
                "nll": round(float(per.mean()), 4), "nll_std": round(float(per.std()), 4)}
    res = {"members": len(paths), "raw": st(probs), "fused": st(fused),
           "subgoal_cos": round(float((protos * mv).sum(1).mean()), 4)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
