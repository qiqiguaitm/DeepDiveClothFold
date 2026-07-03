#!/usr/bin/env python
"""P0 production milestone model: pooled MLP with the verified H-battery wins.

Input  = [pooled DINOv3-H 1280 | prev-milestone latent 1280 (H1) | current-milestone
          latent 1280 (H3) | proprio 14]  (= 3854-D)
Loss   = CVaR-CE milestone (H4, mean+variance) + episode-medoid subgoal cosine (on-manifold)
Fusion = graph prior, lambda swept (H5, optimum >= 0.5)
Ensemble of N members (different init). Saves members + summary (best lambda).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402


class ProdNet(nn.Module):
    def __init__(self, din, num_m, ld=1280, hid=512, depth=2):
        super().__init__()
        layers = [nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid)]
        self.trunk = nn.Sequential(*layers)
        self.cls = nn.Linear(hid, num_m)
        self.proto = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, ld))

    def forward(self, x):
        h = self.trunk(x)
        return self.cls(h), F.normalize(self.proto(h), dim=-1)


def build_feat(z, proto_tbl):
    cur = z["current"].astype(np.float32)
    pooled, prev_oh, state = cur[:, :1280], cur[:, 1280:1280 + 38], cur[:, -14:]
    prev_id = prev_oh.argmax(1)
    prev_lat = np.where(prev_id[:, None] < len(proto_tbl), proto_tbl[np.clip(prev_id, 0, len(proto_tbl) - 1)], 0.0)
    cur_lat = proto_tbl[np.clip(z["current_milestone"].astype(np.int64), 0, len(proto_tbl) - 1)]
    return np.concatenate([pooled, prev_lat, cur_lat, state], 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--save_dir", default="lmwm/checkpoints/prod_milestone", type=Path)
    ap.add_argument("--out", default="lmwm/outputs/prod_milestone/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    proto_tbl = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    feat = build_feat(z, proto_tbl); din = feat.shape[1]
    y = z["future_milestone"].astype(np.int64); num_m = int(y.max()) + 1
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    trans = np.load(args.graph_npz)["transition_probs"].astype(np.float64); trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    tri = ti[ok[ti]]
    X = torch.from_numpy(feat); Y = torch.from_numpy(y); M = torch.from_numpy(med); tok = torch.from_numpy(ok)
    Xv = torch.from_numpy(feat[vi]).to(dev); yv = y[vi]; okv = ok[vi]; medv = med[vi]
    prior_v = trans[z["current_milestone"][vi].astype(np.int64)]

    args.save_dir.mkdir(parents=True, exist_ok=True)
    probs = None; protos = None
    for j in range(args.members):
        torch.manual_seed(100 + j)
        m = ProdNet(din, num_m).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
        for s in range(args.steps):
            bi = tri[np.random.randint(0, len(tri), 2048)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, pr = m(X[bi].to(dev))
                per = F.cross_entropy(lg, Y[bi].to(dev), reduction="none")   # H4 CVaR-CE
                k = max(1, int(0.1 * len(per)))
                loss = 0.5 * per.mean() + 0.5 * torch.topk(per, k).values.mean() \
                    + 5.0 * (1 - (pr * M[bi].to(dev)).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval()
        pp, gg = [], []
        with torch.no_grad():
            for s in range(0, len(vi), 8192):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, pr = m(Xv[s:s + 8192])
                pp.append(F.softmax(lg.float(), -1).cpu().numpy()); gg.append(F.normalize(pr.float(), -1).cpu().numpy())
        pp = np.concatenate(pp); gg = np.concatenate(gg)
        probs = pp if probs is None else probs + pp
        protos = gg if protos is None else protos + gg
        torch.save({"model": m.state_dict(), "din": din, "num_m": num_m}, args.save_dir / f"member_{j}.pt")
        print(f"member {j} saved", flush=True)
    probs /= args.members; protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)

    def stat(p):
        per = -np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, 1)); rank = (np.argsort(-p, 1) == yv[:, None]).argmax(1)
        return {"top1": round(float((rank == 0).mean()), 4), "top5": round(float((rank < 5).mean()), 4),
                "nll": round(float(per.mean()), 4), "nll_std": round(float(per.std()), 4)}

    def fuse(lam):
        lp = (1 - lam) * np.log(np.clip(probs, 1e-12, 1)) + lam * np.log(np.clip(prior_v, 1e-12, 1))
        lp -= lp.max(1, keepdims=True); e = np.exp(lp); return e / e.sum(1, keepdims=True)

    sweep = {f"{lam:.2f}": stat(fuse(lam))["top1"] for lam in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
    best_lam = max(sweep, key=sweep.get)
    res = {"members": args.members, "input_dim": din, "raw": stat(probs),
           "lambda_sweep_top1": sweep, "best_lambda": float(best_lam),
           "fused_best": stat(fuse(float(best_lam))),
           "subgoal_cos": round(float((protos[okv] * medv[okv]).sum(1).mean()), 4),
           "vs_baseline": {"mlp_prod_ens_fuse": 0.459, "lawm_fulldata": 0.382}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
