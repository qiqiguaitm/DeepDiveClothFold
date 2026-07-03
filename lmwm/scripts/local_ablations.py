#!/usr/bin/env python
"""Local fast-verification battery (H1-H5) on pooled augin data.

H1 prev-milestone latent vs one-hot   | H2 multi-task aux regularizes milestone
H3 +current-milestone latent          | H4 CVaR-CE variance vs mean
H5 graph-fusion lambda sweep (concavity, ~0.3 optimal)
Each is a quick MLP A/B; reports fused top1/top5/NLL(+std).
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
from lmwm.data import split_indices  # noqa: E402


class Net(nn.Module):
    def __init__(self, din, num_m, ld=1280, hid=512, subgoal=False):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid),
                                   nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid))
        self.cls = nn.Linear(hid, num_m)
        self.proto = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, ld)) if subgoal else None

    def forward(self, x):
        h = self.trunk(x)
        p = F.normalize(self.proto(h), dim=-1) if self.proto is not None else None
        return self.cls(h), p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="lmwm/outputs/local_ablations/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    cur = z["current"].astype(np.float32)
    pooled, prev_oh, state = cur[:, :1280], cur[:, 1280:1280 + 38], cur[:, -14:]
    y = z["future_milestone"].astype(np.int64); num_m = int(y.max()) + 1
    cur_m = z["current_milestone"].astype(np.int64)
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    proto_tbl = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    prev_id = prev_oh.argmax(1)
    prev_lat = np.where(prev_id[:, None] < len(proto_tbl), proto_tbl[np.clip(prev_id, 0, len(proto_tbl) - 1)], 0.0).astype(np.float32)
    cur_lat = proto_tbl[np.clip(cur_m, 0, len(proto_tbl) - 1)].astype(np.float32)

    trans = np.load(args.graph_npz)["transition_probs"].astype(np.float64); trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior_v = trans[cur_m[vi]]

    def fuse(p, lam=0.3):
        lp = (1 - lam) * np.log(np.clip(p, 1e-12, 1)) + lam * np.log(np.clip(prior_v, 1e-12, 1))
        lp -= lp.max(1, keepdims=True); e = np.exp(lp); return e / e.sum(1, keepdims=True)

    yv = y[vi]
    def stat(p):
        per = -np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, 1)); rank = (np.argsort(-p, 1) == yv[:, None]).argmax(1)
        return {"top1": round(float((rank == 0).mean()), 4), "top5": round(float((rank < 5).mean()), 4),
                "nll": round(float(per.mean()), 4), "nll_std": round(float(per.std()), 4)}

    def train(feat, subgoal=False, cvar=False):
        X = torch.from_numpy(feat); din = feat.shape[1]
        Y = torch.from_numpy(y); M = torch.from_numpy(med); tok = torch.from_numpy(ok)
        model = Net(din, num_m, subgoal=subgoal).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
        for s in range(args.steps):
            bi = ti[np.random.randint(0, len(ti), 2048)]
            lg, pr = model(X[bi].to(dev))
            per = F.cross_entropy(lg, Y[bi].to(dev), reduction="none")
            ce = (0.5 * per.mean() + 0.5 * torch.topk(per, 205).values.mean()) if cvar else per.mean()
            loss = ce
            if subgoal:
                m = tok[bi].to(dev)
                if m.any():
                    loss = loss + 5.0 * (1 - (pr[m] * M[bi].to(dev)[m]).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            ps = [F.softmax(model(X[vi[s:s + 8192]].to(dev))[0], -1).cpu().numpy() for s in range(0, len(vi), 8192)]
        return np.concatenate(ps)

    R = {}
    augin = np.concatenate([pooled, prev_oh, state], 1)
    augin_lat = np.concatenate([pooled, prev_lat, state], 1)

    print("H1 prev-latent vs one-hot ...", flush=True)
    R["H1_prev_onehot"] = stat(fuse(train(augin)))
    R["H1_prev_latent"] = stat(fuse(train(augin_lat)))

    print("H2 multi-task aux ...", flush=True)
    R["H2_milestone_only"] = stat(fuse(train(augin, subgoal=False)))
    R["H2_multitask"] = stat(fuse(train(augin, subgoal=True)))

    print("H3 +current-milestone latent ...", flush=True)
    R["H3_plus_current_latent"] = stat(fuse(train(np.concatenate([augin_lat, cur_lat], 1))))

    print("H4 CVaR-CE ...", flush=True)
    R["H4_ce"] = stat(fuse(train(augin_lat, cvar=False)))
    R["H4_cvar_ce"] = stat(fuse(train(augin_lat, cvar=True)))

    print("H5 lambda sweep ...", flush=True)
    p_raw = train(augin_lat)
    R["H5_lambda"] = {f"{lam:.1f}": stat(fuse(p_raw, lam))["top1"] for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
