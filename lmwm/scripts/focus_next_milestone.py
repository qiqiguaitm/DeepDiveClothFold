#!/usr/bin/env python
"""Focused model: predict ONLY P(next milestone). No greedy/max-product/subgoal heads.

Question: does dropping the other heads (they compete for trunk capacity) improve the
next-milestone distribution vs the multi-head UnifiedLMWM greedy head? Trains N focused
single-head members (augin, real_future, CE) and reports single / ensemble / +graph-fuse
top1/3/5 + NLL(+std/CVaR), against the known unified baselines.

Unified reference (same held-out): greedy single 0.408 / +fuse 0.434 ; ensemble+fuse 0.459.
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


class FocusNet(nn.Module):
    """augin -> trunk(512x2) -> single 37-way next-milestone head. Nothing else."""

    def __init__(self, din, num_m, hid=512, depth=2):
        super().__init__()
        layers = [nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hid, num_m)

    def forward(self, x):
        return self.head(self.trunk(x))


def ce_loss(logits, y, tail_mode=None, tail_w=0.5, cvar_q=0.1):
    per = F.cross_entropy(logits, y, reduction="none")
    if tail_mode == "cvar":
        k = max(1, int(cvar_q * len(per)))
        worst = torch.topk(per, k).values.mean()
        return (1 - tail_w) * per.mean() + tail_w * worst
    return per.mean()


def stats(probs, y):
    per = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))
    rank = (np.argsort(-probs, 1) == y[:, None]).argmax(1)
    k10 = max(1, int(0.1 * len(per)))
    return {"top1": round(float((rank == 0).mean()), 4), "top3": round(float((rank < 3).mean()), 4),
            "top5": round(float((rank < 5).mean()), 4), "nll": round(float(per.mean()), 4),
            "nll_std": round(float(per.std()), 4), "cvar10": round(float(np.sort(per)[-k10:].mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--members", type=int, default=4)      # 3 plain + 1 cvar
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--out", default="lmwm/outputs/focus_milestone/summary.json", type=Path)
    ap.add_argument("--save_dir", default="lmwm/checkpoints/focus_milestone", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    X = z["current"].astype(np.float32); din = X.shape[1]
    y = z["future_milestone"].astype(np.int64); num_m = int(y.max()) + 1
    Xt, Yt = torch.from_numpy(X[ti]), torch.from_numpy(y[ti])
    Xv = torch.from_numpy(X[vi]).to(dev); yv = y[vi]
    cur_v = z["current_milestone"][vi].astype(np.int64)
    ntr = len(ti)

    trans = np.load(args.graph_npz)["transition_probs"].astype(np.float64)
    trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior = trans[cur_v]
    def fuse(p):
        lp = (1 - args.lam) * np.log(np.clip(p, 1e-12, 1)) + args.lam * np.log(np.clip(prior, 1e-12, 1))
        lp -= lp.max(1, keepdims=True); e = np.exp(lp); return e / e.sum(1, keepdims=True)

    def train(init_seed, tail_mode):
        torch.manual_seed(init_seed)
        m = FocusNet(din, num_m, args.hidden, args.depth).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
        for s in range(args.steps):
            bi = torch.randint(0, ntr, (2048,))
            loss = ce_loss(m(Xt[bi].to(dev)), Yt[bi].to(dev), tail_mode)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            ps = [F.softmax(m(Xv[s:s + 8192]), -1).cpu().numpy() for s in range(0, len(vi), 8192)]
        return m, np.concatenate(ps)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    probs_all = []
    for j in range(args.members):
        tail = "cvar" if j == args.members - 1 else None
        m, p = train(100 + j, tail)
        probs_all.append(p)
        torch.save({"model": m.state_dict(), "in_dim": din, "num_m": num_m,
                    "hidden": args.hidden, "depth": args.depth, "tail": tail}, args.save_dir / f"member_{j}.pt")
        print(f"  member {j} ({'cvar' if tail else 'plain'}): single top1={stats(p, yv)['top1']:.4f}", flush=True)

    res = {}
    single = probs_all[0]
    ens = np.mean(probs_all, axis=0)
    ens_plain = np.mean(probs_all[:-1], axis=0)  # exclude cvar for max-mean
    for tag, p in [("focus_single", single), ("focus_single+fuse", fuse(single)),
                   ("focus_ens_plain+fuse", fuse(ens_plain)), ("focus_ens_all+fuse", fuse(ens))]:
        res[tag] = stats(p, yv)
        r = res[tag]
        print(f"{tag:26s} top1={r['top1']:.4f} top3={r['top3']:.4f} top5={r['top5']:.4f} "
              f"nll={r['nll']:.3f}(std {r['nll_std']:.2f} cvar {r['cvar10']:.2f})", flush=True)

    res["_unified_reference"] = {"greedy_single": 0.408, "greedy_single_fuse": 0.434, "ensemble_fuse": 0.459}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
