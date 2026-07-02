#!/usr/bin/env python
"""Does the MILESTONE-LEVEL path (m_{t-1}, m_{t-2}, ...) + current latent predict
m_{t+1} better than the current latent alone?

Unlike consecutive-frame history (redundant), the milestone path is a coarse,
longer-horizon signal that could disambiguate route/order for a far-horizon jump.
Tests with a LEARNED combiner (same MLP class as LMWM) so the frame+path fusion
is fair (not a cosine-append artifact). Held-out episodes, predict real next-unique
milestone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_full(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16); valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard); gi = z["gidx"].astype(np.int64)
        feat[gi] = z["feat"]; valid[gi] = z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32); fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], fv


class Probe(nn.Module):
    def __init__(self, in_dim, num_m, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, num_m))

    def forward(self, x):
        return self.net(x)


def train_eval(Xtr, ytr, Xva, yva, num_m, dev, steps=2000, bs=4096):
    model = Probe(Xtr.shape[1], num_m).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr = torch.from_numpy(Xtr).to(dev); ytr = torch.from_numpy(ytr).to(dev)
    Xva = torch.from_numpy(Xva).to(dev); yva = torch.from_numpy(yva).to(dev)
    n = len(Xtr)
    for _ in range(steps):
        bi = torch.randint(0, n, (bs,), device=dev)
        loss = F.cross_entropy(model(Xtr[bi]), ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(Xva)
        p = F.softmax(logits, -1)
        top1 = float((logits.argmax(1) == yva).float().mean())
        nll = float(F.cross_entropy(logits, yva))
    return {"top1": top1, "nll": nll}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", default="lmwm/outputs/ceiling_diag/milestone_path.json", type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    g = np.load(args.graph_npz); proto = g["prototype_table"].astype(np.float32); num_m = len(proto)
    E, FR, F_ = load_full(args.feature_dir)
    assign = np.empty(len(F_), dtype=np.int64)
    for i in range(0, len(F_), 32768):
        assign[i:i + 32768] = (F_[i:i + 32768] @ proto.T).argmax(1)
    START = num_m

    feats, cur, p1, p2, nxt, ep = [], [], [], [], [], []
    for e in np.unique(E):
        loc = np.where(E == e)[0]; order = loc[np.argsort(FR[loc])]; seq = assign[order]
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        ms = [int(seq[s]) for s in st]
        for i in range(len(ms) - 1):
            feats.append(F_[order[en[i] - 1]]); cur.append(ms[i])
            p1.append(ms[i - 1] if i >= 1 else START); p2.append(ms[i - 2] if i >= 2 else START)
            nxt.append(ms[i + 1]); ep.append(e)
    feats = np.stack(feats).astype(np.float32)
    cur = np.array(cur); p1 = np.array(p1); p2 = np.array(p2); nxt = np.array(nxt).astype(np.int64); ep = np.array(ep)

    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    iv = np.array([x in val for x in ep]); itr = ~iv

    def onehot(a, k):
        o = np.zeros((len(a), k), dtype=np.float32); o[np.arange(len(a)), a] = 1.0; return o

    oh1 = onehot(p1, num_m + 1); oh2 = onehot(p2, num_m + 1); ohc = onehot(cur, num_m)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    variants = {
        "frame_only": feats,
        "frame+path_K1": np.concatenate([feats, oh1], 1),
        "frame+path_K2": np.concatenate([feats, oh1, oh2], 1),
        "path_only_K2_no_frame": np.concatenate([ohc, oh1, oh2], 1),
    }
    res = {}
    for name, X in variants.items():
        res[name] = train_eval(X[itr], nxt[itr], X[iv], nxt[iv], num_m, dev)
        print(f"{name:24s} top1={res[name]['top1']:.4f} nll={res[name]['nll']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"num_milestones": num_m, "n_train": int(itr.sum()), "n_val": int(iv.sum()), "variants": res}, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
