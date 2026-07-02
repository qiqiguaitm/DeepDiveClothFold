#!/usr/bin/env python
"""Test LaWM-style STATE (proprioception) input for LMWM, and whether it stacks
with the milestone-path signal. CRAVE showed proprio disambiguates the visual
fold-start vs fold-end alias (arm extended vs retracted) -- orthogonal to the
frame. Learned MLP probe, held-out, predict real next-unique milestone.

Variants: frame_only | frame+path(K1) | frame+state | frame+path+state.
State (14-D observation.state) is z-scored; joined from kai0 parquets by frame_index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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

    def forward(self, x): return self.net(x)


def train_eval(Xtr, ytr, Xva, yva, num_m, dev, steps=2000, bs=4096):
    m = Probe(Xtr.shape[1], num_m).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr = torch.from_numpy(Xtr).to(dev); ytr = torch.from_numpy(ytr).to(dev)
    Xva = torch.from_numpy(Xva).to(dev); yva = torch.from_numpy(yva).to(dev)
    for _ in range(steps):
        bi = torch.randint(0, len(Xtr), (bs,), device=dev)
        loss = F.cross_entropy(m(Xtr[bi]), ytr[bi]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        lg = m(Xva)
        return {"top1": float((lg.argmax(1) == yva).float().mean()), "nll": float(F.cross_entropy(lg, yva))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", default="lmwm/outputs/ceiling_diag/state_probe.json", type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    g = np.load(args.graph_npz); proto = g["prototype_table"].astype(np.float32); num_m = len(proto)
    E, FR, F_ = load_full(args.feature_dir)
    assign = np.empty(len(F_), dtype=np.int64)
    for i in range(0, len(F_), 32768):
        assign[i:i + 32768] = (F_[i:i + 32768] @ proto.T).argmax(1)
    START = num_m
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])

    # stage-representative frames (last frame of each stage with a next stage)
    gidx, cur, p1, nxt, ep, frs = [], [], [], [], [], []
    for e in np.unique(E):
        loc = np.where(E == e)[0]; order = loc[np.argsort(FR[loc])]; seq = assign[order]
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        ms = [int(seq[s]) for s in st]
        for i in range(len(ms) - 1):
            gi = order[en[i] - 1]
            gidx.append(gi); cur.append(ms[i]); p1.append(ms[i - 1] if i >= 1 else START)
            nxt.append(ms[i + 1]); ep.append(int(e)); frs.append(int(FR[gi]))
    gidx = np.array(gidx); cur = np.array(cur); p1 = np.array(p1); nxt = np.array(nxt).astype(np.int64)
    ep = np.array(ep); frs = np.array(frs)
    feats = F_[gidx]

    # join state from parquet by (episode, frame_index)
    state = np.zeros((len(gidx), 14), dtype=np.float32)
    for e in np.unique(ep):
        pq = args.dataset_root / f"data/chunk-{e // cs:03d}/episode_{e:06d}.parquet"
        df = pd.read_parquet(pq, columns=["observation.state", "frame_index"])
        arr = np.stack(df["observation.state"].to_numpy()).astype(np.float32)  # (T,14)
        m = ep == e
        fi = np.clip(frs[m], 0, len(arr) - 1)
        state[m] = arr[fi]
    state = (state - state.mean(0)) / (state.std(0) + 1e-6)  # z-score

    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    iv = np.array([x in val for x in ep]); itr = ~iv
    oh1 = np.zeros((len(gidx), num_m + 1), np.float32); oh1[np.arange(len(gidx)), p1] = 1.0

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    variants = {
        "frame_only": feats,
        "frame+path_K1": np.concatenate([feats, oh1], 1),
        "frame+state": np.concatenate([feats, state], 1),
        "frame+path+state": np.concatenate([feats, oh1, state], 1),
    }
    res = {}
    for name, X in variants.items():
        res[name] = train_eval(X[itr], nxt[itr], X[iv], nxt[iv], num_m, dev)
        print(f"{name:22s} top1={res[name]['top1']:.4f} nll={res[name]['nll']:.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"num_milestones": num_m, "variants": res}, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
