#!/usr/bin/env python
"""Does conditioning on the PREVIOUS milestone (path history) improve next-milestone
prediction beyond the current frame? Tests H2 (cluster aliasing) with a signal the
current frame cannot contain: which visit of an aliased cluster this is.

Two bias-free evaluations on held-out episodes (predict real next-unique milestone):
  1. Discrete count model: ctx in {cur, cur+prev, cur+time, cur+prev+time}.
  2. kNN over the continuous frame feature: frame-only vs frame + prev-milestone
     one-hot (does prev beat the frame-representation ceiling?).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


def load_full(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, t, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), idx["T"].astype(np.float64), int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16); valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard); gi = z["gidx"].astype(np.int64)
        feat[gi] = z["feat"]; valid[gi] = z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32); fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], t[valid], fv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", default="lmwm/outputs/ceiling_diag/prev_milestone.json", type=Path)
    ap.add_argument("--n_tbins", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    g = np.load(args.graph_npz); proto = g["prototype_table"].astype(np.float32); num_m = len(proto)
    E, FR, T, F = load_full(args.feature_dir)
    assign = np.empty(len(F), dtype=np.int64)
    for i in range(0, len(F), 32768):
        assign[i:i + 32768] = (F[i:i + 32768] @ proto.T).argmax(1)
    START = num_m  # "no previous stage" id

    # Per stage i (with a next stage): last-frame feature, cur_m, prev_m, next_m, tfrac.
    feats, cur_a, prev_a, next_a, tfrac_a, ep_a = [], [], [], [], [], []
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]; seq = assign[order]
        tn = T[order]; tf = (tn - tn.min()) / (tn.max() - tn.min() + 1e-9)
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        ms = [int(seq[s]) for s in st]
        for i in range(len(ms) - 1):
            last = en[i] - 1
            feats.append(F[order[last]]); cur_a.append(ms[i]); prev_a.append(ms[i - 1] if i > 0 else START)
            next_a.append(ms[i + 1]); tfrac_a.append(float(tf[last])); ep_a.append(ep)
    feats = np.stack(feats); cur_a = np.array(cur_a); prev_a = np.array(prev_a)
    next_a = np.array(next_a); tfrac_a = np.array(tfrac_a); ep_a = np.array(ep_a)

    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    isval = np.array([e in val_eps for e in ep_a])

    # ---- 1. discrete count model, bias-free held-out ----
    def tbin(tf): return min(args.n_tbins - 1, int(tf * args.n_tbins))

    def run_ctx(name, keyfn, alpha=0.5):
        tables = defaultdict(Counter)
        for i in np.where(~isval)[0]:
            tables[keyfn(i)][int(next_a[i])] += 1
        nll = hit = 0.0; tot = 0; logM = np.log(num_m)
        for i in np.where(isval)[0]:
            c = tables.get(keyfn(i)); y = int(next_a[i])
            if not c:
                nll += logM; tot += 1; continue
            tot_c = sum(c.values()) + alpha * num_m
            nll += -np.log((c.get(y, 0) + alpha) / tot_c)
            hit += (c.most_common(1)[0][0] == y); tot += 1
        return {"top1": hit / tot, "nll": nll / tot}

    discrete = {
        "cur": run_ctx("cur", lambda i: int(cur_a[i])),
        "cur+prev": run_ctx("cur+prev", lambda i: (int(cur_a[i]), int(prev_a[i]))),
        "cur+time": run_ctx("cur+time", lambda i: (int(cur_a[i]), tbin(tfrac_a[i]))),
        "cur+prev+time": run_ctx("cur+prev+time", lambda i: (int(cur_a[i]), int(prev_a[i]), tbin(tfrac_a[i]))),
    }
    for k, v in discrete.items():
        print(f"[discrete] {k:16s} top1={v['top1']:.4f} nll={v['nll']:.4f}", flush=True)

    # ---- 2. kNN continuous: frame vs frame+prev-onehot ----
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    knn = {}
    for tag, alpha in [("frame_only", 0.0), ("frame+prev(a0.5)", 0.5), ("frame+prev(a1.0)", 1.0)]:
        oh = np.zeros((len(feats), num_m + 1), dtype=np.float32)
        oh[np.arange(len(feats)), prev_a] = alpha
        base = np.concatenate([feats, oh], axis=1) if alpha > 0 else feats
        base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-8)
        keys = torch.from_numpy(base[~isval]).to(dev); ktgt = torch.from_numpy(next_a[~isval]).to(dev)
        q = torch.from_numpy(base[isval]).to(dev); qy = next_a[isval]
        k = 50; top1 = 0; nll = 0.0; tot = 0
        for s in range(0, len(q), 1024):
            sim = q[s:s + 1024] @ keys.T
            _, ii = sim.topk(k, 1)
            votes = torch.zeros(ii.shape[0], num_m, device=dev)
            votes.scatter_add_(1, ktgt[ii], torch.ones_like(ii, dtype=torch.float))
            p = (votes + 0.1) / (votes.sum(1, keepdim=True) + 0.1 * num_m)
            yb = torch.from_numpy(qy[s:s + 1024]).to(dev)
            top1 += int((p.argmax(1) == yb).sum()); nll += float(-torch.log(p[torch.arange(len(ii)), yb]).sum()); tot += len(ii)
        knn[tag] = {"top1": top1 / tot, "nll": nll / tot}
        print(f"[kNN k=50] {tag:18s} top1={top1/tot:.4f} nll={nll/tot:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"discrete": discrete, "knn": knn, "num_milestones": num_m}, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
