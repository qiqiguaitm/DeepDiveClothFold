#!/usr/bin/env python
"""Appearance-invariant subgoal via TRANSITION CODE + forward(current, code).

The code is a learned embedding of the (current_milestone -> next_milestone) pair --
pure milestone ids, zero appearance -- so forward(current_obs, code) renders the subgoal
in the current garment's appearance. Verified on unseen vis_base garments.

Paths compared (subgoal cos vs true next-medoid):
  absolute                     : feat -> next_medoid (bakes in kai0 appearance)
  forward + inverse oracle     : forward(cur, inverse(cur, true_next))     [uses true next latent]
  forward + transition[cur,TRUE next_m]   : code from milestone pair only  [appearance-invariant]
  forward + transition[cur,GREEDY next_m] : deployable (graph greedy next) [appearance-invariant]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lmwm.data import split_indices  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


class MLP(nn.Module):
    def __init__(self, din, dout, hid=512, l2=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, dout))
        self.l2 = l2

    def forward(self, x):
        o = self.net(x)
        return F.normalize(o, dim=-1) if self.l2 else o


def build_visbase(root, camera, proto, enc, n_eps, stride):
    vids = []
    for info in sorted(glob.glob(str(root / "*/meta/info.json"))):
        base = Path(info).parent.parent
        for mp4 in sorted(glob.glob(str(base / f"videos/chunk-*/{camera}/episode_*.mp4"))):
            vids.append(mp4)
    rng = np.random.default_rng(0); rng.shuffle(vids)
    cur, med, cm, nm = [], [], [], []
    done = 0
    for mp4 in vids:
        if done >= n_eps:
            break
        cap = cv2.VideoCapture(mp4); frames = []; i = 0
        while True:
            okf, im = cap.read()
            if not okf:
                break
            if i % stride == 0:
                frames.append(cv2.resize(im[:, :, ::-1], (256, 256)))
            i += 1
        cap.release()
        if len(frames) < 4:
            continue
        lat = enc.encode_pooled(np.stack(frames)).astype(np.float32)
        latn = lat / (np.linalg.norm(lat, axis=1, keepdims=True) + 1e-8)
        seq = (latn @ proto.T).argmax(1)
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        med_s = [latn[s + int((latn[s:e] @ proto[seq[s]]).argmax())] for s, e in zip(st, en)]
        for si in range(len(st) - 1):
            for f in range(st[si], en[si]):
                cur.append(latn[f]); med.append(med_s[si + 1]); cm.append(seq[st[si]]); nm.append(seq[st[si + 1]])
        done += 1
        if done % 20 == 0:
            print(f"  vis_base {done}/{n_eps} eps, {len(cur)} pairs", flush=True)
    return (np.array(cur, np.float32), np.array(med, np.float32),
            np.array(cm, np.int64), np.array(nm, np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--visbase_root", default="kai0/data/Task_A/vis_base/v1", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_eps", type=int, default=120)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--code_dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="lmwm/outputs/appearance_gen/transition_code.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    g = np.load(args.graph_npz); proto = g["prototype_table"].astype(np.float32); num_m = len(proto)
    trans = g["transition_probs"].astype(np.float64); greedy_next = (trans / trans.sum(1, keepdims=True).clip(1e-12)).argmax(1)
    z = np.load(args.pairs)
    pooled = z["current"][:, :1280].astype(np.float32); state = z["current"][:, -14:].astype(np.float32)
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    cm = z["current_milestone"].astype(np.int64); nm = z["future_milestone"].astype(np.int64)
    n = len(cm); tri, _ = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode"); tri = tri.numpy(); tri = tri[ok[tri]]
    feat = np.concatenate([pooled, state], 1).astype(np.float32)
    Xp = torch.from_numpy(pooled); Md = torch.from_numpy(med); Ff = torch.from_numpy(feat)
    CM = torch.from_numpy(cm); NM = torch.from_numpy(nm)

    inv = MLP(2560, args.code_dim).to(dev); fwd = MLP(1280 + args.code_dim, 1280, l2=True).to(dev)
    tcode = nn.Embedding(num_m * num_m, args.code_dim).to(dev)
    absm = MLP(feat.shape[1], 1280, l2=True).to(dev)
    o1 = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=5e-4, weight_decay=1e-5)
    o3 = torch.optim.AdamW(absm.parameters(), lr=5e-4, weight_decay=1e-5)
    print("stage1: inverse/forward + absolute ...", flush=True)
    for s in range(args.steps):
        bi = tri[np.random.randint(0, len(tri), 1024)]
        cur = Xp[bi].to(dev); nxt = Md[bi].to(dev); ft = Ff[bi].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l1 = (1 - (fwd(torch.cat([cur, inv(torch.cat([cur, nxt], -1))], -1)) * nxt).sum(-1)).mean()
        o1.zero_grad(); l1.backward(); o1.step()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l3 = (1 - (absm(ft) * nxt).sum(-1)).mean()
        o3.zero_grad(); l3.backward(); o3.step()
    for m in (inv, fwd, absm):
        m.eval()
    for p in fwd.parameters():
        p.requires_grad_(False)

    print("stage2: transition-code table (forward frozen) ...", flush=True)
    o2 = torch.optim.AdamW(tcode.parameters(), lr=1e-3, weight_decay=0.0)
    for s in range(args.steps):
        bi = tri[np.random.randint(0, len(tri), 1024)]
        cur = Xp[bi].to(dev); nxt = Md[bi].to(dev); idx = (CM[bi] * num_m + NM[bi]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c = tcode(idx); l = (1 - (fwd(torch.cat([cur, c], -1)) * nxt).sum(-1)).mean()
        o2.zero_grad(); l.backward(); o2.step()
    tcode.eval()

    print("building vis_base ...", flush=True)
    enc = load_encoder("dinov3-h", device=str(dev))
    vc, vm, vcm, vnm = build_visbase(args.visbase_root, args.camera, proto, enc, args.n_eps, args.stride)
    print(f"vis_base: {len(vc)} pairs", flush=True)

    def evalset(cur_np, med_np, cm_np, nm_np, feat_np, tag):
        cur = torch.from_numpy(cur_np).to(dev); nxt = med_np
        gnext = greedy_next[cm_np]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            orc = fwd(torch.cat([cur, inv(torch.cat([cur, torch.from_numpy(med_np).to(dev)], -1))], -1)).float().cpu().numpy()
            tc_true = fwd(torch.cat([cur, tcode(torch.from_numpy(cm_np * num_m + nm_np).to(dev))], -1)).float().cpu().numpy()
            tc_greedy = fwd(torch.cat([cur, tcode(torch.from_numpy(cm_np * num_m + gnext).to(dev))], -1)).float().cpu().numpy()
        r = {"forward_inverse_oracle": round(float((orc * nxt).sum(1).mean()), 4),
             "forward_transition_TRUE": round(float((tc_true * nxt).sum(1).mean()), 4),
             "forward_transition_GREEDY": round(float((tc_greedy * nxt).sum(1).mean()), 4)}
        if feat_np is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ab = absm(torch.from_numpy(feat_np).to(dev)).float().cpu().numpy()
            r["absolute"] = round(float((ab * nxt).sum(1).mean()), 4)
        r["n"] = len(cur_np)
        return {tag: r}

    res = {}
    res.update(evalset(pooled[tri[:6000]], med[tri[:6000]], cm[tri[:6000]], nm[tri[:6000]], feat[tri[:6000]], "kai0_in_distribution"))
    res.update(evalset(vc, vm, vcm, vnm, None, "visbase_UNSEEN_appearance"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
