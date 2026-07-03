#!/usr/bin/env python
"""Clean cross-appearance test on vis_base (genuinely unseen garments).

Train inverse/forward + predictor + absolute baseline on kai0_base; build (current,
next-medoid) pairs on a sample of vis_base episodes (assign to the existing 37 kai0
milestones); evaluate subgoal cos on vis_base.

  absolute                : feat -> next_medoid (bakes in kai0 appearance)
  forward + oracle code   : forward(current, inverse(current, true_next))  [mechanism ceiling]
  forward + predicted code: forward(current, predictor(feat))              [deployable]
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


def build_visbase_pairs(root, camera, proto, enc, n_eps, stride, dev):
    """Sample vis_base episodes -> encode top_head frames -> assign milestones ->
    episode-medoid stages -> (current_latent, next_medoid_latent) pairs."""
    vids = []
    for info in sorted(glob.glob(str(root / "*/meta/info.json"))):
        base = Path(info).parent.parent
        cs = int(json.loads(Path(info).read_text())["chunks_size"])
        for mp4 in sorted(glob.glob(str(base / f"videos/chunk-*/{camera}/episode_*.mp4"))):
            vids.append(mp4)
    rng = np.random.default_rng(0); rng.shuffle(vids)
    cur_lat, med_lat = [], []
    done = 0
    for mp4 in vids:
        if done >= n_eps:
            break
        cap = cv2.VideoCapture(mp4); frames = []
        i = 0
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
        seq = (latn @ proto.T).argmax(1)                              # milestone per frame
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        medoids = []
        for s, e in zip(st, en):
            rel = (latn[s:e] @ proto[seq[s]]).argmax()
            medoids.append(latn[s + int(rel)])
        for si in range(len(st) - 1):
            for f in range(st[si], en[si]):
                cur_lat.append(latn[f]); med_lat.append(medoids[si + 1])
        done += 1
        if done % 20 == 0:
            print(f"  vis_base {done}/{n_eps} eps, {len(cur_lat)} pairs", flush=True)
    return np.array(cur_lat, np.float32), np.array(med_lat, np.float32)


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
    ap.add_argument("--out", default="lmwm/outputs/appearance_gen/visbase.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    proto = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    z = np.load(args.pairs)
    pooled = z["current"][:, :1280].astype(np.float32)
    state = z["current"][:, -14:].astype(np.float32)
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    n = len(z["current_milestone"]); tri, _ = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    tri = tri.numpy(); tri = tri[ok[tri]]
    feat = np.concatenate([pooled, state], 1).astype(np.float32)
    Xp = torch.from_numpy(pooled); Md = torch.from_numpy(med); Ff = torch.from_numpy(feat)

    inv = MLP(2560, args.code_dim).to(dev); fwd = MLP(1280 + args.code_dim, 1280, l2=True).to(dev)
    predm = MLP(feat.shape[1], args.code_dim).to(dev); absm = MLP(feat.shape[1], 1280, l2=True).to(dev)
    o1 = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=5e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(predm.parameters(), lr=5e-4, weight_decay=1e-5)
    o3 = torch.optim.AdamW(absm.parameters(), lr=5e-4, weight_decay=1e-5)
    print("training on kai0_base ...", flush=True)
    for s in range(args.steps):
        bi = tri[np.random.randint(0, len(tri), 1024)]
        cur = Xp[bi].to(dev); nxt = Md[bi].to(dev); ft = Ff[bi].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            code = inv(torch.cat([cur, nxt], -1)); rec = fwd(torch.cat([cur, code], -1))
            l1 = (1 - (rec * nxt).sum(-1)).mean()
        o1.zero_grad(); l1.backward(); o1.step()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec2 = fwd(torch.cat([cur, predm(ft)], -1)); l2 = (1 - (rec2 * nxt).sum(-1)).mean()
        o2.zero_grad(); l2.backward(); o2.step()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l3 = (1 - (absm(ft) * nxt).sum(-1)).mean()
        o3.zero_grad(); l3.backward(); o3.step()
    for m in (inv, fwd, predm, absm):
        m.eval()

    print("building vis_base pairs ...", flush=True)
    enc = load_encoder("dinov3-h", device=str(dev))
    vc, vm = build_visbase_pairs(args.visbase_root, args.camera, proto, enc, args.n_eps, args.stride, dev)
    print(f"vis_base: {len(vc)} pairs", flush=True)

    def ev(cur_np, med_np, feat_np, tag):
        cur = torch.from_numpy(cur_np).to(dev); nxt = med_np
        ft = torch.from_numpy(feat_np).to(dev) if feat_np is not None else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            orc = fwd(torch.cat([cur, inv(torch.cat([cur, torch.from_numpy(med_np).to(dev)], -1))], -1)).float().cpu().numpy()
        r = {"forward_oracle_code": round(float((orc * nxt).sum(1).mean()), 4)}
        if ft is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ab = absm(ft).float().cpu().numpy(); fp = fwd(torch.cat([cur, predm(ft)], -1)).float().cpu().numpy()
            r["absolute"] = round(float((ab * nxt).sum(1).mean()), 4)
            r["forward_predicted_code"] = round(float((fp * nxt).sum(1).mean()), 4)
        return {tag: {**r, "n": len(cur_np)}}

    # vis_base has no proprio state -> predicted-code/absolute use current-pooled-only feat variant:
    # we trained predm/absm on [pooled|state]; for vis_base build feat with zero state (appearance still in pooled).
    vfeat = np.concatenate([vc, np.zeros((len(vc), 14), np.float32)], 1)
    res = {}
    res.update(ev(pooled[tri[:6000]], med[tri[:6000]], feat[tri[:6000]], "kai0_in_distribution"))
    res.update(ev(vc, vm, vfeat, "visbase_UNSEEN_appearance"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
