#!/usr/bin/env python
"""LaWM native baseline: inverse/forward reconstruction on ADJACENT (and short-horizon)
frames -- its designed regime -- as a reference point for our milestone-jump numbers.

For frame gaps g in {1,2,4,8} (stride-10 sampled frames, so ~0.3s per step):
  identity        : cos(g_t, g_{t+g})                              [how similar frames are]
  forward+oracle  : cos(forward(g_t, inverse(g_t, g_{t+g})), g_{t+g})   [LaWM recon ceiling]
  forward+pred    : cos(forward(g_t, predictor(g_t)), g_{t+g})     [predict next from current only]
Contrast with milestone-jump forward+oracle (~0.93 held-out, ~0.97 in-dist).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov3h_decoder import load_features, l2  # noqa: E402


class MLP(nn.Module):
    def __init__(self, din, dout, hid=512, l2n=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, dout))
        self.l2n = l2n

    def forward(self, x):
        o = self.net(x)
        return F.normalize(o, dim=-1) if self.l2n else o


def build_pairs(E, Fn, gap, eps_set):
    a, b = [], []
    for e in np.unique(E):
        loc = np.where(E == e)[0]
        loc = loc[np.argsort(loc)]
        if len(loc) <= gap:
            continue
        i = loc[:-gap]; j = loc[gap:]
        a.append(i); b.append(j)
    a = np.concatenate(a); b = np.concatenate(b)
    is_tr = np.array([E[x] in eps_set for x in a])
    return a, b, is_tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h")
    ap.add_argument("--gaps", default="1,2,4,8")
    ap.add_argument("--code_dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default="lmwm/outputs/appearance_gen/lawm_adjacent.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    E, FR, Fb = load_features(Path(args.feature_dir))
    Fn = l2(Fb.astype(np.float32))
    ueps = np.unique(E); rng = np.random.default_rng(2026); rng.shuffle(ueps)
    tr_eps = set(ueps[:int(0.8 * len(ueps))].tolist())
    Ft = torch.from_numpy(Fn)

    res = {}
    for gap in [int(x) for x in args.gaps.split(",")]:
        a, b, is_tr = build_pairs(E, Fn, gap, tr_eps)
        tr = np.where(is_tr)[0]; te = np.where(~is_tr)[0]
        inv = MLP(2560, args.code_dim).to(dev); fwd = MLP(1280 + args.code_dim, 1280, l2n=True).to(dev)
        prd = MLP(1280, args.code_dim).to(dev)
        o1 = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=5e-4, weight_decay=1e-5)
        o2 = torch.optim.AdamW(prd.parameters(), lr=5e-4, weight_decay=1e-5)
        for s in range(args.steps):
            bi = tr[np.random.randint(0, len(tr), 1024)]
            ga = Ft[a[bi]].to(dev); gb = Ft[b[bi]].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l1 = (1 - (fwd(torch.cat([ga, inv(torch.cat([ga, gb], -1))], -1)) * gb).sum(-1)).mean()
            o1.zero_grad(); l1.backward(); o1.step()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l2b = (1 - (fwd(torch.cat([ga, prd(ga)], -1)) * gb).sum(-1)).mean()
            o2.zero_grad(); l2b.backward(); o2.step()
        inv.eval(); fwd.eval(); prd.eval()
        with torch.no_grad():
            teb = te[rng.choice(len(te), min(20000, len(te)), replace=False)]
            ga = Ft[a[teb]].to(dev); gb = Ft[b[teb]].to(dev); gbn = Fn[b[teb]]
            ident = float((Fn[a[teb]] * gbn).sum(1).mean())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                orc = fwd(torch.cat([ga, inv(torch.cat([ga, gb], -1))], -1)).float().cpu().numpy()
                fp = fwd(torch.cat([ga, prd(ga)], -1)).float().cpu().numpy()
        res[f"gap_{gap}"] = {"identity_cos": round(ident, 4),
                             "forward_oracle": round(float((orc * gbn).sum(1).mean()), 4),
                             "forward_predicted": round(float((fp * gbn).sum(1).mean()), 4),
                             "n_test": len(teb)}
        print(f"gap {gap}: identity={res[f'gap_{gap}']['identity_cos']} "
              f"forward_oracle={res[f'gap_{gap}']['forward_oracle']} "
              f"forward_pred={res[f'gap_{gap}']['forward_predicted']}", flush=True)

    res["_milestone_jump_reference"] = {"forward_oracle_in_dist": 0.971, "forward_oracle_heldout_visbase": 0.934}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
