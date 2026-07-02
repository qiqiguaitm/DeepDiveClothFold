#!/usr/bin/env python
"""Aligned-to-LaWM quality experiment.

Replicates LaWM's LAM protocol on our pooled DINOv3-H features so our number is
comparable to LaWM's logged `cos_sim_metric`:
  - fixed horizon ~1.6s (LaWM frame_dt_sec=1.6); our cache is 3Hz -> h=5 frames.
  - target = FUTURE frame feature z_{t+h} (continuous), not a discrete milestone.
  - loss = smooth_l1(recon, target, beta=0.1)  (LaWM's loss_type).
  - optimizer = AdamW lr 3e-4 wd 1e-2 + warmup (LaWM).
  - metric = cos_sim(recon, target)  (LaWM's cos_sim_metric).

Two model variants (both faithful to LaWM):
  A. forward_only: z_t -> z_hat_{t+h}      (honest pure forecast; usable at inference)
  B. inverse_forward: u_t = Inv([z_t; z_{t+h}])(code_dim=32, LN); z_hat = Fwd([z_t; u_t])
     -- matches LaWM's reported cos_sim_metric (reconstruction GIVEN the transition
     code, which the policy predicts at deploy time).

Caveat: domain differs (kai0 folding, DINOv3-H pooled) vs LaWM (LIBERO/RoboTwin,
DINOv3 patch), so this aligns the PROTOCOL, not the domain.
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


def mlp(sizes, ln=False):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.GELU()]
        if ln:
            layers += [nn.LayerNorm(sizes[i + 1])]
    layers += [nn.Linear(sizes[-2], sizes[-1])]
    return nn.Sequential(*layers)


class ForwardOnly(nn.Module):
    def __init__(self, d=1280, h=512):
        super().__init__(); self.net = mlp([d, h, h, d], ln=True)

    def forward(self, zt, zf=None):
        return self.net(zt)


class InverseForward(nn.Module):
    """LaWM-style: inverse code from (z_t, z_future); forward decode future from (z_t, u)."""
    def __init__(self, d=1280, code=32, h=512):
        super().__init__()
        self.inv = mlp([2 * d, h, code], ln=True)
        self.u_ln = nn.LayerNorm(code)          # norm_latents: ln
        self.fwd = mlp([d + code, h, h, d], ln=True)

    def forward(self, zt, zf):
        u = self.u_ln(self.inv(torch.cat([zt, zf], -1)))
        return self.fwd(torch.cat([zt, u], -1))


def train_eval(model, Ztr, Ftr, Zva, Fva, dev, steps=3000, bs=2048):
    model = model.to(dev)
    # LaWM optimizer: AdamW lr 3e-4 wd 1e-2 + linear warmup
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    Ztr = torch.from_numpy(Ztr).to(dev); Ftr = torch.from_numpy(Ftr).to(dev)
    Zva = torch.from_numpy(Zva).to(dev); Fva = torch.from_numpy(Fva).to(dev)
    for _ in range(steps):
        bi = torch.randint(0, len(Ztr), (bs,), device=dev)
        recon = model(Ztr[bi], Ftr[bi])
        loss = F.smooth_l1_loss(recon, Ftr[bi], beta=0.1)   # LaWM loss_type + beta
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        rec = model(Zva, Fva)
        cos = F.cosine_similarity(rec, Fva, dim=-1).mean().item()
        l1 = F.l1_loss(rec, Fva).item()
    return {"cos_sim_metric": cos, "l1_loss_metric": l1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--out", default="lmwm/outputs/lawm_align/summary.json", type=Path)
    ap.add_argument("--horizons", type=int, nargs="+", default=[3, 5])  # cache-frames (3Hz): ~1.0s, ~1.7s
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    E, FR, F_ = load_full(args.feature_dir)
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out = {"note": "protocol aligned to LaWM (h~1.6s, future-feature regression, smooth_l1 b=0.1, AdamW 3e-4/1e-2); domain=kai0 DINOv3-H pooled", "horizons": {}}
    for h in args.horizons:
        zt, zf, isval, persist = [], [], [], []
        for ep in np.unique(E):
            loc = np.where(E == ep)[0]; o = loc[np.argsort(FR[loc])]; f = F_[o]
            if len(f) <= h:
                continue
            a, b = f[:-h], f[h:]
            zt.append(a); zf.append(b)
            isval.append(np.full(len(a), ep in val)); persist.append((a * b).sum(1))
        zt = np.concatenate(zt); zf = np.concatenate(zf); isval = np.concatenate(isval); persist = np.concatenate(persist)
        itr = ~isval
        res_fwd = train_eval(ForwardOnly(), zt[itr], zf[itr], zt[isval], zf[isval], dev)
        res_if = train_eval(InverseForward(), zt[itr], zf[itr], zt[isval], zf[isval], dev)
        out["horizons"][f"h{h}_~{h*0.33:.1f}s"] = {
            "persistence_cos_baseline": float(persist[isval].mean()),
            "forward_only_forecast": res_fwd,
            "inverse_forward_LaWM_style": res_if,
            "n_val": int(isval.sum()),
        }
        print(f"h={h} (~{h*0.33:.1f}s): persist={persist[isval].mean():.4f} | "
              f"forward_only cos={res_fwd['cos_sim_metric']:.4f} | "
              f"inverse_forward(LaWM) cos={res_if['cos_sim_metric']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
