#!/usr/bin/env python
"""Validate the multimodality hypothesis: a conditional flow-matching subgoal head
(samples multiple milestone+1 latents) vs a deterministic regression head.

If the future is genuinely multimodal (~13 branches), a regression head averages the
modes and caps at cos ~0.874; a generative head that SAMPLES should, over N samples,
cover the true mode -> best-of-N cos should exceed 0.874 and approach the L2 oracle
(~0.90). Pooled augin data (fast, full data, no image encoding).

Reports, on held-out:
  regression   : cos mean (single deterministic output)
  flow (diff.) : single-sample cos, best-of-8 cos, centroid-of-8 cos, coverage(cos>0.9)
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


def timeemb(t, d=128):
    half = d // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class RegHead(nn.Module):
    def __init__(self, din, ld=1280, hid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, ld))

    def forward(self, cond):
        return F.normalize(self.net(cond), dim=-1)


class FlowHead(nn.Module):
    """Velocity field v(x_t, t, cond) for rectified flow over the (standardized) latent."""

    def __init__(self, din, ld=1280, hid=1024):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid))
        self.tproj = nn.Sequential(nn.Linear(128, hid), nn.GELU())
        self.net = nn.Sequential(nn.Linear(ld + hid + hid, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, ld))

    def forward(self, x, t, cond_emb):
        return self.net(torch.cat([x, self.tproj(timeemb(t)), cond_emb], -1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--nsample", type=int, default=8)
    ap.add_argument("--ode_steps", type=int, default=10)
    ap.add_argument("--out", default="lmwm/outputs/diffusion_subgoal/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    X = z["current"].astype(np.float32); din = X.shape[1]
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    ti, vi = ti[ok[ti]], vi[ok[vi]]

    # standardize target latent so per-dim variance ~1 (flow-matching needs data~N scale)
    mu = med[ti].mean(0); sd = med[ti].std(0) + 1e-6
    Xt = torch.from_numpy(X[ti]); Mt = torch.from_numpy((med[ti] - mu) / sd)
    Xv = torch.from_numpy(X[vi]).to(dev); Mv = torch.from_numpy(med[vi]).to(dev)  # Mv = true UNIT medoid
    muT = torch.from_numpy(mu).to(dev); sdT = torch.from_numpy(sd).to(dev)
    ntr = len(ti); ld = med.shape[1]

    def cosstat(cos):
        return {"mean": round(float(cos.mean()), 4), "std": round(float(cos.std()), 4),
                "p05": round(float(np.percentile(cos, 5)), 4), "cov_gt0.9": round(float((cos > 0.9).mean()), 4)}

    # ---------- regression baseline ----------
    reg = RegHead(din).to(dev)
    opt = torch.optim.AdamW(reg.parameters(), lr=5e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
    for s in range(args.steps):
        bi = torch.randint(0, ntr, (1024,))
        pred = reg(Xt[bi].to(dev))
        tgt = F.normalize(Mt[bi].to(dev) * sdT + muT, dim=-1)  # back to unit target
        loss = (1 - (pred * tgt).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    reg.eval()
    with torch.no_grad():
        rc = []
        for s in range(0, len(vi), 8192):
            rc.append((reg(Xv[s:s + 8192]) * Mv[s:s + 8192]).sum(-1).cpu().numpy())
    reg_cos = np.concatenate(rc)
    print(f"regression  cos={reg_cos.mean():.4f}", flush=True)

    # ---------- flow-matching head ----------
    flow = FlowHead(din).to(dev)
    opt = torch.optim.AdamW(flow.parameters(), lr=5e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
    for s in range(args.steps):
        bi = torch.randint(0, ntr, (1024,))
        x1 = Mt[bi].to(dev); cond = flow.cond(Xt[bi].to(dev))
        x0 = torch.randn_like(x1); t = torch.rand(len(x1), device=dev)
        xt = (1 - t)[:, None] * x0 + t[:, None] * x1
        v = flow.net(torch.cat([xt, flow.tproj(timeemb(t)), cond], -1))
        loss = F.mse_loss(v, x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    flow.eval()

    @torch.no_grad()
    def sample(condX, nsamp):
        cond = flow.cond(condX)                             # [B,hid]
        B = condX.shape[0]
        outs = []
        for _ in range(nsamp):
            x = torch.randn(B, ld, device=dev)
            for k in range(args.ode_steps):
                t = torch.full((B,), k / args.ode_steps, device=dev)
                x = x + (1.0 / args.ode_steps) * flow.net(torch.cat([x, flow.tproj(timeemb(t)), cond], -1))
            outs.append(F.normalize(x * sdT + muT, dim=-1))   # un-standardize -> unit
        return torch.stack(outs, 1)                          # [B,nsamp,ld]

    single, best, centroid = [], [], []
    with torch.no_grad():
        for s in range(0, len(vi), 2048):
            cx = Xv[s:s + 2048]; tgt = Mv[s:s + 2048]
            samp = sample(cx, args.nsample)                  # [b,N,ld]
            coss = (samp * tgt[:, None]).sum(-1)             # [b,N]
            single.append(coss[:, 0].cpu().numpy())
            best.append(coss.max(1).values.cpu().numpy())
            cen = F.normalize(samp.mean(1), dim=-1)
            centroid.append((cen * tgt).sum(-1).cpu().numpy())
    single, best, centroid = map(np.concatenate, (single, best, centroid))
    print(f"flow single={single.mean():.4f} best-of-{args.nsample}={best.mean():.4f} centroid={centroid.mean():.4f}", flush=True)

    res = {"regression": {"cos_mean": round(float(reg_cos.mean()), 4)},
           "flow_single_sample": cosstat(single),
           f"flow_best_of_{args.nsample}": cosstat(best),
           "flow_centroid": cosstat(centroid),
           "L2_oracle_reference": 0.90}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
