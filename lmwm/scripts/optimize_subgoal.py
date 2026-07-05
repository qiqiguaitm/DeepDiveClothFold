#!/usr/bin/env python
"""Optimize the patch-grid subgoal predictor for VLA (û_T). Builds BOTH horizons the user wants:
  --mode nearfuture  : target = grid H steps ahead in the 3Hz index (LaWM-style, dynamics-aware)
  --mode milestone   : target = next-stage medoid grid (semantic milestone+1)

For each, trains forward-from-current (the mechanism the pooled path won with; the old deploy CNN
skipped it): inverse(g_t,g_f)->code (teacher); forward(g_t,code)->g_f; predm(g_t)->code (DEPLOY, no
future peek). Reports oracle-cos (true code), DEPLOY-cos (predm code), persistence-cos, per code_dim.

Beats the unconditional CNN baseline (milestone deploy grid-cos 0.653). One (mode,code_dim) per GPU
for parallel sweep across gf3 8 + local 2.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_lawm_patch import load_index, read_imgs, InverseEnc, ForwardDec  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


class PredM(nn.Module):
    """current grid -> code (deploy predictor, no future peek). Same conv trunk as InverseEnc but
    single-grid input."""
    def __init__(self, din, code_dim, hid=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(din, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(),   # 16->8
            nn.Conv2d(hid, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(),   # 8->4
        )
        self.head = nn.Linear(hid, code_dim); self.ln = nn.LayerNorm(code_dim)

    def forward(self, gt):
        return self.ln(self.head(self.conv(gt).mean((2, 3))))


def build_pairs(E, FR, Fn, proto, mode, horizon, val_eps, seed, pord=None):
    """Return train/val lists of (cur_gidx, target_gidx). Target term per mode:
      nearfuture      - fixed-horizon frame (frame h steps ahead).
      milestone       - V1 "temporal-next milestone": medoid of the NEXT stage in temporal frame order.
      milestone_value - V2 "progress-next milestone": medoid of the episode-library milestone with the
                        smallest CRAVE value strictly greater than the current stage's value (monotone
                        in progress; never regresses). Needs pord (per-prototype CRAVE value).
    """
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        tgt = va if ep in val_eps else tr
        if mode == "nearfuture":
            for i in range(len(order) - horizon):
                tgt.append((int(order[i]), int(order[i + horizon])))
            continue
        # milestone / milestone_value share the SAME stage segmentation + medoids; only target differs
        seq = (Fn[order] @ proto.T).argmax(1)
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        seg_m, seg_med, seg_last = [], [], []
        for s, e in zip(st, en):
            m = int(seq[s]); sub = order[s:e]; med = int(sub[(Fn[sub] @ proto[m]).argmax()])
            seg_m.append(m); seg_med.append(med); seg_last.append(int(order[e - 1]))
        if mode == "milestone":                                          # V1: temporal-next
            for i in range(len(seg_m) - 1):
                tgt.append((seg_last[i], seg_med[i + 1]))
        else:                                                            # V2: progress(value)-next
            lib = {}                                                     # distinct milestone -> (value, medoid)
            for m, med in zip(seg_m, seg_med):
                lib.setdefault(m, (float(pord[m]), med))
            libsorted = sorted(lib.values())                             # ascending CRAVE value
            for i, m in enumerate(seg_m):
                v = float(pord[m])
                nxt = [med for (val, med) in libsorted if val > v + 1e-6]
                if nxt:
                    tgt.append((seg_last[i], nxt[0]))
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nearfuture", "milestone", "milestone_value"], required=True)
    ap.add_argument("--horizon", type=int, default=5, help="nearfuture: steps ahead in 3Hz index (5≈1.7s)")
    ap.add_argument("--code_dim", type=int, default=64)
    ap.add_argument("--arch", choices=["cnn", "convattn", "transformer"], default="cnn")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--code_head", choices=["deterministic", "vae"], default="deterministic")
    ap.add_argument("--kl_weight", type=float, default=1e-3)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_val", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="", help="json out path; default derived from mode/horizon/code_dim")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    szt = f"_{args.arch}" + (f"_w{args.width}d{args.depth}" if args.arch != "cnn" else "") + ("_vae" if args.code_head == "vae" else "")
    tag = f"{args.mode}{'_h'+str(args.horizon) if args.mode=='nearfuture' else ''}_cd{args.code_dim}{szt}"
    out = Path(args.out) if args.out else Path(f"lmwm/outputs/subgoal_opt/{tag}.json")

    E, FR, Fn = load_index(args.feature_dir)
    _rg = np.load("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = _rg["prototype_table"].astype(np.float32); pord = _rg["pord"].astype(np.float32)
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())

    tr, va = build_pairs(E, FR, Fn, proto, args.mode, args.horizon, val_eps, args.seed, pord=pord)
    tr = tr[:args.n_train]; va = va[:args.n_val]
    uniq = sorted(set([g for p in tr + va for g in p])); u2k = {g: k for k, g in enumerate(uniq)}
    print(f"[{tag}] {len(tr)} train + {len(va)} val pairs, {len(uniq)} unique frames", flush=True)

    enc_imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, np.array(uniq), 256, 128)
    enc = load_encoder("dinov3-h", device=dev)
    grids = enc.encode_grid(enc_imgs).astype(np.float32); din = grids.shape[1]
    gmu, gsd = grids.mean(), grids.std() + 1e-6
    GZ = torch.from_numpy(((grids - gmu) / gsd).astype(np.float32))

    tra = torch.from_numpy(np.array([u2k[c] for c, _ in tr])); trb = torch.from_numpy(np.array([u2k[n] for _, n in tr]))
    vaa = np.array([u2k[c] for c, _ in va]); vab = np.array([u2k[n] for _, n in va])

    import torch.nn as nn
    from lam_arch import build_lam, nparams
    inv, fwd, predm = build_lam(args.arch, din, args.code_dim, args.width, args.depth)
    inv, fwd, predm = inv.to(dev), fwd.to(dev), predm.to(dev)
    vae = args.code_head == "vae"; cd = args.code_dim
    vae_inv = nn.Linear(cd, 2 * cd).to(dev) if vae else None      # code -> (mu, logvar) VAE heads (LaWM-style)
    vae_pm = nn.Linear(cd, 2 * cd).to(dev) if vae else None

    def reparam(head, h, sample):
        mu, lv = head(h).chunk(2, -1); lv = lv.clamp(-8, 8)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if sample else mu
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        return z, mu, lv, kl

    xtra = (nparams(vae_inv) + nparams(vae_pm)) if vae else 0
    n_params = nparams(inv) + nparams(fwd) + nparams(predm) + xtra
    n_deploy = nparams(predm) + nparams(fwd) + (nparams(vae_pm) if vae else 0)
    print(f"[{tag}] arch={args.arch} head={args.code_head} params total={n_params/1e6:.1f}M deploy={n_deploy/1e6:.1f}M", flush=True)
    p1 = list(inv.parameters()) + list(fwd.parameters()) + (list(vae_inv.parameters()) if vae else [])
    p2 = list(predm.parameters()) + (list(vae_pm.parameters()) if vae else [])
    o1 = torch.optim.AdamW(p1, lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(p2, lr=2e-4, weight_decay=1e-5)
    print(f"[{tag}] training ...", flush=True)
    for step in range(args.steps):
        sel = torch.randint(0, len(tra), (32,))
        gt = GZ[tra[sel]].to(dev); gf = GZ[trb[sel]].to(dev)
        if vae:
            z, _, _, kl = reparam(vae_inv, inv(gt, gf), True)
            l1 = F.smooth_l1_loss(fwd(gt, z), gf, beta=1.0) + args.kl_weight * kl
        else:
            l1 = F.smooth_l1_loss(fwd(gt, inv(gt, gf)), gf, beta=1.0)
        o1.zero_grad(); l1.backward(); o1.step()
        if vae:
            zp, _, _, klp = reparam(vae_pm, predm(gt), True)
            l2 = F.smooth_l1_loss(fwd(gt, zp), gf, beta=1.0) + args.kl_weight * klp
        else:
            l2 = F.smooth_l1_loss(fwd(gt, predm(gt)), gf, beta=1.0)
        o2.zero_grad(); l2.backward(); o2.step()
    for m in [inv, fwd, predm] + ([vae_inv, vae_pm] if vae else []):
        m.eval()

    def cos(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    co, cd_, cp, cbest = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(vaa), 256):
            gt = GZ[vaa[s:s + 256]].to(dev); gf = GZ[vab[s:s + 256]].to(dev); gtr = f(gf)
            if vae:
                _, mu_i, _, _ = reparam(vae_inv, inv(gt, gf), False); oracle = fwd(gt, mu_i)
                _, mu_p, lv_p, _ = reparam(vae_pm, predm(gt), False); deploy = fwd(gt, mu_p)
                best = None
                for _ in range(8):                                # best-of-8 posterior samples = multimodal coverage
                    ck = cos(f(fwd(gt, mu_p + torch.randn_like(mu_p) * (0.5 * lv_p).exp())), gtr)
                    best = ck if best is None else np.maximum(best, ck)
                cbest.append(best)
            else:
                oracle = fwd(gt, inv(gt, gf)); deploy = fwd(gt, predm(gt))
            co.append(cos(f(oracle), gtr)); cd_.append(cos(f(deploy), gtr)); cp.append(cos(f(gt), gtr))
    res = {"mode": args.mode, "horizon": args.horizon if args.mode == "nearfuture" else None,
           "code_dim": args.code_dim, "arch": args.arch, "width": args.width, "depth": args.depth,
           "code_head": args.code_head, "kl_weight": args.kl_weight,
           "params_M": round(n_params / 1e6, 1), "deploy_params_M": round(n_deploy / 1e6, 1),
           "n_train": len(tr), "n_val": len(va),
           "oracle_grid_cos": round(float(np.concatenate(co).mean()), 4),
           "deploy_grid_cos": round(float(np.concatenate(cd_).mean()), 4),
           "deploy_bestof8_cos": round(float(np.concatenate(cbest).mean()), 4) if vae else None,
           "persistence_grid_cos": round(float(np.concatenate(cp).mean()), 4),
           "baseline_uncond_cnn_deploy": 0.653}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    ckd = {"inv": inv.state_dict(), "fwd": fwd.state_dict(), "predm": predm.state_dict(),
           "code_dim": args.code_dim, "din": din, "gmu": float(gmu), "gsd": float(gsd),
           "mode": args.mode, "horizon": args.horizon, "arch": args.arch, "width": args.width,
           "depth": args.depth, "code_head": args.code_head}
    if vae:
        ckd["vae_inv"] = vae_inv.state_dict(); ckd["vae_pm"] = vae_pm.state_dict()
    torch.save(ckd, out.with_suffix(".pt"))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
