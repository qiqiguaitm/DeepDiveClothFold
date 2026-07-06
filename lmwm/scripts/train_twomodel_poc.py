#!/usr/bin/env python
"""Two-model PoC (local), in the FAITHFUL pi0.5 SigLIP space (E2 decision):

  Stage-1 (MDN, multimodal):  gist(current) -> mixture-of-K over next-milestone gist z_hat   [identity]
  Stage-2 (deterministic):    (current grid, z_hat) -> next-milestone grid                   [grounding]

Rationale (from analyses C/E1): identity is multimodal (~3 branches) -> Stage-1 must be a distribution
(MDN); appearance is ~unimodal given identity -> Stage-2 can be deterministic. best-of-N sampling lives
in Stage-1 (identity), NOT on the grid code (which gave only +0.02).

Money comparison: run --K 4 (multimodal) vs --K 1 (== unimodal regression). best-of-N deploy grid-cos
should lift for K=4 and NOT for K=1 -> confirms the multimodality is recoverable when sampled on the
right (identity) axis. Compare single-sample deploy to E2 single-stage 0.716.

Stage-2 is trained with the TRUE next gist (teacher); at eval z_hat comes from Stage-1 (deploy).
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import load_index, read_imgs, ForwardDec  # noqa: E402
from optimize_subgoal import build_pairs  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402

PI05_NPZ = "/vePFS/tim/workspace/openpi_cache/paligemma_weights/pt_224.npz"
PI05_NPZ_GF3 = "/vePFS-North-E/vis_robot/openpi_cache/paligemma_weights/pt_224.npz"


class MDN(nn.Module):
    """gist -> mixture of K diagonal Gaussians over next-milestone gist (dim D)."""
    def __init__(self, D, K, hid=1024):
        super().__init__()
        self.K, self.D = K, D
        self.trunk = nn.Sequential(nn.Linear(D, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        self.pi = nn.Linear(hid, K)
        self.mu = nn.Linear(hid, K * D)
        self.ls = nn.Linear(hid, K * D)

    def forward(self, x):
        h = self.trunk(x); B = x.shape[0]
        return (self.pi(h),
                self.mu(h).view(B, self.K, self.D),
                self.ls(h).view(B, self.K, self.D).clamp(-6, 4))

    def nll(self, x, y):                                   # y (B,D)
        logit, mu, ls = self(x)
        logpi = F.log_softmax(logit, -1)                   # (B,K)
        var = (2 * ls).exp()
        comp = -0.5 * (((y[:, None] - mu) ** 2) / var + 2 * ls + np.log(2 * np.pi)).sum(-1)  # (B,K)
        return -(torch.logsumexp(logpi + comp, -1)).mean()

    @torch.no_grad()
    def deploy_mean(self, x):                              # highest-pi component mean
        logit, mu, _ = self(x)
        return mu[torch.arange(len(x)), logit.argmax(-1)]

    @torch.no_grad()
    def sample(self, x, n):                                # (n,B,D)
        logit, mu, ls = self(x)
        pi = F.softmax(logit, -1); out = []
        for _ in range(n):
            k = torch.multinomial(pi, 1).squeeze(1)
            m = mu[torch.arange(len(x)), k]; s = ls[torch.arange(len(x)), k].exp()
            out.append(m + torch.randn_like(m) * s)
        return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4, help="MDN components; K=1 == unimodal regression")
    ap.add_argument("--mode", default="milestone_viterbi", choices=["milestone_viterbi", "milestone_value"])
    ap.add_argument("--code_dim", type=int, default=1152, help="z_hat = next gist dim (== SigLIP dim)")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_val", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bestof", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid_cache", default="", help="load precomputed SigLIP grids+pairs (gf3: no videos/pt_224 needed)")
    ap.add_argument("--save_grid_cache", default="", help="encode then save cache to this .npz and exit")
    ap.add_argument("--pi05_npz", default="", help="pt_224.npz path; auto local/gf3 if empty")
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)
    msid = None                                                             # per-uniq-frame milestone id (identity eval)

    if args.grid_cache:                                                      # gf3 path: everything precomputed
        z = np.load(args.grid_cache)
        GZ = torch.from_numpy(z["GZ"].astype(np.float32)); gist = GZ.mean((2, 3))
        gmu, gsd = float(z["gmu"]), float(z["gsd"]); din = GZ.shape[1]
        tra, trb, vaa, vab = z["tra"], z["trb"], z["vaa"], z["vab"]
        print(f"[K={args.K}] cache {args.grid_cache}: {len(tra)} train + {len(vaa)} val, grid {tuple(GZ.shape)}", flush=True)
    else:
        E, FR, Fn = load_index(args.feature_dir)
        rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
        proto = rg["prototype_table"].astype(np.float32); pord = rg["pord"].astype(np.float32)
        rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
        val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
        tr, va = build_pairs(E, FR, Fn, proto, args.mode, 5, val_eps, args.seed, pord=pord)
        tr = tr[:args.n_train]; va = va[:args.n_val]
        uniq = sorted(set([g for p in tr + va for g in p])); u2k = {g: k for k, g in enumerate(uniq)}
        print(f"[K={args.K}] {len(tr)} train + {len(va)} val, {len(uniq)} frames", flush=True)

        msid = (Fn[np.array(uniq)] @ proto.T).argmax(1)                     # DINOv3 milestone id per uniq frame
        imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, np.array(uniq), 224, 128)
        enc = SiglipBigVision(npz, device=dev)
        grids = enc.encode_grid(imgs); din = grids.shape[1]                 # (N,1152,16,16)
        gmu, gsd = grids.mean(), grids.std() + 1e-6
        GZ = torch.from_numpy(((grids - gmu) / gsd).astype(np.float32))     # CPU
        gist = GZ.mean((2, 3))                                              # (N,1152) pooled, normalized
        tra = np.array([u2k[c] for c, _ in tr]); trb = np.array([u2k[n] for _, n in tr])
        vaa = np.array([u2k[c] for c, _ in va]); vab = np.array([u2k[n] for _, n in va])
        if args.save_grid_cache:
            Path(args.save_grid_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_grid_cache, GZ=GZ.numpy().astype(np.float16), gmu=gmu, gsd=gsd,
                     tra=tra, trb=trb, vaa=vaa, vab=vab)
            print(f"saved cache -> {args.save_grid_cache} (grid {tuple(GZ.shape)})", flush=True); return

    mdn = MDN(din, args.K).to(dev)
    fwd = ForwardDec(din, args.code_dim).to(dev)
    om = torch.optim.AdamW(mdn.parameters(), lr=2e-4, weight_decay=1e-5)
    of = torch.optim.AdamW(fwd.parameters(), lr=2e-4, weight_decay=1e-5)
    for step in range(args.steps):
        sel = torch.randint(0, len(tra), (64,))
        gcur = gist[tra[sel]].to(dev); gnext = gist[trb[sel]].to(dev)
        lm = mdn.nll(gcur, gnext)                                           # Stage-1 MDN
        om.zero_grad(); lm.backward(); om.step()
        Gcur = GZ[tra[sel]].to(dev); Gnext = GZ[trb[sel]].to(dev)
        lf = F.smooth_l1_loss(fwd(Gcur, gnext), Gnext, beta=1.0)            # Stage-2 teacher = true gist
        of.zero_grad(); lf.backward(); of.step()
    mdn.eval(); fwd.eval()

    def cos(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    unn = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    co, cdep, cbest, cp = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(vaa), 128):
            Gcur = GZ[vaa[s:s + 128]].to(dev); Gnext = GZ[vab[s:s + 128]].to(dev)
            gcur = gist[vaa[s:s + 128]].to(dev); gnext = gist[vab[s:s + 128]].to(dev)
            gtr = unn(Gnext)
            co.append(cos(unn(fwd(Gcur, gnext)), gtr))                       # oracle (true gist)
            zdep = mdn.deploy_mean(gcur)
            cdep.append(cos(unn(fwd(Gcur, zdep)), gtr))                      # deploy (Stage-1 mean)
            best = None
            for zi in mdn.sample(gcur, args.bestof):                        # best-of-N (Stage-1 samples)
                ck = cos(unn(fwd(Gcur, zi)), gtr)
                best = ck if best is None else np.maximum(best, ck)
            cbest.append(best)
            cp.append(cos(unn(Gcur), gtr))                                   # persistence
    identity = None                                                         # top-N identity hit (the axis where MM shows)
    if msid is not None:
        Mn = int(msid.max()) + 1; gnp = gist.numpy()
        protos = np.stack([gnp[msid == m].mean(0) if (msid == m).any() else np.zeros(din, np.float32)
                           for m in range(Mn)])
        protosL = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        true_next = msid[vab]; hits = {1: 0, 2: 0, 3: 0, 5: 0}
        with torch.no_grad():
            for s in range(0, len(vaa), 256):
                samp = mdn.sample(gist[vaa[s:s + 256]].to(dev), 8).cpu().numpy()   # (8,B,din)
                tn = true_next[s:s + 256]
                pred = np.stack([(zi / (np.linalg.norm(zi, 1, keepdims=True) + 1e-8)) @ protosL.T
                                 for zi in samp]).argmax(-1)                       # (8,B)
                for bi in range(len(tn)):
                    rk = list(dict.fromkeys(pred[:, bi].tolist()))                 # unique in sample order
                    for N in hits:
                        hits[N] += int(tn[bi] in rk[:N])
        identity = {f"top{N}_hit": round(hits[N] / len(true_next), 4) for N in hits}

    res = {"K": args.K, "space": "pi05_siglip_faithful", "code_dim": args.code_dim, "identity_topN": identity,
           "n_train": len(tr), "n_val": len(va), "bestof": args.bestof,
           "oracle_grid_cos(true gist)": round(float(np.concatenate(co).mean()), 4),
           "deploy_grid_cos(Stage1 mean)": round(float(np.concatenate(cdep).mean()), 4),
           "bestof%d_grid_cos" % args.bestof: round(float(np.concatenate(cbest).mean()), 4),
           "persistence": round(float(np.concatenate(cp).mean()), 4),
           "bestof_gain_over_deploy": round(float(np.concatenate(cbest).mean() - np.concatenate(cdep).mean()), 4),
           "ref_E2_singlestage_deploy": 0.716}
    outp = REPO / f"lmwm/outputs/twomodel_poc_{args.mode}_K{args.K}.json"
    outp.write_text(json.dumps(res, indent=2))
    ckp = REPO / f"lmwm/checkpoints/twomodel/{args.mode}_K{args.K}.pt"
    ckp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mdn": mdn.state_dict(), "fwd": fwd.state_dict(), "K": args.K, "din": din,
                "code_dim": args.code_dim, "gmu": float(gmu), "gsd": float(gsd), "mode": args.mode}, ckp)
    print(json.dumps(res, indent=2), flush=True); print(f"saved ckpt -> {ckp}", flush=True)


if __name__ == "__main__":
    main()
