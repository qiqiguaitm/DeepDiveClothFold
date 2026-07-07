#!/usr/bin/env python
"""LMWM = Milestone PREDICTOR + GENERATOR (π0.5 SigLIP space). Naming:

  PREDICTOR  — predicts "which value-milestone next" as a compact code z (~128d):
     · teacher form  MilestonePredictorTeacher (= InverseEnc): (g_t, g_future) -> z   [train only, sees future]
     · deploy form   MilestonePredictor (MDN):  g_t -> K-Gaussian mixture over z      [deploy takes mode: smooth]
  GENERATOR  — MilestoneGenerator (AdaLN): (current grid g_t, code) -> next-milestone grid ĝ
     · g_t is the spatial canvas; code modulates via shift/scale/gate (zero-init gate = starts from g_t)

  L = smooth_l1(Generator(g_t, teacher_z), g_f) + lift·relu(cos(ĝ,g_t)-cos(ĝ,g_f))   [recon + anti-persistence]
      + MilestonePredictor.nll(g_t, teacher_z.detach())                              [distill teacher code]
      (+ center_w·CE(cen_head(z), next_milestone_id)  cluster-center identity anchor, in train_ablation)

Compact bottleneck (Generator NEEDS the code) + AdaLN (current not copyable) + teacher (clean distill
target) + lift fight the v1 persistence collapse. Generalizes (learned continuous code, no retrieval).
[legacy class names ForwardAdaLN/PredMDN kept as aliases]
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
from train_lawm_patch import load_index, read_imgs, InverseEnc  # noqa: E402

# --- naming: Predictor(teacher=inverse, deploy=MilestonePredictor) + Generator(MilestoneGenerator) ---
MilestonePredictorTeacher = InverseEnc  # PREDICTOR (teacher form): (g_t, g_future)->code, training only
from optimize_subgoal import build_pairs  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402

PI05_NPZ = "/vePFS/tim/workspace/openpi_cache/paligemma_weights/pt_224.npz"
PI05_NPZ_GF3 = "/vePFS-North-E/vis_robot/openpi_cache/paligemma_weights/pt_224.npz"


class MilestoneGenerator(nn.Module):  # GENERATOR: (current grid, milestone code) -> next-milestone grid
    """生成器: 当前 grid 当画布, 码经 AdaLN(shift/scale/gate, zero-init) 逐层调制成下一 milestone grid."""
    def __init__(self, din, code_dim, hid=512, nblk=4):
        super().__init__()
        self.nblk, self.hid = nblk, hid
        self.proj = nn.Conv2d(din, hid, 3, 1, 1)
        self.gn = nn.ModuleList([nn.GroupNorm(8, hid) for _ in range(nblk)])
        self.blk = nn.ModuleList([nn.Sequential(nn.Conv2d(hid, hid, 3, 1, 1), nn.GELU(),
                                                nn.Conv2d(hid, hid, 3, 1, 1)) for _ in range(nblk)])
        self.mod = nn.Linear(code_dim, nblk * 3 * hid)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)      # zero-init: gate 0 -> output=proj(g_t)
        self.out = nn.Conv2d(hid, din, 3, 1, 1)

    def forward(self, gt, code):
        h = self.proj(gt)
        m = self.mod(code).view(-1, self.nblk, 3, self.hid)
        for i in range(self.nblk):
            sh, sc, ga = m[:, i, 0], m[:, i, 1], m[:, i, 2]                 # (B,hid)
            hn = self.gn[i](h) * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
            h = h + ga[:, :, None, None] * self.blk[i](hn)
        return self.out(h)


class MilestonePredictor(nn.Module):  # PREDICTOR (deploy head): current gist -> multimodal milestone code
    """预测器(部署头): 当前 gist -> K 个高斯混合 over 码 z; deploy 取 mode(平滑单点)."""
    def __init__(self, in_dim, C, K, hid=1024):
        super().__init__()
        self.K, self.C = K, C
        self.trunk = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        self.pi = nn.Linear(hid, K); self.mu = nn.Linear(hid, K * C); self.ls = nn.Linear(hid, K * C)

    def forward(self, x):
        h = self.trunk(x); B = x.shape[0]
        return self.pi(h), self.mu(h).view(B, self.K, self.C), self.ls(h).view(B, self.K, self.C).clamp(-6, 4)

    def nll(self, x, z):
        logit, mu, ls = self(x); logpi = F.log_softmax(logit, -1); var = (2 * ls).exp()
        comp = -0.5 * (((z[:, None] - mu) ** 2) / var + 2 * ls + np.log(2 * np.pi)).sum(-1)
        return -(torch.logsumexp(logpi + comp, -1)).mean()

    @torch.no_grad()
    def deploy_mean(self, x):
        logit, mu, _ = self(x); return mu[torch.arange(len(x)), logit.argmax(-1)]

    @torch.no_grad()
    def sample(self, x, n):
        logit, mu, ls = self(x); pi = F.softmax(logit, -1); out = []
        for _ in range(n):
            k = torch.multinomial(pi, 1).squeeze(1)
            out.append(mu[torch.arange(len(x)), k] + torch.randn_like(mu[:, 0]) * ls[torch.arange(len(x)), k].exp())
        return torch.stack(out)


def cosr(a, b):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--code_dim", type=int, default=128)
    ap.add_argument("--mode", default="milestone_viterbi")
    ap.add_argument("--lift_w", type=float, default=1.0)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_val", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--bestof", type=int, default=8)
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)

    E, FR, Fn = load_index(args.feature_dir)
    rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = rg["prototype_table"].astype(np.float32); pord = rg["pord"].astype(np.float32)
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    tr, va = build_pairs(E, FR, Fn, proto, args.mode, 5, val_eps, args.seed, pord=pord)
    tr = tr[:args.n_train]; va = va[:args.n_val]
    uniq = sorted(set([g for p in tr + va for g in p])); u2k = {g: k for k, g in enumerate(uniq)}
    print(f"[v2 K={args.K} cd={args.code_dim}] {len(tr)} train + {len(va)} val, {len(uniq)} frames", flush=True)

    imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, np.array(uniq), 224, 128)
    enc = SiglipBigVision(npz, device=dev)
    grids = enc.encode_grid(imgs, bs=32); din = grids.shape[1]
    gmu, gsd = grids.mean(), grids.std() + 1e-6
    GZ = torch.from_numpy(((grids - gmu) / gsd).astype(np.float32)); gist = GZ.mean((2, 3))
    tra = np.array([u2k[c] for c, _ in tr]); trb = np.array([u2k[n] for _, n in tr])
    vaa = np.array([u2k[c] for c, _ in va]); vab = np.array([u2k[n] for _, n in va])

    inv = InverseEnc(din, args.code_dim).to(dev)
    fwd = MilestoneGenerator(din, args.code_dim).to(dev)
    predm = MilestonePredictor(din, args.code_dim, args.K).to(dev)
    o1 = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(predm.parameters(), lr=2e-4, weight_decay=1e-5)
    npar = lambda m: sum(p.numel() for p in m.parameters())
    print(f"[v2] inverse {npar(inv)/1e6:.1f}M fwd {npar(fwd)/1e6:.1f}M predm {npar(predm)/1e6:.1f}M", flush=True)
    for step in range(args.steps):
        sel = torch.randint(0, len(tra), (64,))
        Gc = GZ[tra[sel]].to(dev); Gf = GZ[trb[sel]].to(dev); gc = gist[tra[sel]].to(dev)
        z = inv(Gc, Gf)                                                    # teacher
        gh = fwd(Gc, z)
        ghf, gcf, gff = gh.flatten(1), Gc.flatten(1), Gf.flatten(1)
        lift = torch.relu(cosr(ghf, gcf) - cosr(ghf, gff)).mean()          # anti-persistence
        l1 = F.smooth_l1_loss(gh, Gf) + args.lift_w * lift
        o1.zero_grad(); l1.backward(); o1.step()
        l2 = predm.nll(gc, z.detach())                                     # distill teacher code
        o2.zero_grad(); l2.backward(); o2.step()
    inv.eval(); fwd.eval(); predm.eval()

    def cn(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    co, cd_, cb, cp = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(vaa), 128):
            Gc = GZ[vaa[s:s + 128]].to(dev); Gf = GZ[vab[s:s + 128]].to(dev); gc = gist[vaa[s:s + 128]].to(dev)
            gtr = f(Gf)
            co.append(cn(f(fwd(Gc, inv(Gc, Gf))), gtr))
            zdep = predm.deploy_mean(gc); cd_.append(cn(f(fwd(Gc, zdep)), gtr))
            best = None
            for zi in predm.sample(gc, args.bestof):
                ck = cn(f(fwd(Gc, zi)), gtr); best = ck if best is None else np.maximum(best, ck)
            cb.append(best); cp.append(cn(f(Gc), gtr))
    res = {"arch": "v2 inverse-teacher + AdaLN + MDN-code + lift", "K": args.K, "code_dim": args.code_dim,
           "lift_w": args.lift_w, "n_train": len(tr), "n_val": len(va),
           "oracle_grid_cos": round(float(np.concatenate(co).mean()), 4),
           "deploy_grid_cos": round(float(np.concatenate(cd_).mean()), 4),
           "bestof%d_grid_cos" % args.bestof: round(float(np.concatenate(cb).mean()), 4),
           "persistence": round(float(np.concatenate(cp).mean()), 4),
           "bestof_gain": round(float(np.concatenate(cb).mean() - np.concatenate(cd_).mean()), 4)}
    (REPO / f"lmwm/outputs/twomodel_v2_{args.mode}_K{args.K}.json").write_text(json.dumps(res, indent=2))
    ckp = REPO / f"lmwm/checkpoints/twomodel_v2/{args.mode}_K{args.K}.pt"
    ckp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"inv": inv.state_dict(), "fwd": fwd.state_dict(), "predm": predm.state_dict(),
                "K": args.K, "din": din, "code_dim": args.code_dim, "gmu": float(gmu), "gsd": float(gsd),
                "mode": args.mode, "arch": "v2"}, ckp)
    print(json.dumps(res, indent=2), flush=True); print(f"saved -> {ckp}", flush=True)


ForwardAdaLN = MilestoneGenerator      # back-compat alias
PredMDN = MilestonePredictor            # back-compat alias

if __name__ == "__main__":
    main()
