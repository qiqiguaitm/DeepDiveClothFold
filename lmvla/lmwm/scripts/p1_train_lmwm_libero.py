#!/usr/bin/env python
"""P1: 训 LMWM(teacher InverseEnc + generator MilestoneGenerator) on LIBERO DINOv3-base pairs。
生成器 = LaWM decoder 替身: (current grid[B,768,16,16], code[B,32]) -> next-milestone grid。
自包含(内联模型), 数据 = p1_libero_milestone_pairs 产的 pairs.npz + libero_dinov3base/ep*.npz。
用法: srpo python p1_train_lmwm_libero.py [--steps N] [--maxep M] [--smoke]
"""
import os, sys, argparse, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

FEAT = "/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
PAIRS = "/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_milestone/pairs.npz"
DIN, PGRID = 768, 16

class InverseEnc(nn.Module):  # teacher: (g_t,g_f)->code, 看未来 milestone+1
    def __init__(self, din, code_dim, hid=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2*din, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(),
            nn.Conv2d(hid, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU())
        self.head = nn.Linear(hid, code_dim); self.ln = nn.LayerNorm(code_dim)
    def forward(self, gt, gf):
        return self.ln(self.head(self.conv(torch.cat([gt, gf], 1)).mean((2, 3))))

class MilestoneGenerator(nn.Module):  # = LaWM decoder 替身: (grid, code)->next-milestone grid
    def __init__(self, din, code_dim, hid=512, nblk=4):
        super().__init__()
        self.nblk, self.hid = nblk, hid
        self.proj = nn.Conv2d(din, hid, 3, 1, 1)
        self.gn = nn.ModuleList([nn.GroupNorm(8, hid) for _ in range(nblk)])
        self.blk = nn.ModuleList([nn.Sequential(nn.Conv2d(hid, hid, 3, 1, 1), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1)) for _ in range(nblk)])
        self.mod = nn.Linear(code_dim, nblk*3*hid); nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.out = nn.Conv2d(hid, din, 3, 1, 1)
    def forward(self, gt, code):
        h = self.proj(gt); m = self.mod(code).view(-1, self.nblk, 3, self.hid)
        for i in range(self.nblk):
            sh, sc, ga = m[:, i, 0], m[:, i, 1], m[:, i, 2]
            hn = self.gn[i](h) * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
            h = h + ga[:, :, None, None] * self.blk[i](hn)
        return self.out(h)

class MilestonePredictorGrid(nn.Module):  # deploy 头: grid -> MDN over code
    def __init__(self, in_dim, C, K, hid=1024, cw=256):
        super().__init__()
        self.K, self.C = K, C
        self.enc = nn.Sequential(nn.Conv2d(in_dim, cw, 3, 2, 1), nn.GroupNorm(8, cw), nn.GELU(),
                                 nn.Conv2d(cw, cw, 3, 2, 1), nn.GroupNorm(8, cw), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(cw, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        self.pi = nn.Linear(hid, K); self.mu = nn.Linear(hid, K*C); self.ls = nn.Linear(hid, K*C)
    def forward(self, G):
        h = self.trunk(self.enc(G).mean((2, 3))); B = G.shape[0]
        return self.pi(h), self.mu(h).view(B, self.K, self.C), self.ls(h).view(B, self.K, self.C).clamp(-6, 4)
    def nll(self, G, z):
        logit, mu, ls = self(G); logpi = F.log_softmax(logit, -1); var = (2*ls).exp()
        comp = -0.5 * (((z[:, None]-mu)**2)/var + 2*ls + np.log(2*np.pi)).sum(-1)
        return -(torch.logsumexp(logpi+comp, -1)).mean()

def cosr(a, b): return (a*b).sum(1) / (a.norm(dim=1)*b.norm(dim=1)+1e-8)

def load_grid(cache, ep):
    if ep not in cache:
        g = np.load(f"{FEAT}/ep{ep}.npz")["grid"].astype(np.float32)  # [N,256,768]
        cache[ep] = g.reshape(len(g), PGRID, PGRID, DIN).transpose(0, 3, 1, 2)  # [N,768,16,16]
    return cache[ep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--code_dim", type=int, default=32)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lift_w", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="/home/tim/workspace/deepdive_kai0/lmvla/lmwm/checkpoints/lmwm_libero_dinov3base")
    ap.add_argument("--pairs", default=PAIRS)
    args = ap.parse_args()
    dev = "cuda"

    P = np.load(args.pairs)
    cur_ep, cur_fi, tgt_fi = P["cur_ep"], P["cur_fi"], P["tgt_fi"]
    print(f"[pairs] {len(cur_ep)} 对", flush=True)
    cache = {}
    inv = InverseEnc(DIN, args.code_dim).to(dev)
    gen = MilestoneGenerator(DIN, args.code_dim).to(dev)
    prd = MilestonePredictorGrid(DIN, args.code_dim, args.K).to(dev)
    o1 = torch.optim.AdamW(list(inv.parameters())+list(gen.parameters()), lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(prd.parameters(), lr=2e-4, weight_decay=1e-5)

    steps = 300 if args.smoke else args.steps
    def batch():
        idx = np.random.randint(0, len(cur_ep), args.bs)
        gt = np.stack([load_grid(cache, int(cur_ep[i]))[int(cur_fi[i])] for i in idx])
        gf = np.stack([load_grid(cache, int(cur_ep[i]))[int(tgt_fi[i])] for i in idx])
        return torch.from_numpy(gt).to(dev), torch.from_numpy(gf).to(dev)
    for step in range(steps):
        gt, gf = batch()
        z = inv(gt, gf)
        pred = gen(gt, z)
        l_rec = F.smooth_l1_loss(pred, gf)
        pf, pt = pred.flatten(1), gf.flatten(1)
        l_lift = F.relu(cosr(pred.flatten(1), gt.flatten(1)) - cosr(pf, pt)).mean()
        l_dist = prd.nll(gt, z.detach())
        (l_rec + args.lift_w*l_lift).backward(retain_graph=True); o1.step(); o1.zero_grad()
        l_dist.backward(); o2.step(); o2.zero_grad()
        if step % 50 == 0 or step == steps-1:
            with torch.no_grad():
                rec_cos = cosr(pred.flatten(1), pt).mean().item()
                persist = cosr(gt.flatten(1), pt).mean().item()  # 当前帧 vs 目标(持久基线)
            print(f"step {step}: rec={l_rec.item():.4f} lift={l_lift.item():.4f} dist={l_dist.item():.3f} "
                  f"| recon_cos={rec_cos:.3f} (持久基线 {persist:.3f})", flush=True)
    if not args.smoke:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"inv": inv.state_dict(), "gen": gen.state_dict(), "prd": prd.state_dict(),
                    "code_dim": args.code_dim, "din": DIN}, f"{args.out}/lmwm.pt")
        print(f"[save] {args.out}/lmwm.pt", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
