#!/usr/bin/env python
"""V5(a) 内在对比: 训 LMWM(同 p1_train 模型)在指定 pairs 上, episode 级 held-out 评前向预测。
held-out recon_cos = cos(生成的下一态 grid, 真目标 grid) [teacher-forced code];
persist = cos(当前帧 grid, 目标 grid)(不动基线)。gain = recon−persist = 前向建模增量。
比 milestone pairs vs r-脊 pairs 哪个 world-model 前向预测更好(内在, sim-SR 为最终判据)。
用法: srpo python p1_train_lmwm_heldout.py --pairs <pairs.npz> --tag <name> [--steps N]
"""
import os, sys, argparse, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
FEAT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
DIN, PGRID = 768, 16

class InverseEnc(nn.Module):
    def __init__(s, din, cd, hid=256):
        super().__init__(); s.conv = nn.Sequential(nn.Conv2d(2*din, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(), nn.Conv2d(hid, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU()); s.head = nn.Linear(hid, cd); s.ln = nn.LayerNorm(cd)
    def forward(s, gt, gf): return s.ln(s.head(s.conv(torch.cat([gt, gf], 1)).mean((2, 3))))
class Gen(nn.Module):
    def __init__(s, din, cd, hid=512, nb=4):
        super().__init__(); s.nb, s.hid = nb, hid; s.proj = nn.Conv2d(din, hid, 3, 1, 1)
        s.gn = nn.ModuleList([nn.GroupNorm(8, hid) for _ in range(nb)]); s.blk = nn.ModuleList([nn.Sequential(nn.Conv2d(hid, hid, 3, 1, 1), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1)) for _ in range(nb)])
        s.mod = nn.Linear(cd, nb*3*hid); nn.init.zeros_(s.mod.weight); nn.init.zeros_(s.mod.bias); s.out = nn.Conv2d(hid, din, 3, 1, 1)
    def forward(s, gt, code):
        h = s.proj(gt); m = s.mod(code).view(-1, s.nb, 3, s.hid)
        for i in range(s.nb):
            sh, sc, ga = m[:, i, 0], m[:, i, 1], m[:, i, 2]; hn = s.gn[i](h)*(1+sc[:, :, None, None])+sh[:, :, None, None]; h = h + ga[:, :, None, None]*s.blk[i](hn)
        return s.out(h)
def cosr(a, b): return (a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)+1e-8)
def load_grid(cache, ep):
    if ep not in cache:
        g = np.load(f"{FEAT}/ep{ep}.npz")["grid"].astype(np.float32); cache[ep] = g.reshape(len(g), PGRID, PGRID, DIN).transpose(0, 3, 1, 2)
    return cache[ep]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--cd", type=int, default=32)
    a = ap.parse_args(); dev = "cuda"; rng = np.random.RandomState(0)
    P = np.load(a.pairs); ce, cf, tf = P["cur_ep"], P["cur_fi"], P["tgt_fi"]
    eps = np.unique(ce); rng.shuffle(eps); k = int(len(eps)*0.85); tre, hoe = set(eps[:k].tolist()), set(eps[k:].tolist())
    tri = np.where(np.isin(ce, list(tre)))[0]; hoi = np.where(np.isin(ce, list(hoe)))[0]
    print(f"[{a.tag}] {len(ce)} pairs | train {len(tri)} / held-out {len(hoi)} ({len(hoe)} eps)", flush=True)
    cache = {}; inv = InverseEnc(DIN, a.cd).to(dev); gen = Gen(DIN, a.cd).to(dev)
    opt = torch.optim.AdamW(list(inv.parameters())+list(gen.parameters()), 2e-4, weight_decay=1e-5)
    def grab(ii):
        gt = np.stack([load_grid(cache, int(ce[i]))[int(cf[i])] for i in ii]); gf = np.stack([load_grid(cache, int(ce[i]))[int(tf[i])] for i in ii])
        return torch.from_numpy(gt).to(dev), torch.from_numpy(gf).to(dev)
    @torch.no_grad()
    def evalho():
        inv.eval(); gen.eval(); rc = []; ps = []
        for _ in range(30):
            ii = hoi[rng.randint(0, len(hoi), 128)]; gt, gf = grab(ii); z = inv(gt, gf); pr = gen(gt, z)
            rc.append(cosr(pr.flatten(1), gf.flatten(1)).mean().item()); ps.append(cosr(gt.flatten(1), gf.flatten(1)).mean().item())
        inv.train(); gen.train(); return float(np.mean(rc)), float(np.mean(ps))
    for step in range(a.steps):
        ii = tri[rng.randint(0, len(tri), a.bs)]; gt, gf = grab(ii); z = inv(gt, gf); pr = gen(gt, z)
        loss = F.smooth_l1_loss(pr, gf) + F.relu(cosr(pr.flatten(1), gt.flatten(1))-cosr(pr.flatten(1), gf.flatten(1))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == a.steps-1:
            rc, ps = evalho(); print(f"[{a.tag}] step {step}: held-out recon_cos={rc:.4f} persist={ps:.4f} GAIN={rc-ps:+.4f}", flush=True)
    rc, ps = evalho(); print(f"[{a.tag}] FINAL held-out recon_cos={rc:.4f} persist={ps:.4f} GAIN={rc-ps:+.4f}\n{a.tag}_DONE", flush=True)

if __name__ == "__main__":
    main()
