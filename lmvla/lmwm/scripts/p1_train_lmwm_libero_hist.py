#!/usr/bin/env python
"""V7.1-B: 历史感知 LMWM 生成器(生成器专属 k 帧历史通道, VLA 主干不变)。
动机(§4.11): 别名(task8 双moka壶/task9 遮挡)源于无状态逐帧预测器——每帧从别名帧重定位→优柔+翻转。
V7.1-B: 只给生成器额外喂 k 帧历史特征, 历史里"已放壶A""推了门"消歧; 主VLA(num_frames=2)/VLM/teacher 全不动。
⚠️ 生成器真正"学会用历史纠正歧义 code"发生在主 VLA 训练(code 来自 VLM 有噪); 本预训练用 teacher oracle code,
   历史仅作 init(hist_fuse 初始化成"选当前帧"→ 起点≈单帧生成器, 再在主训学历史)。产出 lmwm_hist.pt 供主训 init。
用法: CUDA_VISIBLE_DEVICES=0 srpo python p1_train_lmwm_libero_hist.py --pairs <r-脊pairs> --khist 4 --out <dir>
"""
import os, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

FEAT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
PAIRS = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_rvalley/pairs.npz"
DIN, PGRID = 768, 16

class InverseEnc(nn.Module):  # teacher (g_t,g_f)->code, 不变(仍看当前+目标)
    def __init__(self, din, code_dim, hid=256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2*din, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(),
                                  nn.Conv2d(hid, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU())
        self.head = nn.Linear(hid, code_dim); self.ln = nn.LayerNorm(code_dim)
    def forward(self, gt, gf):
        return self.ln(self.head(self.conv(torch.cat([gt, gf], 1)).mean((2, 3))))

class HistMilestoneGenerator(nn.Module):
    """(gt_hist[B,k,din,P,P], code[B,code_dim]) -> next-milestone grid[B,din,P,P]。
    hist_fuse 1x1 conv 融合 k 帧, 初始化成"仅选最后一帧(=当前)"→ 起点等价单帧生成器。"""
    def __init__(self, din, code_dim, hid=512, nblk=4, khist=4):
        super().__init__()
        self.khist, self.din, self.nblk, self.hid = khist, din, nblk, hid
        self.hist_fuse = nn.Conv2d(khist*din, din, 1)
        nn.init.zeros_(self.hist_fuse.weight); nn.init.zeros_(self.hist_fuse.bias)
        with torch.no_grad():                       # init: 只选最后一帧(当前) = 恒等
            for d in range(din):
                self.hist_fuse.weight[d, (khist-1)*din + d, 0, 0] = 1.0
        self.proj = nn.Conv2d(din, hid, 3, 1, 1)
        self.gn = nn.ModuleList([nn.GroupNorm(8, hid) for _ in range(nblk)])
        self.blk = nn.ModuleList([nn.Sequential(nn.Conv2d(hid, hid, 3, 1, 1), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1)) for _ in range(nblk)])
        self.mod = nn.Linear(code_dim, nblk*3*hid); nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.out = nn.Conv2d(hid, din, 3, 1, 1)
    def forward(self, gt_hist, code):
        B, k, D, P, _ = gt_hist.shape
        gt = self.hist_fuse(gt_hist.reshape(B, k*D, P, P))
        h = self.proj(gt); m = self.mod(code).view(-1, self.nblk, 3, self.hid)
        for i in range(self.nblk):
            sh, sc, ga = m[:, i, 0], m[:, i, 1], m[:, i, 2]
            hn = self.gn[i](h) * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
            h = h + ga[:, :, None, None] * self.blk[i](hn)
        return self.out(h)

class MilestonePredictorGrid(nn.Module):  # deploy MDN 头(不变, 看当前帧)
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
        g = np.load(f"{FEAT}/ep{ep}.npz")["grid"].astype(np.float32)
        cache[ep] = g.reshape(len(g), PGRID, PGRID, DIN).transpose(0, 3, 1, 2)
    return cache[ep]

def hist_stack(grid, fi, k):  # 取 [fi-k+1 .. fi] k 帧, 头部不足则重复首帧 -> [k,din,P,P]
    idx = [max(0, fi-k+1+j) for j in range(k)]
    return grid[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--code_dim", type=int, default=32); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lift_w", type=float, default=1.0); ap.add_argument("--khist", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/checkpoints/lmwm_libero_hist")
    ap.add_argument("--pairs", default=PAIRS)
    args = ap.parse_args(); dev = "cuda"
    P = np.load(args.pairs); cur_ep, cur_fi, tgt_fi = P["cur_ep"], P["cur_fi"], P["tgt_fi"]
    print(f"[pairs] {len(cur_ep)} 对 | khist={args.khist}", flush=True)
    cache = {}
    inv = InverseEnc(DIN, args.code_dim).to(dev)
    gen = HistMilestoneGenerator(DIN, args.code_dim, khist=args.khist).to(dev)
    prd = MilestonePredictorGrid(DIN, args.code_dim, args.K).to(dev)
    o1 = torch.optim.AdamW(list(inv.parameters())+list(gen.parameters()), lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(prd.parameters(), lr=2e-4, weight_decay=1e-5)
    steps = 300 if args.smoke else args.steps
    def batch():
        idx = np.random.randint(0, len(cur_ep), args.bs)
        gh, gf, gc = [], [], []
        for i in idx:
            g = load_grid(cache, int(cur_ep[i])); fi = int(cur_fi[i])
            gh.append(hist_stack(g, fi, args.khist)); gc.append(g[fi]); gf.append(g[int(tgt_fi[i])])
        return (torch.from_numpy(np.stack(gh)).to(dev), torch.from_numpy(np.stack(gc)).to(dev),
                torch.from_numpy(np.stack(gf)).to(dev))
    for step in range(steps):
        gh, gc, gf = batch()
        z = inv(gc, gf)                       # teacher 仍看当前+目标
        pred = gen(gh, z)                     # 生成器吃 k 帧历史
        l_rec = F.smooth_l1_loss(pred, gf)
        l_lift = F.relu(cosr(pred.flatten(1), gc.flatten(1)) - cosr(pred.flatten(1), gf.flatten(1))).mean()
        l_dist = prd.nll(gc, z.detach())
        (l_rec + args.lift_w*l_lift).backward(retain_graph=True); o1.step(); o1.zero_grad()
        l_dist.backward(); o2.step(); o2.zero_grad()
        if step % 50 == 0 or step == steps-1:
            with torch.no_grad():
                rec_cos = cosr(pred.flatten(1), gf.flatten(1)).mean().item()
                persist = cosr(gc.flatten(1), gf.flatten(1)).mean().item()
            print(f"step {step}: rec={l_rec.item():.4f} lift={l_lift.item():.4f} dist={l_dist.item():.3f} | recon_cos={rec_cos:.3f} (持久基线 {persist:.3f})", flush=True)
    if not args.smoke:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"inv": inv.state_dict(), "gen": gen.state_dict(), "prd": prd.state_dict(),
                    "code_dim": args.code_dim, "din": DIN, "khist": args.khist}, f"{args.out}/lmwm.pt")
        print(f"[save] {args.out}/lmwm.pt", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
