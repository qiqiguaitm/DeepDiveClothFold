"""检验 P1 生成器是否真的在用 code(条件坍缩检测)。"""
import numpy as np, torch, sys
sys.path.insert(0, "lmwm/scripts")
from p1_train_lmwm_libero import InverseEnc, MilestoneGenerator, MilestonePredictorGrid, load_grid, cosr, DIN
dev = "cuda"
ck = torch.load("lmwm/checkpoints/lmwm_libero_dinov3base/lmwm.pt", map_location="cpu", weights_only=False)
cd = ck["code_dim"]
inv = InverseEnc(DIN, cd).to(dev); inv.load_state_dict(ck["inv"]); inv.eval()
gen = MilestoneGenerator(DIN, cd).to(dev); gen.load_state_dict(ck["gen"]); gen.eval()
prd = MilestonePredictorGrid(DIN, cd, 4).to(dev); prd.load_state_dict(ck["prd"]); prd.eval()

# AdaLN 调制层权重规模(zero-init, 若没学起来则仍≈0)
W = ck["gen"]["mod.weight"]; B = ck["gen"]["mod.bias"]
print(f"[AdaLN mod] |W|mean={W.abs().mean():.6f}  |W|max={W.abs().max():.6f}  |b|mean={B.abs().mean():.6f}")

P = np.load("lmwm/data/libero_milestone/pairs.npz")
sel = P["pair_task"] == 1
eps = np.unique(P["cur_ep"][sel])[:6]
cache = {}
d_zero, d_rand, d_perm, base = [], [], [], []
for e in eps:
    g = load_grid(cache, int(e)); G = torch.from_numpy(g).to(dev); N = len(G)
    last = G[-1:].expand(N, -1, -1, -1)
    with torch.no_grad():
        z = inv(G, last)
        o_z = gen(G, z).flatten(1)
        o_0 = gen(G, torch.zeros_like(z)).flatten(1)
        o_r = gen(G, torch.randn_like(z)).flatten(1)
        o_p = gen(G, z[torch.randperm(N, device=dev)]).flatten(1)   # 打乱 code(错配)
    d_zero.append(cosr(o_z, o_0).cpu().numpy()); d_rand.append(cosr(o_z, o_r).cpu().numpy())
    d_perm.append(cosr(o_z, o_p).cpu().numpy()); base.append(cosr(o_z, G.flatten(1)).cpu().numpy())
    cache.clear()
f = lambda x: float(np.concatenate(x).mean())
print(f"\ncos(gen(G,z_true), gen(G, z=0   )) = {f(d_zero):.4f}   ← 1.0 表示 code 完全无效")
print(f"cos(gen(G,z_true), gen(G, z~N(0,1))) = {f(d_rand):.4f}")
print(f"cos(gen(G,z_true), gen(G, z 打乱  )) = {f(d_perm):.4f}")
print(f"cos(gen(G,z_true), G_current       ) = {f(base):.4f}   ← 1.0 表示生成器≈恒等复制")
