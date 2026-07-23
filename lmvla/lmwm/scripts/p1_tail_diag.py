"""P1 尾巴诊断: task idx1(白杯+布丁) 全帧 persist/deploy/oracle vs 位置, 标注训练/未训练区。"""
import numpy as np, torch, glob, sys, os
sys.path.insert(0, "lmwm/scripts")
from p1_train_lmwm_libero import InverseEnc, MilestoneGenerator, MilestonePredictorGrid, load_grid, cosr, DIN, PGRID

dev = "cuda"
CK = "lmwm/checkpoints/lmwm_libero_dinov3base/lmwm.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
cd = ck["code_dim"]
inv = InverseEnc(DIN, cd).to(dev); inv.load_state_dict(ck["inv"]); inv.eval()
gen = MilestoneGenerator(DIN, cd).to(dev); gen.load_state_dict(ck["gen"]); gen.eval()
prd = MilestonePredictorGrid(DIN, cd, 4).to(dev); prd.load_state_dict(ck["prd"]); prd.eval()
print(f"[ckpt] code_dim={cd} loaded", flush=True)

def deploy_code(G):
    logit, mu, ls = prd(G)
    return mu[torch.arange(len(G)), logit.argmax(1)]        # 主分量均值

P = np.load("lmwm/data/libero_milestone/pairs.npz")
TASK = 1
sel = P["pair_task"] == TASK
eps_t = np.unique(P["cur_ep"][sel])
trained = {}                                                 # ep -> set(已建对的 frame idx)
for e, fi in zip(P["cur_ep"][sel], P["cur_fi"][sel]):
    trained.setdefault(int(e), set()).add(int(fi))
print(f"[task {TASK}] {len(eps_t)} episodes, {sel.sum()} pairs", flush=True)

NB = 10                                                       # 10 个位置分箱
acc = {k: [[] for _ in range(NB)] for k in ("persist", "deploy", "oracle")}
tr_frac = [[] for _ in range(NB)]
cache = {}
for e in eps_t[:40]:
    g = load_grid(cache, int(e))                              # [N,768,16,16]
    N = len(g)
    G = torch.from_numpy(g).to(dev)
    last = G[-1:].expand(N, -1, -1, -1)
    with torch.no_grad():
        zd = deploy_code(G); zo = inv(G, last)
        pd_ = gen(G, zd).flatten(1); po = gen(G, zo).flatten(1)
        lf = last.flatten(1)
        cp = cosr(G.flatten(1), lf).cpu().numpy()
        cdp = cosr(pd_, lf).cpu().numpy()
        co = cosr(po, lf).cpu().numpy()
    tset = trained.get(int(e), set())
    for i in range(N):
        b = min(int(i / N * NB), NB - 1)
        acc["persist"][b].append(cp[i]); acc["deploy"][b].append(cdp[i]); acc["oracle"][b].append(co[i])
        tr_frac[b].append(1.0 if i in tset else 0.0)
    cache.clear()

print(f"\n{'位置':>8} {'训练覆盖':>9} {'persist':>9} {'deploy':>9} {'oracle':>9} {'oracle-deploy':>14}")
print("-" * 66)
for b in range(NB):
    p = np.mean(acc["persist"][b]); d = np.mean(acc["deploy"][b]); o = np.mean(acc["oracle"][b])
    tf = np.mean(tr_frac[b])
    flag = "  ← 未训练区" if tf < 0.2 else ""
    print(f"{b*10:>3}-{(b+1)*10:>3}% {tf:>9.2f} {p:>9.3f} {d:>9.3f} {o:>9.3f} {o-d:>14.3f}{flag}")
