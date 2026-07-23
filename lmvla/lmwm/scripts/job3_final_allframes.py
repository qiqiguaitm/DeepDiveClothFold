"""Job3: 终版模型 + Job2 同口径(全帧对末帧) -> 与 P1 可直接相减。"""
import numpy as np, torch, sys, json
from pathlib import Path
REPO = Path("/home/tim/workspace/deepdive_kai0/lmvla")
sys.path.insert(0, str(REPO / "lmwm/scripts")); sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import load_index
from train_ablation import build_pairs_abl
from train_multitask import _read_libero
from train_twomodel_v2 import MilestonePredictor, MilestoneGenerator
from crave.encoders import load_encoder

dev = "cuda"
FD = "crave/data/libero10_dinov3base"; ROOT = "/vePFS/tim/workspace/LIBERO_fastwam/libero_10_no_noops_lerobot"
GRAPH = "lmwm/data/recurrence_graphs/libero10_dinov3base/recurrence_graph.npz"
ck = torch.load(REPO / "lmwm/checkpoints/libero10_lmwm_sharedpca.pt", map_location="cpu", weights_only=False)
din, cdim, K, gmu, gsd = ck["din"], ck["code_dim"], ck["K"], ck["gmu"], ck["gsd"]
predm = MilestonePredictor(din, cdim, K).to(dev); predm.load_state_dict(ck["predm"]); predm.eval()
fwd = MilestoneGenerator(din, cdim).to(dev); fwd.load_state_dict(ck["fwd"]); fwd.eval()

E, FR, Fn = load_index(REPO / FD)
g = np.load(REPO / GRAPH); proto = g["prototype_table"].astype(np.float32); pord = g["pord"].astype(np.float32)
protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
rng = np.random.default_rng(2026); eps_all = np.unique(E); rng.shuffle(eps_all)
val_eps = set(eps_all[:max(1, int(round(len(eps_all)*0.2)))].tolist())
tr, va = build_pairs_abl(E, FR, Fn, proto, protoL, pord, "seglast", val_eps, 2026)
trained_g = set(p[0] for p in tr) | set(p[0] for p in va)     # 有 pair 的全局帧 idx

import json as _j
tasks = {_j.loads(l)["task_index"]: _j.loads(l)["task"] for l in open(f"{ROOT}/meta/tasks.jsonl")}
n2i = {v: k for k, v in tasks.items()}
ep2task = {_j.loads(l)["episode_index"]: n2i[_j.loads(l)["tasks"][0]] for l in open(f"{ROOT}/meta/episodes.jsonl")}
enc = load_encoder("dinov3-base", device=dev)

NB, NEP = 10, 5
by_task = {}
glob_lift = [[] for _ in range(NB)]; glob_cov = [[] for _ in range(NB)]
glob_p = [[] for _ in range(NB)]; glob_d = [[] for _ in range(NB)]
task_eps = {}
for ep in np.unique(E):
    task_eps.setdefault(ep2task.get(int(ep), -1), []).append(ep)

for t, teps in sorted(task_eps.items()):
    for ep in teps[:NEP]:
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        if len(order) < 12: continue
        ie, _ = _read_libero(Path(ROOT), "observation.images.image", E, FR, order, 224, 128)
        gr = enc.encode_grid(ie, bs=64).astype(np.float32)
        gr = gr / (np.linalg.norm(gr, axis=1, keepdims=True) + 1e-8)
        G = torch.from_numpy(((gr - gmu) / gsd).astype(np.float32)).to(dev)
        N = len(G); gist = G.mean((2, 3))
        with torch.no_grad():
            pred = fwd(G, predm.deploy_mean(gist))
        f = lambda x: (x.detach().cpu().numpy()*gsd + gmu).reshape(len(x), -1)
        A = f(G); Pr = f(pred); lastv = A[-1:]
        cn = lambda a, b: (a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-8)
        cp = cn(A, np.repeat(lastv, N, 0)); cd_ = cn(Pr, np.repeat(lastv, N, 0))
        for i in range(N):
            b = min(int(i/N*NB), NB-1)
            glob_lift[b].append(cd_[i]-cp[i]); glob_cov[b].append(1.0 if int(order[i]) in trained_g else 0.0)
            glob_p[b].append(cp[i]); glob_d[b].append(cd_[i])
            by_task.setdefault(t, [[] for _ in range(NB)])[b].append(cd_[i]-cp[i])
    print(f"  task {t} done", flush=True)

print(f"\n{'位置':>9} {'训练覆盖':>9} {'persist':>9} {'deploy':>9} {'lift':>10}")
print("-"*52)
for b in range(NB):
    print(f"{b*10:>3}-{(b+1)*10:>3}% {np.mean(glob_cov[b]):>9.2f} {np.mean(glob_p[b]):>9.3f} {np.mean(glob_d[b]):>9.3f} {np.mean(glob_lift[b]):>+10.4f}")
print("-"*52)
end_lifts = {t: float(np.mean(v[-1])) for t, v in by_task.items() if v[-1]}
print(f"\n终版 end_lift (最末 10%) per task:")
for t in sorted(end_lifts): print(f"  [{t:>2}] {end_lifts[t]:+.4f}  {tasks.get(t,'?')[:46]}")
print(f"\n终版 平均 end_lift = {np.mean(list(end_lifts.values())):+.4f}")
json.dump({"per_bin_lift": [float(np.mean(x)) for x in glob_lift],
           "per_bin_cov": [float(np.mean(x)) for x in glob_cov],
           "end_lift_per_task": end_lifts}, open(REPO/"lmwm/outputs/job3_final_allframes.json","w"), indent=1)
