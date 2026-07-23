"""Job1: 终版 arch (dinov3base 空间) libero10 的 per-LIBERO-task deploy/persist/lift。"""
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
CFG = dict(fdir="crave/data/libero10_dinov3base",
           root="/vePFS/tim/workspace/LIBERO_fastwam/libero_10_no_noops_lerobot",
           cam="observation.images.image",
           graph="lmwm/data/recurrence_graphs/libero10_dinov3base/recurrence_graph.npz")
ck = torch.load(REPO / "lmwm/checkpoints/libero10_lmwm_sharedpca.pt", map_location="cpu", weights_only=False)
din, cdim, K, gmu, gsd = ck["din"], ck["code_dim"], ck["K"], ck["gmu"], ck["gsd"]
predm = MilestonePredictor(din, cdim, K).to(dev); predm.load_state_dict(ck["predm"]); predm.eval()
fwd = MilestoneGenerator(din, cdim).to(dev); fwd.load_state_dict(ck["fwd"]); fwd.eval()
print(f"[ckpt] din={din} code={cdim} K={K} teacher={ck['teacher']}", flush=True)

E, FR, Fn = load_index(REPO / CFG["fdir"])
g = np.load(REPO / CFG["graph"]); proto = g["prototype_table"].astype(np.float32); pord = g["pord"].astype(np.float32)
protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
rng = np.random.default_rng(2026); eps = np.unique(E); rng.shuffle(eps)
val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
tr, va = build_pairs_abl(E, FR, Fn, proto, protoL, pord, "seglast", val_eps, 2026)
print(f"[pairs] train={len(tr)} val={len(va)}", flush=True)

import json as _j
R = CFG["root"]
tasks = {_j.loads(l)["task_index"]: _j.loads(l)["task"] for l in open(f"{R}/meta/tasks.jsonl")}
n2i = {v: k for k, v in tasks.items()}
ep2task = {_j.loads(l)["episode_index"]: n2i[_j.loads(l)["tasks"][0]] for l in open(f"{R}/meta/episodes.jsonl")}

va = va[:6000]
uniq = sorted(set([p[0] for p in va] + [p[1] for p in va])); u2k = {gx: k for k, gx in enumerate(uniq)}
print(f"[frames] encoding {len(uniq)} unique frames ...", flush=True)
ie, _ = _read_libero(Path(CFG["root"]), CFG["cam"], E, FR, np.array(uniq), 224, 128)
enc = load_encoder("dinov3-base", device=dev)
grids = enc.encode_grid(ie, bs=32)
gf32 = grids.astype(np.float32)
grids = (gf32 / (np.linalg.norm(gf32, axis=1, keepdims=True) + 1e-8)).astype(np.float32)  # per-patch L2 (终版约定)
GZ = torch.from_numpy(((grids - gmu) / gsd).astype(np.float32))
gist = GZ.mean((2, 3))
print(f"[enc] grids {grids.shape}", flush=True)

def cn(a, b): return (a*b).sum(1) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1) + 1e-8)
f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
per = {}
with torch.no_grad():
    for s in range(0, len(va), 128):
        blk = va[s:s+128]
        a = np.array([u2k[p[0]] for p in blk]); b = np.array([u2k[p[1]] for p in blk])
        Gc = GZ[a].to(dev); Gf = GZ[b].to(dev); gc = gist[a].to(dev)
        gtr = f(Gf)
        dep = cn(f(fwd(Gc, predm.deploy_mean(gc))), gtr); per_ = cn(f(Gc), gtr)
        for k, p in enumerate(blk):
            t = ep2task.get(int(E[p[0]]), -1)
            per.setdefault(t, {"d": [], "p": []})
            per[t]["d"].append(dep[k]); per[t]["p"].append(per_[k])

print(f"\n{'task':>4} {'n':>5} {'deploy':>8} {'persist':>8} {'lift':>9}  描述")
print("-" * 92)
rows = []
for t in sorted(per):
    d = float(np.mean(per[t]["d"])); pp = float(np.mean(per[t]["p"])); n = len(per[t]["d"])
    rows.append(dict(task=t, n=n, deploy=round(d, 4), persist=round(pp, 4), lift=round(d-pp, 4), desc=tasks.get(t, "?")))
    mark = " ←白杯+布丁" if t == 8 else ""
    print(f"{t:>4} {n:>5} {d:>8.4f} {pp:>8.4f} {d-pp:>+9.4f}  {tasks.get(t,'?')[:44]}{mark}")
allд = np.concatenate([per[t]["d"] for t in per]); allp = np.concatenate([per[t]["p"] for t in per])
print("-" * 92)
print(f"{'ALL':>4} {len(allд):>5} {allд.mean():>8.4f} {allp.mean():>8.4f} {allд.mean()-allp.mean():>+9.4f}")
json.dump(rows, open(REPO / "lmwm/outputs/job1_final_pertask.json", "w"), indent=1, ensure_ascii=False)
