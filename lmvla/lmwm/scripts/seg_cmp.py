import numpy as np, glob, json, sys
sys.path.insert(0, "crave/src")
from crave.utils.dp import viterbi_forward

FD = "crave/data/libero10_dinov3base"
idx = np.load(f"{FD}/index.npz"); E_all, FR_all = idx["E"], idx["FR"]
gid, feat = [], []
for f in sorted(glob.glob(f"{FD}/shard_*.npz")):
    d = np.load(f); gid.append(d["gidx"]); feat.append(d["feat"])
gid = np.concatenate(gid); feat = np.concatenate(feat).astype(np.float32)
E = E_all[gid]; FR = FR_all[gid]
Fn = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)

rg = np.load("lmwm/data/recurrence_graphs/libero10_dinov3base/recurrence_graph.npz")
proto = rg["prototype_table"].astype(np.float32); pord = rg["pord"].astype(np.float32)
protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

R = "/vePFS/tim/workspace/LIBERO_fastwam/libero_10_no_noops_lerobot"
tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{R}/meta/tasks.jsonl")}
name2i = {v: k for k, v in tasks.items()}
ep2task = {}
for l in open(f"{R}/meta/episodes.jsonl"):
    d = json.loads(l); ep2task[d["episode_index"]] = name2i[d["tasks"][0]]

rank = np.argsort(np.argsort(pord))           # cluster id -> progress rank
inv_rank = np.argsort(pord)                   # rank -> cluster id

def tail_of(ms):
    """final-segment span fraction + n_segments + final-milestone start fraction"""
    ch = np.where(np.diff(ms) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(ms)]])
    return (en[-1] - st[-1]) / len(ms), len(st), st[-1] / len(ms)

res = {}
for ep in np.unique(E):
    loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
    Fq = Fn[order]
    if len(Fq) < 10: continue
    dist = np.linalg.norm(Fq[:, None] - protoL[None], axis=2)
    # --- final-arch: Viterbi ---
    ms_v = viterbi_forward(dist, pord, up=3.0, down=25.0, hard_start=True)
    # --- P1 emulation: argmin + median(w=5) + cummax (on PROGRESS RANK) ---
    raw = rank[dist.argmin(1)]
    n = len(raw); w = 5; sm = raw.copy()
    for i in range(n):
        sm[i] = int(np.median(raw[max(0, i - w): i + w + 1]))
    ms_c = inv_rank[np.maximum.accumulate(sm)]
    t = ep2task.get(int(ep), -1)
    res.setdefault(t, []).append((tail_of(ms_v), tail_of(ms_c)))

print(f"{'task':>4} {'描述':<58} {'Viterbi尾巴':>10} {'cummax尾巴':>10} {'V段数':>6} {'C段数':>6}")
print("-" * 100)
rows = []
for t in sorted(res):
    v = np.array([r[0] for r in res[t]]); c = np.array([r[1] for r in res[t]])
    rows.append((t, v[:, 0].mean(), c[:, 0].mean(), v[:, 1].mean(), c[:, 1].mean()))
    mark = " ←白杯+布丁" if t == 8 else (" ←两moka" if t == 3 else (" ←微波炉" if t == 2 else ""))
    print(f"{t:>4} {tasks[t][:56]:<58} {v[:,0].mean():>10.3f} {c[:,0].mean():>10.3f} {v[:,1].mean():>6.1f} {c[:,1].mean():>6.1f}{mark}")
a = np.array([r[1:] for r in rows])
print("-" * 100)
print(f"{'均值':>4} {'':<58} {a[:,0].mean():>10.3f} {a[:,1].mean():>10.3f} {a[:,2].mean():>6.1f} {a[:,3].mean():>6.1f}")
