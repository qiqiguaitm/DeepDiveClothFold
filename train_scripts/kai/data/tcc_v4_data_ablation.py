#!/usr/bin/env python
"""TCC v4: 把聚类线验证的三条数据经验移植进 TCC 训练采样, 单变量消融。
数据处理改进 (对照 v3 的 random-eps + random-frames + N200):
  (S) 时间分层抽帧: [0,1] 分 T 桶各取一帧 -> 对齐骨架时间均匀 (借鉴 时间分桶 图31/32)
  (G) item 分组 batch: episode 按均值特征聚 G 组, 以 p_same 概率同组成 batch
      -> cycle-consistency 在恒定外观内对齐"阶段"而非被迫跨外观 (借鉴 §2.8 item分组)
  (N) 规模 200->500 (借鉴 §2.11 规模律)
读出固定 = 逐参考 argmax 中位数 (v3 最优); 评测 = kai0 held-out 50 GT eps。
用法: python tcc_v4_data_ablation.py --steps 1000 --out temp/tcc_v4
"""
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.cluster import KMeans
from scipy.stats import kendalltau, pearsonr
import sys
sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
from xirl.losses import compute_tcc_loss

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_advantage"
CACHE = REPO / "temp/tcc_kai0_armmask/feat_cache"
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--batch-eps", type=int, default=8)
ap.add_argument("--T", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--G", type=int, default=6, help="item 分组数")
ap.add_argument("--p-same", type=float, default=0.7, help="同组 batch 概率")
ap.add_argument("--knn-refs", type=int, default=30)
ap.add_argument("--out", default="temp/tcc_v4")
args = ap.parse_args()
out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)
dev = "cpu"
chunks_size = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)

zp = np.load(REPO / "temp/recurrence_v0_kai0/embeddings.npz")
EVAL = sorted(set(zp["ep_ids"].tolist()))
all_eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.npz"))
pool = np.random.RandomState(0).permutation([e for e in all_eps if e not in set(EVAL)]).tolist()

def load_raw(e):
    img = np.load(CACHE / f"ep{e}.npz")["f"]; n = len(img)
    st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                                  columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    return img, np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1)

print("[v4] loading up to 500 train + eval features ...")
RAW = {}
for e in pool[:500] + EVAL:
    try: RAW[e] = load_raw(e)
    except Exception: pass
EVALu = [e for e in EVAL if e in RAW]
GT = {}
for e in EVALu:
    g = pd.read_parquet(DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                        columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy()
    GT[e] = g[np.minimum(np.arange(len(RAW[e][0])) * 10, len(g) - 1)]

def make_feat_fn(train_eps):
    P = np.concatenate([RAW[e][1] for e in train_eps])
    mu, sd = P.mean(0), P.std(0) + 1e-8
    def feat(e):
        p = (RAW[e][1] - mu) / sd; p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
        i = RAW[e][0] / (np.linalg.norm(RAW[e][0], axis=1, keepdims=True) + 1e-9)
        return np.concatenate([i, p], 1).astype(np.float32)
    return feat

def strat_frames(n, T):
    edges = np.linspace(0, n, T + 1).astype(int)
    ix = [np.random.randint(edges[i], max(edges[i] + 1, min(edges[i + 1], n))) for i in range(T)]
    return np.array(sorted(set(ix)))
def rand_frames(n, T):
    return np.sort(np.random.choice(n, size=min(T, n), replace=n < T))

class Head(nn.Module):
    def __init__(self, din, dh=256, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.GELU(),
                                 nn.Linear(dh, dh), nn.GELU(), nn.Linear(dh, dout))
    def forward(self, x): return self.net(x)

def train_variant(name, n_train, grouped, stratified, p_same):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    TRAIN = pool[:n_train]; TRAIN = [e for e in TRAIN if e in RAW]
    feat = make_feat_fn(TRAIN)
    F = {e: feat(e) for e in TRAIN + EVALu}
    DIN = F[TRAIN[0]].shape[1]
    # item 分组
    groups = None
    if grouped:
        epm = np.stack([F[e].mean(0) for e in TRAIN]); epm /= np.linalg.norm(epm, axis=1, keepdims=True) + 1e-9
        gl = KMeans(args.G, n_init=4, random_state=0).fit_predict(epm)
        groups = [[TRAIN[i] for i in range(len(TRAIN)) if gl[i] == g] for g in range(args.G)]
        groups = [g for g in groups if len(g) >= args.batch_eps]
    fsamp = strat_frames if stratified else rand_frames
    head = Head(DIN); opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)
    losses = []
    for step in range(args.steps):
        if groups and random.random() < p_same:
            bes = random.sample(random.choice(groups), args.batch_eps)
        else:
            bes = random.sample(TRAIN, args.batch_eps)
        embs, idxs, lens = [], [], []
        for e in bes:
            f = F[e]; ix = fsamp(len(f), args.T)
            embs.append(head(torch.from_numpy(f[ix])))
            idxs.append(torch.from_numpy(ix).long()); lens.append(len(f))
        loss = compute_tcc_loss(embs=torch.stack(embs), idxs=torch.stack(idxs),
            seq_lens=torch.tensor(lens), stochastic_matching=False, normalize_embeddings=True,
            loss_type="regression_mse", similarity_type="l2", num_cycles=20, cycle_length=2,
            temperature=0.1, label_smoothing=0.1, variance_lambda=0.001, huber_delta=0.1,
            normalize_indices=True)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    head.eval()
    def hemb(x):
        with torch.no_grad(): z = head(torch.from_numpy(x)).numpy()
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    REFS = TRAIN[:args.knn_refs]
    REs = [hemb(F[r]) for r in REFS]; RTs = [np.arange(len(z)) / max(1, len(z) - 1) for z in REs]
    ts, rs, ms = [], [], []
    for e in EVALu:
        g = GT[e]
        if g.std() < 1e-6: continue
        z = hemb(F[e]); preds = [RTs[k][(z @ REs[k].T).argmax(1)] for k in range(len(REFS))]
        v = np.median(np.stack(preds), 0)
        ts.append(kendalltau(v, g)[0]); rs.append(pearsonr(v, g)[0]); ms.append(np.abs(v - g).mean())
    r = dict(tau=float(np.nanmean(ts)), r=float(np.nanmean(rs)), mae=float(np.nanmean(ms)),
             loss=float(np.mean(losses[-100:])), n_train=len(TRAIN))
    print(f"  {name:<34} tau={r['tau']:.3f} Pearson={r['r']:.3f} MAE={r['mae']:.3f} (loss {r['loss']:.3f})")
    if name.startswith("D"):
        torch.save(head.state_dict(), out / "tcc_head_v4.pt")
    return r

print("[v4] ===== 数据处理消融 (kai0 held-out 50 GT, readout=per-ref-median) =====")
results = {}
results["v3 baseline (random eps+frames, N200)"] = train_variant("v3", 200, False, False, 0)
results["A +stratified frames (N200)"]           = train_variant("A", 200, False, True, 0)
results["B +grouped batches p=0.7 (N200)"]       = train_variant("B", 200, True, True, 0.7)
results["C +grouped p=1.0 (N200)"]               = train_variant("C", 200, True, True, 1.0)
results["D B-recipe + N500"]                     = train_variant("D", 500, True, True, 0.7)
json.dump(results, open(out / "ablation.json", "w"), indent=2, ensure_ascii=False)
print(f"\n[v4] -> {out}/ablation.json")
