#!/usr/bin/env python
"""TCC v3: 修复 §2.4 冻结特征版失败的两个根因后重试 TCC 路线。
根因诊断 (post-hoc, 借助后续发现):
  ① v2 value = -||emb-goal|| —— 图33 已证 初态(布平摊)≈终态(布叠好), goal 距离在叠衣上
     结构性失效 (开局即高 → 负相关 τ=-0.31)
  ② v2 输入 = 未掩膜 DINOv2 CLS —— §2.6/2.10 已证 armmask⊕proprio 才是有效特征
修复:
  输入 = armmask patch-mean ⊕ proprio(state+Δstate, z-score)   [§2.10 配方]
  读出 = 对齐-进度 (soft-NN 到 K 条参考 ep 的归一化时间加权平均, GTCC 式)
  对照 = ① raw 特征同读出(零训练基线) ② TCC head + goal 距离(复现 v2 失效模式)
评测 = held-out 50 GT eps (V0 协议), tau/Pearson/MAE。
用法: python tcc_v3_armmask.py --steps 1500 --out temp/tcc_v3_kai0
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kendalltau, pearsonr

sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
from xirl.losses import compute_tcc_loss  # noqa

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_advantage"
CACHE = REPO / "temp/tcc_kai0_armmask/feat_cache"

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=200)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--batch-eps", type=int, default=8)
ap.add_argument("--frames-per-ep", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--knn-refs", type=int, default=30, help="进度读出用的参考 episode 数")
ap.add_argument("--tau-soft", type=float, default=0.1, help="soft-NN 温度")
ap.add_argument("--out", default="temp/tcc_v3_kai0")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
out = REPO / args.out
out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
chunks_size = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)

# ---- 数据: EVAL = V0 50 GT eps; train = 池中前 n-train (与 EVAL 不相交) ----
zp = np.load(REPO / "temp/recurrence_v0_kai0/embeddings.npz")
EVAL = sorted(set(zp["ep_ids"].tolist()))
all_eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.npz"))
pool = np.random.RandomState(0).permutation([e for e in all_eps if e not in set(EVAL)]).tolist()
TRAIN = pool[: args.n_train]

def load_feat(e):
    img = np.load(CACHE / f"ep{e}.npz")["f"]
    n = len(img)
    pq = DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    dst = np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])
    return img, np.concatenate([st, dst], 1)

print("[v3] loading features ...")
IMGS, PRPS = {}, {}
for e in TRAIN + EVAL:
    try:
        IMGS[e], PRPS[e] = load_feat(e)
    except Exception as ex:
        print(f"  skip ep{e}: {type(ex).__name__}")
TRAIN = [e for e in TRAIN if e in IMGS]
EVALu = [e for e in EVAL if e in IMGS]
allP = np.concatenate([PRPS[e] for e in TRAIN])
PMU, PSD = allP.mean(0), allP.std(0) + 1e-8
def feat(e):
    p = (PRPS[e] - PMU) / PSD
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    i = IMGS[e] / (np.linalg.norm(IMGS[e], axis=1, keepdims=True) + 1e-9)
    return np.concatenate([i, p], 1).astype(np.float32)
F = {e: feat(e) for e in TRAIN + EVALu}
DIN = F[TRAIN[0]].shape[1]
print(f"[v3] train {len(TRAIN)} eps, eval {len(EVALu)} eps, din={DIN}, dev={dev}")

GT = {}
for e in EVALu:
    pq = DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet"
    g = pd.read_parquet(pq, columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy()
    n = len(F[e])
    GT[e] = g[np.minimum(np.arange(n) * 10, len(g) - 1)]

# ---- 读出: soft-NN 对齐进度 ----
REFS = TRAIN[: args.knn_refs]
def progress_readout(emb_fn):
    """emb_fn: np(n,din)->np(n,d). 返回 {eval_ep: value 曲线}"""
    ref_emb, ref_t = [], []
    for r in REFS:
        z = emb_fn(F[r])
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        ref_emb.append(z)
        ref_t.append(np.arange(len(z)) / max(1, len(z) - 1))
    RE = np.concatenate(ref_emb); RT = np.concatenate(ref_t)
    vals = {}
    for e in EVALu:
        z = emb_fn(F[e])
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        sim = z @ RE.T / args.tau_soft
        sim -= sim.max(1, keepdims=True)
        w = np.exp(sim); w /= w.sum(1, keepdims=True)
        vals[e] = w @ RT
    return vals

def evaluate(vals, name):
    ts, rs, ms = [], [], []
    for e in EVALu:
        g = GT[e]
        if g.std() < 1e-6: continue
        v = vals[e]
        ts.append(kendalltau(v, g)[0]); rs.append(pearsonr(v, g)[0]); ms.append(np.abs(v - g).mean())
    print(f"  {name:<28} tau={np.nanmean(ts):.3f}  Pearson={np.nanmean(rs):.3f}  MAE={np.nanmean(ms):.3f}")
    return dict(tau=float(np.nanmean(ts)), r=float(np.nanmean(rs)), mae=float(np.nanmean(ms)))

print("\n[v3] ===== 基线: raw armmask⊕proprio + 对齐-进度读出 (零训练) =====")
res = {"raw_align": evaluate(progress_readout(lambda x: x), "raw feat + align-progress")}

# ---- TCC head 训练 ----
class Head(nn.Module):
    def __init__(self, din, dh=256, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.GELU(),
                                 nn.Linear(dh, dh), nn.GELU(), nn.Linear(dh, dout))
    def forward(self, x): return self.net(x)

head = Head(DIN).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)
T = args.frames_per_ep
losses = []
print(f"\n[v3] training TCC head {args.steps} steps on {dev} ...")
for step in range(args.steps):
    bes = random.sample(TRAIN, args.batch_eps)
    embs, idxs, lens = [], [], []
    for e in bes:
        f = F[e]; n = len(f)
        ix = np.sort(np.random.choice(n, size=min(T, n), replace=n < T))
        embs.append(head(torch.from_numpy(f[ix]).to(dev)))
        idxs.append(torch.from_numpy(ix).long())
        lens.append(n)
    loss = compute_tcc_loss(
        embs=torch.stack(embs), idxs=torch.stack(idxs).to(dev),
        seq_lens=torch.tensor(lens).to(dev),
        stochastic_matching=False, normalize_embeddings=True,
        loss_type="regression_mse", similarity_type="l2",
        num_cycles=20, cycle_length=2, temperature=0.1,
        label_smoothing=0.1, variance_lambda=0.001,
        huber_delta=0.1, normalize_indices=True)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss))
    if (step + 1) % 100 == 0:
        print(f"  step {step+1}/{args.steps} loss {np.mean(losses[-100:]):.4f}")
torch.save(head.state_dict(), out / "tcc_head_v3.pt")
head.eval()

def head_emb(x):
    with torch.no_grad():
        return head(torch.from_numpy(x).to(dev)).cpu().numpy()

print("\n[v3] ===== TCC head + 对齐-进度读出 =====")
res["tcc_align"] = evaluate(progress_readout(head_emb), "TCC head + align-progress")

print("\n[v3] ===== TCC head + goal 距离 (复现 v2 失效模式对照) =====")
with torch.no_grad():
    goal = torch.stack([head(torch.from_numpy(F[e][-3:]).to(dev)).mean(0) for e in TRAIN[:50]]).mean(0)
gvals = {}
for e in EVALu:
    with torch.no_grad():
        z = head(torch.from_numpy(F[e]).to(dev))
        gvals[e] = (-torch.norm(z - goal, dim=-1)).cpu().numpy()
res["tcc_goal"] = evaluate(gvals, "TCC head + goal-dist (v2式)")

json.dump({**res, "loss_first100": float(np.mean(losses[:100])), "loss_last100": float(np.mean(losses[-100:])),
           "n_train": len(TRAIN), "steps": args.steps}, open(out / "eval_v3.json", "w"), indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
axes[0].plot(np.convolve(losses, np.ones(50)/50, mode="valid"))
axes[0].set_title(f"TCC loss (last={np.mean(losses[-100:]):.3f}; v2塌缩位=0.0815)", fontsize=9)
axes[0].axhline(1/12, color="r", ls="--", lw=0.8); axes[0].grid(alpha=.3)
va = progress_readout(head_emb); vr = progress_readout(lambda x: x)
for ax, e in zip(axes[1:], [x for x in EVALu if GT[x].std() > 1e-6][:3]):
    xx = np.arange(len(GT[e]))/3
    ax.plot(xx, GT[e], "k-", lw=2, label="GT")
    ax.plot(xx, vr[e], "-", color="#888", lw=1, label="raw align")
    ax.plot(xx, va[e], "-", color="#2ca02c", lw=1.6, label="TCC align")
    ax.plot(xx, (gvals[e]-gvals[e].min())/(np.ptp(gvals[e])+1e-9), "r--", lw=1, alpha=.6, label="goal-dist (norm)")
    ax.set_title(f"ep{e}", fontsize=9); ax.legend(fontsize=6); ax.grid(alpha=.3); ax.set_ylim(-0.05, 1.1)
fig.suptitle("TCC v3: armmask+proprio features, align-progress readout vs goal-dist", fontsize=10)
fig.tight_layout(); fig.savefig(out / "tcc_v3_curves.png", dpi=120)
print(f"\n[v3] outputs -> {out}/")
