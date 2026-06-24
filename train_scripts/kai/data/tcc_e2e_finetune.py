#!/usr/bin/env python
"""TCC 端到端微调 (§2.4.3 未来工作#1): 不冻结 DINOv2, 用 cycle-consistency 学
progress-aware backbone, 突破 frozen 上限 (v3 frozen τ≈0.75)。
- 输入 = 原图 (3Hz 224, temp/tcc_e2e_frames 缓存) → DINOv2 patch-mean (384) ⊕ proprio (28)
- --mode finetune: 解冻 backbone 末 K 块 + head (backbone lr 1e-5, head lr 1e-3)
  --mode frozen:   backbone 全冻, 仅训 head (= 同管线 frozen 对照, 隔离端到端效应)
- 损失 = XIRL compute_tcc_loss (同 v3 配置); 评测 = kai0 held-out 50 GT, per-ref-argmax-median
用法: CUDA_VISIBLE_DEVICES=0 python tcc_e2e_finetune.py --mode finetune --out temp/tcc_e2e_ft
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoModel
from scipy.stats import kendalltau, pearsonr
sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
from xirl.losses import compute_tcc_loss

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["finetune", "frozen"], default="finetune")
ap.add_argument("--unfreeze-blocks", type=int, default=4)
ap.add_argument("--n-train", type=int, default=250)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--batch-eps", type=int, default=8)
ap.add_argument("--T", type=int, default=24)
ap.add_argument("--lr-head", type=float, default=1e-3)
ap.add_argument("--lr-bb", type=float, default=1e-5)
ap.add_argument("--knn-refs", type=int, default=30)
ap.add_argument("--out", default="temp/tcc_e2e_ft")
args = ap.parse_args()
random.seed(0); np.random.seed(0); torch.manual_seed(0)
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_advantage"
FRAMES = REPO / "temp/tcc_e2e_frames/kai0"
CACHE = REPO / "temp/tcc_kai0_armmask/feat_cache"
OUT = REPO / args.out; OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
chunks_size = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)
IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

zp = np.load(REPO / "temp/recurrence_v0_kai0/embeddings.npz")
EVAL = sorted(set(zp["ep_ids"].tolist()))
all_eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.npz"))
pool = np.random.RandomState(0).permutation([e for e in all_eps if e not in set(EVAL)]).tolist()
TRAIN = [e for e in pool[:args.n_train] if (FRAMES / f"ep{e}.npz").exists()]
EVALu = [e for e in EVAL if (FRAMES / f"ep{e}.npz").exists()]
print(f"[e2e:{args.mode}] train {len(TRAIN)} eval {len(EVALu)}")

FR = {}   # ep -> uint8 frames [n,224,224,3]  (RAM, 463GB available)
PR = {}   # ep -> proprio [n,28]
for e in TRAIN + EVALu:
    FR[e] = np.load(FRAMES / f"ep{e}.npz")["frames"]
    n = len(FR[e])
    st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                                  columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    PR[e] = np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1).astype(np.float32)
allp = np.concatenate([PR[e] for e in TRAIN]); PMU, PSD = allp.mean(0), allp.std(0) + 1e-8
for e in PR:
    p = (PR[e] - PMU) / PSD; PR[e] = (p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
GT = {}
for e in EVALu:
    g = pd.read_parquet(DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                        columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy()
    GT[e] = g[np.minimum(np.arange(len(FR[e])) * 10, len(g) - 1)]

bb = AutoModel.from_pretrained("facebook/dinov2-small").to(dev)
if args.mode == "frozen":
    bb.eval()
    for p in bb.parameters(): p.requires_grad_(False)
    bb_params = []
else:
    for p in bb.parameters(): p.requires_grad_(False)
    bb_params = []
    for blk in bb.encoder.layer[-args.unfreeze_blocks:]:
        for p in blk.parameters(): p.requires_grad_(True); bb_params.append(p)
    for p in bb.layernorm.parameters(): p.requires_grad_(True); bb_params.append(p)
head = nn.Sequential(nn.Linear(412, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128)).to(dev)
opt = torch.optim.AdamW([{"params": head.parameters(), "lr": args.lr_head},
                         {"params": bb_params, "lr": args.lr_bb}], weight_decay=1e-5)

def imgs_to_emb(frames_u8, prop, train_mode):
    x = torch.from_numpy(frames_u8).to(dev).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMEAN) / ISTD
    ctx = torch.enable_grad() if (train_mode and args.mode == "finetune") else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        vis = bb(x).last_hidden_state[:, 1:].mean(1).float()
    vis = vis / (vis.norm(dim=-1, keepdim=True) + 1e-9)
    p = torch.from_numpy(prop).to(dev)
    return head(torch.cat([vis, p], -1))

losses = []
for step in range(args.steps):
    bes = random.sample(TRAIN, args.batch_eps)
    embs, idxs, lens = [], [], []
    for e in bes:
        n = len(FR[e]); ix = np.sort(np.random.choice(n, size=min(args.T, n), replace=n < args.T))
        z = imgs_to_emb(FR[e][ix], PR[e][ix], True)
        embs.append(z); idxs.append(torch.from_numpy(ix).long()); lens.append(n)
    if args.mode == "frozen": head.train()
    loss = compute_tcc_loss(embs=torch.stack(embs), idxs=torch.stack(idxs).to(dev),
        seq_lens=torch.tensor(lens).to(dev), stochastic_matching=False, normalize_embeddings=True,
        loss_type="regression_mse", similarity_type="l2", num_cycles=20, cycle_length=2,
        temperature=0.1, label_smoothing=0.1, variance_lambda=0.001, huber_delta=0.1, normalize_indices=True)
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    if (step + 1) % 100 == 0:
        print(f"  [{args.mode}] step {step+1}/{args.steps} loss {np.mean(losses[-100:]):.4f}", flush=True)

bb.eval(); head.eval()
@torch.no_grad()
def emb_ep(e):
    out = []
    for b in range(0, len(FR[e]), 128):
        out.append(imgs_to_emb(FR[e][b:b+128], PR[e][b:b+128], False).cpu().numpy())
    z = np.concatenate(out); return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

REFS = TRAIN[:args.knn_refs]
REs = [emb_ep(r) for r in REFS]; RTs = [np.arange(len(z)) / max(1, len(z) - 1) for z in REs]
ts, rs, ms = [], [], []
for e in EVALu:
    g = GT[e]
    if g.std() < 1e-6: continue
    z = emb_ep(e); preds = [RTs[k][(z @ REs[k].T).argmax(1)] for k in range(len(REFS))]
    v = np.median(np.stack(preds), 0)
    ts.append(kendalltau(v, g)[0]); rs.append(pearsonr(v, g)[0]); ms.append(np.abs(v - g).mean())
res = dict(mode=args.mode, tau=float(np.nanmean(ts)), r=float(np.nanmean(rs)), mae=float(np.nanmean(ms)),
           loss_last=float(np.mean(losses[-100:])), n_train=len(TRAIN), steps=args.steps)
print(f"\n[e2e:{args.mode}] RESULT tau={res['tau']:.3f} Pearson={res['r']:.3f} MAE={res['mae']:.3f}")
json.dump(res, open(OUT / f"eval_{args.mode}.json", "w"), indent=2)
torch.save({"head": head.state_dict(), "bb": bb.state_dict() if args.mode == "finetune" else None},
           OUT / f"tcc_e2e_{args.mode}.pt")
print(f"[e2e:{args.mode}] -> {OUT}/")
