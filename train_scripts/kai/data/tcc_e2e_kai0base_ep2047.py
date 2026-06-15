"""端到端 TCC (kai0_base, 末4块微调) 在 ep2047 上输出连续进度, 供四方对比。
帧 temp/tcc_e2e_frames/kai0base, proprio kai0_base parquet. ep2047 排除出训练(held-out)。
输出: temp/_e2e_kai0base_ep2047.npz (v_e2e 3Hz)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoModel
sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
from xirl.losses import compute_tcc_loss
np.random.seed(0); torch.manual_seed(0)
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
BASE = REPO / "kai0/data/Task_A/kai0_base"
FR = REPO / "temp/tcc_e2e_frames/kai0base"
csB = json.load(open(BASE / "meta/info.json"))["chunks_size"]
TEST = 2047; dev = "cuda"
IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
eps = sorted(int(p.stem[2:]) for p in FR.glob("ep*.npz"))
TRAIN = [e for e in eps if e != TEST]
print(f"e2e train {len(TRAIN)} eps (held-out ep{TEST})", flush=True)

def prop(e, n):
    st = np.stack(pd.read_parquet(BASE / "data" / f"chunk-{e//csB:03d}" / f"episode_{e:06d}.parquet",
                                  columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    return np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1).astype(np.float32)
IMG, PR = {}, {}
for e in eps:
    IMG[e] = np.load(FR / f"ep{e}.npz")["frames"]; PR[e] = prop(e, len(IMG[e]))
allp = np.concatenate([PR[e] for e in TRAIN]); MU, SD = allp.mean(0), allp.std(0) + 1e-8
for e in PR:
    p = (PR[e] - MU) / SD; PR[e] = (p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

bb = AutoModel.from_pretrained("facebook/dinov2-small").to(dev)
for p in bb.parameters(): p.requires_grad_(False)
bbp = []
for blk in bb.encoder.layer[-4:]:
    for p in blk.parameters(): p.requires_grad_(True); bbp.append(p)
for p in bb.layernorm.parameters(): p.requires_grad_(True); bbp.append(p)
head = nn.Sequential(nn.Linear(412, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128)).to(dev)
opt = torch.optim.AdamW([{"params": head.parameters(), "lr": 1e-3}, {"params": bbp, "lr": 1e-5}], weight_decay=1e-5)

def emb(fr_u8, pr, train):
    x = torch.from_numpy(fr_u8).to(dev).permute(0, 3, 1, 2).float() / 255.0; x = (x - IMEAN) / ISTD
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        vis = bb(x).last_hidden_state[:, 1:].mean(1).float()
    vis = vis / (vis.norm(dim=-1, keepdim=True) + 1e-9)
    return head(torch.cat([vis, torch.from_numpy(pr).to(dev)], -1))

for step in range(1000):
    bes = list(np.random.choice(TRAIN, 8, replace=False)); embs, idxs, lens = [], [], []
    for e in bes:
        n = len(IMG[e]); ix = np.sort(np.random.choice(n, size=24, replace=n < 24))
        embs.append(emb(IMG[e][ix], PR[e][ix], True)); idxs.append(torch.from_numpy(ix).long()); lens.append(n)
    loss = compute_tcc_loss(embs=torch.stack(embs), idxs=torch.stack(idxs).to(dev), seq_lens=torch.tensor(lens).to(dev),
        stochastic_matching=False, normalize_embeddings=True, loss_type="regression_mse", similarity_type="l2",
        num_cycles=20, cycle_length=2, temperature=0.1, label_smoothing=0.1, variance_lambda=0.001, huber_delta=0.1, normalize_indices=True)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 200 == 0: print(f"  step {step+1} loss {float(loss):.4f}", flush=True)
bb.eval(); head.eval()
@torch.no_grad()
def embep(e):
    o = []
    for b in range(0, len(IMG[e]), 128): o.append(emb(IMG[e][b:b+128], PR[e][b:b+128], False).cpu().numpy())
    z = np.concatenate(o); return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
def med(a, w=9):
    h = w // 2; return np.array([np.median(a[max(0, j-h):j+h+1]) for j in range(len(a))])
REFS = TRAIN[:30]; REs = [embep(e) for e in REFS]; RTs = [np.arange(len(z))/max(1, len(z)-1) for z in REs]
zq = embep(TEST); preds = [RTs[k][(zq @ REs[k].T).argmax(1)] for k in range(len(REFS))]
v = med(np.median(np.stack(preds), 0))
np.savez(REPO / "temp/_e2e_kai0base_ep2047.npz", v_e2e=v)
print(f"e2e ep{TEST}: end{v[-1]:.2f} mono{np.mean(np.diff(v)>=-1e-6):.0%} len{len(v)}", flush=True); print("DONE", flush=True)
