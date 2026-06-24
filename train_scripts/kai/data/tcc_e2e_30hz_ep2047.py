"""端到端 TCC (kai0_base 末4块) 30Hz 逐帧 readout + 时序平滑, 验证端到端是 30Hz 连续化最终形态。
对照 frozen-TCC-30Hz(_solve_ep2047_30hz.npz, 单调81%)。ep2047 held-out。
输出 temp/_e2e30_ep2047.npz + docs/.../e2e_30hz_ep2047.png
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, av, matplotlib, os
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from transformers import AutoModel
sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
from xirl.losses import compute_tcc_loss
_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
np.random.seed(0); torch.manual_seed(0)
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
BASE = REPO / "kai0/data/Task_A/kai0_base"
FR = REPO / "temp/tcc_e2e_frames/kai0base"          # 3Hz 训练帧
csB = json.load(open(BASE / "meta/info.json"))["chunks_size"]
TEST = 2047; dev = "cuda"
IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
eps = sorted(int(p.stem[2:]) for p in FR.glob("ep*.npz")); TRAIN = [e for e in eps if e != TEST]

def prop3(e, n):
    st = np.stack(pd.read_parquet(BASE / "data" / f"chunk-{e//csB:03d}" / f"episode_{e:06d}.parquet",
                                  columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    return np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1).astype(np.float32)
IMG, PR = {}, {}
for e in eps:
    IMG[e] = np.load(FR / f"ep{e}.npz")["frames"]; PR[e] = prop3(e, len(IMG[e]))
allp = np.concatenate([PR[e] for e in TRAIN]); MU, SD = allp.mean(0), allp.std(0) + 1e-8
def pnorm(p): q = (p - MU) / SD; return (q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
for e in PR: PR[e] = pnorm(PR[e])

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
    if (step + 1) % 250 == 0: print(f"  step {step+1} loss {float(loss):.4f}", flush=True)
bb.eval(); head.eval()
torch.save({"head": head.state_dict(), "bb": bb.state_dict()}, REPO / "temp/_e2e_kai0base_model.pt")

@torch.no_grad()
def embed_frames(frames_u8, pr):
    out = []
    for b in range(0, len(frames_u8), 128):
        out.append(emb(frames_u8[b:b+128], pr[b:b+128], False).cpu().numpy())
    z = np.concatenate(out); return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
# refs (3Hz)
REFS = TRAIN[:30]; REs = [embed_frames(IMG[e], PR[e]) for e in REFS]; RTs = [np.arange(len(z))/max(1,len(z)-1) for z in REs]

# ---- ep2047 全 2629 帧 (30Hz) 解码 + proprio(原生 30Hz) ----
def crop224(im):
    s = 224/min(im.shape[:2]); import cv2
    g = cv2.resize(im, (round(im.shape[1]*s), round(im.shape[0]*s))); h, w = g.shape[:2]; y, x = (h-224)//2, (w-224)//2
    return g[y:y+224, x:x+224]
mp4 = BASE/"videos"/f"chunk-{TEST//csB:03d}"/"observation.images.top_head"/f"episode_{TEST:06d}.mp4"
c = av.open(str(mp4)); frs = [crop224(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]; c.close()
frs = np.stack(frs).astype(np.uint8); n30 = len(frs)
st = np.stack(pd.read_parquet(BASE/"data"/f"chunk-{TEST//csB:03d}"/f"episode_{TEST:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())[:n30]
pr30 = pnorm(np.concatenate([st, np.vstack([np.zeros((1,14)), np.diff(st, axis=0)])], 1).astype(np.float32))
zq = embed_frames(frs, pr30)
preds = [RTs[k][(zq @ REs[k].T).argmax(1)] for k in range(len(REFS))]
raw30 = np.median(np.stack(preds), 0)
def med(a, w): h = w//2; return np.array([np.median(a[max(0,j-h):j+h+1]) for j in range(len(a))])
e2e_med = med(raw30, 27)                    # 时序平滑①: 中值 0.9s
# 时序平滑②: 因果 EMA(更贴在线使用)
ema = np.zeros_like(e2e_med); ema[0] = e2e_med[0]
for i in range(1, len(ema)): ema[i] = 0.92*ema[i-1] + 0.08*e2e_med[i]

z = np.load(REPO/"temp/_solve_ep2047_30hz.npz"); frozen30, crave30, ae30 = z["tcc"], z["crave"], z["ae"]
NF = min(n30, len(frozen30));
def mono(v): return np.mean(np.diff(v) >= -1e-6)
def rough(v): return np.mean(np.abs(np.diff(v, 2)))
e2e_med, ema, frozen30 = e2e_med[:NF], ema[:NF], frozen30[:NF]
print(f"30Hz mono: e2e中值{mono(e2e_med):.0%} e2e-EMA{mono(ema):.0%} frozen{mono(frozen30):.0%}", flush=True)
print(f"30Hz 抖动(2nd-diff): e2e{rough(e2e_med):.4f} ema{rough(ema):.4f} frozen{rough(frozen30):.4f}", flush=True)
print(f"end: e2e{e2e_med[-1]:.2f} ema{ema[-1]:.2f} frozen{frozen30[-1]:.2f}", flush=True)
np.savez(REPO/"temp/_e2e30_ep2047.npz", e2e=e2e_med, ema=ema, raw=raw30[:NF])

x = np.arange(NF)
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(x, frozen30, color="#888", lw=1.3, alpha=.8, label=f"frozen-TCC 30Hz (单调{mono(frozen30):.0%} 抖{rough(frozen30):.4f})")
ax.plot(x, e2e_med, color="#2ca02c", lw=2.0, label=f"端到端-TCC 30Hz 中值 (单调{mono(e2e_med):.0%} 抖{rough(e2e_med):.4f})")
ax.plot(x, ema, color="#9467bd", lw=1.6, ls="--", label=f"端到端-TCC 30Hz 因果EMA (单调{mono(ema):.0%} 抖{rough(ema):.4f})")
ax.axhline(1, color="#999", ls=":", lw=1); ax.set_xlim(0, NF); ax.set_ylim(-0.05, 1.12)
ax.set_xlabel("frame (30Hz)"); ax.set_ylabel("value"); ax.grid(alpha=.25); ax.legend(fontsize=9, loc="upper left")
ax.set_title(f"kai0_base ep{TEST} 30Hz: 端到端-TCC (中值/EMA时序平滑) vs frozen-TCC — 端到端更平滑更单调?", fontsize=11.5)
out = REPO/"docs/visualization/cross_episode_recurrence_value/e2e_30hz_ep2047.png"
fig.tight_layout(); fig.savefig(out, dpi=125); print("SAVED", out, flush=True); print("DONE", flush=True)
