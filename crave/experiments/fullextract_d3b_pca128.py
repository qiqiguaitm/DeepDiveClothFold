#!/usr/bin/env python
"""DINOv3-base PCA→128D 全量 kai0_base 处理: 提取→PCA→聚类→Viterbi→label。
存 milestones + example value curves。
"""
import glob, time, numpy as np, av, cv2, torch
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0")
OUT_DIR = REPO / "temp/crave_d3b_pca128"; OUT_DIR.mkdir(parents=True, exist_ok=True)
NEPS = 3055; K_MAX = 160
rng = np.random.RandomState(42)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])

# ---- Step 1: Extract DINOv3-base for ALL eps ----
from transformers import AutoModel
D3B = "/vePFS/shock/.CACHE/hf_cache/hub/dinov3-vitb16-pretrain-lvd1689m"
print("loading DINOv3-base...", flush=True)
d3b = AutoModel.from_pretrained(D3B, torch_dtype=torch.float16, local_files_only=True).cuda().eval()

from crave.config import resolve_dataset; from crave.data import kai0
cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz")
E_all, FR_all, T_all = zf["E"], zf["FR"], zf["T"]
all_eps = sorted(set(E_all.tolist()))
print(f"{len(all_eps)} eps total", flush=True)

# Save features per-ep to be memory-efficient
FEAT_DIR = OUT_DIR / "feats"; FEAT_DIR.mkdir(exist_ok=True)
T_vals, E_vals = [], []
n_done = 0; t0 = time.time()
for e in all_eps:
    fp = FEAT_DIR / f"ep{e}.npy"
    if fp.exists(): n_done += 1; continue
    loc = np.where(E_all == e)[0]; o = np.argsort(FR_all[loc]); loc = loc[o]; fr = FR_all[loc]
    vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
    cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
    imgs = [frames[i] for i in fr if i < len(frames)]
    if len(imgs) < 3: continue
    imgs_t = torch.from_numpy(np.stack([cv2.resize(im, (256, 256)).transpose(2, 0, 1) for im in imgs])).float().cuda() / 255.0
    imgs_t = (imgs_t - torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    with torch.no_grad():
        out = d3b(imgs_t).last_hidden_state[:, 5:].mean(1).cpu().numpy().astype(np.float32)
        np.save(fp, out)
    n_done += 1
    if n_done % 200 == 0:
        el = time.time() - t0; eta = el / n_done * (len(all_eps) - n_done) / 60
        print(f"  [{n_done}/{len(all_eps)}] ep{e} ({el/60:.0f}min, ~{eta:.0f}min left)", flush=True)
del d3b; torch.cuda.empty_cache()
print(f"extracted {n_done} eps in {(time.time()-t0)/60:.1f}min", flush=True)

# ---- Step 2: Load all features + PCA to 128D ----
print("loading features for PCA...", flush=True)
all_F_list, all_T_list, all_E_list = [], [], []
for e in all_eps:
    fp = FEAT_DIR / f"ep{e}.npy"
    if not fp.exists(): continue
    f = np.load(fp); all_F_list.append(f)
    loc = np.where(E_all == e)[0]; o = np.argsort(FR_all[loc]); loc = loc[o]
    all_T_list.append(T_all[loc][:len(f)]); all_E_list.append(np.full(len(f), e))
all_F = np.concatenate(all_F_list); all_T = np.concatenate(all_T_list); all_E = np.concatenate(all_E_list)
print(f"{len(all_F)} total frames", flush=True)

print("PCA 768→128D...", flush=True)
t0 = time.time()
pca = PCA(n_components=128, random_state=0)
F_pca = l2(pca.fit_transform(l2(all_F)))
var = pca.explained_variance_ratio_.sum()
print(f"  var={var:.3f} ({time.time()-t0:.0f}s)", flush=True)

# ---- Step 3: Overcluster + Otsu k=1.0 ----
print("Overcluster + Otsu k=1.0...", flush=True)
K0 = int(np.clip(round(0.55 * np.sqrt(len(F_pca))), 64, 320))
t0 = time.time()
km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=8192, max_iter=30).fit(F_pca)
labs = km.labels_
print(f"  K0={K0} ({time.time()-t0:.0f}s)", flush=True)

clusters = []
for k in range(K0):
    mk = (labs == k)
    if mk.sum() < 10: continue
    cov = len(set(all_E[mk].tolist())) / len(all_eps)
    tpos = float(all_T[mk].mean())
    tstd = float(all_T[mk].std())
    clusters.append({"k": k, "cov": cov, "tpos": tpos, "tstd": tstd, "size": mk.sum()})

cs = np.array([c["cov"] for c in clusters])
med, std = np.median(cs), np.std(cs)
tau = float(med + 1.0 * std)  # k=1.0
sel = sorted([c for c in clusters if c["cov"] >= tau], key=lambda c: c["tpos"])
M = len(sel)
print(f"  τ={tau:.3f} (med={med:.3f} std={std:.3f}), K_eff={M} milestones", flush=True)

# Isotonic order + milestone stats
Cs = l2(np.array([km.cluster_centers_[c["k"]] for c in sel], dtype=np.float32))
Pord_raw = np.array([c["tpos"] for c in sel], dtype=np.float64)
Pord = np.asarray(IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pord_raw), dtype=np.float64)

# Per-milestone stats
covs_sel = np.array([c["cov"] for c in sel])
tstds_sel = np.array([c["tstd"] for c in sel])
print(f"  Pord: [{Pord[0]:.3f}, {Pord[-1]:.3f}]")
print(f"  coverage: mean={covs_sel.mean():.3f} median={np.median(covs_sel):.3f} min={covs_sel.min():.3f}")
print(f"  Tstd:    mean={tstds_sel.mean():.4f} median={np.median(tstds_sel):.4f} max={tstds_sel.max():.4f}")

# Save milestones
np.savez(OUT_DIR / "milestones.npz", C=Cs, Pord=Pord, tau=tau, K0=K0, M=M,
         covs=covs_sel, tstds=tstds_sel, pca_mean=pca.mean_, pca_components=pca.components_)
print(f"saved milestones to {OUT_DIR}/milestones.npz", flush=True)

# ---- Step 4: Example value curves (10 random eps) ----
LAM = 16.0
bins = np.unique(np.concatenate([[0.0], Pord, [1.0]]))
nb = len(bins); cb = [int(np.searchsorted(bins, p)) for p in Pord]
pen = LAM * np.abs(bins[:, None] - bins[None])

def vit_value(Fq):
    d = np.linalg.norm(Fq[:, None] - Cs[None], axis=2)
    em = np.full((len(Fq), nb), 1e3)
    for ci in range(M): em[:, cb[ci]] = np.minimum(em[:, cb[ci]], d[:, ci])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]
    BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; k = tr.argmin(1)
        cost = em[j] + tr[np.arange(nb), k]; BP[j] = k
    cost[nb - 1] -= 2
    s = int(cost.argmin()); path = np.zeros(len(Fq), int); path[-1] = s
    for j in range(len(Fq) - 2, -1, -1): s = BP[j + 1][s]; path[j] = s
    return bins[path]

# Sample 10 eps for visualization
import pandas as pd
CSQ = 1000
viz_eps = sorted(rng.choice(all_eps, 10, replace=False))
fig, axs = plt.subplots(2, 5, figsize=(22, 8)); axs = axs.ravel()
dd_all = []
for k, e in enumerate(viz_eps):
    loc = np.where(E_all == e)[0]; o = np.argsort(FR_all[loc]); loc = loc[o]
    Fq = F_pca[np.isin(all_E, [e])]  # get features for this ep
    if len(Fq) < 3: continue
    v3 = vit_value(Fq)
    # Interpolate to 30Hz
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy())
    n30 = len(st); xi = np.linspace(0, 1, len(v3)); xo = np.linspace(0, 1, n30)
    v30 = np.interp(xo, xi, v3)
    lo, hi = v30.min(), v30.max()
    if hi > lo + 1e-6: v30 = (v30 - lo) / (hi - lo)
    x = np.linspace(0, 1, n30)
    dd = float((np.maximum.accumulate(v30) - v30).max())
    dd_all.append(dd)
    axs[k].step(x, v30, where="post", color="#1f77b4", lw=1.2, alpha=.8)
    for p in Pord: axs[k].axhline(p, color="grey", lw=.3, alpha=.4)
    axs[k].set_title(f"ep{e}  n={n30}  dd={dd:.2f}", fontsize=9)
    axs[k].set_ylim(-.03, 1.03); axs[k].grid(alpha=.2)
    axs[k].set_xlabel("T"); axs[k].set_ylabel("value")
fig.suptitle(f"DINOv3-base PCA→128D · {M} milestones · λ={LAM} · k=1.0 · 10 example eps", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .95])
out_png = OUT_DIR / "example_values.png"; fig.savefig(out_png, dpi=120)
print(f"SAVED {out_png}", flush=True)
print(f"max drawdown: mean={np.mean(dd_all):.3f} max={max(dd_all):.3f}", flush=True)

# ---- Step 5: Per-milestone coverage bar chart ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.bar(range(M), covs_sel, color=plt.cm.viridis(np.linspace(0.2, 0.9, M)), alpha=.85)
ax1.set_xlabel("milestone index (ordered by progress)"); ax1.set_ylabel("cross-episode coverage")
ax1.set_title(f"{M} milestones · coverage (mean={covs_sel.mean():.3f})"); ax1.grid(alpha=.25, axis="y")
ax2.bar(range(M), tstds_sel, color=plt.cm.viridis(np.linspace(0.2, 0.9, M)), alpha=.85)
ax2.set_xlabel("milestone index"); ax2.set_ylabel("per-cluster T-std")
ax2.set_title(f"Per-cluster temporal std (mean={tstds_sel.mean():.4f})"); ax2.grid(alpha=.25, axis="y")
fig.suptitle(f"DINOv3-base PCA→128D · k=1.0 · {len(all_eps)} eps · milestone quality", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .93])
out_png2 = OUT_DIR / "milestone_quality.png"; fig.savefig(out_png2, dpi=120)
print(f"SAVED {out_png2}", flush=True)

print("\nDONE. Output:", OUT_DIR, flush=True)
