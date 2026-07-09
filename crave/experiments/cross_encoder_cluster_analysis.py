#!/usr/bin/env python
"""跨编码器聚类分析:固定 K=48,对比 DINOV2-L/DINOV3-H/Wan-VAE/SigLIP2 的聚类时间纯度。
指标:per-cluster temporal std(归一时间 T∈[0,1] 的 std,越低=越时间凝聚=越好的 milestone)。
额外对比 grid vs pooled 空间(对 DINOV3-H)。
Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=crave/src:lmwm/src:crave/experiments python crave/experiments/cross_encoder_cluster_analysis.py
"""
import sys, glob, time, numpy as np, cv2, av
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crave.render import setup_mpl
from crave.config import resolve_dataset
from crave.data import kai0
plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0")
K_FIX = 48
N_EPS = 200
rng = np.random.RandomState(42)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w); r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]; return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])

# ---- load all 3Hz cached features (same E/FR/T across dino, dinov3h, wan) ----
def load_shards(idx_path, shard_dir, dim):
    zf = np.load(idx_path); E, FR, T = zf["E"], zf["FR"], zf["T"]; N = int(zf["n"])
    feat = np.zeros((N, dim), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(shard_dir / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    return E, FR, T, feat, valid

E2, FR2, T2, d2, v2 = load_shards(REPO / "temp/crave_full/index_dino.npz", REPO / "temp/crave_full/dino", 1024)
E3, FR3, T3, d3, v3 = load_shards(REPO / "temp/crave_full_dinov3h/index.npz", REPO / "temp/crave_full_dinov3h", 1280)
EW, FRW, TW, dW, vW = load_shards(REPO / "temp/crave_full/index_wan.npz", REPO / "temp/crave_full/wan", 12288)
assert np.array_equal(E2, E3) and np.array_equal(FR2, FR3), "frame index mismatch!"

vi = np.where(v2 & v3 & vW)[0]
Ev, FRv, Tv = E2[vi], FR2[vi], T2[vi]
img2, img3, imgW = l2(d2[vi].astype(np.float32)), l2(d3[vi].astype(np.float32)), l2(dW[vi].astype(np.float32))
ep_list = sorted(set(Ev.tolist()))
print(f"loaded {len(vi)} aligned frames, {len(ep_list)} eps", flush=True)

# sample eps
samp_eps = sorted(rng.choice(ep_list, N_EPS, replace=False))
mask = np.isin(Ev, samp_eps)
Evs, FRvs, Tvs = Ev[mask], FRv[mask], Tv[mask]
i2s, i3s, iWs = img2[mask], img3[mask], imgW[mask]
print(f"sample: {len(i2s)} frames from {len(samp_eps)} eps", flush=True)

# ---- SigLIP2 encoder (standalone, no HF) ----
from transformers import AutoModel, AutoImageProcessor
siglip_path = "/vePFS/tim/workspace/hf_cache/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
print("loading SigLIP2...", flush=True)
siglip_model = AutoModel.from_pretrained(siglip_path, torch_dtype="float16").cuda().eval()
siglip_proc = AutoImageProcessor.from_pretrained(siglip_path)
import torch
cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
def siglip_encode(e, fr_idx):
    """encode SigLIP pooled features for specific frames of an episode"""
    vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
    cap = av.open(str(vid)); all_frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
    imgs = [all_frames[i] for i in fr_idx if i < len(all_frames)]
    if not imgs: return np.zeros((0, 1536), np.float16)
    with torch.no_grad():
        inp = siglip_proc(images=list(imgs), return_tensors="pt").to("cuda")
        out = siglip_model(**inp).pooler_output  # SigLIP2 SO400M: 1152D
        return out.float().cpu().numpy().astype(np.float32)

# extract SigLIP for sampled frames (batch by ep)
print("extracting SigLIP...", flush=True)
siglip_feat = np.zeros((len(Evs), 1152), np.float32)
for ei, e in enumerate(samp_eps):
    loc = np.where(Evs == e)[0]; o = np.argsort(FRvs[loc]); loc = loc[o]; fr = FRvs[loc]
    sf = siglip_encode(e, fr)
    if len(sf) == len(loc): siglip_feat[loc] = sf
    else:
        for j, idx in enumerate(loc):
            if j < len(sf): siglip_feat[idx] = sf[j]
    if (ei + 1) % 20 == 0: print(f"  SigLIP {ei+1}/{len(samp_eps)}", flush=True)
siglip_feat = l2(siglip_feat.astype(np.float32))
del siglip_model; torch.cuda.empty_cache()

# ---- grid-space features for DINOV3-H (same sampled frames) ----
from crave.encoders import load_encoder
enc = load_encoder("dinov3-h")
print("extracting DINOV3-H grid...", flush=True)
grid_feats = np.zeros((len(Evs), 1280 * 256), np.float16)  # flattened grid
for ei, e in enumerate(samp_eps):
    loc = np.where(Evs == e)[0]; o = np.argsort(FRvs[loc]); loc = loc[o]; fr = FRvs[loc]
    vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
    cap = av.open(str(vid)); all_frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
    imgs = [all_frames[i] for i in fr if i < len(all_frames)]
    if imgs:
        grids = enc.encode_grid(imgs)  # (n, 1280, 16, 16)
        grid_feats[loc[:len(grids)]] = grids.reshape(len(grids), -1).astype(np.float16)
    if (ei + 1) % 20 == 0: print(f"  grid {ei+1}/{len(samp_eps)}", flush=True)
grid_flat = l2(grid_feats.astype(np.float32))
# PCA-reduce grid for clustering
n_valid = (grid_flat.std(1) > 1e-6).sum()
print(f"grid valid frames: {n_valid}/{len(grid_flat)}; PCA (256D)...", flush=True)
pca = PCA(n_components=256, random_state=0).fit(grid_flat[grid_flat.std(1) > 1e-6])
grid_pca = pca.transform(grid_flat)

# ---- common pooling for "pooled-grid" (same as encode_pooled, spatial mean) ----
# i3s IS encode_pooled (=grid spatial mean). grid_pca is the full spatial layout.
# Also: grid pooled variant = PCA on grid spatial mean (same as i3s but PCA'd to 256D for fair comparison)
pooled_pca = PCA(n_components=256, random_state=0).fit_transform(i3s)

# ---- cluster each encoder space with same K ----
encoders = {
    "DINOv2-L(1024D)": i2s,
    "DINOv3-H(1280D)": i3s,
    "Wan-VAE(12288D)": iWs,
    "SigLIP2-SO400M(1152D)": siglip_feat,
    "D3H-grid(327kD→PCA256)": grid_pca,
    "D3H-pooled→PCA256": pooled_pca,
}

results = {}
for name, F in encoders.items():
    t0 = time.time()
    valid = F.std(1) > 1e-6
    Fv = F[valid]; Tv_valid = Tvs[valid]
    km = KMeans(K_FIX, n_init=3, random_state=0).fit(Fv)
    labs = km.labels_
    # per-cluster temporal std
    t_stds = [float(np.nanstd(Tv_valid[labs == k])) if (labs == k).sum() > 1 else np.nan for k in range(K_FIX)]
    t_stds = [x for x in t_stds if not np.isnan(x)]
    # coverage: fraction of eps that visit each cluster
    n_ep = len(samp_eps)
    cov = [len(set(Evs[valid][labs == k].tolist())) / n_ep for k in range(K_FIX)]
    mean_cov = np.mean(cov)
    elapsed = time.time() - t0
    results[name] = {"t_stds": t_stds, "cov": cov, "mean_std": np.mean(t_stds), "median_std": np.median(t_stds),
                     "mean_cov": mean_cov, "n_clusters_used": len(t_stds)}
    print(f"  {name}: mean_tstd={np.mean(t_stds):.4f} median_tstd={np.median(t_stds):.4f} mean_cov={mean_cov:.3f} ({elapsed:.1f}s)", flush=True)

# ---- figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.5))
names_plot = list(results.keys()); colors = plt.cm.tab10(np.linspace(0, 1, len(names_plot)))
# boxplot of per-cluster temporal std
bp_data = [results[n]["t_stds"] for n in names_plot]
bp = ax1.boxplot(bp_data, patch_artist=True, tick_labels=[n[:18] for n in names_plot])
for patch, c in zip(bp["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(0.7)
ax1.set_ylabel("per-cluster temporal std (越低越凝聚)"); ax1.set_title(f"K={K_FIX} · {N_EPS} eps · within-cluster temporal variance"); ax1.grid(alpha=0.25)
# bar: mean temporal std
means = [results[n]["mean_std"] for n in names_plot]
bars = ax2.bar(range(len(names_plot)), means, color=colors, alpha=0.8)
ax2.set_xticks(range(len(names_plot))); ax2.set_xticklabels([n[:18] for n in names_plot], rotation=30, ha="right")
ax2.set_ylabel("mean per-cluster temporal std"); ax2.set_title("聚类时间纯度 (越低越好)"); ax2.grid(alpha=0.25, axis="y")
for b, v in zip(bars, means): ax2.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
fig.suptitle(f"跨编码器聚类时间纯度对比 · K={K_FIX} 固定 · {N_EPS} ep · DINOv2/DINOv3/Wan-VAE/SigLIP2 + grid-vs-pooled 消融", fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94]); out = "crave/docs/visualization/encoders/cross_encoder_cluster_purity.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=120); print("SAVED", out)
