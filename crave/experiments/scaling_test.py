#!/usr/bin/env python
"""DINOv2-base PCA→64D 数据 scaling 测试: 逐步增加 ep 数, K/Tstd/τ 是否收敛。"""
import glob, time, numpy as np, av, cv2, torch
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0"); rng = np.random.RandomState(42)
MAX_EPS = 500  # start with 500 eps for speed

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]; return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])

# Load dinov2-base
from transformers import AutoModel
print("loading DINOv2-base...", flush=True)
d2b = AutoModel.from_pretrained("/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-base",
                                 torch_dtype=torch.float16).cuda().eval()

from crave.config import resolve_dataset; from crave.data import kai0
cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E, FR, T = zf["E"], zf["FR"], zf["T"]
all_eps = sorted(set(E.tolist()))
samp = sorted(rng.choice(all_eps, MAX_EPS, replace=False))
print(f"sampled {MAX_EPS} eps from {len(all_eps)} total", flush=True)

# Extract features for all 500 eps
print("extracting DINOv2-base features...", flush=True)
feats_by_ep = {}; t0 = time.time()
for ei, e in enumerate(samp):
    loc = np.where(E == e)[0]; o = np.argsort(FR[loc]); loc = loc[o]; fr = FR[loc]
    vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
    cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
    imgs = [frames[i] for i in fr if i < len(frames)]
    if len(imgs) < 3: continue
    imgs_t = torch.from_numpy(np.stack([cv2.resize(im, (224, 224)).transpose(2, 0, 1) for im in imgs])).float().cuda() / 255.0
    imgs_t = (imgs_t - torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    with torch.no_grad():
        out = d2b(imgs_t).last_hidden_state[:, 1:].mean(1).cpu().numpy().astype(np.float32)
        feats_by_ep[e] = (l2(out), T[loc][:len(out)])
    if (ei + 1) % 100 == 0: print(f"  {ei+1}/{MAX_EPS} ({(time.time()-t0)/60:.1f}min)", flush=True)
del d2b; torch.cuda.empty_cache()
print(f"extracted {len(feats_by_ep)} eps in {(time.time()-t0)/60:.1f}min", flush=True)

# PCA on all 500 eps features
all_F = np.concatenate([v[0] for v in feats_by_ep.values()])
pca = PCA(n_components=64, random_state=0).fit(all_F)
print(f"PCA256 fitted, var={pca.explained_variance_ratio_.sum():.3f}", flush=True)

# Scaling test: increasing N
N_vals = [25, 50, 100, 200, 300, 400, min(500, len(samp))]
results = []
for N in N_vals:
    subsamp = samp[:N]
    F_sub = np.concatenate([feats_by_ep[e][0] for e in subsamp])
    T_sub = np.concatenate([feats_by_ep[e][1] for e in subsamp])
    E_sub = np.concatenate([np.full(len(feats_by_ep[e][1]), e) for e in subsamp])
    F_pca = l2(pca.transform(F_sub))
    # Overcluster+Otsu
    K0 = int(np.clip(round(0.55 * np.sqrt(len(F_sub))), 32, 160))
    km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=4096, max_iter=30).fit(F_pca)
    labs = km.labels_
    clusters = []
    for k in range(K0):
        mk = (labs == k)
        if mk.sum() < 5: continue
        clusters.append({"t_std": float(np.nanstd(T_sub[mk])), "cov": len(set(E_sub[mk].tolist())) / N})
    if not clusters: continue
    cs = [c["cov"] for c in clusters]; tau = float(np.median(cs) + 0.5 * np.std(cs))
    sel = [c for c in clusters if c["cov"] >= tau]
    tstds = [c["t_std"] for c in sel]
    results.append({"N": N, "K0": K0, "K_eff": len(sel), "tau": tau,
                    "Tstd": float(np.mean(tstds)), "Tstd_med": float(np.median(tstds))})
    print(f"N={N}: K₀={K0}→K_eff={len(sel)} τ={tau:.3f} Tstd={np.mean(tstds):.4f}", flush=True)

# Figure
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
Ns = [r["N"] for r in results]
# K_eff
axs[0].plot(Ns, [r["K_eff"] for r in results], "o-", color="#1f77b4", lw=2, ms=8)
axs[0].set_xlabel("N eps"); axs[0].set_ylabel("K_eff"); axs[0].set_title("Milestone 数 vs 数据量")
axs[0].grid(alpha=.25)
# Tstd
axs[1].plot(Ns, [r["Tstd"] for r in results], "o-", color="#2ca02c", lw=2, ms=8)
axs[1].set_xlabel("N eps"); axs[1].set_ylabel("mean Tstd"); axs[1].set_title("时间纯度 vs 数据量")
axs[1].grid(alpha=.25)
# τ
axs[2].plot(Ns, [r["tau"] for r in results], "o-", color="#ff7f0e", lw=2, ms=8)
axs[2].set_xlabel("N eps"); axs[2].set_ylabel("τ (coverage threshold)"); axs[2].set_title("Otsu 阈值 vs 数据量")
axs[2].grid(alpha=.25)
fig.suptitle(f"DINOv2-base PCA→64D 数据 Scaling 测试 · K_eff/Tstd/τ 收敛性", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .94])
out = "crave/docs/visualization/encoders/scaling_test.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=130)
print("SAVED", out, flush=True)
