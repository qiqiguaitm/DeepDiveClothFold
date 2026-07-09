#!/usr/bin/env python
"""DINOv2 scale ablation: base(768D) vs large(1024D). Uses xiezhicong's cached dinov2-base."""
import sys, glob, time, numpy as np, av, cv2, torch
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0"); NEPS = 100; rng = np.random.RandomState(42)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])

def load_shards(idx_path, shard_dir, dim):
    zf = np.load(idx_path); E, FR, T = zf["E"], zf["FR"], zf["T"]; N = int(zf["n"])
    feat = np.zeros((N, dim), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(shard_dir / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    return E, FR, T, feat, valid

# Load DINOv2-large cached features
E2, FR2, T2, d2, v2 = load_shards(REPO / "temp/crave_full/index_dino.npz", REPO / "temp/crave_full/dino", 1024)
vi = np.where(v2)[0]; Ev, Tv = E2[vi], T2[vi]
img2 = l2(d2[vi].astype(np.float32))
samp = sorted(rng.choice(sorted(set(Ev.tolist())), NEPS, replace=False))
m = np.isin(Ev, samp); Evs, Tvs = Ev[m], Tv[m]
img2m = img2[m]

# Extract DINOv2-base features for same eps
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E, FR, T = zf["E"], zf["FR"], zf["T"]
from transformers import AutoModel
print("loading DINOv2-base from xiezhicong cache...", flush=True)
d2b = AutoModel.from_pretrained("/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-base",
                                 torch_dtype=torch.float16).cuda().eval()
from crave.config import resolve_dataset; from crave.data import kai0
cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
d2b_feats = []; t0 = time.time()
for ei, e in enumerate(samp):
    loc = np.where(E == e)[0]; o = np.argsort(FR[loc]); loc = loc[o]; fr = FR[loc]
    vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
    cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
    imgs = [frames[i] for i in fr if i < len(frames)]
    if len(imgs) < 3: continue
    # DINOv2 manual encode: resize to 224, normalize
    imgs_t = torch.from_numpy(np.stack([cv2.resize(im, (224, 224)).transpose(2, 0, 1) for im in imgs])).float().cuda() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    imgs_t = (imgs_t - mean) / std
    with torch.no_grad():
        out = d2b(imgs_t).last_hidden_state[:, 1:].mean(1).cpu().numpy().astype(np.float32)
        d2b_feats.append(out)
    if (ei + 1) % 20 == 0: print(f"  {ei+1}/{len(samp)} ({(time.time()-t0)/60:.1f}min)", flush=True)
d2b_all = l2(np.concatenate(d2b_feats))
n = min(len(d2b_all), len(img2m))
d2b_all, img2m_cut, Evs_cut, Tvs_cut = d2b_all[:n], img2m[:n], Evs[:n], Tvs[:n]
del d2b; torch.cuda.empty_cache()

# Overcluster + Otsu for both
def otsu_threshold(values):
    v = np.array(values); return 0.0 if len(v) < 3 else float(np.median(v) + 0.5 * np.std(v))

encoders = {"DINOv2-base (768D)": d2b_all, "DINOv2-large (1024D)": img2m_cut}
all_results = []
for name, F in encoders.items():
    F = np.ascontiguousarray(F); Nf = len(F)
    K0 = int(np.clip(round(0.55 * np.sqrt(Nf)), 32, 160))
    km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=4096, max_iter=30).fit(l2(F))
    labs = km.labels_
    clusters = []
    for k in range(K0):
        mk = (labs == k)
        if mk.sum() < 5: continue
        clusters.append({"t_std": float(np.nanstd(Tvs_cut[mk])),
                         "cov": len(set(Evs_cut[mk].tolist())) / NEPS})
    tau = otsu_threshold([c["cov"] for c in clusters])
    sel = [c for c in clusters if c["cov"] >= tau]
    tstds = np.array([c["t_std"] for c in sel])
    r = {"name": name, "K0": K0, "K_eff": len(sel), "tau": tau,
         "mean_tstd": float(np.mean(tstds)),
         "mean_cov": float(np.mean([c["cov"] for c in sel]))}
    all_results.append(r)
    print(f"  {name}: K={K0}→{len(sel)} τ={tau:.3f} Tstd={r['mean_tstd']:.4f} cov={r['mean_cov']:.3f}", flush=True)

# Merge with previous 4 encoders
prev = {"DINOv3-H (1280D)": 0.2192, "Wan-VAE (12288D)": 0.2684, "SigLIP2 (1152D)": 0.2805}
for r in all_results: prev[r["name"]] = r["mean_tstd"]
names = list(prev.keys()); cols = plt.cm.viridis(np.linspace(0, .85, len(names)))
fig, ax = plt.subplots(figsize=(13, 4.8))
means = [prev[n] for n in names]
bars = ax.bar(range(len(names)), means, color=cols, alpha=.85)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n[:24] for n in names], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("mean progress std (↓好)"); ax.set_title(f"DINOv2 规模消融 + 全编码器 · Overcluster+Otsu · {NEPS}ep")
ax.grid(alpha=.25, axis="y")
for b, v in zip(bars, means): ax.text(b.get_x() + b.get_width() / 2, v + .002, f"{v:.4f}", ha="center", fontsize=8)
fig.tight_layout(); out = "crave/docs/visualization/encoders/dinov2_scale_ablation.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=120)
print("SAVED", out, flush=True)

print("\n===== DINOv2 规模消融 + 全编码器 =====")
for n in names: print(f"  {n:<28s} {prev[n]:.4f}")
