#!/usr/bin/env python
"""PCA 降维消融: 所有编码器统一降维到相同 dim, Overcluster+Otsu 聚类, 对比 T-std。
编码器: DINOv2-base/L, DINOv3-H, Wan-VAE, SigLIP2
目标维度: 32, 64, 128, 256, 512, 768 + raw
"""
import glob, time, numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0"); NEPS = 100; rng = np.random.RandomState(42)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def load_shards(idx_path, shard_dir, dim):
    zf = np.load(idx_path); E, FR, T = zf["E"], zf["FR"], zf["T"]; N = int(zf["n"])
    feat = np.zeros((N, dim), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(shard_dir / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    return E, FR, T, feat, valid

def load_siglip():
    import torch, safetensors.torch, av, cv2
    from transformers import AutoModel, AutoImageProcessor
    siglip = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384", torch_dtype=torch.float16,
                                        cache_dir="/vePFS/tim/workspace/hf_cache/hub_default")
    proc = AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384",
                                               cache_dir="/vePFS/tim/workspace/hf_cache/hub_default")
    CKPT = str(REPO / "kai0/checkpoints/pi05_base/pytorch/model.safetensors")
    PREFIX = "paligemma_with_expert.paligemma.model.vision_tower."
    state = siglip.state_dict()
    with safetensors.safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if "vision_tower" not in k: continue
            local = k.replace(PREFIX, "")
            if local in state and f.get_tensor(k).shape == state[local].shape:
                state[local] = f.get_tensor(k).to(torch.float16)
    siglip.load_state_dict(state); siglip.cuda().eval()
    from crave.config import resolve_dataset; from crave.data import kai0
    cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
    zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E, FR = zf["E"], zf["FR"]
    ep_list = sorted(set(E.tolist())); samp = sorted(rng.choice(ep_list, NEPS, replace=False))
    def crop224(rgb):
        h, w = rgb.shape[:2]; s = 224 / min(h, w)
        r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
        hh, ww = r.shape[:2]; return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])
    feats, T_all, Ev_sig = [], [], []
    for e in samp:
        loc = np.where(E == e)[0]; o = np.argsort(FR[loc]); loc = loc[o]; fr = FR[loc]
        vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
        cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
        imgs = [frames[i] for i in fr if i < len(frames)]
        if len(imgs) < 3: continue
        with torch.no_grad():
            inp = proc(images=list(imgs), return_tensors="pt").to("cuda")
            feats.append(siglip.get_image_features(**inp).cpu().numpy().astype(np.float32))
            T_all.append(zf["T"][loc][:len(feats[-1])]); Ev_sig.append(E[loc][:len(feats[-1])])
    del siglip; torch.cuda.empty_cache()
    return np.concatenate(feats), np.concatenate(T_all), np.concatenate(Ev_sig)

def extract_dinov2_base():
    import torch, av, cv2
    from transformers import AutoModel
    d2b = AutoModel.from_pretrained("/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-base",
                                     torch_dtype=torch.float16).cuda().eval()
    from crave.config import resolve_dataset; from crave.data import kai0
    cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
    zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E, FR = zf["E"], zf["FR"]
    ep_list = sorted(set(E.tolist())); samp = sorted(rng.choice(ep_list, NEPS, replace=False))
    def crop224(rgb):
        h, w = rgb.shape[:2]; s = 224 / min(h, w)
        r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
        hh, ww = r.shape[:2]; return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])
    feats, T_all, Ev_all = [], [], []
    for e in samp:
        loc = np.where(E == e)[0]; o = np.argsort(FR[loc]); loc = loc[o]; fr = FR[loc]
        vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
        cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
        imgs = [frames[i] for i in fr if i < len(frames)]
        if len(imgs) < 3: continue
        imgs_t = torch.from_numpy(np.stack([cv2.resize(im, (224, 224)).transpose(2, 0, 1) for im in imgs])).float().cuda() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        imgs_t = (imgs_t - mean) / std
        with torch.no_grad():
            out = d2b(imgs_t).last_hidden_state[:, 1:].mean(1).cpu().numpy().astype(np.float32)
            feats.append(out); T_all.append(zf["T"][loc][:len(out)]); Ev_all.append(E[loc][:len(out)])
    del d2b; torch.cuda.empty_cache()
    return np.concatenate(feats), np.concatenate(T_all), np.concatenate(Ev_all)

def otsu_threshold(values):
    v = np.array(values); return 0.0 if len(v) < 3 else float(np.median(v) + 0.5 * np.std(v))

def eval_cluster(F, Ts, Es, name):
    N = len(F); K0 = int(np.clip(round(0.55 * np.sqrt(N)), 32, 160))
    km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=4096, max_iter=30).fit(l2(F))
    labs = km.labels_
    clusters = []
    for k in range(K0):
        mk = labs == k
        if mk.sum() < 5: continue
        clusters.append({"t_std": float(np.nanstd(Ts[mk])), "cov": len(set(Es[mk].tolist())) / NEPS})
    tau = otsu_threshold([c["cov"] for c in clusters])
    sel = [c for c in clusters if c["cov"] >= tau]
    tstds = np.array([c["t_std"] for c in sel])
    return float(np.mean(tstds)), len(sel), tau

# ---- load all features ----
print("loading cached features...", flush=True)
E2, FR2, T2, d2, v2 = load_shards(REPO / "temp/crave_full/index_dino.npz", REPO / "temp/crave_full/dino", 1024)
E3, FR3, T3, d3, v3 = load_shards(REPO / "temp/crave_full_dinov3h/index.npz", REPO / "temp/crave_full_dinov3h", 1280)
EW, FRW, TW, dW, vW = load_shards(REPO / "temp/crave_full/index_wan.npz", REPO / "temp/crave_full/wan", 12288)
vi = np.where(v2 & v3 & vW)[0]; Ev, Tv = E2[vi], T2[vi]
img2 = l2(d2[vi].astype(np.float32)); img3 = l2(d3[vi].astype(np.float32)); imgW = l2(dW[vi].astype(np.float32))
samp = sorted(rng.choice(sorted(set(Ev.tolist())), NEPS, replace=False))
m = np.isin(Ev, samp); Evs, Tvs = Ev[m], Tv[m]

print("extracting SigLIP2...", flush=True)
sigF, sigT, sigEv = load_siglip()
print("extracting DINOv2-base...", flush=True)
d2bF, d2bT, d2bEv = extract_dinov2_base()

# Align all to min length
encoders_raw = {
    "DINOv2-base (768D)": d2bF,
    "DINOv2-large (1024D)": img2[m],
    "DINOv3-H (1280D)": img3[m],
    "Wan-VAE (12288D)": imgW[m],
    "SigLIP2 (1152D)": sigF,
}
T_enc = {"DINOv2-base (768D)": d2bT, "DINOv2-large (1024D)": Tvs,
         "DINOv3-H (1280D)": Tvs, "Wan-VAE (12288D)": Tvs, "SigLIP2 (1152D)": sigT}
E_enc = {"DINOv2-base (768D)": d2bEv, "DINOv2-large (1024D)": Evs,
         "DINOv3-H (1280D)": Evs, "Wan-VAE (12288D)": Evs, "SigLIP2 (1152D)": sigEv}

# PCA sweep for each encoder
TARGET_DIMS = [32, 64, 128, 256, 512, 768]
all_results = {}  # encoder -> [(dim_label, tstd, k_eff)]

for enc_name, F_raw in encoders_raw.items():
    F_raw = np.ascontiguousarray(F_raw)
    Ts = T_enc[enc_name][:len(F_raw)]; Es = E_enc[enc_name][:len(F_raw)]
    raw_dim = F_raw.shape[1]
    # raw baseline
    t0 = time.time()
    tstd, k_eff, tau = eval_cluster(F_raw, Ts, Es, f"{enc_name} raw")
    elapsed = time.time() - t0
    all_results.setdefault(enc_name, []).append((f"raw {raw_dim}D", tstd, k_eff))
    # Only PCA to dims < raw_dim
    valid_dims = [d for d in TARGET_DIMS if d < raw_dim]
    for d in valid_dims:
        t0 = time.time()
        pca = PCA(n_components=d, random_state=0)
        F_pca = l2(pca.fit_transform(F_raw))
        tstd, k_eff, tau = eval_cluster(F_pca, Ts, Es, f"PCA→{d}D")
        elapsed = time.time() - t0
        all_results[enc_name].append((f"PCA→{d}D", tstd, k_eff))
    best = min(all_results[enc_name], key=lambda x: x[1])
    print(f"{enc_name}: best={best[0]} Tstd={best[1]:.4f} K={best[2]}", flush=True)

# ---- Figure: multi-line plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
colors = {"DINOv2-base (768D)": "#1f77b4", "DINOv2-large (1024D)": "#2ca02c",
          "DINOv3-H (1280D)": "#ff7f0e", "Wan-VAE (12288D)": "#d62728",
          "SigLIP2 (1152D)": "#9467bd"}
markers = {"DINOv2-base (768D)": "o", "DINOv2-large (1024D)": "s",
           "DINOv3-H (1280D)": "D", "Wan-VAE (12288D)": "^",
           "SigLIP2 (1152D)": "v"}

for enc_name, rows in all_results.items():
    dims = [int(r[0].split(" ")[-1].replace("D", "")) for r in rows]
    tstds = [r[1] for r in rows]
    ks = [r[2] for r in rows]
    # sort by dim
    order = np.argsort(dims)
    dims = np.array(dims)[order]; tstds = np.array(tstds)[order]; ks = np.array(ks)[order]
    ax1.plot(dims, tstds, "-o", c=colors[enc_name], marker=markers[enc_name],
             lw=2, markersize=8, label=enc_name[:20], alpha=.85)
    # Mark best point
    best_idx = np.argmin(tstds)
    ax1.scatter([dims[best_idx]], [tstds[best_idx]], s=120, c=colors[enc_name],
                edgecolors="k", linewidth=1.5, zorder=5)

ax1.set_xlabel("feature dimension"); ax1.set_ylabel("mean progress std (↓好)")
ax1.set_title(f"PCA 降维消融 — 所有编码器 · {NEPS}ep")
ax1.legend(fontsize=7); ax1.grid(alpha=.25); ax1.invert_xaxis()

# Panel 2: best per encoder (bar)
enc_names = list(all_results.keys())
best_tstds = [min(r[1] for r in all_results[enc]) for enc in enc_names]
best_dims = [min(all_results[enc], key=lambda x: x[1])[0] for enc in enc_names]
bars = ax2.bar(range(len(enc_names)), best_tstds, color=[colors[e] for e in enc_names], alpha=.85)
ax2.set_xticks(range(len(enc_names))); ax2.set_xticklabels([f"{e[:18]}\n{best_dims[i]}" for i, e in enumerate(enc_names)], fontsize=7)
ax2.set_ylabel("best mean progress std"); ax2.set_title("各编码器最优降维后 T-std"); ax2.grid(alpha=.25, axis="y")
for b, v in zip(bars, best_tstds):
    ax2.text(b.get_x() + b.get_width() / 2, v + .002, f"{v:.4f}", ha="center", fontsize=8)

fig.suptitle("PCA 降维全编码器消融: 同一目标维度, Overcluster+Otsu 聚类时间纯度", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .94])
out = "crave/docs/visualization/encoders/pca_all_encoders.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=130)
print("SAVED", out, flush=True)

# Text summary
print("\n===== 全编码器 PCA 消融总结 =====")
print(f"{'Encoder':<25s} {'best_dim':>10s} {'best_Tstd':>10s} {'raw_Tstd':>10s} {'improve':>8s}")
for enc_name in enc_names:
    rows = all_results[enc_name]
    best = min(rows, key=lambda x: x[1])
    raw = rows[0]
    imp = (raw[1] - best[1]) / raw[1] * 100 if raw[1] > 0 else 0
    print(f"{enc_name:<25s} {best[0]:>10s} {best[1]:>10.4f} {raw[1]:>10.4f} {imp:>7.1f}%")
