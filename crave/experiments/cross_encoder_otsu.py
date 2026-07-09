#!/usr/bin/env python
"""在 Overcluster+Otsu (自适应K) 下重测跨编码器聚类时间纯度。
编码器: DINOv2-L / DINOv3-H / Wan-VAE / SigLIP2-SO400M (pi05 weights)
全部使用 K₀=0.55√N overcluster → Otsu coverage filter → 报告自适应K + T-std。
"""
import glob, time, numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0"); NEPS = 100; rng = np.random.RandomState(42)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def load(idx_path, shard_dir, dim):
    zf = np.load(idx_path); E, FR, T = zf["E"], zf["FR"], zf["T"]; N = int(zf["n"])
    feat = np.zeros((N, dim), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(shard_dir / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    return E, FR, T, feat, valid

def load_siglip():
    """Load SigLIP2 from pi05 checkpoint + cached HF model."""
    import torch, safetensors.torch, av, cv2
    from transformers import AutoModel, AutoImageProcessor
    # Load standard SigLIP2 architecture
    siglip = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384", torch_dtype=torch.float16,
                                        cache_dir="/vePFS/tim/workspace/hf_cache/hub_default")
    proc = AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384",
                                               cache_dir="/vePFS/tim/workspace/hf_cache/hub_default")
    # Map pi05 vision_tower weights
    CKPT = str(REPO / "kai0/checkpoints/pi05_base/pytorch/model.safetensors")
    PREFIX = "paligemma_with_expert.paligemma.model.vision_tower."
    state = siglip.state_dict(); mapped = 0
    with safetensors.safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if "vision_tower" not in k: continue
            local = k.replace(PREFIX, "")
            if local in state and f.get_tensor(k).shape == state[local].shape:
                state[local] = f.get_tensor(k).to(torch.float16); mapped += 1
    siglip.load_state_dict(state); siglip.cuda().eval()
    print(f"  SigLIP2 mapped {mapped} vision_tower keys from pi05", flush=True)

    # Extract features for sampled eps at 3Hz
    from crave.config import resolve_dataset; from crave.data import kai0
    cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
    zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E, FR = zf["E"], zf["FR"]
    ep_list = sorted(set(E.tolist())); samp = sorted(rng.choice(ep_list, NEPS, replace=False))
    def crop224(rgb):
        h, w = rgb.shape[:2]; s = 224 / min(h, w)
        r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
        hh, ww = r.shape[:2]; return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])
    feats, T_all, Ev_all = [], [], []
    for ei, e in enumerate(samp):
        loc = np.where(E == e)[0]; o = np.argsort(FR[loc]); loc = loc[o]; fr = FR[loc]
        vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
        cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
        imgs = [frames[i] for i in fr if i < len(frames)]
        if len(imgs) < 3: continue
        with torch.no_grad():
            inp = proc(images=list(imgs), return_tensors="pt").to("cuda")
            vf = siglip.get_image_features(**inp).cpu().numpy().astype(np.float32)
            feats.append(vf); T_all.append(zf["T"][loc][:len(vf)]); Ev_all.append(E[loc][:len(vf)])
    del siglip; torch.cuda.empty_cache()
    return np.concatenate(feats), np.concatenate(T_all), np.concatenate(Ev_all), samp

def otsu_threshold(values):
    """Simple Otsu: median + 0.5*std threshold."""
    v = np.array(values)
    if len(v) < 3: return 0.0
    return float(np.median(v) + 0.5 * np.std(v))

def adaptive_cluster(F, Evs, Tvs, name):
    """Overcluster + Otsu → adaptive K metrics."""
    N = len(F)
    K0 = int(np.clip(round(0.55 * np.sqrt(N)), 32, 160))
    t0 = time.time()
    km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=4096, max_iter=30).fit(F)
    labs = km.labels_
    # Per-cluster coverage
    clusters = []
    for k in range(K0):
        mk = (labs == k)
        if mk.sum() < 5: continue
        cov = len(set(Evs[mk].tolist())) / NEPS
        t_std = float(np.nanstd(Tvs[mk]))
        clusters.append({"k": k, "cov": cov, "t_std": t_std, "size": mk.sum()})
    if not clusters: return None
    covs = np.array([c["cov"] for c in clusters])
    tau = otsu_threshold(covs)
    sel = [c for c in clusters if c["cov"] >= tau]
    K_eff = len(sel)
    tstds = np.array([c["t_std"] for c in sel])
    elapsed = time.time() - t0
    return {"name": name, "K0": K0, "K_eff": K_eff, "tau": tau,
            "mean_tstd": float(np.mean(tstds)), "median_tstd": float(np.median(tstds)),
            "mean_cov": float(np.mean([c["cov"] for c in sel])),
            "time": elapsed}

# ---- Load aligned features ----
E2, FR2, T2, d2, v2 = load(REPO / "temp/crave_full/index_dino.npz", REPO / "temp/crave_full/dino", 1024)
E3, FR3, T3, d3, v3 = load(REPO / "temp/crave_full_dinov3h/index.npz", REPO / "temp/crave_full_dinov3h", 1280)
EW, FRW, TW, dW, vW = load(REPO / "temp/crave_full/index_wan.npz", REPO / "temp/crave_full/wan", 12288)
vi = np.where(v2 & v3 & vW)[0]
Ev, Tv = E2[vi], T2[vi]
img2 = l2(d2[vi].astype(np.float32)); img3 = l2(d3[vi].astype(np.float32)); imgW = l2(dW[vi].astype(np.float32))
samp = sorted(rng.choice(sorted(set(Ev.tolist())), NEPS, replace=False))
m = np.isin(Ev, samp)
Evs, Tvs = Ev[m], Tv[m]

print(f"{len(img2[m])} frames, {NEPS} eps", flush=True)
print("extracting SigLIP2...", flush=True)
sigF, sigT, sigE, _ = load_siglip()

encoders = {
    "DINOv2-L(1024D)": img2[m],
    "DINOv3-H(1280D)": img3[m],
    "Wan-VAE(12288D)": imgW[m],
    "SigLIP2-SO400M(1152D)": sigF,
}
Ev_enc = {"DINOv2-L(1024D)": Evs, "DINOv3-H(1280D)": Evs, "Wan-VAE(12288D)": Evs,
          "SigLIP2-SO400M(1152D)": sigE}
Tv_enc = {"DINOv2-L(1024D)": Tvs, "DINOv3-H(1280D)": Tvs, "Wan-VAE(12288D)": Tvs,
          "SigLIP2-SO400M(1152D)": sigT}

results = []
for name, F in encoders.items():
    F = np.ascontiguousarray(F)
    r = adaptive_cluster(F, Ev_enc[name], Tv_enc[name], name)
    if r:
        print(f"  {r['name']}: K₀={r['K0']}→K_eff={r['K_eff']} τ={r['tau']:.3f} mean_Tstd={r['mean_tstd']:.4f} cov={r['mean_cov']:.3f} ({r['time']:.0f}s)", flush=True)
        results.append(r)

# ---- Figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
names = [r["name"] for r in results]
cols = plt.cm.tab10(np.linspace(0, 1, len(names)))

# Bar: mean T-std
means_t = [r["mean_tstd"] for r in results]
bars = ax1.bar(range(len(names)), means_t, color=cols, alpha=.85)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels([n[:22] for n in names], rotation=15, ha="right", fontsize=9)
ax1.set_ylabel("mean per-cluster T-std (↓好)")
ax1.set_title(f"Overcluster+Otsu(自适应K) · {NEPS} ep · 时间纯度")
ax1.grid(alpha=.25, axis="y")
for b, v in zip(bars, means_t):
    ax1.text(b.get_x() + b.get_width() / 2, v + .002, f"{v:.4f}", ha="center", fontsize=8)

# Scatter: K_eff vs T-std (bubble size = mean_cov)
for i, r in enumerate(results):
    ax2.scatter(r["K_eff"], r["mean_tstd"], s=r["mean_cov"] * 800, c=[cols[i]],
                edgecolors="k", linewidth=.5, alpha=.8)
    ax2.annotate(r["name"][:18], (r["K_eff"], r["mean_tstd"]), fontsize=8, xytext=(5, 5),
                 textcoords="offset points")
ax2.set_xlabel("effective K (Otsu 自适应)"); ax2.set_ylabel("mean T-std"); ax2.set_title("K 自适应 vs 时间纯度 (泡大小=coverage, 理想:右下角)")
ax2.grid(alpha=.25)

fig.suptitle("跨编码器聚类时间纯度 · Overcluster+Otsu 自适应K · 不是固定 K=48", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .94])
out = "crave/docs/visualization/encoders/cross_encoder_otsu.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=120)
print("SAVED", out, flush=True)

# Summary table
print("\n===== Overcluster+Otsu 自适应K 跨编码器对比 =====", flush=True)
print(f"{'Encoder':<28s} {'K₀':>4s} {'K_eff':>6s} {'τ':>6s} {'T-std':>8s} {'cov':>6s}", flush=True)
for r in results:
    print(f"{r['name']:<28s} {r['K0']:>4d} {r['K_eff']:>6d} {r['tau']:>6.3f} {r['mean_tstd']:>8.4f} {r['mean_cov']:>6.3f}", flush=True)
