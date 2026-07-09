#!/usr/bin/env python
"""最终编码器统一对比: DINOv2-L/DINOv3-H/Wan-VAE/SigLIP2 + (DINOv2-g + V-JEPA2 if available).
全部 Overcluster+Otsu 自适应 K + per-cluster progress profile.
"""
import glob, time, os, numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
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

# ---- encoders that need extraction (not cached as shards) ----
def extract_siglip():
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
    feats = []
    for e in samp:
        loc = np.where(E == e)[0]; o = np.argsort(FR[loc]); loc = loc[o]; fr = FR[loc]
        vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
        cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
        imgs = [frames[i] for i in fr if i < len(frames)]
        if len(imgs) < 3: continue
        with torch.no_grad():
            inp = proc(images=list(imgs), return_tensors="pt").to("cuda")
            feats.append(siglip.get_image_features(**inp).cpu().numpy().astype(np.float32))
    del siglip; torch.cuda.empty_cache()
    return l2(np.concatenate(feats))

def try_load_dinov2g():
    """Try loading DINOv2-giant if weights are downloaded."""
    import torch
    from transformers import AutoModel
    cache_dir = "/vePFS/tim/workspace/hf_cache/hub_default"
    snapshots = list(Path(cache_dir).glob("models--facebook--dinov2-giant/snapshots/*"))
    if not snapshots: return None
    path = str(snapshots[0])
    if not (Path(path) / "pytorch_model.bin").exists() and not (Path(path) / "model.safetensors").exists():
        return None
    print("  DINOv2-g weights found, loading...", flush=True)
    model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    from crave.config import resolve_dataset; from crave.data import kai0
    cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
    zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E, FR = zf["E"], zf["FR"]
    ep_list = sorted(set(E.tolist())); samp = sorted(rng.choice(ep_list, NEPS, replace=False))
    def crop224(rgb):
        h, w = rgb.shape[:2]; s = 224 / min(h, w)
        r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
        hh, ww = r.shape[:2]; return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])
    feats = []
    for e in samp:
        loc = np.where(E == e)[0]; o = np.argsort(FR[loc]); loc = loc[o]; fr = FR[loc]
        vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
        cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
        imgs = [frames[i] for i in fr if i < len(frames)]
        if len(imgs) < 3: continue
        with torch.no_grad():
            hs = model(**model.vision_processor(images=imgs, return_tensors="pt").to("cuda")).last_hidden_state
            feats.append(hs[:, 1:].mean(1).cpu().numpy().astype(np.float32))  # skip CLS, mean pool
    del model; torch.cuda.empty_cache()
    return l2(np.concatenate(feats))

# ---- load cached features ----
print("loading DINOv2/DINOv3/Wan shards...", flush=True)
E2, FR2, T2, d2, v2 = load_shards(REPO / "temp/crave_full/index_dino.npz", REPO / "temp/crave_full/dino", 1024)
E3, FR3, T3, d3, v3 = load_shards(REPO / "temp/crave_full_dinov3h/index.npz", REPO / "temp/crave_full_dinov3h", 1280)
EW, FRW, TW, dW, vW = load_shards(REPO / "temp/crave_full/index_wan.npz", REPO / "temp/crave_full/wan", 12288)
vi = np.where(v2 & v3 & vW)[0]; Ev, Tv = E2[vi], T2[vi]
img2 = l2(d2[vi].astype(np.float32)); img3 = l2(d3[vi].astype(np.float32)); imgW = l2(dW[vi].astype(np.float32))
samp = sorted(rng.choice(sorted(set(Ev.tolist())), NEPS, replace=False))
m = np.isin(Ev, samp); Evs, Tvs = Ev[m], Tv[m]

encoders = {
    "DINOv2-L (1024D)": img2[m],
    "DINOv3-H (1280D)": img3[m],
    "Wan-VAE (12288D)": imgW[m],
}

# SigLIP
print("extracting SigLIP2...", flush=True)
encoders["SigLIP2 (1152D)"] = extract_siglip()

# DINOv2-g (if available)
print("checking DINOv2-g...", flush=True)
d2g = try_load_dinov2g()
if d2g is not None:
    # align: d2g uses same samp as others but extracted fresh — ensure frame count matches
    n_exp = len(encoders["DINOv2-L (1024D)"])
    if abs(len(d2g) - n_exp) < n_exp * 0.1:  # within 10%
        encoders["DINOv2-g (1536D)"] = d2g[:n_exp] if len(d2g) >= n_exp else np.pad(d2g, ((0, n_exp - len(d2g)), (0, 0)))
    else:
        print(f"  WARN: DINOv2-g frame mismatch ({len(d2g)} vs {n_exp}), skipping", flush=True)

# ---- Overcluster + Otsu per encoder ----
def otsu_threshold(values):
    v = np.array(values); return 0.0 if len(v) < 3 else float(np.median(v) + 0.5 * np.std(v))

results = []
for name, F in encoders.items():
    F = np.ascontiguousarray(F)
    N = len(F); K0 = int(np.clip(round(0.55 * np.sqrt(N)), 32, 160))
    t0 = time.time()
    km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=4096, max_iter=30).fit(l2(F))
    labs = km.labels_
    clusters = []
    for k in range(K0):
        mk = (labs == k)
        if mk.sum() < 5: continue
        t_std = float(np.nanstd(Tvs[mk])); mean_t = float(Tvs[mk].mean())
        cov = len(set(Evs[mk].tolist())) / NEPS
        clusters.append({"k": k, "cov": cov, "t_std": t_std, "mean_t": mean_t, "size": mk.sum()})
    tau = otsu_threshold([c["cov"] for c in clusters])
    sel = [c for c in clusters if c["cov"] >= tau]
    elapsed = time.time() - t0
    tstds = np.array([c["t_std"] for c in sel])
    covs = np.array([c["cov"] for c in sel])
    mean_ts = np.array([c["mean_t"] for c in sel])
    # gap uniformity
    sorted_mts = np.sort(mean_ts); gaps = np.diff(sorted_mts)
    gap_cv = float(np.std(gaps) / (np.mean(gaps) + 1e-6))
    r = {"name": name, "K0": K0, "K_eff": len(sel), "tau": tau,
         "mean_tstd": float(np.mean(tstds)), "median_tstd": float(np.median(tstds)),
         "mean_cov": float(np.mean(covs)), "gap_cv": gap_cv,
         "sel": sel, "all": clusters, "time": elapsed}
    results.append(r)
    print(f"  {name}: K₀={K0}→{len(sel)} τ={tau:.3f} Tstd={r['mean_tstd']:.4f} cov={r['mean_cov']:.3f} gapCV={gap_cv:.3f} ({elapsed:.0f}s)", flush=True)

# ---- Figure ----
fig, axs = plt.subplots(2, 2, figsize=(17, 11)); names = [r["name"] for r in results]; cols = plt.cm.tab10(np.linspace(0, 1, len(names)))

# 1. Bar: mean T-std
ax = axs[0, 0]
bars = ax.bar(range(len(names)), [r["mean_tstd"] for r in results], color=cols, alpha=.85)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n[:20] for n in names], rotation=15, ha="right", fontsize=8)
ax.set_ylabel("mean T-std (↓好)"); ax.set_title("时间纯度"); ax.grid(alpha=.25, axis="y")
for b, v in zip(bars, [r["mean_tstd"] for r in results]):
    ax.text(b.get_x() + b.get_width() / 2, v + .002, f"{v:.4f}", ha="center", fontsize=7)

# 2. Scatter: per-cluster mean_T vs std_T
ax = axs[0, 1]
for i, r in enumerate(results):
    sel = r["sel"]
    xs = [c["mean_t"] for c in sel]; ys = [c["t_std"] for c in sel]
    sz = [c["cov"] * 300 for c in sel]
    ax.scatter(xs, ys, s=sz, c=[cols[i]], alpha=.7, edgecolors="k", linewidth=.3, label=f"{r['name'][:18]} (K={len(sel)})")
ax.set_xlabel("cluster mean progress T"); ax.set_ylabel("cluster temporal std (↓紧)")
ax.set_title("Per-cluster progress profile"); ax.legend(fontsize=6); ax.grid(alpha=.25); ax.set_xlim(-.02, 1.02)

# 3. K_eff vs Tstd
ax = axs[1, 0]
for i, r in enumerate(results):
    ax.scatter(r["K_eff"], r["mean_tstd"], s=200, c=[cols[i]], edgecolors="k", linewidth=.5, zorder=3)
    ax.annotate(r["name"][:18], (r["K_eff"], r["mean_tstd"]), fontsize=7, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("effective K (Otsu)"); ax.set_ylabel("mean T-std"); ax.set_title("自适应K vs 时间纯度 (理想: 右下角)"); ax.grid(alpha=.25)

# 4. Coverage vs gap uniformity
ax = axs[1, 1]
for i, r in enumerate(results):
    ax.scatter(r["mean_cov"], r["gap_cv"], s=150, c=[cols[i]], edgecolors="k", linewidth=.5, alpha=.8)
    ax.annotate(r["name"][:18], (r["mean_cov"], r["gap_cv"]), fontsize=7, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("mean coverage (↑全民)"); ax.set_ylabel("gap CV (↓均匀)"); ax.set_title("覆盖率 vs milestone均匀度 (理想: 右下角)"); ax.grid(alpha=.25)

fig.suptitle(f"最终编码器统一对比 · Overcluster+Otsu(自适应K) · {NEPS} ep · {len(encoders)} encoders", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .94])
out = "crave/docs/visualization/encoders/encoder_final_compare.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=130)
print("SAVED", out, flush=True)

# Summary table
print(f"\n{'Encoder':<25s} {'K₀':>4s} {'K_eff':>6s} {'τ':>6s} {'T-std':>8s} {'cov':>6s} {'gapCV':>6s} {'time':>6s}", flush=True)
for r in results:
    print(f"{r['name']:<25s} {r['K0']:>4d} {r['K_eff']:>6d} {r['tau']:>6.3f} {r['mean_tstd']:>8.4f} {r['mean_cov']:>6.3f} {r['gap_cv']:>6.3f} {r['time']:>5.0f}s", flush=True)
