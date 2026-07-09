#!/usr/bin/env python
"""per-cluster progress profile: 每个 cluster 的 mean(T) + std(T), 跨编码器对比。
指标: 时间位置(mean T) vs 时间凝聚度(std T),按 encoder 着色。
低 std + 均匀 mean 分布 = 好的 milestone 集合。
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
    import torch, safetensors.torch, av, cv2
    from transformers import AutoModel, AutoImageProcessor
    siglip = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384", torch_dtype=torch.float16,
                                        cache_dir="/vePFS/tim/workspace/hf_cache/hub_default")
    proc = AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384",
                                               cache_dir="/vePFS/tim/workspace/hf_cache/hub_default")
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
            vf = siglip.get_image_features(**inp).cpu().numpy().astype(np.float32)
            feats.append(vf); T_all.append(zf["T"][loc][:len(vf)]); Ev_sig.append(E[loc][:len(vf)])
    del siglip; torch.cuda.empty_cache()
    return np.concatenate(feats), np.concatenate(T_all), np.concatenate(Ev_sig)

# ---- load ----
E2, FR2, T2, d2, v2 = load(REPO / "temp/crave_full/index_dino.npz", REPO / "temp/crave_full/dino", 1024)
E3, FR3, T3, d3, v3 = load(REPO / "temp/crave_full_dinov3h/index.npz", REPO / "temp/crave_full_dinov3h", 1280)
EW, FRW, TW, dW, vW = load(REPO / "temp/crave_full/index_wan.npz", REPO / "temp/crave_full/wan", 12288)
vi = np.where(v2 & v3 & vW)[0]; Ev, Tv = E2[vi], T2[vi]
img2 = l2(d2[vi].astype(np.float32)); img3 = l2(d3[vi].astype(np.float32)); imgW = l2(dW[vi].astype(np.float32))
samp = sorted(rng.choice(sorted(set(Ev.tolist())), NEPS, replace=False))
m = np.isin(Ev, samp); Evs, Tvs = Ev[m], Tv[m]
print("extracting SigLIP2...", flush=True)
sigF, sigT, sigEv = load_siglip()

encoders = {
    "DINOv2-L": img2[m],
    "DINOv3-H": img3[m],
    "Wan-VAE": imgW[m],
    "SigLIP2": sigF,
}
T_enc = {"DINOv2-L": Tvs, "DINOv3-H": Tvs, "Wan-VAE": Tvs, "SigLIP2": sigT}
Ev_enc = {"DINOv2-L": Evs, "DINOv3-H": Evs, "Wan-VAE": Evs, "SigLIP2": sigEv}

# ---- Overcluster + Otsu per encoder, collect per-cluster (mean_T, std_T, cov) ----
def otsu_threshold(values):
    v = np.array(values)
    if len(v) < 3: return 0.0
    return float(np.median(v) + 0.5 * np.std(v))

profiles = {}
for name, F in encoders.items():
    F = np.ascontiguousarray(F); Tv_e = T_enc[name]
    N = len(F); K0 = int(np.clip(round(0.55 * np.sqrt(N)), 32, 160))
    km = MiniBatchKMeans(K0, n_init=3, random_state=0, batch_size=4096, max_iter=30).fit(l2(F))
    labs = km.labels_
    clusters = []
    for k in range(K0):
        mk = (labs == k)
        if mk.sum() < 5: continue
        mean_t = float(Tv_e[mk].mean())
        std_t = float(Tv_e[mk].std())
        ev_arr = Ev_enc[name]
        cov = len(set(ev_arr[mk].tolist())) / NEPS
        clusters.append({"mean_t": mean_t, "std_t": std_t, "cov": cov, "size": mk.sum()})
    tau = otsu_threshold([c["cov"] for c in clusters])
    sel = [c for c in clusters if c["cov"] >= tau]
    # all clusters (pre-filter) for full progress profile
    profiles[name] = {"all": clusters, "selected": sel, "tau": tau, "K0": K0, "K_eff": len(sel)}
    print(f"{name}: K₀={K0}→K_eff={len(sel)} τ={tau:.3f}", flush=True)

# ---- Figure: per-cluster progress profile (all clusters vs selected) ----
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
names_e = list(encoders.keys()); cols_e = {"DINOv2-L": "#1f77b4", "DINOv3-H": "#2ca02c", "Wan-VAE": "#ff7f0e", "SigLIP2": "#d62728"}

# Panel 1: Selected clusters — mean_T vs std_T
ax = axs[0, 0]
for name in names_e:
    sel = profiles[name]["selected"]
    if not sel: continue
    xs = [c["mean_t"] for c in sel]; ys = [c["std_t"] for c in sel]
    sizes = [c["cov"] * 400 for c in sel]
    ax.scatter(xs, ys, s=sizes, c=cols_e[name], alpha=.7, edgecolors="k", linewidth=.3, label=f"{name} (K={len(sel)})")
ax.set_xlabel("cluster mean progress T"); ax.set_ylabel("cluster temporal std (↓紧)")
ax.set_title("Selected milestones: 时间位置 vs 时间凝聚度")
ax.legend(fontsize=7); ax.grid(alpha=.25); ax.set_xlim(-.02, 1.02)

# Panel 2: All overclustered clusters (grey=filtered, colored=selected)
ax = axs[0, 1]
for name in names_e:
    p = profiles[name]; all_c = p["all"]; sel_c = p["selected"]
    # filtered
    filt = [c for c in all_c if c not in sel_c]
    if filt:
        xs = [c["mean_t"] for c in filt]; ys = [c["std_t"] for c in filt]
        ax.scatter(xs, ys, s=15, c="grey", alpha=.25, edgecolors="none")
    # selected
    if sel_c:
        xs = [c["mean_t"] for c in sel_c]; ys = [c["std_t"] for c in sel_c]
        ax.scatter(xs, ys, s=50, c=cols_e[name], alpha=.8, edgecolors="k", linewidth=.3, label=name)
ax.axhline(y=0.3, color="red", ls="--", lw=.8, alpha=.4, label="std=0.3 参考线")
ax.set_xlabel("cluster mean progress T"); ax.set_ylabel("cluster temporal std")
ax.set_title(f"Overcluster(K₀={profiles['DINOv2-L']['K0']})→Otsu filter: 灰=被过滤, 彩=保留")
ax.legend(fontsize=7); ax.grid(alpha=.25); ax.set_xlim(-.02, 1.02)

# Panel 3: Progress histogram — distribution of selected clusters' mean_T
ax = axs[1, 0]
for name in names_e:
    sel = profiles[name]["selected"]
    if not sel:
        ax.text(0.5, 0.5, f"{name}: no clusters", transform=ax.transAxes, ha="center")
        continue
    ts = [c["mean_t"] for c in sel]
    ax.hist(ts, bins=15, alpha=.4, color=cols_e[name], label=name)
ax.set_xlabel("cluster mean progress T"); ax.set_ylabel("count")
ax.set_title("Milestone 时间分布 (均匀=好, 偏斜=差)")
ax.legend(fontsize=7); ax.grid(alpha=.25)

# Panel 4: std_T vs coverage (selected clusters)
ax = axs[1, 1]
for name in names_e:
    sel = profiles[name]["selected"]
    if not sel: continue
    xs = [c["cov"] for c in sel]; ys = [c["std_t"] for c in sel]
    ax.scatter(xs, ys, s=50, c=cols_e[name], alpha=.7, edgecolors="k", linewidth=.3, label=name)
ax.set_xlabel("cross-episode coverage"); ax.set_ylabel("cluster temporal std (↓紧)")
ax.set_title("Coverage vs 时间凝聚度 (理想: 右上角=高覆盖+紧)")
ax.legend(fontsize=7); ax.grid(alpha=.25)

fig.suptitle("Per-cluster Progress Profile: mean(T) + std(T) + coverage · 跨编码器 · Overcluster+Otsu", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .95])
out = "crave/docs/visualization/encoders/cluster_progress_profile.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=130)
print("SAVED", out, flush=True)

# Summary table
print("\n===== per-encoder progress profile =====")
for name in names_e:
    p = profiles[name]; sel = p["selected"]
    if not sel:
        print(f"{name}: NO selected clusters")
        continue
    means = np.array([c["mean_t"] for c in sel]); stds = np.array([c["std_t"] for c in sel])
    covs = np.array([c["cov"] for c in sel])
    # uniformity of coverage across progress: Gini of mean_t spacing
    sorted_means = np.sort(means); gaps = np.diff(sorted_means)
    gap_cv = float(np.std(gaps) / (np.mean(gaps) + 1e-6))  # lower = more uniform
    print(f"{name:15s} K={len(sel):2d}  mean_T∈[{means.min():.2f},{means.max():.2f}]  mean_std={stds.mean():.3f}  gap_CV={gap_cv:.3f}  mean_cov={covs.mean():.3f}",
          flush=True)
