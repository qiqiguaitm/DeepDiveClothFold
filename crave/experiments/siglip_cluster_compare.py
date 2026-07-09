#!/usr/bin/env python
"""SigLIP2 双版本聚类:vision-only(1152D) + text-projected(6D),用 pi05 checkpoint 的 vision_tower 权重."""
import torch, numpy as np, safetensors.torch, av, cv2, time, glob
from pathlib import Path
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from sklearn.cluster import MiniBatchKMeans
from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path("/home/tim/workspace/deepdive_kai0"); K = 48; NEPS = 200; rng = np.random.RandomState(42)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])

zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E, FR, T = zf["E"], zf["FR"], zf["T"]
ep_list = sorted(set(E.tolist())); samp_eps = sorted(rng.choice(ep_list, NEPS, replace=False))

print("loading SigLIP2 from pi05...", flush=True)
siglip = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384", torch_dtype=torch.float16,
                                    cache_dir="/vePFS/tim/workspace/hf_cache/hub_default")
proc = AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384",
                                           cache_dir="/vePFS/tim/workspace/hf_cache/hub_default")
tok = AutoTokenizer.from_pretrained("google/siglip2-so400m-patch14-384",
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
print(f"mapped {mapped} vision_tower keys", flush=True)

# Text prototypes
text_prompts = [
    "a photo of cloth spread flat on table, task just started",
    "a photo of robot hand approaching the cloth",
    "a photo of robot grasping the cloth",
    "a photo of robot lifting the cloth up",
    "a photo of robot folding the cloth",
    "a photo of robot placing cloth, task completed",
]
with torch.no_grad():
    ti = tok(text_prompts, padding=True, return_tensors="pt").to("cuda")
    text_feat = siglip.get_text_features(**ti)
    text_feat = l2(text_feat.cpu().numpy().astype(np.float32))

from crave.config import resolve_dataset; from crave.data import kai0
cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
v_feats, t_feats, Tv_all, Ev_all = [], [], [], []
t0 = time.time()
for ei, e in enumerate(samp_eps):
    loc = np.where(E == e)[0]; o = np.argsort(FR[loc]); loc = loc[o]; fr = FR[loc]
    vid = DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"
    cap = av.open(str(vid)); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
    imgs = [frames[i] for i in fr if i < len(frames)]
    if len(imgs) < 3: continue
    with torch.no_grad():
        inp = proc(images=list(imgs), return_tensors="pt").to("cuda")
        vf = siglip.get_image_features(**inp).cpu().numpy().astype(np.float32)
        vf_n = l2(vf); tf_proj = vf_n @ text_feat.T  # (N, 6) cosine sim to 6 text prototypes
        v_feats.append(vf); t_feats.append(tf_proj)
        Tv_all.append(T[loc][:len(vf)]); Ev_all.append(E[loc][:len(vf)])
    if (ei + 1) % 30 == 0: print(f"  {ei + 1}/{len(samp_eps)} ({(time.time() - t0) / 60:.1f}min)", flush=True)
del siglip; torch.cuda.empty_cache()

v_all = np.concatenate(v_feats); t_all = np.concatenate(t_feats)
Tv = np.concatenate(Tv_all); Ev = np.concatenate(Ev_all)

variants = {"SigLIP-vision(1152D)": v_all, "SigLIP-text-proj(6D)": t_all}
results = {}
for name, F in variants.items():
    t0 = time.time()
    km = MiniBatchKMeans(K, n_init=3, random_state=0, batch_size=2048, max_iter=30).fit(l2(F))
    labs = km.labels_
    t_std = [float(np.nanstd(Tv[labs == k])) if (labs == k).sum() > 4 else np.nan for k in range(K)]
    t_std = [x for x in t_std if not np.isnan(x)]
    cov = np.mean([len(set(Ev[labs == k].tolist())) / NEPS for k in range(K) if (labs == k).sum() > 4])
    results[name] = {"mean_std": np.mean(t_std), "median": np.median(t_std), "cov": cov}
    print(f"  {name}: mean_Tstd={np.mean(t_std):.4f} median={np.median(t_std):.4f} cov={cov:.3f} ({(time.time()-t0):.0f}s)", flush=True)

all_r = {"DINOv2-L(1024D)": 0.2063, "DINOv3-H(1280D)": 0.2095, "Wan-VAE(12288D)": 0.2456}
all_r.update({k: v["mean_std"] for k, v in results.items()})
names = list(all_r.keys())
fig, ax = plt.subplots(figsize=(14, 5))
means = [all_r[n] for n in names]
colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
bars = ax.bar(range(len(names)), means, color=colors[:len(names)], alpha=.85)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n[:28] for n in names], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("mean per-cluster T-std (↓好)"); ax.set_title(f"跨编码器聚类时间纯度 · K={K} · {NEPS}ep · SigLIP双版本")
ax.grid(alpha=.25, axis="y")
for b, v in zip(bars, means): ax.text(b.get_x() + b.get_width() / 2, v + .002, f"{v:.4f}", ha="center", fontsize=8)
fig.tight_layout(); out = "crave/docs/visualization/encoders/cross_encoder_all5.png"
Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=120); print("SAVED", out)
