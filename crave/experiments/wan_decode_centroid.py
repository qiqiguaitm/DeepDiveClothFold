"""用 WanVAEEncoder 加载 Wan2.2 VAE, 给选中簇的最近帧编码→平均→解码。
从原始视频读帧(不走 DINOv3 缓存), 避免兼容问题。"""
import sys, warnings, glob, gc
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crave_full_7b_centroid as C
from crave.encoders import load_encoder
from crave.render import setup_mpl
from sklearn.cluster import MiniBatchKMeans
import torch

DEV = "cuda"
OUTD = C.OUTD
z = np.load(OUTD / "index.npz"); E, FR, T, N = z["E"], z["FR"], z["T"], int(z["n"])
feat = np.zeros((N, C.DIM), np.float16); valid = np.zeros(N, bool)
for f in sorted(glob.glob(str(OUTD / "shard_*.npz"))):
    s = np.load(f); feat[s["gidx"]] = s["feat"]; valid[s["gidx"]] = s["valid"]
vi = np.where(valid)[0]; F = C.L2(feat[vi].astype(np.float32))
Tv, Ev = T[vi], E[vi]; ne = len(set(E.tolist()))

# 找簇 cstar (同配方)
fit = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
km = MiniBatchKMeans(120, random_state=0, batch_size=4096, n_init=3).fit(F[fit])
cen = km.cluster_centers_; lab = km.predict(F)
cov = np.array([len(set(Ev[lab == c].tolist())) / ne for c in range(120)])
tpos = np.array([Tv[lab == c].mean() for c in range(120)])
tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(120)])
cand = [c for c in range(120) if 0.2 <= tpos[c] <= 0.8 and tstd[c] < np.median(tstd[tstd < 9])]
cstar = max(cand, key=lambda c: cov[c]) if cand else int(np.argmax(cov))
print(f"cstar={cstar} cov={cov[cstar]:.1%} P≈{tpos[cstar]:.2f}")

loc = np.where(lab == cstar)[0]; d = np.linalg.norm(F[loc] - cen[cstar], axis=1)
order = loc[np.argsort(d)]
near_global = vi[order[:48]]  # 48 个最近帧的全局索引
near_eps = [int(Ev[li]) for li in near_global]

# 获取真实帧: 用 DINOv3-H 编码器的 pyav decode 抓取 RGB(224 crop)
enc = load_encoder(C.ENC)
_, imgs_raw, ok_raw = C._grids_for(enc, near_global, E, FR)
keep = np.where(ok_raw)[0]
if len(keep) < 8: keep = np.arange(min(8, len(near_global)))
print(f"有效帧: {len(keep)}/{len(near_global)}")
enc.unload(); gc.collect(); torch.cuda.empty_cache()

# resize 224→256 for Wan VAE
import cv2
kept_imgs = []
for k in keep:
    im = imgs_raw[k]
    kept_imgs.append(cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA))

# 过 Wan2.2 VAE
from crave.config.encoders import EncoderSpec
from crave.encoders.wan_vae import WanVAEEncoder
spec = EncoderSpec("wan-vae","wan_vae",str(Path("/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/Wan2.2-TI2V-5B-Diffusers")),48*16*16,"fp16",256,16,0)
wan = WanVAEEncoder(spec, DEV)

# 编码有效帧 → latent (48,16,16) → 平均 → 解码
zs = wan.encode_latents(kept_imgs, bs=16)  # (K,48,16,16)
z_mean = zs.mean(0, keepdims=True)          # (1,48,16,16)
dec = wan.decode(z_mean)[0]                  # (256,256,3)
print(f"Wan2.2 VAE 解码: {dec.shape}")

# 也解码最近一张单帧作为对照
dec_near = wan.decode(zs[:1])[0]

# 保存
from PIL import Image
mat_dir = Path("/vePFS/tim/workspace/deepdive_kai0/crave/docs/visualization/pipeline_material")
Image.fromarray(dec).save(str(mat_dir / "wan_decoded_centroid.png"))
Image.fromarray(dec_near).save(str(mat_dir / "wan_decoded_nearest.png"))
print(f"Saved wan_decoded_centroid.png ({dec.shape})")

# 对比图
plt = setup_mpl()
fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
axes[0].imshow(cv2.resize(kept_imgs[0], (256,256))); axes[0].axis("off"); axes[0].set_title("最近真实帧", fontsize=9)
axes[1].imshow(dec_near); axes[1].axis("off"); axes[1].set_title("单帧 Wan 解码", fontsize=9)
axes[2].imshow(dec); axes[2].axis("off"); axes[2].set_title(f"簇中心 Wan 解码\n({len(keep)}帧平均)", fontsize=9)
fig.suptitle(f"簇 c{cstar} (cov={cov[cstar]:.0%} P={tpos[cstar]:.2f})", fontsize=10)
fig.savefig(str(mat_dir / "wan_decoded_compare.png"), dpi=140, bbox_inches="tight")
print("SAVED wan_decoded_compare.png")
