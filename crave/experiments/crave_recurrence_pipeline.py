"""§2.1 灵感来源 pipeline 示意图:
左1: 6 条不同 episode 距簇中心最近的帧
左2: 每帧的 DINOv3-H 1280D latent (热力条)
中:  簇中心 1280D latent (热力条)
右:  簇中心解码图
展示: 真实帧 → DINOv3 编码 → latent 空间 → 聚类 → 簇中心 → 解码回图
"""
import sys, warnings, argparse, glob, gc
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crave_full_7b_centroid as C
from crave.decoding.decoder import train_dec
from crave.encoders import load_encoder
from crave.render import setup_mpl
from sklearn.cluster import MiniBatchKMeans
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

plt = setup_mpl()

ap = argparse.ArgumentParser()
ap.add_argument("--k0", type=int, default=120)
ap.add_argument("--ncol", type=int, default=6)
ap.add_argument("--nn", type=int, default=24)
ap.add_argument("--save-frames", type=str, default=None,
                help="instead of composite, save 6 raw frames to this dir")
a = ap.parse_args()

if a.save_frames:
    OUTD = C.OUTD
    z = np.load(OUTD / "index.npz"); E, FR, T, N = z["E"], z["FR"], z["T"], int(z["n"])
    feat = np.zeros((N, C.DIM), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / "shard_*.npz"))):
        s = np.load(f); feat[s["gidx"]] = s["feat"]; valid[s["gidx"]] = s["valid"]
    vi = np.where(valid)[0]; F = C.L2(feat[vi].astype(np.float32))
    Tv, Ev = T[vi], E[vi]; ne = len(set(E.tolist()))
    fit = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
    km = MiniBatchKMeans(a.k0, random_state=0, batch_size=4096, n_init=3).fit(F[fit])
    cen = km.cluster_centers_; lab = km.predict(F)
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(a.k0)])
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(a.k0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(a.k0)])
    cand = [c for c in range(a.k0) if 0.2 <= tpos[c] <= 0.8 and tstd[c] < np.median(tstd[tstd < 9])]
    cstar = max(cand, key=lambda c: cov[c]) if cand else int(np.argmax(cov))
    loc = np.where(lab == cstar)[0]; d = np.linalg.norm(F[loc] - cen[cstar], axis=1)
    order = loc[np.argsort(d)]
    seen, picks = set(), []
    for li in order:
        e = int(Ev[li]); seen.add(e); picks.append(li)
        if len(picks) >= a.ncol: break
    near_global = vi[picks]; near_eps = [int(Ev[li]) for li in picks]
    nn_global = vi[order[:a.nn]]
    print(f"选中簇 c={cstar}: cov={cov[cstar]:.1%} progress≈{tpos[cstar]:.2f}")
    print(f"6 episodes: {near_eps}")
    enc = load_encoder(C.ENC)
    _, th_near, ok_near = C._grids_for(enc, near_global, E, FR)
    rs = np.random.RandomState(1); tr = vi[rs.choice(len(vi), min(3000, len(vi)), replace=False)]
    g_tr, im_tr, ok_tr = C._grids_for(enc, tr, E, FR); keep = np.where(ok_tr)[0]
    gg_nn, _, ok_nn = C._grids_for(enc, nn_global, E, FR)
    cen_grid = gg_nn[np.where(ok_nn)[0]].mean(0)
    enc.unload(); gc.collect(); import torch; torch.cuda.empty_cache()
    decode = train_dec(g_tr[keep], im_tr[keep], C.DIM, dec="small", epochs=45)
    dec_cen = decode(cen_grid[None].astype(np.float32))[0]
    out_dir = Path(a.save_frames); out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for j in range(len(near_eps)):
        if ok_near[j]:
            Image.fromarray(th_near[j]).save(str(out_dir / f"ep{near_eps[j]}.png"))
    Image.fromarray(dec_cen).save(str(out_dir / "cluster_center_decoded.png"))
    np.save(str(out_dir / "latents_6frames.npy"), C.L2(feat[near_global[:a.ncol]].astype(np.float32)))
    np.save(str(out_dir / "cluster_center_latent.npy"), cen[cstar])
    print(f"素材已保存到 {out_dir}")
    raise SystemExit(0)

OUTD = C.OUTD
z = np.load(OUTD / "index.npz"); E, FR, T, N = z["E"], z["FR"], z["T"], int(z["n"])
feat = np.zeros((N, C.DIM), np.float16); valid = np.zeros(N, bool)
for f in sorted(glob.glob(str(OUTD / "shard_*.npz"))):
    s = np.load(f); feat[s["gidx"]] = s["feat"]; valid[s["gidx"]] = s["valid"]
vi = np.where(valid)[0]; F = C.L2(feat[vi].astype(np.float32))
Tv, Ev = T[vi], E[vi]; ne = len(set(E.tolist()))

fit = np.random.RandomState(0).choice(len(vi), min(len(vi), 120000), replace=False)
km = MiniBatchKMeans(a.k0, random_state=0, batch_size=4096, n_init=3).fit(F[fit])
cen = km.cluster_centers_; lab = km.predict(F)
cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(a.k0)])
tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(a.k0)])
tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(a.k0)])
cand = [c for c in range(a.k0) if 0.2 <= tpos[c] <= 0.8 and tstd[c] < np.median(tstd[tstd < 9])]
cstar = max(cand, key=lambda c: cov[c]) if cand else int(np.argmax(cov))
print(f"选中簇 c={cstar}: cov={cov[cstar]:.1%} progress≈{tpos[cstar]:.2f} σ={tstd[cstar]:.3f}")

loc = np.where(lab == cstar)[0]; d = np.linalg.norm(F[loc] - cen[cstar], axis=1)
order = loc[np.argsort(d)]
seen, picks = set(), []
for li in order:
    e = int(Ev[li])
    if e in seen: continue
    seen.add(e); picks.append(li)
    if len(picks) >= a.ncol: break
near_global = vi[picks]; near_eps = [int(Ev[li]) for li in picks]
nn_global = vi[order[:a.nn]]

# Encode + decode
enc = load_encoder(C.ENC)
rs = np.random.RandomState(1); tr = vi[rs.choice(len(vi), min(3000, len(vi)), replace=False)]
g_tr, im_tr, ok_tr = C._grids_for(enc, tr, E, FR); keep = np.where(ok_tr)[0]
gg_nn, _, ok_nn = C._grids_for(enc, nn_global, E, FR)
_, th_near, ok_near = C._grids_for(enc, near_global, E, FR)
th_near = [th_near[i] for i in range(len(near_global)) if ok_near[i]]  # only valid
near_eps = [near_eps[i] for i in range(len(near_global)) if ok_near[i]]
ncol = min(len(th_near), a.ncol)
cen_valid = np.where(ok_nn)[0]
if len(cen_valid) < 3: cen_valid = np.where(ok_nn)[0]

# Get latents for the 6 frames: features already in F[near_global]
near_feats = C.L2(feat[near_global[:ncol]].astype(np.float32))  # 1280D latents
cen_feat = cen[cstar]  # cluster center 1280D

cen_grid = gg_nn[cen_valid].mean(0) if len(cen_valid) > 0 else gg_nn[:a.nn].mean(0)
enc.unload(); gc.collect()
import torch; torch.cuda.empty_cache()

from numpy.linalg import norm as L2n
print("Training 16→128 decoder ...")
decode = train_dec(g_tr[keep], im_tr[keep], C.DIM, dec="small", epochs=45)
dec_cen = decode(cen_grid[None].astype(np.float32))[0]
print("Decode done")

# ============ 构图: 左1 | 左2 | 中 | 右 ============
fig = plt.figure(figsize=(16, 4.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 0.6, 0.4, 0.6], wspace=0.15)

# ---- 左1: 6 帧真实图 ----
ax1 = fig.add_subplot(gs[0])
margin = 0.02
for j in range(ncol):
    ny, nx = th_near[j].shape[:2]
    x0 = j / ncol + margin; y0 = 0.05
    w = 1/ncol - 2*margin; h = 0.9
    # Place in a grid within the subplot
    ax_ep = ax1.inset_axes([x0, y0, w, h])
    ax_ep.imshow(th_near[j]); ax_ep.axis("off")
    ax_ep.set_title(f"ep{near_eps[j]}", fontsize=7, pad=1)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
ax1.set_title("不同 episode 距簇中心最近帧", fontsize=9, fontweight="bold")

# ---- 左2: 每帧的 1280D latent 热力条 ----
ax2 = fig.add_subplot(gs[1])
# Create stacked heatmap: each row = one frame's 1280D latent
latent_img = np.stack(near_feats)  # (ncol, 1280)
# Downsample for display: average pool to 256 columns
bs = 5
latent_show = latent_img.reshape(ncol, -1, bs).mean(2)  # (ncol, 256)
im2 = ax2.imshow(latent_show, aspect="auto", cmap="viridis", norm=Normalize(-0.5, 0.5))
ax2.set_yticks(range(ncol)); ax2.set_yticklabels([f"ep{e}" for e in near_eps], fontsize=6)
ax2.set_xticks([]); ax2.set_xlabel("DINOv3-H 1280D 维度 (下采样)", fontsize=7)
ax2.set_title("每帧 latent 特征", fontsize=9, fontweight="bold")

# 箭头: 左1 → 左2
ax1.annotate("", xy=(1.02, 0.5), xytext=(-0.02, 0.5),
             fontsize=16, ha="center", va="center",
             arrowprops=dict(arrowstyle="->", lw=2, color="#7c3aed"),
             xycoords="axes fraction", textcoords="axes fraction")

# ---- 中: 簇中心 latent 热力条 ----
ax3 = fig.add_subplot(gs[2])
cen_show = cen_feat.reshape(-1, bs).mean(1)  # (256,)
ax3.imshow(cen_show[None, :], aspect="auto", cmap="viridis", norm=Normalize(-0.5, 0.5))
ax3.set_yticks([]); ax3.set_xticks([])
ax3.set_title("簇中心 latent", fontsize=9, fontweight="bold", color="#7c3aed")

# 箭头: 左2 → 中
ax2.annotate("", xy=(1.02, 0.5), xytext=(-0.02, 0.5),
             arrowprops=dict(arrowstyle="->", lw=2, color="#7c3aed"),
             xycoords="axes fraction", textcoords="axes fraction")

# ---- 右: 簇中心解码图 ----
ax4 = fig.add_subplot(gs[3])
ax4.imshow(dec_cen); ax4.axis("off")
ax4.set_title(f"簇中心解码\n(覆盖 {cov[cstar]:.0%} ep)", fontsize=9, fontweight="bold", color="#7c3aed")

# 箭头: 中 → 右
ax3.annotate("", xy=(1.02, 0.5), xytext=(-0.02, 0.5),
             arrowprops=dict(arrowstyle="->", lw=2, color="#7c3aed"),
             xycoords="axes fraction", textcoords="axes fraction")

# ---- pipeline 标签 ----
fig.text(0.01, 0.96, "Pipeline:", fontsize=8, color="#555")
fig.text(0.06, 0.96, "真实帧 → DINOv3 编码 → 聚类 → 簇中心 → 解码回图", fontsize=8, color="#555")

fig.suptitle(f"同一 milestone (进度≈{tpos[cstar]:.2f}, 覆盖 {cov[cstar]:.0%} episodes) 的 pipeline 拆解",
             fontsize=11, y=1.04)

out_doc = C.REPO / "crave/docs/visualization/crave_milestone_pipeline.png"
out_web = C.REPO / "web/showcase/content/img/crave_milestone_pipeline.png"
fig.savefig(out_doc, dpi=140, bbox_inches="tight"); print("SAVED", out_doc, flush=True)
fig.savefig(out_web, dpi=140, bbox_inches="tight"); print("SAVED", out_web, flush=True)
