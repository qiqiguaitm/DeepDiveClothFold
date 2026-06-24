#!/usr/bin/env python
"""建机械臂 patch 原型库(臂掩膜第一步, §5.7 缓解a):
取 smooth800 c4(臂占画面)簇的代表帧 → DINOv2 patch tokens → KMeans(k=24)→
按颜色启发(暗色金属/橙色线缆)保留臂簇为原型 → arm_prototypes.npz。
CPU 可跑(帧数少)。同时输出掩膜可视化供人工验证。
"""
import numpy as np, torch, av
from pathlib import Path
from sklearn.cluster import KMeans

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
OUT = REPO / "temp/armmask"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# 1) 找 c4 帧(smooth800 V0 KMeans 确定性重拟合)
z = np.load(REPO / "temp/recurrence_v0/embeddings.npz")
feats, ep_ids, fr_idx = z["feats"], z["ep_ids"], z["fr_idx"]
km = KMeans(n_clusters=48, n_init=4, random_state=0).fit(feats)
m = np.where(km.labels_ == 4)[0]
d = np.linalg.norm(feats[m] - km.cluster_centers_[4], axis=1)
seen, picks = set(), []
for i in m[np.argsort(d)]:
    if ep_ids[i] not in seen:
        seen.add(ep_ids[i]); picks.append((int(ep_ids[i]), int(fr_idx[i])))
    if len(picks) == 10:
        break
print("c4 帧:", picks)

# 2) 解码 + patch tokens
from transformers import AutoImageProcessor, AutoModel
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
enc = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
base = REPO / "kai0/data/Task_A/self_built/A_new_smooth_800/base/videos/chunk-000/observation.images.top_head"

def load_frame(ep, fr, size=224):
    c = av.open(str(base / f"episode_{ep:06d}.mp4"))
    img = None
    for i, f in enumerate(c.decode(video=0)):
        if i == fr:
            h, w = f.height, f.width
            s = size / min(h, w)
            g = f.reformat(width=round(w*s), height=round(h*s), format="rgb24")
            a = g.to_ndarray(format="rgb24")
            hh, ww = a.shape[:2]
            img = a[(hh-size)//2:(hh+size)//2, (ww-size)//2:(ww+size)//2]
            break
    c.close()
    return img

imgs = [load_frame(ep, fr) for ep, fr in picks]
imgs = [im for im in imgs if im is not None]
P = 224 // 14  # 16x16 patch grid
allp, allrgb = [], []
with torch.no_grad():
    for im in imgs:
        px = proc(images=[im], return_tensors="pt").to(dev)
        toks = enc(**px).last_hidden_state[0, 1:]                # (256, 384)
        toks = torch.nn.functional.normalize(toks, dim=-1).cpu().numpy()
        allp.append(toks)
        rgb = im.reshape(P, 14, P, 14, 3).mean((1, 3)) / 255.0   # (16,16,3) per-patch mean RGB
        allrgb.append(rgb.reshape(-1, 3))
allp = np.concatenate(allp); allrgb = np.concatenate(allrgb)
print("patches:", allp.shape)

# 3) patch 聚类 + 颜色启发选臂簇
pk = KMeans(n_clusters=24, n_init=4, random_state=0).fit(allp)
import colorsys
proto, keep_info = [], []
for c in range(24):
    sel = pk.labels_ == c
    r, g, b = allrgb[sel].mean(0)
    h, s_, v = colorsys.rgb_to_hsv(r, g, b)
    dark = v < 0.35                                  # 暗色金属臂
    orange = (0.02 < h < 0.12) and s_ > 0.35 and v > 0.3   # 橙色线缆
    if dark or orange:
        proto.append(pk.cluster_centers_[c])
        keep_info.append((c, f"v={v:.2f}", f"h={h:.2f}", "dark" if dark else "orange"))
proto = np.array(proto)
proto = proto / np.linalg.norm(proto, axis=1, keepdims=True)
print(f"arm prototypes: {len(proto)}/24  {keep_info}")
np.savez_compressed(OUT / "arm_prototypes.npz", proto=proto)

# 4) 掩膜可视化(6 张随机普通帧: 臂 patch 涂红)
import random, json
eps_all = [json.loads(l)["episode_index"] for l in open(REPO / "kai0/data/Task_A/self_built/A_new_smooth_800/base/meta/episodes.jsonl")]
random.seed(1)
test = [(random.choice(eps_all), fr) for fr in (100, 400, 800)][:3] + picks[:2]
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(test), figsize=(3.6 * len(test), 4))
THR = 0.6
for ax, (ep, fr) in zip(axes, test):
    im = load_frame(ep, fr)
    if im is None:
        continue
    with torch.no_grad():
        px = proc(images=[im], return_tensors="pt").to(dev)
        toks = enc(**px).last_hidden_state[0, 1:]
        toks = torch.nn.functional.normalize(toks, dim=-1).cpu().numpy()
    sim = (toks @ proto.T).max(1)
    rgb = im.reshape(P, 14, P, 14, 3).mean((1, 3)) / 255.0
    import colorsys as cs
    hsv = np.array([[cs.rgb_to_hsv(*rgb[i, j]) for j in range(P)] for i in range(P)])
    col = (hsv[..., 2] < 0.30) | ((hsv[..., 0] > 0.02) & (hsv[..., 0] < 0.12) & (hsv[..., 1] > 0.4))
    mask = (sim.reshape(P, P) > THR) | col
    ov = im.copy().astype(float)
    mm = np.kron(mask, np.ones((14, 14), bool))
    ov[mm] = ov[mm] * 0.3 + np.array([255, 0, 0]) * 0.7
    ax.imshow(ov.astype(np.uint8)); ax.set_title(f"ep{ep} f{fr} masked={mask.mean():.0%}", fontsize=8)
    ax.axis("off")
fig.suptitle(f"arm mask overlay (proto-sim>{THR} OR dark/orange color)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "mask_overlay_check.png", dpi=110)
print("overlay ->", OUT / "mask_overlay_check.png")
