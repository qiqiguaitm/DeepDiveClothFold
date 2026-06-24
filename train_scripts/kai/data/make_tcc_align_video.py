#!/usr/bin/env python
"""TCC v3 两 episode 对齐演示视频 (§2.4.1 配套):
[左] ep A 实时播放  [右] ep B 中与 A 当前帧"对齐"的帧 (TCC 嵌入 argmax 余弦匹配, 中值平滑)
[下] 对齐路径: A 归一时间 -> B 匹配归一时间 (绿=TCC, 灰=raw 特征对照, 虚线=对角参考) + 游标
用法: python make_tcc_align_video.py <epA> <epB> <out_mp4>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, av, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

EPA, EPB, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/kai0_advantage"
CACHE = REPO / "temp/tcc_kai0_armmask/feat_cache"
chunks_size = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)
STRIDE = 10

# ---- 特征 (v3 协议: PMU/PSD 来自同一 200 train eps) ----
zp = np.load(REPO / "temp/recurrence_v0_kai0/embeddings.npz")
EVAL = sorted(set(zp["ep_ids"].tolist()))
all_eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.npz"))
TRAIN = np.random.RandomState(0).permutation([e for e in all_eps if e not in set(EVAL)]).tolist()[:200]

def load_feat(e):
    img = np.load(CACHE / f"ep{e}.npz")["f"]
    n = len(img)
    st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                                  columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    return img, np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1)

print("loading train stats ...")
allP = []
for e in TRAIN:
    try:
        allP.append(load_feat(e)[1])
    except Exception:
        pass
allP = np.concatenate(allP)
PMU, PSD = allP.mean(0), allP.std(0) + 1e-8

def feat(e):
    img, p = load_feat(e)
    p = (p - PMU) / PSD
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    i = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    return np.concatenate([i, p], 1).astype(np.float32)

FA, FB = feat(EPA), feat(EPB)
nA, nB = len(FA), len(FB)
print(f"epA={EPA} ({nA}f@3Hz), epB={EPB} ({nB}f@3Hz)")

class Head(nn.Module):
    def __init__(self, din=412, dh=256, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.GELU(),
                                 nn.Linear(dh, dh), nn.GELU(), nn.Linear(dh, dout))
    def forward(self, x): return self.net(x)

head = Head()
head.load_state_dict(torch.load(REPO / "temp/tcc_v3_kai0/tcc_head_v3.pt"))
head.eval()
def hemb(x):
    with torch.no_grad():
        z = head(torch.from_numpy(x)).numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
def nrm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

match_tcc = median_filter((hemb(FA) @ hemb(FB).T).argmax(1), size=5)
match_raw = median_filter((nrm(FA) @ nrm(FB).T).argmax(1), size=5)
print("alignment computed")

def cam_path(e):
    for cam in ("observation.images.top_head", "top_head"):
        p = DS / "videos" / f"chunk-{e // chunks_size:03d}" / cam / f"episode_{e:06d}.mp4"
        if p.is_file():
            return p

MAXSIDE = 400
def decode_at_stride(e, n):
    """ep 的 3Hz 帧表 (idx k -> 原帧 k*10)"""
    frames = {}
    c = av.open(str(cam_path(e)))
    want = set(min(k * STRIDE, 10**9) for k in range(n))
    for i, f in enumerate(c.decode(video=0)):
        if i in want:
            s = min(1.0, MAXSIDE / max(f.height, f.width))
            g = f.reformat(width=int(f.width*s)//2*2, height=int(f.height*s)//2*2, format="rgb24") if s < 1 else f
            frames[i // STRIDE] = g.to_ndarray(format="rgb24")
    c.close()
    return frames
print("decoding epB ...")
BF = decode_at_stride(EPB, nB)

def stream_cam(path):
    c = av.open(str(path))
    for f in c.decode(video=0):
        s = min(1.0, MAXSIDE / max(f.height, f.width))
        g = f.reformat(width=int(f.width*s)//2*2, height=int(f.height*s)//2*2, format="rgb24") if s < 1 else f
        yield g.to_ndarray(format="rgb24")
    c.close()
def nframes(path):
    c = av.open(str(path)); n = c.streams.video[0].frames or sum(1 for _ in c.decode(video=0)); c.close(); return n

L = min(nframes(cam_path(EPA)), nA * STRIDE)
frame0 = next(stream_cam(cam_path(EPA)))

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0], hspace=0.25, wspace=0.08)
ax_a = fig.add_subplot(gs[0, 0]); ax_a.axis("off")
im_a = ax_a.imshow(frame0)
ax_a.set_title(f"ep{EPA} (playing, held-out)", fontsize=10)
ax_b = fig.add_subplot(gs[0, 1]); ax_b.axis("off")
im_b = ax_b.imshow(BF[int(match_tcc[0])])
ttl_b = ax_b.set_title("", fontsize=10, color="#2ca02c")
ax_p = fig.add_subplot(gs[1, :])
xA = np.arange(nA) / max(1, nA - 1)
ax_p.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=.5, label="diagonal (same pacing)")
ax_p.plot(xA, match_raw / max(1, nB - 1), "-", color="#999", lw=1.2, label="raw feature match")
ax_p.plot(xA, match_tcc / max(1, nB - 1), "-", color="#2ca02c", lw=1.8, label="TCC v3 match")
dot, = ax_p.plot([0], [match_tcc[0] / max(1, nB - 1)], "o", ms=9, color="#2ca02c", mec="k")
cur = ax_p.axvline(0, color="gray", lw=1.2)
ax_p.set_xlabel(f"ep{EPA} normalized time"); ax_p.set_ylabel(f"matched ep{EPB} time")
ax_p.set_title("alignment path: same task stage matched across different pacing", fontsize=9)
ax_p.legend(fontsize=8); ax_p.grid(alpha=.3); ax_p.set_xlim(0, 1); ax_p.set_ylim(-0.02, 1.02)
fig.suptitle(f"TCC v3 cross-episode alignment: ep{EPA} ↔ ep{EPB} (kai0, both held-out from training)", fontsize=11)

def render(img, t):
    im_a.set_data(img)
    a3 = min(t // STRIDE, nA - 1)
    j = int(match_tcc[a3])
    im_b.set_data(BF[min(j, nB - 1)])
    ttl_b.set_text(f"ep{EPB} ALIGNED frame (t={j/max(1,nB-1):.2f}) — TCC match")
    dot.set_data([xA[a3]], [j / max(1, nB - 1)])
    cur.set_xdata([xA[a3], xA[a3]])
    fig.canvas.draw()
    return np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])

f0 = render(frame0, 0); H, W = f0.shape[:2]; H -= H % 2; W -= W % 2
oc = av.open(OUT, mode="w"); st = oc.add_stream("libx264", rate=30)
st.width, st.height, st.pix_fmt = W, H, "yuv420p"; st.options = {"crf": "20"}
t = 0
for img in stream_cam(cam_path(EPA)):
    if t >= L: break
    vf = av.VideoFrame.from_ndarray(render(img, t)[:H, :W], format="rgb24")
    for pkt in st.encode(vf): oc.mux(pkt)
    t += 1
for pkt in st.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {OUT} {W}x{H} {t}f")
