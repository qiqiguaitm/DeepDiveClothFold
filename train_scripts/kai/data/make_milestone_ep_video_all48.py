"""单 episode 全 48 簇对账视频:
[左上] 实时相机  [左下] 当前簇参考帧(另一 episode 质心最近帧, 全 48 簇都有)
[中上] 本 ep 逐帧 48 簇序列(灰=全集, 彩=milestone, 红圈=当前)  [中下] V_milestone
[右]   48 簇覆盖率全表(当前簇实时高亮)
用法: python make_milestone_ep_video_all48.py <tag smooth800|kai0> <ep> <out_mp4>
"""
import json, sys
from pathlib import Path
import numpy as np, av, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans

TAG, EP, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
CFG = {
    "smooth800": dict(probe="temp/recurrence_v0/embeddings.npz",
                      cache="temp/tcc_smooth800_armmask/feat_cache",
                      ds="kai0/data/Task_A/self_built/A_new_smooth_800/base"),
    "kai0": dict(probe="temp/recurrence_v0_kai0/embeddings.npz",
                 cache="temp/tcc_kai0_armmask/feat_cache",
                 ds="kai0/data/Task_A/kai0_advantage"),
}[TAG]
ds, cache = REPO / CFG["ds"], REPO / CFG["cache"]
chunks_size = json.load(open(ds / "meta/info.json")).get("chunks_size", 1000)
STRIDE = 10
K = 48

# ---- 挖掘 (armmask, V0 50ep, 确定性, 与 audit/视频脚本同协议) ----
zp = np.load(REPO / CFG["probe"])
mined = sorted(set(zp["ep_ids"].tolist()))
F, E, T, FR = [], [], [], []
for e in mined:
    f = np.load(cache / f"ep{e}.npz")["f"]
    n = len(f)
    F.append(f); E.append(np.full(n, e)); T.append(np.arange(n) / max(1, n - 1)); FR.append(np.arange(n) * STRIDE)
F = np.concatenate(F); E = np.concatenate(E); T = np.concatenate(T); FR = np.concatenate(FR)
km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(F)
lab_all = km.labels_
n_ep = len(mined)
cov = np.array([len(set(E[lab_all == c].tolist())) / n_ep for c in range(K)])
tpos = np.array([T[lab_all == c].mean() if (lab_all == c).any() else 0.5 for c in range(K)])
ms = sorted(np.argsort(cov)[-10:].tolist(), key=lambda c: tpos[c])
idx = {c: i + 1 for i, c in enumerate(ms)}
print("milestones:", [(f"M{i+1}", int(c), f"{cov[c]:.0%}") for i, c in enumerate(ms)])

def cam_path(e):
    for cam in ("observation.images.top_head", "top_head"):
        p = ds / "videos" / f"chunk-{e // chunks_size:03d}" / cam / f"episode_{e:06d}.mp4"
        if p.is_file():
            return p

def grab_frame(e, fr):
    c = av.open(str(cam_path(e)))
    img = None
    for i, f in enumerate(c.decode(video=0)):
        if i == fr:
            img = f.to_ndarray(format="rgb24"); break
    c.close(); return img

# ---- 本 episode 逐帧分配 ----
fE = np.load(cache / f"ep{EP}.npz")["f"]
labE = km.predict(fE)
rawE = np.array([idx.get(c, 0) for c in labE])
passedE, seen = [], set()
for r in rawE:
    if r > 0:
        seen.add(r)
    passedE.append(len(seen) / 10)
passedE = np.array(passedE)
v30 = np.repeat(passedE, STRIDE)

# ---- 全 48 簇参考帧 (仅本 ep 实际经过的簇才需要; 来自 != EP 的质心最近帧) ----
need = sorted(set(labE.tolist()))
print(f"ep{EP} passes through {len(need)}/48 clusters; grabbing reference frames ...")
canon = {}
for c in need:
    m = np.where((lab_all == c) & (E != EP))[0]
    if not len(m):
        m = np.where(lab_all == c)[0]
    d = np.linalg.norm(F[m] - km.cluster_centers_[c], axis=1)
    i = m[np.argmin(d)]
    canon[c] = (grab_frame(int(E[i]), int(FR[i])), int(E[i]), int(FR[i]))
print("reference frames ready:", len(canon))

# ---- 渲染 ----
MAXSIDE = 400
def stream_cam(path):
    c = av.open(str(path))
    for f in c.decode(video=0):
        s = min(1.0, MAXSIDE / max(f.height, f.width))
        g = f.reformat(width=int(f.width*s)//2*2, height=int(f.height*s)//2*2, format="rgb24") if s < 1 else f
        yield g.to_ndarray(format="rgb24")
    c.close()
def nframes(path):
    c = av.open(str(path)); n = c.streams.video[0].frames or sum(1 for _ in c.decode(video=0)); c.close(); return n

L = min(nframes(cam_path(EP)), len(labE) * STRIDE)
frame0 = next(stream_cam(cam_path(EP)))
print("frames", L)

mscolors = matplotlib.colormaps["tab10"]
fig = plt.figure(figsize=(17.5, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.55, 0.62], height_ratios=[1.4, 1.0],
                      hspace=0.24, wspace=0.16)
ax_cam = fig.add_subplot(gs[0, 0]); ax_cam.axis("off")
im_cam = ax_cam.imshow(frame0)
ax_cam.set_title(f"LIVE: {TAG} ep{EP}", fontsize=10)
ax_can = fig.add_subplot(gs[1, 0]); ax_can.axis("off")
c0 = int(labE[0])
im_can = ax_can.imshow(canon[c0][0])
ttl_can = ax_can.set_title("", fontsize=9, color="#9467bd")

# 中上: 48 簇序列
ax_seq = fig.add_subplot(gs[0, 1])
x3 = np.arange(len(labE)) / 3
nonms = np.array([c not in idx for c in labE])
ax_seq.scatter(x3[nonms], labE[nonms], s=8, c="#bbb", zorder=1, label="non-milestone")
for k, c in enumerate(ms):
    hit = labE == c
    if hit.any():
        ax_seq.scatter(x3[hit], labE[hit], s=26, color=mscolors(k % 10), zorder=2)
cur_seq = ax_seq.axvline(0, color="gray", lw=1.2)
ring, = ax_seq.plot([x3[0]], [labE[0]], "o", ms=13, mfc="none", mec="red", mew=2, zorder=3)
ax_seq.set_ylim(-1.5, K + 0.5); ax_seq.set_xlim(0, L / 30)
ax_seq.set_ylabel("cluster id (0-47)")
ax_seq.set_title("per-frame assignment over ALL 48 clusters (gray = non-milestone, colored = milestone, red ring = now)", fontsize=9)
ax_seq.grid(alpha=.2)

# 中下: V_milestone
ax_v = fig.add_subplot(gs[1, 1], sharex=ax_seq)
ax_v.plot(np.arange(L) / 30, v30[:L], "-", color="#9467bd", lw=1.6)
dot_v, = ax_v.plot([0], [v30[0]], "o", color="#9467bd", ms=8, mec="k")
cur_v = ax_v.axvline(0, color="gray", lw=1.2)
ax_v.set_ylim(-0.05, 1.1); ax_v.set_xlabel("seconds")
ax_v.set_ylabel("V_milestone")
ax_v.set_title("V(t) = #distinct milestones first-entered so far / 10", fontsize=9)
ax_v.grid(alpha=.25)

# 右: 48 簇覆盖率全表 (2 列 × 24 行), 当前簇高亮
ax_tab = fig.add_subplot(gs[:, 2]); ax_tab.axis("off")
ax_tab.set_title(f"coverage of ALL {K} clusters\n(% of {n_ep} mined eps visiting it)", fontsize=9)
cellpos = {}
for c in range(K):
    col, row = divmod(c, 24)
    x = 0.04 + col * 0.52
    y = 1 - (row + 1) / 25.5
    if c in idx:
        k = idx[c]
        txt, color, w = f"c{c:02d} {cov[c]:>4.0%} M{k}", mscolors((k - 1) % 10), "bold"
    else:
        txt, color, w = f"c{c:02d} {cov[c]:>4.0%}", "#666", "normal"
    ax_tab.text(x, y, txt, fontsize=7.6, color=color, fontweight=w,
                family="monospace", transform=ax_tab.transAxes, va="center")
    cellpos[c] = (x - 0.02, y)
hl = Rectangle((0, 0), 0.46, 1 / 26, transform=ax_tab.transAxes,
               facecolor="yellow", alpha=0.45, edgecolor="red", lw=1.2, zorder=0)
ax_tab.add_patch(hl)
fig.suptitle(f"{TAG} ep{EP}: ALL-48-cluster audit — live frame / current-cluster reference / full coverage table / V_milestone", fontsize=11)

state = [-1]
def render(camimg, t):
    im_cam.set_data(camimg)
    t3 = min(t // STRIDE, len(labE) - 1)
    c = int(labE[t3])
    if c != state[0]:
        img, ce, cf = canon[c]
        im_can.set_data(img)
        mtag = f" = M{idx[c]}" if c in idx else " (non-milestone)"
        ttl_can.set_text(f"NOW IN c{c}{mtag}, cov {cov[c]:.0%} — reference: ep{ce} f{cf}")
        ttl_can.set_color(mscolors((idx[c] - 1) % 10) if c in idx else "#666")
        x, y = cellpos[c]
        hl.set_xy((x, y - 1 / 52))
        state[0] = c
    cur_seq.set_xdata([t / 30, t / 30]); cur_v.set_xdata([t / 30, t / 30])
    ring.set_data([x3[t3]], [c])
    dot_v.set_data([t / 30], [v30[t]])
    fig.canvas.draw()
    return np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])

f0 = render(frame0, 0); H, Wd = f0.shape[:2]; H -= H % 2; Wd -= Wd % 2
oc = av.open(OUT, mode="w"); st = oc.add_stream("libx264", rate=30)
st.width, st.height, st.pix_fmt = Wd, H, "yuv420p"; st.options = {"crf": "20"}
t = 0
for camimg in stream_cam(cam_path(EP)):
    if t >= L: break
    vf = av.VideoFrame.from_ndarray(render(camimg, t)[:H, :Wd], format="rgb24")
    for pkt in st.encode(vf): oc.mux(pkt)
    t += 1
for pkt in st.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {OUT} {Wd}x{H} {t}f")
