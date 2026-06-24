"""单 episode × milestone 对账视频:
[左上] 实时相机  [左下] 当前命中 milestone 的标准帧(另一 episode 的质心代表帧)
[右上] milestone 阶梯 (M1-M10, 各标覆盖率, 命中点亮 + 游标)  [右下] V_milestone
用法: python make_milestone_ep_video.py <tag smooth800|kai0> <ep> <out_mp4>
"""
import json, sys
from pathlib import Path
import numpy as np, av, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
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

# ---- 挖掘 (armmask, V0 50ep, 确定性) ----
zp = np.load(REPO / CFG["probe"])
mined = sorted(set(zp["ep_ids"].tolist()))
F, E, T, FR = [], [], [], []
for e in mined:
    f = np.load(cache / f"ep{e}.npz")["f"]
    n = len(f)
    F.append(f); E.append(np.full(n, e)); T.append(np.arange(n) / max(1, n - 1)); FR.append(np.arange(n) * STRIDE)
F = np.concatenate(F); E = np.concatenate(E); T = np.concatenate(T); FR = np.concatenate(FR)
km = KMeans(n_clusters=48, n_init=4, random_state=0).fit(F)
lab_all = km.labels_
n_ep = len(mined)
cov = np.array([len(set(E[lab_all == c].tolist())) / n_ep for c in range(48)])
tpos = np.array([T[lab_all == c].mean() for c in range(48)])
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

# ---- 每个 milestone 的标准帧 (来自 != EP 的质心最近帧) ----
canon = {}
for k, c in enumerate(ms):
    m = np.where((lab_all == c) & (E != EP))[0]
    d = np.linalg.norm(F[m] - km.cluster_centers_[c], axis=1)
    i = m[np.argmin(d)]
    canon[k + 1] = (grab_frame(int(E[i]), int(FR[i])), int(E[i]), int(FR[i]))
    print(f"  M{k+1}=c{c} canon: ep{int(E[i])} f{int(FR[i])}")

# ---- 本 episode 的逐帧分配 (3Hz → 30Hz upsample) ----
fE = np.load(cache / f"ep{EP}.npz")["f"]
labE = km.predict(fE)
rawE = np.array([idx.get(c, 0) for c in labE])          # 3Hz milestone 等级
passedE = []
seen = set()
for r in rawE:
    if r > 0:
        seen.add(r)
    passedE.append(len(seen) / 10)
passedE = np.array(passedE)
raw30 = np.repeat(rawE, STRIDE)
v30 = np.repeat(passedE, STRIDE)
hit_t3 = {k: np.where(rawE == k)[0] for k in range(1, 11)}   # 每 milestone 的命中时刻(3Hz)

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

L = min(nframes(cam_path(EP)), len(raw30))
frame0 = next(stream_cam(cam_path(EP)))
print("frames", L)

fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5], hspace=0.25, wspace=0.12)
ax_cam = fig.add_subplot(gs[0, 0]); ax_cam.axis("off")
im_cam = ax_cam.imshow(frame0)
ax_cam.set_title(f"LIVE: {TAG} ep{EP}", fontsize=10)
ax_can = fig.add_subplot(gs[1, 0]); ax_can.axis("off")
im_can = ax_can.imshow(np.zeros_like(canon[1][0]))
ttl_can = ax_can.set_title("milestone reference: (none yet)", fontsize=9, color="#9467bd")

mscolors = cm.get_cmap("tab10")
ax_lad = fig.add_subplot(gs[0, 1])
x3 = np.arange(len(rawE)) / 3
for k in range(1, 11):
    c = ms[k - 1]
    ax_lad.axhline(k, color="#eee", lw=4, zorder=0)
    if len(hit_t3[k]):
        ax_lad.scatter(hit_t3[k] / 3, np.full(len(hit_t3[k]), k), s=22, color=mscolors((k-1) % 10), zorder=2)
    ax_lad.text(1.01, k, f"M{k}=c{c}  cov={cov[c]:.0%}  t̄={tpos[c]:.2f}",
                fontsize=7.5, va="center", color=mscolors((k-1) % 10),
                transform=ax_lad.get_yaxis_transform())
cur_lad = ax_lad.axvline(0, color="gray", lw=1.2)
dot_lad, = ax_lad.plot([], [], "o", ms=12, mfc="none", mec="red", mew=2)
ax_lad.set_ylim(0.3, 10.7); ax_lad.set_xlim(0, L / 30)
ax_lad.set_ylabel("milestone"); ax_lad.set_title("milestone hits (dots) + coverage labels", fontsize=9)
ax_lad.grid(axis="x", alpha=.2)

ax_v = fig.add_subplot(gs[1, 1], sharex=ax_lad)
ax_v.plot(np.arange(L) / 30, v30[:L], "-", color="#9467bd", lw=1.6)
dot_v, = ax_v.plot([0], [v30[0]], "o", color="#9467bd", ms=8, mec="k")
cur_v = ax_v.axvline(0, color="gray", lw=1.2)
ax_v.set_ylim(-0.05, 1.1); ax_v.set_xlabel("seconds"); ax_v.set_ylabel("V_milestone")
ax_v.grid(alpha=.25)
fig.suptitle(f"{TAG} ep{EP}: live frame vs milestone reference state vs coverage — single-episode audit", fontsize=11)

last_k = [0]
def render(camimg, t):
    im_cam.set_data(camimg)
    t3 = min(t // STRIDE, len(rawE) - 1)
    k = int(rawE[t3])
    if k > 0 and k != last_k[0]:
        img, ce, cf = canon[k]
        im_can.set_data(img)
        c = ms[k - 1]
        ttl_can.set_text(f"NOW HIT M{k}=c{c} (cov {cov[c]:.0%}) — reference from ep{ce} f{cf}")
        last_k[0] = k
    cur_lad.set_xdata([t / 30, t / 30]); cur_v.set_xdata([t / 30, t / 30])
    if k > 0:
        dot_lad.set_data([t / 30], [k])
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
