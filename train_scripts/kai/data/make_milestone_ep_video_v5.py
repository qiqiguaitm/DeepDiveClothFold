"""V2 标签版单 episode 进度视频 (v5):
- milestone 按 P_k(门控首入中位数)重排序, 阶梯 y 轴 = 校准进度值(非均匀)
- 循环型 milestone (runs/ep>1.5 或 runs>1.3且隔段重入>0.2) 灰显, 不计入 V
- V 面板: V2 校准阶梯(绿, cummax P_k) vs 旧等步长(红) vs 对照
用法: python make_milestone_ep_video_v5.py <tag> <ep> <out_mp4>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, av, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG, EP, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
N, K, M = 500, 96, 20
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
CFG = {
    "smooth800": dict(cache="temp/tcc_smooth800_armmask/feat_cache",
                      ds="kai0/data/Task_A/self_built/A_new_smooth_800/base"),
    "kai0": dict(cache="temp/tcc_kai0_armmask/feat_cache",
                 ds="kai0/data/Task_A/kai0_advantage"),
}[TAG]
ds, cache = REPO / CFG["ds"], REPO / CFG["cache"]
chunks_size = json.load(open(ds / "meta/info.json")).get("chunks_size", 1000)
STRIDE = 10
from sklearn.cluster import KMeans

def load_proprio(e, n):
    pq = ds / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * STRIDE, len(st) - 1)]
    return np.concatenate([st, np.vstack([np.zeros((1, st.shape[1])), np.diff(st, axis=0)])], 1)

# ---- 挖掘 (held-out: 排除目标 ep) ----
all_eps = sorted(int(p.stem[2:]) for p in cache.glob("ep*.npz"))
mined = sorted(np.random.RandomState(0).permutation([e for e in all_eps if e != EP])[:N].tolist())
IMG, PRP, E, T, FR = [], [], [], [], []
for e in mined:
    try:
        f = np.load(cache / f"ep{e}.npz")["f"]; n = len(f); p = load_proprio(e, n)
    except Exception:
        continue
    IMG.append(f); PRP.append(p)
    E.append(np.full(n, e)); T.append(np.arange(n) / max(1, n - 1)); FR.append(np.arange(n) * STRIDE)
IMG = np.concatenate(IMG); PRP = np.concatenate(PRP)
E = np.concatenate(E); T = np.concatenate(T); FR = np.concatenate(FR)
n_ep = len(set(E.tolist()))
PMU, PSD = PRP.mean(0), PRP.std(0) + 1e-8
PRPn = (PRP - PMU) / PSD; PRPn /= np.linalg.norm(PRPn, axis=1, keepdims=True)
F = np.concatenate([IMG / np.linalg.norm(IMG, axis=1, keepdims=True), PRPn], 1)
print(f"mining {n_ep} eps, {len(F)} frames; KMeans k={K} ...")
km = KMeans(n_clusters=K, n_init=2, random_state=0).fit(F)
lab_all = km.labels_
cov = np.array([len(set(E[lab_all == c].tolist())) / n_ep for c in range(K)])
ms_cov = np.argsort(cov)[-M:].tolist()

# ---- V2 标签: P_k(门控首入中位) + 循环检测 ----
mined_set = sorted(set(E.tolist()))
def gated_runs(e, c):
    m = np.where(E == e)[0]; hit = m[lab_all[m] == c]
    runs = []; s = None; prev = None
    for i in hit:
        if prev is None or i != prev + 1:
            if s is not None: runs.append((s, prev))
            s = i
        prev = i
    if s is not None: runs.append((s, prev))
    return [r for r in runs if r[1] - r[0] >= 1]

Pk, cyc = {}, {}
for c in ms_cov:
    fe, nr, re_, vis = [], [], 0, 0
    for e in mined_set:
        rs = gated_runs(e, c)
        if not rs: continue
        vis += 1; fe.append(T[rs[0][0]]); nr.append(len(rs))
        if len(rs) >= 2:
            m = np.where(E == e)[0]
            for (a1, b1), (a2, b2) in zip(rs[:-1], rs[1:]):
                if any(x in ms_cov and x != c for x in lab_all[[j for j in m if b1 < j < a2]]):
                    re_ += 1; break
    Pk[c] = float(np.median(fe)) if fe else 0.5
    runs = float(np.mean(nr)) if nr else 1.0
    cyc[c] = runs > 1.5 or (runs > 1.3 and re_ / max(1, vis) > 0.2)
ms = sorted(ms_cov, key=lambda c: Pk[c])           # P_k 排序 (V2)
idx = {c: i + 1 for i, c in enumerate(ms)}
abs_ms = [c for c in ms if not cyc[c]]
print("V2 milestones (sorted by P_k):")
for i, c in enumerate(ms):
    print(f"  M{i+1:>2}=c{c:02d} cov={cov[c]:.0%} P_k={Pk[c]:.2f} {'[CYCLIC, 不计入V]' if cyc[c] else ''}")

def cam_path(e):
    for cam in ("observation.images.top_head", "top_head"):
        p = ds / "videos" / f"chunk-{e // chunks_size:03d}" / cam / f"episode_{e:06d}.mp4"
        if p.is_file():
            return p

def grab_frame(e, fr):
    c = av.open(str(cam_path(e))); img = None
    for i, f in enumerate(c.decode(video=0)):
        if i == fr:
            img = f.to_ndarray(format="rgb24"); break
    c.close(); return img

canon = {}
for k, c in enumerate(ms):
    m = np.where(lab_all == c)[0]
    d = np.linalg.norm(F[m] - km.cluster_centers_[c], axis=1)
    i = m[np.argmin(d)]
    canon[k + 1] = (grab_frame(int(E[i]), int(FR[i])), int(E[i]), int(FR[i]))
print("canon ready")

# ---- 目标 ep (held-out): 门控分配 + V2/V_uniform ----
fE = np.load(cache / f"ep{EP}.npz")["f"]; nE = len(fE)
pE = (load_proprio(EP, nE) - PMU) / PSD; pE /= np.linalg.norm(pE, axis=1, keepdims=True)
gF = np.concatenate([fE / np.linalg.norm(fE, axis=1, keepdims=True), pE], 1)
D = np.linalg.norm(gF[:, None, :] - km.cluster_centers_[None], axis=2)
labE = D.argmin(1)
dsrt = np.sort(D, axis=1); marginE = dsrt[:, 0] / dsrt[:, 1]
def conf(j):
    dwell = (j + 1 < nE and labE[j + 1] == labE[j]) or (j > 0 and labE[j - 1] == labE[j])
    return dwell or marginE[j] <= 0.8
rawE = np.array([idx.get(labE[j], 0) if conf(j) else 0 for j in range(nE)])
v2 = np.zeros(nE); vu = np.zeros(nE)
cur = 0.0; seen = set()
for j in range(nE):
    k = rawE[j]
    if k > 0:
        c = ms[k - 1]
        if c not in seen:
            seen.add(c)
            if not cyc[c]:
                cur = max(cur, Pk[c])
    v2[j] = cur
    vu[j] = len([c for c in seen if not cyc[c]]) / max(1, len(abs_ms))
v2_30, vu_30, raw30 = np.repeat(v2, STRIDE), np.repeat(vu, STRIDE), np.repeat(rawE, STRIDE)
hit_t3 = {k: np.where(rawE == k)[0] for k in range(1, M + 1)}
print(f"ep{EP}: abs-hit {len([c for c in seen if not cyc[c]])}/{len(abs_ms)}, final V2={v2[-1]:.2f}")

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

L = min(nframes(cam_path(EP)), nE * STRIDE)
frame0 = next(stream_cam(cam_path(EP)))
mscolors = matplotlib.colormaps["tab20"]
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.65], height_ratios=[1.35, 1.0], hspace=0.25, wspace=0.13)
ax_cam = fig.add_subplot(gs[0, 0]); ax_cam.axis("off")
im_cam = ax_cam.imshow(frame0)
ax_cam.set_title(f"LIVE: {TAG} ep{EP} (held-out)", fontsize=10)
ax_can = fig.add_subplot(gs[1, 0]); ax_can.axis("off")
im_can = ax_can.imshow(np.zeros_like(canon[1][0]))
ttl_can = ax_can.set_title("milestone reference: (none yet)", fontsize=9, color="#9467bd")

# 阶梯面板: y = P_k (真实进度值!), 非均匀
ax_lad = fig.add_subplot(gs[0, 1])
for k in range(1, M + 1):
    c = ms[k - 1]
    y = Pk[c]
    col = "#999" if cyc[c] else mscolors((k - 1) % 20)
    ax_lad.axhline(y, color="#eee", lw=2, zorder=0)
    if len(hit_t3[k]):
        ax_lad.scatter(hit_t3[k] / 3, np.full(len(hit_t3[k]), y), s=14,
                       color=col, marker="x" if cyc[c] else "o", zorder=2)
    ylab = 0.02 + (k - 1) / (M - 1) * 0.96      # 标签等距排列, 细线连到 P_k 高度
    ax_lad.annotate(f"M{k}=c{c} {cov[c]:.0%} P={Pk[c]:.2f}" + (" CYC" if cyc[c] else ""),
                    xy=(1.0, y), xycoords=ax_lad.get_yaxis_transform(),
                    xytext=(1.06, ylab), textcoords=ax_lad.get_yaxis_transform(),
                    fontsize=6.0, va="center", color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.6))
cur_lad = ax_lad.axvline(0, color="gray", lw=1.2)
dot_lad, = ax_lad.plot([], [], "o", ms=10, mfc="none", mec="red", mew=2)
ax_lad.set_ylim(-0.03, 1.03); ax_lad.set_xlim(0, L / 30)
ax_lad.set_ylabel("calibrated progress P_k")
ax_lad.set_title("V2 labels: y = CALIBRATED progress (non-uniform); gray X = cyclic (excluded from V)", fontsize=9)
ax_lad.grid(axis="x", alpha=.2)

ax_v = fig.add_subplot(gs[1, 1], sharex=ax_lad)
ax_v.plot(np.arange(L) / 30, v2_30[:L], "-", color="#2ca02c", lw=1.9, label="V2 calibrated (cummax P_k)")
ax_v.plot(np.arange(L) / 30, vu_30[:L], "-", color="#d62728", lw=1.1, alpha=.7, label="V1 uniform (k/M, for contrast)")
dot_v, = ax_v.plot([0], [v2_30[0]], "o", color="#2ca02c", ms=8, mec="k")
cur_v = ax_v.axvline(0, color="gray", lw=1.2)
ax_v.set_ylim(-0.05, 1.05); ax_v.set_xlabel("seconds"); ax_v.set_ylabel("V")
ax_v.legend(fontsize=8, loc="lower right"); ax_v.grid(alpha=.25)
fig.suptitle(f"{TAG} ep{EP}: V2 stage labels — P_k-ordered milestones, calibrated step heights, cyclic excluded", fontsize=11)

last_k = [0]
def render(camimg, t):
    im_cam.set_data(camimg)
    t3 = min(t // STRIDE, nE - 1)
    k = int(rawE[t3])
    if k > 0 and k != last_k[0]:
        img, ce, cf = canon[k]
        im_can.set_data(img)
        c = ms[k - 1]
        tagc = " [CYCLIC — no V credit]" if cyc[c] else ""
        ttl_can.set_text(f"NOW HIT M{k}=c{c} P_k={Pk[c]:.2f} (cov {cov[c]:.0%}){tagc} — ref ep{ce} f{cf}")
        ttl_can.set_color("#999" if cyc[c] else "#9467bd")
        last_k[0] = k
    cur_lad.set_xdata([t / 30, t / 30]); cur_v.set_xdata([t / 30, t / 30])
    if k > 0:
        dot_lad.set_data([t / 30], [Pk[ms[k - 1]]])
    dot_v.set_data([t / 30], [v2_30[t]])
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
