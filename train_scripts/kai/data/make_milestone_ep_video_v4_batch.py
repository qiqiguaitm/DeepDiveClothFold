"""§2.11 终版配方 + 置信门控(v4): 首入需 连续驻留>=2帧 或 margin<=0.8: N=500/k=96/M=20, img⊕proprio, 目标 ep 不入挖掘集(held-out)。
布局同图26, 阶梯扩到 M 级。
用法: python make_milestone_ep_video_v4.py <tag> <ep> <out_mp4> [N=500] [K=96] [M=20]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, av, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

TAG = sys.argv[1]
EPS = [int(x) for x in sys.argv[2].split(",")]
OUTDIR = sys.argv[3]
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

def load_proprio(e, n):
    pq = ds / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * STRIDE, len(st) - 1)]
    dst = np.vstack([np.zeros((1, st.shape[1])), np.diff(st, axis=0)])
    return np.concatenate([st, dst], 1)

# ---- 挖掘集: 全缓存除目标 ep, 打乱取 N 条 (held-out) ----
all_eps = sorted(int(p.stem[2:]) for p in cache.glob("ep*.npz"))
pool = [e for e in all_eps if e not in set(EPS)]
mined = sorted(np.random.RandomState(0).permutation(pool)[:N].tolist())
IMG, PRP, E, T, FR = [], [], [], [], []
for e in mined:
    try:
        f = np.load(cache / f"ep{e}.npz")["f"]
        n = len(f)
        p = load_proprio(e, n)
    except Exception:
        continue
    IMG.append(f); PRP.append(p)
    E.append(np.full(n, e)); T.append(np.arange(n) / max(1, n - 1)); FR.append(np.arange(n) * STRIDE)
IMG = np.concatenate(IMG); PRP = np.concatenate(PRP)
E = np.concatenate(E); T = np.concatenate(T); FR = np.concatenate(FR)
n_ep = len(set(E.tolist()))
PMU, PSD = PRP.mean(0), PRP.std(0) + 1e-8
PRPn = (PRP - PMU) / PSD; PRPn /= np.linalg.norm(PRPn, axis=1, keepdims=True)
IMGn = IMG / np.linalg.norm(IMG, axis=1, keepdims=True)
F = np.concatenate([IMGn, PRPn], 1)
print(f"mining: {n_ep} eps, {len(F)} frames; KMeans k={K} ...")
km = KMeans(n_clusters=K, n_init=2, random_state=0).fit(F)
lab_all = km.labels_
cov = np.array([len(set(E[lab_all == c].tolist())) / n_ep for c in range(K)])
tpos = np.array([T[lab_all == c].mean() for c in range(K)])
ms = sorted(np.argsort(cov)[-M:].tolist(), key=lambda c: tpos[c])
idx = {c: i + 1 for i, c in enumerate(ms)}
print(f"milestones (N={N},k={K},M={M}):")
for i, c in enumerate(ms):
    print(f"  M{i+1:>2}=c{c:02d} cov={cov[c]:.0%} t̄={tpos[c]:.2f}")

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

canon = {}
for k, c in enumerate(ms):
    m = np.where(lab_all == c)[0]
    d = np.linalg.norm(F[m] - km.cluster_centers_[c], axis=1)
    i = m[np.argmin(d)]
    canon[k + 1] = (grab_frame(int(E[i]), int(FR[i])), int(E[i]), int(FR[i]))
print("canon frames ready")

# ---- 逐条渲染 ----
for EP in EPS:
    OUT = f"{OUTDIR}/milestone_ep_{TAG}_{EP}_v4gated_sync.mp4"
    print(f"\n===== ep{EP} =====")
    # ---- 目标 ep (held-out) 逐帧分配 ----
    fE = np.load(cache / f"ep{EP}.npz")["f"]
    nE = len(fE)
    pE = (load_proprio(EP, nE) - PMU) / PSD
    pE /= np.linalg.norm(pE, axis=1, keepdims=True)
    gFq = np.concatenate([fE / np.linalg.norm(fE, axis=1, keepdims=True), pE], 1)
    Dq = np.linalg.norm(gFq[:, None, :] - km.cluster_centers_[None], axis=2)
    labE = Dq.argmin(1)
    ds_ = np.sort(Dq, axis=1)
    marginE = ds_[:, 0] / ds_[:, 1]
    rawE0 = np.array([idx.get(c, 0) for c in labE])
    # 置信门控: 命中仅当 连续驻留>=2帧 或 margin<=0.8 (纯状态判据)
    def gated(j):
        if rawE0[j] == 0: return 0
        dwell = (j + 1 < len(labE) and labE[j + 1] == labE[j]) or (j > 0 and labE[j - 1] == labE[j])
        return rawE0[j] if (dwell or marginE[j] <= 0.8) else 0
    rawE = np.array([gated(j) for j in range(len(labE))])
    killed = int(((rawE0 > 0) & (rawE == 0)).sum())
    print(f"gating: {killed} low-confidence hits suppressed ({killed/max(1,(rawE0>0).sum()):.0%} of raw hits)")
    passedE, seen = [], set()
    for r in rawE:
        if r > 0:
            seen.add(r)
        passedE.append(len(seen) / M)
    passedE = np.array(passedE)
    v30 = np.repeat(passedE, STRIDE)
    hit_t3 = {k: np.where(rawE == k)[0] for k in range(1, M + 1)}
    print(f"ep{EP}: hit {len(set(rawE[rawE>0].tolist()))}/{M} milestones, final V={passedE[-1]:.2f}")
    print("first-entry table (gated):")
    for k in range(1, M + 1):
        h = np.where(rawE == k)[0]
        h0 = np.where(rawE0 == k)[0]
        a = f"{h[0]/(len(rawE)-1):.2f}" if len(h) else "-"
        b = f"{h0[0]/(len(rawE0)-1):.2f}" if len(h0) else "-"
        print(f"  M{k:>2}=c{ms[k-1]:02d} t̄={tpos[ms[k-1]]:.2f}: raw首入={b} gated首入={a}")
    
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
    
    L = min(nframes(cam_path(EP)), len(rawE) * STRIDE)
    frame0 = next(stream_cam(cam_path(EP)))
    
    mscolors = matplotlib.colormaps["tab20"]
    fig = plt.figure(figsize=(15.5, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.6], height_ratios=[1.35, 1.0], hspace=0.25, wspace=0.13)
    ax_cam = fig.add_subplot(gs[0, 0]); ax_cam.axis("off")
    im_cam = ax_cam.imshow(frame0)
    ax_cam.set_title(f"LIVE: {TAG} ep{EP} (HELD-OUT from mining)", fontsize=10)
    ax_can = fig.add_subplot(gs[1, 0]); ax_can.axis("off")
    im_can = ax_can.imshow(np.zeros_like(canon[1][0]))
    ttl_can = ax_can.set_title("milestone reference: (none yet)", fontsize=9, color="#9467bd")
    
    ax_lad = fig.add_subplot(gs[0, 1])
    for k in range(1, M + 1):
        c = ms[k - 1]
        ax_lad.axhline(k, color="#eee", lw=3, zorder=0)
        if len(hit_t3[k]):
            ax_lad.scatter(hit_t3[k] / 3, np.full(len(hit_t3[k]), k), s=16, color=mscolors((k-1) % 20), zorder=2)
        ax_lad.text(1.01, k, f"M{k}=c{c} {cov[c]:.0%} t̄{tpos[c]:.2f}",
                    fontsize=6.2, va="center", color=mscolors((k-1) % 20),
                    transform=ax_lad.get_yaxis_transform())
    cur_lad = ax_lad.axvline(0, color="gray", lw=1.2)
    dot_lad, = ax_lad.plot([], [], "o", ms=10, mfc="none", mec="red", mew=2)
    ax_lad.set_ylim(0.3, M + 0.7); ax_lad.set_xlim(0, L / 30)
    ax_lad.set_ylabel("milestone"); ax_lad.set_title(f"M={M} milestone hits + coverage — final recipe N={N}/k={K}/M={M}", fontsize=9)
    ax_lad.grid(axis="x", alpha=.2)
    
    ax_v = fig.add_subplot(gs[1, 1], sharex=ax_lad)
    ax_v.plot(np.arange(L) / 30, v30[:L], "-", color="#9467bd", lw=1.6)
    dot_v, = ax_v.plot([0], [v30[0]], "o", color="#9467bd", ms=8, mec="k")
    cur_v = ax_v.axvline(0, color="gray", lw=1.2)
    ax_v.set_ylim(-0.05, 1.1); ax_v.set_xlabel("seconds"); ax_v.set_ylabel(f"V_milestone (/{M})")
    ax_v.grid(alpha=.25)
    fig.suptitle(f"{TAG} ep{EP}: final recipe (img armmask ⊕ proprio, N={N}/k={K}/M={M}, held-out) — single-episode audit", fontsize=11)
    
    last_k = [0]
    def render(camimg, t):
        im_cam.set_data(camimg)
        t3 = min(t // STRIDE, len(rawE) - 1)
        k = int(rawE[t3])
        if k > 0 and k != last_k[0]:
            img, ce, cf = canon[k]
            im_can.set_data(img)
            c = ms[k - 1]
            ttl_can.set_text(f"NOW HIT M{k}=c{c} (cov {cov[c]:.0%}) — reference: ep{ce} f{cf}")
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
    