#!/usr/bin/env python
"""F2 (§4.4.10) V2.1 退步回落 value 与 rollout 视频同步可视化。
[上] autonomy rollout top_head 实时播放 (7676 帧)
[下] V2.1 退步回落 (绿) vs monotone cummax (红) + 退步事件标注 + ΔV<0 区段 + 游标
挖掘/规则与 f2_rollout_regression_test.py 完全一致 (确定性)。
用法: python make_f2_value_sync_video.py <out_mp4>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, av
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_MP4 = sys.argv[1]
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DS = REPO / "kai0/data/Task_A/self_built/A_new_smooth_800/base"
CACHE = REPO / "temp/tcc_smooth800_armmask/feat_cache"
chunks_size = json.load(open(DS / "meta/info.json")).get("chunks_size", 1000)

# ===== 与 f2_rollout_regression_test.py 一致的挖掘 + 规则 =====
all_eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.npz"))
mined = sorted(np.random.RandomState(0).permutation([e for e in all_eps if e != 660])[:500].tolist())
def load_raw(ds, cache, e):
    img = np.load(cache / f"ep{e}.npz")["f"]
    n = len(img)
    st = np.stack(pd.read_parquet(ds / "data" / f"chunk-{e // chunks_size:03d}" / f"episode_{e:06d}.parquet",
                                  columns=["observation.state"])["observation.state"].to_numpy())
    st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
    return img, np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1)
print("[sync] mining ...")
IMG, PRP, E, T = [], [], [], []
for e in mined:
    try:
        i, p = load_raw(DS, CACHE, e)
    except Exception:
        continue
    IMG.append(i); PRP.append(p)
    E.append(np.full(len(i), e)); T.append(np.arange(len(i)) / max(1, len(i) - 1))
IMG = np.concatenate(IMG); PRP = np.concatenate(PRP); E = np.concatenate(E); T = np.concatenate(T)
MU, SD = PRP.mean(0), PRP.std(0) + 1e-8
def mkfeat(img, prp):
    p = (prp - MU) / SD; p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    i = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    return np.concatenate([i, p], 1).astype(np.float32)
F = mkfeat(IMG, PRP)
n_ep = len(set(E.tolist()))
km = KMeans(n_clusters=96, n_init=2, random_state=0).fit(F)
lab = km.labels_
cov = np.array([len(set(E[lab == c].tolist())) / n_ep for c in range(96)])
tpos = np.array([T[lab == c].mean() for c in range(96)])
ms = sorted(np.argsort(cov)[-20:].tolist(), key=lambda c: tpos[c])
def gated_runs_idx(idx_arr):
    runs = []; s = None; prev = None
    for i in idx_arr:
        if prev is None or i != prev + 1:
            if s is not None: runs.append((s, prev))
            s = i
        prev = i
    if s is not None: runs.append((s, prev))
    return [r for r in runs if r[1] - r[0] >= 1]
Pk, cyc = {}, {}
for c in ms:
    fe, nr, re_, vis, starts = [], [], 0, 0, []
    for e in set(E.tolist()):
        m = np.where(E == e)[0]
        rs = gated_runs_idx(m[lab[m] == c].tolist())
        if not rs: continue
        vis += 1; fe.append(T[rs[0][0]]); nr.append(len(rs)); starts += [T[r[0]] for r in rs]
        if len(rs) >= 2:
            for (a1, b1), (a2, b2) in zip(rs[:-1], rs[1:]):
                if any(x in ms and x != c for x in lab[[j for j in m if b1 < j < a2]]):
                    re_ += 1; break
    Pk[c] = float(np.median(fe)) if fe else tpos[c]
    v1 = np.mean(nr) > 1.5 if nr else False
    X = np.array(starts).reshape(-1, 1)
    v2 = False
    if len(X) >= 10:
        v2 = (GaussianMixture(1, random_state=0).fit(X).bic(X) -
              GaussianMixture(2, random_state=0).fit(X).bic(X)) > 10
    v3 = (re_ / max(1, vis)) > 0.2
    cyc[c] = int(v1) + int(v2) + int(v3) >= 2
anchors = sorted([c for c in ms if not cyc[c]], key=lambda c: Pk[c])
aset = set(anchors)

AUTO = REPO / "temp/autonomy"
img = np.load(REPO / "temp/tcc_autonomy_armmask/feat_cache/ep0.npz")["f"]
n = len(img)
st = np.stack(pd.read_parquet(AUTO / "data/chunk-000/episode_000000.parquet",
                              columns=["observation.state"])["observation.state"].to_numpy())
st = st[np.minimum(np.arange(n) * 10, len(st) - 1)]
fA = mkfeat(img, np.concatenate([st, np.vstack([np.zeros((1, 14)), np.diff(st, axis=0)])], 1))
D = np.linalg.norm(fA[:, None, :] - km.cluster_centers_[None], axis=2)
lA = D.argmin(1); ds_ = np.sort(D, axis=1); mg = ds_[:, 0] / ds_[:, 1]
DELTA = 0.15
vm = np.zeros(n); vr = np.zeros(n); cm = 0.0; cr = 0.0
seen = set(); events = []
j = 0
while j < n:
    c = lA[j]; k = j
    while k + 1 < n and lA[k + 1] == c:
        k += 1
    runlen = k - j + 1
    conf = runlen >= 2 or mg[j] <= 0.8
    if c in aset and conf:
        if c not in seen:
            seen.add(c); cm = max(cm, Pk[c]); cr = max(cr, Pk[c])
        else:
            cm = max(cm, Pk[c])
            if Pk[c] <= cr - DELTA and runlen >= 3:
                events.append((j, cr, Pk[c])); cr = Pk[c]
            else:
                cr = max(cr, Pk[c])
    vm[j:k + 1] = cm; vr[j:k + 1] = cr
    j = k + 1
print(f"[sync] events: {[(int(e[0]*10), f'{e[1]:.2f}->{e[2]:.2f}') for e in events]}")
vr30 = np.repeat(vr, 10); vm30 = np.repeat(vm, 10)

# ===== 渲染 =====
MAXSIDE = 480
def stream_cam(path):
    c = av.open(str(path))
    for f in c.decode(video=0):
        s = min(1.0, MAXSIDE / max(f.height, f.width))
        g = f.reformat(width=int(f.width*s)//2*2, height=int(f.height*s)//2*2, format="rgb24") if s < 1 else f
        yield g.to_ndarray(format="rgb24")
    c.close()
cam = AUTO / "videos/chunk-000/top_head/episode_000000.mp4"
frame0 = next(stream_cam(cam))
L = min(7676, n * 10)

fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.18)
ax_cam = fig.add_subplot(gs[0]); ax_cam.axis("off")
im_cam = ax_cam.imshow(frame0)
ttl = ax_cam.set_title("", fontsize=11)
ax_v = fig.add_subplot(gs[1])
x30 = np.arange(L)
ax_v.plot(x30, vm30[:L], "-", color="#d62728", lw=1.2, alpha=.8, label="V monotone (cummax)")
ax_v.plot(x30, vr30[:L], "-", color="#2ca02c", lw=2.0, label="V2.1 regression rule")
for j_, a, b in events:
    ax_v.axvline(j_ * 10, color="#2ca02c", ls="--", lw=1.0, alpha=.6)
    ax_v.annotate(f"DROP {a:.2f}→{b:.2f}", (j_ * 10, b), xytext=(j_ * 10 + 120, b - 0.1),
                  fontsize=9, color="#2ca02c", fontweight="bold")
cur = ax_v.axvline(0, color="gray", lw=1.4)
dot, = ax_v.plot([0], [vr30[0]], "o", ms=10, color="#2ca02c", mec="k")
ax_v.set_xlim(0, L); ax_v.set_ylim(-0.05, 1.08)
ax_v.set_xlabel("frame (30Hz)"); ax_v.set_ylabel("V")
ax_v.legend(fontsize=9, loc="lower right"); ax_v.grid(alpha=.3)
fig.suptitle("autonomy rollout (3-round folding, 2 human disturbances): V2.1 regression-rule value sync", fontsize=12)

def status(t):
    for j_, a, b in events:
        if abs(t - j_ * 10) < 60:
            return f"⚠ REGRESSION DETECTED: V {a:.2f} → {b:.2f}", "#c0392b"
    return f"frame {t}  V={vr30[min(t, L-1)]:.2f}", "#333333"

oc = av.open(OUT_MP4, mode="w"); stv = oc.add_stream("libx264", rate=30)
f0 = None
t = 0
for img_ in stream_cam(cam):
    if t >= L: break
    im_cam.set_data(img_)
    s, col = status(t)
    ttl.set_text(s); ttl.set_color(col)
    cur.set_xdata([t, t]); dot.set_data([t], [vr30[min(t, L - 1)]])
    fig.canvas.draw()
    arr = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    if f0 is None:
        H, W = arr.shape[:2]; H -= H % 2; W -= W % 2
        stv.width, stv.height, stv.pix_fmt = W, H, "yuv420p"; stv.options = {"crf": "21"}
        f0 = True
    vf = av.VideoFrame.from_ndarray(arr[:H, :W], format="rgb24")
    for pkt in stv.encode(vf): oc.mux(pkt)
    t += 1
    if t % 1000 == 0:
        print(f"  rendered {t}/{L}")
for pkt in stv.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {OUT_MP4} {W}x{H} {t}f")
