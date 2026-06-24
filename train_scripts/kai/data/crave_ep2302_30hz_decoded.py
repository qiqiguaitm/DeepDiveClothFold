"""ep2302 30Hz 解析(新 enc/dec):DINOv2-large 编码器 + small 解码器。
产出:① value 曲线(30Hz)② milestone 随时间变化 ③ 每个 milestone 用**簇中心解码图**表示。
全程 large-enc:挖矿/聚类/value 用 large 池化特征;milestone 代表 = 簇内 large patch-grid 平均 → small 解码器解码。
Run: kai0/.venv/bin/python train_scripts/kai/data/crave_ep2302_30hz_decoded.py [--mine-n 200] [--k 24]
输出: docs/visualization/cross_episode_recurrence_value/crave_ep2302_30hz_decoded.png
"""
import argparse, json, sys, time, os
import numpy as np, av, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crave_decoder_scale_ablation import (REPO, DS, ARM, RAW, cs, loadep, mkp, decode_images, encode_grids,
                                          train_dec, crop224, camp, RES, P, dev)
from crave_readout import smooth_monotone   # 报告用的 fps 标定平滑(移动平均+re-clip)
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value/centroid_decoder"
LARGE = "/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large"; DIM = 1024
# value bins 不再固定:bins = milestone 进度位置 Pord + 端点 → NB 随自适应 milestone 数自动定,value↔milestone 精确对应
def dpHB(emit, bins, lam):
    nb = len(bins); pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
    cost = np.full(nb, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, nb), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(nb), k]; bp[j] = k
    cost[nb - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return bins[path]


def med(a, w):
    h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


def _encode(out, frames_bgr, fps):
    """GPU NVENC 优先(h264_nvenc)→ libx264 → cv2 mp4v 回退。"""
    H, W = frames_bgr[0].shape[:2]
    for codec in ("h264_nvenc", "libx264"):
        try:
            cont = av.open(str(out), "w")
            st = cont.add_stream(codec, rate=fps); st.width, st.height, st.pix_fmt = W, H, "yuv420p"
            for f in frames_bgr:
                vf = av.VideoFrame.from_ndarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), format="rgb24")
                for pkt in st.encode(vf): cont.mux(pkt)
            for pkt in st.encode(): cont.mux(pkt)
            cont.close(); return codec
        except Exception:
            try: cont.close()
            except Exception: pass
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames_bgr: vw.write(f)
    vw.release(); return "mp4v"


def render_video(frames, v, ms_idx, proto, Pord, out, fps=30):
    """合成视频(充分用 CPU/GPU):多核并行 compose + GPU NVENC 编码。frames=RGB crop224。"""
    from concurrent.futures import ThreadPoolExecutor
    W, H = 1000, 540; px0, py0, pw, ph = 510, 40, 470, 250; n = len(v)
    panel0 = np.full((ph, pw, 3), 255, np.uint8)
    for p in Pord:
        y = int((1 - p) * (ph - 1)); cv2.line(panel0, (0, y), (pw - 1, y), (235, 235, 235), 1)
    pts = np.array([[int(i / n * (pw - 1)), int((1 - v[i]) * (ph - 1))] for i in range(n)], np.int32)
    cv2.polylines(panel0, [pts], False, (60, 160, 44), 2)
    thumbs = {m: cv2.cvtColor(cv2.resize(proto[m], (210, 210)), cv2.COLOR_RGB2BGR) for m in range(len(proto))}

    def compose(t):
        cv = np.full((H, W, 3), 255, np.uint8)
        cv[40:520, 15:495] = cv2.cvtColor(cv2.resize(frames[t], (480, 480)), cv2.COLOR_RGB2BGR)
        cv2.putText(cv, f"ep2302  frame {t}/{n}  (30Hz)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        p = panel0.copy(); x = int(t / n * (pw - 1)); yy = int((1 - v[t]) * (ph - 1))
        cv2.line(p, (x, 0), (x, ph - 1), (180, 180, 180), 1); cv2.circle(p, (x, yy), 5, (0, 0, 220), -1)
        cv[py0:py0 + ph, px0:px0 + pw] = p
        cv2.putText(cv, f"CRAVE value = {v[t]:.2f}", (px0, py0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 160, 44), 2, cv2.LINE_AA)
        m = int(ms_idx[t]); cv[py0 + ph + 30:py0 + ph + 240, px0:px0 + 210] = thumbs[m]
        cv2.putText(cv, f"milestone m{m}  P={Pord[m]:.2f}", (px0, py0 + ph + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 90, 160), 2, cv2.LINE_AA)
        cv2.putText(cv, "decoded cluster-center", (px0 + 220, py0 + ph + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 90, 160), 1, cv2.LINE_AA)
        return cv
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 8))) as ex:
        comp = list(ex.map(compose, range(n)))      # 多核并行合成, 保序
    codec = _encode(out, comp, fps)
    print(f"  video encoded via {codec}", flush=True)


def render_transition_3row(fr, eg, decf, proto, medoid, cross, uniq, Pord, out):
    """milestone 跳变帧四行对比:① 原始跳变帧 ② 该帧 encode→decode 重建 ③ 簇中心解码图 ④ 离簇心最近真实帧(medoid)。"""
    sel = list(uniq) if len(uniq) <= 16 else [uniq[i] for i in np.linspace(0, len(uniq) - 1, 16).round().astype(int)]
    NS = len(sel)
    recon = decf(np.stack([eg[cross[m]] for m in sel]).astype(np.float32))   # 批量解码跳变帧自身 grid
    rowimgs = [[cv2.resize(fr[cross[m]], (128, 128)) for m in sel],
               [recon[j] for j in range(NS)],
               [proto[m] for m in sel],
               [medoid[m] for m in sel]]
    labels = ["(1) original\ntransition frame", "(2) encode->decode\n(large + small dec)", "(3) decoded\ncluster-center", "(4) nearest real frame\nto center (medoid)"]
    fig, axes = plt.subplots(4, NS, figsize=(1.35 * NS, 5.8))
    for r in range(4):
        for j in range(NS):
            ax = axes[r, j]; ax.imshow(rowimgs[r][j]); ax.axis("off")
            if r == 0: ax.set_title(f"m{sel[j]} f{cross[sel[j]]}\nP={Pord[sel[j]]:.2f}", fontsize=7)
        axes[r, 0].set_ylabel(labels[r], fontsize=8.5, rotation=0, ha="right", va="center", labelpad=2)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values(): sp.set_visible(False)
    fig.suptitle("ep2302 milestone transition frames — (1) original / (2) encode→decode recon / (3) decoded centroid / (4) nearest real (medoid)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=125, bbox_inches="tight"); plt.close(fig)


def decode_all_frames(e):
    """解 ep 全部 30fps 帧 → crop224 list (顺序)。"""
    imgs = []
    c = av.open(str(camp(e)))
    for f in c.decode(video=0): imgs.append(crop224(f.to_ndarray(format="rgb24")))
    c.close(); return imgs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mine-n", type=int, default=200); ap.add_argument("--k", type=int, default=96, help="过聚类数 K0(milestone 由 Otsu 自适应选出)")
    a = ap.parse_args(); t0 = time.time(); EP = 2302

    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = [e for e in sorted(int(p.stem[2:]) for p in ARM.glob("ep*.npz")) if e in rawset and e != EP]
    mined = sorted(np.random.RandomState(0).permutation(all_eps)[:a.mine_n].tolist())
    # 挖矿帧(3Hz 缓存帧位置)→ 但用 large-enc 重抽 grid。复用 mv pool: 每 ep 全部 3Hz 帧
    pool_idx, E_, FR_, T_ = [], [], [], []
    for e in mined:
        _, _, _, n = loadep(e)
        for i in range(n): pool_idx.append((e, i)); E_.append(e); FR_.append(i * 10); T_.append(i / max(1, n - 1))
    E_ = np.array(E_); FR_ = np.array(FR_); T_ = np.array(T_)
    print(f"挖矿 {len(mined)} ep, {len(pool_idx)} 帧; large-enc 抽 grid ...", flush=True)
    # 并行解码 + large 编码(挖矿帧)
    fake_pool = np.arange(len(pool_idx))
    imgs224, valid = decode_images(fake_pool, E_, FR_, t0)
    imgs128 = np.stack([cv2.resize(imgs224[i], (RES, RES), interpolation=cv2.INTER_AREA) for i in range(len(imgs224))]).astype(np.uint8)
    grids = encode_grids(imgs224, valid, LARGE, DIM)                       # (Nf,1024,16,16)
    pooled = grids.reshape(len(grids), DIM, -1).mean(2).astype(np.float32)
    pooled /= (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)
    print(f"  grids {grids.shape}  ({time.time()-t0:.0f}s)", flush=True)

    # 聚类 + **自适应**选 milestone(CRAVE V2.4: 过聚类 KMeans-K0 → 覆盖率 → Otsu 自动阈值,milestone 数据自适应)
    def otsu(xs):
        s = np.sort(xs); best_t, best_v = s[0], -1
        for t in np.unique(s):
            lo, hi = xs[xs < t], xs[xs >= t]
            if len(lo) == 0 or len(hi) == 0: continue
            v = (len(lo) / len(xs)) * (len(hi) / len(xs)) * (lo.mean() - hi.mean()) ** 2
            if v > best_v: best_v, best_t = v, t
        return best_t
    K0 = a.k  # 过聚类数(默认 96)
    km = KMeans(K0, n_init=3, random_state=0).fit(pooled); lab = km.labels_; cen = km.cluster_centers_
    n_ep = len(mined)
    tpos = np.array([T_[lab == c].mean() for c in range(K0)])
    cov = np.array([len(set(E_[lab == c].tolist())) / n_ep for c in range(K0)])   # 覆盖率
    tau = otsu(cov); sel = [c for c in range(K0) if cov[c] >= tau]                 # Otsu 自动选 milestone
    sel = sorted(sel, key=lambda c: tpos[c])                                       # 按进度排
    print(f"自适应 milestone:过聚类 {K0} → Otsu τ(cov)={tau:.3f} → 选出 {len(sel)} 个", flush=True)
    Cp = cen[sel].copy(); Cp /= (np.linalg.norm(Cp, axis=1, keepdims=True) + 1e-9); Pord = tpos[sel]
    # 自适应 value bins:把 bins 放在 milestone 进度位置上(+端点 0/1)→ NB 随 milestone 数自动定
    bins = np.unique(np.concatenate([[0.0], Pord, [1.0]])).astype(np.float64)
    cb = [int(np.searchsorted(bins, p)) for p in Pord]
    print(f"自适应 value bins:NB={len(bins)}(= {len(sel)} milestone + 端点),value↔milestone 精确对应", flush=True)
    order = sel  # 兼容下游命名

    # 训 small 解码器(large grid → 128 图)
    print("训 small 解码器(large grid→img)...", flush=True)
    decf = train_dec(grids, imgs128, DIM, "small", 55)
    # 每个被选 milestone 的簇中心解码图(簇内 large grid 平均 → decode)
    proto = {}; medoid = {}
    for oi, c in enumerate(sel):
        mem = np.where(lab == c)[0]
        if len(mem):
            proto[oi] = decf(grids[mem].astype(np.float32).mean(0)[None])[0]
            ni = mem[np.argmin(np.linalg.norm(pooled[mem] - cen[c], axis=1))]   # 离簇心最近的真实挖矿帧
            medoid[oi] = imgs128[ni]
        else:
            proto[oi] = np.zeros((RES, RES, 3), np.uint8); medoid[oi] = np.zeros((RES, RES, 3), np.uint8)

    # ===== ep2302 30Hz =====
    print("ep2302 30Hz 抽帧 + large grid ...", flush=True)
    fr224 = decode_all_frames(EP); n30 = len(fr224)
    eg = encode_grids(np.array(fr224), np.ones(n30, bool), LARGE, DIM)
    ep_pooled = eg.reshape(n30, DIM, -1).mean(2).astype(np.float32); ep_pooled /= (np.linalg.norm(ep_pooled, axis=1, keepdims=True) + 1e-9)
    d = np.linalg.norm(ep_pooled[:, None] - Cp[None], axis=2); em = np.full((n30, len(bins)), 1e3)
    for ci in range(len(Cp)): em[:, cb[ci]] = np.minimum(em[:, cb[ci]], d[:, ci])
    v_dp = dpHB(em, bins, lam=80.0)                              # DP 阶梯(对齐 milestone bins)→ 定 milestone
    ms_idx = np.array([int(np.sum(Pord <= v_dp[t] + 1e-9)) - 1 for t in range(n30)]); ms_idx = np.clip(ms_idx, 0, len(order) - 1)
    v = smooth_monotone(v_dp, fps=30.0)                         # 参考报告:fps 标定移动平均 → 平滑连续 value(显示用)
    print(f"  ep2302 {n30} 帧, value(smooth) {v.min():.2f}→{v.max():.2f}, 访问 milestone {sorted(set(ms_idx.tolist()))}", flush=True)

    # 访问到的 milestone(按出现顺序)
    visited = []
    for m in ms_idx:
        if not visited or visited[-1] != m: visited.append(int(m))
    uniq = sorted(set(visited))
    cross = {m: int(np.argmax(ms_idx >= m)) for m in uniq}  # 首达帧

    # ===== 图:value 曲线 + milestone 变化 + 簇中心解码图 =====
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 0.7, 1.6], hspace=0.32)
    x = np.arange(n30)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(x, v, color="#2ca02c", lw=2, label="CRAVE value (large-enc, 30Hz)")
    for m in uniq: ax0.axhline(Pord[m], color="#ddd", ls=":", lw=.6)
    for m in uniq:
        cf = cross[m]; ax0.scatter([cf], [v[cf]], s=40, color="#d7191c", zorder=5)
    ax0.set_ylabel("value (0→1)"); ax0.set_xlim(0, n30); ax0.set_ylim(-0.03, 1.05); ax0.grid(alpha=.25)
    ax0.legend(loc="lower right"); ax0.set_title(f"ep2302 · 30Hz · {n30} frames · value curve (DINOv2-large milestones, K={a.k})")
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.step(x, ms_idx, where="post", color="#7b5aa6", lw=1.5); ax1.set_ylabel("milestone idx"); ax1.set_xlabel("frame (30Hz)"); ax1.grid(alpha=.25)
    ax1.set_title("milestone change over time")
    # 簇中心解码图 strip(访问到的 milestone, 按进度排,均匀铺开 + 引线到首达帧)
    ax2 = fig.add_subplot(gs[2]); ax2.axis("off")
    NS = len(uniq); slots = np.linspace(0.03, 0.97, NS)
    for slot, m in zip(slots, uniq):
        axin = ax2.inset_axes([slot - 0.45 / NS, 0.30, 0.9 / NS, 0.62])
        axin.imshow(proto[m]); axin.axis("off"); axin.set_title(f"m{m} P={Pord[m]:.2f}\nf{cross[m]}", fontsize=7)
    ax2.set_title("each milestone = decoded cluster-center image (large-enc grid avg → small decoder)", fontsize=10, loc="left")
    fig.suptitle("ep2302 30Hz CRAVE analysis — value + milestones (milestones shown as DECODED cluster-center images)", fontsize=13, y=0.995)
    fig.savefig(OUTV / "crave_ep2302_30hz_decoded.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED crave_ep2302_30hz_decoded.png  total {time.time()-t0:.0f}s", flush=True)
    # 存 ground-truth bundle 供逐帧验证
    np.savez(REPO / "temp/crave_a1a2/ep2302_bundle.npz", v=v.astype(np.float32), ms_idx=ms_idx.astype(np.int32),
             Pord=Pord.astype(np.float32), proto=np.array([proto[i] for i in range(len(order))], np.uint8), NB=len(bins), bins=bins.astype(np.float32))
    print("SAVED ep2302_bundle.npz", flush=True)
    render_transition_3row(fr224, eg, decf, proto, medoid, cross, uniq, Pord, OUTV / "crave_ep2302_transition_3row.png")
    print("SAVED crave_ep2302_transition_3row.png", flush=True)
    print("渲染视频 ...", flush=True)
    render_video(fr224, v, ms_idx, proto, Pord, OUTV / "crave_ep2302_30hz_decoded.mp4", fps=30)
    print(f"SAVED crave_ep2302_30hz_decoded.mp4  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
