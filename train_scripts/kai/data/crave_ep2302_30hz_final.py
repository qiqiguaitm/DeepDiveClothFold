"""最终方法 ep2302 30Hz 全套: image⊕proprio 聚类 + precedence/isotonic + Wan medoid + 解耦 status flag。
输出文件夹 temp/crave_interp_ep2302_30hz_final/:
  01_value_milestone.png  主分析(value+milestone时间线+Wan medoid条)
  02_truncation_validate.png  裁剪验证(截30/50/70/90/100% → value停在真实进度 + is_complete flag)
  03_visitation_validate.png  到访验证(逐帧 value/milestone到访/完成残差de_end/ood)
  ep2302_fwd_vs_reverse_30hz.mp4  正放裁剪 vs 倒放 对齐(各带value游标+完成状态)
跑法: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_ep2302_30hz_final.py
"""
import sys, glob, time, subprocess, tempfile, os
sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
import numpy as np, cv2, torch, pandas as pd
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from crave_decoder_scale_ablation import REPO, DS, cs, grab_ep, encode_grids, train_dec
from crave_ep2302_30hz_decoded import decode_all_frames
from crave_readout import smooth_monotone
LARGE = "/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large"; DIM = 1024
OUTD = REPO / "temp/crave_full"; WAN = "checkpoints/Wan2.2-TI2V-5B-Diffusers"; dev = "cuda"; ENC = "dino"; EP = 2302
OUT = REPO / "temp/crave_interp_ep2302_30hz_final"; OUT.mkdir(exist_ok=True)


def otsu(xs):
    s = np.unique(np.sort(xs)); bt, bv = s[0], -1
    for t in s:
        lo, hi = xs[xs < t], xs[xs >= t]
        if len(lo) and len(hi):
            v = (len(lo) / len(xs)) * (len(hi) / len(xs)) * (lo.mean() - hi.mean()) ** 2
            if v > bv: bv, bt = v, t
    return bt


def med(a, w):
    h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


def viterbi(emit, bins, lam, eb=2.0):
    NB = len(bins); pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
    cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= eb; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return bins[path]


def viterbi_rev(emit, bins, lam, eb=2.0):                      # 倒放DP: 起帧自由(不强锚折好态→1), 末帧锚 平摊→value0(bin0)
    NB = len(bins); pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
    cost = emit[0].copy().astype(float); bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[0] -= eb; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return bins[path]


def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def gpu_kmeans(F, K, iters=25, chunk=30000, seed=0):
    g = torch.device("cuda"); Fg = torch.from_numpy(np.ascontiguousarray(F, np.float32)).to(g); Nn = Fg.shape[0]
    torch.manual_seed(seed); C = Fg[torch.randperm(Nn, device=g)[:K]].clone(); lab = torch.zeros(Nn, dtype=torch.long, device=g)
    for it in range(iters):
        Cn = (C * C).sum(1)
        for s in range(0, Nn, chunk):
            Fc = Fg[s:s + chunk]; lab[s:s + chunk] = ((Fc * Fc).sum(1, keepdim=True) - 2 * Fc @ C.T + Cn[None]).argmin(1)
        nc = torch.zeros_like(C); cnt = torch.zeros(K, device=g)
        nc.index_add_(0, lab, Fg); cnt.index_add_(0, lab, torch.ones(Nn, device=g)); msk = cnt > 0; C[msk] = nc[msk] / cnt[msk, None]
    return C.cpu().numpy(), lab.cpu().numpy()


def main():
    t0 = time.time()
    zf = np.load(OUTD / f"index_{ENC}.npz"); E, FR, T, N = zf["E"], zf["FR"], zf["T"], int(zf["n"])
    feat = np.zeros((N, 1024), np.float16); valid = np.zeros(N, bool)
    for f in sorted(glob.glob(str(OUTD / ENC / "shard_*.npz"))):
        z = np.load(f); feat[z["gidx"]] = z["feat"]; valid[z["gidx"]] = z["valid"]
    vi = np.where(valid)[0]; img = feat[vi].astype(np.float32); img /= (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    Ev, FRv, Tv = E[vi], FR[vi], T[vi]; ep_list = sorted(set(Ev.tolist()))
    P = np.zeros((len(vi), 28), np.float32); raw_state = {}
    for e in ep_list:
        loc = np.where(Ev == e)[0]; o = np.argsort(FRv[loc]); loc = loc[o]
        st = np.stack(pd.read_parquet(DS / "data" / f"chunk-{e//cs:03d}" / f"episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
        if e == EP: raw_state[e] = st
        P[loc] = mkp(st[np.minimum(FRv[loc], len(st) - 1)])
    PMU, PSD = P.mean(0), P.std(0) + 1e-8; Pn = (P - PMU) / PSD; Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9)
    F = np.concatenate([img, Pn], 1); K0 = int(np.clip(round(0.55 * np.sqrt(len(vi))), 96, 320))
    print(f"GPU KMeans K0={K0} ...", flush=True); cen, lab = gpu_kmeans(F, K0); ne = len(ep_list)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(Ev[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel)
    fe = np.full((ne, M), np.nan)
    for ei, e in enumerate(ep_list):
        fi = np.where(Ev == e)[0]; labe = lab[fi]; te = Tv[fi]
        for m in range(M):
            hit = te[labe == sel[m]]
            if len(hit): fe[ei, m] = hit.min()
    Pk = np.array([np.nanmedian(fe[:, m]) for m in range(M)])
    Pbef = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i != j:
                both = np.isfinite(fe[:, i]) & np.isfinite(fe[:, j])
                if both.sum() >= 5: Pbef[i, j] = float(np.mean(fe[both, i] < fe[both, j]))
    soft = np.nansum(np.where(np.isnan(Pbef), 0.0, Pbef), 1); prec = list(np.argsort(-soft))
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(M), Pk[prec]); Pord = np.asarray(iso, float)
    order = [sel[p] for p in prec]; C = cen[order]
    from sklearn.cluster import KMeans
    SP = np.concatenate([F[np.where(Ev == e)[0][np.argsort(Tv[np.where(Ev == e)[0]])][:2]] for e in ep_list])
    EPp = np.concatenate([F[np.where(Ev == e)[0][np.argsort(Tv[np.where(Ev == e)[0]])][-2:]] for e in ep_list])
    startK = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_; endK = KMeans(8, n_init=2, random_state=0).fit(EPp).cluster_centers_
    de_tr = np.array([float(np.linalg.norm(F[np.where(Ev == e)[0][np.argmax(Tv[np.where(Ev == e)[0]])]][None] - endK, axis=1).min()) for e in ep_list])
    de_thr = float(np.quantile(de_tr, 0.90)) * 1.3
    print(f"全量 {ne}ep → {M} milestones, de_end_thr={de_thr:.2f}", flush=True)
    bins = np.linspace(0, 1, 41); cb = [int(np.argmin(abs(bins - Pord[m]))) for m in range(M)]

    def readout(Fq):                                            # → value, de_end(每帧到endK), ood(每帧到最近milestone)
        nn = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nn, 41), 1e3)
        for m in range(M): em[:, cb[m]] = np.minimum(em[:, cb[m]], d[:, m])
        ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1); tx = np.arange(nn) / nn
        em[:, 0] = np.minimum(em[:, 0], np.where(tx < 0.3, ds, ds + (tx - 0.3) * 6))   # 只锚起始 平摊→value0; 不锚完成(不强拉→1)
        v = smooth_monotone(med(viterbi(em, bins, 8.0, 0.0), 5), fps=3.0)              # eb=0 去 end_bonus
        return v, de, d.min(1)

    # ---- ep2302 3Hz 正向 ----
    fi = np.where(Ev == EP)[0]; oo = np.argsort(FRv[fi]); fi = fi[oo]; Fq = F[fi]; n3 = len(fi); fr3 = FRv[fi]
    v3, de3, ood3 = readout(Fq)
    ms3 = np.array([int(np.argmin(np.abs(Pord - v3[t]))) for t in range(n3)])
    is_comp = de3[-3:].min() <= de_thr
    print(f"正向: {n3}帧 value {v3.min():.2f}→{v3.max():.2f}, is_complete={is_comp}", flush=True)

    # ---- 逆序(倒放视频) DP 平滑: 图像+proprio 按倒序 reorder(forward Δ 正确配对), 翻转anchor(起=折好value1, 末=平摊value0)
    #      在倒放特征上跑 reverse-DP(viterbi_rev) → 平滑 1→0(仍是倒放轨迹独立计算, 非翻转正向曲线)
    rev = list(range(n3 - 1, -1, -1)); Fq_rev = Fq[rev]
    dR = np.linalg.norm(Fq_rev[:, None] - C[None], axis=2); emR = np.full((n3, 41), 1e3)
    for m in range(M): emR[:, cb[m]] = np.minimum(emR[:, cb[m]], dR[:, m])
    dsR = np.linalg.norm(Fq_rev[:, None] - startK[None], axis=2).min(1)
    deR = np.linalg.norm(Fq_rev[:, None] - endK[None], axis=2).min(1); txR = np.arange(n3) / n3
    emR[:, 0] = np.minimum(emR[:, 0], np.where(txR > 0.6, dsR, dsR + (0.6 - txR) * 6))      # 只锚末→平摊态(value0); 不锚折好→1
    v3_rev = smooth_monotone(med(viterbi_rev(emR, bins, 8.0), 5)[::-1], fps=3.0)[::-1]      # 单调下降平滑(反向单调化)
    print(f"逆序(倒放,DP): value {v3_rev[:5].mean():.2f}(起,应~1) → {v3_rev[-5:].mean():.2f}(末,应~0)", flush=True)

    # ---- 簇中心解码图: 簇内成员 large patch-grid 平均(=簇中心) → small 解码器解码(合成质心, 非 medoid) ----
    print("簇中心解码: 抽成员 grid + 训 small 解码器 ...", flush=True)
    NSAMP = 24; samp = []; rng = []
    for mi, c in enumerate(order):
        loc = np.where(lab == c)[0]
        if len(loc) > NSAMP: loc = loc[np.linspace(0, len(loc) - 1, NSAMP).astype(int)]
        s0 = len(samp)
        for gi in vi[loc]:
            im = grab_ep(int(E[gi]), [int(FR[gi])]).get(int(FR[gi]))
            if im is not None: samp.append(cv2.resize(im, (224, 224)))
        rng.append((s0, len(samp)))
    samp = np.stack(samp)
    grids = encode_grids(samp, np.ones(len(samp), bool), LARGE, DIM)        # (Ns,1024,16,16) large patch-grid
    imgs128 = np.stack([cv2.resize(im, (128, 128)) for im in samp])
    decf = train_dec(grids, imgs128, DIM, "small", 55)                      # small 解码器: grid → 128 图
    proto = {}; medoid = {}
    for mi, c in enumerate(order):                                          # 簇中心解码 + 最近真实帧(medoid)做参照
        s0, e0 = rng[mi]
        proto[mi] = cv2.resize(decf(grids[s0:e0].mean(0)[None])[0], (96, 96)) if e0 > s0 else np.zeros((96, 96, 3), np.uint8)
        loc = np.where(lab == c)[0]; gi = vi[loc[int(np.argmin(np.linalg.norm(F[loc] - cen[c], axis=1)))]]
        im = grab_ep(int(E[gi]), [int(FR[gi])]).get(int(FR[gi]))
        medoid[mi] = cv2.resize(im, (96, 96)) if im is not None else np.zeros((96, 96, 3), np.uint8)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # ===== 01 主分析 =====
    fig = plt.figure(figsize=(15, 7)); gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.3, 2])
    ax1 = fig.add_subplot(gs[0]); ax1.plot(v3, color="#1a7f37", lw=2, label="CRAVE value (img⊕proprio, precedence+isotonic)")
    for p in Pord: ax1.axhline(p, color="0.92", lw=.5)
    ax1.set_ylim(-.02, 1.02); ax1.set_ylabel("value"); ax1.legend(fontsize=9); ax1.set_title(f"ep2302 30Hz 最终方法 — {M} milestones — value {v3.min():.2f}→{v3.max():.2f}, is_complete={is_comp}")
    ax2 = fig.add_subplot(gs[1]); ax2.step(range(n3), ms3, where="post", color="#9c27b0"); ax2.set_ylabel("milestone"); ax2.set_xlabel("frame(3Hz)"); ax2.set_title("milestone 到访", fontsize=9)
    nsh = min(16, M); shi = [int(x) for x in np.linspace(0, M - 1, nsh).round()]
    for k, mi in enumerate(shi):
        axp = fig.add_subplot(gs[2].subgridspec(1, nsh)[0, k]); axp.imshow(proto[mi]); axp.axis("off"); axp.set_title(f"m{mi}\n{Pord[mi]:.2f}", fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "01_value_milestone.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ===== 02 裁剪验证 =====
    fig, ax = plt.subplots(figsize=(11, 5)); cols = {0.3: "#d62728", 0.5: "#e45756", 0.7: "#f0a020", 0.9: "#4c78a8", 1.0: "#1a7f37"}
    for fr in [0.3, 0.5, 0.7, 0.9, 1.0]:
        k = max(3, int(n3 * fr)); vk, dek, _ = readout(Fq[:k]); comp = dek[-3:].min() <= de_thr
        conf = float(np.clip((1.2 * de_thr - dek[-3:].min()) / (0.4 * de_thr), 0, 1))
        ax.plot(np.linspace(0, fr, k), vk, color=cols[fr], lw=2, label=f"截{int(fr*100)}%: 末value={vk[-2:].mean():.2f} | {'完成' if comp else '未完成'}(conf={conf:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.4, label="参照 value=进度")
    ax.set_xlim(0, 1.02); ax.set_ylim(-.02, 1.02); ax.set_xlabel("截断进度"); ax.set_ylabel("value"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("裁剪验证: value 停在真实进度(贴合完整曲线, 不被拉到1.0) + 完成flag 正确判未完成")
    fig.tight_layout(); fig.savefig(OUT / "02_truncation_validate.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ===== 03 到访验证(逐帧 value/milestone/完成残差/ood)=====
    fig, axs = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    axs[0].plot(v3, color="#1a7f37", lw=2); axs[0].set_ylabel("value"); axs[0].set_ylim(-.02, 1.02); axs[0].grid(alpha=.3); axs[0].set_title("ep2302 到访验证: 上=value 中=milestone到访 下=完成残差de_end / OOD(越低越在轨)")
    axs[1].step(range(n3), ms3, where="post", color="#9c27b0"); axs[1].set_ylabel("milestone idx"); axs[1].grid(alpha=.3)
    axs[2].plot(de3, color="#d62728", lw=1.5, label="de_end(到完成态endK)"); axs[2].plot(ood3, color="#4c78a8", lw=1, alpha=.7, label="ood(到最近milestone)")
    axs[2].axhline(de_thr, color="k", ls="--", lw=1, label=f"完成阈 thr={de_thr:.2f}"); axs[2].set_ylabel("residual"); axs[2].set_xlabel("frame(3Hz)"); axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "03_visitation_validate.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ===== 04 簇中心解码 vs 最近真实帧(medoid) 参照, 按 value 顺序 =====
    PR = 18; NBd = (M + PR - 1) // PR
    fig, axes = plt.subplots(2 * NBd, PR, figsize=(PR * 0.82, 2 * NBd * 1.18)); axes = np.atleast_2d(axes)
    for ax in axes.ravel(): ax.axis("off")
    for mi in range(M):
        b, col = mi // PR, mi % PR
        axes[2 * b, col].imshow(proto[mi]); axes[2 * b, col].set_title(f"m{mi} v={Pord[mi]:.2f}", fontsize=6)
        axes[2 * b + 1, col].imshow(medoid[mi])
    for b in range(NBd):
        axes[2 * b, 0].set_ylabel("centroid", fontsize=7, rotation=90); axes[2 * b, 0].axis("on"); axes[2 * b, 0].set_xticks([]); axes[2 * b, 0].set_yticks([])
        axes[2 * b + 1, 0].set_ylabel("medoid", fontsize=7, rotation=90); axes[2 * b + 1, 0].axis("on"); axes[2 * b + 1, 0].set_xticks([]); axes[2 * b + 1, 0].set_yticks([])
    fig.suptitle("Milestones in value order — top: decoded cluster centroid | bottom: nearest real frame (medoid)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "04_centroid_vs_medoid.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    print("SAVED 01/02/03/04 png", flush=True)

    # ===== mp4: 单视频(正向相机) + 三条 value 曲线 同步游标 =====
    print("解 30Hz 帧 ...", flush=True); frames = decode_all_frames(EP); n30 = len(frames)
    xi = np.linspace(0, 1, n3); xo = np.linspace(0, 1, n30)
    v_fwd = np.interp(xo, xi, v3)                              # ① 正序 value
    v_revflip = np.interp(xo, xi, v3_rev)                     # ② 逆序 value(倒放视频→现聚类匹配, 验证应 1→0)
    # ③ 裁剪: 先把视频裁到 frc, 对裁剪后的视频重新评估 value(不是评估完整再截), 裁掉部分保持 0
    frc = 0.5; k3 = max(3, int(n3 * frc)); kcut = int(n30 * frc)
    vt3 = readout(Fq[:k3])[0]
    v_trunc = np.zeros(n30); v_trunc[:kcut] = np.interp(np.linspace(0, 1, kcut), np.linspace(0, 1, k3), vt3)
    ms30 = ms3[np.clip((xo * (n3 - 1)).round().astype(int), 0, n3 - 1)]   # 每30Hz帧的milestone(随簇跳变)
    # 预渲三子图背景(静态), 逐帧只画游标(快)
    fig2, axs = plt.subplots(3, 1, figsize=(7.8, 6.4))
    curves = [(v_fwd, "(1) Forward value  (start-anchored only, no completion pull to 1)", "#1a7f37"),
              (v_revflip, "(2) Reverse value  (reversed video, reverse-DP, swept RIGHT->LEFT, 1 -> 0)", "#d62728"),
              (v_trunc, f"(3) Truncated value  (cut video at {int(frc*100)}% then re-evaluate, 0 after)", "#1f77b4")]
    for ax, (cv, ttl, col) in zip(axs, curves):
        ax.plot(np.arange(n30), cv, color=col, lw=2); ax.set_ylim(-.03, 1.03); ax.set_ylabel("value", fontsize=8)
        ax.set_title(ttl, fontsize=11, loc="left"); ax.grid(alpha=.3)
    axs[0].set_xlim(0, n30 - 1); axs[1].set_xlim(0, n30 - 1); axs[2].set_xlim(0, n30 - 1)   # ②曲线 1→0(左折好,右平摊); 游标右→左(mp4起始帧=倒放终止帧)
    axs[-1].set_xlabel("frame (30Hz, synced to video)", fontsize=8); fig2.tight_layout()
    fig2.canvas.draw(); Wpx, Hpx = fig2.canvas.get_width_height()
    bg = np.frombuffer(fig2.canvas.buffer_rgba(), np.uint8).reshape(Hpx, Wpx, 4)[..., :3][..., ::-1].copy()
    boxes = [(b.x0, b.x1, Hpx - b.y1, Hpx - b.y0) for b in (ax.get_window_extent() for ax in axs)]
    plt.close(fig2)
    Wl = 360; FONT = cv2.FONT_HERSHEY_SIMPLEX; td = tempfile.mkdtemp()
    for t in range(n30):
        cur = bg.copy(); frac = t / max(1, n30 - 1)
        for bi, (x0, x1, yt, yb) in enumerate(boxes):
            px = int(x1 - frac * (x1 - x0)) if bi == 1 else int(x0 + frac * (x1 - x0))   # ②游标右→左
            cv2.line(cur, (px, int(yt)), (px, int(yb)), (30, 30, 30), 1)
        cam = cv2.resize(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR), (Wl, 280))    # 相机
        mid = int(ms30[t]); medimg = cv2.resize(cv2.cvtColor(proto[mid], cv2.COLOR_RGB2BGR), (Wl, 300))  # 当前簇中心 Wan 解码图
        lbl = np.full((42, Wl, 3), 28, np.uint8)
        cv2.putText(lbl, f"milestone m{mid}  value={v_fwd[t]:.2f}", (8, 28), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        left = np.vstack([cam, lbl, medimg]); Hl = left.shape[0]
        curR = cv2.resize(cur, (int(cur.shape[1] * Hl / cur.shape[0]), Hl))
        cv2.imwrite(f"{td}/{t:05d}.png", np.hstack([left, curR]))
    mp4 = OUT / "ep2302_value_aligned_30hz.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", f"{td}/%05d.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in glob.glob(f"{td}/*.png"): os.remove(f)
    old = OUT / "ep2302_fwd_vs_reverse_30hz.mp4"
    if old.exists(): old.unlink()
    print(f"SAVED {mp4.name}  total {time.time()-t0:.0f}s", flush=True)
    print("DIR:", OUT, flush=True)


if __name__ == "__main__":
    main()
