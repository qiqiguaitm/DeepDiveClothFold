"""泛化验证: 当前架构(DINOv2-large 图像⊕proprio 聚类 + precedence/isotonic + 簇中心解码) 在 VIS/XVLA/coffee 全量聚类,
每数据集抽2个偏长 ep, 渲染 图像+forward value 视频, 配簇中心解码图。
跑法: HF_HUB_OFFLINE=1 .venv_wanvae/bin/python train_scripts/kai/data/crave_generalize.py <vis|xvla|coffee>
"""
import sys, glob, time, json, subprocess, tempfile, os
from pathlib import Path
try:                                                          # 仅在缺依赖时回退 kai0 venv(srpo py3.10 不能借 py3.11 包,会 shadow transformers)
    import numpy as np, cv2, torch, pandas as pd
except ImportError:
    sys.path.append("/vePFS/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages")
    import numpy as np, cv2, torch, pandas as pd
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/train_scripts/kai/data")
from crave_decoder_scale_ablation import REPO, encode_grids, train_dec
from crave_readout import smooth_monotone
dev = "cuda"; P = 16
# ============ encoder selection (env CRAVE_ENC: dinov2 | dinov3h | dinov3_7b_int8 | dinov3_7b) ============
# 全部在 256/224 下出 16x16 grid (decoder 固定 16->128); dinov3 必须 bf16(fp16 溢出 NaN), prefix=1CLS+4reg=5
_ENCODERS = {
    "dinov2":         dict(path="/vePFS/xiezhicong/.cache/huggingface/hub/dinov2-large", dim=1024, nprefix=1, dtype="fp16", res=224),
    "dinov3h":        dict(path="/vePFS/xiezhicong/.cache/huggingface/hub/dinov3-vith16plus-pretrain-lvd1689m", dim=1280, nprefix=5, dtype="bf16", res=256),
    "dinov3_7b_int8": dict(path=str(REPO / "temp/dinov3_7b_int8"), dim=4096, nprefix=5, dtype="int8", res=256),
    "dinov3_7b":      dict(path=str(REPO / "temp/dinov3_7b"), dim=4096, nprefix=5, dtype="bf16", res=256),
}
_ENC = os.environ.get("CRAVE_ENC", "dinov2"); _EC = _ENCODERS[_ENC]
LARGE = _EC["path"]; DIM = _EC["dim"]; NPREFIX = _EC["nprefix"]; ENC_DTYPE = _EC["dtype"]; ENC_RES = _EC["res"]
print(f"[encoder] {_ENC}: {LARGE} dim={DIM} nprefix={NPREFIX} dtype={ENC_DTYPE} res={ENC_RES}", flush=True)
OUTB = REPO / f"temp/crave_generalize{'' if _ENC == 'dinov2' else '_' + _ENC}"

CFG = {
    "vis": dict(kind="lerobot2", root="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/pure_vis600/base",
                cam="observation.images.top_head", stride=10, maxep=560),
    "xvla": dict(kind="hdf5", root="/vePFS/tim/workspace/deepdive_kai0/xvla/data/xvla_soft_fold/0707_11pm_stage_1_stage2new_new_cam_very_slow",
                 cam="cam_high", stride=10, maxep=168),
    "coffee": dict(kind="lerobotv3", root="/vePFS/tim/workspce/hf_cache/hub_default/datasets--lerobot--aloha_static_coffee/snapshots/b144896feb1f37398a862927b22cd3abdf005a6b",
                   cam="observation.images.cam_high", stride=16, maxep=50,
                   statecache=str(REPO / "temp/generalization_value_eval/coffee/feat_cache")),
}


def mkp(s): return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def mkp_gap(s, g):                                             # Δ 用 g 帧间隔(30Hz读出时匹配3Hz聚类的Δ尺度)
    d = np.zeros_like(s); d[g:] = s[g:] - s[:-g]
    return np.concatenate([s, d], 1)


def med(a, w):
    h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


def otsu(xs):
    s = np.unique(np.sort(xs)); bt, bv = s[0], -1
    for t in s:
        lo, hi = xs[xs < t], xs[xs >= t]
        if len(lo) and len(hi):
            v = (len(lo) / len(xs)) * (len(hi) / len(xs)) * (lo.mean() - hi.mean()) ** 2
            if v > bv: bv, bt = v, t
    return bt


def viterbi(emit, bins, lam, eb=0.0):
    NB = len(bins); pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
    cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= eb; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return bins[path]


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


# ============ DINOv2-large pooled encoder (load once) ============
def make_enc():
    from transformers import AutoImageProcessor, AutoModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    proc = AutoImageProcessor.from_pretrained(LARGE)
    if ENC_DTYPE == "int8":
        from transformers import BitsAndBytesConfig
        enc = AutoModel.from_pretrained(LARGE, quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()  # bnb 自动放 GPU
    elif ENC_DTYPE == "bf16":
        enc = AutoModel.from_pretrained(LARGE, torch_dtype=torch.bfloat16).to(dev).eval()                          # dinov3: fp16 溢出, 必须 bf16
    else:
        enc = AutoModel.from_pretrained(LARGE).half().to(dev).eval()                                              # dinov2: fp16 加速
    return proc, enc


def enc_pooled(imgs224, proc, enc, bs=128):                    # 大 batch GPU 加速; dinov3 用 bf16, res=256→16x16
    out = np.zeros((len(imgs224), DIM), np.float32)
    tdt = torch.float16 if ENC_DTYPE == "fp16" else torch.bfloat16   # int8/bf16 输入走 bf16
    sz = {"height": ENC_RES, "width": ENC_RES}
    for b in range(0, len(imgs224), bs):
        with torch.no_grad():
            px = proc(images=imgs224[b:b + bs], return_tensors="pt", size=sz).to(dev)
            px = {k: (v.to(tdt) if torch.is_floating_point(v) else v) for k, v in px.items()}
            out[b:b + bs] = enc(**px).last_hidden_state[:, NPREFIX:].mean(1).float().cpu().numpy()
    return out


# ============ per-dataset loaders → (frames224_rgb, state(n,14), thumb128, native_idx) at ~3Hz ============
def load_ep(ds, cfg, e, strd=None):
    st = strd if strd is not None else cfg["stride"]
    if cfg["kind"] == "lerobot2":
        root = Path(cfg["root"])
        df = pd.read_parquet(root / "data/chunk-000" / f"episode_{e:06d}.parquet", columns=["observation.state"])
        state_full = np.stack(df["observation.state"].to_numpy())
        cap = cv2.VideoCapture(str(root / "videos/chunk-000" / cfg["cam"] / f"episode_{e:06d}.mp4"))
        frames = []; i = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % st == 0: frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
    elif cfg["kind"] == "hdf5":
        import h5py
        with h5py.File(Path(cfg["root"]) / f"episode_{e}.hdf5", "r") as h:
            state_full = h["observations/qpos"][:]
            jpg = h["observations/images/" + cfg["cam"]]
            frames = [cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg[i], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for i in range(0, len(state_full), st)]
    elif cfg["kind"] == "lerobotv3":
        import av
        epm = pd.read_parquet(Path(cfg["root"]) / "meta/episodes/chunk-000/file-000.parquet")
        row = epm[epm["episode_index"] == e].iloc[0]; f0, f1 = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        s3 = np.load(Path(cfg["statecache"]) / f"ep{e}.npz")["state"]            # feat_cache 3Hz state
        cont = av.open(str(Path(cfg["root"]) / "videos" / cfg["cam"] / "chunk-000/file-000.mp4")); frames = []
        for gi, fr in enumerate(cont.decode(video=0)):
            if gi >= f1: break
            if gi >= f0 and (gi - f0) % st == 0: frames.append(fr.to_ndarray(format="rgb24"))
        cont.close(); n = len(frames)
        xs = np.linspace(0, 1, len(s3)); xo = np.linspace(0, 1, n)                # 插值到当前帧率
        state = np.stack([np.interp(xo, xs, s3[:, j]) for j in range(s3.shape[1])], 1)
        f224 = [cv2.resize(f, (224, 224)) for f in frames]; th = [cv2.resize(f, (128, 128)) for f in frames]
        return f224, state, th, np.arange(n)
    idx = np.arange(0, len(state_full), st)
    state = state_full[idx]; n = min(len(frames), len(state)); frames = frames[:n]; state = state[:n]
    f224 = [cv2.resize(f, (224, 224)) for f in frames]; th = [cv2.resize(f, (128, 128)) for f in frames]
    return f224, state, th, idx[:n]


def list_eps(ds, cfg):
    if cfg["kind"] == "lerobot2": eps = sorted(int(p.stem[8:]) for p in (Path(cfg["root"]) / "data/chunk-000").glob("episode_*.parquet"))
    elif cfg["kind"] == "hdf5": eps = sorted(int(p.stem.split("_")[1]) for p in Path(cfg["root"]).glob("episode_*.hdf5"))
    elif cfg["kind"] == "lerobotv3": eps = sorted(int(p.stem[2:]) for p in Path(cfg["statecache"]).glob("ep*.npz"))
    return eps[:cfg["maxep"]]


def load_ep_native(ds, cfg, e):                               # 原生帧率: 所有帧 + 原生 state(coffee 用3Hz插值)
    if cfg["kind"] == "lerobot2":
        root = Path(cfg["root"])
        state = np.stack(pd.read_parquet(root / "data/chunk-000" / f"episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
        cap = cv2.VideoCapture(str(root / "videos/chunk-000" / cfg["cam"] / f"episode_{e:06d}.mp4")); frames = []
        while True:
            ok, fr = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release(); fps = 30
    elif cfg["kind"] == "hdf5":
        import h5py
        with h5py.File(Path(cfg["root"]) / f"episode_{e}.hdf5", "r") as h:
            state = h["observations/qpos"][:]; jpg = h["observations/images/" + cfg["cam"]]
            frames = [cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg[i], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for i in range(len(state))]
        fps = 30
    elif cfg["kind"] == "lerobotv3":
        import av
        epm = pd.read_parquet(Path(cfg["root"]) / "meta/episodes/chunk-000/file-000.parquet")
        row = epm[epm["episode_index"] == e].iloc[0]; f0, f1 = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        cont = av.open(str(Path(cfg["root"]) / "videos" / cfg["cam"] / "chunk-000/file-000.mp4")); frames = []
        for gi, fr in enumerate(cont.decode(video=0)):
            if gi >= f1: break
            if gi >= f0: frames.append(fr.to_ndarray(format="rgb24"))
        cont.close()
        s3 = np.load(Path(cfg["statecache"]) / f"ep{e}.npz")["state"]; xs = np.linspace(0, 1, len(s3)); xo = np.linspace(0, 1, len(frames))
        state = np.stack([np.interp(xo, xs, s3[:, j]) for j in range(s3.shape[1])], 1); fps = 50
    n = min(len(frames), len(state)); frames = frames[:n]; state = state[:n]
    return [cv2.resize(f, (224, 224)) for f in frames], state, fps


def main(ds):
    t0 = time.time(); cfg = CFG[ds]; OUT = OUTB / ds; OUT.mkdir(parents=True, exist_ok=True)
    eps = list_eps(ds, cfg); print(f"[{ds}] {len(eps)} eps, 全帧数抽特征(DINOv2-large pooled⊕proprio, vel Δ gap={cfg['stride']})...", flush=True)
    proc, enc = make_enc()
    POOL, STATE, EPID, TPOS, THUMB, NIDX, eplen = [], [], [], [], [], [], {}
    for k, e in enumerate(eps):
        try: f224, state, th, nidx = load_ep(ds, cfg, e, strd=1)          # 全帧数(30Hz/原生)聚类
        except Exception as ex: print(f"  ep{e} skip ({ex})", flush=True); continue
        if len(f224) < 5: continue
        pooled = enc_pooled(f224, proc, enc); pooled /= (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)
        POOL.append(pooled); STATE.append(mkp_gap(state, cfg["stride"])); n = len(f224)  # 速度Δ用gap保持有意义
        EPID.append(np.full(n, e)); TPOS.append(np.arange(n) / max(1, n - 1)); THUMB += th; NIDX.append(nidx); eplen[e] = n
        if (k + 1) % 25 == 0: print(f"  {k+1}/{len(eps)} ({time.time()-t0:.0f}s)", flush=True)
    img = np.concatenate(POOL); Pm = np.concatenate(STATE); E = np.concatenate(EPID); Tv = np.concatenate(TPOS); NIDX = np.concatenate(NIDX)
    THUMB = np.stack(THUMB)                                    # 保留 enc 给 30Hz 读出
    PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8
    Pn = (Pm - PMU) / PSD; Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-9)
    F = np.concatenate([img, Pn], 1); ne = len(eps); N = len(F)
    K0 = int(np.clip(round(0.55 * np.sqrt(N)), 64, 320)); print(f"[{ds}] N={N} K0={K0} 聚类...", flush=True)
    cen, lab = gpu_kmeans(F, K0)
    tpos = np.array([Tv[lab == c].mean() if (lab == c).any() else 0 for c in range(K0)])
    cov = np.array([len(set(E[lab == c].tolist())) / ne if (lab == c).any() else 0 for c in range(K0)])
    tstd = np.array([Tv[lab == c].std() if (lab == c).sum() > 2 else 9.0 for c in range(K0)])
    tau_cov = otsu(cov); tau_pur = float(np.percentile(tstd[tstd < 9], 60))
    cand = sorted([c for c in range(K0) if cov[c] >= tau_cov and tstd[c] <= tau_pur], key=lambda c: tpos[c])
    g0 = max(0.006, 0.5 / max(len(cand), 1)); sel = []
    for c in cand:
        if not sel or tpos[c] - tpos[sel[-1]] >= g0: sel.append(c)
        elif cov[c] > cov[sel[-1]]: sel[-1] = c
    M = len(sel); eps_sorted = sorted(set(E.tolist()))
    fe = np.full((len(eps_sorted), M), np.nan)
    for ei, e in enumerate(eps_sorted):
        fi = np.where(E == e)[0]; labe = lab[fi]; te = Tv[fi]
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
    SP = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][:2]] for e in eps_sorted])
    EPp = np.concatenate([F[np.where(E == e)[0][np.argsort(Tv[np.where(E == e)[0]])][-2:]] for e in eps_sorted])
    sk = KMeans(8, n_init=2, random_state=0).fit(SP).cluster_centers_; ek = KMeans(8, n_init=2, random_state=0).fit(EPp).cluster_centers_
    de_tr = np.array([float(np.linalg.norm(F[np.where(E == e)[0][np.argmax(Tv[np.where(E == e)[0]])]][None] - ek, axis=1).min()) for e in eps_sorted])
    de_thr = float(np.quantile(de_tr, 0.9)) * 1.3              # 完成残差阈(末帧到 endK > 阈 ⇒ 未完成/失败)
    print(f"[{ds}] {M} milestones (precedence+isotonic)", flush=True)
    _dv = Pord[:, None] - Pord[None]
    PEN = np.where(_dv >= 0, 3.0 * _dv, 25.0 * (-_dv))         # forward(value up) cheap=3, backward(value down) expensive=25 (forward-biased Viterbi)

    def readout(Fq, fps=3.0):                                  # Viterbi 直接在 M 个 milestone 上 → 帧直接分簇, value=Pord[簇]
        nn = len(Fq); emit = np.linalg.norm(Fq[:, None] - C[None], axis=2)   # (nn,M) 到各簇中心距离
        dsx = np.linalg.norm(Fq[:, None] - sk[None], axis=2).min(1); tx = np.arange(nn) / nn
        emit[:, 0] = np.minimum(emit[:, 0], np.where(tx < 0.3, dsx, dsx + (tx - 0.3) * 6))   # 起始锚→milestone 0
        cost = np.full(M, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((nn, M), int)            # 硬起始: 帧0=milestone0(value0), 防起始别名

        for j in range(1, nn):
            tr = cost[None, :] + PEN; k = tr.argmin(1); cost = emit[j] + tr[np.arange(M), k]; bp[j] = k
        ms = np.zeros(nn, int); ms[-1] = int(cost.argmin())
        for j in range(nn - 2, -1, -1): ms[j] = bp[j + 1, ms[j + 1]]
        mw = max(5, int(round(5 * fps / 3))) | 1
        return smooth_monotone(med(Pord[ms], mw), fps=fps), ms   # 原始 value(保留循环 milestone 的真实震荡) + 帧分配簇

    # ---- 簇中心解码 + medoid ----
    print(f"[{ds}] 簇中心解码(grid 平均→small decoder)...", flush=True)
    NS = 20; samp = []; rng = []
    for mi, c in enumerate(order):
        loc = np.where(lab == c)[0]
        if len(loc) > NS: loc = loc[np.linspace(0, len(loc) - 1, NS).astype(int)]
        s0 = len(samp)
        for gi in loc: samp.append(cv2.resize(THUMB[gi], (224, 224)))
        rng.append((s0, len(samp)))
    samp = np.stack(samp); grids = encode_grids(samp, np.ones(len(samp), bool), LARGE, DIM, NPREFIX, ENC_RES, ENC_DTYPE)
    imgs128 = np.stack([cv2.resize(s, (128, 128)) for s in samp]); decf = train_dec(grids, imgs128, DIM, "small", 55)
    proto = {}; medoid = {}
    for mi, c in enumerate(order):
        s0, e0 = rng[mi]
        proto[mi] = cv2.resize(decf(grids[s0:e0].mean(0)[None])[0], (96, 96)) if e0 > s0 else np.zeros((96, 96, 3), np.uint8)
        loc = np.where(lab == c)[0]; gi = loc[int(np.argmin(np.linalg.norm(F[loc] - cen[c], axis=1)))]
        medoid[mi] = cv2.resize(THUMB[gi], (96, 96))

    # ---- 04 簇中心 vs medoid 画廊 ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    PR = 18; NBd = (M + PR - 1) // PR
    fig, axes = plt.subplots(2 * NBd, PR, figsize=(PR * 0.82, 2 * NBd * 1.18)); axes = np.atleast_2d(axes)
    for ax in axes.ravel(): ax.axis("off")
    for mi in range(M):
        b, col = mi // PR, mi % PR
        axes[2 * b, col].imshow(proto[mi]); axes[2 * b, col].set_title(f"m{mi} v={Pord[mi]:.2f}", fontsize=6)
        axes[2 * b + 1, col].imshow(medoid[mi])
    fig.suptitle(f"[{ds}] {M} milestones in value order — top: decoded centroid | bottom: nearest real (medoid)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"{ds}_centroid_gallery.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ---- 2 个偏长 ep: 30Hz(原生帧率) forward value 视频 ----
    long2 = sorted(eplen, key=lambda e: -eplen[e])[:2]; print(f"[{ds}] 2 长 ep: {long2}; 30Hz 读出...", flush=True)
    for e in long2:
        f224, staten, fpsd = load_ep_native(ds, cfg, e)       # 原生帧率帧 + state
        imgn = enc_pooled(f224, proc, enc); imgn /= (np.linalg.norm(imgn, axis=1, keepdims=True) + 1e-9)
        Pg = mkp_gap(staten, cfg["stride"]); Pg = (Pg - PMU) / PSD; Pg /= (np.linalg.norm(Pg, axis=1, keepdims=True) + 1e-9)
        Fq = np.concatenate([imgn, Pg], 1); nn = len(Fq)
        v, ms = readout(Fq, fps=fpsd)                         # ms = 帧真实分配簇(非 value 反推); v 已 cummax
        de_end = float(np.linalg.norm(Fq[-3:][:, None] - ek[None], axis=2).min()); comp = de_end <= de_thr   # 完成 flag
        print(f"  ep{e}: {nn}帧@{fpsd}Hz value {v.min():.2f}->{v.max():.2f}", flush=True)
        # 图
        fig = plt.figure(figsize=(13, 5.5)); gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.4])
        ax0 = fig.add_subplot(gs[0]); ax0.plot(v, color="#1a7f37", lw=2)
        for p in Pord: ax0.axhline(p, color="0.93", lw=.5)
        ax0.set_ylim(-.02, 1.02); ax0.set_ylabel("forward value")
        ax0.set_title(f"[{ds}] ep{e} forward value (len {nn}@{fpsd}Hz) — {M} ms — {'COMPLETE' if comp else 'INCOMPLETE'} (resid {de_end:.2f}/thr {de_thr:.2f})",
                      color=("#1a7f37" if comp else "#d62728"))
        ax1 = fig.add_subplot(gs[1]); ax1.step(range(nn), ms, where="post", color="#9c27b0"); ax1.set_ylabel("milestone"); ax1.set_xlabel(f"frame({fpsd}Hz)")
        fig.tight_layout(); fig.savefig(OUT / f"{ds}_ep{e}_value.png", dpi=120, bbox_inches="tight"); plt.close(fig)
        if os.environ.get("NOVIDEO"):                          # 只出图模式: 跳过视频渲染
            print(f"  SAVED {ds}_ep{e}_value.png (NOVIDEO)", flush=True); continue
        # 视频(原生fps): 相机 + 簇中心解码 + value 游标
        figc, axc = plt.subplots(figsize=(6.5, 3.0)); axc.plot(np.arange(nn), v, color="#1a7f37", lw=2)
        axc.set_ylim(-.02, 1.02); axc.set_xlim(0, nn - 1); axc.set_ylabel("forward value", fontsize=8); axc.set_xlabel(f"frame({fpsd}Hz)", fontsize=8); axc.grid(alpha=.3)
        figc.tight_layout(); figc.canvas.draw(); Wp, Hp = figc.canvas.get_width_height()
        bg = np.frombuffer(figc.canvas.buffer_rgba(), np.uint8).reshape(Hp, Wp, 4)[..., :3][..., ::-1].copy()
        bb = axc.get_window_extent(); bx0, bx1, byt, byb = bb.x0, bb.x1, Hp - bb.y1, Hp - bb.y0; plt.close(figc)
        Wl = 360; FONT = cv2.FONT_HERSHEY_SIMPLEX; td = tempfile.mkdtemp()
        for t in range(nn):
            cur = bg.copy(); px = int(bx0 + (t / max(1, nn - 1)) * (bx1 - bx0)); cv2.line(cur, (px, int(byt)), (px, int(byb)), (30, 30, 30), 1)
            cam = cv2.resize(cv2.cvtColor(f224[t], cv2.COLOR_RGB2BGR), (Wl, 280))
            mi = int(ms[t]); cd = cv2.resize(cv2.cvtColor(proto[mi], cv2.COLOR_RGB2BGR), (Wl, 250))
            lbl = np.full((40, Wl, 3), 28, np.uint8); cv2.putText(lbl, f"m{mi}  value={v[t]:.2f}", (8, 27), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
            left = np.vstack([cam, lbl, cd]); Hl = left.shape[0]; curR = cv2.resize(cur, (int(cur.shape[1] * Hl / cur.shape[0]), Hl))
            cv2.imwrite(f"{td}/{t:05d}.png", np.hstack([left, curR]))
        mp4 = OUT / f"{ds}_ep{e}_value.mp4"
        enc_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]   # GPU 视频编码
        r = subprocess.run(["ffmpeg", "-y", "-framerate", str(fpsd), "-i", f"{td}/%05d.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            *enc_args, "-pix_fmt", "yuv420p", str(mp4)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:                                  # nvenc 不可用则回退 CPU
            subprocess.run(["ffmpeg", "-y", "-framerate", str(fpsd), "-i", f"{td}/%05d.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in glob.glob(f"{td}/*.png"): os.remove(f)
        print(f"  SAVED {mp4.name} (value {v.min():.2f}->{v.max():.2f})", flush=True)
    del enc; torch.cuda.empty_cache()
    json.dump({"ds": ds, "n_eps": len(eps), "N": int(N), "K0": K0, "M": M, "long2": long2,
               "value_range": [float(v.min()), float(v.max())]}, open(OUT / f"{ds}_summary.json", "w"), indent=2)
    print(f"[{ds}] DONE {time.time()-t0:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
