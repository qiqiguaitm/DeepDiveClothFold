#!/usr/bin/env python
"""CRAVE vs KAI0-AE 可解释性分析(基于簇原型 / 典型簇)。

回答: 为什么 ep808 的 CRAVE value 上升? 为什么上升到这个值? 典型簇是哪些?
做法:
  ① 挖 dagger milestone 模型 → 每个 milestone 簇取**原型帧**(挖矿集里离簇心最近的真实相机帧)= 典型簇。
  ② 把 ep808 整段按三档(pos 上升 / normal 平台 / neg 退步)拆分; 每帧匹配最近 milestone 簇。
  ③ value[t] = 该帧所匹配 milestone 的挖掘进度 → 用原型帧解释"像哪个簇 → 故 value=该簇进度"。
  ④ 拆出的代表段 + KAI0-AE 对齐进一条短视频: 相机 | CRAVE value/档 | KAI0-AE value/档 | 当前匹配的典型簇原型。
输出目录: temp/crave_interp_ep808/  (clusters_gallery.png, seg_*.png, crave_vs_kai0ae_interp_ep808.mp4, README.md)
KAI0-AE = pi05 + value_head 监督回归(absolute_value), 取自 A_smooth800_dagger_all_awbc。
"""
import json, os, cv2
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crave_readout import smooth_monotone

_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

REPO = Path("/home/tim/workspace/deepdive_kai0")
# ===== 数据源(env 可切换; 默认 smooth800_dagger; kai0_base 用 triple 缓存) =====
CACHE = os.environ.get("INTERP_CACHE", "sep")          # sep(分离 raw/armmask "f"键 + parquet state) | triple(单 npz raw/armmask/state)
DS = REPO / os.environ.get("INTERP_DS", "kai0/data/Task_A/self_built/A_smooth800_dagger_all")
FC = REPO / os.environ.get("INTERP_FC", "temp/tcc_smooth800_dagger_armmask/feat_cache")   # triple: 唯一缓存; sep: armmask 缓存
RAW = REPO / os.environ.get("INTERP_RAW_FC", "temp/tcc_smooth800_dagger_raw/feat_cache")  # sep 专用
_aw = os.environ.get("INTERP_AW", "kai0/data/Task_A/self_built/A_smooth800_dagger_all_awbc")
HAVE_AE = _aw.lower() != "none"
AW = REPO / _aw if HAVE_AE else None
MINE_MAX = int(os.environ.get("INTERP_MINE_MAX", "0"))  # >0: 仅 episode_index < 此值参与挖矿(域纯净, 如 kai0_base 取 <3055)
EP = int(os.environ.get("INTERP_EP", "808"))   # 目标 episode
TAG = os.environ.get("INTERP_TAG", "")         # 输出文件夹后缀(如 _30hz)
OUT = REPO / f"temp/crave_interp_ep{EP}{TAG}"; OUT.mkdir(exist_ok=True)
# 频率相关(3Hz 挖矿默认; 30Hz 挖矿: DT=10 LAM=80 MEDW=45 STRIDE=1, 见 crave_mine_freq_compare.py)
DT = int(os.environ.get("INTERP_DT", "1"))         # proprio Δ 帧距
LAM = float(os.environ.get("INTERP_LAM", "8"))     # Viterbi-DP 转移惩罚
MEDW = int(os.environ.get("INTERP_MEDW", "9"))     # value 中值窗
STRIDE = int(os.environ.get("INTERP_STRIDE", "10"))  # 特征帧→视频帧 步长(3Hz=10, 30Hz=1)
MINE_N = 500; W = 50; EPS = 0.02
MERGE_NORMAL = 50  # 短 normal(≤此帧数, ~advantage 窗 W)若两侧同为 pos/pos 或 neg/neg → 融合进该类(段内微停顿, 非独立保持相)
csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]
csAW = json.load(open(AW / "meta/info.json"))["chunks_size"] if HAVE_AE else csDS
# 多数据源原型支持: sources.json {cache_idx:[dataset_rel, real_ep]}(挖矿集跨 kai0_base+kai0_dagger 时用)
_srcp = os.environ.get("INTERP_SOURCES", "")
SOURCES = json.load(open(REPO / _srcp)) if _srcp and (REPO / _srcp).exists() else None
_DSMAP = {"kai0_base": REPO / "kai0/data/Task_A/kai0_base", "kai0_dagger": REPO / "kai0/data/Task_A/kai0_dagger"}
_DSCS = {k: json.load(open(v / "meta/info.json"))["chunks_size"] for k, v in _DSMAP.items()} if SOURCES else {}
RGB = {1: (0.17, 0.63, 0.17), 0: (0.6, 0.6, 0.6), -1: (0.84, 0.15, 0.16)}
BGR = {1: (44, 160, 44), 0: (150, 150, 150), -1: (214, 39, 40)}; NAME = {1: "POSITIVE", 0: "NORMAL", -1: "NEGATIVE"}
print(f"[数据源] cache={CACHE} FC={FC.name} DS={DS.name} AE={'有' if HAVE_AE else '无(kai0_base未标注)'} mine<{MINE_MAX or '全部'}", flush=True)


def lpst(e, n):
    pq = DS / "data" / f"chunk-{e//csDS:03d}" / f"episode_{e:06d}.parquet"
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
    return st[np.minimum(np.arange(n) * 10, len(st) - 1)]


def loadep(e):
    if CACHE == "triple":
        d = np.load(FC / f"ep{e}.npz"); a, r, s = d["armmask"], d["raw"], d["state"]
        n = min(len(a), len(r), len(s))
        s = np.clip(np.nan_to_num(s[:n].astype(np.float64)), -10, 10)   # 防离群/NaN state 污染 z-score
        return a[:n], r[:n], s, n
    a = np.load(FC / f"ep{e}.npz")["f"]; r = np.load(RAW / f"ep{e}.npz")["f"]
    n = min(len(a), len(r)); return a[:n], r[:n], lpst(e, n), n


def mkp(s):   # state ⊕ Δstate(隔 DT 帧, 对齐时间语义; 3Hz DT=1, 30Hz DT=10)
    d = np.zeros_like(s); d[DT:] = s[DT:] - s[:-DT]
    return np.concatenate([s, d], 1)
def med(a, w):
    h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])


# ===== 挖矿(canonical) + 记录每帧来源 (ep, 特征帧idx) 以便取原型相机帧 =====
if CACHE == "triple":
    all_eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
else:
    rawset = set(int(p.stem[2:]) for p in RAW.glob("ep*.npz"))
    all_eps = sorted(e for e in (int(p.stem[2:]) for p in FC.glob("ep*.npz")) if e in rawset)
mine_pool = [e for e in all_eps if (MINE_MAX == 0 or e < MINE_MAX)]   # 域纯净过滤(kai0_base: <3055)
mined = sorted(np.random.RandomState(0).permutation(mine_pool)[:min(MINE_N, len(mine_pool))].tolist())
if EP not in mined: mined = sorted(mined + [EP])
print(f"挖矿 {len(mined)} eps", flush=True)
Sall = [loadep(e)[2] for e in mined]; Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8


def emb(a_, r_, st):
    an = a_ / (np.linalg.norm(a_, axis=1, keepdims=True) + 1e-8); rn = r_ / (np.linalg.norm(r_, axis=1, keepdims=True) + 1e-8)
    Pn = ((mkp(st) - PMU) / PSD); Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-8)   # +eps 防零范数→NaN
    return np.concatenate([rn, an, Pn], 1)


A, R, S, T, E, FI, SP, EP_ = [], [], [], [], [], [], [], []
for e in mined:
    aa, rr, st, n = loadep(e); g = emb(aa, rr, st)
    A.append(aa); R.append(rr); S.append(st); T.append(np.arange(n) / max(1, n - 1))
    E.append(np.full(n, e)); FI.append(np.arange(n)); SP.append(g[:2]); EP_.append(g[-2:])
A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T)
E = np.concatenate(E); FI = np.concatenate(FI)
G = emb(A, R, S)
km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
N = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
Pstart = {}
for e in sorted(set(E.tolist())):
    m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Pstart[e] = float(np.median(tpos[nn]))
cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / N) for c in range(96)])
# 纯 episode 覆盖率(无 start-bonus,= cluster_stats / all_clusters 口径)
covE = np.array([len(set(E[lab == c].tolist())) / N if (lab == c).any() else 0.0 for c in range(96)])
eff = cov_n.copy(); TAU = float(np.quantile(cov_n, 0.5))   # 默认值(coverage 模式会重算 eff/TAU,供 all_clusters 图用)
SELECT = os.environ.get("INTERP_SELECT", "coverage")   # coverage(默认,全留 cov_n≥τ,不均衡不去重,最大化细节) | balanced(覆盖+进度NMS) | adaptive(每带top3)
TAU_Q = float(os.environ.get("INTERP_TAU_Q", "0.5"))
if SELECT == "coverage":
    # 选择基于**纯 episode 覆盖率 covE**(真复现度,不被 start-bonus 虚高)+ ① 稀疏度补偿 ② Otsu 自适应阈值。
    # (start-bonus 只用于 value 计算的早期锚定,不进选择 → 早期低真覆盖的簇不再被强行抬上来)
    DELTA = float(os.environ.get("INTERP_DENS_DELTA", "0.05")); BETA = float(os.environ.get("INTERP_SPARSE_BETA", "0.10"))
    dens = np.array([int(np.sum(np.abs(tpos - tpos[c]) <= DELTA)) for c in range(96)])  # 进度邻域簇数(含自身)
    spars = (dens.max() - dens) / max(1, dens.max() - dens.min())     # [0,1], 1=最稀疏
    eff = np.clip(covE + BETA * spars, 0, 1.2)                        # 真覆盖率 + 稀疏补偿
    if "INTERP_TAU_Q" in os.environ:                                  # 显式给分位则用分位, 否则 Otsu 自适应
        TAU = float(np.quantile(eff, TAU_Q)); _tmode = f"{TAU_Q}分位"
    else:
        h, e = np.histogram(eff, bins=64); cc = (e[:-1] + e[1:]) / 2; tot = h.sum(); w0 = s0 = 0.0; TAU = float(cc[0]); best = -1
        for i in range(64):
            w0 += h[i]; s0 += h[i] * cc[i]; w1 = tot - w0
            if w0 and w1:
                m0 = s0 / w0; m1 = (h[i + 1:] * cc[i + 1:]).sum() / w1; v = w0 * w1 * (m0 - m1) ** 2
                if v > best: best, TAU = v, float(e[i + 1])
        _tmode = "Otsu自适应"
    sel = sorted([c for c in range(96) if eff[c] >= TAU], key=lambda c: tpos[c])
    print(f"[覆盖率选择] eff=covE+{BETA}·稀疏(邻域±{DELTA}); 阈值 τ={TAU:.2f}({_tmode}) → {len(sel)} milestone "
          f"(纯覆盖率 {min(covE[c] for c in sel):.0%}-{max(covE[c] for c in sel):.0%})", flush=True)
elif SELECT == "balanced":
    # 均衡(无保底): 按 cov_n(含 start-bonus,已保证早/晚 milestone 可比) 降序 + 进度 NMS(最小间隔 MIN_GAP)。
    # cov_n<τ 的进度区间**不强行塞** milestone(早期真复现的簇靠 start-bonus 已能过 τ);允许进度空隙 → 分布大体均匀但不强制。
    TAU = float(np.quantile(cov_n, TAU_Q)); MIN_GAP = float(os.environ.get("INTERP_MIN_GAP", "0.03"))
    cand = sorted([c for c in range(96) if cov_n[c] >= TAU], key=lambda c: -cov_n[c])
    sel = []
    for c in cand:
        if all(abs(tpos[c] - tpos[s]) >= MIN_GAP for s in sel): sel.append(c)
    sel = sorted(set(sel), key=lambda c: tpos[c])
    print(f"[均衡选择·无保底] cov_n≥τ={TAU:.2f}({TAU_Q}分位,含start-bonus) + 进度NMS间隔{MIN_GAP} → {len(sel)} milestone "
          f"(纯覆盖率 {min(covE[c] for c in sel):.0%}-{max(covE[c] for c in sel):.0%})", flush=True)
else:   # adaptive(旧): 每 0.1 带 top-≤3(cov_n 含 start-bonus), 保底1个
    NBINS, CAP_PB = 10, 3; TAU = float(np.quantile(cov_n, TAU_Q))
    bk = np.linspace(0, 1, NBINS + 1); sel = []
    for b in range(NBINS):
        inb = sorted([c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]], key=lambda c: -cov_n[c])
        if not inb: continue
        above = [c for c in inb if cov_n[c] >= TAU][:CAP_PB]
        sel += above if above else inb[:1]
    sel = sorted(set(sel), key=lambda c: tpos[c])
    print(f"[自适应选择] τ={TAU:.2f} cap={CAP_PB} → {len(sel)} milestone", flush=True)

# 严格只取 N 个 milestone(要求高、图更净): 按真覆盖率 covE 降序 + 进度 NMS(均匀铺满任务进度)
TOPN = int(os.environ.get("INTERP_TOPN", "0"))
if TOPN > 0:
    GAP = float(os.environ.get("INTERP_TOPN_GAP", "0.06"))
    cand = [c for c in sorted(range(96), key=lambda c: -covE[c]) if covE[c] > 0 and (lab == c).any()]
    picked = []
    for c in cand:
        if all(abs(tpos[c] - tpos[s]) >= GAP for s in picked): picked.append(c)
        if len(picked) >= TOPN: break
    if len(picked) < TOPN:                                   # NMS 太严没凑够 → 放宽补齐
        for c in cand:
            if c not in picked: picked.append(c)
            if len(picked) >= TOPN: break
    sel = sorted(set(picked), key=lambda c: tpos[c])
    TAU = float(min(covE[c] for c in sel))                   # 供 all_clusters 图的阈值线
    print(f"[TOPN 严格选择] 取 covE 最高的 {len(sel)} 个 milestone(进度NMS间隔≥{GAP}); "
          f"纯覆盖率 {min(covE[c] for c in sel):.0%}-{max(covE[c] for c in sel):.0%}", flush=True)


def gr(idx):
    o = []; s0 = None; pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s0 is not None: o.append((s0, pv))
            s0 = i
        pv = i
    if s0 is not None: o.append((s0, pv))
    return [x for x in o if x[1] - x[0] >= 1]


Pk = {}
for c in sel:
    fe = []
    for e in sorted(set(E.tolist())):
        m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
        if rs: fe.append(T[rs[0][0]])
    Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
order = sorted(sel, key=lambda c: Pk[c]); C = allC[order]; Pord = np.array([Pk[c] for c in order])
startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP_)).cluster_centers_
NB = 21; bins = np.linspace(0, 1, NB); cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]
NM = len(order); print(f"milestones={NM} Pord {Pord.min():.2f}-{Pord.max():.2f}", flush=True)

# ===== 数据/簇 体量分析(用多少数据 → 分成多少簇, 是否合理) =====
Ntot = len(G); ksize = np.bincount(lab, minlength=96)
nmall = np.empty(Ntot, int)
for i0 in range(0, Ntot, 20000):
    nmall[i0:i0 + 20000] = np.linalg.norm(G[i0:i0 + 20000, None] - C[None], axis=2).argmin(1)
mcount = np.bincount(nmall, minlength=NM)
Nep = len(set(E.tolist()))   # 覆盖率 = 含此 milestone(最近归属)的 demo 比例 = 跨 episode 复现度
cover = np.array([len(set(E[nmall == k].tolist())) / Nep for k in range(NM)])
print(f"[覆盖率] {NM} milestone(进度序)各覆盖: {[f'{c:.0%}' for c in cover]}", flush=True)
# 全部 96 个 KMeans 簇的覆盖率 + 选中标记 → 解释 milestone 数量从何而来
cov96 = np.array([len(set(E[lab == c].tolist())) / Nep if (lab == c).any() else 0.0 for c in range(96)])
sel_set = set(order); oo = np.argsort(tpos)
figc, axc = plt.subplots(figsize=(16, 4.5))
bar_c = ["#3b6fb0" if int(oo[i]) in sel_set else "#d9d9d9" for i in range(96)]
axc.bar(np.arange(96), cov96[oo] * 100, color=bar_c, label="pure episode coverage covE (bar)")
axc.plot(np.arange(96), eff[oo] * 100, "k^", ms=4, label="eff = covE + sparsity bonus (selection score)")
axc.plot(np.arange(96), cov_n[oo] * 100, ".", color="#ccc", ms=2.5, label="cov_n (with start-bonus, ref only)")
_thr_lab = f"top-{TOPN} cutoff covE={TAU:.0%}" if TOPN > 0 else f"Otsu adaptive threshold tau={TAU:.0%}"
axc.axhline(TAU * 100, color="r", ls="--", lw=1.2, label=_thr_lab)
axc.set_xticks([]); axc.set_xlabel("96 KMeans clusters (sorted by mined progress ->, left=task start, right=done)")
axc.set_ylabel("coverage %"); axc.set_ylim(0, 108); axc.grid(alpha=.2, axis="y"); axc.legend(fontsize=8.5, ncol=3, loc="upper center")
if TOPN > 0:
    _rule = f"strict top-{len(order)}: highest covE + progress-NMS spread over the task"
elif SELECT == "coverage":
    _rule = f"keep-all coverage: all clusters with cov_n>=tau (unbalanced, no dedup) -> {len(order)} milestones"
elif SELECT == "balanced":
    _rule = f"balanced: cov_n>=tau + progress-NMS (gap {os.environ.get('INTERP_MIN_GAP','0.03')}) -> {len(order)} milestones"
else:
    _rule = f"adaptive: per-0.1-band cov_n>=tau top-<=3 + fallback -> {len(order)} milestones"
axc.set_title(f"All 96 clusters coverage - blue = selected milestone ({len(order)}) / grey = unselected. {_rule}", fontsize=10)
figc.tight_layout(); figc.savefig(OUT / "all_clusters_coverage.png", dpi=110); plt.close(figc)
print("SAVED", OUT / "all_clusters_coverage.png", flush=True)
COVER_FLOOR = float(os.environ.get("INTERP_COVER_FLOOR", "0"))   # 丢弃覆盖率<此的 milestone(非真复现态)
if COVER_FLOOR > 0:
    low = [k for k in range(NM) if cover[k] < COVER_FLOOR]
    if low:
        print(f"[覆盖率过滤] floor={COVER_FLOOR:.0%} → 丢弃 {len(low)} 个低覆盖 milestone: "
              f"{[(round(float(Pord[k]),2), f'{cover[k]:.0%}') for k in low]}", flush=True)
        keep = [k for k in range(NM) if cover[k] >= COVER_FLOOR]
        order = [order[k] for k in keep]; C = allC[order]; Pord = np.array([Pk[c] for c in order])
        cb = [[int(np.argmin(abs(bins - Pk[c])))] for c in order]; NM = len(order)
        for i0 in range(0, Ntot, 20000):
            nmall[i0:i0 + 20000] = np.linalg.norm(G[i0:i0 + 20000, None] - C[None], axis=2).argmin(1)
        mcount = np.bincount(nmall, minlength=NM)
        cover = np.array([len(set(E[nmall == k].tolist())) / Nep for k in range(NM)])
        print(f"[覆盖率过滤] 保留 {NM} milestone, 覆盖率 min{cover.min():.0%} 中位{np.median(cover):.0%}", flush=True)
if SOURCES:   # 挖矿集数据源分布(kai0_base vs kai0_dagger)
    _mix = {}
    for e in mined: _mix[SOURCES.get(str(e), ["?"])[0]] = _mix.get(SOURCES.get(str(e), ["?"])[0], 0) + 1
    print(f"[数据源分布] 挖矿 {len(mined)} ep = " + " + ".join(f"{v} {k}" for k, v in sorted(_mix.items())), flush=True)
FPS_FEAT = 30.0 / STRIDE; FREQ = f"{FPS_FEAT:.0f}Hz"  # STRIDE10=3Hz, STRIDE1=30Hz
sec_total = Ntot / FPS_FEAT
print(f"[体量] 挖矿频率 {FREQ}(DT={DT} LAM={LAM:.0f} MEDW={MEDW}); {len(mined)} ep, {Ntot} 帧({FREQ}≈{sec_total/60:.0f}分钟demo); KMeans-96 → 选 {NM} 个 milestone", flush=True)
print(f"[体量] 每 milestone 覆盖率(含此里程碑的 demo 比例): 中位 {np.median(cover):.0%} min {cover.min():.0%} max {cover.max():.0%}", flush=True)
print(f"[体量] milestone 进度间距 中位 {np.median(np.diff(Pord)):.3f} (越匀越好), 覆盖 {Pord.min():.2f}-{Pord.max():.2f}", flush=True)
figs, axs = plt.subplots(1, 2, figsize=(12, 3.8))
axs[0].bar(range(NM), cover * 100, color="#3b6fb0"); axs[0].axhline(cover.mean() * 100, color="r", ls="--", lw=1, label=f"mean {cover.mean():.0%}")
axs[0].set_xlabel("milestone (by progress)"); axs[0].set_ylabel("coverage % (demos containing this milestone)"); axs[0].set_ylim(0, 105)
axs[0].set_title(f"per-milestone cross-episode coverage ({Nep} demos)", fontsize=10); axs[0].legend(fontsize=8); axs[0].grid(alpha=.2, axis="y")
axs[1].plot(Pord, range(NM), "o-", color="#1a9641"); axs[1].set_xlabel("mined progress Pord"); axs[1].set_ylabel("milestone index"); axs[1].set_xlim(-.02, 1.02)
axs[1].set_title(f"milestone progress coverage (median gap {np.median(np.diff(Pord)):.2f})", fontsize=10); axs[1].grid(alpha=.25)
figs.suptitle(f"CRAVE scale ({FREQ} mining): {len(mined)}ep / {Ntot} frames -> KMeans-96 -> {NM} milestones", fontsize=11)
figs.tight_layout(); figs.savefig(OUT / "cluster_stats.png", dpi=110); plt.close(figs)
print("SAVED", OUT / "cluster_stats.png", flush=True)


def dpHB(emit, lam=LAM):
    pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
    cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return bins[path]


def value_nm(aa, rr, st):
    """返回 v3(阶梯) + 每帧最近 milestone idx(在 order 中)。"""
    Fq = emb(aa, rr, st); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - C[None], axis=2); em = np.full((nq, NB), 1e3)
    for ci in range(NM):
        for b in cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
    ds = np.linalg.norm(Fq[:, None] - startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - endK[None], axis=2).min(1)
    tn = np.arange(nq) / nq
    em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
    em[:, NB - 1] = np.minimum(em[:, NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
    return med(dpHB(em), MEDW), d.argmin(1)


# ===== ① 簇原型(典型簇): 每个 milestone 取挖矿集里离簇心最近的真实相机帧 =====
def grab(e, vframe):
    if SOURCES is not None and str(e) in SOURCES:   # 源感知: 原型可能来自 kai0_base 或 kai0_dagger
        dskey, real = SOURCES[str(e)]; dsp = _DSMAP[dskey]; cs = _DSCS[dskey]
        mp4 = dsp / f"videos/chunk-{real//cs:03d}/observation.images.top_head/episode_{real:06d}.mp4"
    else:
        mp4 = DS / f"videos/chunk-{e//csDS:03d}/observation.images.top_head/episode_{e:06d}.mp4"
    cap = cv2.VideoCapture(str(mp4)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(vframe)); ok, fr = cap.read(); cap.release()
    return fr[:, :, ::-1] if ok else np.zeros((480, 640, 3), np.uint8)


proto = []  # (milestone_idx_in_order, cluster_progress, ep, feat_idx, image)
for k in range(NM):
    dist = np.linalg.norm(G - C[k], axis=1); gi = int(dist.argmin())
    e, fj = int(E[gi]), int(FI[gi]); img = grab(e, fj * STRIDE)
    proto.append((k, float(Pord[k]), e, fj, img))
print(f"原型帧 {len(proto)} 个已取", flush=True)

# 画廊
ncol = 5; nrow = int(np.ceil(NM / ncol))
figg, axg = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.95), constrained_layout=True)
axg = np.atleast_2d(axg)
for idx in range(nrow * ncol):
    ax = axg[idx // ncol, idx % ncol]; ax.axis("off")
    if idx < NM:
        k, p, e, fj, img = proto[idx]; ax.imshow(img)
        _src = f"{SOURCES[str(e)][0]} ep{SOURCES[str(e)][1]}" if (SOURCES and str(e) in SOURCES) else f"ep{e}"
        ax.set_title(f"milestone #{k}  progress={p:.2f}\n(from {_src})", fontsize=8.5)
figg.suptitle(f"CRAVE milestone prototype vocabulary ({NM} milestones, sorted by progress)\n"
              "each cluster = a recurrent state across demos; value[t] = progress of the cluster the frame looks most like", fontsize=12)
figg.savefig(OUT / "clusters_gallery.png", dpi=110, bbox_inches="tight"); plt.close(figg)
print("SAVED", OUT / "clusters_gallery.png", flush=True)

# ===== ② ep808: CRAVE 连续 value + 最近簇; KAI0-AE value; 三档 =====
aa, rr, st, n3 = loadep(EP); v3, nm3 = value_nm(aa, rr, st)
NF = len(pd.read_parquet(DS / f"data/chunk-{EP//csDS:03d}/episode_{EP:06d}.parquet", columns=["frame_index"]))
cv = np.repeat(v3, STRIDE)[:NF]
if len(cv) < NF: cv = np.concatenate([cv, np.full(NF - len(cv), cv[-1])])
cv = smooth_monotone(cv, fps=30.0)
if HAVE_AE:
    ae_v = pd.read_parquet(AW / f"data/chunk-{EP//csAW:03d}/episode_{EP:06d}.parquet", columns=["absolute_value"])["absolute_value"].to_numpy().astype(float)
    n = min(NF, len(cv), len(ae_v)); cv, ae_v = cv[:n], ae_v[:n]
else:                       # kai0_base 无 KAI0-AE 标注 → 仅 CRAVE
    n = min(NF, len(cv)); cv = cv[:n]; ae_v = np.full(n, np.nan)
# 展示用 milestone = 与连续 value 一致的那一档(value=该簇挖掘进度), 直接解释"为什么是这个值"
nm30 = np.array([int(np.argmin(np.abs(Pord - cv[i]))) for i in range(n)])


def adv(v, w=W):
    return np.array([np.clip(v[min(i + w, len(v) - 1)] - v[i], -1, 1) for i in range(len(v))])


def three(a): return np.where(a > EPS, 1, np.where(a < -EPS, -1, 0))


def runs(cls):
    out = []; s = 0
    for i in range(1, len(cls) + 1):
        if i == len(cls) or cls[i] != cls[s]: out.append((s, i - 1, int(cls[s]))); s = i
    return out


def merge_small_normal(cls, max_len):
    """短 normal 段(≤max_len)若被同一非 normal 类(pos/pos 或 neg/neg)两侧夹住 → 并入该类。
    迭代到收敛(相邻同类自动合并后可能暴露新的可并 normal)。"""
    cls = cls.copy(); merged = 0; mframes = 0
    while True:
        R = runs(cls); changed = False
        for i in range(1, len(R) - 1):
            s0, s1, c = R[i]
            if c == 0 and (s1 - s0 + 1) <= max_len and R[i - 1][2] == R[i + 1][2] != 0:
                cls[s0:s1 + 1] = R[i - 1][2]; merged += 1; mframes += s1 - s0 + 1; changed = True; break
        if not changed: break
    return cls, merged, mframes


ccls_raw = three(adv(cv)); acls = three(adv(ae_v)) if HAVE_AE else np.zeros(len(cv), int)
ccls, nmrg, nmf = merge_small_normal(ccls_raw, MERGE_NORMAL)   # 段定义用融合后(段内微停顿归入升/降)
fcr = {c: float((ccls_raw == c).mean()) for c in (1, 0, -1)}
fc = {c: float((ccls == c).mean()) for c in (1, 0, -1)}
fa = {c: float((acls == c).mean()) for c in (1, 0, -1)} if HAVE_AE else {1: 0, 0: 0, -1: 0}
print(f"CRAVE 原始 pos{fcr[1]:.0%}/normal{fcr[0]:.0%}/neg{fcr[-1]:.0%}", flush=True)
_ae_str = f" | KAI0-AE pos{fa[1]:.0%}/normal{fa[0]:.0%}/neg{fa[-1]:.0%}" if HAVE_AE else " | KAI0-AE 无(kai0_base 未标注)"
print(f"CRAVE 融合 pos{fc[1]:.0%}/normal{fc[0]:.0%}/neg{fc[-1]:.0%} (并 {nmrg} 个小 normal={nmf}帧){_ae_str}", flush=True)

# ===== ③ 拆分段(用融合后的 ccls): 连续同档 run, 取每类代表段 =====
segs = runs(ccls)
reps = {}
for c in (1, 0, -1):
    cand = [sg for sg in segs if sg[2] == c and (sg[1] - sg[0]) > 30]
    if cand: reps[c] = max(cand, key=lambda sg: sg[1] - sg[0])
print(f"段数 {len(segs)}; 代表段 pos={reps.get(1)} normal={reps.get(0)} neg={reps.get(-1)}", flush=True)

# 段-原型蒙太奇: 每个代表段 取起/中/止 3 帧 + 各自匹配的典型簇原型
def seg_montage(c, sg, fname):
    s0, s1, _ = sg; picks = [s0, (s0 + s1) // 2, s1]
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 7.8), constrained_layout=True)
    for col, fi in enumerate(picks):
        cam = grab(EP, fi); ax[0, col].imshow(cam); ax[0, col].axis("off")
        ax[0, col].set_title(f"ep{EP} frame {fi}  CRAVE={NAME[ccls[fi]]}\nvalue={cv[fi]:.2f}", fontsize=9, color=RGB[ccls[fi]])
        mk = nm30[fi]; pk, pp, pe, pj, pimg = proto[mk]
        ax[1, col].imshow(pimg); ax[1, col].axis("off")
        ax[1, col].set_title(f"value={cv[fi]:.2f} -> milestone #{pk}\n(progress={pp:.2f}, demo ep{pe})", fontsize=9, color="#3b6fb0")
    cls_name = {1: "POSITIVE (rising)", 0: "NORMAL (plateau)", -1: "NEGATIVE (regression)"}[c]
    fig.suptitle(f"{cls_name}: ep{EP} frames [{s0}-{s1}] - top = camera frames, bottom = matched milestone prototypes\n"
                 f"value {cv[s0]:.2f}->{cv[s1]:.2f}: matched cluster goes from #{nm30[s0]} (prog {Pord[nm30[s0]]:.2f}) to #{nm30[s1]} (prog {Pord[nm30[s1]]:.2f})",
                 fontsize=11, color=RGB[c])
    fig.savefig(OUT / fname, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("SAVED", OUT / fname, flush=True)
    return picks


seg_info = {}
for c, tag in [(1, "pos"), (0, "normal"), (-1, "neg")]:
    if c in reps: seg_montage(c, reps[c], f"seg_{tag}.png"); seg_info[c] = reps[c]

# ===== ④ 簇-帧对应: ep 经过的 milestone 序列(nm30, 与视频一致)+ 簇跳变处的原视频帧, 按簇归类 =====
def cluster_runs(a, minlen=15):
    out = []; s = 0
    for i in range(1, len(a) + 1):
        if i == len(a) or a[i] != a[s]:
            if i - s >= minlen: out.append((int(a[s]), s, i - 1))   # (cluster_k, vstart, vend) 30fps 视频帧
            s = i
    return out
clr = cluster_runs(nm30, minlen=15)
# ===== 对齐图: 仅展示这 NM 个 milestone 与 ep 的对齐关系(每行一个 milestone: 原型 + 命中帧占用条 + 顶部 value 曲线) =====
rows = list(range(NM))   # 全部 milestone(已按进度排序)
figt = plt.figure(figsize=(13, 0.55 * NM + 1.6))
gst = figt.add_gridspec(NM + 1, 2, width_ratios=[1, 9], height_ratios=[1.5] + [1] * NM, wspace=0.02, hspace=0.12)
axv = figt.add_subplot(gst[0, 1])
axv.plot(np.arange(n), cv, color="#2ca02c", lw=1.8); axv.set_xlim(0, n); axv.set_ylim(0, 1.05)
axv.set_xticks([]); axv.set_ylabel("value", fontsize=8); axv.grid(alpha=.2)
axv.set_title(f"ep{EP}: alignment to the {NM} milestone clusters - blue = frames matched to that milestone (top: CRAVE value)", fontsize=11)
for ri, ck in enumerate(rows):
    axp = figt.add_subplot(gst[ri + 1, 0]); axp.imshow(proto[ck][4]); axp.axis("off")
    axp.set_title(f"#{ck} p={Pord[ck]:.2f}", fontsize=7.5, color="#3b6fb0")
    axb = figt.add_subplot(gst[ri + 1, 1])
    axb.fill_between(np.arange(n), 0, (nm30 == ck).astype(float), step="mid", color="#3b6fb0", alpha=.85)
    axb.set_xlim(0, n); axb.set_ylim(0, 1); axb.set_yticks([])
    for sp in ("top", "right", "left"): axb.spines[sp].set_visible(False)
    if ri < NM - 1: axb.set_xticks([])
    else: axb.set_xlabel("ep frame (30fps)")
figt.savefig(OUT / "cluster_timeline.png", dpi=120, bbox_inches="tight"); plt.close(figt)
print("SAVED", OUT / "cluster_timeline.png", flush=True)
from collections import defaultdict
bycl = defaultdict(list)
for ck, vs, ve in clr: bycl[ck].append((vs, ve))
_md = [f"# ep{EP} 簇-帧对应(按簇归类)\n",
       f"ep{EP} 经过 {ncl} 段(nm30,与视频/`cluster_timeline.png` 一致),访问 {len(bycl)} 个不同 milestone 簇。",
       f"同一簇可能对应**多段帧**(状态在该 milestone 反复出现)。\n",
       f"| 典型簇 | 进度 Pord | ep{EP} 命中帧段(可多段) | 总帧 |", "|---|---|---|---|"]
for ck in sorted(bycl, key=lambda k: Pord[k]):
    sgs = bycl[ck]; tot = sum(e - s + 1 for s, e in sgs)
    _md.append(f"| #{ck} | {Pord[ck]:.2f} | {', '.join(f'[{s}-{e}]' for s, e in sgs)} | {tot} |")
(OUT / "cluster_frames.md").write_text("\n".join(_md), encoding="utf-8")
print("SAVED", OUT / "cluster_frames.md", flush=True)

np.savez(OUT / "_cache.npz", cv=cv, ae_v=ae_v, ccls=ccls, acls=acls, nm30=nm30, Pord=Pord, n=n,
         have_ae=int(HAVE_AE), ep=EP,
         reps_pos=reps.get(1, (-1, -1, 1)), reps_normal=reps.get(0, (-1, -1, 0)), reps_neg=reps.get(-1, (-1, -1, -1)))
# 原型缩略图存盘供视频脚本用
np.save(OUT / "_proto_imgs.npy", np.array([cv2.resize(p[4], (240, 180)) for p in proto]))
np.save(OUT / "_proto_meta.npy", np.array([[p[0], p[1], p[2]] for p in proto], dtype=float))
print("INTERP_CLUSTERS_DONE", flush=True)
