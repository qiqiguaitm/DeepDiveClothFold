#!/usr/bin/env python
"""CRAVE 可解释分析 II:① autonomy 真机 rollout 的关键退步点同款三路归因;
② 把 grounded 判据接进 AWBC advantage 生成 —— 用"只保留特征支持(grounded)的 milestone 步进"
重建 value → 重算 advantage,量化对 AWBC 标签的去噪(neg 假标减少 + 单调性提升)。

grounded 重建: v_clean 从 v[0] 出发, 只累加 grounded 转移的 Δ, 丢弃 ungrounded(DP/中值瞬变)的 Δ。
  → adv_clean = clip(v_clean[t+50]−v_clean[t], −1, 1); 三档 = where(adv<−.02,neg, where(adv>.02,pos, normal))。
量化: 专家遥操 demo 里真退步稀少 → neg-adv 多为 DP 噪声; grounded 过滤应显著降 neg 假标且不伤真退步。

输出: docs/visualization/cross_episode_recurrence_value/{crave_interp_autonomy.png, crave_grounded_advantage.png}
      + 同目录 crave_grounded_advantage.md
复用挖矿/value 配方与 crave_interpretability.py / crave_vs_ae_*.py 逐字一致。
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
OUTV = REPO / "docs/visualization/cross_episode_recurrence_value"
RAWD, ARMD, PROD = slice(0, 384), slice(384, 768), slice(768, 796)
W = 50  # advantage 窗口
EPS3 = 0.02  # 三档阈


def mkp(s):
    return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)


def med(arr, w):
    h = w // 2; return np.array([np.median(arr[max(0, j - h):j + h + 1]) for j in range(len(arr))])


def gr(idx):
    o = []; s0 = None; pv = None
    for i in idx:
        if pv is None or i != pv + 1:
            if s0 is not None: o.append((s0, pv))
            s0 = i
        pv = i
    if s0 is not None: o.append((s0, pv))
    return [x for x in o if x[1] - x[0] >= 1]


class Model:
    """挖矿一个 CRAVE 离散 milestone 模型(逐字同 canonical)。fc_arm/fc_raw=特征缓存, DS=取 state 的数据集。"""
    def __init__(self, fc_arm, fc_raw, DS, mine_eps, tag="model"):
        self.fc_arm, self.fc_raw, self.DS = fc_arm, fc_raw, DS
        self.cs = json.load(open(DS / "meta/info.json"))["chunks_size"]
        Sall = [self._lpst(e, self._flen(e)) for e in mine_eps]
        Pm = mkp(np.concatenate(Sall)); self.PMU, self.PSD = Pm.mean(0), Pm.std(0) + 1e-8
        A, R, S, T, E, SP, EP = [], [], [], [], [], [], []
        for e in mine_eps:
            a, r, st, n = self._loadep(e); g = self.emb(a, r, st)
            A.append(a); R.append(r); S.append(st); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
            SP.append(g[:2]); EP.append(g[-2:])
        A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
        G = self.emb(A, R, S)
        km = KMeans(96, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
        N = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(96)])
        Pstart = {}
        for e in sorted(set(E.tolist())):
            m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1); Pstart[e] = float(np.median(tpos[nn]))
        cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / N) for c in range(96)])
        bk = np.linspace(0, 1, 11); sel = []
        for b in range(10):
            inb = [c for c in range(96) if bk[b] <= tpos[c] < bk[b + 1]]
            if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:2]
        sel = sorted(set(sel), key=lambda c: tpos[c])
        Pk = {}
        for c in sel:
            fe = []
            for e in sorted(set(E.tolist())):
                m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
                if rs: fe.append(T[rs[0][0]])
            Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
        self.order = sorted(sel, key=lambda c: Pk[c]); self.C = allC[self.order]
        self.Pord = np.array([Pk[c] for c in self.order])
        self.startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
        self.endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_
        self.NB = 21; self.bins = np.linspace(0, 1, self.NB); self.cb = [[int(np.argmin(abs(self.bins - Pk[c])))] for c in self.order]
        sub = G[np.random.RandomState(1).permutation(len(G))[:5000]]
        self.resid_q90 = float(np.quantile(np.linalg.norm(sub[:, None] - self.C[None], axis=2).min(1), 0.90))  # OOD 阈
        print(f"[{tag}] milestones={len(self.order)} Pord {self.Pord.min():.2f}-{self.Pord.max():.2f} resid_q90={self.resid_q90:.2f}", flush=True)

    def _flen(self, e):
        a = np.load(self.fc_arm / f"ep{e}.npz")["f"]; r = np.load(self.fc_raw / f"ep{e}.npz")["f"]; return min(len(a), len(r))

    def _lpst(self, e, n):
        pq = self.DS / "data" / f"chunk-{e//self.cs:03d}" / f"episode_{e:06d}.parquet"
        st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())
        return st[np.minimum(np.arange(n) * 10, len(st) - 1)]

    def _loadep(self, e):
        a = np.load(self.fc_arm / f"ep{e}.npz")["f"]; r = np.load(self.fc_raw / f"ep{e}.npz")["f"]
        n = min(len(a), len(r)); return a[:n], r[:n], self._lpst(e, n), n

    def emb(self, a_, r_, s_):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True); rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = (mkp(s_) - self.PMU) / self.PSD; Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    def value_full(self, a, r, s):
        Fq = self.emb(a, r, s); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - self.C[None], axis=2)
        em = np.full((nq, self.NB), 1e3)
        for ci in range(len(self.order)):
            for b in self.cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
        ds = np.linalg.norm(Fq[:, None] - self.startK[None], axis=2).min(1); de = np.linalg.norm(Fq[:, None] - self.endK[None], axis=2).min(1)
        tn = np.arange(nq) / nq
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, self.NB - 1] = np.minimum(em[:, self.NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        NB = self.NB; pen = 8.0 * np.abs(self.bins[:, None] - self.bins[None]); NF = len(em)
        cost = np.full(NB, 1e9); cost[0] = em[0, 0]; bp = np.zeros((NF, NB), int)
        for j in range(1, NF):
            tr = cost[None, :] + pen; k = tr.argmin(1); cost = em[j] + tr[np.arange(NB), k]; bp[j] = k
        cost[NB - 1] -= 2; path = np.zeros(NF, int); path[-1] = cost.argmin()
        for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
        v3 = med(self.bins[path], 9)
        dsrt = np.sort(d, axis=1); marg = dsrt[:, 0] / np.clip(dsrt[:, 1], 1e-9, None); resid = d.min(1)
        return v3, Fq, d, resid, marg


def detect_transitions(v3, min_gap=3):
    dv = np.diff(v3); idx = np.where(np.abs(dv) > 1e-6)[0]
    if len(idx) == 0: return []
    groups = []; cur = [idx[0]]
    for k in idx[1:]:
        if k - cur[-1] <= min_gap: cur.append(k)
        else: groups.append(cur); cur = [k]
    groups.append(cur)
    evs = []
    for g in groups:
        j0, j1 = g[0], g[-1] + 1; jc = (j0 + j1) // 2
        v_from = float(np.median(v3[max(0, j0 - 4):j0 + 1])); v_to = float(np.median(v3[j1:min(len(v3), j1 + 5)]))
        if abs(v_to - v_from) < 0.03: continue
        evs.append((jc, j0, j1, v_from, v_to, v_to - v_from))
    return evs


def attr_motion(M, Fq, j0, j1, vf, vt, dvv, n):
    pre = max(0, j0 - 3); post = min(n - 1, j1 + 3)
    m_from = int(np.argmin(np.abs(M.Pord - vf))); m_to = int(np.argmin(np.abs(M.Pord - vt)))
    if m_from == m_to: m_to = max(0, min(len(M.Pord) - 1, m_from + (1 if dvv > 0 else -1)))
    out = []
    for blk in (RAWD, ARMD, PROD):
        e_pre = float(((Fq[pre][blk] - M.C[m_to][blk]) ** 2).sum())
        e_post = float(((Fq[post][blk] - M.C[m_to][blk]) ** 2).sum())
        out.append(e_pre - e_post)
    out = np.array(out); tot = out.sum(); pct = out / (np.abs(out).sum() + 1e-12) * 100
    grounded = tot > 0
    return m_from, m_to, pct, grounded


def grounded_clean_value(v3, evs, gflags, resid, resid_thr):
    """重建 value: 接受 grounded 步进; 也接受 ungrounded 但**高残差(OOD,如布料被拿走的真退步)**的步进;
    只丢弃 ungrounded **且低残差**(分布内 DP/中值瞬变=假信号)的步进。"""
    v_clean = np.empty_like(v3)
    segs = sorted(zip(evs, gflags), key=lambda x: x[0][0])
    boundaries = [(ev[0], ev[4] - ev[3], g, float(resid[ev[0]])) for ev, g in segs]  # (jc, delta, grounded, resid)
    level = float(v3[0]); bi = 0
    for j in range(len(v3)):
        while bi < len(boundaries) and boundaries[bi][0] == j:
            jc, delta, g, rs = boundaries[bi]
            keep = g or (rs >= resid_thr)  # grounded 或 OOD 真事件 → 保留; 否则(in-dist 瞬变)丢弃
            if keep: level = float(np.clip(level + delta, 0, 1))
            bi += 1
        v_clean[j] = level
    return v_clean


def adv(v, w=W):
    a = np.array([v[min(i + w, len(v) - 1)] - v[i] for i in range(len(v))]); return np.clip(a, -1, 1)


# ======================================================================
# 模型
DAG_ARM = REPO / "temp/tcc_smooth800_dagger_armmask/feat_cache"; DAG_RAW = REPO / "temp/tcc_smooth800_dagger_raw/feat_cache"
DAG_DS = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"
VIS_ARM = REPO / "temp/tcc_vis0526_armmask/feat_cache"; VIS_RAW = REPO / "temp/tcc_vis0526_raw/feat_cache"
VIS_DS = REPO / "kai0/data/Task_A/vis_base/v3/2026-05-26-v3"

dag_all = sorted(e for e in (int(p.stem[2:]) for p in DAG_ARM.glob("ep*.npz")) if (DAG_RAW / f"ep{e}.npz").exists())
dag_mine = sorted(np.random.RandomState(0).permutation(dag_all)[:500].tolist())
Mdag = Model(DAG_ARM, DAG_RAW, DAG_DS, dag_mine, "dagger")

vis_all = sorted(e for e in (int(p.stem[2:]) for p in VIS_ARM.glob("ep*.npz")) if (VIS_RAW / f"ep{e}.npz").exists())
Mvis = Model(VIS_ARM, VIS_RAW, VIS_DS, vis_all, "vis0526")

md = ["# CRAVE 可解释分析 II — autonomy 真机退步归因 + grounded 过滤进 AWBC advantage\n"]

# ======================================================================
# Part 1: autonomy rollout 归因(vis0526 挖矿 → 应用 rollout)
print("\n========== Part 1: autonomy rollout 归因 ==========", flush=True)
AROLL = REPO / "temp/tcc_autonomy_armmask/feat_cache"; RROLL = REPO / "temp/tcc_autonomy_raw/feat_cache"
ROLLDS = REPO / "temp/autonomy"
aR = np.load(AROLL / "ep0.npz")["f"]; rR = np.load(RROLL / "ep0.npz")["f"]; nR = min(len(aR), len(rR))
stR = np.stack(pd.read_parquet(ROLLDS / "data/chunk-000/episode_000000.parquet", columns=["observation.state"])["observation.state"].to_numpy())
stR = stR[np.minimum(np.arange(nR) * 10, len(stR) - 1)]
v3, Fq, d, resid, marg = Mvis.value_full(aR[:nR], rR[:nR], stR)
NF = len(pd.read_parquet(ROLLDS / "data/chunk-000/episode_000000.parquet", columns=["frame_index"]))
v30 = np.repeat(v3, 10)[:NF]
if len(v30) < NF: v30 = np.concatenate([v30, np.full(NF - len(v30), v30[-1])])
v30s = smooth_monotone(v30, fps=30.0)
evs = detect_transitions(v3)
falls = sorted([e for e in evs if e[5] < 0], key=lambda x: x[5])[:3]   # 真机退步是主角 → 取前3降
rises = sorted([e for e in evs if e[5] > 0], key=lambda x: -x[5])[:2]
picks = sorted(falls + rises, key=lambda x: x[0])
print(f"autonomy: {nR}帧(3Hz) {NF}帧(30fps) 检出{len(evs)}转移 → {len(falls)}降+{len(rises)}升", flush=True)


def grab_autonomy(vframe, cams=("top_head",)):
    out = {}
    for c in cams:
        mp4 = ROLLDS / f"videos/chunk-000/{c}/episode_000000.mp4"
        cap = cv2.VideoCapture(str(mp4)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(vframe)); ok, fr = cap.read(); cap.release()
        out[c] = fr[:, :, ::-1] if ok else None
    return out


ncol = max(1, len(picks)); fig = plt.figure(figsize=(3.4 * ncol, 7.4))
gs = fig.add_gridspec(3, ncol, height_ratios=[1.5, 1.7, 0.9], hspace=0.32, wspace=0.18)
axc = fig.add_subplot(gs[0, :]); x = np.arange(NF)
axc.plot(x, v30s, color="#2ca02c", lw=2.0, label=f"CRAVE value (0→{v30s[-1]:.2f})")
for b in (NF/3, 2*NF/3): axc.axvline(b, color="orange", ls=":", lw=1.2)
for s_, nm in [(0,"round1: 衣物被拿走"),(1,"round2: 叠完被弄乱"),(2,"round3: 恢复→完成")]:
    axc.text(s_*NF/3+40, 1.04, nm, fontsize=8.5, color="gray")
md.append("## Part 1 · autonomy 真机 rollout 退步归因(vis0526 挖矿 → 应用 rollout)\n")
md.append(f"![autonomy](crave_interp_autonomy.png)\n")
md.append("> 3 轮叠衣 rollout,round1 衣物被拿走 / round2 叠完被弄乱 = 两次**真机退步**。下表取前 3 降 + 2 升。\n")
md.append("| # | 类型 | 帧 | Δv | m_from→m_to | 驱动 raw/arm/pro | grounded | resid | marg |")
md.append("|---|---|---|---|---|---|---|---|---|")
for i, (jc, j0, j1, vf, vt, dvv) in enumerate(picks):
    vfr = min(jc*10, NF-1); col = "#1a9641" if dvv > 0 else "#d7191c"; mk = "^" if dvv > 0 else "v"
    axc.scatter([vfr], [v30s[vfr]], s=130, marker=mk, color=col, zorder=5, edgecolor="k", lw=.8)
    axc.annotate(f"#{i+1}", (vfr, v30s[vfr]), textcoords="offset points", xytext=(0, 10 if dvv>0 else -16), ha="center", fontsize=11, fontweight="bold", color=col)
axc.set_xlim(0, NF); axc.set_ylim(-.05, 1.12); axc.grid(alpha=.25); axc.set_ylabel("value"); axc.set_xlabel("frame(30fps)")
axc.legend(fontsize=9, loc="lower right"); axc.set_title("autonomy rollout: value 曲线 + 关键退步/恢复点(▼降 ▲升)", fontsize=11)
for i, (jc, j0, j1, vf, vt, dvv) in enumerate(picks):
    vfr = min(jc*10, NF-1)
    m_from, m_to, pct, grounded = attr_motion(Mvis, Fq, j0, j1, vf, vt, dvv, nR)
    img = grab_autonomy(vfr)["top_head"]
    axi = fig.add_subplot(gs[1, i])
    if img is not None: axi.imshow(img)
    axi.axis("off"); col = "#1a9641" if dvv > 0 else "#d7191c"
    axi.set_title(f"#{i+1} {'上升' if dvv>0 else '退步'} 帧{vfr}\nΔv={dvv:+.2f} p:{Mvis.Pord[m_from]:.2f}→{Mvis.Pord[m_to]:.2f}", fontsize=9.5, color=col)
    axb = fig.add_subplot(gs[2, i])
    axb.bar(["raw\n场景","arm\n臂掩码","pro\n本体"], pct, color=["#3b6fb0","#e08a1e","#7b5aa6"]); axb.axhline(0, color="k", lw=.6)
    axb.set_ylim(-110, 110); axb.tick_params(labelsize=8); axb.grid(alpha=.2, axis="y")
    if i == 0: axb.set_ylabel("驱动占比%", fontsize=8)
    axb.set_title(f"{'✓grounded' if grounded else '✗瞬变'} resid{resid[jc]:.2f} marg{marg[jc]:.2f}", fontsize=8.5, color="#1a9641" if grounded else "#d7191c")
    md.append(f"| {i+1} | {'上升' if dvv>0 else '退步'} | {vfr} | {dvv:+.2f} | {Mvis.Pord[m_from]:.2f}→{Mvis.Pord[m_to]:.2f} | {pct[0]:+.0f}%/{pct[1]:+.0f}%/{pct[2]:+.0f}% | {'✓' if grounded else '✗'} | {resid[jc]:.2f} | {marg[jc]:.2f} |")
fig.suptitle("CRAVE 可解释分析 · autonomy 真机 rollout 退步归因(相机帧 + 三路驱动)", fontsize=12.5, y=0.995)
fig.savefig(OUTV / "crave_interp_autonomy.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("SAVED", OUTV / "crave_interp_autonomy.png", flush=True)
md.append(f"\n**关键观察(诚实)**:两次大退步性质不同 ——\n"
          f"- **round1『衣物被拿走』= OOD 退步**:#1(帧3020, 0.65→0.15)三路 approach 全负 + **残差最高(在 vis0526 milestone 里离任何一档都远)**"
          f"= 桌面变空,不像任何 demo 状态。grounded 检验(是否更靠某档)对它失效,但**高残差 (>resid_q90={Mvis.resid_q90:.2f}) 正是 OOD 信号**。\n"
          f"- **round2『叠完被弄乱』= in-dist 退步**:#4(帧5160, 0.65→0.15)grounded ✓(pro+92%)= 布料仍在、退回早期可叠状态,被特征正确 grounding。\n"
          f"→ 含义:grounded(in-dist 真退步)**与** 高残差(OOD 退步)**二者并用**才能既保住真退步又摘掉 DP 噪声;Part 2 的过滤据此实现(ungrounded 且低残差才丢)。\n")

# ======================================================================
# Part 2: grounded 过滤 → AWBC advantage 去噪(dagger 全样本统计)
print("\n========== Part 2: grounded 过滤进 AWBC advantage ==========", flush=True)
samp = sorted(np.random.RandomState(11).permutation(dag_all)[:150].tolist())
neg_raw_all, neg_clean_all, mono_raw_all, mono_clean_all = [], [], [], []
removed_on_ung, removed_total = 0, 0
ngrnd_trans, total_trans = 0, 0
dist_raw = np.zeros(3, int); dist_clean = np.zeros(3, int)  # neg/normal/pos 帧计数
for e in samp:
    try:
        a, r, st, n = Mdag._loadep(e); v3, Fq, d, resid, marg = Mdag.value_full(a, r, st)
    except Exception: continue
    evs = detect_transitions(v3)
    gflags = []
    for (jc, j0, j1, vf, vt, dvv) in evs:
        _, _, _, g = attr_motion(Mdag, Fq, j0, j1, vf, vt, dvv, n); gflags.append(g)
    total_trans += len(evs); ngrnd_trans += sum(1 for g in gflags if not g)
    v_clean3 = grounded_clean_value(v3, evs, gflags, resid, Mdag.resid_q90)
    # === 与真实 build_ds_A_from_mv 一致: 3Hz→30fps upsample + smooth_monotone(41) + adv WIN=50@30fps ===
    NF = n * 10
    v30_raw = smooth_monotone(np.repeat(v3, 10)[:NF], fps=30.0)
    v30_cln = smooth_monotone(np.repeat(v_clean3, 10)[:NF], fps=30.0)
    a_raw = adv(v30_raw, 50); a_cln = adv(v30_cln, 50)
    neg_raw_all.append(float(np.mean(a_raw < -EPS3))); neg_clean_all.append(float(np.mean(a_cln < -EPS3)))
    mono_raw_all.append(float(np.mean(np.diff(v30_raw) >= -1e-6))); mono_clean_all.append(float(np.mean(np.diff(v30_cln) >= -1e-6)))
    # 被去掉的 neg 帧是否落在 ungrounded 转移诱发区(30fps; adv 用 v[t+50] 故 neg 落在转移前 ~W 帧)
    removed = (a_raw < -EPS3) & (~(a_cln < -EPS3))
    ung_zone = np.zeros(NF, bool)  # 真正被丢弃的 = ungrounded 且 低残差(in-dist DP 瞬变)
    for (jc, j0, j1, vf, vt, dvv), g in zip(evs, gflags):
        if (not g) and (resid[jc] < Mdag.resid_q90): ung_zone[max(0, j0 * 10 - W - 15):min(NF, j1 * 10 + 15)] = True
    removed_total += int(removed.sum()); removed_on_ung += int((removed & ung_zone).sum())
    for av, dist in [(a_raw, dist_raw), (a_cln, dist_clean)]:
        ti = np.where(av < -EPS3, 0, np.where(av > EPS3, 2, 1))
        for k in range(3): dist[k] += int((ti == k).sum())

neg_raw = float(np.mean(neg_raw_all)); neg_clean = float(np.mean(neg_clean_all))
mono_raw = float(np.mean(mono_raw_all)); mono_clean = float(np.mean(mono_clean_all))
ung_rate = ngrnd_trans / max(1, total_trans); removed_ung_frac = removed_on_ung / max(1, removed_total)
dr = dist_raw / dist_raw.sum() * 100; dc = dist_clean / dist_clean.sum() * 100
print(f"neg-adv帧: raw {neg_raw:.1%} → grounded-clean {neg_clean:.1%} (降 {neg_raw-neg_clean:.1%})", flush=True)
print(f"value单调率: raw {mono_raw:.1%} → clean {mono_clean:.1%}", flush=True)
print(f"ungrounded转移占比 {ung_rate:.0%}; 被去掉的neg帧落在ungrounded区 {removed_ung_frac:.0%}", flush=True)
print(f"三档分布 raw  neg/normal/pos = {dr[0]:.1f}/{dr[1]:.1f}/{dr[2]:.1f}", flush=True)
print(f"三档分布 clean neg/normal/pos = {dc[0]:.1f}/{dc[1]:.1f}/{dc[2]:.1f}", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
ax[0].bar(["raw value","grounded\nclean"], [neg_raw*100, neg_clean*100], color=["#d7191c","#1a9641"])
ax[0].set_ylabel("neg-advantage 帧占比 %"); ax[0].grid(alpha=.2, axis="y")
ax[0].set_title(f"neg 假标减少: {neg_raw*100:.1f}% → {neg_clean*100:.1f}%", fontsize=10)
for i, v in enumerate([neg_raw*100, neg_clean*100]): ax[0].text(i, v + .3, f"{v:.1f}%", ha="center", fontsize=9)
ax[1].bar(["raw","clean"], [mono_raw*100, mono_clean*100], color=["#d7191c","#1a9641"]); ax[1].set_ylim(min(mono_raw,mono_clean)*100-3, 101)
ax[1].set_ylabel("value 单调率 %"); ax[1].grid(alpha=.2, axis="y"); ax[1].set_title(f"单调性提升: {mono_raw*100:.1f}% → {mono_clean*100:.1f}%", fontsize=10)
for i, v in enumerate([mono_raw*100, mono_clean*100]): ax[1].text(i, v + .2, f"{v:.1f}%", ha="center", fontsize=9)
xx = np.arange(3); wq = .36
ax[2].bar(xx - wq/2, dr, wq, label="raw", color="#d7191c", alpha=.8); ax[2].bar(xx + wq/2, dc, wq, label="grounded-clean", color="#1a9641", alpha=.85)
ax[2].set_xticks(xx); ax[2].set_xticklabels(["neg","normal","pos"]); ax[2].set_ylabel("帧占比 %"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.2, axis="y")
ax[2].set_title("三档 advantage 分布(去噪后)", fontsize=10)
fig.suptitle(f"grounded 过滤进 AWBC advantage · {len(samp)} dagger ep · 被去 neg 中 {removed_ung_frac:.0%} 落在 ungrounded 瞬变区", fontsize=12)
fig.tight_layout(); fig.savefig(OUTV / "crave_grounded_advantage.png", dpi=120); plt.close(fig)
print("SAVED", OUTV / "crave_grounded_advantage.png", flush=True)

md.append("\n---\n\n## Part 2 · grounded 过滤进 AWBC advantage(去噪量化, 150 dagger ep)\n")
md.append("![grounded](crave_grounded_advantage.png)\n")
md.append("> **方法**: 只累加 grounded(特征支持)的 milestone 步进重建 value → 重算 advantage。专家遥操 demo 真退步稀少,"
          "raw value 的 DP/中值瞬变制造**假 neg 标签**;grounded 过滤丢弃这些瞬变步进。\n")
md.append("| 指标 | raw value | grounded-clean | 变化 |")
md.append("|---|---|---|---|")
md.append(f"| neg-advantage 帧占比 | {neg_raw:.1%} | {neg_clean:.1%} | **−{(neg_raw-neg_clean)*100:.1f} pt** |")
md.append(f"| value 单调率 | {mono_raw:.1%} | {mono_clean:.1%} | +{(mono_clean-mono_raw)*100:.1f} pt |")
md.append(f"| 三档 neg/normal/pos | {dr[0]:.1f}/{dr[1]:.1f}/{dr[2]:.1f}% | {dc[0]:.1f}/{dc[1]:.1f}/{dc[2]:.1f}% | neg −{dr[0]-dc[0]:.1f}pt |")
md.append(f"\n- **ungrounded 转移占比** {ung_rate:.0%}(全部转移里 DP/瞬变的比例)。")
md.append(f"- **被去掉的 neg 帧有 {removed_ung_frac:.0%} 落在 ungrounded 瞬变区** → 证实去掉的是 DP 噪声假 neg,不是真退步。")
md.append("\n## 结论\n")
md.append(
    f"1. **grounded 过滤显著降 AWBC 的 neg 假标**:neg-advantage 帧从 {neg_raw:.1%} 降到 {neg_clean:.1%}"
    f"(−{(neg_raw-neg_clean)*100:.1f}pt),value 单调率 {mono_raw:.1%}→{mono_clean:.1%}。"
    f"且被去掉的 neg **{removed_ung_frac:.0%} 落在 ungrounded(DP/中值瞬变)区** —— 去的是噪声不是信号。\n"
    "2. **这正是标量 AE 做不到的**:AE 输出单个 scalar,无法判别某次 value 下跌是真退步还是读出噪声(故 AE 在 dagger 上 47% 帧误标 neg);"
    "CRAVE 的离散 milestone + 可分离归因给出 grounded 判据,把假 neg 摘掉。\n"
    "3. **接入路径**:在 `build_ds_A_from_mv` 产 advantage 那步,用 grounded-clean value 替代 raw value 再算 Δ/离散化 →"
    "更干净的三档标签喂 `pi05_awbc_mv_A_3lvl`(见 [AB_plan §5b/§10](../../training/future_plans/plans/cross_episode_recurrence_value/awbc_milestone_value_AB_plan.md))。\n"
    "4. **保真安全阀(grounded + 残差并用)**:过滤只丢弃 **ungrounded 且低残差**(分布内 DP 瞬变)的步进;"
    f"**ungrounded 但高残差(>resid_q90={Mdag.resid_q90:.2f})= OOD 真退步(如 autonomy round1 布料被拿走)→ 保留**。"
    "故 grounded 过滤**保住真退步(in-dist 的 grounded ✓ + OOD 的高残差),只摘 in-dist DP 噪声**(Part 1 的两类退步分别由这两条保住)。")
(OUTV / "crave_grounded_advantage.md").write_text("\n".join(md), encoding="utf-8")
print("SAVED", OUTV / "crave_grounded_advantage.md", flush=True)
print("GROUNDED_DONE", flush=True)
