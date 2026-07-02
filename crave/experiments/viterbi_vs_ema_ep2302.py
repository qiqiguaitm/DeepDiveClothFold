#!/usr/bin/env python
"""ep2302: Viterbi-DP vs EMA 对比实验 — Viterbi 是动态规划还是 fancy smoother?

回答两个问题:
  1. 无 Viterbi 时 progress 跳变的根因是什么(真问题 vs 心理安慰)?
  2. EMA(指数滑动平均)能否替代 Viterbi? 如果不能,差在哪?
"""
import os, sys, numpy as np
from pathlib import Path

REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
DS = REPO / "kai0/data/Task_A/kai0_base"
FC = REPO / "temp/crave_kai0bd/feat_cache"   # armmask+raw+state @3Hz
OUT = REPO / "temp/viterbi_vs_ema_ep2302"; OUT.mkdir(exist_ok=True, parents=True)
EP = 2302; STRIDE = 10; csDS = 500

# ===== load features for ep2302 @3Hz =====
d = np.load(FC / f"ep{EP}.npz")
A = d["armmask"]; R = d["raw"]; S = d["state"]
n = min(len(A), len(R), len(S))
A, R, S = A[:n], R[:n], S[:n]
S = np.clip(np.nan_to_num(S.astype(np.float64)), -10, 10)
print(f"[load] ep{EP}: {n} frames @3Hz (≈{n*STRIDE} frames @30Hz)")

# ===== build embedding (same as interp_clusters) =====
def mkp(s, dt=1):
    d = np.zeros_like(s); d[dt:] = s[dt:] - s[:-dt]
    return np.concatenate([s, d], 1)

# load all kai0_base features for mining
all_eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
mine_pool = [e for e in all_eps if e < 3055]  # kai0_base only
rng = np.random.RandomState(0)
mined = sorted(rng.permutation(mine_pool)[:min(200, len(mine_pool))].tolist())
if EP not in mined: mined = sorted(mined + [EP])
print(f"[mine] {len(mined)} eps for mining", flush=True)

Sall = [];
for e in mined:
    d2 = np.load(FC / f"ep{e}.npz"); s2 = d2["state"]
    s2 = np.clip(np.nan_to_num(s2.astype(np.float64)), -10, 10)
    Sall.append(s2)
Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

def emb(aa, rr, st):
    an = aa / (np.linalg.norm(aa, axis=1, keepdims=True) + 1e-8)
    rn = rr / (np.linalg.norm(rr, axis=1, keepdims=True) + 1e-8)
    Pn = (mkp(st) - PMU) / PSD
    Pn /= (np.linalg.norm(Pn, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([rn, an, Pn], 1)

# mine features (all mined episodes) + track per-frame episode
G_mine = []; E_mine = []
for e in mined:
    d2 = np.load(FC / f"ep{e}.npz")
    aa, rr, st = d2["armmask"][:], d2["raw"][:], d2["state"][:]
    st = np.clip(np.nan_to_num(st.astype(np.float64)), -10, 10)
    n2 = min(len(aa), len(rr), len(st))
    G_mine.append(emb(aa[:n2], rr[:n2], st[:n2]))
    E_mine.append(np.full(n2, e))
G_all = np.concatenate(G_mine); E_all = np.concatenate(E_mine)

# ===== KMeans → milestones =====
from sklearn.cluster import KMeans
K0 = 96
km = KMeans(K0, n_init=2, random_state=0).fit(G_all); lab = km.labels_; C_all = km.cluster_centers_
N_ep = len(mined); T_mine = np.concatenate([np.arange(len(g)) / max(1, len(g)-1) for g in G_mine])
tpos = np.array([T_mine[lab == c].mean() if (lab == c).any() else 0.5 for c in range(K0)])

# Select milestones by coverage
covE = np.array([len(set(E_all[lab == c].tolist())) / N_ep if (lab == c).any() else 0.0 for c in range(K0)])
TAU = float(np.quantile(covE, 0.5))
sel = sorted([c for c in range(K0) if covE[c] >= TAU], key=lambda c: tpos[c])
C = C_all[sel]; Pord = tpos[sel]; NM = len(sel)
print(f"[milestones] {NM} selected from K={K0} (coverage≥{TAU:.2f})", flush=True)
print(f"  Pord range: [{Pord.min():.3f}, {Pord.max():.3f}]", flush=True)

# ===== ep2302: compute distances to each milestone =====
Ge = emb(A, R, S)
dist = np.linalg.norm(Ge[:, None] - C[None], axis=2)  # (n, NM)

# ① Raw argmin progress
raw_idx = dist.argmin(1)
raw_progress = Pord[raw_idx]

# ② EMA variants
def ema(x, alpha):
    y = np.zeros_like(x); y[0] = x[0]
    for i in range(1, len(x)): y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

ema_hard = ema(raw_progress, alpha=0.3)   # 快速响应(~3f有效窗)
ema_med  = ema(raw_progress, alpha=0.1)   # 中等(~10f有效窗)
ema_soft = ema(raw_progress, alpha=0.05)  # 慢速(~20f有效窗)

# ③ Viterbi-DP (same as dpHB)
NB = 21; b = np.linspace(0, 1, NB)
LAM = 8.0; MEDW = 9

# emission field: map each milestone to nearest bin
em = np.full((n, NB), 1e3)
for ci in range(NM):
    bin_idx = np.abs(b - Pord[ci]).argmin()
    em[:, bin_idx] = np.minimum(em[:, bin_idx], dist[:, ci])

# anchor: start/end
ds = dist[:, np.abs(Pord).argmin()]; de = dist[:, np.abs(Pord - 1.0).argmin()]
tn = np.arange(n) / n
em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
em[:, NB-1] = np.minimum(em[:, NB-1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))

# Viterbi forward + backward
pen = LAM * np.abs(b[:, None] - b[None])
cost = np.full(NB, 1e9); cost[0] = em[0, 0]
bp_trace = np.zeros((n, NB), int)
for j in range(1, n):
    tr = cost[None, :] + pen; k = tr.argmin(1)
    cost = em[j] + tr[np.arange(NB), k]; bp_trace[j] = k
cost[NB-1] -= 2.0
path = np.zeros(n, int); path[-1] = cost.argmin()
for j in range(n-2, -1, -1): path[j] = bp_trace[j+1, path[j+1]]
viterbi_progress = b[path]

# median post-filter (same as interp_clusters)
from scipy.ndimage import median_filter
viterbi_med = median_filter(viterbi_progress, MEDW)

# ===== Metrics =====
def mono_rate(v):
    return np.mean(np.diff(v) >= -0.001)

def jitter(v):
    return np.mean(np.abs(np.diff(v)))

def n_flips(v, eps=0.02):
    """count sign flips of meaningful magnitude"""
    d = np.diff(v); flips = 0
    for i in range(1, len(d)):
        if d[i] * d[i-1] < 0 and abs(d[i]) > eps and abs(d[i-1]) > eps:
            flips += 1
    return flips

print(f"\n{'='*60}")
print(f"{'Method':<25s} {'mono':>8s} {'jitter':>8s} {'flips':>6s} {'end_v':>8s} {'start_v':>8s}")
print(f"{'-'*60}")
for name, v in [("raw argmin", raw_progress),
                ("EMA α=0.30 (fast)", ema_hard),
                ("EMA α=0.10 (medium)", ema_med),
                ("EMA α=0.05 (slow)", ema_soft),
                ("Viterbi-DP (raw)", viterbi_progress),
                ("Viterbi-DP + med", viterbi_med)]:
    print(f"{name:<25s} {mono_rate(v):8.4f} {jitter(v):8.4f} {n_flips(v):6d} {v[-1]:8.4f} {v[0]:8.4f}")

# ===== Plot =====
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
t = np.arange(n) * STRIDE / 30.0  # seconds

# Panel A: Raw vs EMA vs Viterbi
ax = axes[0]
ax.plot(t, raw_progress, 'gray', alpha=0.35, lw=0.6, label='raw argmin (mono={:.3f})'.format(mono_rate(raw_progress)))
ax.plot(t, ema_med, '#e8a030', alpha=0.7, lw=1.2, label='EMA α=0.10 (mono={:.3f})'.format(mono_rate(ema_med)))
ax.plot(t, viterbi_med, '#1a9641', lw=2.0, label='Viterbi-DP+med (mono={:.3f})'.format(mono_rate(viterbi_med)))
ax.set_ylabel('progress'); ax.set_title('ep2302: raw argmin vs EMA vs Viterbi-DP'); ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.2)

# Panel B: EMA variants sweep
ax = axes[1]
ax.plot(t, raw_progress, 'gray', alpha=0.25, lw=0.6, label='raw')
ax.plot(t, ema_hard, '#d62728', alpha=0.6, lw=1.0, label='EMA α=0.30 fast (mono={:.3f})'.format(mono_rate(ema_hard)))
ax.plot(t, ema_med, '#e8a030', alpha=0.7, lw=1.2, label='EMA α=0.10 med (mono={:.3f})'.format(mono_rate(ema_med)))
ax.plot(t, ema_soft, '#2ca02c', alpha=0.7, lw=1.2, label='EMA α=0.05 slow (mono={:.3f})'.format(mono_rate(ema_soft)))
ax.set_ylabel('progress'); ax.set_title('EMA variants across α'); ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.2)

# Panel C: Viterbi raw vs +med filter
ax = axes[2]
ax.plot(t, viterbi_progress, '#9467bd', alpha=0.7, lw=1.0, label='Viterbi raw (mono={:.3f})'.format(mono_rate(viterbi_progress)))
ax.plot(t, viterbi_med, '#1a9641', lw=2.0, label='Viterbi+med9 (mono={:.3f})'.format(mono_rate(viterbi_med)))
ax.set_xlabel('time (s)'); ax.set_ylabel('progress')
ax.set_title('Viterbi: raw DP output vs +median filter'); ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.2)

fig.tight_layout(); fig.savefig(OUT / "viterbi_vs_ema_ep2302.png", dpi=130, bbox_inches="tight")
print(f"\nSAVED {OUT / 'viterbi_vs_ema_ep2302.png'}", flush=True)

# ===== Bonus: check the Voronoi jitter mechanism =====
# For each frame transition, check: did the argmin change cluster? And by how much in Pord?
raw_changes = np.where(np.diff(raw_idx) != 0)[0]
pord_jumps = np.abs(np.diff(raw_progress))[raw_changes]
print(f"\n[Mechanism check] {len(raw_changes)}/{n-1} frame transitions change cluster")
print(f"  Pord jump magnitude: mean={pord_jumps.mean():.3f} median={np.median(pord_jumps):.3f} max={pord_jumps.max():.3f}")
print(f"  Large jumps (>0.10): {np.mean(pord_jumps > 0.10)*100:.1f}%")
print(f"  Fraction of frames that change cluster: {len(raw_changes)/(n-1)*100:.1f}%")

# Save intermediate data
np.savez(OUT / "comparison_data.npz",
         raw_progress=raw_progress, raw_idx=raw_idx,
         ema_hard=ema_hard, ema_med=ema_med, ema_soft=ema_soft,
         viterbi_progress=viterbi_progress, viterbi_med=viterbi_med,
         Pord=Pord, dist=dist, C=C, t=t)
print(f"SAVED {OUT / 'comparison_data.npz'}", flush=True)
