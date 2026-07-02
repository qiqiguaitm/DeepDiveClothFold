#!/usr/bin/env python
"""ep2302: 深入分析 raw argmin 跳变机制。

回答: 在没有 Viterbi 的情况下,逐帧 argmin 为什么跳? 跳变时发生了什么?

分析维度:
  A. 每次跳变的 winner-runner 间距 (margin) — 是"差一点点"还是"差很多"?
  B. 跳变的 Pord 跨度 — 跳到相近簇 vs 跳到远处?
  C. 连续帧的特征变化 vs 簇间距离 — 特征走一小步,argmin 跳一大步?
  D. 每个簇的"势力范围"稳定性 — 哪些区域容易跳?
  E. 如果只看 top-3 候选,有多少跳变可以避免?
"""
import os, sys, numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
FC = REPO / "temp/crave_kai0bd/feat_cache"
OUT = REPO / "temp/argmin_jump_analysis_ep2302"; OUT.mkdir(exist_ok=True, parents=True)
EP = 2302; STRIDE = 10

# ===== load features + build KMeans model (same as before) =====
def load_ep(e):
    d = np.load(FC / f"ep{e}.npz"); a, r, s = d["armmask"], d["raw"], d["state"]
    n = min(len(a), len(r), len(s)); s = np.clip(np.nan_to_num(s[:n].astype(np.float64)), -10, 10)
    return a[:n], r[:n], s, n

def mkp(s, dt=1):
    d = np.zeros_like(s); d[dt:] = s[dt:] - s[:-dt]; return np.concatenate([s, d], 1)

all_eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
mine_pool = [e for e in all_eps if e < 3055]
rng = np.random.RandomState(0); N_MINE = 200
mined = sorted(rng.permutation(mine_pool)[:min(N_MINE, len(mine_pool))].tolist())
if EP not in mined: mined = sorted(mined + [EP])

Sall = []; feature_list = []
for e in mined:
    aa, rr, st, n = load_ep(e); Sall.append(st)
    an = aa / (np.linalg.norm(aa, axis=1, keepdims=True) + 1e-8)
    rn = rr / (np.linalg.norm(rr, axis=1, keepdims=True) + 1e-8)
    feature_list.append(np.concatenate([rn, an], 1))

Pm = mkp(np.concatenate(Sall)); PMU, PSD = Pm.mean(0), Pm.std(0) + 1e-8

def emb(aa, rr, st):
    """Embed without state (768D = 384+384), matching KMeans model training."""
    an = aa / (np.linalg.norm(aa, axis=1, keepdims=True) + 1e-8)
    rn = rr / (np.linalg.norm(rr, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([rn, an], 1)

G_all = np.concatenate(feature_list)
km = KMeans(96, n_init=2, random_state=0).fit(G_all)
C_all = km.cluster_centers_; lab = km.labels_
T_all = np.concatenate([np.arange(len(f)) / max(1, len(f)-1) for f in feature_list])
E_all = np.concatenate([np.full(len(f), mined[i]) for i, f in enumerate(feature_list)])
tpos = np.array([T_all[lab == c].mean() if (lab == c).any() else 0.5 for c in range(96)])
covE = np.array([len(set(E_all[lab == c])) / N_MINE if (lab == c).any() else 0.0 for c in range(96)])
TAU = float(np.quantile(covE, 0.5))
sel = sorted([c for c in range(96) if covE[c] >= TAU], key=lambda c: tpos[c])
C = C_all[sel]; Pord = tpos[sel]; NM = len(sel)
print(f"[model] {NM} milestones from K=96, coverage threshold={TAU:.2f}", flush=True)

# ===== ep2302: compute per-frame distances =====
aa, rr, st, n_feat = load_ep(EP)
Ge = emb(aa, rr, st)
dist = np.linalg.norm(Ge[:, None] - C[None], axis=2)  # (n_feat, NM)
raw_idx = dist.argmin(1)       # selected cluster per frame
raw_progress = Pord[raw_idx]   # corresponding progress

# top-3 sorted distances per frame
top3_idx = np.argpartition(dist, 3, axis=1)[:, :3]
top3_sorted = np.take_along_axis(dist, np.argsort(dist, axis=1)[:, :3], axis=1)
top3_clusters = np.take_along_axis(dist, top3_idx, axis=1)
# actually: we want sorted
sorted_dist = np.sort(dist, axis=1)     # (n, NM)
sorted_idx = np.argsort(dist, axis=1)   # (n, NM)

margin = sorted_dist[:, 1] - sorted_dist[:, 0]  # how close is runner-up?
runner_up_idx = sorted_idx[:, 1]
runner_up_pord = Pord[runner_up_idx]
rank2_dist = sorted_dist[:, 1]

print(f"\n[ep{EP}] {n_feat} frames @3Hz (≈{n_feat*STRIDE} frames @30Hz)")
print(f"  dist range: [{dist.min():.4f}, {dist.max():.4f}]")
print(f"  margin (1st-2nd) range: [{margin.min():.4f}, {margin.max():.4f}] median={np.median(margin):.4f}")
print(f"  margin < 0.02: {np.mean(margin < 0.02)*100:.1f}%  (very close calls)")
print(f"  margin < 0.05: {np.mean(margin < 0.05)*100:.1f}%  (close calls)")
print(f"  margin < 0.10: {np.mean(margin < 0.10)*100:.1f}%  (moderate calls)")

# ===== A. Jump analysis =====
jump_mask = np.diff(raw_idx) != 0
jump_frames = np.where(jump_mask)[0]  # indices where frame t→t+1 changes cluster
n_jumps = len(jump_frames)
print(f"\n[A] Jumps: {n_jumps}/{n_feat-1} transitions ({n_jumps/(n_feat-1)*100:.1f}%)")

jump_data = []
for jf in jump_frames:
    old_c = raw_idx[jf]; new_c = raw_idx[jf+1]
    old_p = Pord[old_c]; new_p = Pord[new_c]
    pord_jump = abs(new_p - old_p)
    # margin at frame t (before jump) and t+1 (after jump)
    m_before = margin[jf]
    m_after = margin[jf+1]
    # feature delta
    feat_delta = np.linalg.norm(Ge[jf+1] - Ge[jf])
    # distance to old/new centers
    d_old_before = dist[jf, old_c]; d_new_before = dist[jf, new_c]
    d_old_after = dist[jf+1, old_c]; d_new_after = dist[jf+1, new_c]
    # flip type: which one was already close before?
    flip_type = "crossing" if d_new_before < d_old_before else \
                ("drift" if feat_delta > np.median(margin) else "noise")
    jump_data.append({
        'frame': jf, 'old_c': old_c, 'new_c': new_c,
        'old_p': old_p, 'new_p': new_p, 'pord_jump': pord_jump,
        'margin_before': m_before, 'margin_after': m_after,
        'feat_delta': feat_delta,
        'd_old_before': d_old_before, 'd_new_before': d_new_before,
        'd_old_after': d_old_after, 'd_new_after': d_new_after,
    })

jump_pord = np.array([j['pord_jump'] for j in jump_data])
jump_margin = np.array([j['margin_before'] for j in jump_data])
jump_feat = np.array([j['feat_delta'] for j in jump_data])

print(f"  Pord jump: mean={jump_pord.mean():.3f} median={np.median(jump_pord):.3f} max={jump_pord.max():.3f}")
print(f"  margin_before jump: mean={jump_margin.mean():.4f} median={np.median(jump_margin):.4f}")
print(f"  Large jumps (>0.10 Pord): {np.mean(jump_pord > 0.10)*100:.1f}%")
print(f"  Small margin (<0.05) + Large Pord jump (>0.10): {np.mean((jump_margin < 0.05) & (jump_pord > 0.10))*100:.1f}%")
print(f"  feature delta: mean={jump_feat.mean():.4f} median={np.median(jump_feat):.4f}")

# ===== B. Per-frame: what's the runner-up's Pord? =====
runner_pord_gap = np.abs(Pord[raw_idx] - runner_up_pord)
print(f"\n[B] Runner-up analysis:")
print(f"  runner Pord gap: mean={runner_pord_gap.mean():.3f} median={np.median(runner_pord_gap):.3f}")
print(f"  runner Pord gap > 0.10: {np.mean(runner_pord_gap > 0.10)*100:.1f}%  (distant runner-up)")
print(f"  runner Pord gap > 0.20: {np.mean(runner_pord_gap > 0.20)*100:.1f}%  (very distant runner-up)")
print(f"  At jump frames: runner gap mean={runner_pord_gap[jump_frames].mean():.3f}")

# ===== C. Feature space trajectory smoothness =====
feat_step = np.linalg.norm(np.diff(Ge, axis=0), axis=1)
# Inter-cluster distances (all pairs)
C_dist = np.linalg.norm(C[:, None] - C[None], axis=2)
avg_C_dist = C_dist[np.triu_indices(NM, 1)].mean()
print(f"\n[C] Feature space geometry:")
print(f"  mean feature step: {feat_step.mean():.4f}  median: {np.median(feat_step):.4f}")
print(f"  mean inter-cluster distance: {avg_C_dist:.4f}")
print(f"  step/ICD ratio: {feat_step.mean()/avg_C_dist:.4f}  (small=features move slowly between clusters)")

# ===== D. "Soft" value: weighted by inverse distance (top-5) =====
def soft_value(dist, Pord, topk=5, temp=1.0):
    """Weighted average progress using inverse-distance softmax over top-K."""
    idx = np.argpartition(dist, topk, axis=1)[:, :topk]
    d_top = np.take_along_axis(dist, idx, axis=1)
    w = np.exp(-d_top / temp)
    w /= w.sum(1, keepdims=True)
    p_top = Pord[idx]
    return np.sum(w * p_top, axis=1), idx

soft_v, soft_idx = soft_value(dist, Pord, topk=5, temp=0.05)
soft_jitter = np.mean(np.abs(np.diff(soft_v)))

print(f"\n[D] Soft assignment (top-5, τ=0.05):")
print(f"  jitter: {soft_jitter:.4f} (vs raw argmin {np.mean(np.abs(np.diff(raw_progress))):.4f})")
print(f"  mono: {np.mean(np.diff(soft_v) >= -0.001):.4f} (vs raw {np.mean(np.diff(raw_progress) >= -0.001):.4f})")
print(f"  end value: {soft_v[-1]:.4f} (vs raw {raw_progress[-1]:.4f})")

# ===== PLOTS =====
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

t_sec = np.arange(n_feat) * STRIDE / 30.0

# (A) Distance to top-5 clusters over time
ax = fig.add_subplot(gs[0, :])
colors = plt.cm.tab10(np.linspace(0, 1, 5))
for i in range(5):
    ax.plot(t_sec, sorted_dist[:, i], color=colors[i], lw=0.5, alpha=0.7,
            label=f'rank {i+1} (median={np.median(sorted_dist[:, i]):.3f})')
ax.set_xlabel('time (s)'); ax.set_ylabel('L2 distance')
ax.set_title(f'(A) ep{EP}: distance to top-5 closest milestones over time')
ax.legend(fontsize=7, ncol=5); ax.grid(alpha=0.2)

# (B) Margin (1st - 2nd) + jumps marked
ax = fig.add_subplot(gs[1, 0])
ax.plot(t_sec, margin, 'k-', lw=0.4, alpha=0.6)
ax.scatter(t_sec[jump_frames], margin[jump_frames], c='red', s=4, alpha=0.5, zorder=5)
ax.axhline(0.02, color='orange', ls='--', lw=0.8, label='0.02 (very close)')
ax.axhline(0.05, color='red', ls='--', lw=0.8, label='0.05 (close)')
ax.axhline(np.median(margin), color='blue', ls=':', lw=1, label=f'median={np.median(margin):.3f}')
ax.set_xlabel('time (s)'); ax.set_ylabel('margin (dist_2nd - dist_1st)')
ax.set_title(f'(B) Winner-runner margin: {np.mean(margin < 0.05)*100:.0f}% < 0.05')
ax.legend(fontsize=7); ax.grid(alpha=0.2)

# (C) Raw argmin progress with jump Pord-gap coloring
ax = fig.add_subplot(gs[1, 1])
ax.plot(t_sec, raw_progress, 'gray', lw=0.5, alpha=0.5)
sc = ax.scatter(t_sec[jump_frames], raw_progress[jump_frames],
                c=jump_pord, cmap='Reds', s=8, alpha=0.7, vmin=0, vmax=0.5)
ax.scatter(t_sec[jump_frames+1], raw_progress[jump_frames+1],
           c=jump_pord, cmap='Reds', s=8, alpha=0.4, vmin=0, vmax=0.5)
plt.colorbar(sc, ax=ax, label='Pord jump magnitude')
ax.set_xlabel('time (s)'); ax.set_ylabel('progress')
ax.set_title(f'(C) Raw argmin: {n_jumps} jumps, {np.mean(jump_pord>0.1)*100:.0f}% are >0.10 Pord')
ax.grid(alpha=0.2)

# (D) Pord jump vs margin scatter
ax = fig.add_subplot(gs[1, 2])
ax.scatter(jump_margin, jump_pord, alpha=0.5, s=12, c='#d62728')
ax.axhline(0.10, color='orange', ls='--', lw=0.8, label='Pord gap 0.10')
ax.axvline(0.05, color='blue', ls='--', lw=0.8, label='margin 0.05')
# Quadrant stats
q_ll = np.mean((jump_margin < 0.05) & (jump_pord < 0.10))
q_lh = np.mean((jump_margin < 0.05) & (jump_pord > 0.10))
q_hl = np.mean((jump_margin > 0.05) & (jump_pord > 0.10))
ax.text(0.25, 0.95, f'low margin+small gap: {q_ll:.0%}\nlow margin+LARGE gap: {q_lh:.0%}\nhigh margin+large gap: {q_hl:.0%}',
        transform=ax.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('margin (1st-2nd distance)'); ax.set_ylabel('Pord jump')
ax.set_title(f'(D) Pord jump vs margin: {q_lh*100:.0f}% are close-call large-jump')
ax.legend(fontsize=7); ax.grid(alpha=0.2)

# (E) Running: raw vs soft vs Viterbi
ax = fig.add_subplot(gs[2, 0])
# Quick Viterbi for comparison
from scipy.ndimage import median_filter
LAM = 8.0; NB = 21; b = np.linspace(0, 1, NB)
em = np.full((n_feat, NB), 1e3)
for ci in range(NM):
    bi = np.abs(b - Pord[ci]).argmin(); em[:, bi] = np.minimum(em[:, bi], dist[:, ci])
ds = dist[:, np.abs(Pord).argmin()]; de = dist[:, np.abs(Pord-1.0).argmin()]
tn = np.arange(n_feat)/n_feat
em[:,0] = np.minimum(em[:,0], np.where(tn<0.3, ds, ds+(tn-0.3)*6))
em[:,-1] = np.minimum(em[:,-1], np.where(tn>0.6, de, de+(0.6-tn)*6))
pen = LAM * np.abs(b[:,None]-b[None])
cost = np.full(NB,1e9); cost[0]=em[0,0]; bp_v=np.zeros((n_feat,NB),int)
for j in range(1,n_feat): tr=cost[None,:]+pen; k=tr.argmin(1); cost=em[j]+tr[np.arange(NB),k]; bp_v[j]=k
cost[-1]-=2; path=np.zeros(n_feat,int); path[-1]=cost.argmin()
for j in range(n_feat-2,-1,-1): path[j]=bp_v[j+1,path[j+1]]
v_vit = median_filter(b[path], 9)

ax.plot(t_sec, raw_progress, 'gray', lw=0.3, alpha=0.5, label=f'raw argmin (mono={np.mean(np.diff(raw_progress)>=-0.001):.3f})')
ax.plot(t_sec, soft_v, '#e8a030', lw=1.2, alpha=0.8, label=f'soft top-5 τ=0.05 (mono={np.mean(np.diff(soft_v)>=-0.001):.3f})')
ax.plot(t_sec, v_vit, '#1a9641', lw=1.5, label=f'Viterbi (mono={np.mean(np.diff(v_vit)>=-0.001):.3f})')
ax.set_xlabel('time (s)'); ax.set_ylabel('progress')
ax.set_title('(E) Comparison: raw vs soft vs Viterbi'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

# (F) Feature space geometry: intra-cluster distance vs inter-cluster
ax = fig.add_subplot(gs[2, 1])
# For each frame, compute: distance to assigned cluster vs avg distance to all clusters
intra = np.array([dist[i, raw_idx[i]] for i in range(n_feat)])
inter = dist.mean(1)
ax.plot(t_sec, intra, 'b-', lw=0.5, alpha=0.5, label=f'intra-cluster (assigned) median={np.median(intra):.3f}')
ax.plot(t_sec, inter, 'r-', lw=0.5, alpha=0.5, label=f'inter-cluster (avg all) median={np.median(inter):.3f}')
ratio = intra / inter
ax2 = ax.twinx()
ax2.plot(t_sec, ratio, 'gray', lw=0.3, alpha=0.5)
ax2.set_ylabel('intra/inter ratio', color='gray')
ax.set_xlabel('time (s)'); ax.set_ylabel('distance')
ax.set_title(f'(F) Intra vs inter cluster distance (ratio median={np.median(ratio):.3f})')
ax.legend(fontsize=7); ax.grid(alpha=0.2)

# (G) Jump Pord distribution
ax = fig.add_subplot(gs[2, 2])
bins_jump = np.linspace(0, 0.8, 40)
ax.hist(jump_pord, bins=bins_jump, color='#d62728', alpha=0.7, edgecolor='black', lw=0.3)
ax.axvline(np.median(jump_pord), color='black', ls='--', lw=1.5, label=f'median={np.median(jump_pord):.3f}')
ax.set_xlabel('Pord jump'); ax.set_ylabel('count')
ax.set_title(f'(G) Pord jump distribution ({n_jumps} jumps)')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

fig.suptitle(f'ep{EP} Raw Argmin Jump Mechanism Analysis\n'
             f'{NM} milestones (KMeans K=96), {N_MINE} training episodes, 3Hz',
             fontsize=13, fontweight='bold')
fig.savefig(OUT / "jump_mechanism_analysis.png", dpi=150, bbox_inches="tight")
print(f"\nSAVED {OUT / 'jump_mechanism_analysis.png'}", flush=True)

# ===== Save detailed data =====
np.savez(OUT / "jump_analysis_data.npz",
         dist=dist, raw_idx=raw_idx, raw_progress=raw_progress,
         sorted_dist=sorted_dist, sorted_idx=sorted_idx, margin=margin,
         jump_frames=jump_frames, jump_pord=jump_pord, jump_margin=jump_margin,
         jump_feat=jump_feat, Pord=Pord, C=C, soft_v=soft_v, v_vit=v_vit,
         t_sec=t_sec)
print(f"SAVED {OUT / 'jump_analysis_data.npz'}", flush=True)

# ===== Detailed per-jump report (top 20 worst jumps) =====
print("\n" + "="*100)
print("TOP 20 WORST JUMPS (largest Pord gap):")
print(f"{'frame':>6s} {'time':>7s} {'old_c':>5s} {'new_c':>5s} {'old_p':>7s} {'new_p':>7s} {'Δp':>7s} {'margin':>8s} {'feat_Δ':>8s} {'type':>10s}")
print("-"*100)
sorted_jumps = sorted(jump_data, key=lambda j: -j['pord_jump'])
for j in sorted_jumps[:20]:
    ftype = "crossing" if j['d_new_before'] < j['d_old_before'] else \
            ("drift" if j['feat_delta'] > np.median(margin) else "noise")
    print(f"{j['frame']:6d} {t_sec[j['frame']]:7.1f}s {j['old_c']:5d} {j['new_c']:5d} "
          f"{j['old_p']:7.3f} {j['new_p']:7.3f} {j['pord_jump']:7.3f} "
          f"{j['margin_before']:8.4f} {j['feat_delta']:8.4f} {ftype:>10s}")

print(f"\n[DONE] See {OUT}/ for full analysis")
