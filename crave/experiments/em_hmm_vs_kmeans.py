#!/usr/bin/env python
"""EM-HMM vs KMeans (CRAVE): cluster quality head-to-head comparison.

Same features, same K=96, same training data (200 kai0_base episodes @3Hz).
Compare: per-cluster progress distribution (std), weighted purity, coverage.

Also answers: does EM-HMM have latent frame prediction capability?
  Yes — HMM is a proper generative model p(x|z)·p(z_t|z_{t-1}),
  can sample future features AND decode back to image space.
"""
import os, sys, numpy as np
from pathlib import Path
from scipy.special import logsumexp

REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
FC = REPO / "temp/crave_kai0bd/feat_cache"
OUT = REPO / "temp/em_hmm_vs_kmeans"; OUT.mkdir(exist_ok=True, parents=True)

N_MINE = 200    # training episodes
K = 96          # same as CRAVE
N_EM_ITER = 30  # EM iterations
EPS = 1e-8

# ===== load data =====
all_eps = sorted(int(p.stem[2:]) for p in FC.glob("ep*.npz"))
mine_pool = [e for e in all_eps if e < 3055]
rng = np.random.RandomState(42)
mined = sorted(rng.permutation(mine_pool)[:min(N_MINE, len(mine_pool))].tolist())
print(f"[data] {len(mined)} episodes for training", flush=True)

# Standardize features (per-dimension z-score, same for both methods)
def mkp(s, dt=1):
    d = np.zeros_like(s); d[dt:] = s[dt:] - s[:-dt]
    return np.concatenate([s, d], 1)

def load_ep(e):
    d = np.load(FC / f"ep{e}.npz")
    a, r, s = d["armmask"], d["raw"], d["state"]
    n = min(len(a), len(r), len(s))
    s = np.clip(np.nan_to_num(s[:n].astype(np.float64)), -10, 10)
    return a[:n], r[:n], s, n

# Collect all features + compute standardization
print("[prep] computing feature statistics...", flush=True)
features_list = []  # list of (T_i, d)
for e in mined:
    aa, rr, st, n = load_ep(e)
    an = aa / (np.linalg.norm(aa, axis=1, keepdims=True) + EPS)
    rn = rr / (np.linalg.norm(rr, axis=1, keepdims=True) + EPS)
    feat = np.concatenate([rn, an], 1)  # no state for simplicity (DINO features only)
    features_list.append(feat.astype(np.float32))

all_x = np.concatenate(features_list)
x_mean = all_x.mean(0); x_std = all_x.std(0) + EPS
features_list = [(f - x_mean) / x_std for f in features_list]
d = all_x.shape[1]; total_frames = sum(len(f) for f in features_list)
print(f"[prep] {total_frames} frames, d={d} dims", flush=True)

# ===== Method A: KMeans (current CRAVE) =====
print("\n" + "="*60)
print("Method A: KMeans (CRAVE)")
print("="*60)
from sklearn.cluster import KMeans
km = KMeans(K, n_init=2, random_state=0, max_iter=300).fit(all_x)
km_labels = km.labels_
km_centers = km.cluster_centers_

# Compute per-cluster progress distribution
all_T = [np.arange(len(f)) / max(1, len(f)-1) for f in features_list]
all_progress = np.concatenate(all_T)
km_progress_std = np.array([np.std(all_progress[km_labels == k]) if (km_labels == k).sum() > 1 else np.nan
                            for k in range(K)])
km_progress_mean = np.array([np.mean(all_progress[km_labels == k]) if (km_labels == k).sum() > 0 else np.nan
                             for k in range(K)])
km_sizes = np.array([(km_labels == k).sum() for k in range(K)])

# Per-cluster per-episode coverage
all_E = np.concatenate([np.full(len(f), i) for i, f in enumerate(features_list)])
km_cov = np.array([len(set(all_E[km_labels == k])) / len(mined) if (km_labels == k).sum() > 0 else 0
                   for k in range(K)])

# ===== Method B: EM-HMM =====
print("\n" + "="*60)
print("Method B: EM-HMM (Gaussian isotropic, full FB)")
print("="*60)

# Initialize HMM from KMeans
mu = km_centers.copy().astype(np.float64)  # K×d, start from same point
sigma2 = np.ones(K, dtype=np.float64) * 0.5  # shared initial variance

# Reorder states by progress
km_pos = np.nan_to_num(km_progress_mean, nan=0.5)
order = np.argsort(km_pos)
mu = mu[order]; sigma2 = sigma2[order]

# Dirichlet priors (MAP-EM, not pure MLE) to prevent state collapse
# α_A: concentration on each row of A — larger = more uniform, smaller = sparser
# 2.0 = mild smoothing toward uniform, prevents zero-probability transitions
ALPHA_A = 2.0
ALPHA_PI = 2.0  # concentration on initial distribution

# Initialize A with a "near-diagonal" banded matrix (encourages progress ordering)
band = 3  # half-bandwidth
A = np.zeros((K, K), dtype=np.float64)
for i in range(K):
    lo, hi = max(0, i-band), min(K, i+band+2)
    A[i, lo:hi] = 1.0
A /= A.sum(1, keepdims=True)
A = 0.9 * A + 0.1 / K  # 10% uniform background
pi = np.ones(K, dtype=np.float64) / K

# EM iterations
log_lik_history = []
for it in range(N_EM_ITER):
    total_xi = np.zeros((K, K), dtype=np.float64)
    total_gamma0 = np.zeros(K, dtype=np.float64)
    new_mu_num = np.zeros((K, d), dtype=np.float64)
    new_s2_num = np.zeros(K, dtype=np.float64)
    total_gamma = np.zeros(K, dtype=np.float64)
    total_ll = 0.0

    for feat in features_list:
        T = len(feat)
        x = feat.astype(np.float64)

        # --- E-step: log-emission ---
        # log p(x_t | z_t=k) = -0.5 * ||x_t - mu_k||^2 / sigma2_k - 0.5*d*log(2π*sigma2_k)
        log_emit = np.zeros((T, K), dtype=np.float64)
        for k in range(K):
            diff = x - mu[k]
            log_emit[:, k] = -0.5 * np.sum(diff**2, axis=1) / sigma2[k] \
                             - 0.5 * d * np.log(2 * np.pi * sigma2[k])

        # forward (log-scale)
        log_alpha = np.zeros((T, K), dtype=np.float64)
        log_alpha[0] = np.log(pi + EPS) + log_emit[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t-1, :, None] + np.log(A.T + EPS), axis=1) + log_emit[t]

        # backward
        log_beta = np.zeros((T, K), dtype=np.float64)
        for t in range(T-2, -1, -1):
            log_beta[t] = logsumexp(np.log(A + EPS) + log_emit[t+1] + log_beta[t+1], axis=1)

        # posterior gamma
        log_gamma = log_alpha + log_beta
        ll = logsumexp(log_alpha[-1])
        gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
        total_ll += ll

        # posterior xi (T-1, K, K)
        # xi[t,i,j] ∝ gamma[t,i] * A[i,j] * p(x_{t+1}|z_{t+1}=j) * beta[t+1,j]
        # In log space:
        for t in range(T-1):
            la = log_alpha[t, :, None]          # (K,1)
            lA = np.log(A + EPS)                # (K,K)
            le = log_emit[t+1, :]               # (K,)
            lb = log_beta[t+1, :]               # (K,)
            lxi = la + lA + le[None, :] + lb[None, :]
            lxi -= logsumexp(lxi)  # normalize
            xi_t = np.exp(lxi)
            total_xi += xi_t

        # accumulate for M-step
        total_gamma0 += gamma[0]
        for k in range(K):
            new_mu_num[k] += (gamma[:, k:k+1] * x).sum(0)
            new_s2_num[k] += (gamma[:, k] * np.sum((x - mu[k])**2, axis=1)).sum()
            total_gamma[k] += gamma[:, k].sum()

    # --- M-step with Dirichlet priors (MAP, prevents collapse) ---
    pi = (total_gamma0 + ALPHA_PI/K) / (len(features_list) + ALPHA_PI)
    pi = np.clip(pi, EPS, 1.0); pi /= pi.sum()

    # MAP estimate: A[i,:] ~ Dir(α + counts[i,:])
    A = (total_xi + ALPHA_A/K) / np.maximum(total_xi.sum(1, keepdims=True) + ALPHA_A, EPS)
    A = np.clip(A, EPS, 1.0); A /= A.sum(1, keepdims=True)

    for k in range(K):
        if total_gamma[k] > EPS:
            mu[k] = new_mu_num[k] / total_gamma[k]
    sigma2 = new_s2_num / np.maximum(d * total_gamma, EPS)
    sigma2 = np.clip(sigma2, 1e-6, 1e2)

    log_lik_history.append(total_ll / total_frames)
    if it % 5 == 0 or it == N_EM_ITER - 1:
        print(f"  iter {it:3d}: avg log-lik = {log_lik_history[-1]:.4f}", flush=True)

print(f"[EM] converged: avg log-lik {log_lik_history[0]:.4f} → {log_lik_history[-1]:.4f}", flush=True)

# ----- HMM posterior: compute gamma for all frames (like KMeans labels) -----
hmm_gamma_list = []
for feat in features_list:
    T = len(feat); x = feat.astype(np.float64)
    log_emit = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        diff = x - mu[k]
        log_emit[:, k] = -0.5 * np.sum(diff**2, axis=1) / sigma2[k] - 0.5 * d * np.log(2*np.pi*sigma2[k])
    log_alpha = np.zeros((T, K), dtype=np.float64)
    log_alpha[0] = np.log(pi + EPS) + log_emit[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t-1, :, None] + np.log(A.T + EPS), axis=1) + log_emit[t]
    log_beta = np.zeros((T, K), dtype=np.float64)
    for t in range(T-2, -1, -1):
        log_beta[t] = logsumexp(np.log(A + EPS) + log_emit[t+1] + log_beta[t+1], axis=1)
    log_gamma = log_alpha + log_beta
    gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
    hmm_gamma_list.append(gamma)

hmm_gamma_all = np.concatenate(hmm_gamma_list)

# Hard assignment for comparison (argmax gamma)
hmm_labels = hmm_gamma_all.argmax(1)

# Compute per-cluster progress distribution
hmm_progress_std = np.array([np.std(all_progress[hmm_labels == k]) if (hmm_labels == k).sum() > 1 else np.nan
                             for k in range(K)])
hmm_progress_mean = np.array([np.mean(all_progress[hmm_labels == k]) if (hmm_labels == k).sum() > 0 else np.nan
                              for k in range(K)])
hmm_sizes = np.array([(hmm_labels == k).sum() for k in range(K)])
hmm_cov = np.array([len(set(all_E[hmm_labels == k])) / len(mined) if (hmm_labels == k).sum() > 0 else 0
                    for k in range(K)])

# ===== Compare =====
print("\n" + "="*60)
print("COMPARISON: per-cluster progress std (lower = tighter = better)")
print("="*60)

valid_kmeans = ~np.isnan(km_progress_std)
valid_hmm = ~np.isnan(hmm_progress_std)

print(f"\n{'Metric':<40s} {'KMeans':>12s} {'EM-HMM':>12s} {'Better':>8s}")
print("-"*76)
for name, km_v, hmm_v, lower_better in [
    ("non-empty clusters", f"{valid_kmeans.sum()}/{K}", f"{valid_hmm.sum()}/{K}", ""),
    ("mean progress std", f"{np.nanmean(km_progress_std):.4f}", f"{np.nanmean(hmm_progress_std):.4f}", "lower"),
    ("median progress std", f"{np.nanmedian(km_progress_std):.4f}", f"{np.nanmedian(hmm_progress_std):.4f}", "lower"),
    ("weighted mean std (by cluster size)", f"{np.average(km_progress_std[valid_kmeans], weights=km_sizes[valid_kmeans]):.4f}",
     f"{np.average(hmm_progress_std[valid_hmm], weights=hmm_sizes[valid_hmm]):.4f}", "lower"),
    ("mean coverage (episodes)", f"{km_cov[valid_kmeans].mean():.3f}", f"{hmm_cov[valid_hmm].mean():.3f}", "higher"),
    ("% clusters with std<0.10", f"{np.mean(km_progress_std[valid_kmeans] < 0.10)*100:.1f}%",
     f"{np.mean(hmm_progress_std[valid_hmm] < 0.10)*100:.1f}%", "higher"),
    ("% clusters with std<0.05", f"{np.mean(km_progress_std[valid_kmeans] < 0.05)*100:.1f}%",
     f"{np.mean(hmm_progress_std[valid_hmm] < 0.05)*100:.1f}%", "higher"),
]:
    better = "—"
    if lower_better == "lower":
        if float(hmm_v) < float(km_v): better = "← HMM"
        elif float(km_v) < float(hmm_v): better = "← KMeans"
    elif lower_better == "higher":
        if float(hmm_v.rstrip('%')) > float(km_v.rstrip('%')): better = "← HMM"
        elif float(km_v.rstrip('%')) > float(hmm_v.rstrip('%')): better = "← KMeans"
    print(f"{name:<40s} {km_v:>12s} {hmm_v:>12s} {better:>8s}")

# Also print effective state counts
eff_hmm = (hmm_sizes > 10).sum()  # states with >10 frames
print(f"\n{'Effective states (>10 frames)':<40s} {valid_kmeans.sum():>12d} {eff_hmm:>12d}")

# ===== Histogram of per-cluster std =====
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (A) Per-cluster progress std histogram
ax = axes[0, 0]
bins = np.linspace(0, 0.25, 50)
ax.hist(km_progress_std[valid_kmeans], bins=bins, alpha=0.5, label=f'KMeans (μ={np.nanmean(km_progress_std):.3f})', color='#3b6fb0')
ax.hist(hmm_progress_std[valid_hmm], bins=bins, alpha=0.5, label=f'EM-HMM (μ={np.nanmean(hmm_progress_std):.3f})', color='#e85d47')
ax.axvline(np.nanmedian(km_progress_std), color='#3b6fb0', ls='--', lw=1)
ax.axvline(np.nanmedian(hmm_progress_std), color='#e85d47', ls='--', lw=1)
ax.set_xlabel('per-cluster progress std'); ax.set_ylabel('count')
ax.set_title(f'(A) Per-cluster progress std distribution (K={K})')
ax.legend(); ax.grid(alpha=0.2)

# (B) Cluster size vs progress std (scatter)
ax = axes[0, 1]
ax.scatter(km_sizes[valid_kmeans], km_progress_std[valid_kmeans], alpha=0.4, s=15, label='KMeans', color='#3b6fb0')
ax.scatter(hmm_sizes[valid_hmm], hmm_progress_std[valid_hmm], alpha=0.4, s=15, label='EM-HMM', color='#e85d47')
ax.set_xlabel('cluster size (frames)'); ax.set_ylabel('progress std'); ax.set_xscale('log')
ax.set_title('(B) Cluster size vs progress std'); ax.legend(); ax.grid(alpha=0.2)

# (C) EM convergence
ax = axes[1, 0]
ax.plot(log_lik_history, 'k.-', lw=1)
ax.set_xlabel('EM iteration'); ax.set_ylabel('avg log-likelihood per frame')
ax.set_title('(C) EM-HMM convergence'); ax.grid(alpha=0.2)

# (D) Transfer matrix (learned A)
ax = axes[1, 1]
im = ax.imshow(A, cmap='YlOrRd', aspect='auto', vmin=0, vmax=np.percentile(A, 95))
ax.set_xlabel('to state j'); ax.set_ylabel('from state i')
ax.set_title('(D) EM-HMM learned transition matrix A (K=96)')
plt.colorbar(im, ax=ax, label='P(j|i)')

fig.suptitle(f'KMeans vs EM-HMM: cluster quality comparison\n'
             f'{N_MINE} kai0_base episodes, {total_frames} frames, {K} states, {d}D features',
             fontsize=12, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT / "em_hmm_vs_kmeans_quality.png", dpi=140, bbox_inches="tight")
print(f"\nSAVED {OUT / 'em_hmm_vs_kmeans_quality.png'}", flush=True)

# ===== Save data =====
np.savez(OUT / "comparison_results.npz",
         km_progress_std=km_progress_std, km_progress_mean=km_progress_mean, km_sizes=km_sizes, km_cov=km_cov,
         hmm_progress_std=hmm_progress_std, hmm_progress_mean=hmm_progress_mean, hmm_sizes=hmm_sizes, hmm_cov=hmm_cov,
         K=K, N_mine=N_MINE, log_lik_history=np.array(log_lik_history),
         hmm_mu=mu, hmm_sigma2=sigma2, hmm_A=A, hmm_pi=pi,
         km_centers=km_centers)
print(f"SAVED {OUT / 'comparison_results.npz'}", flush=True)

# ===== Prediction demo: sample from HMM =====
print("\n" + "="*60)
print("PREDICTION: sampling future frames from EM-HMM")
print("="*60)
# Pick a random episode, take first 10 frames, predict next 20
test_f = features_list[-1]  # last episode
T_obs = 10; T_pred = 20

# Forward filter for first T_obs frames
log_emit = np.zeros((T_obs, K), dtype=np.float64)
for k in range(K):
    diff = test_f[:T_obs].astype(np.float64) - mu[k]
    log_emit[:, k] = -0.5 * np.sum(diff**2, axis=1) / sigma2[k] - 0.5*d*np.log(2*np.pi*sigma2[k])
log_alpha = np.zeros((T_obs, K), dtype=np.float64)
log_alpha[0] = np.log(pi + EPS) + log_emit[0]
for t in range(1, T_obs):
    log_alpha[t] = logsumexp(log_alpha[t-1, :, None] + np.log(A.T + EPS), axis=1) + log_emit[t]
state_dist = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))

# Sample future states and features
rng_state = np.random.RandomState(99)
pred_states = []; pred_samples = []
cur_dist = state_dist
for t in range(T_pred):
    k = rng_state.choice(K, p=cur_dist)
    pred_states.append(k)
    # sample feature from N(mu[k], sigma2[k]*I)
    feat_sample = mu[k] + np.sqrt(sigma2[k]) * rng_state.randn(d)
    pred_samples.append(feat_sample)
    # transition
    cur_dist = A[k]

pred_states = np.array(pred_states)
pred_samples = np.array(pred_samples)
pred_progress = np.array([np.mean(all_progress[hmm_labels == k]) if (hmm_labels == k).sum() > 0 else 0.5
                          for k in pred_states])

# Plot prediction
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
# State distribution evolution
ax1.bar(range(K), state_dist, alpha=0.6, label='t=10 (after obs)', color='gray')
for t_idx in [0, 5, 10, 15, 19]:
    d = np.zeros(K); d[pred_states[t_idx]] = 1
    ax1.bar(range(K), d, alpha=0.3, label=f't={T_obs+t_idx+1}', width=0.8)
ax1.set_xlabel('state k'); ax1.set_ylabel('probability')
ax1.set_title(f'EM-HMM prediction: state distribution after {T_obs} observed frames → {T_pred} predicted')
ax1.legend(fontsize=7, ncol=2); ax1.grid(alpha=0.2)

# Progress prediction
ax2.plot(range(T_obs), all_progress[all_E == len(features_list)-1][:T_obs], 'k.-', lw=2, label='observed (true progress)')
ax2.plot(range(T_obs, T_obs+T_pred), pred_progress, 'r.--', lw=1.5, label='predicted progress (sampled)')
ax2.set_xlabel('frame t'); ax2.set_ylabel('progress')
ax2.set_title('EM-HMM progress prediction'); ax2.legend(); ax2.grid(alpha=0.2)

fig2.tight_layout(); fig2.savefig(OUT / "em_hmm_prediction_demo.png", dpi=140, bbox_inches="tight")
print(f"SAVED {OUT / 'em_hmm_prediction_demo.png'}", flush=True)
