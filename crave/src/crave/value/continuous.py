"""ContinuousValue — TCC frozen-feature head + similarity-field DP + soft sub-bin readout.

Ported verbatim from crave_value.ContinuousValue. Requires torch + the google-research
xirl TCC loss (external path). Produces a per-frame continuous value in [0,1].
"""
from __future__ import annotations

import numpy as np

from crave.utils.array import med
from crave.utils.dp import viterbi
from crave.value.features import FeatureSpace

XIRL_PATH = "/vePFS/tim/workspace/recurrence_research/google-research/xirl"


class ContinuousValue:
    def __init__(self, fs: FeatureSpace, eps, n_refs=30, seed=0):
        import torch
        import torch.nn as nn
        self.torch, self.nn = torch, nn
        self.fs = fs; self.eps = list(eps); self.n_refs = n_refs; self.seed = seed
        self.Gd = {e: fs.emb_ep(e).astype(np.float32) for e in self.eps}
        self.din = self.Gd[self.eps[0]].shape[1]; self.head = None; self.bank = None; self.bankt = None

    def _make_head(self):
        nn = self.nn
        return nn.Sequential(nn.Linear(self.din, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128))

    def train_head(self, steps=1200, batch_eps=8, T=32, lr=1e-3, log=print):
        import sys
        sys.path.insert(0, XIRL_PATH)
        from xirl.losses import compute_tcc_loss
        torch = self.torch; np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.head = self._make_head(); opt = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-5)
        for step in range(steps):
            bes = list(np.random.choice(self.eps, min(batch_eps, len(self.eps)), replace=False))
            E_, I_, L_ = [], [], []
            for e in bes:
                f = self.Gd[e]; m = len(f); ix = np.sort(np.random.choice(m, size=T, replace=m < T))
                E_.append(self.head(torch.from_numpy(f[ix]))); I_.append(torch.from_numpy(ix).long()); L_.append(m)
            loss = compute_tcc_loss(embs=torch.stack(E_), idxs=torch.stack(I_), seq_lens=torch.tensor(L_),
                stochastic_matching=False, normalize_embeddings=True, loss_type="regression_mse", similarity_type="l2",
                num_cycles=20, cycle_length=2, temperature=0.1, label_smoothing=0.1, variance_lambda=0.001,
                huber_delta=0.1, normalize_indices=True)
            opt.zero_grad(); loss.backward(); opt.step()
            if log and (step + 1) % 400 == 0: log(f"[ContinuousValue] step {step+1} loss {float(loss):.4f}")
        self.head.eval()
        REFS = list(self.eps)[:self.n_refs]
        self.bank = np.concatenate([self._embed(self.Gd[e]) for e in REFS])
        self.bankt = np.concatenate([np.arange(len(self.Gd[e])) / max(1, len(self.Gd[e]) - 1) for e in REFS])
        return self

    def _embed(self, g796):
        torch = self.torch
        with torch.no_grad():
            z = self.head(torch.from_numpy(np.ascontiguousarray(g796, dtype=np.float32))).numpy()
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    def value(self, a, r, s, NB=201, lam=0.2, soft_w=8, soft_temp=0.03, smooth=9, exclude_ep=None):
        assert self.head is not None, "先 train_head()"
        bank, bankt = self.bank, self.bankt
        zq = self._embed(self.fs.emb(a, r, s)); n = len(zq); sim = zq @ bank.T
        bins = np.linspace(0, 1, NB); binid = np.clip((bankt * (NB - 1)).round().astype(int), 0, NB - 1)
        simb = np.full((n, NB), -9.0)
        for b in range(NB):
            c = np.where(binid == b)[0]
            if len(c): simb[:, b] = sim[:, c].max(1)
        rmax = simb.max(1, keepdims=True); rmin = np.where(simb > -8, simb, np.inf).min(1, keepdims=True)
        emit = np.where(simb > -8, (rmax - simb) / (rmax - rmin + 1e-6), 1.0)
        _, path = viterbi(emit, bins, lam=lam, end_bonus=2.0)
        cont = np.zeros(n)
        for i in range(n):
            lo = max(0, path[i] - soft_w); hi = min(NB, path[i] + soft_w + 1); ss = simb[i, lo:hi].copy()
            if (ss <= -8).all(): cont[i] = bins[path[i]]; continue
            ss[ss <= -8] = ss[ss > -8].min(); w = np.exp((ss - ss.max()) / soft_temp); w /= w.sum()
            cont[i] = (w * bins[lo:hi]).sum()
        return med(cont, smooth)
