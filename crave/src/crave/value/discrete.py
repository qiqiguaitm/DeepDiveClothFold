"""DiscreteValue — V2.4 discrete-milestone progress ladder.

Ported verbatim from crave_value.DiscreteValue (KMeans + coverage correction + progress
binning + precedence/isotonic ordering + start/end anchors + Viterbi-DP), with the only
change being that `med`/`viterbi`/`mkp`/`loadep` now come from the crave package. Behavior
is byte-for-byte identical to the legacy class.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from crave.data.cache import loadep
from crave.utils.array import med
from crave.utils.dp import viterbi
from crave.value.features import FeatureSpace


class DiscreteValue:
    def __init__(self, fs: FeatureSpace, eps, k=96, log=print,
                 select="fixed", nbins=10, topN=2, cap_pb=3, tau_q=0.5,
                 order="time", prec_min_co=5, cond_end=False):
        self.fs = fs; self.select = select; self.order_mode = order; self.prec_min_co = prec_min_co
        A, R, S, T, E, SP, EP = [], [], [], [], [], [], []
        for e in eps:
            a, r, s, n = loadep(fs.fc, e); g = fs.emb(a, r, s)
            A.append(a); R.append(r); S.append(s); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
            SP.append(g[:2]); EP.append(g[-2:])
        A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
        G = fs.emb(A, R, S)
        km = KMeans(k, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
        N = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(k)])
        Pstart = {}
        for e in sorted(set(E.tolist())):
            m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1)
            Pstart[e] = float(np.median(tpos[nn]))
        cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / N) for c in range(k)])
        bk = np.linspace(0, 1, nbins + 1); sel = []; self.tau = None
        if select == "adaptive":
            self.tau = float(np.quantile(cov_n, tau_q))
            for b in range(nbins):
                inb = sorted([c for c in range(k) if bk[b] <= tpos[c] < bk[b + 1]], key=lambda c: -cov_n[c])
                if not inb: continue
                above = [c for c in inb if cov_n[c] >= self.tau][:cap_pb]
                sel += above if above else inb[:1]
        else:
            for b in range(nbins):
                inb = [c for c in range(k) if bk[b] <= tpos[c] < bk[b + 1]]
                if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:topN]
        sel = sorted(set(sel), key=lambda c: tpos[c])

        def gr(idx):
            o = []; s0 = None; pv = None
            for i in idx:
                if pv is None or i != pv + 1:
                    if s0 is not None: o.append((s0, pv))
                    s0 = i
                pv = i
            if s0 is not None: o.append((s0, pv))
            return [x for x in o if x[1] - x[0] >= 1]

        eps_sorted = sorted(set(E.tolist())); ns = len(sel)
        fe_mat = np.full((len(eps_sorted), ns), np.nan)
        for ei, e in enumerate(eps_sorted):
            m = np.where(E == e)[0]
            for si, c in enumerate(sel):
                rs = gr(m[lab[m] == c].tolist())
                if rs: fe_mat[ei, si] = T[rs[0][0]]
        Pk = {c: (float(np.nanmedian(fe_mat[:, si])) if np.isfinite(fe_mat[:, si]).any() else float(tpos[c])) for si, c in enumerate(sel)}
        if self.order_mode == "precedence":
            from sklearn.isotonic import IsotonicRegression
            Pbef = np.full((ns, ns), np.nan)
            for i in range(ns):
                for j in range(ns):
                    if i == j: continue
                    both = np.isfinite(fe_mat[:, i]) & np.isfinite(fe_mat[:, j])
                    if both.sum() >= self.prec_min_co: Pbef[i, j] = float(np.mean(fe_mat[both, i] < fe_mat[both, j]))
            soft = np.nansum(np.where(np.isnan(Pbef), 0.0, Pbef), axis=1)
            prec = list(np.argsort(-soft))
            iso = IsotonicRegression(increasing=True).fit_transform(np.arange(ns), np.array([Pk[sel[si]] for si in prec]))
            self.order = [sel[si] for si in prec]
            self.Pord = np.asarray(iso, float)
            self.Pk = {self.order[kk]: float(iso[kk]) for kk in range(ns)}
        else:
            self.order = sorted(sel, key=lambda c: Pk[c])
            self.Pord = np.array([Pk[c] for c in self.order]); self.Pk = Pk
        self.C = allC[self.order]
        log(f"[DiscreteValue] select={select} milestones: {len(self.order)}"
            + (f" (tau={self.tau:.2f})" if self.tau is not None else "")
            + f"  前段(P<0.5): {sum(1 for c in self.order if Pk[c] < 0.5)}")
        self.startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
        self.endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_
        self.cond_end = cond_end
        de_tr = np.array([float(np.linalg.norm(ep[:, None] - self.endK[None], axis=2).min()) for ep in EP])
        self.de_end_thr = float(np.quantile(de_tr, 0.90)) * 1.3
        self.NB = 21; self.bins = np.linspace(0, 1, self.NB)
        self.cb = [[int(np.argmin(abs(self.bins - Pk[c])))] for c in self.order]

    def status(self, a, r, s):
        Fq = self.fs.emb(a, r, s)
        return self._status(Fq, np.linalg.norm(Fq[:, None] - self.C[None], axis=2))

    def _status(self, Fq, d):
        nq = len(Fq); thr = float(self.de_end_thr)
        de = np.linalg.norm(Fq[:, None] - self.endK[None], axis=2).min(1)
        de_end = float(np.min(de[-3:])) if nq >= 3 else float(de[-1])
        ood = d.min(1)
        return {"is_complete": bool(de_end <= thr), "complete_conf": float(np.clip((1.2 * thr - de_end) / (0.4 * thr + 1e-9), 0.0, 1.0)),
                "de_end": de_end, "de_end_thr": thr, "ood": ood, "ood_frac": float(np.mean(ood > thr))}

    def value(self, a, r, s, ret_lab=False, ret_status=False):
        Fq = self.fs.emb(a, r, s); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - self.C[None], axis=2)
        em = np.full((nq, self.NB), 1e3)
        for ci in range(len(self.order)):
            for b in self.cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
        ds = np.linalg.norm(Fq[:, None] - self.startK[None], axis=2).min(1)
        de = np.linalg.norm(Fq[:, None] - self.endK[None], axis=2).min(1)
        tn = np.arange(nq) / nq
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, self.NB - 1] = np.minimum(em[:, self.NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        de_end = float(np.min(de[-3:])) if nq >= 3 else float(de[-1])
        eb = 2.0 * float(np.clip((self.de_end_thr - de_end) / (0.3 * self.de_end_thr + 1e-9), 0.0, 1.0)) if self.cond_end else 2.0
        v = med(viterbi(em, self.bins, lam=8.0, end_bonus=eb)[0], 9)
        if ret_status:
            return v, self._status(Fq, d)
        if ret_lab:
            dsrt = np.sort(d, axis=1); marg = dsrt[:, 0] / np.clip(dsrt[:, 1], 1e-9, None)
            return v, d.argmin(1), marg
        return v
