#!/usr/bin/env python
"""Analysis C: is the milestone+1 target uni- or multi-modal?

Decides whether best-of-N / a generative (flow) Stage-2 head can help at all, or whether the
oracle->deploy gap is irreducible info loss. Two DISTINCT multimodalities, both in the SAME
pooled DINOv3-H space the milestone_viterbi target is built in (load_index -> L2-normed pooled):

  (1) IDENTITY  : which milestone comes next. Entropy/eff-branches of the segment->segment
                  transition, visit-weighted. High -> task branches -> Stage-1 must be multimodal.
  (2) APPEARANCE: given you are at the end of milestone k (current frame g_t), how spread is the
                  next-milestone per-episode target g_f? Measured 3 ways:
                    a. per-episode target vs GLOBAL prototype cos (are episode targets ~ the centroid?)
                    b. within-transition (same k->k') spread across episodes (pure appearance modes)
                    c. CONDITIONAL spread: for each pair, spread of g_f among pairs whose g_t is a
                       near-neighbor -> Var[g_f | g_t]. This is exactly what best-of-N must exploit.

Verdict: if identity eff-branches ~1 AND conditional spread << marginal AND target~centroid, the
target is near-unimodal -> best-of-N is a red herring, +0.02 is honest. Else -> flow Stage-2 worth it.

NOTE: pooled is a LOWER bound on grid multimodality (grid has more spatial DOF). If pooled says
multimodal, grid is at least as multimodal; if pooled says unimodal, grid may still add spatial modes.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import load_index  # noqa: E402
from crave.utils.dp import viterbi_forward  # noqa: E402


def cosrow(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def main():
    feat_dir = REPO / "temp/crave_full_dinov3h"
    rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = rg["prototype_table"].astype(np.float32)
    pord = rg["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    E, FR, Fn = load_index(feat_dir)  # Fn: (N,1280) L2-normalized pooled
    print(f"loaded {len(E)} frames, {len(np.unique(E))} episodes, dim {Fn.shape[1]}", flush=True)

    # ---- build the EXACT milestone_viterbi segments/pairs (pooled) ----
    segs = []          # (ep, cur_mid, next_mid, cur_gidx, tgt_gidx)
    ident = []         # (cur_mid, next_mid) between consecutive segments
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        Fq = Fn[order]
        emit = np.linalg.norm(Fq[:, None] - protoL[None], axis=2)
        ms = viterbi_forward(emit, pord, up=3.0, down=25.0, hard_start=True)
        ch = np.where(np.diff(ms) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(ms)]])
        seg_m, seg_med, seg_last = [], [], []
        for s, e in zip(st, en):
            m = int(ms[s]); sub = order[s:e]
            seg_med.append(int(sub[(Fq[s:e] @ protoL[m]).argmax()]))
            seg_last.append(int(order[e - 1])); seg_m.append(m)
        for i in range(len(seg_m) - 1):
            ident.append((seg_m[i], seg_m[i + 1]))
            segs.append((ep, seg_m[i], seg_m[i + 1], seg_last[i], seg_med[i + 1]))
    print(f"{len(segs)} milestone->milestone pairs, {len(np.unique([s[0] for s in segs]))} episodes", flush=True)

    # =========================================================================
    # (1) IDENTITY multimodality: which milestone next
    # =========================================================================
    nextdist = defaultdict(Counter)
    for a, b in ident:
        nextdist[a][b] += 1
    rows = []
    for a, cnt in nextdist.items():
        tot = sum(cnt.values()); p = np.array(list(cnt.values()), float) / tot
        H = float(-(p * np.log(p + 1e-12)).sum())
        rows.append({"src": a, "visits": tot, "n_next": len(cnt),
                     "entropy": H, "eff_branches": float(np.exp(H)),
                     "top_prob": float(p.max())})
    tot_all = sum(r["visits"] for r in rows)
    w = lambda key: sum(r["visits"] * r[key] for r in rows) / tot_all
    identity = {
        "n_source_milestones": len(rows),
        "visit_weighted_eff_branches": round(w("eff_branches"), 3),
        "visit_weighted_top_prob": round(w("top_prob"), 3),
        "frac_transitions_deterministic(top>0.9)": round(
            sum(r["visits"] for r in rows if r["top_prob"] > 0.9) / tot_all, 3),
        "frac_transitions_branchy(eff>=2)": round(
            sum(r["visits"] for r in rows if r["eff_branches"] >= 2.0) / tot_all, 3),
    }

    # =========================================================================
    # (2) APPEARANCE multimodality
    # =========================================================================
    gt = Fn[np.array([s[3] for s in segs])]           # current (end of seg i)
    gf = Fn[np.array([s[4] for s in segs])]           # per-episode next-milestone target
    cur_m = np.array([s[1] for s in segs]); nxt_m = np.array([s[2] for s in segs])

    persist = float(cosrow(gt, gf).mean())            # headroom: how far target is from current
    gf_vs_centroid = float(cosrow(gf, protoL[nxt_m]).mean())  # (a) episode target vs global prototype

    # (b) within-transition spread across episodes (same k->k')
    groups = defaultdict(list)
    for i, s in enumerate(segs):
        groups[(s[1], s[2])].append(i)
    rng0 = np.random.default_rng(1)

    def top_pc(Gc, iters=15):                          # cheap top principal component (power iteration)
        v = rng0.standard_normal(Gc.shape[1]); v /= np.linalg.norm(v) + 1e-8
        for _ in range(iters):
            v = Gc.T @ (Gc @ v); v /= np.linalg.norm(v) + 1e-8
        return v

    tights, mm_gain = [], []
    for key, idxs in groups.items():
        if len(idxs) < 8:
            continue
        if len(idxs) > 300:                            # cap huge groups (SVD/PC cost); random subsample
            idxs = list(rng0.permutation(idxs)[:300])
        G = gf[idxs]                                   # already L2-normed
        mu = G.mean(0); mu = mu / (np.linalg.norm(mu) + 1e-8)
        within = float((G @ mu).mean())                # ~1 = tight/unimodal
        tights.append((within, len(idxs)))
        # crude 2-mode test: split by top principal component, compare within-cos
        Gc = G - G.mean(0)
        pc = top_pc(Gc)
        proj = Gc @ pc
        A, B = G[proj < 0], G[proj >= 0]
        if len(A) >= 2 and len(B) >= 2:
            def tc(X):
                m = X.mean(0); m /= np.linalg.norm(m) + 1e-8
                return (X @ m).mean()
            two = (len(A) * tc(A) + len(B) * tc(B)) / len(G)
            mm_gain.append((two - within, len(idxs)))   # >0.03 => splitting helps => bimodal-ish
    tw = np.array([t for t, _ in tights]); twn = np.array([n for _, n in tights])
    within_tight = float((tw * twn).sum() / twn.sum()) if len(tw) else float("nan")
    mg = np.array([g for g, _ in mm_gain]); mgn = np.array([n for _, n in mm_gain])
    twomode_gain = float((mg * mgn).sum() / mgn.sum()) if len(mg) else float("nan")
    frac_bimodal = float((mgn[mg > 0.03].sum() / mgn.sum())) if len(mg) else float("nan")

    # (c) CONDITIONAL spread: Var[g_f | g_t] via g_t near-neighbors (subsample for O(P^2))
    rng = np.random.default_rng(0)
    P = len(segs); sub = rng.permutation(P)[:min(2500, P)]
    Gt, Gf = gt[sub], gf[sub]
    S = Gt @ Gt.T                                       # cos sim of current frames (L2-normed)
    marg_mu = Gf.mean(0); marg_mu /= np.linalg.norm(marg_mu) + 1e-8
    marginal_spread = float(1 - (Gf @ marg_mu).mean())  # 1-cos to global mean of targets
    cond_spreads = []
    for i in range(len(sub)):
        nn = np.argsort(-S[i])[1:21]                    # top-20 nearest current frames
        H = Gf[nn]
        m = H.mean(0); m /= np.linalg.norm(m) + 1e-8
        cond_spreads.append(1 - float((H @ m).mean()))
    conditional_spread = float(np.mean(cond_spreads))
    # ratio ~1 => g_t doesn't pin g_f (uncertainty/MM possible); ~0 => g_t determines g_f (unimodal)
    cond_ratio = round(conditional_spread / (marginal_spread + 1e-8), 3)

    appearance = {
        "headroom_persistence_cos(gt,gf)": round(persist, 4),
        "episode_target_vs_global_centroid_cos": round(gf_vs_centroid, 4),
        "within_transition_tightness_cos": round(within_tight, 4),
        "two_mode_split_gain": round(twomode_gain, 4),
        "frac_transitions_bimodal(gain>0.03)": round(frac_bimodal, 3),
        "marginal_target_spread(1-cos)": round(marginal_spread, 4),
        "conditional_target_spread|gt(1-cos)": round(conditional_spread, 4),
        "conditional/marginal_ratio": cond_ratio,
    }

    # ---- verdict ----
    id_uni = identity["visit_weighted_eff_branches"] < 1.3
    app_uni = (gf_vs_centroid > 0.85) and (cond_ratio < 0.4) and (frac_bimodal < 0.15)
    verdict = ("NEAR-UNIMODAL: best-of-N is a red herring; oracle->deploy gap is info loss."
               if (id_uni and app_uni) else
               "MULTIMODAL signal present: Stage-2 flow / best-of-N can plausibly help "
               f"(identity_branchy={not id_uni}, appearance_multimodal={not app_uni}).")

    out = {"identity": identity, "appearance": appearance, "verdict": verdict,
           "n_pairs": len(segs), "n_transition_groups_used": int(twn.size)}
    (REPO / "lmwm/outputs").mkdir(parents=True, exist_ok=True)
    (REPO / "lmwm/outputs/milestone_multimodality.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
