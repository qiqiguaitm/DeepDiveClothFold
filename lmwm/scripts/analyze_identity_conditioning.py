#!/usr/bin/env python
"""Two cheap identity measurements for V2 (milestone_value) and V3.1 (milestone_viterbi), to settle
whether Stage-1 must be DISTRIBUTIONAL (the two-model cornerstone).

M1 FRAME-CONDITIONED identity entropy:
    index-conditioned eff-branches (known: 2.8 / 4.1) conditions only on the current milestone INDEX.
    Here we condition on the actual current FRAME (pooled DINOv3-H), WITHIN the same current milestone,
    via k-NN. If frame-conditioned eff-branches << index-conditioned -> the observation resolves the
    branch -> a point predictor could suffice. If ~equal -> genuinely multimodal given observation ->
    Stage-1 MUST model a distribution.

M2 COARSE-GRANULARITY branching:
    re-bin the 37 milestones into K coarse progress-ordered stages; recompute eff-branches. If it
    collapses toward 1 as K shrinks -> fine branching was CRAVE over-segmentation. If it stays >2 ->
    real task branching.
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


def eff_from_counter(cnt):
    tot = sum(cnt.values())
    if tot == 0:
        return 0.0, 0.0
    p = np.array(list(cnt.values()), float) / tot
    return float(np.exp(-(p * np.log(p + 1e-12)).sum())), float(p.max())


def collect(E, FR, Fn, proto, protoL, pord):
    """return per-mode list of (cur_gidx, cur_ms, next_ms), FAITHFUL to optimize_subgoal:
       V3.1 milestone_viterbi -> Viterbi-monotone segments; V2 milestone_value -> ARGMAX segments."""
    out = {"milestone_value": [], "milestone_viterbi": []}
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        Fq = Fn[order]
        # V3.1 viterbi segments
        emit = np.linalg.norm(Fq[:, None] - protoL[None], axis=2)
        ms = viterbi_forward(emit, pord, up=3.0, down=25.0, hard_start=True)
        chv = np.where(np.diff(ms) != 0)[0] + 1
        stv = np.concatenate([[0], chv]); env = np.concatenate([chv, [len(ms)]])
        vseg_last, vseg_m = [], []
        for s, e in zip(stv, env):
            vseg_last.append(int(order[e - 1])); vseg_m.append(int(ms[s]))
        for i in range(len(vseg_m) - 1):
            out["milestone_viterbi"].append((vseg_last[i], vseg_m[i], vseg_m[i + 1]))
        # V2 milestone_value on ARGMAX segments (== optimize_subgoal), value-next milestone in library
        seq = (Fq @ proto.T).argmax(1)
        cha = np.where(np.diff(seq) != 0)[0] + 1
        sta = np.concatenate([[0], cha]); ena = np.concatenate([cha, [len(seq)]])
        aseg_last, aseg_m = [], []
        for s, e in zip(sta, ena):
            aseg_last.append(int(order[e - 1])); aseg_m.append(int(seq[s]))
        present = sorted({m: float(pord[m]) for m in aseg_m}.items(), key=lambda kv: kv[1])
        vals = [m for m, _ in present]
        for i in range(len(aseg_m)):
            cv = float(pord[aseg_m[i]])
            nxt = [m for m in vals if float(pord[m]) > cv + 1e-6]
            if nxt:
                out["milestone_value"].append((aseg_last[i], aseg_m[i], int(nxt[0])))
    return out


def m1_frame_conditioned(pairs, Fn, knn=30, n_query=2500, seed=0):
    cur_g = np.array([p[0] for p in pairs]); cur_ms = np.array([p[1] for p in pairs])
    nxt = np.array([p[2] for p in pairs])
    Fc = Fn[cur_g]                                            # L2-normed pooled current-frame feats
    rng = np.random.default_rng(seed)
    # index-conditioned (visit-weighted) over THIS pair set
    idx_eff = []
    for k in np.unique(cur_ms):
        c = Counter(nxt[cur_ms == k].tolist()); e, _ = eff_from_counter(c)
        idx_eff.append((e, int((cur_ms == k).sum())))
    tot = sum(n for _, n in idx_eff)
    index_cond = sum(e * n for e, n in idx_eff) / tot
    # frame-conditioned: within same cur_ms, k-NN by current-frame cos
    by_ms = {k: np.where(cur_ms == k)[0] for k in np.unique(cur_ms)}
    q = rng.permutation(len(pairs))[:min(n_query, len(pairs))]
    loc_eff = []
    for i in q:
        cand = by_ms[cur_ms[i]]
        if len(cand) < knn + 1:
            continue
        sims = Fc[cand] @ Fc[i]
        top = cand[np.argsort(-sims)[:knn]]
        e, _ = eff_from_counter(Counter(nxt[top].tolist()))
        loc_eff.append(e)
    frame_cond = float(np.mean(loc_eff)) if loc_eff else float("nan")
    return {
        "index_conditioned_eff_branches": round(float(index_cond), 3),
        "frame_conditioned_eff_branches": round(frame_cond, 3),
        "resolved_fraction": round(1 - frame_cond / (index_cond + 1e-9), 3),
        "n_query": len(loc_eff), "knn": knn,
    }


def m2_coarse(pairs, pord, Ks=(37, 18, 12, 8, 5)):
    cur_ms = np.array([p[1] for p in pairs]); nxt = np.array([p[2] for p in pairs])
    rank = np.argsort(np.argsort(pord))                      # progress rank of each milestone
    res = {}
    for K in Ks:
        binid = (rank.astype(float) / 37 * K).astype(int).clip(0, K - 1)
        cb, nb = binid[cur_ms], binid[nxt]
        nd = defaultdict(Counter)
        for a, b in zip(cb, nb):
            if a != b:
                nd[a][b] += 1
        rows = [(sum(c.values()),) + eff_from_counter(c) for c in nd.values()]
        T = sum(r[0] for r in rows) or 1
        res[f"K={K}"] = {"eff_branches": round(sum(r[0] * r[1] for r in rows) / T, 3),
                         "frac_branchy(eff>=2)": round(sum(r[0] for r in rows if r[1] >= 2) / T, 3)}
    return res


def main():
    E, FR, Fn = load_index(REPO / "temp/crave_full_dinov3h")
    rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = rg["prototype_table"].astype(np.float32); pord = rg["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    pm = collect(E, FR, Fn, proto, protoL, pord)
    out = {}
    for mode in ["milestone_value", "milestone_viterbi"]:
        out[mode] = {"n_pairs": len(pm[mode]),
                     "M1_frame_conditioned": m1_frame_conditioned(pm[mode], Fn),
                     "M2_coarse_granularity": m2_coarse(pm[mode], pord)}
        print(mode, json.dumps(out[mode], indent=2), flush=True)
    (REPO / "lmwm/outputs/identity_conditioning.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
