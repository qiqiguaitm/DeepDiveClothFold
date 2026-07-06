#!/usr/bin/env python
"""Confirm: is the IDENTITY multimodality (which milestone comes next) strong under EACH milestone+1
target construction we used, or only under milestone_viterbi? Computes visit-weighted effective
branches of the (current_milestone -> next_milestone) distribution for every mode in build_pairs.

Modes:
  milestone (V1)         : argmax-assign segments, next = next temporal segment's milestone
  milestone_value (V2)   : next = smallest-CRAVE-value milestone > current present in the episode
  progress_delta (V3)    : target frame at progress+delta; identity = its argmax milestone (next!=cur)
  milestone_viterbi(V3.1): Viterbi-monotone segments, next = next segment's milestone
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
from crave.utils.signal import smooth_monotone  # noqa: E402
from crave.utils.array import med  # noqa: E402


def eff_branches(trans):
    """visit-weighted effective branches + top-prob over (src->dst) transitions (dst!=src)."""
    nd = defaultdict(Counter)
    for a, b in trans:
        if a != b:
            nd[a][b] += 1
    rows = []
    for a, cnt in nd.items():
        tot = sum(cnt.values()); p = np.array(list(cnt.values()), float) / tot
        H = -(p * np.log(p + 1e-12)).sum()
        rows.append((tot, float(np.exp(H)), float(p.max())))
    T = sum(r[0] for r in rows) or 1
    return {
        "n_transitions": int(sum(r[0] for r in rows)),
        "eff_branches": round(sum(r[0] * r[1] for r in rows) / T, 3),
        "top_prob": round(sum(r[0] * r[2] for r in rows) / T, 3),
        "frac_deterministic(top>0.9)": round(sum(r[0] for r in rows if r[2] > 0.9) / T, 3),
        "frac_branchy(eff>=2)": round(sum(r[0] for r in rows if r[1] >= 2.0) / T, 3),
    }


def main():
    E, FR, Fn = load_index(REPO / "temp/crave_full_dinov3h")
    rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = rg["prototype_table"].astype(np.float32); pord = rg["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    T = {"milestone": [], "milestone_value": [], "progress_delta": [], "milestone_viterbi": []}
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        Fq = Fn[order]
        assign = (Fq @ proto.T).argmax(1)                       # per-frame argmax milestone

        # V1 milestone: argmax segments
        ch = np.where(np.diff(assign) != 0)[0] + 1
        segm = assign[np.concatenate([[0], ch])]
        for i in range(len(segm) - 1):
            T["milestone"].append((int(segm[i]), int(segm[i + 1])))

        # V2 milestone_value: next = smallest value milestone > current, present in this episode
        present = sorted({int(m): float(pord[m]) for m in segm}.items(), key=lambda kv: kv[1])
        vals = [m for m, _ in present]
        for i in range(len(segm)):
            cv = float(pord[segm[i]])
            nxt = [m for m in vals if float(pord[m]) > cv + 1e-6]
            if nxt:
                T["milestone_value"].append((int(segm[i]), int(nxt[0])))

        # V3 progress_delta: target frame at progress+0.15; identity = its argmax milestone
        emit = np.linalg.norm(Fq[:, None] - protoL[None], axis=2)
        ms = viterbi_forward(emit, pord, up=3.0, down=25.0, hard_start=True)
        value = smooth_monotone(med(pord[ms], 5), fps=3.0)
        for i in range(len(order)):
            jj = int(np.searchsorted(value, value[i] + 0.15))
            if i < jj < len(order):
                T["progress_delta"].append((int(assign[i]), int(assign[jj])))

        # V3.1 milestone_viterbi: viterbi segments
        chv = np.where(np.diff(ms) != 0)[0] + 1
        segv = ms[np.concatenate([[0], chv])]
        for i in range(len(segv) - 1):
            T["milestone_viterbi"].append((int(segv[i]), int(segv[i + 1])))

    out = {mode: eff_branches(tr) for mode, tr in T.items()}
    (REPO / "lmwm/outputs/identity_branching_bymode.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
