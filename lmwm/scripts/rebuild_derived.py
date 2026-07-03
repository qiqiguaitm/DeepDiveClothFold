#!/usr/bin/env python
"""Rebuild derived data in the NEW unified encode_pooled space (after reencode_pooled_unified).

Keeps milestone SEMANTICS (per-frame assignment from the old space, since the graph/
transitions depend only on ids), recomputes prototypes / medoids / pair latents in the
new space. Outputs pairs_next_unique_augin_v2.npz + recurrence_graph_v2.npz.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov3h_decoder import load_features, l2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_bank", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--new_bank", default="temp/crave_full_dinov3h_v2", type=Path)
    ap.add_argument("--old_pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--old_graph", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out_pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin_v2.npz")
    ap.add_argument("--out_graph", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph_v2.npz")
    args = ap.parse_args()

    # ---- new features by gidx ----
    idx = np.load(args.old_bank / "index.npz"); E, FR = idx["E"].astype(np.int64), idx["FR"].astype(np.int64); n = int(idx["n"])
    new_feat = np.zeros((n, 1280), np.float32)
    seen = np.zeros(n, bool)
    for f in sorted(glob.glob(str(args.new_bank / "feat_*.npz"))):
        z = np.load(f); g = z["gidx"].astype(np.int64); new_feat[g] = z["feat"].astype(np.float32); seen[g] = True
    assert seen.all(), f"missing {int((~seen).sum())} frames in new bank"
    new_n = l2(new_feat)

    # ---- old per-frame milestone assignment (semantic; from old feat + old centers C) ----
    Eo, FRo, Fo = load_features(args.old_bank)                     # valid old frames
    C = np.load(args.old_bank / "milestones_uniform_dinov3h.npz")["C"].astype(np.float32)
    assign = (l2(Fo.astype(np.float32)) @ l2(C).T).argmax(1)       # milestone per (valid) frame
    # map (E,FR)->assignment; valid frames only
    key = Eo.astype(np.int64) * 100000 + FRo.astype(np.int64)
    a_by_key = dict(zip(key.tolist(), assign.tolist()))
    all_key = E * 100000 + FR
    full_assign = np.array([a_by_key.get(int(k), -1) for k in all_key])
    num_m = len(C)

    # ---- new prototypes = mean of new features per milestone ----
    C_new = np.zeros((num_m, 1280), np.float32)
    for m in range(num_m):
        sel = new_feat[full_assign == m]
        C_new[m] = sel.mean(0) if len(sel) else C[m]
    C_new_n = l2(C_new)

    # ---- gidx by (E,FR) for pair lookup ----
    g_by_key = {int(k): g for g, k in enumerate(all_key)}

    # ---- per-episode stages + new medoids ----
    ep_stage_med = {}                                             # ep -> list of (start_fr, milestone, medoid_new_latent)
    for e in np.unique(E):
        loc = np.where(E == e)[0]; order = loc[np.argsort(FR[loc])]
        seq = full_assign[order]
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        stages = []
        for s, en_ in zip(st, en):
            gl = order[s:en_]; m = int(seq[s])
            medg = gl[(new_n[gl] @ C_new_n[m]).argmax()]
            stages.append((int(FR[order[s]]), m, new_n[medg]))
        ep_stage_med[int(e)] = stages

    # ---- rebuild pairs ----
    z = dict(np.load(args.old_pairs))
    eps = z["episode_id"].astype(np.int64); ts = z["t"].astype(np.int64); m = len(eps)
    cur_old = z["current"].astype(np.float32)                     # [m, 1332] = pooled|prev_oh(38)|state(14)
    new_pooled = np.zeros((m, 1280), np.float32); new_med = np.zeros((m, 1280), np.float32)
    for i in range(m):
        g = g_by_key.get(int(eps[i] * 100000 + ts[i]))
        new_pooled[i] = new_n[g] if g is not None else l2(cur_old[i:i + 1, :1280])[0]
        stages = ep_stage_med[int(eps[i])]
        # find current stage index by start_fr <= t, then next stage's medoid
        si = max(j for j in range(len(stages)) if stages[j][0] <= ts[i])
        new_med[i] = stages[min(si + 1, len(stages) - 1)][2]
    z["current"] = np.concatenate([new_pooled, cur_old[:, 1280:]], 1).astype(np.float32)
    z["next_medoid"] = new_med.astype(np.float32)
    Path(args.out_pairs).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_pairs, **z)

    # ---- new graph: prototype_table = C_new, transition_probs unchanged ----
    g = dict(np.load(args.old_graph))
    g["prototype_table"] = C_new.astype(np.float32)
    Path(args.out_graph).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_graph, **g)

    # sanity: assignment vs stored current_milestone
    match = float((full_assign[[g_by_key[int(eps[i] * 100000 + ts[i])] for i in range(min(2000, m)) if int(eps[i] * 100000 + ts[i]) in g_by_key]]
                   == z["current_milestone"][:2000][[i for i in range(min(2000, m)) if int(eps[i] * 100000 + ts[i]) in g_by_key]]).mean())
    print(f"assignment vs stored current_milestone match={match:.3f}")
    print(f"saved {args.out_pairs} ({m} pairs) + {args.out_graph} (proto_table {C_new.shape})")


if __name__ == "__main__":
    main()
