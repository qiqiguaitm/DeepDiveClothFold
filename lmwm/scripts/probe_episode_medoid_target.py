#!/usr/bin/env python
"""Probe the user's proposal: represent each stage by its EPISODE-LOCAL medoid
latent (real frame closest to the cluster centroid), not the global centroid.

Measures, on the DINOv3-H cache:
  M1: cos(episode_medoid, global_centroid) -- how far medoids deviate from centroid.
  M2: cross-episode consistency -- mean pairwise cos among a milestone's episode
      medoids. Low => medoids are episode-specific (proposal changes the target a
      lot); high => medoids ~= centroid (proposal ~= current).
  M3: target-shift for a next-stage predictor. The current proto head predicts
      ~the global centroid. Against the medoid target its score is
      cos(centroid_next, true_next_episode_medoid). We also check whether the
      *current frame* carries episode-specific info about the next medoid beyond
      the centroid (regression-to-mean test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_full(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    feat = np.zeros((n, 1280), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        gi = z["gidx"].astype(np.int64)
        feat[gi] = z["feat"]; valid[gi] = z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32)
    fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], fv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", default="lmwm/outputs/ceiling_diag/medoid_probe.json", type=Path)
    args = ap.parse_args()

    g = np.load(args.graph_npz)
    proto = g["prototype_table"].astype(np.float32)  # l2 centroids
    num_m = len(proto)
    E, FR, F = load_full(args.feature_dir)
    assign = np.empty(len(F), dtype=np.int64)
    for i in range(0, len(F), 32768):
        assign[i:i + 32768] = (F[i:i + 32768] @ proto.T).argmax(1)

    # Per episode: compress into stages; per stage store (milestone, medoid latent).
    # transitions: (cur_milestone, next_milestone, next_medoid) per stage boundary.
    medoids_by_m: dict[int, list[np.ndarray]] = {m: [] for m in range(num_m)}
    cos_med_cent = []
    trans = []  # (cur_m, next_m, cur_medoid, next_medoid)
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]
        order = loc[np.argsort(FR[loc])]
        seq = assign[order]
        # run boundaries
        change = np.where(np.diff(seq) != 0)[0] + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(seq)]])
        stage_m, stage_med = [], []
        for s, e in zip(starts, ends):
            m = int(seq[s])
            sub = F[order[s:e]]
            med = sub[(sub @ proto[m]).argmax()]  # frame in run closest to centroid
            stage_m.append(m); stage_med.append(med)
            medoids_by_m[m].append(med)
            cos_med_cent.append(float(med @ proto[m]))
        for i in range(len(stage_m) - 1):
            trans.append((stage_m[i], stage_m[i + 1], stage_med[i], stage_med[i + 1]))

    # M1
    m1_mean, m1_std = float(np.mean(cos_med_cent)), float(np.std(cos_med_cent))

    # M2: cross-episode consistency per milestone (sampled pairwise cos)
    rng = np.random.default_rng(0)
    m2 = []
    for m, meds in medoids_by_m.items():
        if len(meds) < 2:
            continue
        arr = np.stack(meds)
        k = min(len(arr), 200)
        sub = arr[rng.choice(len(arr), k, replace=False)]
        sims = sub @ sub.T
        iu = np.triu_indices(k, 1)
        m2.append(float(sims[iu].mean()))
    m2_mean = float(np.mean(m2))

    # M3: predicting the next-stage medoid.
    #  (a) centroid predictor: cos(proto[next_m], true_next_medoid)
    #  (b) current-medoid carry-over: cos(cur_medoid, true_next_medoid) (naive persistence)
    #  (c) does current frame beat centroid? centroid is E[medoid|next_m]; if (a) already
    #      high, episode detail is largely unpredictable (regression-to-mean).
    ca, cb = [], []
    for cur_m, next_m, cur_med, next_med in trans:
        ca.append(float(proto[next_m] @ next_med))
        cb.append(float(cur_med @ next_med))
    m3_centroid = float(np.mean(ca))
    m3_persist = float(np.mean(cb))

    summary = {
        "num_milestones": num_m,
        "num_stages": int(sum(len(v) for v in medoids_by_m.values())),
        "num_transitions": len(trans),
        "M1_cos_medoid_vs_centroid": {"mean": m1_mean, "std": m1_std},
        "M2_cross_episode_medoid_consistency_mean_cos": m2_mean,
        "M3_predict_next_medoid": {
            "centroid_predictor_cos": m3_centroid,
            "persistence_cur_medoid_cos": m3_persist,
        },
        "interpretation_hooks": {
            "our_proto_head_cos_to_centroid_target": 0.94,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
