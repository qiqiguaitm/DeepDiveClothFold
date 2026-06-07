#!/usr/bin/env python3
"""Compute 14D action + state normalization stats for wam_fold_v1 (cosmos-framework format).
Outputs both quantile (cosmos training) and mean/std (GWP eval z-score) stats.
Samples episodes for speed; quantiles are stable on ~300k frames."""
import json, glob, sys, numpy as np, pandas as pd
from pathlib import Path

# usage: compute_action_stats.py [N_EP] [ROOT] [OUT_JSON]
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_train")
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/data/stats/visrobot01.json")
parquets = sorted(glob.glob(str(ROOT / "data" / "chunk-*" / "episode_*.parquet")))
if not parquets:
    print("NO PARQUETS FOUND at", ROOT); sys.exit(1)
# uniform sample across the set
idx = np.linspace(0, len(parquets) - 1, min(N_EP, len(parquets))).astype(int)
sel = [parquets[i] for i in idx]
print(f"sampling {len(sel)} / {len(parquets)} episodes")
acts, states = [], []
for i, p in enumerate(sel):
    try:
        df = pd.read_parquet(p, columns=["action", "observation.state"])
    except Exception as e:
        print("skip", p, e); continue
    acts.append(np.stack(df["action"].to_numpy()))
    states.append(np.stack(df["observation.state"].to_numpy()))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(sel)}")
A = np.concatenate(acts, 0).astype(np.float64)   # [N,14]
S = np.concatenate(states, 0).astype(np.float64)
print("action frames:", A.shape, "state frames:", S.shape)

def stats(X):
    return dict(
        mean=X.mean(0).tolist(), std=(X.std(0) + 1e-8).tolist(),
        min=X.min(0).tolist(), max=X.max(0).tolist(),
        q01=np.quantile(X, 0.01, 0).tolist(), q99=np.quantile(X, 0.99, 0).tolist(),
    )

out = {
    "global": {"action": stats(A), "observation.state": stats(S)},
    "n_frames": int(A.shape[0]), "n_episodes": len(sel), "action_dim": A.shape[1],
}
dst = OUT
dst.write_text(json.dumps(out, indent=2))
print("WROTE", dst)
print("action mean:", [round(x, 4) for x in out["global"]["action"]["mean"]])
print("action std :", [round(x, 4) for x in out["global"]["action"]["std"]])
print("STATS_DONE")
