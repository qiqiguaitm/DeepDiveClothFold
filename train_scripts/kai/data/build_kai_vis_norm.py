"""Compute PER-DOMAIN norm_stats for kai_vis_merged (C2 of per-DS-norm + conditioning run).

Single pass over the merged dataset's parquets, bucket each frame by `task_index`
(0=kai, 1=vis), accumulate two RunningStats → two norm_stats. Identical numerics to
norm_stats_from_dataset.compute_norm_stats (pad to action_dim, [-pi,pi] filter, batch=32).

Writes (norm_stats.json lives in the dataset dir per the asset_id=repo_id convention):
  <merged>/norm_domain0_kai/norm_stats.json   (kai = kai0_base + kai0_dagger frames)
  <merged>/norm_domain1_vis/norm_stats.json   (vis = A_smooth800_dagger_full frames)
  <merged>/norm_stats.json                     (= vis norm; single fallback for create_base_config
                                                + output Unnormalize on the vis deploy target)

Run with kai0 venv.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from norm_stats_from_dataset import _process, _ensure_openpi_on_path

DOMAIN_DIR = {0: "norm_domain0_kai", 1: "norm_domain1_vis"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/kai_vis_merged")
    ap.add_argument("--action-dim", type=int, default=32)
    a = ap.parse_args()
    _ensure_openpi_on_path()
    import openpi.shared.normalize as normalize

    base = Path(a.merged)
    pqs = sorted(str(p) for p in (base / "data").rglob("*.parquet"))
    if not pqs:
        sys.exit(f"no parquet under {base}/data")
    print(f"computing per-domain norm over {len(pqs)} episodes ...", flush=True)

    rs = {0: {"state": normalize.RunningStats(), "actions": normalize.RunningStats()},
          1: {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}}
    counts = {0: 0, 1: 0}
    buf = {0: {"state": [], "actions": []}, 1: {"state": [], "actions": []}}

    def flush(dom):
        for key in ("state", "actions"):
            if buf[dom][key]:
                arr = np.stack(buf[dom][key])
                for i in range(0, len(arr), 32):
                    rs[dom][key].update(arr[i:i + 32])
                buf[dom][key] = []

    for fp in tqdm(pqs, desc="per-domain norm"):
        df = pd.read_parquet(fp, columns=["observation.state", "action", "task_index"])
        dom = int(np.asarray(df["task_index"].iloc[0]).reshape(-1)[0])
        for i in range(len(df)):
            buf[dom]["state"].append(_process(np.array(df["observation.state"].iloc[i]), a.action_dim))
            buf[dom]["actions"].append(_process(np.array(df["action"].iloc[i]), a.action_dim))
        counts[dom] += len(df)
        flush(dom)   # flush per-episode to bound memory (still batch=32 inside)

    for dom in (0, 1):
        stats = {key: rs[dom][key].get_statistics() for key in ("state", "actions")}
        outdir = base / DOMAIN_DIR[dom]
        outdir.mkdir(exist_ok=True)
        normalize.save(outdir, stats)
        print(f"✅ domain{dom} ({DOMAIN_DIR[dom]}): frames={counts[dom]} → {outdir}/norm_stats.json", flush=True)
        if dom == 1:   # vis = single fallback + output norm (deploy target)
            normalize.save(base, stats)
            print(f"✅ single fallback (vis) → {base}/norm_stats.json", flush=True)


if __name__ == "__main__":
    main()
