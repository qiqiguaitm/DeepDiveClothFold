"""A1 (AWBC milestone-value A臂): build ds_A from V2.4 mv_value sidecar.

For each episode of A_smooth800_dagger_all (drop the |corr|<0.5 bad eps), add the adapter columns
(same names as pi0-AE Stage 2 → downstream discretize/AWBC unchanged):
  absolute_value[t]     = mv_value[t]
  absolute_advantage[t] = clip(mv_value[min(t+50,end)] − mv_value[t], −1, 1)   (Δ=50, no sign flip)
Renumber contiguous, symlink videos to source realpath, copy meta, compute norm_stats.
Output: kai0/data/Task_A/self_built/dagger_all_mvA   (then discretize_advantage.py + AWBC).

Run: kai0/.venv/bin/python train_scripts/kai/data/build_ds_A_from_mv.py
"""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_no_release import per_episode_stats  # noqa: E402

ROOT = Path("/home/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built")
SRC = ROOT / "A_smooth800_dagger_all"
MV = Path("/home/tim/workspace/deepdive_kai0/temp/mv_value_full")
OUT = ROOT / "dagger_all_mvA"
CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
PROMPT = "Flatten and fold the cloth."
WIN = 50
SMOOTH = 41   # moving-average window: ramp the V2.4 milestone staircase → continuous progress
              # (else 58.6% of 50-frame diffs are exactly 0 → discretize can't quantile-match C's 25.2%neg)


def smooth_monotone(v, w=SMOOTH):
    """Edge-padded moving average → continuous; re-clip to [0,1]. Staircase plateaus become ramps."""
    if len(v) < 3 or w < 2:
        return v.astype(np.float32)
    h = w // 2
    vp = np.concatenate([np.full(h, v[0]), v, np.full(h, v[-1])])
    k = np.ones(w, dtype=np.float64) / w
    vs = np.convolve(vp, k, mode="valid")[: len(v)]
    return np.clip(vs, 0.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-norm", action="store_true")
    a = ap.parse_args()
    cs = json.loads((SRC / "meta" / "info.json").read_text())["chunks_size"]
    corr = json.load(open(MV / "corr.json"))
    bad = set(corr["bad_lt0.5"])
    src_eps = [int(p.stem.split("_")[1]) for p in sorted((SRC / "data").glob("chunk-*/episode_*.parquet"))]
    keep = [e for e in src_eps if e not in bad]
    print(f"source {len(src_eps)} ep, drop {len(bad)} bad → {len(keep)} ep", flush=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / "chunk-000").mkdir(parents=True)
    (OUT / "meta").mkdir()
    for cam in CAMERAS:
        (OUT / "videos" / "chunk-000" / cam).mkdir(parents=True)

    eps_meta, stats_out = [], []
    total_frames = 0
    for new_ep, e in enumerate(keep):
        sc = e // cs
        df = pd.read_parquet(SRC / "data" / f"chunk-{sc:03d}" / f"episode_{e:06d}.parquet")
        mv = np.load(MV / f"ep{e}.npy").astype(np.float32)
        n = len(df)
        if len(mv) != n:  # defensive align
            mv = np.resize(mv, n) if len(mv) else np.zeros(n, np.float32)
        # staircase → continuous (smooth only; NO time-prior). discretize 用 advantage>=0 → positive:
        # milestone-value 里 plateau(adv==0=守住里程碑不退步)= 正, 只有真退步(adv<0)= 负.
        mv = smooth_monotone(mv)
        adv = np.clip(mv[np.minimum(np.arange(n) + WIN, n - 1)] - mv, -1.0, 1.0).astype(np.float32)
        df = df.reset_index(drop=True)
        df["episode_index"] = np.int64(new_ep)
        df["index"] = np.arange(total_frames, total_frames + n, dtype=np.int64)
        df["absolute_value"] = mv
        df["absolute_advantage"] = adv
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       OUT / "data" / "chunk-000" / f"episode_{new_ep:06d}.parquet")
        for cam in CAMERAS:
            sv = SRC / "videos" / f"chunk-{sc:03d}" / cam / f"episode_{e:06d}.mp4"
            if not (sv.exists() or sv.is_symlink()):
                raise FileNotFoundError(f"missing video {sv}")
            os.symlink(str(sv.resolve()), OUT / "videos" / "chunk-000" / cam / f"episode_{new_ep:06d}.mp4")
        eps_meta.append({"episode_index": new_ep, "tasks": [PROMPT], "length": n, "src_ep": e})
        stats_out.append({"episode_index": new_ep, "stats": per_episode_stats(df[["observation.state", "action"]])})
        total_frames += n
        if (new_ep + 1) % 300 == 0:
            print(f"  {new_ep+1}/{len(keep)}", flush=True)

    info = json.loads((SRC / "meta" / "info.json").read_text())
    info["total_episodes"] = len(keep)
    info["total_frames"] = total_frames
    info["total_videos"] = len(keep) * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(1000, len(keep))
    info["splits"] = {"train": f"0:{len(keep)}"}
    (OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (OUT / "meta" / "episodes.jsonl").open("w") as f:
        for em in eps_meta:
            f.write(json.dumps(em) + "\n")
    with (OUT / "meta" / "episodes_stats.jsonl").open("w") as f:
        for st in stats_out:
            f.write(json.dumps(st) + "\n")
    (OUT / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
    print(f"  built {len(keep)} ep / {total_frames} frames -> {OUT}", flush=True)

    if not a.no_norm:
        from norm_stats_from_dataset import compute_norm_stats
        print("  computing norm_stats (action_dim=32)...", flush=True)
        compute_norm_stats(str(OUT), action_dim=32)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
