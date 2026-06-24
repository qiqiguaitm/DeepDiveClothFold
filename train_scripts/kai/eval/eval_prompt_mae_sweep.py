#!/usr/bin/env python3
"""同一 ckpt 在多个 prompt 下的 action MAE 对照 (paired, 单次加载).

针对 AWBC ckpt: 测 "Advantage: positive" / "Advantage: negative" / 裸 prompt
三种文本条件下, 模型复现 *好* 演示动作的 MAE@{1,10,25,50}。

设计要点:
  - 策略只加载一次, 每个 episode 视频只解码一次。
  - 三个 prompt 用 *完全相同* 的 (obs, GT) 帧 → 差异纯粹来自文本条件 (paired)。
  - 除 overall MAE, 另拆 joints(12 维) vs grippers(dim 6,13) — 折叠"决策"主要在夹爪。
  - 额外报 "条件化强度": mean|a_pos - a_neg|, 看模型到底有没有理睬 advantage token。

用法 (从 kai0/ 跑):
  CUDA_VISIBLE_DEVICES=2 OPENPI_EXTRA_CONFIG=<ckpt>/train_config.json \
  XLA_FLAGS="--xla_gpu_autotune_level=0" \
  .venv/bin/python ../train_scripts/kai/eval/eval_prompt_mae_sweep.py \
    --config pi05_flatten_fold_A_0518_v2_201 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_v0/awbc_step49999 \
    --val  data/Task_A/self_built/A_new_pure_200_val \
    --n-frames 50
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

HORIZONS = (1, 10, 25, 50)
GRIPPER_DIMS = (6, 13)
JOINT_DIMS = tuple(i for i in range(14) if i not in GRIPPER_DIMS)

PROMPTS = {
    "positive": "Flatten and fold the cloth. Advantage: positive",
    "negative": "Flatten and fold the cloth. Advantage: negative",
    "plain":    "Flatten and fold the cloth.",
}


def read_video_frames(path: Path, n_frames: int) -> np.ndarray:
    import av
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    out = []
    for frame in container.decode(stream):
        out.append(frame.to_ndarray(format="rgb24"))
        if len(out) >= n_frames:
            break
    container.close()
    arr = np.stack(out[:n_frames], axis=0)
    if arr.shape[0] < n_frames:
        arr = np.concatenate([arr, np.repeat(arr[-1:], n_frames - arr.shape[0], 0)], 0)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--n-frames", type=int, default=50, help="每个 episode 抽样的 query 帧数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import openpi.training.config as C
    from openpi.policies import policy_config as PC

    cfg = C.get_config(args.config)
    t0 = time.time()
    pol = PC.create_trained_policy(cfg, args.ckpt)
    print(f"[load] policy ready in {time.time()-t0:.1f}s", flush=True)

    val = Path(args.val)
    episodes = [json.loads(l) for l in (val / "meta" / "episodes.jsonl").read_text().splitlines()]

    # acc_mae[prompt][horizon] -> list of per-frame MAE (over all dims)
    acc = {p: {h: [] for h in HORIZONS} for p in PROMPTS}
    acc_j = {p: {h: [] for h in HORIZONS} for p in PROMPTS}   # joints only
    acc_g = {p: {h: [] for h in HORIZONS} for p in PROMPTS}   # grippers only
    cond_strength = []   # mean|a_pos - a_neg| over full chunk, per query frame
    n_infer = 0
    t_start = time.time()

    for ei, ep in enumerate(episodes):
        ep_idx = ep["episode_index"]; L = ep["length"]
        df = pq.read_table(val / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet").to_pandas()
        state = np.stack([np.asarray(x) for x in df["observation.state"]])   # (L,14)
        action = np.stack([np.asarray(x) for x in df["action"]])            # (L,14)
        cams = {}
        for cam in ("top_head", "hand_left", "hand_right"):
            vp = val / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
            cams[cam] = read_video_frames(vp, L)

        hi = L - max(HORIZONS) - 1
        if hi <= 0:
            continue
        q = np.linspace(0, hi, min(args.n_frames, hi + 1)).astype(int)

        for k in q:
            base_obs = {
                "images": {c: cams[c][k] for c in cams},
                "state": state[k],
            }
            preds = {}
            for p, ptext in PROMPTS.items():
                r = pol.infer({**base_obs, "prompt": ptext})
                preds[p] = np.asarray(r["actions"])  # (50,14)
                n_infer += 1
            cl = min(min(pr.shape[0] for pr in preds.values()), max(HORIZONS))
            cond_strength.append(float(np.mean(np.abs(preds["positive"][:cl] - preds["negative"][:cl]))))
            for p in PROMPTS:
                for h in HORIZONS:
                    if h > cl:
                        continue
                    gt = action[k + 1: k + 1 + h]      # (h,14)
                    ph = preds[p][:h]                  # (h,14)
                    d = np.abs(gt - ph)
                    acc[p][h].append(float(d.mean()))
                    acc_j[p][h].append(float(d[:, list(JOINT_DIMS)].mean()))
                    acc_g[p][h].append(float(d[:, list(GRIPPER_DIMS)].mean()))
        el = time.time() - t_start
        print(f"  ep{ep_idx:02d} ({ei+1}/{len(episodes)})  frames={len(q)}  infers={n_infer}  elapsed={el:.0f}s", flush=True)

    def red(a):
        return {p: {h: (float(np.mean(a[p][h])) if a[p][h] else None) for h in HORIZONS} for p in PROMPTS}

    summary = {
        "ckpt": args.ckpt, "val": str(val), "n_episodes": len(episodes),
        "n_frames_per_ep": args.n_frames, "n_infer_total": n_infer,
        "prompts": PROMPTS,
        "mae_overall": red(acc),
        "mae_joints": red(acc_j),
        "mae_grippers": red(acc_g),
        "cond_strength_pos_vs_neg": float(np.mean(cond_strength)) if cond_strength else None,
    }

    print("\n================ MAE sweep (lower = closer to GOOD demos) ================")
    hdr = "prompt".ljust(10) + "".join(f"  @{h:<7}" for h in HORIZONS)
    for tag, tbl in (("OVERALL", summary["mae_overall"]),
                     ("JOINTS",  summary["mae_joints"]),
                     ("GRIPPERS",summary["mae_grippers"])):
        print(f"\n--- {tag} ---\n{hdr}")
        for p in PROMPTS:
            row = "".join(f"  {tbl[p][h]:.5f}" if tbl[p][h] is not None else "   n/a   " for h in HORIZONS)
            print(p.ljust(10) + row)
    print(f"\ncond_strength  mean|a_pos - a_neg| over chunk = {summary['cond_strength_pos_vs_neg']:.5f}")
    print("(≈0 → 模型基本无视 advantage token; 越大 → 文本条件越强)")

    out = Path(args.out) if args.out else Path(args.ckpt) / "eval_prompt_mae_sweep.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
