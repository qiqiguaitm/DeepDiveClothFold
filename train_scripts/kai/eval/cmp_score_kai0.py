#!/usr/bin/env python3
"""Score a kai0 JAX policy on the dumped comparison frames (from cmp_dump_frames.py).
Run in kai0/.venv. Same metric as cmp_score_gwp.py:
   mae@h = |pred[h-1]-gt[h-1]|.mean(), averaged over windows (absolute joint space).

Usage (from kai0/):
  XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  ../train_scripts/.../cmp_score_kai0.py --config pi05_flatten_fold_A_0518_v2_201 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_v1/A_smooth800_dagger_full_step49999 \
    --frames_dir /data2/gwp_eval/out/cmp_frames --out /data2/gwp_eval/out/cmp_kai0.json
"""
import argparse, glob, json, os
import numpy as np

HOR = [1, 10, 24, 48]
PROMPT = "Flatten and fold the cloth."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    import dataclasses, json as _json
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _tc
    from openpi.training.config import AssetsConfig
    cfg = _tc.get_config(args.config)
    # 用 ckpt 自带 assets (train_config.json 的 override_asset_id), 否则会去训练集群路径找 norm_stats。
    tcj = _json.load(open(os.path.join(args.ckpt, "train_config.json")))
    asset_id = tcj.get("override_asset_id")
    if asset_id:
        cfg = dataclasses.replace(
            cfg, data=dataclasses.replace(
                cfg.data, assets=AssetsConfig(assets_dir=os.path.join(args.ckpt, "assets"), asset_id=asset_id)))
        print(f"[kai0] asset override -> {args.ckpt}/assets/{asset_id}", flush=True)
    policy = _pc.create_trained_policy(cfg, args.ckpt)
    print("[kai0] policy ready", flush=True)

    files = sorted(glob.glob(os.path.join(args.frames_dir, "win_*.npz")))
    acc = {h: [] for h in HOR}
    for i, fp in enumerate(files):
        z = np.load(fp)
        obs = {"images": {"top_head": z["top_head"], "hand_left": z["hand_left"], "hand_right": z["hand_right"]},
               "state": z["state"].astype(np.float32), "prompt": args.prompt}
        pred = np.asarray(policy.infer(obs)["actions"])[:, :14]  # [chunk,14] absolute joints
        gt = z["gt"]
        L = min(len(pred), len(gt)); ae = np.abs(pred[:L] - gt[:L])
        for h in HOR:
            if h <= L: acc[h].append(float(ae[h - 1].mean()))
        if (i + 1) % 25 == 0: print(f"  {i+1}/{len(files)}", flush=True)
    res = {"model": f"kai0:{os.path.basename(args.ckpt)}", "config": args.config,
           "n_win": len(files), "mae": {h: float(np.mean(acc[h])) for h in HOR}}
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
