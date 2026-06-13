"""Dump a canonical eval frame-set (identical windows/GT to episode_report) for
cross-model comparison (gwp_ans vs kai0). Run in the gwp venv.

Each window -> one npz: {top_head,hand_left,hand_right (HWC uint8), state[14], gt[48,14], ep, f}.
GT = dataset action[f:f+action_chunk] (absolute joints), same as episode_report.
"""
import argparse, os
from collections import OrderedDict
import numpy as np
import torch

from wam_pipeline.eval_watch import build_window_indices, EpisodeFrameCache

VK = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
# gwp view -> kai0 camera name
VK2KAI = {"observation.images.cam_high": "top_head",
          "observation.images.cam_left_wrist": "hand_left",
          "observation.images.cam_right_wrist": "hand_right"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--action_chunk", type=int, default=48)
    ap.add_argument("--exec_horizon", type=int, default=16)
    ap.add_argument("--n_metric_eps", type=int, default=50)
    ap.add_argument("--max_win_per_ep", type=int, default=3)
    ap.add_argument("--frame_cache", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from giga_datasets import load_dataset
    ve = dict(_class_name="LeRobotDataset", data_path=args.val_root, delta_info={"action": args.action_chunk},
              skip_video_decoding=True, embodiment="visrobot01", tolerance_s=1e-3)
    ds = load_dataset([ve])

    idx, gs, info = build_window_indices(args.val_root, "exec", args.exec_horizon, args.action_chunk, args.exec_horizon)
    ep2win = OrderedDict()
    for gi in idx:
        ep2win.setdefault(info[gi][0], []).append(gi)
    metric_eps = list(ep2win.keys())[:args.n_metric_eps]
    fc = EpisodeFrameCache(args.val_root, VK, args.frame_cache)

    n = 0
    manifest = []
    for ki, ep in enumerate(metric_eps):
        if ki + 1 < len(metric_eps):
            fc.prefetch(metric_eps[ki + 1])
        wins = ep2win[ep]
        if len(wins) > args.max_win_per_ep:
            wins = [wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, args.max_win_per_ep).astype(int))]
        fr = fc.get(ep)
        for gi in wins:
            d = ds[int(gi)]
            _, f = info[int(gi)]
            state = d["observation.state"].float().numpy().reshape(-1)[:14]
            gt = d["action"].float().numpy()[:, :14]
            rec = {"state": state.astype(np.float32), "gt": gt.astype(np.float32),
                   "ep": int(ep), "f": int(f)}
            for vk in VK:
                rec[VK2KAI[vk]] = fr[vk][f]  # HWC uint8
            p = os.path.join(args.out_dir, f"win_{n:05d}.npz")
            np.savez_compressed(p, **rec)
            manifest.append(p)
            n += 1
        print(f"  ep{ep} -> {len(wins)} win (total {n})", flush=True)
    with open(os.path.join(args.out_dir, "manifest.txt"), "w") as fh:
        fh.write("\n".join(manifest))
    print(f"DUMPED {n} windows -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
