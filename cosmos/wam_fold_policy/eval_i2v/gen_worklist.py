"""Generate the eval work list: evenly-spaced anchors per (episode,camera,horizon).

Honors "more starts for shorter horizon" via per-horizon anchor counts, but caps the
expensive 64B count (full non-overlapping tiling of 1s = 516 windows/model is infeasible
for the 64B models). Anchors are spread across the whole episode → full-episode phase
coverage + metric-vs-horizon. One anchor per (ep,cam,horizon) is flagged save_video for
the report gallery (cam_high only, to keep it small).
"""
import json, math, argparse

DATA = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_val"
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
SRC = {"1s": 30, "3s": 90, "7s": 236}   # GT source frames @30fps per horizon

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[102, 126, 168])
    ap.add_argument("--n_anchor", type=str, default="1s:8,3s:5,7s:3")  # anchors per (ep,cam,horizon)
    ap.add_argument("--out", default="worklist.json")
    args = ap.parse_args()
    nA = {k: int(v) for k, v in (kv.split(":") for kv in args.n_anchor.split(","))}

    L = {json.loads(l)["episode_index"]: json.loads(l)["length"]
         for l in open(f"{DATA}/meta/episodes.jsonl")}
    units = []
    for ep in args.episodes:
        N = L[ep]
        for cam in CAMS:
            for hz, k in nA.items():
                span = SRC[hz]
                last = max(0, N - span)             # last valid anchor so the GT window fits
                if last == 0:
                    anchors = [0]
                else:
                    anchors = [int(round(i * last / max(1, k - 1))) for i in range(k)]
                    anchors = sorted(set(anchors))
                for j, a in enumerate(anchors):
                    save = (cam == "cam_high" and j == len(anchors) // 2)  # mid anchor, head cam
                    units.append([ep, cam, hz, a, bool(save)])
    json.dump(units, open(args.out, "w"))
    n_save = sum(1 for u in units if u[4])
    print(f"{len(units)} work units -> {args.out}  (save_video={n_save})")
    # per-horizon breakdown
    from collections import Counter
    c = Counter(u[2] for u in units)
    print("per-horizon:", dict(c), " x3 models =", {k: v * 3 for k, v in c.items()})

if __name__ == "__main__":
    main()
