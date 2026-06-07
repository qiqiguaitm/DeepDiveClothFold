#!/usr/bin/env python3
"""Detect open-loop grasp-retry motifs in KAI0 Task_A episodes.

Symptom (user 2026-06-07): models trained on 5-19~5-27 execute a fixed grasp
trajectory regardless of whether the cloth was actually caught, then "realize"
the miss and loop back to re-grasp -- i.e. the grasp + following sub-trajectory
became OPEN-LOOP / vision-ignoring, only the action *order* was learned.

Hypothesis (= H2 in data_root_cause_probe_results.md §4.3 #2): the training data
itself contains failed-grasp-then-regrasp loops (operator missed during teleop
and retried). BC copies the retry as a fixed motif and reproduces it open-loop.

This scans the parquet action stream (no video decode) and per episode reports:
  * grasp cycles per arm = open->closed transitions
  * SHORT closures        = closed segment < --short-frames then reopened
                            (a grasp that grabbed nothing and was abandoned)
  * ABORTED-in-place      = SHORT closure during which the arm barely moved
                            (sum |dq_arm| over the closed segment < --still-rad):
                            the literal "close, nothing happened, reopen, retry"
  * regrasp loops         = >=2 grasp cycles whose pre-grasp arm pose is similar
                            (returned to ~same spot to try again)
  * idle frame fraction   = frames with max|dq_arm| < --idle-eps (policy-idling fuel)

Gripper convention in this data: value in meters, ~0 = CLOSED, ~0.08 = OPEN.

Usage:
  kai0/.venv/bin/python train_scripts/kai/data/scan_grasp_retry.py \
      --root /data1/DATA_IMP/KAI0/Task_A/base \
      --dates 2026-05-19-v2,2026-05-20-v2,...  [--top 20]
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pyarrow.parquet as pq

GRIP = {"L": 6, "R": 13}
ARM = {"L": list(range(0, 6)), "R": list(range(7, 13))}

# gripper hysteresis thresholds (meters): 0=closed .. 0.08=open
CLOSED = 0.020
OPEN = 0.045


def segments_closed(g):
    """Return list of (start, end_exclusive) frame ranges where gripper is held
    CLOSED, using hysteresis: enter closed at g<CLOSED, leave at g>OPEN."""
    segs = []
    state = "open"  # start assumed open-ish; first sample fixes it
    s = None
    for i, v in enumerate(g):
        if state == "open":
            if v < CLOSED:
                state = "closed"; s = i
        else:  # closed
            if v > OPEN:
                segs.append((s, i)); state = "open"; s = None
    if state == "closed":
        segs.append((s, len(g)))
    return segs


def scan_episode(path, short_frames, still_rad, idle_eps, pose_tol):
    t = pq.read_table(path, columns=["action"]).to_pandas()
    a = np.stack([np.asarray(x) for x in t["action"]]).astype(np.float32)
    L = len(a)
    dq = np.abs(np.diff(a, axis=0))  # [L-1, 14]
    out = {"len": L}
    # idle fraction over both arms
    arm_all = ARM["L"] + ARM["R"]
    idle = (dq[:, arm_all].max(axis=1) < idle_eps).mean() if L > 1 else 0.0
    out["idle_frac"] = float(idle)
    for arm in ("L", "R"):
        g = a[:, GRIP[arm]]
        ad = dq[:, ARM[arm]]  # [L-1, 6]
        segs = segments_closed(g)
        cycles = len(segs)
        short = 0
        aborted = 0
        durations = []
        # pre-grasp arm pose = mean arm joints in the 5 frames before each close
        pre_poses = []
        for (s, e) in segs:
            dur = e - s
            durations.append(dur)
            # arm travel while gripper held closed
            travel = float(ad[s:max(s + 1, e - 1)].sum()) if e - 1 > s else 0.0
            if dur < short_frames:
                short += 1
                if travel < still_rad:
                    aborted += 1
            p0 = max(0, s - 5)
            pre_poses.append(a[p0:s + 1, ARM[arm]].mean(axis=0) if s > 0 else a[0, ARM[arm]])
        # regrasp loops: consecutive cycles whose pre-grasp poses are within pose_tol
        loops = 0
        for i in range(1, len(pre_poses)):
            if np.max(np.abs(pre_poses[i] - pre_poses[i - 1])) < pose_tol:
                loops += 1
        out[arm] = dict(cycles=cycles, short=short, aborted=aborted, loops=loops,
                        durs=durations)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data1/DATA_IMP/KAI0/Task_A/base")
    ap.add_argument("--dates", default="2026-05-19-v2,2026-05-20-v2,2026-05-21-v2,"
                    "2026-05-22-v2,2026-05-26-v2,2026-05-27-v2")
    ap.add_argument("--short-frames", type=int, default=18, help="closure shorter than this = candidate failed grasp (~0.6s @30fps)")
    ap.add_argument("--still-rad", type=float, default=0.25, help="sum|dq_arm| over closure below this = aborted-in-place")
    ap.add_argument("--idle-eps", type=float, default=0.0020, help="max|dq_arm|/frame below this = idle frame")
    ap.add_argument("--pose-tol", type=float, default=0.12, help="pre-grasp arm-pose match (rad) for regrasp loop")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    rows = []
    for d in dates:
        for f in sorted(glob.glob(f"{args.root}/{d}/data/chunk-000/*.parquet")):
            try:
                r = scan_episode(f, args.short_frames, args.still_rad, args.idle_eps, args.pose_tol)
            except Exception as ex:
                print("ERR", f, ex); continue
            r["date"] = d; r["ep"] = os.path.basename(f)
            rows.append(r)

    def tot(r, k):
        return r["L"][k] + r["R"][k]

    # per-day aggregate
    print(f"{'date':16s} {'n':>4s} {'med_len':>7s} {'idle%':>6s} "
          f"{'cyc/ep':>6s} {'short/ep':>8s} {'abort/ep':>8s} {'loop/ep':>7s} {'%ep_abort':>9s} {'%ep_loop':>8s}")
    for d in dates:
        dr = [r for r in rows if r["date"] == d]
        if not dr:
            continue
        n = len(dr)
        med_len = int(np.median([r["len"] for r in dr]))
        idle = 100 * np.mean([r["idle_frac"] for r in dr])
        cyc = np.mean([tot(r, "cycles") for r in dr])
        sh = np.mean([tot(r, "short") for r in dr])
        ab = np.mean([tot(r, "aborted") for r in dr])
        lp = np.mean([tot(r, "loops") for r in dr])
        pab = 100 * np.mean([tot(r, "aborted") > 0 for r in dr])
        plp = 100 * np.mean([tot(r, "loops") > 0 for r in dr])
        print(f"{d:16s} {n:4d} {med_len:7d} {idle:6.1f} {cyc:6.2f} {sh:8.2f} {ab:8.2f} {lp:7.2f} {pab:9.1f} {plp:8.1f}")

    # worst episodes by (aborted + loops)
    rows.sort(key=lambda r: (tot(r, "aborted") + tot(r, "loops"), tot(r, "short")), reverse=True)
    print(f"\n=== top {args.top} worst episodes (aborted-in-place + regrasp-loops) ===")
    print(f"{'date':16s} {'ep':>18s} {'len':>5s} {'idle%':>6s} "
          f"{'Lcyc':>4s} {'Lsh':>4s} {'Lab':>4s} {'Llp':>4s} | {'Rcyc':>4s} {'Rsh':>4s} {'Rab':>4s} {'Rlp':>4s}")
    for r in rows[:args.top]:
        print(f"{r['date']:16s} {r['ep']:>18s} {r['len']:5d} {100*r['idle_frac']:6.1f} "
              f"{r['L']['cycles']:4d} {r['L']['short']:4d} {r['L']['aborted']:4d} {r['L']['loops']:4d} | "
              f"{r['R']['cycles']:4d} {r['R']['short']:4d} {r['R']['aborted']:4d} {r['R']['loops']:4d}")

    # global
    n = len(rows)
    print(f"\n=== global ({n} eps) ===")
    print(f"  mean grasp cycles/ep     = {np.mean([tot(r,'cycles') for r in rows]):.2f}")
    print(f"  mean short closures/ep   = {np.mean([tot(r,'short') for r in rows]):.2f}")
    print(f"  mean aborted-in-place/ep = {np.mean([tot(r,'aborted') for r in rows]):.2f}")
    print(f"  mean regrasp-loops/ep    = {np.mean([tot(r,'loops') for r in rows]):.2f}")
    print(f"  eps with >=1 aborted     = {100*np.mean([tot(r,'aborted')>0 for r in rows]):.1f}%")
    print(f"  eps with >=1 regrasp-loop= {100*np.mean([tot(r,'loops')>0 for r in rows]):.1f}%")
    print(f"  eps with >=2 aborted     = {100*np.mean([tot(r,'aborted')>=2 for r in rows]):.1f}%")

    if args.out:
        for r in rows:
            r.pop("L", None); r.pop("R", None)
        json.dump(rows, open(args.out, "w"))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
