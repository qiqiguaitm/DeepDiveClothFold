"""LeRobot(WAM)数据集健康体检 —— 只读,有问题返回非零。

校验项(逐 episode):
  - meta 一致:episodes.jsonl 索引 0..N-1 连续且数=info.total_episodes;
    episodes_stats.jsonl 覆盖全部 ep;tasks.jsonl 存在;codebase_version。
  - parquet 完整:文件存在、行数=episodes.jsonl 的 length、帧合计=info.total_frames。
  - state/action:形状 (n,14)、全 finite、无尖刺(|value|<=阈值,默认 10)。
  - frame_index=0..n-1;timestamp≈frame_index/fps(容差 1e-3);episode_index 列恒等于 ep;
    global index 全局连续无缝。
  - 视频齐全:3 个相机 key 每集 mp4 都在(仅查存在,不解码)。

用法:
  python -m scripts.wam_pipeline.health_check                 # 默认体检 wam_fold_v1/{vis,kai}
  python -m scripts.wam_pipeline.health_check <root> [<root> ...]
  python -m scripts.wam_pipeline.health_check <root> --spike-threshold 10
"""

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

DEFAULT_ROOTS = [
    "../kai0/data/wam_fold_v1/visrobot01",
    "../kai0/data/wam_fold_v1/kairobot01",
]
CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def check(root, spike):
    errs, warns = [], []
    info = json.load(open(f"{root}/meta/info.json"))
    fps, cs = info["fps"], info["chunks_size"]
    tmpl, vtmpl = info["data_path"], info["video_path"]
    te, tf = info["total_episodes"], info["total_frames"]

    eps = [json.loads(l) for l in open(f"{root}/meta/episodes.jsonl") if l.strip()]
    idxs = [e["episode_index"] for e in eps]
    lens = {e["episode_index"]: e["length"] for e in eps}
    if idxs != list(range(len(idxs))):
        errs.append("episodes.jsonl 索引非 0..N-1 连续")
    if len(eps) != te:
        errs.append(f"episodes.jsonl 条数 {len(eps)} != info.total_episodes {te}")

    stat_path = f"{root}/meta/episodes_stats.jsonl"
    if not os.path.isfile(stat_path):
        errs.append("缺 meta/episodes_stats.jsonl")
    else:
        st_idx = sorted(json.loads(l)["episode_index"] for l in open(stat_path) if l.strip())
        if st_idx != sorted(idxs):
            errs.append(f"episodes_stats 覆盖 {len(st_idx)} != episodes {len(idxs)}")
    if not os.path.isfile(f"{root}/meta/tasks.jsonl"):
        errs.append("缺 meta/tasks.jsonl")
    if info.get("codebase_version") != "v2.1":
        warns.append(f"codebase_version={info.get('codebase_version')}")

    running_index, sum_len = 0, 0
    agg = dict(spike=0, bad_timestamp=0, bad_frame_index=0, nonfinite=0,
               missing_video=0, row_mismatch=0, global_index_gap=0, missing_parquet=0)
    bad_eps = []
    for ep in idxs:
        chunk = ep // cs
        pqf = os.path.join(root, tmpl.format(episode_chunk=chunk, episode_index=ep))
        if not os.path.isfile(pqf):
            agg["missing_parquet"] += 1; bad_eps.append(ep); continue
        d = pq.read_table(pqf, columns=["observation.state", "action", "timestamp",
                                        "frame_index", "episode_index", "index"]).to_pydict()
        n = len(d["frame_index"])
        if n != lens[ep]:
            agg["row_mismatch"] += 1; bad_eps.append(ep)
        S = np.asarray([np.asarray(x) for x in d["observation.state"]], dtype=np.float64)
        A = np.asarray([np.asarray(x) for x in d["action"]], dtype=np.float64)
        if S.shape != (n, 14) or A.shape != (n, 14):
            errs.append(f"ep{ep} state/action 维度异常 {S.shape}/{A.shape}"); bad_eps.append(ep); continue
        if not (np.isfinite(S).all() and np.isfinite(A).all()):
            agg["nonfinite"] += 1; bad_eps.append(ep)
        if (np.abs(S) > spike).any() or (np.abs(A) > spike).any():
            agg["spike"] += 1; bad_eps.append(ep)
        fi = np.asarray(d["frame_index"])
        if not np.array_equal(fi, np.arange(n)):
            agg["bad_frame_index"] += 1; bad_eps.append(ep)
        ts = np.asarray(d["timestamp"], dtype=np.float64)
        if n > 1 and np.abs(np.diff(ts) - 1.0 / fps).max() > 1e-3:
            agg["bad_timestamp"] += 1; bad_eps.append(ep)
        if not (np.asarray(d["episode_index"]) == ep).all():
            errs.append(f"ep{ep} episode_index 列不一致"); bad_eps.append(ep)
        if not np.array_equal(np.asarray(d["index"]), np.arange(running_index, running_index + n)):
            agg["global_index_gap"] += 1
        running_index += n; sum_len += n
        for ck in CAM_KEYS:
            vp = os.path.join(root, vtmpl.format(episode_chunk=chunk, video_key=ck, episode_index=ep))
            if not os.path.isfile(vp):
                agg["missing_video"] += 1
    if sum_len != tf:
        errs.append(f"帧数合计 {sum_len} != info.total_frames {tf}")
    for k, v in agg.items():
        if v:
            errs.append(f"{k}: {v} 处")
    return dict(root=root, te=te, tf=tf, fps=fps, errs=errs, warns=warns, bad_eps=sorted(set(bad_eps))[:20])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=DEFAULT_ROOTS,
                    help="数据集根目录(含 meta/ data/);省略则体检默认两本体")
    ap.add_argument("--spike-threshold", type=float, default=10.0)
    args = ap.parse_args()
    roots = args.roots or DEFAULT_ROOTS

    any_err = False
    for root in roots:
        r = check(root, args.spike_threshold)
        name = os.path.basename(root.rstrip("/"))
        print(f"\n===== {name} =====")
        print(f"  episodes={r['te']}  frames={r['tf']}  fps={r['fps']}")
        if r["warns"]:
            print(f"  WARN: {r['warns']}")
        if r["errs"]:
            any_err = True
            print("  ❌ 问题:")
            for e in r["errs"]:
                print(f"     - {e}")
            if r["bad_eps"]:
                print(f"     异常 ep(前20): {r['bad_eps']}")
        else:
            print("  ✅ 通过(meta一致/帧索引连续/时间戳规整/无尖刺/有限/视频齐全/stats覆盖)")
    print("\nHEALTH_DONE")
    sys.exit(1 if any_err else 0)


if __name__ == "__main__":
    main()
