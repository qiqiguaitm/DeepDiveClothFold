#!/usr/bin/env python3
"""A_v4_base_dagger_launchtrim (Approach B: 切片复用现有 adv_est_v1 标签, 不重打标).
launchpoint-trim plan §2.2 主实验(任务②-analog). 单变量 vs 任务②(plus_freshdagger) = dagger clip 起爆点前裁.

⚠️ adv_est_v1 ckpt 已全网删除, 但 absolute_advantage 标签仍在 labeled 源里; 且 AE 的 absolute_advantage
   = V(f0,f_{n+int})−V(f0,f_n) 差分, progress(f0) 近似抵消 → 参考帧无关 → 裁剪后切片复用是忠实近似,
   也正是任务② 的做法(复用现成标签). 故直接从 labeled 源切片 advantage, 免 GPU 重打标 + 免 AE ckpt.

源(均带 absolute_advantage + group + src/src_ep):
- A_v4_base_dagger: group==base(1200, 整段保留) + group==dagger(806 老dagger 05-29~06-23, 前裁)
- A_v4_freshdagger_ft: group==dagger(506 fresh 06-29~07-03, 前裁)
构造: base 不裁; dagger 逐 clip launch_window(action 臂速, 排夹爪) 双向起爆点前裁 [s:e], 切所有列(含advantage);
      视频从 RAW vis_base/vis_dagger 源按 src/src_ep 重裁(select+PTS归零). 统一列 + 重排 ep + norm.
之后: discretize_advantage.py top-30% binary → task_index pos/neg. 训练 config=pi05_v4_awbc_launchtrim.

KAI0_ROOT 环境变量指定 kai0 根(默认 cnsh; gf3/North-E 上 export 为 North-E 路径).
Run: KAI0_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0 \
     .venv/bin/python train_scripts/kai/data/build_launchtrim_from_labeled.py [--dry-run] [--nproc 96]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_no_release import per_episode_stats, _select_job  # noqa: E402

ROOT = Path(os.environ.get("KAI0_ROOT", "/vePFS/tim/workspace/deepdive_kai0/kai0"))
TASK_A = ROOT / "data" / "Task_A"
BASE_DS = TASK_A / "self_built" / "A_v4_base_dagger"        # base(whole)+old-dagger(trim)
FRESH_DS = TASK_A / "self_built" / "A_v4_freshdagger_ft"    # fresh dagger(trim)
RAW_BASE = TASK_A / "vis_base" / "v4"
RAW_DAG = TASK_A / "vis_dagger" / "v4"
OUT = TASK_A / "self_built" / "A_v4_base_dagger_launchtrim"
CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
# Option A: 复用逐帧 task_index(pos/neg, 100% 完整), 不重 discretize → 与任务② 严格单变量.
# 丢 advantage 列(源里仅 ~85% ep 有 → schema 不一致会 CastError); task_index 才是训练用标签.
KEEP = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
THR, K, M, MIN_LEN, FPS, CHUNK = 0.02, 5, 2, 30, 30, 0


def launch_window(action):
    v = np.concatenate([[0.0], np.linalg.norm(np.diff(action[:, ARM], axis=0), axis=1)])
    vbar = np.convolve(v, np.ones(5) / 5, mode="same")
    above = vbar > THR
    if not above.any():
        return None
    t_start, run = None, 0
    for t in range(len(above)):
        run = run + 1 if above[t] else 0
        if run >= K:
            t_start = t - K + 1; break
    if t_start is None:
        t_start = int(np.argmax(above))
    t_end = int(np.where(above)[0][-1])
    a, b = max(0, t_start - M), min(len(action), t_end + M + 1)
    return (a, b) if b - a >= MIN_LEN else None


def raw_video(group, src, src_ep, cam):
    """RAW 源 mp4(base→vis_base, dagger→vis_dagger), 兼容裸相机名."""
    root = RAW_BASE if group == "base" else RAW_DAG
    for c in (cam, cam.replace("observation.images.", "")):
        p = root / src / "videos" / f"chunk-{CHUNK:03d}" / c / f"episode_{src_ep:06d}.mp4"
        if p.exists() or p.is_symlink():
            return p
    raise FileNotFoundError(f"raw video 缺 {group}/{src}/{cam}/ep{src_ep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--nproc", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="smoke: 只处理前 N base + 前 N dagger")
    a = ap.parse_args()

    # 收集 items: (labeled_ds, ep, src, src_ep, group, trim?)
    items = []
    for e in (json.loads(l) for l in (BASE_DS / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()):
        items.append((BASE_DS, e["episode_index"], e["src"], e["src_ep"], e["group"], e["group"] == "dagger"))
    fresh = [json.loads(l) for l in (FRESH_DS / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    for e in (x for x in fresh if x.get("group") == "dagger"):
        items.append((FRESH_DS, e["episode_index"], e["src"], e["src_ep"], "dagger", True))
    if a.limit:
        base = [x for x in items if x[4] == "base"][:a.limit]
        dag = [x for x in items if x[4] == "dagger"][:a.limit]
        items = base + dag
    nb = sum(1 for x in items if x[4] == "base"); nd = len(items) - nb
    print(f"源: base={nb}(整段) + dagger={nd}(前裁) = {len(items)}ep; KAI0_ROOT={ROOT}", flush=True)
    if a.dry_run:
        print("dry-run"); return

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / f"chunk-{CHUNK:03d}").mkdir(parents=True)
    (OUT / "meta").mkdir()

    eps_meta, stats_out, video_jobs = [], [], []
    total_frames, new_ep, dropped = 0, 0, 0
    for ds, ep, src, src_ep, grp, do_trim in items:
        df = pd.read_parquet(ds / "data" / f"chunk-{CHUNK:03d}" / f"episode_{ep:06d}.parquet")
        df = df[[c for c in KEEP if c in df.columns]].copy()
        n0 = len(df)
        if do_trim:
            w = launch_window(np.stack(df["action"].to_numpy()).astype(np.float64))
            if w is None:
                dropped += 1; continue
            s, e = w
            df = df.iloc[s:e].copy().reset_index(drop=True)
        else:
            s, e = 0, n0
        n = len(df)
        df["episode_index"] = np.int64(new_ep)
        df["index"] = np.arange(total_frames, total_frames + n, dtype=np.int64)
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["timestamp"] = (np.arange(n, dtype=np.float32) / FPS)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       OUT / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet")
        for cam in CAMERAS:
            sv = raw_video(grp, src, src_ep, cam)
            dv = OUT / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{new_ep:06d}.mp4"
            dv.parent.mkdir(parents=True, exist_ok=True)
            if do_trim:
                video_jobs.append((str(sv), str(dv), np.arange(s, e), e - s))
            else:
                os.symlink(str(sv.resolve()), dv)   # base 整段, 免重编码
        eps_meta.append({"episode_index": new_ep, "tasks": ["Flatten and fold the cloth."], "length": n,
                         "src": src, "src_ep": src_ep, "group": grp, "trimmed": bool(do_trim)})
        stats_out.append({"episode_index": new_ep, "stats": per_episode_stats(df)})
        total_frames += n; new_ep += 1
    print(f"写 {new_ep}ep / {total_frames}帧 (dagger 丢弃 {dropped}); 并行重编码 {len(video_jobs)} dagger 视频 (nproc={a.nproc})", flush=True)
    with Pool(a.nproc) as pool:
        for i, _ in enumerate(pool.imap_unordered(_select_job, video_jobs, chunksize=4), 1):
            if i % 1000 == 0:
                print(f"  视频 {i}/{len(video_jobs)}", flush=True)

    info = json.loads((RAW_BASE / eps_meta[0]["src"] / "meta" / "info.json").read_text()) \
        if (RAW_BASE / eps_meta[0]["src"] / "meta" / "info.json").exists() \
        else json.loads((BASE_DS / "meta" / "info.json").read_text())
    info.update({"total_episodes": new_ep, "total_frames": total_frames, "total_tasks": 1,
                 "total_videos": new_ep * len(CAMERAS), "total_chunks": 1,
                 "chunks_size": max(1000, new_ep), "splits": {"train": f"0:{new_ep}"}})
    for k in ("observation.depth.top_head", "intervention"):
        info.get("features", {}).pop(k, None)
    (OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (OUT / "meta" / "episodes.jsonl").open("w") as f:
        for em in eps_meta:
            f.write(json.dumps(em) + "\n")
    with (OUT / "meta" / "episodes_stats.jsonl").open("w") as f:
        for st in stats_out:
            f.write(json.dumps(st) + "\n")
    # tasks.jsonl: 复用源的 pos/neg 映射(Option A 复用逐帧 task_index, 0=neg/1=pos)
    shutil.copy(BASE_DS / "meta" / "tasks.jsonl", OUT / "meta" / "tasks.jsonl")
    ntasks = sum(1 for _ in (OUT / "meta" / "tasks.jsonl").open())
    info["total_tasks"] = ntasks
    (OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"  -> {OUT} ({new_ep}ep/{total_frames}帧, tasks.jsonl {ntasks}条 pos/neg)", flush=True)

    if not a.no_norm:
        from norm_stats_from_dataset import compute_norm_stats
        print("  computing norm_stats (action_dim=32)...", flush=True)
        compute_norm_stats(str(OUT), action_dim=32)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
