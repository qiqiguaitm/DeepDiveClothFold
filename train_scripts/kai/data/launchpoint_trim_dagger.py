#!/usr/bin/env python3
"""dagger clip 双向"起爆点"前裁 (dagger_launchpoint_trim_freeze_fix_plan §1).
逐 dagger clip: 前砍迟疑起手 + 后砍静止收尾, 只留果断动作核心. base 不裁.
臂速排除夹爪(idx 6,13). 视频同步裁 + PTS 归零(复用 build_no_release.select_video_pyav).

--dry-run: 只算+打印各日期裁前后统计(人工核验 THR/K/M), 不写文件.
默认: 裁全 dagger → vis_dagger/v4_launchtrim/<date>/ (parquet + 视频 + meta 重排).
Run: kai0/.venv/bin/python train_scripts/kai/data/launchpoint_trim_dagger.py [--dry-run] [--dates d1,d2]
"""
import argparse, json, shutil, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A")
SRC = ROOT / "vis_dagger" / "v4"
DST = ROOT / "vis_dagger" / "v4_launchtrim"
CAMS = ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]   # 12 臂关节, 排除夹爪 6/13
THR, K, M, MIN_LEN = 0.02, 5, 2, 30
NPROC = 48   # 视频重编码并行度(本机 56 核), 瓶颈=逐帧解 AV1 源

sys.path.insert(0, str(Path(__file__).parent))
from build_no_release import _select_job   # (src,dst,keep_idx,new_len) 选帧重编码+PTS归零, 多进程安全

def launch_window(action: np.ndarray):
    """返回 (t_start, t_end) keep 区间(含 margin), 或 None(丢弃: 纯hold / 裁后过短)."""
    v = np.linalg.norm(np.diff(action[:, ARM], axis=0), axis=1)   # 臂速 (n-1,)
    v = np.concatenate([[0.0], v])                                # 对齐 (n,)
    k = 5
    vbar = np.convolve(v, np.ones(k) / k, mode="same")            # 5帧滑动均值
    above = vbar > THR
    if not above.any():
        return None                                               # 纯 hold → 丢
    # 前起爆: 首个连续 >=K 帧 above 的起点
    t_start = None
    run = 0
    for t in range(len(above)):
        run = run + 1 if above[t] else 0
        if run >= K:
            t_start = t - K + 1; break
    if t_start is None:
        t_start = int(np.argmax(above))                           # 兜底: 首个 above
    t_end = int(np.where(above)[0][-1])                           # 后起爆: 末个 above
    a = max(0, t_start - M); b = min(len(action), t_end + M + 1)
    if b - a < MIN_LEN:
        return None
    return a, b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dates", type=str, default="")
    a = ap.parse_args()
    dates = a.dates.split(",") if a.dates else sorted(d.name for d in SRC.iterdir() if d.name.startswith("2026"))
    tot_in = tot_out = tot_clip = tot_drop = 0
    front_cuts, tail_cuts, kept_lens = [], [], []
    video_jobs = []   # (src_mp4, dst_mp4, keep_idx, new_len) 攒齐后并行编码
    for date in dates:
        sd = SRC / date
        parts = sorted(sd.glob("data/**/*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
        if not parts:
            print(f"{date}: 无 parquet, 跳过"); continue
        if not a.dry_run:
            dd = DST / date
            if dd.exists(): shutil.rmtree(dd)
            (dd / "data" / "chunk-000").mkdir(parents=True)
            if (sd / "meta").exists(): shutil.copytree(sd / "meta", dd / "meta")
        d_in = d_out = d_clip = d_drop = 0
        new_ep = 0
        for p in parts:
            df = pq.read_table(p).to_pandas()
            n = len(df); d_in += n; d_clip += 1
            action = np.stack(df["action"].to_numpy()).astype(np.float64)
            w = launch_window(action)
            if w is None:
                d_drop += 1; tot_drop += 1; continue
            s, e = w
            front_cuts.append(s); tail_cuts.append(n - e); kept_lens.append(e - s)
            d_out += (e - s)
            if a.dry_run:
                continue
            # 裁 parquet(串行, 快)
            sub = df.iloc[s:e].copy().reset_index(drop=True)
            sub["frame_index"] = np.arange(len(sub), dtype=np.int64)
            sub["timestamp"] = (np.arange(len(sub)) / 30.0).astype(np.float32)
            sub["episode_index"] = np.int64(new_ep)
            ep_old = int(p.stem.split("_")[1])
            outp = DST / date / "data" / "chunk-000" / f"episode_{new_ep:06d}.parquet"
            pq.write_table(pa.Table.from_pandas(sub, preserve_index=False), outp)
            # 视频重编码入队(稍后并行, [s:e] 选帧 + PTS归零)
            for cam in CAMS:
                sv = sd / "videos" / f"chunk-000" / cam / f"episode_{ep_old:06d}.mp4"
                dv = DST / date / "videos" / "chunk-000" / cam / f"episode_{new_ep:06d}.mp4"
                dv.parent.mkdir(parents=True, exist_ok=True)
                video_jobs.append((str(sv), str(dv), np.arange(s, e), e - s))
            new_ep += 1
        tot_in += d_in; tot_out += d_out; tot_clip += d_clip
        print(f"{date}: clips {d_clip} (drop {d_drop}) | frames {d_in}→{d_out} ({100*d_out/max(d_in,1):.0f}%保留)", flush=True)
    # 并行重编码所有视频(瓶颈步)
    if video_jobs and not a.dry_run:
        print(f"\n并行重编码 {len(video_jobs)} 视频 (NPROC={NPROC})...", flush=True)
        with Pool(NPROC) as pool:
            for i, _ in enumerate(pool.imap_unordered(_select_job, video_jobs, chunksize=4), 1):
                if i % 500 == 0:
                    print(f"  视频 {i}/{len(video_jobs)}", flush=True)
        print(f"  视频重编码完成 {len(video_jobs)}", flush=True)
    print(f"\n=== 总计: {tot_clip} clips, drop {tot_drop} | frames {tot_in}→{tot_out} ({100*tot_out/max(tot_in,1):.0f}%保留) ===")
    if front_cuts:
        print(f"平均前裁 {np.mean(front_cuts):.0f}帧({np.mean(front_cuts)/30:.2f}s) | 平均后裁 {np.mean(tail_cuts):.0f}帧({np.mean(tail_cuts)/30:.2f}s) | 平均保留 {np.mean(kept_lens):.0f}帧({np.mean(kept_lens)/30:.1f}s)")
        print(f"前裁分位 p10/50/90: {np.percentile(front_cuts,[10,50,90]).round(0)} | 保留长度 p10/50/90: {np.percentile(kept_lens,[10,50,90]).round(0)}")
    print("DRY_RUN_DONE" if a.dry_run else "TRIM_DONE")

if __name__ == "__main__":
    main()
