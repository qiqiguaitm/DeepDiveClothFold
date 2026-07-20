#!/usr/bin/env python3
"""
stitch_dagger_episodes.py — 把同一天的同 rollout_id 的 inference + dagger episodes
按时间序拼接为组合 episode，输出到 dagger 同日期目录的 chunk-001。

拼接逻辑:
  同 rollout_id 的 episodes 按 created_at 排序:
    INF(策略跑) → INF(卡住,terminal=intervention,ends_takeover_id=X)
      → DAG(takeover_id=X, 人类纠错) → INF(恢复继续) → ...

  每帧自动打 dagger_frame_class (Sirius 4 类 {demo,robot,intv,preintv} + 2 类双臂遥操静态伪影;
  完整语义/权重/文献见 docs/training/analysis/chunk001_schema.md):
    0 = robot           自主-正常     robot 自主, 正常执行             [keep,  Sirius robot]
    1 = intv_core       人控-纠错     人类遥操果断纠错核心             [keep,  Sirius intv, 上采样]
    2 = preintv         自主-临失败   机器人"停下点"之前 ℓ 帧的失败先兆 [keep+标记, Sirius preintv,
                                      (真运动, 非静止前奏; 见下)       P*=0 → 正训归零/AWBC 转负]
    3 = hesitation      起手迟疑      接管后低速遥操起手 (静态伪影)     [物理裁掉]
    4 = stationary_tail 静止尾        inf 静止前奏 / dag 末静止 (伪影)  [物理裁掉]
    5 = demo            纯示范        base 示范 (保留码位, 本脚本不产)  [keep]

  设计要点 (回应 review ①, 依据 Sirius RSS2023/IJRR2025 §加权BC + IWR Mandlekar2020):
  - class 不再是 intervention 冗余死列: 保留 preintv(2) 并标记 → 落盘 class = 真 3 分类 {0,1,2}。
  - **静态遥操伪影 (3,4) 仍物理裁掉** (双臂遥操录制伪影 + 接管前静止前奏, 非策略信号, 该删)。
  - **⚠️ 采集流程 ≠ Sirius**: 我们是"打断→机器人停住→人接管", 实测接管前末 15 帧 90% 静止。
    故 preintv **不能**取"接管前 ℓ 帧"(那是静止前奏伪影), 必须取**机器人停下点之前** ℓ 帧的
    真运动 (实测 arm 速度 0.0031 ≈ 正常)。静止前奏 → stationary_tail(4) 裁掉。详见 §3.1 of schema。
  - **ℓ 是时间尺度, 随 FPS 变**: Sirius ℓ=15 @20Hz=0.75s; 本 30 FPS → PREINTV_MARGIN=round(0.75*30)=22
    (照抄 15 只有 0.5s, 偏短)。AWBC 侧 discretize_advantage 已把 class∈{2,3,4} 排除出 positive → 天然一致。
  - 码位 0/1/2/3/4 与历史 chunk-001 兼容; 仅"保留 2 + 停下点切法 + ℓ 随帧率"改变 → 列变活且语义正确。

裁留/打标原则 (Sirius RSS 2023 / IJRR 2025; IWR Mandlekar 2020; ℓ=round(0.75*FPS)):
  ┌────────────────────────┬──────────┬────────────────────────────────┐
  │ 段                     │ class    │ 做法                            │
  ├────────────────────────┼──────────┼────────────────────────────────┤
  │ 犯错前 robot 段        │ 0 robot  │ 留                              │
  │ 停下点前 ℓ 帧 (真失败) │ 2 preintv│ 留+标记 (失败先兆; 正训归零/转负)│
  │ 停下点→接管 静止前奏   │ 4 statl  │ 物理裁掉 (静止, 非失败动态)      │
  │ 接管后犹豫低速段       │ 3 hesit  │ 物理裁掉 (静态遥操伪影)          │
  │ 摇操果断核心           │ 1 intv   │ 留, 上采样                      │
  │ dag 末静止段           │ 4 statl  │ 物理裁掉 (静态遥操伪影)          │
  │ 纠错后恢复→完成        │ 0 robot  │ 留                              │
  └────────────────────────┴──────────┴────────────────────────────────┘

输出:
  dagger/v4/<date>-v4/
    data/chunk-001/episode_NNNNNN.parquet   ← 拼接后的 parquet
    videos/chunk-001/observation.images.{cam}/episode_NNNNNN.mp4  ← 拼接后的视频
    meta/episodes_stitched.jsonl            ← 拼接 episode 的 metadata

用法:
  # Dry-run: 预览一个日期
  python stitch_dagger_episodes.py --date 2026-06-29 --dry-run

  # 执行一个日期
  python stitch_dagger_episodes.py --date 2026-06-29

  # 批量处理所有共同日期
  python stitch_dagger_episodes.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── 常量 ──
DATA_ROOT = Path(os.environ.get("KAI0_DATA_ROOT", "/data1/DATA_IMP/KAI0"))
FPS = 30
CAMERAS = ("top_head", "mid_head", "hand_left", "hand_right")
CAM_DIRS = {f"observation.images.{c}": c for c in CAMERAS}
ARM_DIMS = list(range(0, 6)) + list(range(7, 13))  # 12 arm dims
GRIP_DIMS = [6, 13]

# 裁留阈值 (基于实际数据分布调优)
# preintv 窗长是【时间尺度】, 随 FPS 变: Sirius ℓ=15 @20Hz = 0.75s 人反应时间;
# 本数据 30 FPS → 同样 0.75s 需 round(0.75*30)=22 帧 (照抄 15 只有 0.5s, 偏短 1.5×)。
REACTION_TIME_S = 0.75      # Sirius 人反应时间 (15帧@20Hz)
PREINTV_MARGIN = round(REACTION_TIME_S * FPS)   # =22 @30fps; preintv = 机器人"停下点"之前这么多帧
# 迟疑检测: velocity 连续超过此阈值 N 帧 → 迟疑结束
HESITATION_THR = 5e-3        # rad/frame — 低于此视为迟疑/慢速起手 (arm)
GRIP_HESITATION_THR = 0.01   # gripper 迟疑阈值 — 夹爪慢速 vs 果断抓放
HESITATION_WIN = 3           # 连续帧数
HESITATION_MAX = 30          # 最多检测这么多帧 (1s @30Hz)
# 静止检测
STATIONARY_THR = 3e-3        # rad/frame — 低于此视为已停止 (与 TRIM_THR 一致)
GRIP_STATIONARY_THR = 0.02   # gripper 静止阈值

# 帧类别 (Sirius {demo,robot,intv,preintv} + 2 类双臂遥操静态伪影; 码位与历史 chunk-001 兼容)
# 语义/权重/文献见 docs/training/analysis/chunk001_schema.md
CLASS_ROBOT = 0            # 自主-正常   robot 自主执行 (Sirius robot)
CLASS_INTV_CORE = 1        # 人控-纠错   人类遥操核心 (Sirius intv, 上采样)
CLASS_PREINTV = 2          # 自主-临失败 接管前 ℓ=15 帧, 从自主数据切出的失败先兆 (Sirius preintv)
CLASS_HESITATION = 3       # 起手迟疑    遥操低速起手静态伪影 → 物理裁掉
CLASS_STATIONARY_TAIL = 4  # 静止尾      episode 末静止伪影 → 物理裁掉
CLASS_DEMO = 5             # 纯示范      base demo (保留码位, 本脚本不产)

CLASS_NAMES = {0: "robot", 1: "intv_core", 2: "preintv",
               3: "hesitation", 4: "stationary_tail", 5: "demo"}
# 下游 loss 目标权重 (Sirius w=P*(c)/P(c) 精神): intv 上采样, preintv 正训归零(AWBC 侧转负),
# 静态伪影 mask (None). 供 dataloader/AWBC 参考, 本脚本不施加, 仅落盘 class 列。
CLASS_TRAIN_WEIGHT = {0: 1.0, 1: 2.0, 2: 0.0, 3: None, 4: None, 5: 1.0}

# 物理裁掉的 class = 纯静态遥操伪影 (preintv 保留并标记, 不裁 → 落盘 class 成为真多分类 {0,1,2})
TRIM_CLASSES = {CLASS_HESITATION, CLASS_STATIONARY_TAIL}


# ── helpers ──
def load_episodes_meta(subset: str, date: str) -> list[dict]:
    """读取 episodes.jsonl."""
    path = DATA_ROOT / "Task_A" / subset / "v4" / date / "meta" / "episodes.jsonl"
    if not path.exists():
        return []
    eps = []
    for line in open(path):
        d = json.loads(line)
        d["_subset"] = subset
        d["_date"] = date
        eps.append(d)
    return eps


def get_ep_id(ep: dict) -> int:
    """兼容不同 jsonl 的 episode_id/episode_index 字段."""
    return ep.get("episode_id", ep.get("episode_index", -1))


def read_parquet_states(pq_path: str) -> np.ndarray:
    """读取 parquet 的 observation.state 列, 返回 (N, 14) array."""
    t = pq.read_table(pq_path)
    n = t.num_rows
    states = np.array([t.column("observation.state")[i].as_py() for i in range(n)],
                      dtype=np.float64)
    return states


def compute_arm_velocity(states: np.ndarray) -> np.ndarray:
    """shape (N, 14) → (N,) arm velocity (mean |Δ| over 12 arm dims)."""
    if len(states) < 2:
        return np.zeros(len(states), dtype=np.float64)
    delta = np.abs(np.diff(states[:, ARM_DIMS], axis=0))
    vel = np.zeros(len(states), dtype=np.float64)
    vel[1:] = delta.mean(axis=1)
    return vel


def compute_gripper_velocity(states: np.ndarray) -> np.ndarray:
    """shape (N, 14) → (N,) gripper velocity (max |Δ| over 2 grip dims)."""
    if len(states) < 2:
        return np.zeros(len(states), dtype=np.float64)
    delta = np.abs(np.diff(states[:, GRIP_DIMS], axis=0))
    vel = np.zeros(len(states), dtype=np.float64)
    vel[1:] = delta.max(axis=1)
    return vel


def find_keep_indices(states: np.ndarray, seg_type: str,
                      next_seg_type: str = "") -> np.ndarray:
    """[DEPRECATED — 不再被 stitch 调用] 旧版物理裁剪索引。

    旧行为把 preintv 也裁掉 → 落盘 class 塌成 0/1 (review ① 的死列问题)。
    现在 stitch 改用 `classify_segment` 打真实 class + `TRIM_CLASSES={3,4}` 只裁静态伪影,
    preintv(2) 保留并标记。保留此函数仅为向后兼容 (可能有外部脚本 import)。

    - INF 段: 保留全部, 但如果下一段是 DAG 则裁掉尾部 preintv (最后 PREINTV_MARGIN 帧)
    - DAG 段: 裁掉首部 hesitation 和尾部 stationary.
      hesitation = 首段连续 arm_vel < HESITATION_THR **且** grip_vel < GRIP_HESITATION_THR
      stationary = 尾段连续 arm_vel < STATIONARY_THR **且** grip_vel < GRIP_STATIONARY_THR
      夹爪在动(抓/放)时不算 hesitation/stationary → 保留为 core

    Returns:
        int array of frame indices to keep (0-based within this segment)
    """
    n = len(states)
    arm_vel = compute_arm_velocity(states)
    grip_vel = compute_gripper_velocity(states)

    if seg_type == "inf":
        keep = np.arange(n)
        if next_seg_type == "dag" and n > PREINTV_MARGIN:
            keep = keep[:n - PREINTV_MARGIN]  # 裁掉最后 15 帧 preintv
        return keep

    # DAG 段: 裁首 hesitation + 尾 stationary
    # 首部 hesitation: 连续 arm 和 grip 都低于阈值
    hes_end = 0
    burst = 0
    for i in range(min(n, HESITATION_MAX)):
        arm_slow = arm_vel[i] < HESITATION_THR
        grip_slow = grip_vel[i] < GRIP_HESITATION_THR
        if not arm_slow or not grip_slow:
            burst += 1
            if burst >= HESITATION_WIN:
                hes_end = i - HESITATION_WIN + 1
                break
        else:
            burst = 0

    # 尾部 stationary: 从末尾向前, arm 和 grip 都低于阈值
    stat_start = n
    for i in range(n - 1, hes_end - 1, -1):
        arm_still = arm_vel[i] < STATIONARY_THR
        grip_still = grip_vel[i] < GRIP_STATIONARY_THR
        if arm_still and grip_still:
            stat_start = i
        else:
            break

    return np.arange(hes_end, stat_start)


def classify_segment(states: np.ndarray, seg_type: str,
                     next_seg_type: str = "") -> np.ndarray:
    """对一段 segment 的每一帧分配真实相位 class (stitch 的打标+裁剪都以此为准).

    返回 full-length int8 数组, 取值 {robot(0), intv_core(1), preintv(2), hesitation(3),
    stationary_tail(4)}。stitch 用它: (a) 落盘 dagger_frame_class 列; (b) keep = class∉TRIM_CLASSES。

    inf 段 (接管前): 采集流程是"打断→机器人停住→人接管", 实测接管前末 15 帧 90% 是静止
    (静止前奏中位 0.5s / p90 ~0.9s)。故 **不能** 直接把末 PREINTV_MARGIN 帧当 preintv —
    那落在静止前奏里, 是伪影不是失败动态。正确切法:
      · 找"停下点" = 最后一个 arm_vel>STATIONARY_THR 的帧;
      · 停下点【之后】(静止前奏) → stationary_tail(4), 物理裁掉;
      · 停下点【之前】PREINTV_MARGIN 帧 → preintv(2), 真失败先兆 (实测≈正常速);
      · 更早 → robot(0)。
    dag 段: 全 intv_core, 首部低速标 hesitation, 尾部静止标 stationary_tail。
    """
    n = len(states)
    arm_vel = compute_arm_velocity(states)
    grip_vel = compute_gripper_velocity(states)

    if seg_type == "inf":
        classes = np.full(n, CLASS_ROBOT, dtype=np.int8)
        if next_seg_type == "dag":
            moving = arm_vel > STATIONARY_THR
            if moving.any():
                stop = int(np.max(np.where(moving)[0]))   # 机器人停下点 (最后运动帧)
                classes[stop + 1:] = CLASS_STATIONARY_TAIL  # 静止前奏 → 裁 (TRIM_CLASSES)
                lo = max(0, stop + 1 - PREINTV_MARGIN)
                classes[lo:stop + 1] = CLASS_PREINTV        # 停下点前 ℓ 帧 → 失败先兆
            # moving 全 False (整段静止) → 全 robot, 交由后续无 preintv (退化但安全)
        return classes

    # DAG
    classes = np.full(n, CLASS_INTV_CORE, dtype=np.int8)
    # hesitation
    burst = 0
    hes_end = 0
    for i in range(min(n, HESITATION_MAX)):
        arm_slow = arm_vel[i] < HESITATION_THR
        grip_slow = grip_vel[i] < GRIP_HESITATION_THR
        if not arm_slow or not grip_slow:
            burst += 1
            if burst >= HESITATION_WIN:
                hes_end = i - HESITATION_WIN + 1
                break
        else:
            burst = 0
    if hes_end > 0:
        classes[:hes_end] = CLASS_HESITATION
    # stationary
    for i in range(n - 1, hes_end - 1, -1):
        if arm_vel[i] < STATIONARY_THR and grip_vel[i] < GRIP_STATIONARY_THR:
            classes[i] = CLASS_STATIONARY_TAIL
        else:
            break
    return classes


# ── 主逻辑 ──
def build_rollout_timeline(inf_eps: list[dict], dag_eps: list[dict]) -> list[dict]:
    """按 rollout_id 分组, 构建时间线.

    Returns:
        list of rollout dicts: {
            "rollout_id": int,
            "segments": [
                {"type": "inf", "ep": {...}, "states": ndarray, ...},
                {"type": "dag", "ep": {...}, "states": ndarray, ...},
            ]
        }
    """
    # DAG episode 按 (rollout_id, takeover_id) 索引
    dag_index: dict[tuple, dict] = {}
    for e in dag_eps:
        rid = e.get("rollout_id", -1)
        tid = e.get("takeover_id", -1)
        dag_index[(rid, tid)] = e

    # INF episode 按 rollout_id 分组, 排序
    inf_by_rollout = defaultdict(list)
    for e in inf_eps:
        rid = e.get("rollout_id", -1)
        inf_by_rollout[rid].append(e)

    # DAG episode 也按 rollout_id 分组
    dag_by_rollout = defaultdict(list)
    for e in dag_eps:
        rid = e.get("rollout_id", -1)
        dag_by_rollout[rid].append(e)

    all_rollout_ids = set(inf_by_rollout.keys()) | set(dag_by_rollout.keys())
    all_rollout_ids.discard(-1)

    rollouts = []
    for rid in sorted(all_rollout_ids):
        inf_list = sorted(inf_by_rollout[rid], key=lambda e: e.get("created_at", 0))
        dag_list = sorted(dag_by_rollout[rid], key=lambda e: e.get("created_at", 0))

        # 构建时间线: 每个 INF 的 ends_takeover_id → 在它后面插入对应 DAG
        segments = []
        added_dag_ids = set()
        for e in inf_list:
            term = e.get("terminal", "completed")
            segments.append({"type": "inf", "ep": e, "terminal": term})

            # 如果被 intervention 终止, 插入对应的 DAG (去重)
            if term == "intervention":
                etid = e.get("ends_takeover_id")
                if etid is not None:
                    dag_key = (rid, etid)
                    if dag_key in dag_index and etid not in added_dag_ids:
                        added_dag_ids.add(etid)
                        segments.append({"type": "dag", "ep": dag_index[dag_key],
                                         "takeover_id": etid})

        # 孤儿 DAG: 不在任何 intervention 之后
        for e in dag_list:
            tid = e.get("takeover_id", -1)
            if tid not in added_dag_ids and tid != -1:
                added_dag_ids.add(tid)
                segments.append({"type": "dag", "ep": e, "takeover_id": tid})

        # 按 created_at 排
        segments.sort(key=lambda s: s["ep"].get("created_at", 0))

        rollouts.append({"rollout_id": rid, "segments": segments})

    return rollouts


def get_parquet_path(subset: str, date: str, ep_id: int) -> Path:
    """获取原始 parquet 文件路径."""
    return DATA_ROOT / "Task_A" / subset / "v4" / date / "data" / "chunk-000" / f"episode_{ep_id:06d}.parquet"


def get_video_path(subset: str, date: str, cam_key: str, ep_id: int) -> Path:
    """获取原始视频文件路径."""
    cam_name = CAM_DIRS[cam_key]
    return (DATA_ROOT / "Task_A" / subset / "v4" / date / "videos" / "chunk-000" /
            cam_key / f"episode_{ep_id:06d}.mp4")


def _keep_ranges(keep_indices) -> list[tuple[int, int]]:
    """将 keep 帧索引压缩为连续区间 [(start, end), ...]."""
    if len(keep_indices) == 0:
        return []
    arr = np.asarray(keep_indices)
    ranges = []
    start = arr[0]
    end = start
    for i in range(1, len(arr)):
        if arr[i] == end + 1:
            end = arr[i]
        else:
            ranges.append((int(start), int(end)))
            start = arr[i]
            end = start
    ranges.append((int(start), int(end)))
    return ranges


def concat_videos(src_paths: list[Path], dst_path: Path) -> None:
    """使用 ffmpeg concat demuxer 无损拼接视频 (同编码无需重编码)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # 写 concat 列表文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in src_paths:
            f.write(f"file '{p}'\n")
        concat_file = f.name

    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", concat_file, "-c", "copy", str(dst_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # 如果 concat 失败 (某些 h264 参数不兼容), 回退到重编码
            print(f"  [WARN] concat demuxer 失败, 回退重编码: {result.stderr[:200]}")
            cmd2 = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_file, "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-pix_fmt", "yuv420p", str(dst_path)]
            subprocess.run(cmd2, check=True, capture_output=True)
    finally:
        os.unlink(concat_file)


def stitch_date(date: str, dry_run: bool = False) -> dict:
    """拼接一个日期的所有 rollout.

    Returns:
        dict with stats
    """
    dag_dir = DATA_ROOT / "Task_A" / "dagger" / "v4" / date / "data" / "chunk-001"
    vid_dir = DATA_ROOT / "Task_A" / "dagger" / "v4" / date / "videos" / "chunk-001"
    meta_path = DATA_ROOT / "Task_A" / "dagger" / "v4" / date / "meta" / "episodes_stitched.jsonl"

    # 加载 metadata
    inf_eps = load_episodes_meta("inference", date)
    dag_eps = load_episodes_meta("dagger", date)

    if not inf_eps or not dag_eps:
        print(f"[{date}] 缺少 inference 或 dagger 数据, 跳过")
        return {"date": date, "n_combined": 0, "n_frames": 0}

    # 构建 rollout 时间线
    rollouts = build_rollout_timeline(inf_eps, dag_eps)
    stitchable = [r for r in rollouts if
                  any(s["type"] == "dag" for s in r["segments"]) and
                  any(s["type"] == "inf" and s.get("terminal") == "intervention"
                      for s in r["segments"])]

    print(f"\n[{date}] {len(rollouts)} rollouts, {len(stitchable)} 可拼接")
    if dry_run:
        for r in stitchable[:3]:
            timeline = " → ".join(
                f"{s['type'].upper()}(ep={get_ep_id(s['ep'])}, len={s['ep']['length']})"
                for s in r["segments"])
            print(f"  rollout={r['rollout_id']}: {timeline}")
        if len(stitchable) > 3:
            print(f"  ... (还有 {len(stitchable)-3} 个)")
        return {"date": date, "n_combined": len(stitchable)}

    # ── 执行拼接 ──
    dag_dir.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        meta_path.unlink()

    combined_ep_idx = 0
    total_frames = 0
    all_meta = []
    class_counts = defaultdict(int)

    for r in stitchable:
        segments = r["segments"]
        if len(segments) < 2:
            continue

        # 收集所有段的 keep 帧
        # 先计算每个段要保留哪些帧, 然后从原始 parquet 取对应行, 从原始视频取对应帧范围
        seg_keep_info = []  # [(ep_id, subset, keep_indices, keep_count)]

        for i, seg in enumerate(segments):
            ep = seg["ep"]
            ep_id = get_ep_id(ep)
            subset = ep["_subset"]
            pq_path = get_parquet_path(subset, date, ep_id)
            if not pq_path.exists():
                print(f"  [WARN] {pq_path} 不存在, 跳过")
                continue

            states = read_parquet_states(str(pq_path))
            next_type = segments[i + 1]["type"] if i + 1 < len(segments) else ""

            # 逐帧真实 class (0/1/2/3/4), 再按 TRIM_CLASSES 决定删哪些帧。
            # 与旧 find_keep_indices 阈值同源 (classify_segment 共用 HESITATION/STATIONARY_THR),
            # 但差别在: preintv(2) 不再被裁, 落盘保留其真实 class → 列成为真多分类。
            classes_seg = classify_segment(states, seg["type"], next_seg_type=next_type)
            keep_idx = np.where(~np.isin(classes_seg, list(TRIM_CLASSES)))[0]
            seg_keep_info.append({
                "ep_id": ep_id,
                "subset": subset,
                "keep_idx": keep_idx,
                "classes": classes_seg,      # full-length 真实 class (供写列时按 keep_idx 取)
                "n_keep": len(keep_idx),
                "n_orig": len(states),
                "type": seg["type"],
            })

        if not seg_keep_info:
            continue

        # ── 拼接 parquet: 只取 keep_idx 对应的行 ──
        kept_tables = []
        n_total = 0
        for info in seg_keep_info:
            pq_path = get_parquet_path(info["subset"], date, info["ep_id"])
            t = pq.read_table(str(pq_path))
            keep_mask = np.zeros(t.num_rows, dtype=bool)
            keep_mask[info["keep_idx"]] = True
            kept_tables.append(t.filter(pa.array(keep_mask)))
            n_total += info["n_keep"]

        combined_table = pa.concat_tables(kept_tables)

        # 更新元数据列
        new_ep_idx = pa.array([combined_ep_idx] * n_total, type=pa.int64())
        new_frame_idx = pa.array(list(range(n_total)), type=pa.int64())
        new_index = pa.array(list(range(n_total)), type=pa.int64())
        new_ts = pa.array(np.arange(n_total, dtype=np.float32) / FPS, type=pa.float32())

        # 生成 dagger_frame_class: 落盘每帧【真实相位】(subselect 已裁掉 3/4, 保留 0/1/2)。
        # 不再塌成 inf→0/dag→1 — 那样会丢掉 preintv(2), 使列冗余于 intervention (review ①)。
        classes = np.concatenate(
            [info["classes"][info["keep_idx"]] for info in seg_keep_info]
        ).astype(np.int8)

        cols = {"episode_index": new_ep_idx, "frame_index": new_frame_idx,
                "index": new_index, "timestamp": new_ts}
        for col_name in combined_table.column_names:
            if col_name not in cols:
                cols[col_name] = combined_table.column(col_name)
        cols["dagger_frame_class"] = pa.array(classes, type=pa.int8())

        new_table = pa.table(cols)
        out_pq = dag_dir / f"episode_{combined_ep_idx:06d}.parquet"
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, str(out_pq))

        # ── 拼接视频: 从每个源视频提取 keep 帧范围, 再 concat ──
        for cam_key in CAM_DIRS:
            src_ranges = []  # [(src_video_path, keep_indices), ...]
            for info in seg_keep_info:
                vpath = get_video_path(info["subset"], date, cam_key, info["ep_id"])
                if vpath.exists():
                    src_ranges.append((vpath, info["keep_idx"]))

            if not src_ranges:
                continue

            # 用 ffmpeg 提取每段的 keep 帧到临时文件, 再 concat
            tmp_videos = []
            concat_list = []
            try:
                for vi, (vpath, keep_idx) in enumerate(src_ranges):
                    ranges = _keep_ranges(keep_idx)
                    # 构建 select 表达式
                    parts = [f"between(n,{s},{e})" for s, e in ranges]
                    select_expr = "+".join(parts)
                    # 写到 /tmp 避免被 tosutil sync 抓到未清理的临时文件
                    tmp_path = Path(tempfile.gettempdir()) / f"_stitch_tmp_{date}_{cam_key}_seg{vi}_ep{combined_ep_idx:06d}.mp4"
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)

                    # 如果表达式太长, 用 filter_script
                    if len(select_expr) > 30000:
                        script_file = str(tmp_path) + ".filter"
                        with open(script_file, "w") as f:
                            f.write(f"select='{select_expr}',setpts=N/FRAME_RATE/TB\n")
                        cmd = ["ffmpeg", "-y", "-i", str(vpath),
                               "-filter_script:v", script_file,
                               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                               "-pix_fmt", "yuv420p", "-an", str(tmp_path)]
                        result = subprocess.run(cmd, capture_output=True, timeout=600)
                        os.unlink(script_file)
                    else:
                        cmd = ["ffmpeg", "-y", "-i", str(vpath),
                               "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
                               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                               "-pix_fmt", "yuv420p", "-an", str(tmp_path)]
                        result = subprocess.run(cmd, capture_output=True, timeout=600)

                    if result.returncode != 0:
                        raise RuntimeError(f"ffmpeg select failed: {result.stderr[:300]}")

                    if tmp_path.exists():
                        tmp_videos.append(tmp_path)
                        concat_list.append(str(tmp_path))

                # concat 所有临时视频
                if concat_list:
                    dst_video = vid_dir / cam_key / f"episode_{combined_ep_idx:06d}.mp4"
                    dst_video.parent.mkdir(parents=True, exist_ok=True)  # 缺此→ffmpeg 打不开输出→exit 254
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as cf:
                        for p in concat_list:
                            cf.write(f"file '{p}'\n")
                        concat_file = cf.name
                    # 先试无损 -c copy; nvenc/libx264 段间 DTS/SPS 不一致会让 concat 报
                    # exit 254 → 回退重编码 (libx264, CPU, 不抢 GPU)。
                    _r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                                         "-i", concat_file, "-c", "copy", str(dst_video)],
                                        capture_output=True)
                    if _r.returncode != 0:
                        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                                        "-i", concat_file, "-c:v", "libx264", "-preset",
                                        "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                        str(dst_video)], check=True, capture_output=True)
                    os.unlink(concat_file)
            except Exception as e:
                print(f"  [WARN] 视频拼接失败 {cam_key}: {e}")
            finally:
                for p in tmp_videos:
                    p.unlink(missing_ok=True)

        # ── 写 meta ──
        duration = n_total / FPS
        # 找第一个和最后一个 segment 的 episode 信息
        first_ep = segments[0]["ep"]
        meta_rec = {
            "episode_id": combined_ep_idx,
            "length": n_total,
            "duration_s": round(duration, 3),
            "operator": "stitch",
            "prompt": first_ep.get("prompt", "Flatten and fold the cloth."),
            "template_id": "stitched_dagger",
            "success": segments[-1]["ep"].get("success", True),
            "note": f"stitched from rollout={r['rollout_id']}, "
                    f"{len(segments)} segments: "
                    + " → ".join(f"{s['type']}(ep={get_ep_id(s['ep'])})"
                                 for s in segments),
            "scene_tags": [],
            "created_at": segments[0]["ep"].get("created_at", 0),
            "stitch_rollout_id": r["rollout_id"],
            "stitch_segments": [
                {"type": s["type"], "episode_id": get_ep_id(s["ep"]),
                 "subset": s["ep"]["_subset"], "length": s["ep"]["length"]}
                for s in segments
            ],
        }
        all_meta.append(meta_rec)

        # 统计
        for c in range(6):
            cnt = int((classes == c).sum())
            if cnt > 0:
                class_counts[c] += cnt

        combined_ep_idx += 1
        total_frames += n_total
        print(f"  ep={combined_ep_idx - 1}: rollout={r['rollout_id']}, "
              f"{n_total} frames, {len(segments)} segments")

    # ── 写 meta ──
    if all_meta and not dry_run:
        with open(meta_path, "a") as f:
            for rec in all_meta:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── 汇总 ──
    print(f"\n[{date}] 完成: {len(all_meta)} 组合 episodes, {total_frames} 总帧")
    for c in sorted(class_counts.keys()):
        pct = class_counts[c] / max(1, total_frames) * 100
        print(f"  class {c} ({CLASS_NAMES[c]:20s}): {class_counts[c]:8d} ({pct:5.1f}%)")

    return {"date": date, "n_combined": len(all_meta), "n_frames": total_frames,
            "class_counts": dict(class_counts)}


def main():
    parser = argparse.ArgumentParser(description="拼接同 rollout 的 inference+dagger episodes")
    parser.add_argument("--date", type=str, default=None,
                        help="指定日期 (如 2026-06-29)")
    parser.add_argument("--all", action="store_true",
                        help="处理所有共同日期")
    parser.add_argument("--dry-run", action="store_true",
                        help="只预览, 不写文件")
    args = parser.parse_args()

    if not args.date and not args.all:
        parser.error("需要 --date 或 --all")

    # 找所有共同日期
    inf_dates = {d.name for d in (DATA_ROOT / "Task_A" / "inference" / "v4").glob("*-v4")}
    dag_dates = {d.name for d in (DATA_ROOT / "Task_A" / "dagger" / "v4").glob("*-v4")}
    common = sorted(inf_dates & dag_dates)

    if args.date:
        date = args.date
        if not date.endswith("-v4"):
            date = f"{date}-v4"
        if date not in common:
            print(f"[WARN] {date} 不在共同日期列表中: {common}")
        stitch_date(date, dry_run=args.dry_run)
    elif args.all:
        total_combined = 0
        for date in common:
            r = stitch_date(date, dry_run=args.dry_run)
            total_combined += r["n_combined"]
        print(f"\n{'='*60}")
        print(f"全部完成: {len(common)} 个日期, {total_combined} 组合 episodes")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
