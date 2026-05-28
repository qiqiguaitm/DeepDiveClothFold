"""
Render per-episode videos with progress curve overlay.

Layout per frame:
  ┌──────────────┬──────────────┬──────────────┐  ← 3 cameras (320×240 each)
  │  top_head    │  hand_left   │  hand_right  │
  └──────────────┴──────────────┴──────────────┘
  ┌─────────────────────────────────────────────┐  ← progress panel (960×200)
  │  absolute_value curve + current frame mark  │
  │  relative_advantage curve                   │
  └─────────────────────────────────────────────┘

Uses multiprocessing: one worker per chunk of episodes, assigned to different CPU cores.
No GPU needed (pure video rendering).

Usage (from kai0/):
    uv run python stage_advantage/render_progress_video.py [--out eval_adv_est_out] [--workers 8]
"""

import argparse
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/vePFS/tim/workspace/lerobot")

CHUNKS_SIZE = 1000
DATA_ROOT: Path = None  # type: ignore  — set at runtime via --data

# Output video dimensions
CAM_W, CAM_H   = 320, 240        # each camera cell
PANEL_H        = 200              # progress panel height
OUT_W          = CAM_W * 3       # 960
OUT_H          = CAM_H + PANEL_H # 440
FPS            = 30.0

# Colors (BGR)
POS_COLOR_BGR  = (172, 102, 33)  # blue
NEG_COLOR_BGR  = (77,  96, 214)  # red
GRAY           = (160, 160, 160)
WHITE          = (255, 255, 255)
BLACK          = (0,   0,   0)
DARK_BG        = (30,  30,  30)
GREEN          = (80,  200, 80)
YELLOW         = (0,   220, 220)


def video_paths(ep_idx: int):
    chunk = ep_idx // CHUNKS_SIZE
    base = DATA_ROOT / f"videos/chunk-{chunk:03d}"
    return (
        base / f"observation.images.top_head/episode_{ep_idx:06d}.mp4",
        base / f"observation.images.hand_left/episode_{ep_idx:06d}.mp4",
        base / f"observation.images.hand_right/episode_{ep_idx:06d}.mp4",
    )


def build_curve_lut(results: list, total_frames: int):
    """
    Build per-frame lookup arrays (absolute_value, relative_advantage)
    by linear interpolation from sampled inference results.
    Returns two float32 arrays of length total_frames.
    """
    if not results:
        return np.zeros(total_frames, np.float32), np.zeros(total_frames, np.float32)

    src_fi  = np.array([r["frame_idx"]         for r in results], dtype=np.float32)
    src_av  = np.array([r["absolute_value"]     for r in results], dtype=np.float32)
    src_ra  = np.array([r["relative_advantage"] for r in results], dtype=np.float32)

    all_fi = np.arange(total_frames, dtype=np.float32)
    av_lut = np.interp(all_fi, src_fi, src_av).astype(np.float32)
    ra_lut = np.interp(all_fi, src_fi, src_ra).astype(np.float32)
    return av_lut, ra_lut


def draw_progress_panel(av_lut, ra_lut, cur_frame, total_frames, label, color_bgr):
    """
    Draw the progress panel (OUT_W × PANEL_H, uint8 BGR).
    Shows the full curve in dim color, history in bright color, current frame marker.
    """
    panel = np.full((PANEL_H, OUT_W, 3), 28, dtype=np.uint8)  # near-black bg

    pad_l, pad_r = 60, 20
    pad_t, pad_b = 20, 40
    plot_w = OUT_W  - pad_l - pad_r
    plot_h = PANEL_H - pad_t - pad_b

    def val_to_y(v):
        # v in [-1, 1] → pixel row (top=pad_t, bottom=pad_t+plot_h)
        return int(pad_t + (1.0 - (v + 1.0) / 2.0) * plot_h)

    def fi_to_x(fi):
        return int(pad_l + fi / max(total_frames - 1, 1) * plot_w)

    # Grid lines
    for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        y = val_to_y(v)
        cv2.line(panel, (pad_l, y), (pad_l + plot_w, y),
                 (60, 60, 60) if v != 0 else (90, 90, 90), 1)
        label_txt = f"{v:+.1f}"
        cv2.putText(panel, label_txt, (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, (140, 140, 140), 1, cv2.LINE_AA)

    # Full curves in dim color
    dim_av = tuple(max(0, c // 3) for c in color_bgr)
    dim_ra = (50, 50, 50)
    pts_av_all = np.array([[fi_to_x(i), val_to_y(float(av_lut[i]))]
                           for i in range(total_frames)], dtype=np.int32)
    pts_ra_all = np.array([[fi_to_x(i), val_to_y(float(ra_lut[i]))]
                           for i in range(total_frames)], dtype=np.int32)
    cv2.polylines(panel, [pts_av_all], False, dim_av, 1, cv2.LINE_AA)
    cv2.polylines(panel, [pts_ra_all], False, dim_ra, 1, cv2.LINE_AA)

    # History up to cur_frame in bright color
    if cur_frame > 0:
        pts_av_hist = pts_av_all[:cur_frame + 1]
        pts_ra_hist = pts_ra_all[:cur_frame + 1]
        cv2.polylines(panel, [pts_av_hist], False, color_bgr, 2, cv2.LINE_AA)
        cv2.polylines(panel, [pts_ra_hist], False, (100, 180, 100), 1, cv2.LINE_AA)

    # Current frame vertical line
    cx = fi_to_x(cur_frame)
    cv2.line(panel, (cx, pad_t), (cx, pad_t + plot_h), (255, 255, 255), 1, cv2.LINE_AA)

    # Current value dot
    cy_av = val_to_y(float(av_lut[cur_frame]))
    cv2.circle(panel, (cx, cy_av), 4, color_bgr, -1, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy_av), 4, WHITE, 1, cv2.LINE_AA)

    # Text annotations
    av_val = float(av_lut[cur_frame])
    ra_val = float(ra_lut[cur_frame])
    progress_pct = (av_val + 1.0) / 2.0 * 100.0  # map [-1,1] to [0,100]%

    cv2.putText(panel,
                f"[{label}]  frame {cur_frame}/{total_frames-1}  "
                f"abs_val={av_val:+.3f}  progress~{progress_pct:.0f}%  "
                f"rel_adv={ra_val:+.3f}",
                (pad_l + 4, PANEL_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

    # Axis label
    cv2.putText(panel, "abs_val", (pad_l + 4, pad_t + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_bgr, 1, cv2.LINE_AA)
    cv2.putText(panel, "rel_adv", (pad_l + 60, pad_t + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 180, 100), 1, cv2.LINE_AA)

    return panel


def render_episode(ep_idx: int, results: list, label: str, out_dir: Path):
    """Render one episode video with progress overlay."""
    top_p, left_p, right_p = video_paths(ep_idx)
    if not top_p.exists():
        print(f"  [skip] ep {ep_idx}: missing video")
        return

    color_bgr = POS_COLOR_BGR if label == "POS" else NEG_COLOR_BGR

    # Open all three video captures
    caps = [cv2.VideoCapture(str(p)) for p in (top_p, left_p, right_p)]
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    fps = caps[0].get(cv2.CAP_PROP_FPS) or FPS

    out_path = out_dir / f"ep_{ep_idx:06d}_{label}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (OUT_W, OUT_H))

    # Pre-build interpolated curve lookup
    av_lut, ra_lut = build_curve_lut(results, total_frames)

    for fi in range(total_frames):
        frames = []
        ok = True
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                ok = False
                break
            frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        if not ok:
            break

        # Camera row
        cam_row = np.concatenate(frames, axis=1)  # (CAM_H, OUT_W, 3)

        # Label each camera
        for i, cam_name in enumerate(["top_head", "hand_left", "hand_right"]):
            ox = i * CAM_W + 4
            cv2.putText(cam_row, cam_name, (ox, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)
            cv2.putText(cam_row, cam_name, (ox, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, BLACK, 1, cv2.LINE_AA)

        # Camera dividers
        for x in [CAM_W, CAM_W * 2]:
            cv2.line(cam_row, (x, 0), (x, CAM_H), (60, 60, 60), 1)

        # Progress panel
        panel = draw_progress_panel(av_lut, ra_lut, fi, total_frames, label, color_bgr)

        # Stack vertically
        out_frame = np.concatenate([cam_row, panel], axis=0)  # (OUT_H, OUT_W, 3)
        writer.write(out_frame)

    for cap in caps:
        cap.release()
    writer.release()
    return out_path


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------

def worker(shard: list, out_dir: Path, data_root: str):
    """Process a list of (ep_idx, results, label) tuples."""
    global DATA_ROOT
    DATA_ROOT = Path(data_root)
    from tqdm import tqdm
    pid = os.getpid()
    for ep_idx, results, label in tqdm(shard, desc=f"[pid {pid}]"):
        try:
            p = render_episode(ep_idx, results, label, out_dir)
            if p:
                print(f"  [done] ep {ep_idx} {label} → {p.name}", flush=True)
        except Exception as e:
            print(f"  [error] ep {ep_idx}: {e}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="eval_adv_est_out")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n", type=int, default=100, help="Max episodes to render")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data", type=str,
        default="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_advantage",
        help="Dataset root (must contain videos/)",
    )
    args = parser.parse_args()

    global DATA_ROOT
    DATA_ROOT = Path(args.data)

    out_dir = ROOT / args.out
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    # Load inference results — support both flat {ep: results} and legacy {"positive":…,"negative":…}
    pkl = out_dir / "inference_results.pkl"
    with open(pkl, "rb") as f:
        data = pickle.load(f)

    if "positive" in data and "negative" in data:
        # Legacy format: assign POS/NEG labels
        all_episodes = (
            [(ep, res, "POS") for ep, res in data["positive"].items()] +
            [(ep, res, "NEG") for ep, res in data["negative"].items()]
        )
    else:
        # Flat format: no per-episode label
        all_episodes = [(ep, res, "EP") for ep, res in data.items()]

    # Sample --n episodes
    import random
    rng = random.Random(args.seed)
    if len(all_episodes) > args.n:
        all_episodes = rng.sample(all_episodes, args.n)
    all_episodes.sort(key=lambda x: x[0])

    # Skip already-rendered
    all_work = []
    for ep_idx, results, label in all_episodes:
        p = vid_dir / f"ep_{ep_idx:06d}_{label}.mp4"
        if not p.exists():
            all_work.append((ep_idx, results, label))

    total = len(all_work)
    print(f"[render] {total} episodes to render → {vid_dir}/")
    print(f"[render] data={DATA_ROOT}")
    print(f"[render] workers={args.workers}")

    if total == 0:
        print("[render] all episodes already rendered.")
        return

    # Distribute round-robin
    n = args.workers
    shards = [[] for _ in range(n)]
    for i, item in enumerate(all_work):
        shards[i % n].append(item)

    mp.set_start_method("spawn", force=True)
    processes = []
    for shard in shards:
        if not shard:
            continue
        p = mp.Process(target=worker, args=(shard, vid_dir, str(DATA_ROOT)), daemon=True)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    done = len(list(vid_dir.glob("*.mp4")))
    print(f"[done] {done} videos in {vid_dir}/")


if __name__ == "__main__":
    main()
