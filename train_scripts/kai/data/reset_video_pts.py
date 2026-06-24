#!/usr/bin/env python3
"""Reset video PTS to start at 0 (packet-level remux, NO re-encode) for a lerobot videos/ dir.

Why: v3 front-trim (build_no_release.trim_video_pyav) dropped leading frames but kept their
original PTS, so trimmed videos start at PTS≈cut/fps (e.g. 0.7s) while the parquet `timestamp`
column resets to 0. lerobot's AdvantageLerobotDataset decodes video BY TIMESTAMP (tolerance 1e-4s)
→ query 0.0 vs nearest frame 0.7s → "timestamps violate tolerance". Regular openpi pi05 loads
by frame-index so never hit it; the advantage estimator does.

Fix: rewrite each video's packet PTS/DTS to start at 0 (demux→mux, copy codec). Replaces symlinks
in a tim-owned videos/ dir with real PTS-reset files (isolated; does NOT touch shared v3 sources).
"""
from __future__ import annotations
import argparse, os, sys, tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import av


def remux_reset(src_mp4: str, dst_mp4: str):
    inp = av.open(src_mp4)
    ins = inp.streams.video[0]
    out = av.open(dst_mp4, "w")
    outs = out.add_stream(template=ins)
    first = None
    for pkt in inp.demux(ins):
        if pkt.pts is None or pkt.dts is None:
            continue
        if first is None:
            first = pkt.pts
        pkt.pts -= first
        pkt.dts -= first
        pkt.stream = outs
        out.mux(pkt)
    out.close(); inp.close()


def _job(path: str):
    """Replace a symlink-or-file video with a PTS-reset real file (atomic via temp+rename)."""
    p = Path(path)
    target = os.path.realpath(path)  # follow symlink to the real source video
    tmp = tempfile.mktemp(suffix=".mp4", dir=str(p.parent))
    try:
        remux_reset(target, tmp)
        # verify first frame ~0
        c = av.open(tmp); s = c.streams.video[0]
        f0 = next(c.decode(video=0)); t0 = float(f0.pts * s.time_base); c.close()
        if t0 > 0.01:
            os.unlink(tmp); return (path, f"BAD_PTS {t0:.3f}")
        if p.is_symlink() or p.exists():
            p.unlink()
        os.rename(tmp, path)
        return (path, "ok")
    except Exception as e:
        if os.path.exists(tmp):
            os.unlink(tmp)
        return (path, f"ERR {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos_dir", help="lerobot videos/ dir (chunk-000/<cam>/episode_*.mp4)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    vids = sorted(str(p) for p in Path(a.videos_dir).rglob("*.mp4"))
    print(f"reset PTS on {len(vids)} videos ({a.workers} workers)", flush=True)
    ok = bad = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_job, v): v for v in vids}
        for i, fut in enumerate(as_completed(futs)):
            path, st = fut.result()
            if st == "ok":
                ok += 1
            else:
                bad += 1
                print(f"  ⚠️ {path}: {st}", flush=True)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(vids)} (ok={ok} bad={bad})", flush=True)
    print(f"DONE ok={ok} bad={bad}/{len(vids)}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
