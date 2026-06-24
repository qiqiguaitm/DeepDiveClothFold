#!/usr/bin/env python3
"""Compress backup depth (base/v1) zarr DirectoryStore -> FFV1 .mkv, lossless.

Same codec/path as build_v2_to_v4.trim_depth, but reads plain .zarr DIRS (not .zip)
and keeps ALL frames (backup: no trimming). Per episode: decode every frame from the
zarr dir, pipe raw gray16le to ffmpeg -c:v ffv1 -level 3, verify mkv frame count == N,
then delete the .zarr dir. Updates each date's info.json depth feature -> uint16_ffv1.

Run with kai0 venv python (needs numcodecs):
  /data1/tim/workspace/deepdive_kai0/kai0/.venv/bin/python compress_backup_depth.py [--workers N] [--dry]
"""
import argparse
import glob
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

BK = "/data2/visrobot_backup/datasets/KAI0/Task_A_backup/base/v1"
FPS = 30


def _convert(zdir: str) -> dict:
    """Decode .zarr dir -> FFV1 .mkv (lossless). Returns status; deletes zarr on success."""
    import shutil

    import numcodecs
    za = json.load(open(os.path.join(zdir, ".zarray")))
    N, H, W = za["shape"]
    dt = np.dtype(za["dtype"])
    codec = numcodecs.get_codec(za["compressor"])
    frames = []
    for i in range(N):
        cn = os.path.join(zdir, f"{i}.0.0")
        try:
            with open(cn, "rb") as f:
                fr = np.frombuffer(codec.decode(f.read()), dtype=dt).reshape(H, W)
        except FileNotFoundError:
            fr = np.zeros((H, W), dtype=dt)
        frames.append(np.ascontiguousarray(fr, dtype="<u2"))
    arr = np.stack(frames)
    dst = zdir[:-len(".zarr")] + ".mkv"
    tmp = dst + ".tmp.mkv"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray16le",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0", "-c:v", "ffv1", "-level", "3", tmp]
    p = subprocess.run(cmd, input=arr.tobytes(), capture_output=True)
    if p.returncode != 0:
        return {"zdir": zdir, "ok": False, "err": p.stderr.decode()[:200]}
    # verify frame count
    pr = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", tmp],
                        capture_output=True, text=True)
    try:
        got = int(pr.stdout.strip())
    except ValueError:
        got = -1
    if got != N:
        os.unlink(tmp)
        return {"zdir": zdir, "ok": False, "err": f"frame mismatch got={got} want={N}"}
    os.replace(tmp, dst)
    z_kb = sum(os.path.getsize(os.path.join(zdir, f)) for f in os.listdir(zdir)) // 1024
    m_kb = os.path.getsize(dst) // 1024
    shutil.rmtree(zdir)
    return {"zdir": zdir, "ok": True, "N": N, "z_kb": z_kb, "m_kb": m_kb}


def _update_info(date_dir: str):
    fp = os.path.join(date_dir, "meta", "info.json")
    if not os.path.exists(fp):
        return
    info = json.load(open(fp))
    if "depth_path" in info:
        info["depth_path"] = ("videos/chunk-{episode_chunk:03d}/{video_key}_depth/"
                              "episode_{episode_index:06d}.mkv")
    for k, feat in info.get("features", {}).items():
        if "depth" in k.lower():
            feat["dtype"] = "uint16_ffv1"
            feat["info"] = {"container": "matroska", "codec": "ffv1", "pix_fmt": "gray16le",
                            "unit": "millimeter", "depth.height": 480, "depth.width": 640, "depth.fps": FPS}
    if not os.path.exists(fp + ".bak_predepthffv1"):
        os.replace(fp, fp + ".bak_predepthffv1")
    json.dump(info, open(fp, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    zdirs = sorted(glob.glob(f"{BK}/*/videos/chunk-000/*_depth/episode_*.zarr"))
    print(f"zarr dirs to compress: {len(zdirs)}  (workers={args.workers})", flush=True)
    if args.dry:
        for z in zdirs[:5]:
            print("  ", z)
        return
    done = fail = 0
    z_tot = m_tot = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_convert, z): z for z in zdirs}
        for fu in as_completed(futs):
            r = fu.result()
            if r["ok"]:
                done += 1
                z_tot += r["z_kb"]
                m_tot += r["m_kb"]
                if done % 100 == 0:
                    print(f"  {done}/{len(zdirs)} ok  cum zarr={z_tot//1024}MB -> mkv={m_tot//1024}MB "
                          f"({100*m_tot//max(z_tot,1)}%)", flush=True)
            else:
                fail += 1
                print(f"  FAIL {r['zdir']}: {r.get('err')}", flush=True)
    print(f"convert done: ok={done} fail={fail}", flush=True)
    print(f"  total zarr={z_tot//1024//1024}GB -> mkv={m_tot//1024//1024}GB "
          f"(saved {(z_tot-m_tot)//1024//1024}GB, {100*m_tot//max(z_tot,1)}% of original)", flush=True)
    # update each date's info.json
    for d in sorted(glob.glob(f"{BK}/*/")):
        _update_info(d.rstrip("/"))
    print("info.json updated (depth -> uint16_ffv1)", flush=True)


if __name__ == "__main__":
    main()
