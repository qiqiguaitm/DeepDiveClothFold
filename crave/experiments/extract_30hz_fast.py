#!/usr/bin/env python
"""Fast 30Hz feature extraction with parallel video decoding.
8 CPU workers decode+crop224, main thread GPU-encodes in batches.
Resumable, skip existing.
"""
import argparse, time, av, cv2, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from crave.encoders import load_encoder
from crave.config import resolve_dataset
from crave.data import kai0

REPO = Path("/home/tim/workspace/deepdive_kai0"); OUT = REPO / "temp/crave_30hz_feat_v2"; OUT.mkdir(exist_ok=True, parents=True)

def crop224(rgb):
    h, w = rgb.shape[:2]; s = 224 / min(h, w)
    r = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
    hh, ww = r.shape[:2]
    return np.ascontiguousarray(r[(hh - 224) // 2:(hh + 224) // 2, (ww - 224) // 2:(ww + 224) // 2])

def decode_one(args):
    """Worker: decode one episode, return (ep, frames_rgb_np) or None on error."""
    e, vid_path = args
    try:
        cap = av.open(vid_path); frames = [crop224(f.to_ndarray(format="rgb24")) for f in cap.decode(video=0)]
        cap.close()
        if len(frames) < 3: return None
        return (e, frames)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--decode-workers", type=int, default=8)
    a = ap.parse_args()
    cfg = resolve_dataset("kai0_base"); cs = kai0.chunks_size(cfg.root); DS = Path(cfg.root)
    all_eps = sorted(int(p.stem.split("_")[1]) for p in (DS / "data").glob("chunk-*/episode_*.parquet"))
    mine = [(e, str(DS / f"videos/chunk-{e // cs:03d}/observation.images.top_head/episode_{e:06d}.mp4"))
            for i, e in enumerate(all_eps) if i % a.world == a.rank and not (OUT / f"ep{e}.npy").exists()]
    print(f"[rank{a.rank}/{a.world}] {len(mine)}/{len(all_eps)} eps to do (decode_workers={a.decode_workers})", flush=True)
    enc = load_encoder("dinov3-h")
    t0 = time.time(); done = 0; errs = 0
    # Accumulate frames from multiple eps for batched GPU encode
    BATCH_EPS = 4; buf_eps = []; buf_frames = []
    def flush_buf():
        nonlocal done
        if not buf_frames: return
        all_feats = enc.encode_pooled(sum(buf_frames, []))  # flatten list of lists
        pos = 0
        for e, frs in zip(buf_eps, buf_frames):
            n = len(frs); feat = all_feats[pos:pos + n].astype(np.float16)
            np.save(OUT / f"ep{e}.npy", feat); pos += n; done += 1
        buf_eps.clear(); buf_frames.clear()
    with ProcessPoolExecutor(max_workers=a.decode_workers) as ex:
        for result in ex.map(decode_one, mine):
            if result is None:
                errs += 1; continue
            e, frames = result
            buf_eps.append(e); buf_frames.append(frames)
            if len(buf_eps) >= BATCH_EPS:
                flush_buf()
                if done % 50 == 0:
                    el = time.time() - t0; eta = el / done * (len(mine) - done) / 60
                    print(f"  [{done}/{len(mine)}] ep{e} n={len(frames)}  ({el/60:.0f}min, ~{eta:.0f}min left)", flush=True)
    flush_buf()
    print(f"[rank{a.rank}] DONE {done} eps, {errs} errors in {(time.time()-t0)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
