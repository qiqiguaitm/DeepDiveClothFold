"""Merge per-window side-by-side clips into one continuous video per (model,episode,camera,horizon).

Input : report_assets/<model>__<ep>__<cam>__<hz>__a<anchor>__sbs.mp4   (GT|pred, 480x1286)
Output: report_assets/merged/<model>__<ep>__<cam>__<hz>__merged.mp4     (anchors concatenated in time order)
"""
import glob, os, re, argparse
import numpy as np
import imageio.v3 as iio

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "report_assets")
OUT = os.path.join(ASSETS, "merged")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")   # restrict to one model tag (optional)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pat = f"{args.model}__*__sbs.mp4" if args.model else "*__sbs.mp4"
    files = glob.glob(os.path.join(ASSETS, pat))
    groups = {}
    for f in files:
        b = os.path.basename(f)[:-len("__sbs.mp4")]
        parts = b.split("__")            # tag, ep, cam, hz, a<anchor>
        if len(parts) != 5: continue
        tag, ep, cam, hz, a = parts
        anchor = int(a[1:])
        groups.setdefault((tag, ep, cam, hz), []).append((anchor, f))
    print(f"{len(files)} window clips -> {len(groups)} merged videos", flush=True)
    for (tag, ep, cam, hz), lst in sorted(groups.items()):
        lst.sort()
        frames = []
        for _, f in lst:
            try:
                v = iio.imread(f, plugin="pyav")   # [T,H,W,C]
                frames.append(np.asarray(v))
            except Exception as e:
                print(f"  skip {f}: {e}")
        if not frames: continue
        merged = np.concatenate(frames, axis=0)
        outp = os.path.join(OUT, f"{tag}__{ep}__{cam}__{hz}__merged.mp4")
        iio.imwrite(outp, merged, fps=24, codec="libx264")
    print(f"done -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
