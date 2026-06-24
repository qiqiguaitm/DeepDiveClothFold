"""逐帧全部验证 ep2302 视频:对 mp4 每一帧核对 相机帧 / 解码质心缩略图 / value 游标 三处是否与 ground-truth 一致。
依赖:docs/.../crave_ep2302_30hz_decoded.mp4 + temp/crave_a1a2/ep2302_bundle.npz + ep2302 原视频。
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/crave_ep2302_video_validate.py
"""
from __future__ import annotations

import json

import av
import cv2
import numpy as np

from crave.config import REPO, resolve_dataset
from crave.data import kai0

OUTV = REPO / "crave/docs/visualization/centroid_decoder"; OUTJ = REPO / "temp/crave_a1a2"
EP = 2302
# 与 render_video 完全一致的几何
W, H = 1000, 540; px0, py0, pw, ph = 510, 40, 470, 250


def main():
    cfg = resolve_dataset("kai0_base")
    b = np.load(OUTJ / "ep2302_bundle.npz")
    v, ms_idx, Pord, proto = b["v"], b["ms_idx"], b["Pord"], b["proto"]; n = len(v)
    # 原始 ep2302 相机帧(crop224, 与 render 同)
    mp4src = kai0.video_path(cfg, EP)
    cam_gt = []
    c = av.open(str(mp4src))
    for f in c.decode(video=0): cam_gt.append(kai0.crop224(f.to_ndarray(format="rgb24")))
    c.release() if hasattr(c, "release") else c.close()
    print(f"ep2302 原帧 {len(cam_gt)} | bundle n={n}", flush=True)

    # 逐帧读 mp4 验证
    cap = av.open(str(OUTV / "crave_ep2302_30hz_decoded.mp4"))
    cam_bad = thumb_bad = cursor_bad = blank = 0; vidn = 0
    cam_errs, thumb_errs, cur_errs = [], [], []
    for t, fr in enumerate(cap.decode(video=0)):
        if t >= n: break
        vidn += 1
        img = fr.to_ndarray(format="rgb24")  # RGB
        if img.shape[:2] != (H, W): print(f"  [frame {t}] 尺寸异常 {img.shape}"); continue
        if img.std() < 3: blank += 1
        # ① 相机帧:左 [40:520,15:495] vs crop224(t)→480
        cam_v = img[40:520, 15:495]
        cam_e = cam_gt[t] if t < len(cam_gt) else cam_gt[-1]
        cam_e = cv2.resize(cam_e, (480, 480))
        e1 = float(np.mean(np.abs(cam_v.astype(int) - cam_e.astype(int)))); cam_errs.append(e1)
        if e1 > 18: cam_bad += 1
        # ② 解码质心缩略图:[py0+ph+30:+240, px0:px0+210] vs proto[ms_idx[t]]→210
        th_v = img[py0 + ph + 30:py0 + ph + 240, px0:px0 + 210]
        th_e = cv2.resize(proto[int(ms_idx[t])], (210, 210))
        e2 = float(np.mean(np.abs(th_v.astype(int) - th_e.astype(int)))); thumb_errs.append(e2)
        if e2 > 18: thumb_bad += 1
        # ③ value 游标红点:期望位置 (px0 + x, py0 + yy)
        x = int(t / n * (pw - 1)); yy = int((1 - v[t]) * (ph - 1))
        ex, ey = px0 + x, py0 + yy
        win = img[max(0, ey - 6):ey + 7, max(0, ex - 6):ex + 7].astype(int)
        redness = win[..., 0] - (win[..., 1] + win[..., 2]) / 2  # R 显著高
        ok_cursor = redness.max() > 60 if win.size else False
        cur_errs.append(float(redness.max()) if win.size else -1)
        if not ok_cursor: cursor_bad += 1
        if (t + 1) % 500 == 0: print(f"  验证 {t+1}/{n} ...", flush=True)
    cap.close()

    res = {"video_frames": vidn, "bundle_frames": int(n), "frame_count_match": vidn == n,
           "blank_frames": blank,
           "cam_panel": {"bad(>18)": cam_bad, "mean_abs_err": round(float(np.mean(cam_errs)), 2), "max_err": round(float(np.max(cam_errs)), 2)},
           "decoded_thumb": {"bad(>18)": thumb_bad, "mean_abs_err": round(float(np.mean(thumb_errs)), 2), "max_err": round(float(np.max(thumb_errs)), 2)},
           "value_cursor": {"bad(no red dot)": cursor_bad, "min_redness": round(float(np.min(cur_errs)), 1)},
           "ALL_PASS": (vidn == n and cam_bad == 0 and thumb_bad == 0 and cursor_bad == 0 and blank == 0)}
    json.dump(res, open(OUTJ / "ep2302_video_validation.json", "w"), indent=2, ensure_ascii=False)
    print("\n==== 逐帧验证结果 ====")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
