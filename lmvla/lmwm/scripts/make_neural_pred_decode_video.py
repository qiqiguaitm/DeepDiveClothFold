#!/usr/bin/env python
"""Closed-loop viz of the NEURAL LMWM predictor (unified DINOv3-base space).

For each frame: current DINOv3-base grid -> gist -> MDN predm.deploy_mean -> code ẑ
-> generator fwd(grid, ẑ) -> predicted next-milestone GRID -> GRID decoder -> image.
Everything in DINOv3-base space (no SigLIP), so the neural prediction is directly decodable.

Panels: [ real(t) | neural-predicted milestone+1 | milestone+1 代表帧 (encode→decode) ].
  · neural pred m+1  = decode( generator fwd(cur_grid, ẑ) )        <- the model's prediction
  · m+1 repr frame   = decode( encode_grid( NEXT-SEGMENT medoid frame ) )
                       Segmentation is EXACTLY the LMWM training pipeline (build_pairs_abl): Viterbi-monotone
                       segments, per-segment medoid = frame in THIS episode's segment nearest the cluster
                       center, target = the NEXT segment's medoid. So this panel is literally the frame the
                       generator was TRAINED to render -> directly comparable to the neural prediction.
                       Per-episode (same cloth) => consistent appearance (global medoid would mix cloth colors).

Space alignment: grid decoder trained on per-patch-L2 grids (Gn); LMWM works in
standardized (Gn-gmu)/gsd. So fwd output -> ×gsd+gmu -> re-L2-per-patch -> decode.

Run in CRAVE env: /home/tim/miniconda3/envs/srpo/bin/python
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path("/home/tim/workspace/deepdive_kai0")
sys.path.insert(0, str(REPO / "lmvla/lmwm/scripts"))
sys.path.insert(0, str(REPO / "lmvla/crave/src"))
from train_dinov3base_video_decoder import load_features, l2  # noqa: E402
from train_twomodel_v2 import MilestoneGenerator, MilestonePredictor  # noqa: E402
from crave.decoding import make_decoder  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from crave.utils.dp import viterbi_forward  # noqa: E402  (SAME segmentation as build_pairs_abl training)


def l2c(x):  # L2 over channel axis=1 of (N,768,16,16)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def to_bgr(x):
    return cv2.cvtColor(((np.clip(x, -1, 1).transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def write_h264(tmp, out, fps):
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "20", str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            tmp.unlink(missing_ok=True); return
    shutil.move(str(tmp), str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmwm_ckpt", required=True, type=Path)
    ap.add_argument("--grid_ckpt", required=True, type=Path)
    ap.add_argument("--graph", required=True, type=Path, help="recurrence_graph.npz (prototype_table + pord)")
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--episodes", default="100,300,800")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    dev = args.device; R = 128
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])

    ck = torch.load(args.lmwm_ckpt, map_location=dev, weights_only=False)
    din, cdim, K = ck["din"], ck["code_dim"], ck["K"]; gmu, gsd = ck["gmu"], ck["gsd"]
    fwd = MilestoneGenerator(din, cdim).to(dev); fwd.load_state_dict(ck["fwd"]); fwd.eval()
    predm = MilestonePredictor(din, cdim, K).to(dev); predm.load_state_dict(ck["predm"]); predm.eval()
    gck = torch.load(args.grid_ckpt, map_location=dev, weights_only=False)
    dec = make_decoder(768, gck["dec"]).to(dev); dec.load_state_dict(gck["model"]); dec.eval()
    enc = load_encoder("dinov3-base", device=dev)

    # recurrence graph: milestone centers (pooled DINO) + progress order (for Viterbi segmentation)
    gr = np.load(args.graph); protoL = l2(gr["prototype_table"].astype(np.float32))
    pord = gr["pord"].astype(np.float32); Mn = len(protoL)
    E, FR, F, _ = load_features(args.feature_dir)
    Fl = l2(F.astype(np.float32))

    def read_grouped(ef_list, res):                                          # [(E,FR)] -> uint8 (N,res,res,3) RGB
        out = np.zeros((len(ef_list), res, res, 3), np.uint8)
        by_ep: dict[int, list] = {}
        for i, (e, x) in enumerate(ef_list):
            by_ep.setdefault(int(e), []).append((i, int(x)))
        for e, items in by_ep.items():
            cap = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{e // cs:03d}/{args.camera}/episode_{e:06d}.mp4"))
            for i, x in items:
                cap.set(cv2.CAP_PROP_POS_FRAMES, x); ok, im = cap.read()
                if ok:
                    out[i] = cv2.resize(im[:, :, ::-1], (res, res))
            cap.release()
        return out

    def enc_grid(imgs):                                                      # uint8 (N,224,224,3) -> l2 grid (N,768,16,16)
        G = np.zeros((len(imgs), 768, 16, 16), np.float32)
        for b in range(0, len(imgs), 256):
            G[b:b + 256] = enc.encode_grid(imgs[b:b + 256], bs=64).astype(np.float32)
        return l2c(G)

    def decode(gn):                                                          # (K,768,16,16)->(K,3,R,R)
        with torch.no_grad():
            return torch.cat([dec(torch.from_numpy(gn[s:s + 64]).to(dev)) for s in range(0, len(gn), 64)]).cpu().numpy()

    # NOTE: the "m+1 代表帧" is picked PER-EPISODE (within this episode's next-stage frames), NOT globally.
    # A global medoid would mix cloths of different colors across episodes -> color mismatch with the current
    # episode. Same-episode medoid = same cloth = consistent appearance.

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (R * 3, R + 16))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ep in [int(x) for x in args.episodes.split(",")]:
        m = E == ep
        if not m.any():
            print(f"  ep{ep} absent"); continue
        o = np.argsort(FR[m]); fr = FR[m][o]; feats = Fl[m][o]
        # SAME segmentation as LMWM training (build_pairs_abl): Viterbi-monotone -> segments -> per-seg medoid,
        # target = NEXT segment's medoid. So "m+1 repr" == the exact frame the generator was trained to render.
        d2 = np.linalg.norm(feats[:, None] - protoL[None], axis=2)
        ms = viterbi_forward(d2, pord, up=3.0, down=25.0, hard_start=True)
        ch = np.where(np.diff(ms) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(ms)]])
        seg_med, spans = [], []
        for s, e in zip(st, en):
            mm = int(ms[s]); seg_med.append(int(s + (feats[s:e] @ protoL[mm]).argmax())); spans.append((int(s), int(e)))
        seg_of = np.zeros(len(ms), int)
        for i, (s, e) in enumerate(spans):
            seg_of[s:e] = i
        nseg = len(spans)
        nextmed = np.array([seg_med[min(seg_of[t] + 1, nseg - 1)] for t in range(len(ms))])  # next-seg medoid (local idx)
        reals = read_grouped([(ep, int(x)) for x in fr], R)
        Gn = enc_grid(read_grouped([(ep, int(x)) for x in fr], 224))
        with torch.no_grad():
            GZ = torch.from_numpy(((Gn - gmu) / gsd).astype(np.float32)).to(dev)  # LMWM space
            zhat = predm.deploy_mean(GZ.mean((2, 3)))                        # neural predicted code
            pred = l2c(fwd(GZ, zhat).cpu().numpy() * gsd + gmu)              # un-standardize -> re-L2 per patch
        prd_dec = decode(pred)
        umed = sorted(set(int(x) for x in nextmed))                          # distinct next-seg medoid frames (this ep = same cloth)
        med_gn = enc_grid(read_grouped([(ep, int(fr[li])) for li in umed], 224))
        med_dec = decode(med_gn); med_map = {li: med_dec[i] for i, li in enumerate(umed)}
        for t in range(len(fr)):
            row = np.concatenate([reals[t][:, :, ::-1], to_bgr(prd_dec[t]), to_bgr(med_map[int(nextmed[t])])], axis=1)
            bar = np.zeros((16, R * 3, 3), np.uint8)
            for i, txt in enumerate(["real (t)", "neural pred m+1", "m+1 repr frame"]):
                cv2.putText(bar, txt, (i * R + 2, 12), font, 0.32, (255, 255, 255), 1)
            vw.write(np.concatenate([bar, row], axis=0))
        print(f"  ep{ep}: {len(fr)} frames", flush=True)
    enc.unload(); vw.release()
    write_h264(tmp, args.out, args.fps)
    print(f"wrote {args.out}  ([real | neural pred m+1 | m+1 repr frame], H.264)")


if __name__ == "__main__":
    main()
