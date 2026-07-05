#!/usr/bin/env python
"""Render the WHOLE episode: at every frame, the milestone+1 subgoal predictor
(forward-from-current) predicts the next milestone; compared against the real next-milestone
target, both decoded through the SAME patch decoder (so the comparison isolates PREDICTION
quality from decoder quality).

Per frame t:
  [ current frame t | PREDICTED m+1 (fwd(g_t, predm(g_t)) -> decode)
                    | REAL m+1 (encode(next-stage medoid) -> decode) | REAL m+1 frame ]

Two data sources:
  --episode N            : a kai0_base episode via the cached crave index (in-distribution).
  --raw_video PATH.mp4   : ANY video (e.g. vis_base) read directly; pooled features derived from
                           the grids and stages assigned with the kai0 recurrence prototypes ->
                           this is the CROSS-DATASET generalization test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
import cv2  # noqa: E402
from train_lawm_patch import load_index, read_imgs, ForwardDec  # noqa: E402
from optimize_subgoal import PredM  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402


def load_decoder(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    D = make_decoder(ck["din"], ck["dec"]).to(device); D.load_state_dict(ck["model"]); D.eval()
    mu = torch.from_numpy(np.asarray(ck["mu"])).view(1, -1, 1, 1).to(device)
    sd = torch.from_numpy(np.asarray(ck["sd"])).view(1, -1, 1, 1).to(device)

    def dec(grids_np):
        with torch.no_grad():
            o = D((torch.from_numpy(grids_np.astype(np.float32)).to(device) - mu) / sd).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    return dec, f"{ck.get('dec','?')}{'+GDL' if ck.get('gdl') else ''}"


def label_bar(w, text, h=26, shade=30):
    bar = np.full((h, w, 3), shade, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
    return bar


def read_video_frames(path, enc_res, disp_res, max_frames):
    cap = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im[:, :, ::-1])
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {path}")
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    enc = np.stack([cv2.resize(f, (enc_res, enc_res)) for f in frames]).astype(np.uint8)
    disp = np.stack([cv2.resize(f, (disp_res, disp_res)) for f in frames]).astype(np.uint8)
    return enc, disp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=8)
    ap.add_argument("--raw_video", default="", help="if set, read this mp4 directly (cross-dataset test)")
    ap.add_argument("--max_frames", type=int, default=160)
    ap.add_argument("--predictor", default="lmwm/outputs/subgoal_opt/milestone_cd128.pt")
    ap.add_argument("--decoder", default="lmwm/checkpoints/patch_decoder/patch_dec_big_gdl0.5.pt")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--out", default="lmwm/outputs/milestone_predict_episode.mp4")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    proto = np.load("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")["prototype_table"].astype(np.float32)
    enc = load_encoder("dinov3-h", device=dev)

    if args.raw_video:
        src = f"raw:{Path(args.raw_video).name}"
        enc_imgs, disp_imgs = read_video_frames(args.raw_video, 256, args.cell, args.max_frames)
        print(f"{src}: {len(enc_imgs)} frames; encoding grids ...", flush=True)
        grids = enc.encode_grid(enc_imgs).astype(np.float32); din = grids.shape[1]
        Fn_ep = grids.reshape(len(grids), din, -1).mean(2)                  # pooled = mean over 16x16 patch tokens
        Fn_ep = Fn_ep / (np.linalg.norm(Fn_ep, axis=1, keepdims=True) + 1e-8)
    else:
        src = f"kai0 ep{args.episode}"
        E, FR, Fn = load_index(args.feature_dir)
        loc = np.where(E == args.episode)[0]
        if len(loc) == 0:
            raise SystemExit(f"episode {args.episode} not in index")
        order = loc[np.argsort(FR[loc])]
        enc_imgs, disp_imgs = read_imgs(args.dataset_root, args.camera, E, FR, order, 256, args.cell)
        print(f"{src}: {len(order)} frames; encoding grids ...", flush=True)
        grids = enc.encode_grid(enc_imgs).astype(np.float32); din = grids.shape[1]
        Fn_ep = Fn[order]

    seq = (Fn_ep @ proto.T).argmax(1)
    ch = np.where(np.diff(seq) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
    seg_med, seg_stage = [], []
    for s, e in zip(st, en):
        m = int(seq[s]); seg_med.append(s + int((Fn_ep[s:e] @ proto[m]).argmax())); seg_stage.append(m)
    seg_of = np.zeros(len(seq), int)
    for i, (s, e) in enumerate(zip(st, en)):
        seg_of[s:e] = i
    print(f"{src}: {len(grids)} frames, {len(seg_med)} stage segments", flush=True)

    ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
    cd, gmu, gsd = ck["code_dim"], ck["gmu"], ck["gsd"]
    fwd = ForwardDec(din, cd).to(dev); fwd.load_state_dict(ck["fwd"]); fwd.eval()
    predm = PredM(din, cd).to(dev); predm.load_state_dict(ck["predm"]); predm.eval()
    vae_pm = None
    if "vae_pm" in ck:
        import torch.nn as nn
        vae_pm = nn.Linear(cd, 2 * cd).to(dev); vae_pm.load_state_dict(ck["vae_pm"]); vae_pm.eval()
    decode, dec_label = load_decoder(args.decoder, dev)
    print(f"predictor code_dim={cd} ({ck['mode']}, head={ck.get('code_head','det')}, trained on kai0); decoder={dec_label}", flush=True)

    gz = torch.from_numpy(((grids - gmu) / gsd).astype(np.float32))
    preds_raw = np.zeros_like(grids)
    with torch.no_grad():
        for b in range(0, len(grids), 128):
            gt = gz[b:b + 128].to(dev)
            code = predm(gt)
            if vae_pm is not None:
                code = vae_pm(code).chunk(2, -1)[0]                # deploy = posterior mean (mu)
            preds_raw[b:b + 128] = (fwd(gt, code).cpu().numpy() * gsd + gmu)
    pred_imgs = decode(preds_raw)
    self_dec_imgs = decode(grids)

    def cos(a, b):
        a, b = a.reshape(-1), b.reshape(-1)
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    C = args.cell
    titles = ["current frame t", "PREDICTED m+1 -> decode", "REAL m+1 encode->decode", "REAL m+1 frame"]
    W, H = C * 4, C + 26
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    coss = []
    for j in range(len(grids)):
        si = seg_of[j]; ni = min(si + 1, len(seg_med) - 1)
        tgt = seg_med[ni]; cur_m, tgt_m = seg_stage[si], seg_stage[ni]
        c = cos(preds_raw[j], grids[tgt]); coss.append(c)
        cells = [disp_imgs[j], pred_imgs[j], self_dec_imgs[tgt], disp_imgs[tgt]]
        cells = [cv2.resize(x, (C, C)) for x in cells]
        row = np.concatenate(cells, axis=1)
        subt = [f"m{cur_m} ({j+1}/{len(grids)})", f"pred cos={c:.3f}", f"real m{tgt_m}", f"m{tgt_m}"]
        bars = np.concatenate([label_bar(C, f"{titles[k]} | {subt[k]}") for k in range(4)], axis=1)
        vw.write(np.concatenate([bars, row], axis=0)[:, :, ::-1])
    vw.release()
    print(f"wrote {args.out} ({W}x{H}, {len(grids)} frames @ {args.fps}fps) | mean pred cos={np.mean(coss):.3f}", flush=True)


if __name__ == "__main__":
    main()
