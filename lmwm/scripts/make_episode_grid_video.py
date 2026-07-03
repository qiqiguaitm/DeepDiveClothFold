#!/usr/bin/env python
"""Episode video: left = real episode frames; right (stacked) = the current frame's
PRED grid -> decode  and  TRUE next-medoid -> encode -> decode (decoder ceiling).

The grid predictor is on-manifold (LaWM feature-space loss), the VLA-facing path.
Trains (and caches) the predictor once, then walks a held-out episode frame by frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lmwm.data import split_indices  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402
from track_b2_grid_predict import GridGen, render_medoid_images  # noqa: E402
from train_dinov3h_decoder import load_features, l2  # noqa: E402


def get_predictor(args, dev, din_in, din_grid, enc):
    gpath = Path("lmwm/checkpoints/grid_predictor/G_lawm.pt")
    G = GridGen(din_in, din_grid).to(dev)
    if gpath.exists():
        G.load_state_dict(torch.load(gpath, map_location="cpu", weights_only=False)["model"]); G.eval()
        print("loaded cached grid predictor", flush=True)
        return G
    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, _ = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti = ti.numpy(); ok = np.linalg.norm(z["next_medoid"], axis=1) > 1e-6
    rng = np.random.default_rng(0)
    ti = rng.choice(ti[ok[ti]], min(12000, int(ok[ti].sum())), replace=False)
    print("training on-manifold grid predictor (LaWM loss) ...", flush=True)
    tr_img, _ = render_medoid_images(z, ti, args.dataset_root, args.camera, args.feature_dir, args.res)
    tr_grid = torch.from_numpy(enc.encode_grid(tr_img).astype(np.float32))
    Xt = torch.from_numpy(z["current"][ti].astype(np.float32))
    opt = torch.optim.AdamW(G.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
    for s in range(6000):
        bi = torch.randint(0, len(ti), (64,))
        grid = G(Xt[bi].to(dev)); tg = tr_grid[bi].to(dev)
        rt = grid.flatten(2).transpose(1, 2); tt = tg.flatten(2).transpose(1, 2)
        loss = F.smooth_l1_loss(rt, tt, beta=0.1) + (1.0 - F.cosine_similarity(rt, tt, dim=-1).mean())
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    G.eval()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": G.state_dict()}, gpath)
    print("saved grid predictor", flush=True)
    return G


def label(img, text):
    out = cv2.copyMakeBorder(img, 26, 4, 4, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=1819)
    ap.add_argument("--episodes", default=None, help="comma list; overrides auto-pick")
    ap.add_argument("--max_frames", type=int, default=500)
    ap.add_argument("--native", action="store_true",
                    help="single episode, EVERY real frame at native fps (normal speed; "
                         "prediction held between stride-10 pair frames)")
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--patch_dec", default="lmwm/checkpoints/patch_decoder/patch_dec.pt", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="lmwm/docs/assets/ep_grid_pred_video.mp4", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    R = args.res

    ck = torch.load(args.patch_dec, map_location="cpu", weights_only=False)
    din_grid = int(ck["din"])
    D = make_decoder(din_grid, ck["dec"]).to(dev); D.load_state_dict(ck["model"]); D.eval()
    for p in D.parameters():
        p.requires_grad_(False)
    muT = torch.from_numpy(ck["mu"]).view(1, din_grid, 1, 1).to(dev)
    sdT = torch.from_numpy(ck["sd"]).view(1, din_grid, 1, 1).to(dev)
    def decode(grid): return D((grid - muT) / sdT)
    def d2img(t): return np.clip((t.detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)

    enc = load_encoder("dinov3-h", device=str(dev))
    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = set(vi.numpy().tolist())
    din_in = z["current"].shape[1]
    G = get_predictor(args, dev, din_in, din_grid, enc)

    # episode list: explicit, or auto-pick held-out episodes (longest first) up to max_frames
    ep_ids = z["episode_id"]
    if args.native:
        ep_list = [args.episode]
    elif args.episodes:
        ep_list = [int(x) for x in args.episodes.split(",")]
    else:
        held = {}
        for i in (vi if isinstance(vi, set) else set(vi)):
            held[int(ep_ids[i])] = held.get(int(ep_ids[i]), 0) + 1
        ep_list = [e for e, _ in sorted(held.items(), key=lambda kv: -kv[1])]

    E, FR, Fb = load_features(Path(args.feature_dir))
    Fn = l2(Fb.astype(np.float32))
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)

    BIG = R * 2; RW = 230
    W = BIG + 8 + RW; H = 26 + BIG + 4
    def rlabel(img, text):
        return label(cv2.resize(img, (RW, RW)), text)
    # native mode: use the source video's real fps so playback is normal speed
    out_fps = args.fps
    if args.native:
        c0 = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{args.episode // cs:03d}/{args.camera}/episode_{args.episode:06d}.mp4"))
        native_fps = c0.get(cv2.CAP_PROP_FPS); c0.release()
        out_fps = round(native_fps) if native_fps and native_fps > 1 else 30
        print(f"native fps = {native_fps:.2f} -> out {out_fps}", flush=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), fourcc, out_fps, (W, H))

    written = 0
    for ep in ep_list:
        if written >= args.max_frames:
            break
        idx = np.array([i for i in np.where(ep_ids == ep)[0] if i in vi])
        if len(idx) < 4:
            continue
        idx = idx[np.argsort(z["t"][idx])]
        ts = z["t"][idx].astype(np.int64)
        med = z["next_medoid"][idx].astype(np.float32); med /= np.linalg.norm(med, axis=1, keepdims=True) + 1e-8
        with torch.no_grad():
            pred = decode(G(torch.from_numpy(z["current"][idx].astype(np.float32)).to(dev)))
        pred_imgs = [d2img(pred[i]) for i in range(len(idx))]
        qloc = np.where(E == ep)[0]
        med_g = qloc[(Fn[qloc] @ med.T).argmax(0)]
        uniq, inv = np.unique(med_g, return_inverse=True)
        med_frames = np.stack([frame(int(E[g]), int(FR[g])) for g in uniq])
        ceil = decode(torch.from_numpy(enc.encode_grid(med_frames).astype(np.float32)).to(dev))
        ceil_imgs = [d2img(ceil[j]) for j in range(len(uniq))]
        for i in range(len(idx)):
            if written >= args.max_frames:
                break
            left = label(cv2.resize(frame(ep, int(ts[i])), (BIG, BIG)), f"ep{ep} frame t={int(ts[i])} (real)")
            right = np.vstack([rlabel(pred_imgs[i], "PRED grid -> decode"),
                               rlabel(ceil_imgs[inv[i]], "TRUE medoid enc->decode (ceiling)")])
            right = cv2.resize(right, (RW, left.shape[0]))
            canvas = np.hstack([left, np.full((left.shape[0], 8, 3), 20, np.uint8), right])
            vw.write(cv2.cvtColor(canvas[:H, :W], cv2.COLOR_RGB2BGR))
            written += 1
        print(f"  ep{ep}: +{len(idx)} -> {written}", flush=True)
    vw.release()
    for c in caps.values():
        c.release()
    print(f"saved {args.out} | {written} frames @ {args.fps}fps")


if __name__ == "__main__":
    main()
