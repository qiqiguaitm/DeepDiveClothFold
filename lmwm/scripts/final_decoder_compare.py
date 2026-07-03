#!/usr/bin/env python
"""FINAL decoder comparison on common held-out frames (unified gated DINOv3-H space):
  real | dec_v2 (L1) | dec_gan_v2 (GAN) | flow_b160 (flow-matching, WINNER)
Per decoder: re-encode cos (semantic fidelity), sharpness, pixel L1. Saves montage + json.
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
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402
from flow_decoder_gf3 import UNet  # noqa: E402
from dinov3h_gated import DINOv3HGated  # noqa: E402


def load_pooled(p, dev):
    ck = torch.load(p, map_location="cpu"); R = int(ck["res"])
    D = PooledDecoder(din=int(ck["din"]), res=R).to(dev); D.load_state_dict(ck["model"]); D.eval(); return D, R


@torch.no_grad()
def flow_sample(net, lat, res, dev, ode_steps=25):
    x = torch.randn(len(lat), 3, res, res, device=dev)
    for k in range(ode_steps):
        t = torch.full((len(lat),), k / ode_steps, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = net(x, t, lat)
        x = x + (1.0 / ode_steps) * v.float()
    return np.clip((x.cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec_l1", default="lmwm/checkpoints/dinov3h_decoder/dec_v2.pt")
    ap.add_argument("--dec_gan", default="lmwm/checkpoints/dinov3h_decoder/dec_gan_v2.pt")
    ap.add_argument("--flow", default="lmwm/checkpoints/dinov3h_decoder/dec_best.pt")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h_v2")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--dino_path", default="temp/lmwm_p0/dinov3h")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="lmwm/docs/assets/final_decoder_compare.png", type=Path)
    ap.add_argument("--out_json", default="lmwm/outputs/final_decoder_compare.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    DL, R = load_pooled(args.dec_l1, dev); DG, _ = load_pooled(args.dec_gan, dev)
    fck = torch.load(args.flow, map_location="cpu"); FR_res = int(fck["res"])
    FN = UNet(1280, fck["base"]).to(dev); FN.load_state_dict(fck["model"]); FN.eval()

    E, FRi, Fb = load_features(Path(args.feature_dir)); Fn = l2(Fb.astype(np.float32))
    rng = np.random.default_rng(11); sel = rng.choice(len(E), args.n, replace=False)
    lat = torch.from_numpy(Fn[sel]).to(dev)

    def dec_pooled(D):
        with torch.no_grad():
            o = D(lat).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    rL, rG = dec_pooled(DL), dec_pooled(DG)
    rF = flow_sample(FN, lat, FR_res, dev)

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)
    reals = np.stack([frame(int(E[g]), int(FRi[g])) for g in sel])
    for c in caps.values():
        c.release()

    reenc = DINOv3HGated(args.dino_path, device=str(dev))
    lat_np = Fn[sel]
    def reencode_cos(imgs):
        # imgs may be at flow res; resize to R for fair comparison already done by encoder interp
        re = F.normalize(reenc.encode_pooled_tensor(torch.from_numpy(imgs.astype(np.float32) / 255).permute(0, 3, 1, 2).to(dev)), dim=-1)
        return float((re.float().cpu().numpy() * lat_np).sum(1).mean())
    def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    def sh(a): return float(np.mean([sharp(x) for x in a]))
    def l1(a):
        rr = np.stack([cv2.resize(x, (a.shape[1], a.shape[1])) for x in reals]) if a.shape[1] != R else reals
        return float(np.abs(rr.astype(float) - a.astype(float)).mean() / 255)

    stat = {"real": {"sharp": round(sh(reals), 0)},
            "dec_v2_L1": {"reencode_cos": round(reencode_cos(rL), 4), "sharp": round(sh(rL), 0), "L1": round(l1(rL), 4)},
            "dec_gan_v2_GAN": {"reencode_cos": round(reencode_cos(rG), 4), "sharp": round(sh(rG), 0), "L1": round(l1(rG), 4)},
            "flow_b160_WINNER": {"reencode_cos": round(reencode_cos(rF), 4), "sharp": round(sh(rF), 0), "L1": round(l1(rF), 4)}}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(stat, indent=2), encoding="utf-8")
    print(json.dumps(stat, indent=2), flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [("real", reals), (f"dec_v2 L1\ncos{stat['dec_v2_L1']['reencode_cos']} sh{stat['dec_v2_L1']['sharp']:.0f}", rL),
            (f"dec_gan GAN\ncos{stat['dec_gan_v2_GAN']['reencode_cos']} sh{stat['dec_gan_v2_GAN']['sharp']:.0f}", rG),
            (f"flow_b160 *WIN*\ncos{stat['flow_b160_WINNER']['reencode_cos']} sh{stat['flow_b160_WINNER']['sharp']:.0f}", rF)]
    fig, ax = plt.subplots(4, args.n, figsize=(args.n * 1.5, 6.8))
    for r, (lab, imgs) in enumerate(rows):
        for j in range(args.n):
            ax[r, j].imshow(imgs[j]); ax[r, j].axis("off")
        ax[r, 0].set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("FINAL decoder comparison (unified DINOv3-H space) | flow-matching wins: sharp + faithful, no GAN hallucination / no L1 blur", fontsize=9)
    fig.tight_layout(rect=[0.06, 0, 1, 0.96])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
