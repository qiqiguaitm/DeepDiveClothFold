#!/usr/bin/env python
"""Decoder + RE-ENCODE consistency loss: decode(latent) must re-encode (DINOv3-H) back to
the input latent. Directly optimizes SEMANTIC fidelity (the re-encode cos metric), which
pixel L1 ignores. Base = L1(+GDL); adds lambda*(1-cos(encode(decode), latent)).

Reports val pixel-L1 AND val re-encode cos (vs L1-only baseline ~0.32, GAN ~0.47).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402
from lever_patch_token import read_enc  # noqa: E402
from dinov3h_gated import DINOv3HGated  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h_v2", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--dino_path", default="/vePFS/xiezhicong/.cache/huggingface/hub/dinov3-vith16plus-pretrain-lvd1689m")
    ap.add_argument("--n_pairs", type=int, default=12000)
    ap.add_argument("--reencode_weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--out", default="lmwm/checkpoints/dinov3h_decoder/dec_reencode_v2.pt", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    E, FR, Fb = load_features(args.feature_dir)
    rng = np.random.default_rng(2026)
    sel = rng.choice(len(E), args.n_pairs + 800, replace=False)
    print(f"reading {len(sel)} frames ...", flush=True)
    imgs = read_enc(args.dataset_root, args.camera, E[sel], FR[sel], args.res)
    lat = l2(Fb[sel].astype(np.float32))
    Y = torch.from_numpy(imgs.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous()
    X = torch.from_numpy(lat)
    ntr = args.n_pairs                                        # LOCAL indices into X/Y
    va = np.arange(ntr, len(sel))

    reenc = DINOv3HGated(args.dino_path, device=str(dev))                # frozen re-encoder
    dec = PooledDecoder(din=1280, res=args.res).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    for ep in range(args.epochs):
        perm = torch.randperm(ntr)
        for b in range(0, ntr, args.bs):
            bi = perm[b:b + args.bs]
            x = X[bi].to(dev); y = Y[bi].to(dev)
            pred = dec(x)                                                # [-1,1]
            l_pix = (pred - y).abs().mean() + 0.5 * ((pred - y) ** 2).mean()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                re = reenc.encode_pooled_tensor((pred + 1) / 2)         # differentiable re-encode
            re = F.normalize(re, dim=-1)
            l_re = (1 - (re * x).sum(-1)).mean()
            loss = l_pix + args.reencode_weight * l_re
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            dec.eval()
            with torch.no_grad():
                vp = dec(X[va].to(dev))
                vl1 = float((vp - Y[va].to(dev)).abs().mean() / 2)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vre = F.normalize(reenc.encode_pooled_tensor((vp + 1) / 2), dim=-1)
                vcos = float((vre * X[va].to(dev)).sum(-1).mean())
            def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
            srec = np.clip((vp[:8].cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
            print(f"epoch {ep}: val L1={vl1:.4f} reencode_cos={vcos:.4f} sharp={np.mean([sharp(x) for x in srec]):.0f}", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": dec.state_dict(), "res": args.res, "din": 1280,
                        "meta": {"epoch": ep, "val_reencode_cos": vcos, "val_L1": vl1}}, args.out)
            dec.train()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dec.state_dict(), "res": args.res, "din": 1280,
                "meta": {"reencode_weight": args.reencode_weight, "val_reencode_cos": vcos, "val_L1": vl1}}, args.out)
    print(f"saved {args.out} | val reencode_cos={vcos:.4f} L1={vl1:.4f}")


if __name__ == "__main__":
    main()
