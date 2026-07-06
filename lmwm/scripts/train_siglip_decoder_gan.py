#!/usr/bin/env python
"""Sharper π0.5 SigLIP grid -> image decoder via adversarial (PatchGAN hinge) + L1 + GDL. The plain
L1/GDL decoder (val_L1 0.053) is faithful but soft; a PatchGAN discriminator adds high-frequency
detail (repo practice: GAN decoder = sharp/recognizable). Same GridDecoder arch as the L1 one, so
render_twomodel_video.py loads it unchanged (--dec_ckpt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
import cv2  # noqa: E402
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402
from train_siglip_decoder import GridDecoder  # noqa: E402

PI05_NPZ = "/vePFS/tim/workspace/openpi_cache/paligemma_weights/pt_224.npz"
PI05_NPZ_GF3 = "/vePFS-North-E/vis_robot/openpi_cache/paligemma_weights/pt_224.npz"


class PatchD(nn.Module):
    """PatchGAN on 256x256 RGB -> patch real/fake logits (4 downsamples for 256)."""
    def __init__(self):
        super().__init__()

        def blk(i, o, norm=True):
            L = [nn.Conv2d(i, o, 4, 2, 1)]
            if norm:
                L.append(nn.InstanceNorm2d(o))
            L.append(nn.LeakyReLU(0.2, True))
            return L
        self.net = nn.Sequential(*blk(3, 64, norm=False), *blk(64, 128), *blk(128, 256),
                                 *blk(256, 512), nn.Conv2d(512, 1, 4, 1, 1))

    def forward(self, x):
        return self.net(x)


def sharp(u8):
    return cv2.Laplacian(cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()


def gdl(pr, y):
    return (((pr[:, :, :, 1:] - pr[:, :, :, :-1]) - (y[:, :, :, 1:] - y[:, :, :, :-1])).abs().mean()
            + ((pr[:, :, 1:, :] - pr[:, :, :-1, :]) - (y[:, :, 1:, :] - y[:, :, :-1, :])).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--l1_weight", type=float, default=12.0)
    ap.add_argument("--gdl_weight", type=float, default=2.0)
    ap.add_argument("--out", default="lmwm/checkpoints/siglip_decoder/dec_gan.pt", type=Path)
    ap.add_argument("--sample_png", default="lmwm/outputs/siglip_decoder_gan_samples.png")
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)

    E, FR, _ = load_index(args.feature_dir)
    rng = np.random.default_rng(args.seed)
    uniq = np.array(sorted(set(rng.choice(len(E), min(args.n_pairs, len(E)), replace=False).tolist())))
    imgs224, imgs_tgt = read_imgs(args.dataset_root, args.camera, E, FR, uniq, 224, args.res)
    print(f"read {len(uniq)} frames; SigLIP encoding ...", flush=True)
    enc = SiglipBigVision(npz, device=dev)
    grids = enc.encode_grid(imgs224, bs=32); din = grids.shape[1]

    Y = torch.from_numpy(imgs_tgt.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous()
    X = torch.from_numpy(grids)
    n = len(X); n_val = max(64, n // 10); perm = torch.randperm(n)
    val_i, tr_i = perm[:n_val], perm[n_val:]
    G = GridDecoder(din, args.res).to(dev); D = PatchD().to(dev)
    optG = torch.optim.AdamW(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bs = 48
    for ep in range(args.epochs):
        G.train(); D.train(); p = tr_i[torch.randperm(len(tr_i))]
        for b in range(0, len(p), bs):
            bi = p[b:b + bs]; x, y = X[bi].to(dev), Y[bi].to(dev)
            fake = G(x)
            optD.zero_grad()
            lossD = torch.relu(1 - D(y)).mean() + torch.relu(1 + D(fake.detach())).mean()
            lossD.backward(); optD.step()
            optG.zero_grad()
            lossG = -D(fake).mean() + args.l1_weight * (fake - y).abs().mean() + args.gdl_weight * gdl(fake, y)
            lossG.backward(); optG.step()
        if ep == 0 or (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            G.eval()
            with torch.no_grad():
                vp = G(X[val_i].to(dev)); vl1 = (vp - Y[val_i].to(dev)).abs().mean().item()
                vimg = np.clip((vp[:64].cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
            print(f"epoch {ep + 1}/{args.epochs} val_L1={vl1:.4f} sharp={np.mean([sharp(im) for im in vimg]):.1f}", flush=True)

    G.eval()
    with torch.no_grad():
        vi = val_i[:8]; pr = G(X[vi].to(dev)).cpu()
    to = lambda t: ((t.permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    P, GT = to(pr), to(Y[vi])
    panel = np.concatenate([np.concatenate([P[i], GT[i]], 1) for i in range(8)], 0)[:, :, ::-1]
    Path(args.sample_png).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(args.sample_png, panel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": G.state_dict(), "res": args.res, "din": int(din),
                "input": "pi05 SigLIP grid 1152x16x16", "val_L1": vl1, "loss": "hinge-GAN+L1+GDL"}, args.out)
    print(f"saved {args.out} val_L1={vl1:.4f}; samples -> {args.sample_png}", flush=True)


if __name__ == "__main__":
    main()
