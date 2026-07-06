#!/usr/bin/env python
"""Train a π0.5 SigLIP patch-grid -> RGB decoder, so the NEW-architecture subgoal (Stage-2 predicts a
SigLIP grid) can be VISUALIZED. SigLIP is contrastive/less-reconstruction-rich than DINOv3 (VLA-JEPA),
so this is the risky part -> dumps sample val decodes as PNG to eyeball sharpness before rendering.

  SigLIP grid (1152,16,16) --proj--> (512,16,16) --5x up--> 3x256x256   (L1 + 0.5 L2 + GDL)

Trained on (SigLIP-encoded frame, real frame) pairs. Saves ckpt for render_twomodel_video.py.
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

PI05_NPZ = "/vePFS/tim/workspace/openpi_cache/paligemma_weights/pt_224.npz"
PI05_NPZ_GF3 = "/vePFS-North-E/vis_robot/openpi_cache/paligemma_weights/pt_224.npz"


class GridDecoder(nn.Module):
    """SigLIP grid (din,16,16) -> 3x256x256."""
    def __init__(self, din=1152, res=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(din, 512, 1), nn.GroupNorm(8, 512), nn.GELU())

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.net = nn.Sequential(up(512, 256), up(256, 128), up(128, 64),
                                 nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())  # 16->32->64->128->256

    def forward(self, g):
        return self.net(self.proj(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_pairs", type=int, default=16000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gdl_weight", type=float, default=0.5)
    ap.add_argument("--out", default="lmwm/checkpoints/siglip_decoder/dec.pt", type=Path)
    ap.add_argument("--sample_png", default="lmwm/outputs/siglip_decoder_samples.png")
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)

    E, FR, _ = load_index(args.feature_dir)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(E), min(args.n_pairs, len(E)), replace=False)
    uniq = np.array(sorted(set(sel.tolist())))
    imgs224, imgs_tgt = read_imgs(args.dataset_root, args.camera, E, FR, uniq, 224, args.res)  # enc@224, tgt@res
    print(f"read {len(uniq)} frames", flush=True)
    enc = SiglipBigVision(npz, device=dev)
    grids = enc.encode_grid(imgs224, bs=32)                                # (N,1152,16,16) from 224 -> 16x16
    din = grids.shape[1]

    Y = torch.from_numpy(imgs_tgt.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous()
    X = torch.from_numpy(grids)
    dec = GridDecoder(din, args.res).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    n = len(X); n_val = max(64, n // 10); perm0 = torch.randperm(n)
    val_i, tr_i = perm0[:n_val], perm0[n_val:]; bs = 64
    for ep in range(args.epochs):
        dec.train(); p = tr_i[torch.randperm(len(tr_i))]
        for b in range(0, len(p), bs):
            bi = p[b:b + bs]; x, y = X[bi].to(dev), Y[bi].to(dev)
            pr = dec(x)
            loss = (pr - y).abs().mean() + 0.5 * ((pr - y) ** 2).mean()
            if args.gdl_weight > 0:
                loss = loss + args.gdl_weight * (
                    ((pr[:, :, :, 1:] - pr[:, :, :, :-1]) - (y[:, :, :, 1:] - y[:, :, :, :-1])).abs().mean()
                    + ((pr[:, :, 1:, :] - pr[:, :, :-1, :]) - (y[:, :, 1:, :] - y[:, :, :-1, :])).abs().mean())
            opt.zero_grad(); loss.backward(); opt.step()
        if ep == 0 or (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            dec.eval()
            with torch.no_grad():
                vl = (dec(X[val_i].to(dev)) - Y[val_i].to(dev)).abs().mean().item()
            print(f"epoch {ep + 1}/{args.epochs} val_L1={vl:.4f}", flush=True)

    # dump 8 val [pred | real] pairs stacked, for eyeballing sharpness
    dec.eval()
    with torch.no_grad():
        vi = val_i[:8]; pr = dec(X[vi].to(dev)).cpu()
    def to_u8(t): return ((t.permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    P, G = to_u8(pr), to_u8(Y[vi])
    rows = [np.concatenate([P[i], G[i]], axis=1) for i in range(8)]      # [pred|real] per row
    panel = np.concatenate(rows, axis=0)[:, :, ::-1]                     # RGB->BGR for cv2
    Path(args.sample_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.sample_png, panel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dec.state_dict(), "res": args.res, "din": int(din),
                "input": "pi05 SigLIP grid 1152x16x16", "val_L1": vl}, args.out)
    print(f"saved {args.out} val_L1={vl:.4f}; samples -> {args.sample_png}", flush=True)


if __name__ == "__main__":
    main()
