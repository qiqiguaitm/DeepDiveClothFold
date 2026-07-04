#!/usr/bin/env python
"""Perfect the patch-grid decoder. Sweeps capacity (small/medium/big/xl) + adds a GDL (gradient
difference) sharpness loss + more data/epochs. Current 'small' baseline: val L1 0.0248, sharp 222
vs real 1034 (soft). One (dec,gdl) per GPU for parallel sweep.

Saves ckpt in the SAME format as track_b1_patch_decoder (loadable by make_decoder_compare3_vis).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import make_decoder, P as GRID_P  # noqa: E402


def gdl_loss(pred, real):
    """Gradient difference loss: match horizontal+vertical gradients (deterministic sharpener)."""
    pdx = pred[:, :, :, 1:] - pred[:, :, :, :-1]; rdx = real[:, :, :, 1:] - real[:, :, :, :-1]
    pdy = pred[:, :, 1:, :] - pred[:, :, :-1, :]; rdy = real[:, :, 1:, :] - real[:, :, :-1, :]
    return (pdx - rdx).abs().mean() + (pdy - rdy).abs().mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec", default="big", choices=["small", "medium", "big", "xl"])
    ap.add_argument("--gdl", type=float, default=0.5, help="GDL sharpness weight (0=off)")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=24000)
    ap.add_argument("--n_val", type=int, default=1200)
    ap.add_argument("--enc_res", type=int, default=256)
    ap.add_argument("--tgt_res", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    tag = f"{args.dec}{'_gdl'+str(args.gdl) if args.gdl else ''}"
    out = Path(args.out) if args.out else Path(f"lmwm/checkpoints/patch_decoder/patch_dec_{tag}.pt")

    E, FR, _ = load_index(args.feature_dir)
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    is_val = np.array([e in val_eps for e in E])
    tr_g = rng.choice(np.where(~is_val)[0], min(args.n_train, int((~is_val).sum())), replace=False)
    va_g = rng.choice(np.where(is_val)[0], min(args.n_val, int(is_val.sum())), replace=False)
    print(f"[{tag}] {len(tr_g)}+{len(va_g)} frames", flush=True)

    tr_enc, tr_tgt = read_imgs(args.dataset_root, args.camera, E, FR, tr_g, args.enc_res, args.tgt_res)
    va_enc, va_tgt = read_imgs(args.dataset_root, args.camera, E, FR, va_g, args.enc_res, args.tgt_res)
    enc = load_encoder("dinov3-h", device=dev)
    print(f"[{tag}] encoding grids ...", flush=True)
    tr_grid = enc.encode_grid(tr_enc).astype(np.float32); va_grid = enc.encode_grid(va_enc).astype(np.float32)
    din = tr_grid.shape[1]
    mu = tr_grid.mean(axis=(0, 2, 3), dtype=np.float32); sd = tr_grid.std(axis=(0, 2, 3)).astype(np.float32) + 1e-4
    muT = torch.from_numpy(mu).view(1, din, 1, 1).to(dev); sdT = torch.from_numpy(sd).view(1, din, 1, 1).to(dev)
    Y = torch.from_numpy(tr_tgt.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous().to(dev)
    Gg = torch.from_numpy(tr_grid).to(dev)
    D = make_decoder(din, args.dec).to(dev)
    opt = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    n = len(tr_grid); bs = 64
    print(f"[{tag}] training dec={args.dec} gdl={args.gdl} ({args.epochs} ep) ...", flush=True)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, bs):
            bi = perm[b:b + bs]; pred = D((Gg[bi] - muT) / sdT); tgt = Y[bi]
            loss = (pred - tgt).abs().mean() + 0.5 * ((pred - tgt) ** 2).mean()
            if args.gdl:
                loss = loss + args.gdl * gdl_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    D.eval()

    with torch.no_grad():
        rec = []
        for b in range(0, len(va_grid), 256):
            rec.append(D((torch.from_numpy(va_grid[b:b + 256]).to(dev) - muT) / sdT).cpu().numpy())
        rec = np.concatenate(rec)
    rec_u8 = np.clip((rec.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    l1 = float(np.abs(va_tgt.astype(float) - rec_u8.astype(float)).mean())
    def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    vs = float(np.mean([sharp(x) for x in rec_u8])); rs = float(np.mean([sharp(x) for x in va_tgt]))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": D.state_dict(), "mu": mu, "sd": sd, "din": din, "dec": args.dec, "res": args.tgt_res,
                "grid_P": GRID_P, "val_L1_frac": l1 / 255, "val_sharp": vs, "real_sharp": rs, "gdl": args.gdl}, out)
    res = {"dec": args.dec, "gdl": args.gdl, "val_L1_frac": round(l1 / 255, 4), "val_sharp": round(vs, 0),
           "real_sharp": round(rs, 0), "baseline_small_L1": 0.0248, "baseline_small_sharp": 222}
    Path("lmwm/outputs/decoder_opt").mkdir(parents=True, exist_ok=True)
    Path(f"lmwm/outputs/decoder_opt/{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
