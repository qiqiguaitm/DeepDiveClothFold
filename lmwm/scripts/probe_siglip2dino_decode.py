#!/usr/bin/env python
"""Validate the DINO-sandwich: can a VLA-SigLIP grid be adapted to DINOv3-H space well enough that
decoding STAYS sharp?  (User's Q: SigLIP->adapt->DINO->decode fidelity; and predicted-milestone decode.)

Assets reused (all local):
  encoders : SiglipBigVision (pt_224.npz, 1152 grid) ; DINOv3HGated (vith16plus, 1280 grid)
  decoders : crave patch_dec.pt (DINOv3-H grid 1280x16x16 -> 128^2, the fidelity 'king', val_L1 .0248)
             siglip_decoder GridDecoder dec.pt (SigLIP grid 1152x16x16 -> 256^2) as SigLIP-native ref

Trains adapter A: SigLIP grid -> DINO grid (per-token 1x1 + light 3x3 residual). Then on held-out frames:
  Q(a)  cos(A(sig), dino_true) ; decode L1/sharpness for {dino_true, A(sig), sig-native} vs original.
        -> is SigLIP->DINO near-lossless for appearance? does the sandwich beat SigLIP-native decode?
  Q(b') proxy: our lmwm2 SigLIP generator's predicted milestone -> A -> DINO -> decode, vs decode of the
        TRUE next-milestone DINO grid.  (Lower bound on a DINO-native generator; measures predict+decode.)
Outputs: JSON metrics + a visual panel PNG (fidelity rule: look at pixels).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(REPO / "crave/src"))
import cv2  # noqa: E402
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from train_twomodel_v2 import PI05_NPZ, PI05_NPZ_GF3  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402
from dinov3h_gated import DINOv3HGated  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402

DINO_DIR = "/vePFS/xiezhicong/.cache/huggingface/hub/dinov3-vith16plus-pretrain-lvd1689m"


class GridDecoder(nn.Module):  # SigLIP-native decoder (from train_siglip_decoder.py)
    def __init__(self, din=1152, res=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(din, 512, 1), nn.GroupNorm(8, 512), nn.GELU())
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.net = nn.Sequential(up(512, 256), up(256, 128), up(128, 64), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())
    def forward(self, g): return self.net(self.proj(g))


class Adapter(nn.Module):  # SigLIP grid (1152) -> DINO grid (1280): per-token 1x1 + light spatial residual
    def __init__(self, ci=1152, co=1280, h=1024):
        super().__init__()
        self.inp = nn.Conv2d(ci, h, 1)
        self.res = nn.Sequential(nn.Conv2d(h, h, 3, 1, 1), nn.GroupNorm(16, h), nn.GELU(),
                                 nn.Conv2d(h, h, 3, 1, 1), nn.GroupNorm(16, h), nn.GELU())
        self.out = nn.Conv2d(h, co, 1)
    def forward(self, g):
        x = self.inp(g); x = x + self.res(x); return self.out(x)


def sharp(img):  # Laplacian variance (repo sharpness metric)
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdir", default="temp/crave_full_dinov3h")
    ap.add_argument("--root", default="kai0/data/Task_A/kai0_base")
    ap.add_argument("--cam", default="observation.images.top_head")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--tag", default="s2d_kai0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    npz = PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3
    rng = np.random.default_rng(2026)

    E, FR, Fn = load_index(REPO / args.fdir)
    idx = rng.choice(len(E), min(args.n, len(E)), replace=False)
    ie224, disp = read_imgs(Path(args.root), args.cam, E, FR, idx, 224, 128)     # 224 for SigLIP, 128 disp target
    ie256, _ = read_imgs(Path(args.root), args.cam, E, FR, idx, 256, 128)        # 256 for DINO
    print(f"frames={len(idx)}", flush=True)

    sig_enc = SiglipBigVision(npz, device=dev)
    Gsig = sig_enc.encode_grid(ie224, bs=32).astype(np.float32)                  # (N,1152,16,16)
    del sig_enc; torch.cuda.empty_cache()
    dino = DINOv3HGated(DINO_DIR, device=dev)
    gd = dino.encode_grid(ie256, bs=32)                                          # (N,256,1280) tokens
    Gdino = gd.reshape(len(idx), 16, 16, 1280).transpose(0, 3, 1, 2).astype(np.float32)  # (N,1280,16,16)
    del dino; torch.cuda.empty_cache()

    n = len(idx); nva = max(200, n // 5); tr, va = np.arange(n)[nva:], np.arange(n)[:nva]
    Gs = torch.from_numpy(Gsig).to(dev); Gd = torch.from_numpy(Gdino).to(dev)
    # per-channel norm for adapter target (stabilize)
    dmu = Gd[tr].mean((0, 2, 3), keepdim=True); dsd = Gd[tr].std((0, 2, 3), keepdim=True) + 1e-4
    smu = Gs[tr].mean((0, 2, 3), keepdim=True); ssd = Gs[tr].std((0, 2, 3), keepdim=True) + 1e-4
    Gsn = (Gs - smu) / ssd

    A = Adapter().to(dev)
    opt = torch.optim.AdamW(A.parameters(), lr=2e-4, weight_decay=1e-5)
    for step in range(args.steps):
        b = tr[torch.randint(0, len(tr), (32,)).numpy()]
        pred = A(Gsn[b]) * dsd + dmu
        tgt = Gd[b]
        pn = F.normalize(pred.flatten(2).transpose(1, 2), dim=-1); tn = F.normalize(tgt.flatten(2).transpose(1, 2), dim=-1)
        loss = F.smooth_l1_loss(pred, tgt) + (1 - (pn * tn).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    A.eval()

    # ---- load decoders ----
    pd = torch.load(REPO / "lmwm/checkpoints/patch_decoder/patch_dec.pt", map_location=dev, weights_only=False)
    dec_d = make_decoder(pd["din"], pd["dec"]).to(dev); dec_d.load_state_dict(pd["model"] if "model" in pd else pd["dec_sd"]); dec_d.eval()
    pmu = torch.from_numpy(np.asarray(pd["mu"], np.float32)).view(1, pd["din"], 1, 1).to(dev)
    psd = torch.from_numpy(np.asarray(pd["sd"], np.float32)).view(1, pd["din"], 1, 1).to(dev)
    sig_dec = None
    for cand in ["dec.pt", "dec_gan.pt"]:
        p = REPO / "lmwm/checkpoints/siglip_decoder" / cand
        if p.exists():
            sd = torch.load(p, map_location=dev, weights_only=False)
            try:
                sig_dec = GridDecoder(sd.get("din", 1152), sd.get("res", 256)).to(dev)
                sig_dec.load_state_dict(sd["model"] if "model" in sd else sd); sig_dec.eval(); break
            except Exception as e:
                print(f"siglip dec {cand} load fail: {e}", flush=True); sig_dec = None

    def dec_dino(grid):  # grid (B,1280,16,16) -> uint8 (B,128,128,3)
        with torch.no_grad():
            y = dec_d(((grid - pmu) / psd)); y = ((y.clamp(-1, 1) + 1) * 127.5).byte()
        return y.permute(0, 2, 3, 1).cpu().numpy()

    def dec_sig(grid):
        if sig_dec is None: return None
        with torch.no_grad():
            y = sig_dec(grid); y = ((y.clamp(-1, 1) + 1) * 127.5).byte()
        out = y.permute(0, 2, 3, 1).cpu().numpy()
        return np.stack([cv2.resize(x, (128, 128)) for x in out])

    # ---- eval on held-out ----
    with torch.no_grad():
        Aout = A(Gsn[va]) * dsd + dmu
    cos = float(F.cosine_similarity(F.normalize(Aout.flatten(2).transpose(1, 2), dim=-1),
                                    F.normalize(Gd[va].flatten(2).transpose(1, 2), dim=-1), dim=-1).mean())
    orig = disp[va]
    rec_true = dec_dino(Gd[va]); rec_adpt = dec_dino(Aout); rec_sig = dec_sig(Gs[va])
    def l1(a, b): return round(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() / 255), 4)
    def shp(a): return round(float(np.mean([sharp(x) for x in a])), 1)
    res = {"tag": args.tag, "n_val": len(va), "adapter_feat_cos_sig2dino": round(cos, 4),
           "real_sharp": shp(orig),
           "decode_L1_vs_orig": {"dino_true": l1(rec_true, orig), "sandwich_A(sig)": l1(rec_adpt, orig),
                                 "siglip_native": l1(rec_sig, orig) if rec_sig is not None else None},
           "decode_sharp": {"real": shp(orig), "dino_true": shp(rec_true), "sandwich_A(sig)": shp(rec_adpt),
                            "siglip_native": shp(rec_sig) if rec_sig is not None else None},
           "sandwich_vs_dinotrue_L1": l1(rec_adpt, rec_true)}
    outp = REPO / f"lmwm/outputs/lmwm2/{args.tag}.json"; outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2), flush=True)

    # ---- visual panel ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = ["real", "SigLIP-native", "sandwich A(sig)->DINO", "DINO-true"]
    mats = [orig, rec_sig if rec_sig is not None else orig, rec_adpt, rec_true]
    k = 8; sel = rng.choice(len(va), k, replace=False)
    fig, ax = plt.subplots(len(rows), k, figsize=(2 * k, 2 * len(rows)))
    for r in range(len(rows)):
        for c in range(k):
            ax[r, c].imshow(mats[r][sel[c]]); ax[r, c].axis("off")
        ax[r, 0].set_ylabel(rows[r], fontsize=10, rotation=90); ax[r, 0].axis("on"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
    fig.suptitle(f"SigLIP->DINO adapter decode | feat-cos={cos:.3f} | L1 sandwich={res['decode_L1_vs_orig']['sandwich_A(sig)']} "
                 f"dino-true={res['decode_L1_vs_orig']['dino_true']} sig-native={res['decode_L1_vs_orig']['siglip_native']}", fontsize=11)
    fig.tight_layout(); pngp = REPO / f"lmwm/outputs/lmwm2/{args.tag}.png"; fig.savefig(pngp, dpi=85, bbox_inches="tight")
    print(f"panel -> {pngp}", flush=True)


if __name__ == "__main__":
    main()
