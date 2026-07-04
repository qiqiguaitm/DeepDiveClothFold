#!/usr/bin/env python
"""Best LMWM decoder = conditional flow-matching pixel decoder (dec_best.pt, flow_b160).

Decodes a pooled DINOv3-H latent (1280-D, L2-normalized) into a SHARP, FAITHFUL image via
rectified-flow ODE sampling. Beats L1 (blurry conditional mean, reencode_cos 0.35) and GAN
(sharp but hallucinates, 0.41): flow reaches reencode_cos 0.68 while staying sharp, and
because it samples from the learned image distribution it cannot produce adversarial garbage.

Use as a library:
    from decode_best import load_best_decoder
    dec = load_best_decoder("lmwm/checkpoints/dinov3h_decoder/dec_best.pt", "cuda:0")
    imgs = dec(latents)          # (N,1280) float, L2-normed -> (N,res,res,3) uint8 RGB

CLI demo (decode a few bank frames vs their reals):
    python lmwm/scripts/decode_best.py --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_decoder_gf3 import UNet  # noqa: E402


class BestDecoder:
    """Flow-matching pixel decoder. Callable: (N,1280) L2-normed latents -> (N,res,res,3) uint8."""

    def __init__(self, ckpt_path: str, device: str = "cuda:0", ode_steps: int = 25):
        ck = torch.load(ckpt_path, map_location="cpu")
        self.res = int(ck["res"]); self.ode_steps = ode_steps
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net = UNet(1280, ck["base"]).to(self.dev); self.net.load_state_dict(ck["model"]); self.net.eval()
        self.step = ck.get("step")

    @torch.no_grad()
    def __call__(self, latents, ode_steps: int | None = None,
                 fixed_noise: "torch.Tensor | None" = None, seed: int | None = None) -> np.ndarray:
        """Decode latents. For TEMPORALLY STABLE / DETERMINISTIC output (e.g. per-frame milestone
        hints where consecutive latents are ~equal), pass seed=<int> (or fixed_noise): the ODE
        starts from ONE shared noise so identical latents -> identical images and nearby latents ->
        nearby images (no per-frame flicker). Default (both None) = fresh noise per row = generative."""
        n_steps = ode_steps or self.ode_steps
        lat = latents if torch.is_tensor(latents) else torch.from_numpy(np.asarray(latents, np.float32))
        lat = lat.float().to(self.dev)
        lat = lat / (lat.norm(dim=-1, keepdim=True) + 1e-8)
        if fixed_noise is not None:
            x = fixed_noise.to(self.dev).reshape(1, 3, self.res, self.res).expand(len(lat), -1, -1, -1).clone()
        elif seed is not None:
            g = torch.Generator(device=self.dev).manual_seed(int(seed))
            x = torch.randn(1, 3, self.res, self.res, device=self.dev, generator=g).expand(len(lat), -1, -1, -1).clone()
        else:
            x = torch.randn(len(lat), 3, self.res, self.res, device=self.dev)
        for k in range(n_steps):
            t = torch.full((len(lat),), k / n_steps, device=self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = self.net(x, t, lat)
            x = x + (1.0 / n_steps) * v.float()
        return np.clip((x.cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)


def load_best_decoder(ckpt_path: str = "lmwm/checkpoints/dinov3h_decoder/dec_best.pt",
                      device: str = "cuda:0", ode_steps: int = 25) -> BestDecoder:
    return BestDecoder(ckpt_path, device, ode_steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lmwm/checkpoints/dinov3h_decoder/dec_best.pt")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h_v2")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="lmwm/docs/assets/decode_best_demo.png")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from train_dinov3h_decoder import load_features, l2  # noqa: E402

    dec = load_best_decoder(args.ckpt, args.device)
    E, FR, Fb = load_features(Path(args.feature_dir)); Fn = l2(Fb.astype(np.float32))
    rng = np.random.default_rng(7); sel = rng.choice(len(E), args.n, replace=False)
    imgs = dec(Fn[sel])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, args.n, figsize=(args.n * 1.5, 1.8))
    for j in range(args.n):
        ax[j].imshow(imgs[j]); ax[j].axis("off")
    fig.suptitle(f"dec_best (flow_b160) decode | ckpt step {dec.step} res {dec.res}", fontsize=9)
    fig.tight_layout(); Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"decoded {args.n} latents -> {args.out} (res {dec.res}, ODE {dec.ode_steps} steps)")


if __name__ == "__main__":
    main()
