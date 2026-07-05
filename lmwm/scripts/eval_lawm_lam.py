#!/usr/bin/env python
"""Evaluate the OFFICIAL LaWM LAM (RLinf/LaWAM ckpt jialei02/lawam_lam) on OUR data.

oracle grid-cos = cos(predicted future feature, true future feature) at the 1.6s horizon, in the
LAM's own DINOv3 ViT-B/16 layer(-2) LN-normed space. Directly comparable in spirit to our LMWM
oracle (0.79, ViT-H+ space). --raw_video runs the cross-dataset (vis_base) test.

Because gf3's transformers (4.53.2) can't build dinov3, we monkeypatch transformers.AutoModel with
our pure-torch DINOv3ViTStandalone (ViT-B is non-gated; the DINOv3Encoder LN is affine=False and we
replicate hidden_states[-2]). The 2.9GB ckpt bundles the frozen ViT-B weights under transformers key
names, so we load the LAM's TRAINED parts (encoder/decoder/vq/state_decoder) with strict=False and
serve the frozen encoder from the standalone. videos=dec_videos=[current, future] per the LAM API.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
sys.path.insert(0, str(REPO / "lmwm/vendor/LaWAM"))
import cv2  # noqa: E402
from crave.encoders._dino_vit_standalone import DINOv3ViTStandalone  # noqa: E402
from train_lawm_patch import load_index, read_imgs  # noqa: E402

VITB_DIRS = ["/vePFS-North-E/shared_data/shock/.CACHE/hf_cache/hub/dinov3-vitb16-pretrain-lvd1689m",
             "/vePFS/shock/.CACHE/hf_cache/hub/dinov3-vitb16-pretrain-lvd1689m"]


class _HFOut:
    def __init__(self, last, hs):
        self.last_hidden_state = last; self.hidden_states = tuple(hs)


class StandaloneDinoAM(nn.Module):
    """Mimics a transformers DINOv3 model over the DINOv3Encoder interface, backed by our standalone."""
    def __init__(self, model_dir):
        super().__init__()
        self.m = DINOv3ViTStandalone(model_dir)
        self.config = SimpleNamespace(hidden_size=self.m.hidden, num_hidden_layers=len(self.m.layer))

    def forward(self, pixel_values, output_hidden_states=False):
        last, hs = self.m(pixel_values, return_all_hidden=True)
        return _HFOut(last, hs)


def _stub_heavy_deps():
    """Stub lightning/wandb so `latent_action_model.core.__init__` (imports the Lightning trainer
    module) resolves. We only need LatentLAMModel for inference, never the LightningModule."""
    import types

    class _AnyMod(types.ModuleType):
        def __getattr__(self, name):
            return type(name, (), {})
    for name in ["lightning", "lightning.pytorch", "lightning.pytorch.callbacks",
                 "lightning.pytorch.loggers", "lightning.pytorch.utilities", "wandb"]:
        if name not in sys.modules:
            sys.modules[name] = _AnyMod(name)
    sys.modules["lightning"].LightningModule = object


def _patch_automodel():
    import latent_action_model.core.vjepa_encoder as ve
    vitb = next((d for d in VITB_DIRS if Path(d).exists()), None)
    if vitb is None:
        raise SystemExit("DINOv3 ViT-B/16 weights not found in VITB_DIRS")

    class FakeAM:
        @staticmethod
        def from_pretrained(model_id, **kw):
            return StandaloneDinoAM(vitb)
    ve.AutoModel = FakeAM
    return vitb


def load_lam(ckpt, yaml_path, device):
    import yaml
    from latent_action_model.core.lam_model import LatentLAMModel
    cfg = yaml.safe_load(open(yaml_path))["model"]
    cfg = {k: v for k, v in cfg.items() if k != "ar_prediction"}
    model = LatentLAMModel(**cfg)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {k.replace("lam.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    trained = ("encoder.", "decoder.", "vq.", "state_decoder.")
    bad = [k for k in missing if k.startswith(trained)]
    if bad:
        raise RuntimeError(f"LAM trained-module keys missing from ckpt ({len(bad)}): {bad[:8]}")
    print(f"LAM loaded: {len(missing)} missing (vision_encoder, expected), {len(unexpected)} unexpected (ckpt ViT-B keys)", flush=True)
    return model.to(device).eval()


def imagenet(frames_u8, device):
    x = torch.from_numpy(np.asarray(frames_u8)).to(device).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def kai0_pairs(feature_dir, dataset_root, camera, horizon_s, fps, n_pairs, seed):
    E, FR, _ = load_index(feature_dir)
    rng = np.random.default_rng(seed); gap = int(round(horizon_s * fps))
    cur, fut = [], []
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]; fr = FR[order]
        for i in range(len(order)):
            tgtfr = fr[i] + gap; j = int(np.argmin(np.abs(fr - tgtfr)))
            if j > i and abs(fr[j] - tgtfr) <= gap // 2:
                cur.append(int(order[i])); fut.append(int(order[j]))
    idx = rng.permutation(len(cur))[:n_pairs]
    cur = np.array(cur)[idx]; fut = np.array(fut)[idx]
    uniq = sorted(set(cur.tolist() + fut.tolist())); u2k = {g: k for k, g in enumerate(uniq)}
    imgs, _ = read_imgs(dataset_root, camera, E, FR, np.array(uniq), 256, 256)
    return imgs, np.array([u2k[c] for c in cur]), np.array([u2k[f] for f in fut])


def rawvideo_pairs(path, horizon_s, n_pairs, seed):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(im[:, :, ::-1], (256, 256)))
    cap.release()
    gap = int(round(horizon_s * fps)); rng = np.random.default_rng(seed)
    pairs = [(i, i + gap) for i in range(len(frames) - gap)]
    rng.shuffle(pairs); pairs = pairs[:n_pairs]
    imgs = np.stack(frames).astype(np.uint8)
    return imgs, np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]), fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lmwm/vendor/LaWAM/ckpts/pytorch_model.pt")
    ap.add_argument("--yaml", default="lmwm/vendor/LaWAM/ckpts/dino_large_vae.yaml")
    ap.add_argument("--raw_video", default="")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--horizon_s", type=float, default=1.6)
    ap.add_argument("--fps", type=float, default=30.0, help="kai0 source video fps")
    ap.add_argument("--n_pairs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    _stub_heavy_deps()
    _patch_automodel()
    lam = load_lam(args.ckpt, args.yaml, dev)

    if args.raw_video:
        imgs, ci, fi, fps = rawvideo_pairs(args.raw_video, args.horizon_s, args.n_pairs, args.seed)
        src = f"raw:{Path(args.raw_video).name} fps={fps:.1f}"
    else:
        imgs, ci, fi = kai0_pairs(args.feature_dir, args.dataset_root, args.camera, args.horizon_s, args.fps, args.n_pairs, args.seed)
        src = f"kai0 fps={args.fps}"
    print(f"{src}: {len(ci)} pairs, horizon {args.horizon_s}s", flush=True)

    def cos(a, b):
        a = a.reshape(a.shape[0], -1); b = b.reshape(b.shape[0], -1)
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)

    oc, pc = [], []
    X = imagenet(imgs, dev)
    for s in range(0, len(ci), 32):
        cb = ci[s:s + 32]; fb = fi[s:s + 32]
        vid = torch.stack([X[cb], X[fb]], dim=1)                    # (B,2,3,256,256) = [current, future]
        out = lam.get_latent_action(videos=vid, states=None, dec_videos=vid, predict_future_frame=True)
        recon = out["recon"].float().cpu().numpy()                 # predicted future feat (B,1,K,D)
        tgt = out["tgt"].float().cpu().numpy(); dec_in = out["dec_in"].float().cpu().numpy()
        oc.append(cos(recon, tgt)); pc.append(cos(dec_in, tgt))
    oracle = float(np.concatenate(oc).mean()); persist = float(np.concatenate(pc).mean())
    print(f"\n=== LaWM LAM on {src} (horizon {args.horizon_s}s) ===", flush=True)
    print(f"oracle grid-cos (recon vs true future) = {oracle:.4f}", flush=True)
    print(f"persistence grid-cos (current vs future) = {persist:.4f}", flush=True)
    print(f"lift over persistence = {oracle - persist:+.4f}", flush=True)
    print(f"[compare: our LMWM oracle ~0.79 (ViT-H+ space); note different encoder space]", flush=True)


if __name__ == "__main__":
    main()
