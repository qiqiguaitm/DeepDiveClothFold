#!/usr/bin/env python
"""Compare the two pooled decoders (dec_v2 L1 vs dec_gan_v2 GAN) on the SAME prod v2
predictions, all in the unified space. Columns:
  current(real) | PRED->L1 | PRED->GAN | TRUE medoid->L1 | TRUE medoid->GAN | TRUE medoid(real)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_prod_milestone import ProdNet, build_feat  # noqa: E402
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_dec(p, dev):
    ck = torch.load(p, map_location="cpu"); R = int(ck["res"])
    D = PooledDecoder(din=int(ck["din"]), res=R).to(dev); D.load_state_dict(ck["model"]); D.eval()
    return D, R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=1819)
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin_v2.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph_v2.npz")
    ap.add_argument("--members", default="lmwm/checkpoints/prod_milestone_v2/member_*.pt")
    ap.add_argument("--dec_l1", default="lmwm/checkpoints/dinov3h_decoder/dec_v2.pt")
    ap.add_argument("--dec_gan", default="lmwm/checkpoints/dinov3h_decoder/dec_gan_v2.pt")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h_v2")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--out", default="lmwm/docs/assets/decoder_compare_v2.png", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    DL1, R = load_dec(args.dec_l1, dev); DG, _ = load_dec(args.dec_gan, dev)
    def dec(D, lat):
        with torch.no_grad():
            o = D(torch.from_numpy(l2(np.atleast_2d(lat).astype(np.float32))).to(dev)).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    z = np.load(args.pairs); proto = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    feat = build_feat(z, proto); din = feat.shape[1]
    idx = np.where(z["episode_id"] == args.episode)[0]; idx = idx[np.argsort(z["t"][idx])]
    ts = z["t"][idx].astype(np.int64)
    med = z["next_medoid"][idx].astype(np.float32); med /= np.linalg.norm(med, axis=1, keepdims=True) + 1e-8

    protos = None; X = torch.from_numpy(feat[idx].astype(np.float32)).to(dev)
    for p in sorted(glob.glob(args.members)):
        c = torch.load(p, map_location="cpu"); m = ProdNet(din, len(proto)).to(dev); m.load_state_dict(c["model"]); m.eval()
        with torch.no_grad():
            _, pr = m(X)
        g = F.normalize(pr.float(), -1).cpu().numpy(); protos = g if protos is None else protos + g
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8

    E, FR, Fb = load_features(Path(args.feature_dir)); Fn = l2(Fb.astype(np.float32))
    qloc = np.where(E == args.episode)[0]; med_g = qloc[(Fn[qloc] @ med.T).argmax(0)]
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)

    sel = np.linspace(0, len(idx) - 1, args.rows).astype(int)
    pL, pG = dec(DL1, protos[sel]), dec(DG, protos[sel])
    tL, tG = dec(DL1, med[sel]), dec(DG, med[sel])
    titles = ["current (real)", "PRED -> L1(dec_v2)", "PRED -> GAN(dec_gan_v2)", "TRUE -> L1", "TRUE -> GAN", "TRUE medoid (real)"]
    fig, axes = plt.subplots(len(sel), 6, figsize=(6 * 2.0, len(sel) * 2.1))
    for i, k in enumerate(sel):
        imgs = [frame(args.episode, int(ts[k])), pL[i], pG[i], tL[i], tG[i], frame(int(E[med_g[k]]), int(FR[med_g[k]]))]
        for ci, im in enumerate(imgs):
            a = axes[i, ci]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(titles[ci], fontsize=8)
    for c in caps.values():
        c.release()
    fig.suptitle(f"ep{args.episode} unified-space decoder comparison | L1(dec_v2, soft) vs GAN(dec_gan_v2, sharp) on same prod-v2 predictions", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
