#!/usr/bin/env python
"""Visualize the production milestone+1 model on a held-out episode.

Ensemble (pooled MLP + H1/H3/H4) predicts next milestone (fused) + subgoal latent.
Columns per row: current frame | PRED subgoal -> pooled decode (option-2 synthetic)
| TRUE next-medoid -> pooled decode (decoder ceiling) | TRUE next-medoid real frame.
Annotated with pred/true milestone, top1/top5 hit, subgoal cos.
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
from lmwm.data import split_indices  # noqa: E402
from train_prod_milestone import ProdNet, build_feat  # noqa: E402
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=1819)
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--members", default="lmwm/checkpoints/prod_milestone/member_*.pt")
    ap.add_argument("--decoder", default="lmwm/checkpoints/dinov3h_decoder/dec.pt", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--out", default="lmwm/docs/assets/prod_milestone_ep.png", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.decoder, map_location="cpu"); R = int(ck["res"])
    D = PooledDecoder(din=int(ck["din"]), res=R).to(dev); D.load_state_dict(ck["model"]); D.eval()
    def decode(lat):
        with torch.no_grad():
            o = D(torch.from_numpy(l2(np.atleast_2d(lat).astype(np.float32))).to(dev)).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    z = np.load(args.pairs)
    proto_tbl = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    n = len(z["current_milestone"])
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = set(vi.numpy().tolist())
    feat = build_feat(z, proto_tbl); din = feat.shape[1]
    num_m = int(z["future_milestone"].max()) + 1
    idx = np.array([i for i in np.where(z["episode_id"] == args.episode)[0] if i in vi])
    idx = idx[np.argsort(z["t"][idx])]
    ts = z["t"][idx].astype(np.int64); fut = z["future_milestone"][idx].astype(np.int64)
    cur_m = z["current_milestone"][idx].astype(np.int64)
    med = z["next_medoid"][idx].astype(np.float32); med /= np.linalg.norm(med, axis=1, keepdims=True) + 1e-8

    # ensemble predict
    paths = sorted(glob.glob(args.members)); probs = None; protos = None
    X = torch.from_numpy(feat[idx]).to(dev)
    for p in paths:
        c = torch.load(p, map_location="cpu"); m = ProdNet(din, num_m).to(dev); m.load_state_dict(c["model"]); m.eval()
        with torch.no_grad():
            lg, pr = m(X)
        pp = F.softmax(lg.float(), -1).cpu().numpy(); gg = F.normalize(pr.float(), -1).cpu().numpy()
        probs = pp if probs is None else probs + pp; protos = gg if protos is None else protos + gg
    probs /= len(paths); protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
    trans = np.load(args.graph_npz)["transition_probs"].astype(np.float64); trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior = trans[cur_m]
    lp = (1 - args.lam) * np.log(np.clip(probs, 1e-12, 1)) + args.lam * np.log(np.clip(prior, 1e-12, 1))
    pred_m = lp.argmax(1); rank = (np.argsort(-lp, 1) == fut[:, None]).argmax(1)
    cos = (protos * med).sum(1)

    # real medoid frame via feature bank retrieval (in-episode)
    E, FR, Fb = load_features(Path(args.feature_dir)); Fn = l2(Fb.astype(np.float32))
    qloc = np.where(E == args.episode)[0]; med_g = qloc[(Fn[qloc] @ med.T).argmax(0)]
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if ok else np.zeros((R, R, 3), np.uint8)

    sel = np.linspace(0, len(idx) - 1, args.rows).astype(int)
    dp = decode(protos[sel]); dm = decode(med[sel])
    titles = ["current (real)", "PRED subgoal->decode", "TRUE medoid->decode (ceiling)", "TRUE medoid (real)"]
    fig, axes = plt.subplots(len(sel), 4, figsize=(4 * 2.3, len(sel) * 2.35))
    for i, k in enumerate(sel):
        imgs = [frame(args.episode, int(ts[k])), dp[i], dm[i], frame(int(E[med_g[k]]), int(FR[med_g[k]]))]
        for ci, im in enumerate(imgs):
            a = axes[i, ci]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(titles[ci], fontsize=8)
        good = rank[k] == 0; ok5 = rank[k] < 5
        axes[i, 1].set_ylabel(f"m{cur_m[k]}->m{fut[k]}\npred m{pred_m[k]} {'OK' if good else ('top5' if ok5 else 'x')}\ncos={cos[k]:.2f}", fontsize=6.5)
        for s in axes[i, 1].spines.values():
            s.set_color("#2ca02c" if good else ("#ff7f0e" if ok5 else "#d62728")); s.set_linewidth(2.5)
    acc = float((rank == 0).mean()); a5 = float((rank < 5).mean())
    for c in caps.values():
        c.release()
    fig.suptitle(f"ep{args.episode} (held-out) prod milestone+1 | top1={acc:.0%} top5={a5:.0%} mean cos={cos.mean():.3f} "
                 f"| green=top1 orange=top5 red=miss", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"saved {args.out} | ep{args.episode}: top1={acc:.3f} top5={a5:.3f} cos={cos.mean():.4f}")


if __name__ == "__main__":
    main()
