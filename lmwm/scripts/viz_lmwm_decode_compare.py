#!/usr/bin/env python
"""LMWM-specific decode comparison: decode the PRODUCTION model's PREDICTED subgoal
latent (noisier than a clean encoding) three ways, and measure which best conveys the
TRUE next-medoid -- CRAVE's self-recon winner (retrieval) is not guaranteed optimal here.

Quantitative (held-out subset), all vs the TRUE next-medoid latent:
  latent_cos          : cos(pred, true medoid)                         [ceiling, no decode]
  retrieval           : nearest real frame to PRED -> its DINOv3-H latent cos to true medoid
                        + milestone-match(retrieved, true next)        [snaps to real; can drift]
  synthetic(pooled)   : decode(pred) -> re-encode DINOv3-H -> cos to true medoid  [blurry]
Plus a visual filmstrip: current | pred->pooled-decode | pred->retrieval | true medoid real.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lmwm.data import split_indices  # noqa: E402
from train_prod_milestone import ProdNet, build_feat  # noqa: E402
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--members", default="lmwm/checkpoints/prod_milestone/member_*.pt")
    ap.add_argument("--decoder", default="lmwm/checkpoints/dinov3h_decoder/dec.pt", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_quant", type=int, default=3000)
    ap.add_argument("--viz_episode", type=int, default=1819)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--out", default="lmwm/docs/assets/lmwm_decode_compare.png", type=Path)
    ap.add_argument("--out_json", default="lmwm/outputs/lmwm_decode_compare/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    g = np.load(args.graph_npz); proto_tbl = g["prototype_table"].astype(np.float32)
    n = len(z["current_milestone"])
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = vi.numpy()
    feat = build_feat(z, proto_tbl); din = feat.shape[1]; num_m = len(proto_tbl)
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    vi = vi[ok[vi]]

    paths = sorted(glob.glob(args.members))
    def predict(rows):
        X = torch.from_numpy(feat[rows]).to(dev); protos = None
        for p in paths:
            c = torch.load(p, map_location="cpu"); m = ProdNet(din, num_m).to(dev); m.load_state_dict(c["model"]); m.eval()
            with torch.no_grad():
                _, pr = m(X)
            gg = F.normalize(pr.float(), -1).cpu().numpy(); protos = gg if protos is None else protos + gg
        return protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)

    # feature bank (real latents + frames), milestone assign of each bank frame
    E, FR, Fb = load_features(Path(args.feature_dir)); Fn = l2(Fb.astype(np.float32))
    bank_m = (Fn @ proto_tbl.T).argmax(1)
    Fn_t = torch.from_numpy(Fn).to(dev)

    # decoder + re-encoder
    ck = torch.load(args.decoder, map_location="cpu"); R = int(ck["res"])
    D = PooledDecoder(din=int(ck["din"]), res=R).to(dev); D.load_state_dict(ck["model"]); D.eval()
    def decode(lat):
        with torch.no_grad():
            o = D(torch.from_numpy(l2(np.atleast_2d(lat).astype(np.float32))).to(dev)).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    enc = load_encoder("dinov3-h", device=str(dev))

    # ---- quantitative on a held-out subset ----
    rng = np.random.default_rng(0)
    sub = rng.choice(vi, min(args.n_quant, len(vi)), replace=False)
    pred = predict(sub); tmed = med[sub]; eps = z["episode_id"][sub].astype(np.int64); fut = z["future_milestone"][sub].astype(np.int64)
    lat_cos = (pred * tmed).sum(1)
    # retrieval: nearest bank frame to pred, excluding query episode
    retr_cos = np.zeros(len(sub)); retr_match = np.zeros(len(sub), bool)
    pt = torch.from_numpy(pred.astype(np.float32)).to(dev)
    with torch.no_grad():
        for s in range(0, len(sub), 1024):
            sims = pt[s:s + 1024] @ Fn_t.T                                  # [b, N]
            for j in range(sims.shape[0]):
                mask = torch.from_numpy(E != eps[s + j]).to(dev)
                gi = int(torch.where(mask, sims[j], torch.full_like(sims[j], -2)).argmax().item())
                retr_cos[s + j] = float(Fn[gi] @ tmed[s + j]); retr_match[s + j] = bank_m[gi] == fut[s + j]
    # synthetic: decode(pred) -> re-encode -> cos to true medoid
    syn_cos = []
    for s in range(0, len(sub), 256):
        imgs = decode(pred[s:s + 256])
        re = l2(enc.encode_pooled(imgs).astype(np.float32))
        syn_cos.append((re * tmed[s:s + 256]).sum(1))
    syn_cos = np.concatenate(syn_cos)

    res = {"n": len(sub),
           "latent_cos_ceiling": round(float(lat_cos.mean()), 4),
           "retrieval": {"reencode_cos": round(float(retr_cos.mean()), 4), "milestone_match": round(float(retr_match.mean()), 4)},
           "synthetic_pooled": {"reencode_cos": round(float(syn_cos.mean()), 4)}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))

    # ---- visual filmstrip on one episode ----
    idx = np.array([i for i in np.where(z["episode_id"] == args.viz_episode)[0] if i in set(vi.tolist())])
    idx = idx[np.argsort(z["t"][idx])]
    pv = predict(idx); ts = z["t"][idx].astype(np.int64); medv = med[idx]
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)
    qloc = np.where(E == args.viz_episode)[0]
    sel = np.linspace(0, len(idx) - 1, args.rows).astype(int)
    dp = decode(pv[sel])
    with torch.no_grad():
        pt2 = torch.from_numpy(pv[sel].astype(np.float32)).to(dev)
        retr_g = []
        for j in range(len(sel)):
            sims = pt2[j] @ Fn_t.T; mask = torch.from_numpy(E != args.viz_episode).to(dev)
            retr_g.append(int(torch.where(mask, sims, torch.full_like(sims, -2)).argmax().item()))
    medframe = [qloc[(Fn[qloc] @ medv[k]).argmax()] for k in sel]
    titles = ["current (real)", "PRED->pooled decode (blurry)", "PRED->retrieval (sharp real)", "TRUE medoid (real)"]
    fig, axes = plt.subplots(len(sel), 4, figsize=(4 * 2.3, len(sel) * 2.35))
    for i, k in enumerate(sel):
        imgs = [frame(args.viz_episode, int(ts[k])), dp[i], frame(int(E[retr_g[i]]), int(FR[retr_g[i]])), frame(int(E[medframe[i]]), int(FR[medframe[i]]))]
        for ci, im in enumerate(imgs):
            a = axes[i, ci]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(titles[ci], fontsize=8)
    for c in caps.values():
        c.release()
    fig.suptitle(f"LMWM decode comparison (predicted subgoal) | latent-cos={res['latent_cos_ceiling']} | "
                 f"retrieval reencode-cos={res['retrieval']['reencode_cos']} match={res['retrieval']['milestone_match']} | "
                 f"synthetic reencode-cos={res['synthetic_pooled']['reencode_cos']}", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
