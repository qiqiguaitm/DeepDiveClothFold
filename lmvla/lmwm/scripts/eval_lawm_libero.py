#!/usr/bin/env python
"""Official LaWM LAM (jialei02/lawam_lam, ViT-B DINOv3-base 768D, code32 VAE) evaluated ON LIBERO-Long
-- LaWM's IN-DISTRIBUTION arena. Same 768D space as our LMWM (both = facebook/dinov3-vitb16-lvd1689m),
so oracle/persistence/lift are directly comparable to train_multitask.py's LMWM eval.

LaWM native target = FIXED-HORIZON future (+1.6s). Metrics (each vs its own native target):
    oracle      = cos(LAM.recon(cur,fut), true_future)      teacher-forced (sees future)
    deploy      = cos(LAM.decoder(dec_in, predm(dec_in)), true_future)   deploy (predm, no future)
    persistence = cos(dec_in, true_future)                  copy-current
    lift_oracle = oracle - persistence ; lift_deploy = deploy - persistence

LIBERO videos are AV1 (cv2 can't decode) -> read from pre-decoded frame_cache npy.
Run in CRAVE env: /home/tim/miniconda3/envs/srpo/bin/python
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
sys.path.insert(0, str(REPO / "lmwm/vendor/LaWAM"))
from train_lawm_patch import load_index  # noqa: E402
from eval_lawm_lam import _stub_heavy_deps, _patch_automodel, load_lam, imagenet  # noqa: E402
from measure_lawm_lag import LawmPredM  # noqa: E402

ARENA = {}   # set in main(): {kind, root, camera, fcache} — supports LIBERO(frame_cache) + kai0(cv2 mp4)


def read_libero(E, FR, gidx, res=256):
    """gidx -> (N,res,res,3) uint8 RGB. Dispatches on ARENA globals set in main():
    frames_kind='libero' -> frame_cache npy (AV1 videos); 'kai0' -> cv2 mp4 via read_imgs."""
    import cv2
    from collections import defaultdict
    if ARENA["kind"] == "kai0":                                     # non-AV1 mp4, use train_lawm_patch reader
        from train_lawm_patch import read_imgs
        ie, _ = read_imgs(ARENA["root"], ARENA["camera"], E, FR, np.asarray(gidx), res, res)
        return ie
    if ARENA["kind"] == "lerobotv3":                                # coffee/aloha: concat AV1 video via pyav
        from train_multitask import _read_lerobotv3
        ie, _ = _read_lerobotv3(ARENA["root"], ARENA["camera"], E, FR, np.asarray(gidx), res, res)
        return ie
    out = np.zeros((len(gidx), res, res, 3), np.uint8)              # libero frame_cache
    by_ep = defaultdict(list)
    for k, g in enumerate(gidx):
        by_ep[int(E[g])].append((k, int(FR[g])))
    for ep, items in by_ep.items():
        fp = ARENA["fcache"] / f"chunk-{ep // 1000:03d}/{ARENA['camera']}/episode_{ep:06d}.npy"
        if not fp.exists():
            continue
        arr = np.load(fp)
        for k, fr in items:
            if fr < len(arr):
                out[k] = arr[fr] if res == arr.shape[1] else cv2.resize(arr[fr], (res, res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lmwm/vendor/LaWAM/ckpts_dl/checkpoints/pytorch_model.pt")
    ap.add_argument("--yaml", default="lmwm/vendor/LaWAM/ckpts_dl/dino_large_vae.yaml")
    ap.add_argument("--feature_dir", default="crave/data/libero10_dinov3base", type=Path)
    ap.add_argument("--frames_kind", default="libero", choices=["libero", "kai0", "lerobotv3"])
    ap.add_argument("--dataset_root", default="/vePFS/tim/workspace/LIBERO_fastwam/libero_10_no_noops_lerobot")
    ap.add_argument("--camera", default="observation.images.image")
    ap.add_argument("--arena_name", default="libero10")
    ap.add_argument("--horizon_s", type=float, default=1.6)
    ap.add_argument("--fps", type=float, default=20.0)         # LIBERO 20fps ; kai0 30fps
    ap.add_argument("--n_train_pairs", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="lmwm/outputs/eval_lawm_libero.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    ARENA.update(kind=args.frames_kind, root=Path(args.dataset_root), camera=args.camera,
                 fcache=Path(args.dataset_root) / "frame_cache/resize_256x256")

    _stub_heavy_deps(); _patch_automodel()
    lam = load_lam(args.ckpt, args.yaml, dev)
    for p in lam.parameters():
        p.requires_grad_(False)
    gap = int(round(args.horizon_s * args.fps))
    E, FR, _ = load_index(REPO / args.feature_dir)

    @torch.no_grad()
    def encode_decin(frames_u8):                              # (N,256,256,3)u8 -> dec_in (N,1,K,D)
        X = imagenet(frames_u8, dev); vid = torch.stack([X, X], dim=1)
        return lam.get_latent_action(videos=vid, states=None, dec_videos=vid, predict_future_frame=False)["dec_in"]

    # ---- fixed-horizon pairs (train + val), split by episode ----
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    tr, va = [], []
    for ep in eps:
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]; fr = FR[order]
        for i in range(len(order)):
            j = int(np.argmin(np.abs(fr - (fr[i] + gap))))
            if j > i and abs(fr[j] - fr[i] - gap) <= gap // 2:
                (va if ep in val_eps else tr).append((int(order[i]), int(order[j])))
    rng.shuffle(tr); rng.shuffle(va)
    tr = tr[:args.n_train_pairs]; va = va[:args.n_val]
    print(f"gap={gap}f ({args.horizon_s}s@{args.fps}fps) | {len(tr)} train + {len(va)} val pairs", flush=True)

    # ---- encode all needed frames via LAM's frozen ViT-B (dec_in feats) ----
    uniq = np.array(sorted(set([g for p in tr + va for g in p]))); u2k = {g: k for k, g in enumerate(uniq)}
    print(f"encoding {len(uniq)} frames (LAM ViT-B) ...", flush=True)
    imgs = read_libero(E, FR, uniq, 256)
    feats = []
    for s in range(0, len(uniq), 64):
        feats.append(encode_decin(imgs[s:s + 64]).cpu())
    feats = torch.cat(feats)                                  # (U,1,K,D)
    D = feats.shape[-1]; cdim = getattr(getattr(lam, "vq", None), "code_dim", 32)
    tca = np.array([u2k[c] for c, _ in tr]); tfa = np.array([u2k[f] for _, f in tr])
    vca = np.array([u2k[c] for c, _ in va]); vfa = np.array([u2k[f] for _, f in va])

    # ---- train deploy-predm (LAM frozen), reconstruction loss (mirrors our predm) ----
    predm = LawmPredM(D, cdim).to(dev)
    opt = torch.optim.AdamW(predm.parameters(), lr=2e-4, weight_decay=1e-5)
    print(f"training deploy-predm (code_dim={cdim}) ...", flush=True)
    for step in range(args.steps):
        sel = np.random.randint(0, len(tca), 16)
        din = feats[tca[sel]].to(dev); tgt = feats[tfa[sel]].to(dev)
        recon = lam.decoder(features=din, actions=predm(din))
        loss = F.smooth_l1_loss(recon, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  predm step {step+1}/{args.steps} loss {loss.item():.4f}", flush=True)
    predm.eval()

    # ---- eval on val ----
    def cflat(a, b):
        a = a.reshape(a.shape[0], -1); b = b.reshape(b.shape[0], -1)
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    oc, dp, ps = [], [], []
    with torch.no_grad():
        for s in range(0, len(vca), 64):
            din = feats[vca[s:s+64]].to(dev); tgt = feats[vfa[s:s+64]].to(dev)
            # oracle: teacher-forced recon (uses future)
            X = None
            vid_c = imagenet(read_libero(E, FR, uniq[vca[s:s+64]], 256), dev)
            vid_f = imagenet(read_libero(E, FR, uniq[vfa[s:s+64]], 256), dev)
            vid = torch.stack([vid_c, vid_f], dim=1)
            out = lam.get_latent_action(videos=vid, states=None, dec_videos=vid, predict_future_frame=True)
            recon = out["recon"].float().cpu().numpy(); tg = out["tgt"].float().cpu().numpy()
            dci = out["dec_in"].float().cpu().numpy()
            oc.append(cflat(recon, tg)); ps.append(cflat(dci, tg))
            # deploy: predm code (no future)
            rec_d = lam.decoder(features=din, actions=predm(din)).float().cpu().numpy()
            dp.append(cflat(rec_d, tgt.cpu().numpy()))
    oracle = float(np.concatenate(oc).mean()); persist = float(np.concatenate(ps).mean())
    deploy = float(np.concatenate(dp).mean())

    res = {
        "model": "Official LaWM LAM (jialei02/lawam_lam, ViT-B dinov3-base, code32 VAE) + our deploy-predm",
        "arena": args.arena_name, "frames_kind": args.frames_kind,
        "space": "dinov3-base 768D (facebook/dinov3-vitb16-pretrain-lvd1689m) = SAME as LMWM",
        "target": "fixed-horizon future (+%.1fs)" % args.horizon_s,
        "horizon_s": args.horizon_s, "code_dim": int(cdim), "n_val": len(va),
        args.arena_name: {
            "oracle": round(oracle, 4), "deploy": round(deploy, 4), "persistence": round(persist, 4),
            "lift_oracle": round(oracle - persist, 4), "lift_deploy": round(deploy - persist, 4),
        },
    }
    outp = REPO / args.out; outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
