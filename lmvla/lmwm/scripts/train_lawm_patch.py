#!/usr/bin/env python
"""LaWM-style inverse/forward on DINOv3-H PATCH-GRIDS, joint training.

Mirrors LaWM's LAM: inverse(g_t, g_future) -> compact code u; forward(g_t, u) ->
reconstruct g_future. The reconstruction loss shapes the code, so a tiny code
suffices because the forward is conditioned on the current grid (spatial base) --
this is what makes it faithful where the pooled un-pool failed.

Quantifies:
  - code compactness: sweep code_dim {8,32,128}.
  - future-grid reconstruction fidelity: cos(g_hat, g_future) vs persistence
    baseline cos(g_t, g_future).
  - image fidelity: decode(g_hat) vs real future frame L1 (vs direct decode 2.7%).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import train_dec  # noqa: E402


def load_index(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    shards = sorted(s for s in feature_dir.glob("shard_*.npz") if "_bak" not in s.name)
    dim = int(np.load(shards[0])["feat"].shape[1])                       # infer feat dim (768 base / 1280 H)
    feat = np.zeros((n, dim), dtype=np.float16); valid = np.zeros(n, dtype=bool)
    for shard in shards:
        z = np.load(shard); gi = z["gidx"].astype(np.int64)
        feat[gi] = z["feat"]; valid[gi] = z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32); fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], fv


def read_imgs(dataset_root, camera, E, FR, gidx, enc_res, tgt_res):
    cs = int(json.loads((dataset_root / "meta/info.json").read_text())["chunks_size"])
    ie = np.zeros((len(gidx), enc_res, enc_res, 3), np.uint8)
    it = np.zeros((len(gidx), tgt_res, tgt_res, 3), np.uint8)
    by_ep: dict[int, list[int]] = {}
    for k, gi in enumerate(gidx):
        by_ep.setdefault(int(E[gi]), []).append(k)
    for ep, ks in by_ep.items():
        cap = cv2.VideoCapture(str(dataset_root / f"videos/chunk-{ep // cs:03d}/{camera}/episode_{ep:06d}.mp4"))
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(FR[gidx[k]]))
            ok, fr = cap.read()
            if ok:
                rgb = fr[:, :, ::-1]; ie[k] = cv2.resize(rgb, (enc_res, enc_res)); it[k] = cv2.resize(rgb, (tgt_res, tgt_res))
        cap.release()
    return ie, it


class InverseEnc(nn.Module):
    def __init__(self, din, code_dim, hid=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * din, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(),  # 16->8
            nn.Conv2d(hid, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(),      # 8->4
        )
        self.head = nn.Linear(hid, code_dim); self.ln = nn.LayerNorm(code_dim)

    def forward(self, gt, gf):
        x = self.conv(torch.cat([gt, gf], 1)).mean((2, 3))
        return self.ln(self.head(x))


class ForwardDec(nn.Module):
    def __init__(self, din, code_dim, hid=512):
        super().__init__()
        self.proj = nn.Conv2d(din + code_dim, hid, 3, 1, 1)
        self.body = nn.Sequential(
            nn.GroupNorm(8, hid), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1),
            nn.GroupNorm(8, hid), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1),
            nn.GroupNorm(8, hid), nn.GELU(),
        )
        self.out = nn.Conv2d(hid, din, 3, 1, 1)

    def forward(self, gt, code):
        c = code[:, :, None, None].expand(-1, -1, gt.shape[2], gt.shape[3])
        return self.out(self.body(self.proj(torch.cat([gt, c], 1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--code_dims", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out_dir", default="lmwm/outputs/lawm_patch", type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    dev = "cuda"

    E, FR, Fn = load_index(args.feature_dir)
    proto = np.load("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")["prototype_table"].astype(np.float32)

    # stage transitions: (cur-stage last frame gidx, next-stage medoid gidx), by episode
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    tr_pairs, va_pairs = [], []
    for ep in eps:
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        seq = (Fn[order] @ proto.T).argmax(1)
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        reps = []
        for s, e in zip(st, en):
            m = int(seq[s]); sub = order[s:e]; med = sub[(Fn[sub] @ proto[m]).argmax()]
            reps.append((int(order[e - 1]), int(med)))
        tgt = va_pairs if ep in val_eps else tr_pairs
        for i in range(len(reps) - 1):
            tgt.append((reps[i][0], reps[i + 1][1]))  # (cur rep, next medoid)
    rng.shuffle(tr_pairs); rng.shuffle(va_pairs)
    tr_pairs = tr_pairs[:args.n_train]; va_pairs = va_pairs[:args.n_val]

    # unique frames -> read + encode patch grids
    uniq = sorted(set([g for p in tr_pairs + va_pairs for g in p]))
    u2k = {g: k for k, g in enumerate(uniq)}
    print(f"{len(tr_pairs)} train + {len(va_pairs)} val transitions; {len(uniq)} unique frames", flush=True)
    enc_imgs, tgt_imgs = read_imgs(args.dataset_root, args.camera, E, FR, np.array(uniq), 256, 128)
    enc = load_encoder("dinov3-h", device=dev)
    print("encoding patch grids ...", flush=True)
    grids = enc.encode_grid(enc_imgs).astype(np.float32)  # (U,1280,16,16)
    din = grids.shape[1]
    gmu, gsd = grids.mean(), grids.std() + 1e-6
    gz = ((grids - gmu) / gsd).astype(np.float32)

    # patch decoder for image-space fidelity
    print("training patch decoder ...", flush=True)
    decode = train_dec(grids, tgt_imgs, din, dec="small", epochs=50, device=dev)

    def pairs_arr(pairs):
        a = np.array([u2k[c] for c, _ in pairs]); b = np.array([u2k[n] for _, n in pairs]); return a, b
    tra, trb = pairs_arr(tr_pairs); vaa, vab = pairs_arr(va_pairs)
    GZ = torch.from_numpy(gz)  # keep on CPU, move batches

    def batch(idx_a, idx_b, bs, dev):
        sel = torch.randint(0, len(idx_a), (bs,))
        return GZ[idx_a[sel]].to(dev), GZ[idx_b[sel]].to(dev)

    tra_t, trb_t = torch.from_numpy(tra), torch.from_numpy(trb)
    results = {}
    for cd in args.code_dims:
        inv = InverseEnc(din, cd).to(dev); fwd = ForwardDec(din, cd).to(dev)
        opt = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=2e-4, weight_decay=1e-5)
        for step in range(args.steps):
            gt, gf = batch(tra_t, trb_t, 32, dev)
            u = inv(gt, gf); gf_hat = fwd(gt, u)
            loss = F.smooth_l1_loss(gf_hat, gf, beta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        # eval on val
        inv.eval(); fwd.eval()
        cos_recon, cos_persist, l1_img = [], [], []
        with torch.no_grad():
            for s in range(0, len(vaa), 256):
                gt = GZ[vaa[s:s+256]].to(dev); gf = GZ[vab[s:s+256]].to(dev)
                u = inv(gt, gf); gf_hat = fwd(gt, u)
                # feature cos on raw (un-standardize)
                gh = (gf_hat.cpu().numpy() * gsd + gmu).reshape(len(gf_hat), -1)
                gtr = (gf.cpu().numpy() * gsd + gmu).reshape(len(gf), -1)
                gcur = (gt.cpu().numpy() * gsd + gmu).reshape(len(gt), -1)
                def cos(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
                cos_recon.append(cos(gh, gtr)); cos_persist.append(cos(gcur, gtr))
                # image: decode reconstructed grid vs real future frame
                grid_raw = (gf_hat.cpu().numpy() * gsd + gmu).astype(np.float32)
                rec_img = decode(grid_raw)
                real_img = tgt_imgs[vab[s:s+256]]
                l1_img.append(np.abs(real_img.astype(float) - rec_img.astype(float)).mean(axis=(1, 2, 3)))
        results[str(cd)] = {
            "code_dim": cd, "code_bits_vs_grid": round(cd / (din * 16 * 16), 6),
            "recon_grid_cos": round(float(np.concatenate(cos_recon).mean()), 4),
            "persistence_grid_cos": round(float(np.concatenate(cos_persist).mean()), 4),
            "recon_image_L1_frac": round(float(np.concatenate(l1_img).mean()) / 255, 4),
        }
        print(f"code_dim={cd}: recon_cos={results[str(cd)]['recon_grid_cos']} "
              f"persist_cos={results[str(cd)]['persistence_grid_cos']} "
              f"img_L1={results[str(cd)]['recon_image_L1_frac']}", flush=True)

    summary = {
        "n_train": len(tr_pairs), "n_val": len(va_pairs), "din": din,
        "direct_patch_decode_L1_frac": 0.027, "pooled_decode_L1_frac": 0.062,
        "note": "recon = forward(g_t, inverse(g_t,g_future)); code compact; conditioned on current grid",
        "by_code_dim": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
