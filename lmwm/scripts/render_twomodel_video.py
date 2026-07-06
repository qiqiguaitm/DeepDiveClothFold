#!/usr/bin/env python
"""Render the NEW-architecture milestone+1 prediction as a video, decoded through the π0.5 SigLIP
decoder — keeps the visualization capability in the new (SigLIP) space.

Per episode frame t (panel, left->right):
  [ current frame | PREDICTED m+1  (Stage-2(G_t, MDN(g_t)) -> SigLIP-decode)
                  | REAL m+1 decoded (SigLIP-encode target medoid -> decode)  | REAL m+1 frame ]

Segmentation / m+1 TARGET stay in DINOv3-H + CRAVE Viterbi (the offline label factory); the online
predictor + decoding live entirely in π0.5 SigLIP space. So the middle two cells share ONE decoder and
isolate PREDICTION quality from decoder quality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
import cv2  # noqa: E402
from train_lawm_patch import load_index, read_imgs, ForwardDec  # noqa: E402
from crave.utils.dp import viterbi_forward  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402
from train_twomodel_poc import MDN, PI05_NPZ, PI05_NPZ_GF3  # noqa: E402
from train_siglip_decoder import GridDecoder  # noqa: E402


def bar(w, text, h=24):
    b = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(b, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=8)
    ap.add_argument("--tm_ckpt", default="lmwm/checkpoints/twomodel/milestone_viterbi_K4.pt")
    ap.add_argument("--dec_ckpt", default="lmwm/checkpoints/siglip_decoder/dec.pt")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="lmwm/outputs/twomodel_milestone_ep8.mp4")
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)

    rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = rg["prototype_table"].astype(np.float32); pord = rg["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    E, FR, Fn = load_index(args.feature_dir)
    loc = np.where(E == args.episode)[0]
    if len(loc) == 0:
        raise SystemExit(f"episode {args.episode} not in index")
    order = loc[np.argsort(FR[loc])]
    Fn_ep = Fn[order]                                                        # DINOv3-H pooled (offline, for seg)
    enc_imgs, disp = read_imgs(args.dataset_root, args.camera, E, FR, order, 224, args.cell)
    print(f"ep{args.episode}: {len(order)} frames; SigLIP encoding ...", flush=True)

    # --- Viterbi segments + per-frame m+1 target medoid (DINOv3-H / CRAVE offline) ---
    emit = np.linalg.norm(Fn_ep[:, None] - protoL[None], axis=2)
    seq = viterbi_forward(emit, pord, up=3.0, down=25.0, hard_start=True)
    ch = np.where(np.diff(seq) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
    seg_med = [s + int((Fn_ep[s:e] @ protoL[int(seq[s])]).argmax()) for s, e in zip(st, en)]
    seg_of = np.zeros(len(seq), int)
    for i, (s, e) in enumerate(zip(st, en)):
        seg_of[s:e] = i
    tgt_of = np.array([seg_med[min(seg_of[t] + 1, len(seg_med) - 1)] for t in range(len(seq))])  # m+1 medoid idx

    # --- SigLIP grids (online space) ---
    enc = SiglipBigVision(npz, device=dev)
    Graw = enc.encode_grid(enc_imgs, bs=32)                                  # (T,1152,16,16) raw

    # --- two-model + decoder ---
    tm = torch.load(args.tm_ckpt, map_location="cpu", weights_only=False)
    din, cd, gmu, gsd = tm["din"], tm["code_dim"], tm["gmu"], tm["gsd"]
    mdn = MDN(din, tm["K"]).to(dev); mdn.load_state_dict(tm["mdn"]); mdn.eval()
    fwd = ForwardDec(din, cd).to(dev); fwd.load_state_dict(tm["fwd"]); fwd.eval()
    dc = torch.load(args.dec_ckpt, map_location="cpu", weights_only=False)
    dec = GridDecoder(dc["din"], dc["res"]).to(dev); dec.load_state_dict(dc["model"]); dec.eval()
    print(f"loaded two-model K={tm['K']} + SigLIP decoder (val_L1={dc.get('val_L1')})", flush=True)

    Gn = torch.from_numpy(((Graw - gmu) / gsd).astype(np.float32)).to(dev)   # normalized for two-model
    gist = Gn.mean((2, 3))

    def decode_raw(graw_t):                                                  # (B,1152,16,16) raw -> (B,H,W,3) u8
        with torch.no_grad():
            o = dec(graw_t.to(dev)).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    T = len(order); C = args.cell
    vw = cv2.VideoWriter(str(REPO / args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (C * 4, C + 24))
    with torch.no_grad():
        for s in range(0, T, 64):
            sl = slice(s, min(s + 64, T))
            zt = mdn.deploy_mean(gist[sl])                                   # Stage-1 deploy identity
            Ghat_n = fwd(Gn[sl], zt)                                         # Stage-2 grounding (normalized)
            Ghat_raw = Ghat_n * float(gsd) + float(gmu)
            pred = decode_raw(Ghat_raw)                                      # predicted m+1 decoded
            real_dec = decode_raw(torch.from_numpy(Graw[tgt_of[sl]]))        # real m+1 (encode target->decode)
            for k, t in enumerate(range(sl.start, sl.stop)):
                row = np.concatenate([disp[t], pred[k], real_dec[k], disp[tgt_of[t]]], axis=1)[:, :, ::-1]
                labels = np.concatenate([bar(C, "current"), bar(C, "PRED m+1 (decoded)"),
                                         bar(C, "REAL m+1 (decoded)"), bar(C, "REAL m+1 frame")], axis=1)
                vw.write(np.concatenate([labels, row], axis=0))
    vw.release()
    print(f"saved {REPO / args.out}  ({T} frames)", flush=True)


if __name__ == "__main__":
    main()
