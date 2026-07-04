#!/usr/bin/env python
"""vis_base milestone+1 video @30Hz, 4-panel appearance-gap comparison (zero re-training on vis).

  real (native)                        | current milestone from CRAVE Viterbi
  PRED absolute -> flow                 | prod LMWM proto head; bakes in kai0 appearance
  PRED forward-from-current -> flow     | fwd(current_vis, predictor(obs)); inherits vis appearance
  TRUE milestone+1 -> flow              | CRAVE per-stage medoid (a vis frame) = ground truth

The forward path (inverse/forward/predictor) is trained on the unified-v2 kai0 pairs so its output
lives in the flow decoder's space; because `current_vis` is an input, the decode stays on the vis
(black-garment) appearance manifold while advancing the structure to milestone+1.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lmwm.data import split_indices  # noqa: E402
from train_prod_milestone import ProdNet  # noqa: E402
from make_prod_video_vis import viterbi_assign  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from decode_best import load_best_decoder  # noqa: E402
from make_episode_native_video import label  # noqa: E402


class MLP(nn.Module):
    def __init__(self, din, dout, hid=512, l2=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, dout))
        self.l2 = l2

    def forward(self, x):
        o = self.net(x)
        return F.normalize(o, dim=-1) if self.l2 else o


def train_forward(pairs, dev, code_dim, steps, save_path):
    """inverse(cur,next)->code ; forward(cur,code)->next ; predictor(obs)->code. Pooled L2-normed
    to match vis-inference (unit-norm) latents. Cached to save_path."""
    if save_path.exists():
        c = torch.load(save_path, map_location="cpu"); cd = c["code_dim"]
        fwd = MLP(1280 + cd, 1280, l2=True).to(dev); fwd.load_state_dict(c["fwd"]); fwd.eval()
        predm = MLP(1294, cd).to(dev); predm.load_state_dict(c["predm"]); predm.eval()
        print(f"loaded cached forward model {save_path}", flush=True)
        return fwd, predm
    z = np.load(pairs)
    pooled = z["current"][:, :1280].astype(np.float32)
    pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8)
    state = z["current"][:, -14:].astype(np.float32)
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    n = len(z["current_milestone"]); tri, _ = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    tri = tri.numpy(); tri = tri[ok[tri]]
    feat = np.concatenate([pooled, state], 1).astype(np.float32)
    Xp = torch.from_numpy(pooled); Md = torch.from_numpy(med); Ff = torch.from_numpy(feat)
    inv = MLP(2560, code_dim).to(dev); fwd = MLP(1280 + code_dim, 1280, l2=True).to(dev)
    predm = MLP(feat.shape[1], code_dim).to(dev)
    o1 = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=5e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(predm.parameters(), lr=5e-4, weight_decay=1e-5)
    print("training forward-from-current on kai0 v2 pairs ...", flush=True)
    for s in range(steps):
        bi = tri[np.random.randint(0, len(tri), 1024)]
        cur = Xp[bi].to(dev); nxt = Md[bi].to(dev); ft = Ff[bi].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            code = inv(torch.cat([cur, nxt], -1)); rec = fwd(torch.cat([cur, code], -1))
            l1 = (1 - (rec * nxt).sum(-1)).mean()
        o1.zero_grad(); l1.backward(); o1.step()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec2 = fwd(torch.cat([cur, predm(ft)], -1)); l2 = (1 - (rec2 * nxt).sum(-1)).mean()
        o2.zero_grad(); l2.backward(); o2.step()
    fwd.eval(); predm.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vi = tri[np.random.randint(0, len(tri), 8000)]
        oc = fwd(torch.cat([Xp[vi].to(dev), predm(Ff[vi].to(dev))], -1)).float()
        cos = float((oc * Md[vi].to(dev)).sum(-1).mean())
    print(f"forward predictor kai0 subgoal cos = {cos:.4f}", flush=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fwd": fwd.state_dict(), "predm": predm.state_dict(), "code_dim": code_dim}, save_path)
    return fwd, predm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--root", default="kai0/data/Task_A/vis_base/v4/2026-04-23-v4")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph_v2.npz")
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin_v2.npz")
    ap.add_argument("--members", default="lmwm/checkpoints/prod_milestone_v2/member_*.pt")
    ap.add_argument("--flow", default="lmwm/checkpoints/dinov3h_decoder/dec_best.pt")
    ap.add_argument("--fwd_ckpt", default="lmwm/checkpoints/fwd_from_current/fwd_predm_v2.pt", type=Path)
    ap.add_argument("--code_dim", type=int, default=64)
    ap.add_argument("--fwd_steps", type=int, default=8000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--stay", type=float, default=0.9)
    ap.add_argument("--max_frames", type=int, default=3000)
    ap.add_argument("--out", default="lmwm/docs/assets/visbase_milestone_pred_fwd.mp4", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    g = np.load(args.graph_npz)
    proto_raw = g["prototype_table"].astype(np.float32)
    proto_n = proto_raw / (np.linalg.norm(proto_raw, axis=1, keepdims=True) + 1e-8)
    trans = g["transition_probs"].astype(np.float64); trans = trans / trans.sum(1, keepdims=True).clip(1e-12)

    fwd, predm = train_forward(args.pairs, dev, args.code_dim, args.fwd_steps, args.fwd_ckpt)
    flow = load_best_decoder(args.flow, str(dev))

    vids = ([args.video] if args.video else
            sorted(glob.glob(str(Path(args.root) / f"videos/chunk-*/{args.camera}/episode_*.mp4")))
            or sorted(glob.glob(str(Path(args.root) / f"*/videos/chunk-*/{args.camera}/episode_*.mp4"))))
    vid = vids[0]; print(f"episode: {vid}", flush=True)
    cap = cv2.VideoCapture(vid); frames = []
    while len(frames) < args.max_frames:
        okf, im = cap.read()
        if not okf:
            break
        frames.append(im[:, :, ::-1])
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    out_fps = round(fps) if fps and fps > 1 else 30
    N = len(frames); kt = np.arange(0, N, args.stride)
    print(f"{N} frames, {len(kt)} keyframes", flush=True)

    enc = load_encoder("dinov3-h", device=str(dev))
    lat = enc.encode_pooled(np.stack([cv2.resize(frames[t], (256, 256)) for t in kt])).astype(np.float32)
    latn = lat / (np.linalg.norm(lat, axis=1, keepdims=True) + 1e-8)

    ms = viterbi_assign(latn, proto_n, trans, args.beta, args.stay)
    ch = np.where(np.diff(ms) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(ms)]])
    stage_m = [int(ms[s]) for s in st]
    medoid = [latn[s + int((latn[s:e] @ proto_n[ms[s]]).argmax())] for s, e in zip(st, en)]
    stage_of = np.zeros(len(kt), int)
    for si, (s, e) in enumerate(zip(st, en)):
        stage_of[s:e] = si
    true_next = np.stack([medoid[min(stage_of[k] + 1, len(medoid) - 1)] for k in range(len(kt))])

    # (a) absolute PRED: prod LMWM proto head
    prev_lat = np.zeros((len(kt), 1280), np.float32); cur_lat = np.zeros((len(kt), 1280), np.float32)
    for k in range(len(kt)):
        si = stage_of[k]; cur_lat[k] = proto_raw[stage_m[si]]
        if si > 0:
            prev_lat[k] = proto_raw[stage_m[si - 1]]
    Xprod = torch.from_numpy(np.concatenate([latn, prev_lat, cur_lat, np.zeros((len(kt), 14), np.float32)], 1)).to(dev)
    pred_abs = None
    for p in sorted(glob.glob(args.members)):
        c = torch.load(p, map_location="cpu"); m = ProdNet(c["din"], c["num_m"]).to(dev)
        m.load_state_dict(c["model"]); m.eval()
        with torch.no_grad():
            _, pr = m(Xprod)
        gg = F.normalize(pr.float(), -1).cpu().numpy(); pred_abs = gg if pred_abs is None else pred_abs + gg
    pred_abs /= (np.linalg.norm(pred_abs, axis=1, keepdims=True) + 1e-8)

    # (b) forward-from-current PRED: fwd(current_vis, predictor([pooled|state=0]))
    Xf = torch.from_numpy(latn).to(dev)
    feat_fwd = torch.from_numpy(np.concatenate([latn, np.zeros((len(kt), 14), np.float32)], 1)).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_fwd = fwd(torch.cat([Xf, predm(feat_fwd)], -1)).float().cpu().numpy()
    pred_fwd /= (np.linalg.norm(pred_fwd, axis=1, keepdims=True) + 1e-8)

    def dec(a):
        return np.concatenate([flow(a[s:s + 64]) for s in range(0, len(a), 64)])
    d_abs, d_fwd, d_true = dec(pred_abs), dec(pred_fwd), dec(true_next)
    print("decoded absolute + forward + true (flow)", flush=True)

    R = flow.res; P = 256
    def lab(img, t): return label(cv2.resize(img, (P, P)), t)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def compose(f):
        k = int(np.searchsorted(kt, f, side="right") - 1); k = max(0, min(k, len(kt) - 1))
        p0 = lab(frames[f], f"vis_base real | milestone {stage_m[stage_of[k]]}")
        p1 = lab(d_abs[k], "PRED absolute -> flow")
        p2 = lab(d_fwd[k], "PRED forward-from-current -> flow")
        p3 = lab(d_true[k], "TRUE milestone+1 -> flow")
        gap = np.full((p0.shape[0], 8, 3), 20, np.uint8)
        return np.ascontiguousarray(np.hstack([p0, gap, p1, gap, p2, gap, p3]))
    c0 = compose(0); Hc, Wc = c0.shape[:2]; Hc -= Hc % 2; Wc -= Wc % 2
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (Wc, Hc))
    assert vw.isOpened(), f"VideoWriter failed to open ({Wc}x{Hc})"
    for f in range(N):
        vw.write(cv2.cvtColor(compose(f)[:Hc, :Wc], cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"saved {args.out} | {N} frames @ {out_fps}fps | {len(st)} stages | 4-panel real|abs|fwd|true", flush=True)


if __name__ == "__main__":
    main()
