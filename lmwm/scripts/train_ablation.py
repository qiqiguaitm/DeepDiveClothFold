#!/usr/bin/env python
"""Controlled-variable ablation over the v2 predictor: ONE flag changes per run. Reports the SAME
metric panel (oracle/deploy/best-of-N grid-cos + identity top-N) so configs are directly comparable.
Run the matrix in parallel (gf3 8 + local 2). Lag is a separate follow-up on winners.

Flags (baseline v2 = seglast·medoid·adaln·lift1·code128·K4):
  --target_mode {seglast, allframes}   seglast: (last frame of seg i, next-seg medoid) [v2]
                                       allframes: EVERY frame in seg i -> same next-seg medoid (more data + smooth)
  --teacher {medoid, center, none}     medoid: inverse(g_t,g_f)->z [v2]
                                       center: + aux CE(linear(z)->next milestone id) = cluster-center identity anchor
                                       none:   no inverse; predm trained by reconstruction (v1-style)
  --fwd_arch {adaln, concat}           adaln [v2] vs concat ForwardDec
  --lift_w, --code_dim, --K
target/medoid stay DINOv3-H+CRAVE Viterbi (offline). next-seg medoid is always temporally forward (next
segment), so allframes is inherently forward.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import load_index, read_imgs, InverseEnc, ForwardDec  # noqa: E402
from crave.utils.dp import viterbi_forward  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402
from train_twomodel_v2 import MilestoneGenerator, MilestonePredictor, MilestonePredictorGrid, PI05_NPZ, PI05_NPZ_GF3, cosr  # noqa: E402


def build_pairs_abl(E, FR, Fn, proto, protoL, pord, mode, val_eps, seed):
    """(cur_gidx, tgt_gidx, cur_ms, next_ms) for train/val. Viterbi segments; next-seg medoid target."""
    tr, va = [], []
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        Fq = Fn[order]
        ms = viterbi_forward(np.linalg.norm(Fq[:, None] - protoL[None], axis=2), pord, up=3.0, down=25.0, hard_start=True)
        ch = np.where(np.diff(ms) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(ms)]])
        seg_med, seg_m, spans = [], [], []
        for s, e in zip(st, en):
            m = int(ms[s]); seg_med.append(int(order[s + int((Fq[s:e] @ protoL[m]).argmax())]))
            seg_m.append(m); spans.append((s, e))
        dst = va if ep in val_eps else tr
        for i in range(len(seg_m) - 1):
            tgt = seg_med[i + 1]; nm = seg_m[i + 1]; cm = seg_m[i]
            if mode == "seglast":
                dst.append((int(order[spans[i][1] - 1]), tgt, cm, nm))
            else:                                                          # allframes: every frame in seg i
                for f in range(spans[i][0], spans[i][1]):
                    dst.append((int(order[f]), tgt, cm, nm))
    rng = np.random.default_rng(seed); rng.shuffle(tr); rng.shuffle(va)
    return tr, va


def topn_hit(scores, target, Ns=(1, 3, 5)):
    order = np.argsort(-scores, axis=1)
    return {f"top{n}": round(float(np.mean([t in order[i, :n] for i, t in enumerate(target)])), 4) for n in Ns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_mode", default="seglast", choices=["seglast", "allframes"])
    ap.add_argument("--teacher", default="medoid", choices=["medoid", "center", "none"])
    ap.add_argument("--fwd_arch", default="adaln", choices=["adaln", "concat"])
    ap.add_argument("--pred_input", default="gist", choices=["gist", "grid"])  # predictor sees pooled gist vs full grid
    ap.add_argument("--lift_w", type=float, default=1.0)
    ap.add_argument("--code_dim", type=int, default=128)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--center_w", type=float, default=0.5)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=16000)
    ap.add_argument("--n_val", type=int, default=1600)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--bestof", type=int, default=8)
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device; cd = args.code_dim
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)

    E, FR, Fn = load_index(args.feature_dir)
    rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = rg["prototype_table"].astype(np.float32); pord = rg["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    Mn = proto.shape[0]
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    tr, va = build_pairs_abl(E, FR, Fn, proto, protoL, pord, args.target_mode, val_eps, args.seed)
    tr = tr[:args.n_train]; va = va[:args.n_val]
    uniq = sorted(set([p[0] for p in tr + va] + [p[1] for p in tr + va])); u2k = {g: k for k, g in enumerate(uniq)}
    print(f"[{args.tag}] {len(tr)} train + {len(va)} val, {len(uniq)} frames", flush=True)

    imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, np.array(uniq), 224, 128)
    enc = SiglipBigVision(npz, device=dev)
    grids = enc.encode_grid(imgs, bs=32); din = grids.shape[1]
    gmu, gsd = float(grids.mean()), float(grids.std() + 1e-6)
    GZnp = ((grids - gmu) / gsd).astype(np.float32); del grids
    gist = torch.from_numpy(GZnp.mean((2, 3)))                             # fp32 pooled (small)
    GZ = torch.from_numpy(GZnp).half(); del GZnp                          # fp16 CPU grids (halve RAM vs fp32)
    tra = np.array([u2k[p[0]] for p in tr]); trb = np.array([u2k[p[1]] for p in tr]); trn = np.array([p[3] for p in tr])
    vaa = np.array([u2k[p[0]] for p in va]); vab = np.array([u2k[p[1]] for p in va]); vbn = np.array([p[3] for p in va]); vcm = np.array([p[2] for p in va])

    inv = InverseEnc(din, cd).to(dev) if args.teacher != "none" else None
    fwd = (MilestoneGenerator(din, cd) if args.fwd_arch == "adaln" else ForwardDec(din, cd)).to(dev)
    predm = (MilestonePredictorGrid if args.pred_input == "grid" else MilestonePredictor)(din, cd, args.K).to(dev)
    cen_head = nn.Linear(cd, Mn).to(dev) if args.teacher == "center" else None
    p1 = list(fwd.parameters()) + (list(inv.parameters()) if inv else []) + (list(cen_head.parameters()) if cen_head else [])
    o1 = torch.optim.AdamW(p1, lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(predm.parameters(), lr=2e-4, weight_decay=1e-5)
    for step in range(args.steps):
        sel = torch.randint(0, len(tra), (64,))
        Gc = GZ[tra[sel]].float().to(dev); Gf = GZ[trb[sel]].float().to(dev); gc = gist[tra[sel]].to(dev)
        pc = Gc if args.pred_input == "grid" else gc                       # predictor input: grid vs gist
        nm = torch.from_numpy(trn[sel.numpy()]).long().to(dev)
        if inv is not None:                                                # teacher path
            z = inv(Gc, Gf); gh = fwd(Gc, z)
            lift = torch.relu(cosr(gh.flatten(1), Gc.flatten(1)) - cosr(gh.flatten(1), Gf.flatten(1))).mean()
            l1 = F.smooth_l1_loss(gh, Gf) + args.lift_w * lift
            if cen_head is not None:                                       # cluster-center identity anchor
                l1 = l1 + args.center_w * F.cross_entropy(cen_head(z), nm)
            o1.zero_grad(); l1.backward(); o1.step()
            l2 = predm.nll(pc, z.detach()); o2.zero_grad(); l2.backward(); o2.step()
        else:                                                              # no teacher: predm code reconstructs
            zp = predm.deploy_mean(pc) if False else predm(pc)[1][:, 0]    # use 1st component mean as code
            gh = fwd(Gc, zp)
            lift = torch.relu(cosr(gh.flatten(1), Gc.flatten(1)) - cosr(gh.flatten(1), Gf.flatten(1))).mean()
            loss = F.smooth_l1_loss(gh, Gf) + args.lift_w * lift
            o1.zero_grad(); o2.zero_grad(); loss.backward(); o1.step(); o2.step()
    fwd.eval(); predm.eval()
    if inv:
        inv.eval()

    # SigLIP milestone prototypes (identity retrieval) from val-set gists by DINOv3 milestone id
    msid = (Fn[np.array(uniq)] @ proto.T).argmax(1); gnp = gist.numpy()
    sp = np.stack([gnp[msid == m].mean(0) if (msid == m).any() else np.zeros(din, np.float32) for m in range(Mn)])
    spL = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)

    def cn(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    co, cd_, cb, cp = [], [], [], []; idhit = {1: 0, 3: 0, 5: 0}
    with torch.no_grad():
        for s in range(0, len(vaa), 128):
            Gc = GZ[vaa[s:s + 128]].float().to(dev); Gf = GZ[vab[s:s + 128]].float().to(dev); gc = gist[vaa[s:s + 128]].to(dev)
            pc = Gc if args.pred_input == "grid" else gc
            gtr = f(Gf); zdep = predm.deploy_mean(pc)
            if inv is not None:
                co.append(cn(f(fwd(Gc, inv(Gc, Gf))), gtr))
            cd_.append(cn(f(fwd(Gc, zdep)), gtr)); cp.append(cn(f(Gc), gtr))
            best = None
            for zi in predm.sample(pc, args.bestof):
                ck = cn(f(fwd(Gc, zi)), gtr); best = ck if best is None else np.maximum(best, ck)
            cb.append(best)
        # identity top-N: rank milestones by cos(deploy predicted pooled gist, SigLIP proto)
        idpred = []
        for s in range(0, len(vaa), 128):
            Gc = GZ[vaa[s:s + 128]].float().to(dev); gc = gist[vaa[s:s + 128]].to(dev)
            pc = Gc if args.pred_input == "grid" else gc
            gh = fwd(Gc, predm.deploy_mean(pc)).mean((2, 3)).cpu().numpy()  # predicted pooled gist
            idpred.append(gh)
        idpred = np.concatenate(idpred); idpred /= (np.linalg.norm(idpred, axis=1, keepdims=True) + 1e-8)
        idn = topn_hit(idpred @ spL.T, vbn)
        # HARD REQ ②: predicted milestone's progress-VALUE > current stage's value
        pred_ms = (idpred @ spL.T).argmax(1); cur_ms = vcm  # Viterbi current milestone (consistent w/ target)
        value_fwd = float((pord[pred_ms] > pord[cur_ms]).mean())            # value-forward fraction
        value_true = float((pord[vbn] > pord[cur_ms]).mean())              # target's own value-forward (ref)

    res = {"tag": args.tag, "target_mode": args.target_mode, "teacher": args.teacher, "fwd_arch": args.fwd_arch,
           "pred_input": args.pred_input, "predm_params": sum(p.numel() for p in predm.parameters()),
           "lift_w": args.lift_w, "center_w": args.center_w, "code_dim": cd, "K": args.K, "n_train": len(tr),
           "oracle_grid_cos": round(float(np.concatenate(co).mean()), 4) if co else None,
           "deploy_grid_cos": round(float(np.concatenate(cd_).mean()), 4),
           "bestof%d" % args.bestof: round(float(np.concatenate(cb).mean()), 4),
           "persistence": round(float(np.concatenate(cp).mean()), 4),
           "identity_topN": idn, "value_forward_frac": round(value_fwd, 4), "target_value_forward_ref": round(value_true, 4)}
    outp = REPO / f"lmwm/outputs/ablation/{args.tag}.json"; outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    ckp = REPO / f"lmwm/checkpoints/ablation/{args.tag}.pt"; ckp.parent.mkdir(parents=True, exist_ok=True)
    save = {"fwd": fwd.state_dict(), "predm": predm.state_dict(), "K": args.K, "din": din, "code_dim": cd,
            "gmu": float(gmu), "gsd": float(gsd), "arch": "v2" if args.fwd_arch == "adaln" else "v2concat",
            "fwd_arch": args.fwd_arch, "pred_input": args.pred_input}
    if inv:
        save["inv"] = inv.state_dict()
    torch.save(save, ckp)
    print(json.dumps(res, indent=2), flush=True); print(f"saved -> {ckp}", flush=True)


if __name__ == "__main__":
    main()
