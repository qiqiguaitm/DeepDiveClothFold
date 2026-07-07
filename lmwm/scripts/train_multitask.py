#!/usr/bin/env python
"""Multi-task LMWM world-model training — the cross-task test (P1b/P2).

Trains ONE state-conditioned two-model (inverse-teacher + AdaLN generator + MDN predictor) jointly on
SEVERAL tasks, each with its OWN CRAVE milestones (different counts: kai0 37 / coffee 15 / vis 27 /
xvla 51). Compares 3 anchor forms on the code z:
  - union_ce   : CE over the UNION of all tasks' milestones (global ids, offset per task). Head grows
                 with #tasks -> the discrete-vocab approach; can it even share across counts?
  - progress   : scalar per-task-normalized progress[0,1] regression + monotonic margin (count-agnostic).
  - progress_id: progress scalar + a continuous IDENTITY term (regress z to a fixed random projection of
                 the target milestone's DINOv3-H prototype) -> open-vocabulary, keeps identity/multimodal.

Language is deliberately NOT used (world model stays state-conditioned; task routing is left to the
policy). Eval is PER-TASK (each task assigned to its own prototypes).
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
import cv2  # noqa: E402
from train_lawm_patch import load_index, read_imgs, InverseEnc  # noqa: E402
from train_ablation import build_pairs_abl, topn_hit  # noqa: E402
from train_twomodel_v2 import MilestoneGenerator, MilestonePredictor, PI05_NPZ, PI05_NPZ_GF3, cosr  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402

# per-task data registry: DINOv3-H bank (index+shards) + frames + recurrence graph + frame format
TASKS = {
    "kai0":   dict(fdir="temp/crave_full_dinov3h",  root="kai0/data/Task_A/kai0_base",
                   cam="observation.images.top_head", fmt="kai0",
                   graph="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz"),
    "coffee": dict(fdir="temp/coffee_dinov3h",      root="temp/aloha_static_coffee",
                   cam="observation.images.cam_high", fmt="lerobotv3",
                   graph="lmwm/data/recurrence_graphs/coffee_dinov3h/recurrence_graph.npz"),
    "vis":    dict(fdir="temp/vis_dinov3h",         root="kai0/data/Task_A/vis_base/v1/2026-04-24",
                   cam="observation.images.top_head", fmt="kai0",
                   graph="lmwm/data/recurrence_graphs/vis_dinov3h/recurrence_graph.npz"),
    "xvla":   dict(fdir="temp/xvla_dinov3h",        root="xvla/data/xvla_soft_fold/0707_11pm_stage_1_stage2new_new_cam_very_slow",
                   cam="observation.images.cam_high", fmt="hdf5",
                   graph="lmwm/data/recurrence_graphs/xvla_dinov3h/recurrence_graph.npz"),
}


def read_frames(cfg, E, FR, gidx, enc_res, tgt_res):
    """Dispatch frame reading by format. Returns (enc_imgs [N,enc,enc,3], disp [N,tgt,tgt,3]) uint8."""
    if cfg["fmt"] == "kai0":
        return read_imgs(Path(cfg["root"]), cfg["cam"], E, FR, gidx, enc_res, tgt_res)
    if cfg["fmt"] == "lerobotv3":
        return _read_lerobotv3(Path(cfg["root"]), cfg["cam"], E, FR, gidx, enc_res, tgt_res)
    if cfg["fmt"] == "hdf5":
        return _read_hdf5(Path(cfg["root"]), cfg["cam"], E, FR, gidx, enc_res, tgt_res)
    raise NotImplementedError(cfg["fmt"])


def _read_hdf5(root, camera, E, FR, gidx, enc_res, tgt_res):
    """XVLA HDF5: one episode_<N>.hdf5 per episode; frames = JPEG-encoded bytes at observations/images/<cam>."""
    import h5py
    from collections import defaultdict
    key = camera.replace("observation.images.", "")                        # cam_high
    ie = np.zeros((len(gidx), enc_res, enc_res, 3), np.uint8)
    it = np.zeros((len(gidx), tgt_res, tgt_res, 3), np.uint8)
    by_ep = defaultdict(list)
    for k, g in enumerate(gidx):
        by_ep[int(E[g])].append((k, int(FR[g])))
    for ep, items in by_ep.items():
        fp = root / f"episode_{ep}.hdf5"
        if not fp.exists():
            continue
        with h5py.File(fp, "r") as h:
            ds = h[f"observations/images/{key}"]
            for k, fr in items:
                img = cv2.imdecode(np.frombuffer(bytes(ds[fr]), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ie[k] = cv2.resize(img, (enc_res, enc_res)); it[k] = cv2.resize(img, (tgt_res, tgt_res))
    return ie, it


def _read_lerobotv3(root, camera, E, FR, gidx, enc_res, tgt_res):
    """LeRobot v3: per-camera CONCAT video (videos/<cam>/chunk-*/file-*.mp4), frames in global episode
    order (gi = episode_start + FR). Coffee videos are AV1 -> decode with pyav (cv2 can't). We decode
    sequentially up to the max needed global index, grabbing the wanted frames."""
    import av
    import glob
    from collections import defaultdict
    starts, cum = {}, 0                                                     # per-episode global start offset
    for line in (root / "meta/episodes.jsonl").read_text().splitlines():
        d = json.loads(line); starts[int(d["episode_index"])] = cum; cum += int(d["length"])
    vids = sorted(glob.glob(str(root / f"videos/{camera}/chunk-*/file-*.mp4")))
    ie = np.zeros((len(gidx), enc_res, enc_res, 3), np.uint8)
    it = np.zeros((len(gidx), tgt_res, tgt_res, 3), np.uint8)
    gpos = np.array([starts[int(E[g])] + int(FR[g]) for g in gidx])
    pos2rows = defaultdict(list)
    for k, p in enumerate(gpos):
        pos2rows[int(p)].append(k)
    maxwant = int(gpos.max()) if len(gpos) else -1
    fbase = 0
    for vf in vids:
        if fbase > maxwant:
            break
        container = av.open(vf)
        li = -1
        for li, frame in enumerate(container.decode(container.streams.video[0])):
            gp = fbase + li
            if gp in pos2rows:
                img = frame.to_ndarray(format="rgb24")
                e = cv2.resize(img, (enc_res, enc_res)); t = cv2.resize(img, (tgt_res, tgt_res))
                for k in pos2rows[gp]:
                    ie[k] = e; it[k] = t
            if gp >= maxwant:
                break
        container.close(); fbase += li + 1
    return ie, it


class IdentityAnchor(nn.Module):
    """Continuous identity term: regress z -> fixed random projection of target milestone prototype.
    Open-vocabulary (any prototype projects; no fixed-K table)."""
    def __init__(self, cd, id_dim=64):
        super().__init__(); self.head = nn.Linear(cd, id_dim)

    def loss(self, z, target_id):
        return F.mse_loss(self.head(z), target_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kai0,coffee")
    ap.add_argument("--heldout", default="", help="comma tasks EXCLUDED from training, eval-only (open-vocab/LOO test)")
    ap.add_argument("--anchor", default="progress_id", choices=["union_ce", "progress", "progress_id"])
    ap.add_argument("--teacher", default="inv", choices=["inv", "none"])    # inv=inverse-dynamics teacher+distill; none=direct predictor+generator end-to-end
    ap.add_argument("--center_w", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--id_dim", type=int, default=64)
    ap.add_argument("--code_dim", type=int, default=128)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lift_w", type=float, default=1.0)
    ap.add_argument("--mode", default="seglast")
    ap.add_argument("--per_task_cap", type=int, default=8000, help="max TRAIN pairs per task (balance)")
    ap.add_argument("--val_cap", type=int, default=4000, help="max VAL pairs per task (RAM/speed; caps unique frames read)")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--bestof", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save_ckpt", action="store_true", help="save the trained cross-task predictor (fwd+predm+inv+anchor)")
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)
    datasets = args.datasets.split(",")
    rng = np.random.default_rng(args.seed)
    enc = SiglipBigVision(npz, device=dev)

    heldout = [h for h in args.heldout.split(",") if h]
    all_tasks = [(n, False) for n in datasets] + [(n, True) for n in heldout]
    grids_all, tasks_meta = [], []
    goff = 0; msoff = 0
    for ti, (name, is_ho) in enumerate(all_tasks):
        cfg = TASKS[name]
        E, FR, Fn = load_index(REPO / cfg["fdir"])
        g = np.load(REPO / cfg["graph"]); proto = g["prototype_table"].astype(np.float32); pord = g["pord"].astype(np.float32)
        protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8); M = len(proto)
        eps = np.unique(E); rng.shuffle(eps); val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
        tr, va = build_pairs_abl(E, FR, Fn, proto, protoL, pord, args.mode, val_eps, args.seed)
        if is_ho:
            tr = []                                                        # heldout: eval-only, never trained
        elif len(tr) > args.per_task_cap:
            tr = [tr[i] for i in rng.choice(len(tr), args.per_task_cap, replace=False)]
        if len(va) > args.val_cap:                                         # cap val (RAM/speed): fewer unique frames to read
            va = [va[i] for i in rng.choice(len(va), args.val_cap, replace=False)]
        uniq = sorted(set([p[0] for p in tr + va] + [p[1] for p in tr + va])); u2k = {gi: k for k, gi in enumerate(uniq)}
        ie, _ = read_frames(cfg, E, FR, np.array(uniq), 224, 128)
        grids = enc.encode_grid(ie, bs=32); din = grids.shape[1]
        progn = ((pord - pord.min()) / (pord.max() - pord.min() + 1e-8)).astype(np.float32)
        pdim = proto.shape[1]                                             # DINOv3-H prototype dim (1280), NOT SigLIP grid dim
        idproj = (rng.standard_normal((pdim, args.id_dim)).astype(np.float32) / np.sqrt(pdim)) if ti == 0 else tasks_meta[0]["idproj"]
        idtarget = (protoL @ idproj).astype(np.float32)                    # (M, id_dim) fixed id embedding per milestone
        # SigLIP proto (identity retrieval) per milestone from this task's val gists
        gnp = grids.mean((2, 3))
        msid = (Fn[np.array(uniq)] @ proto.T).argmax(1)
        sp = np.stack([gnp[msid == m].mean(0) if (msid == m).any() else np.zeros(din, np.float32) for m in range(M)])
        spL = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)
        meta = dict(name=name, ti=ti, M=M, msoff=msoff, goff=goff, progn=progn, idtarget=idtarget, idproj=idproj,
                    pord=pord, spL=spL, din=din, is_heldout=is_ho,
                    tr=[(goff + u2k[c], goff + u2k[t], msoff + cm, msoff + nm, cm, nm) for (c, t, cm, nm) in tr],
                    va=[(goff + u2k[c], goff + u2k[t], msoff + cm, msoff + nm, cm, nm) for (c, t, cm, nm) in va])
        tasks_meta.append(meta); grids_all.append(grids); goff += len(uniq)
        if not is_ho:
            msoff += M                                                     # global ms ids span TRAINING tasks only
        print(f"[{name}{' HELDOUT' if is_ho else ''}] M={M} pairs tr={len(meta['tr'])} va={len(meta['va'])} frames={len(uniq)}", flush=True)

    train_metas = [m for m in tasks_meta if not m["is_heldout"]]
    G = np.concatenate(grids_all); din = G.shape[1]; total_M = msoff
    gmu, gsd = float(G.mean()), float(G.std() + 1e-6)                       # SHARED normalization across tasks
    GZ = torch.from_numpy(((G - gmu) / gsd).astype(np.float32)).half(); del G, grids_all
    gist_all = GZ.float().mean((2, 3))
    idproj = train_metas[0]["idproj"]
    idtarget_g = np.concatenate([m["idtarget"] for m in train_metas])       # (total_M, id_dim), TRAINING global-ms
    progn_g = np.concatenate([m["progn"] for m in train_metas])             # (total_M,), TRAINING global-ms
    TR = [p for m in train_metas for p in m["tr"]]; rng.shuffle(TR); TR = np.array(TR)

    inv = InverseEnc(din, args.code_dim).to(dev) if args.teacher == "inv" else None
    fwd = MilestoneGenerator(din, args.code_dim).to(dev)
    predm = MilestonePredictor(din, args.code_dim, args.K).to(dev)
    cd = args.code_dim
    if args.anchor == "union_ce":
        anchor_head = nn.Linear(cd, total_M).to(dev); idanchor = None
    else:
        anchor_head = nn.Linear(cd, 1).to(dev)                             # progress scalar
        idanchor = IdentityAnchor(cd, args.id_dim).to(dev) if args.anchor == "progress_id" else None
    ap_par = list(anchor_head.parameters()) + (list(idanchor.parameters()) if idanchor else [])
    o1 = torch.optim.AdamW(list(fwd.parameters()) + (list(inv.parameters()) if inv else []) + ap_par, lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(predm.parameters(), lr=2e-4, weight_decay=1e-5)
    progn_t = torch.from_numpy(progn_g).to(dev); idt_t = torch.from_numpy(idtarget_g).to(dev)

    def anchor_loss(z, gnm_t, gcm):
        if args.anchor == "union_ce":
            return args.center_w * F.cross_entropy(anchor_head(z), gnm_t)
        ph = anchor_head(z).squeeze(-1).sigmoid()
        la = F.mse_loss(ph, progn_t[gnm_t]) + torch.relu(progn_t[torch.from_numpy(gcm).long().to(dev)] - ph + args.margin).mean()
        if idanchor is not None:
            la = la + idanchor.loss(z, idt_t[gnm_t])
        return args.center_w * la

    for step in range(args.steps):
        sel = torch.randint(0, len(TR), (64,))
        b = TR[sel.numpy()]; ca, cb_, gcm, gnm = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        Gc = GZ[ca].float().to(dev); Gf = GZ[cb_].float().to(dev); gnm_t = torch.from_numpy(gnm).long().to(dev)
        if args.teacher == "inv":                                          # inverse-dynamics teacher + distillation
            z = inv(Gc, Gf); gh = fwd(Gc, z)
            lift = torch.relu(cosr(gh.flatten(1), Gc.flatten(1)) - cosr(gh.flatten(1), Gf.flatten(1))).mean()
            l1 = F.smooth_l1_loss(gh, Gf) + args.lift_w * lift + anchor_loss(z, gnm_t, gcm)
            o1.zero_grad(); l1.backward(); o1.step()
            l2 = predm.nll(gist_all[ca].to(dev), z.detach()); o2.zero_grad(); l2.backward(); o2.step()
        else:                                                              # DIRECT: predictor code -> generator, end-to-end (no teacher)
            z = predm(gist_all[ca].to(dev))[1][:, 0]                       # 1st-component mean (differentiable)
            gh = fwd(Gc, z)
            lift = torch.relu(cosr(gh.flatten(1), Gc.flatten(1)) - cosr(gh.flatten(1), Gf.flatten(1))).mean()
            l1 = F.smooth_l1_loss(gh, Gf) + args.lift_w * lift + anchor_loss(z, gnm_t, gcm)
            o1.zero_grad(); o2.zero_grad(); l1.backward(); o1.step(); o2.step()
    fwd.eval(); predm.eval()
    if inv is not None:
        inv.eval()

    # ---- PER-TASK eval ----
    def cn(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    per_task = {}
    with torch.no_grad():
        for m in tasks_meta:
            va = np.array(m["va"]); spL = m["spL"]; progn = m["progn"]
            vaa, vab, gcm, gnm, lcm, lnm = va[:, 0], va[:, 1], va[:, 2], va[:, 3], va[:, 4], va[:, 5]
            co, cd_, cp, idpred = [], [], [], []
            for s in range(0, len(vaa), 128):
                Gc = GZ[vaa[s:s + 128]].float().to(dev); Gf = GZ[vab[s:s + 128]].float().to(dev)
                gc = gist_all[vaa[s:s + 128]].to(dev); gtr = f(Gf); zdep = predm.deploy_mean(gc)
                if inv is not None:
                    co.append(cn(f(fwd(Gc, inv(Gc, Gf))), gtr))            # oracle only meaningful with teacher
                cd_.append(cn(f(fwd(Gc, zdep)), gtr)); cp.append(cn(f(Gc), gtr))
                idpred.append(fwd(Gc, zdep).mean((2, 3)).cpu().numpy())
            idpred = np.concatenate(idpred); idpred /= (np.linalg.norm(idpred, axis=1, keepdims=True) + 1e-8)
            idn = topn_hit(idpred @ spL.T, lnm)
            pms = (idpred @ spL.T).argmax(1)
            vfwd = float((progn[pms] > progn[lcm]).mean())
            per_task[m["name"]] = {"oracle": round(float(np.concatenate(co).mean()), 4) if co else None,
                                   "deploy": round(float(np.concatenate(cd_).mean()), 4),
                                   "persistence": round(float(np.concatenate(cp).mean()), 4),
                                   "identity_topN": idn, "value_forward_frac": round(vfwd, 4),
                                   "is_heldout": m["is_heldout"], "n_val": len(va)}
    tr_v = [v for v in per_task.values() if not v["is_heldout"]]
    ho_v = [v for v in per_task.values() if v["is_heldout"]]
    mean = lambda vs, k, sub=None: round(float(np.mean([(v[k][sub] if sub else v[k]) for v in vs])), 4) if vs else None
    res = {"tag": args.tag, "datasets": datasets, "heldout": heldout, "anchor": args.anchor, "total_M": total_M,
           "center_w": args.center_w, "per_task": per_task,
           "train_deploy_mean": mean(tr_v, "deploy"), "train_id_top3_mean": mean(tr_v, "identity_topN", "top3"),
           "heldout_deploy_mean": mean(ho_v, "deploy"), "heldout_id_top3_mean": mean(ho_v, "identity_topN", "top3"),
           "heldout_persist_mean": mean(ho_v, "persistence"), "heldout_vfwd_mean": mean(ho_v, "value_forward_frac")}
    outp = REPO / f"lmwm/outputs/multitask/{args.tag}.json"; outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)

    if args.save_ckpt:                                                     # save the trained cross-task predictor
        ck = {"fwd": fwd.state_dict(), "predm": predm.state_dict(), "anchor_head": anchor_head.state_dict(),
              "inv": inv.state_dict() if inv is not None else None, "idproj": idproj,
              "gmu": gmu, "gsd": gsd, "din": din, "code_dim": args.code_dim, "K": args.K,
              "anchor": args.anchor, "teacher": args.teacher, "total_M": total_M, "datasets": datasets,
              "tasks": [{"name": m["name"], "M": int(m["M"]), "msoff": int(m["msoff"]), "pord": m["pord"].tolist()} for m in train_metas]}
        cp = REPO / f"lmwm/checkpoints/{args.tag}.pt"; cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ck, cp); print(f"saved cross-task predictor -> {cp}", flush=True)


if __name__ == "__main__":
    main()
