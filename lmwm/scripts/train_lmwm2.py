#!/usr/bin/env python
"""LMWM-2 P1/P2 (local, intrinsic; no VLA): value-geometric code space + conditioned predictor + probes.

P1 (code space + conditioning):
  - code z = Wproj·L2(gist): frames and milestone centers live in ONE space (proto teacher lineage).
  - L_mono trains a progress DIRECTION u by ranking on TEMPORAL order (episode-internal pairs) — no CRAVE
    values in the loss; CRAVE labels are used as the EXAM only (red line 2).
  - conditioning arms --cond {none|task|prevz|task,prevz}: task-emb (language proxy on single-instruction
    data) and prev-milestone code (teacher-forced zteach[cur_ms] in offline eval; at real deploy this is
    the model's own last committed code — self-contained, no table).
P2 (geometry vs module, judged on the same replay exam):
  - progress: u·z (linear-on-code) vs linear-on-gist vs MLP-on-gist  -> monotonic order accuracy
  - reach:    code-distance threshold vs learned reach-head          -> F1 / AUC
  - verify:   MDN own density vs external verifier MLP               -> AUC (backward + wrong-task negatives)
  - calibration: MDN confidence vs identity correctness              -> ECE + entropy-gate stats
Teacher fixed to proto (settled). Trains fwd generator + conditioned MDN exactly like train_multitask
teacher=proto for comparability.
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
from train_lawm_patch import load_index  # noqa: E402
from train_ablation import build_pairs_abl, topn_hit  # noqa: E402
from train_twomodel_v2 import MilestoneGenerator, MilestonePredictor, PI05_NPZ, PI05_NPZ_GF3, cosr  # noqa: E402
from train_multitask import TASKS, read_frames  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402


def per_sample_nll(predm, x, z):
    """MDN log-density per sample (no reduction) — the model's OWN verifier score."""
    logit, mu, ls = predm(x)
    logpi = F.log_softmax(logit, -1)
    var = (2 * ls).exp()
    comp = -0.5 * (((z[:, None] - mu) ** 2) / var + 2 * ls + np.log(2 * np.pi)).sum(-1)
    return torch.logsumexp(logpi + comp, -1)                                # (B,) logp


def auc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg) + 1e-9))


def best_f1(pos_d, neg_d):
    """Binary 'reached' detector from distances (pos should be SMALL). Sweep threshold, return best F1."""
    ths = np.quantile(np.concatenate([pos_d, neg_d]), np.linspace(0.02, 0.98, 49))
    best = 0.0
    for t in ths:
        tp = (pos_d < t).sum(); fp = (neg_d < t).sum(); fn = (pos_d >= t).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        best = max(best, float(f1))
    return round(best, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kai0,coffee,xvla")
    ap.add_argument("--cond", default="none", help="none | comma of {task,prevz}")
    ap.add_argument("--code_dim", type=int, default=128)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lift_w", type=float, default=1.0)
    ap.add_argument("--mode", default="seglast")
    ap.add_argument("--per_task_cap", type=int, default=8000)
    ap.add_argument("--val_cap", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--probe_steps", type=int, default=3000)
    ap.add_argument("--task_emb_dim", type=int, default=16)
    ap.add_argument("--target", default="code", choices=["code", "gist"], help="code(128) vs raw gist(1152) as teacher/predictor target")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    conds = [c for c in args.cond.split(",") if c and c != "none"]
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)
    datasets = args.datasets.split(",")
    rng = np.random.default_rng(args.seed)
    enc = SiglipBigVision(npz, device=dev)

    # ---------- load tasks (same recipe as train_multitask, + per-frame milestone labels lms) ----------
    grids_all, tasks_meta = [], []
    goff = 0; msoff = 0
    for ti, name in enumerate(datasets):
        cfg = TASKS[name]
        E, FR, Fn = load_index(REPO / cfg["fdir"])
        g = np.load(REPO / cfg["graph"]); proto = g["prototype_table"].astype(np.float32); pord = g["pord"].astype(np.float32)
        protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8); M = len(proto)
        eps = np.unique(E); rng.shuffle(eps); val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
        tr, va = build_pairs_abl(E, FR, Fn, proto, protoL, pord, args.mode, val_eps, args.seed)
        if len(tr) > args.per_task_cap:
            tr = [tr[i] for i in rng.choice(len(tr), args.per_task_cap, replace=False)]
        if len(va) > args.val_cap:
            va = [va[i] for i in rng.choice(len(va), args.val_cap, replace=False)]
        uniq = sorted(set([p[0] for p in tr + va] + [p[1] for p in tr + va])); u2k = {gi: k for k, gi in enumerate(uniq)}
        ie, _ = read_frames(cfg, E, FR, np.array(uniq), 224, 128)
        grids = enc.encode_grid(ie, bs=32); din = grids.shape[1]
        progn = ((pord - pord.min()) / (pord.max() - pord.min() + 1e-8)).astype(np.float32)
        gnp = grids.mean((2, 3))
        lms = (Fn[np.array(uniq)] @ proto.T).argmax(1)                      # per-frame CRAVE milestone (EXAM label)
        sp = np.stack([gnp[lms == m].mean(0) if (lms == m).any() else np.zeros(din, np.float32) for m in range(M)])
        spL = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)
        meta = dict(name=name, ti=ti, M=M, msoff=msoff, goff=goff, progn=progn, pord=pord, spL=spL, din=din,
                    lms=lms,
                    tr=[(goff + u2k[c], goff + u2k[t], msoff + cm, msoff + nm, cm, nm) for (c, t, cm, nm) in tr],
                    va=[(goff + u2k[c], goff + u2k[t], msoff + cm, msoff + nm, cm, nm) for (c, t, cm, nm) in va])
        tasks_meta.append(meta); grids_all.append(grids); goff += len(uniq); msoff += M
        print(f"[{name}] M={M} tr={len(meta['tr'])} va={len(meta['va'])} frames={len(uniq)}", flush=True)

    G = np.concatenate(grids_all); din = G.shape[1]; total_M = msoff
    gmu, gsd = float(G.mean()), float(G.std() + 1e-6)
    GZ = torch.from_numpy(((G - gmu) / gsd).astype(np.float32)).half(); del G, grids_all
    gist_all = GZ.float().mean((2, 3))                                       # normalized gist (predictor input)
    graw = gist_all * gsd + gmu
    gl2 = (graw / (graw.norm(dim=1, keepdim=True) + 1e-8)).numpy()           # L2 raw gist (code-space input)
    frame_task = np.concatenate([np.full(len(m["lms"]), m["ti"]) for m in tasks_meta])
    frame_lms_g = np.concatenate([m["lms"] + m["msoff"] for m in tasks_meta])  # global milestone id per frame
    progn_g = np.concatenate([m["progn"] for m in tasks_meta])
    task_of_ms = np.concatenate([np.full(m["M"], m["ti"]) for m in tasks_meta])

    # ---------- code space: z = Wproj·L2(gist); centers zc from SigLIP milestone centers ----------
    if args.target == "gist":
        args.code_dim = din                                                # predict raw gist (1152), no compression
    Wproj = (rng.standard_normal((din, args.code_dim)).astype(np.float32) / np.sqrt(din))
    sp_g = np.concatenate([m["spL"] for m in tasks_meta])
    zc = (sp_g @ Wproj).astype(np.float32)                                 # (total_M, code_dim) center codes
    zf = (gl2 @ Wproj).astype(np.float32)                                  # (N, code_dim) frame codes
    zc_t = torch.from_numpy(zc).to(dev)
    if args.target == "gist":
        zc = zc.astype(np.float32); zf = zf.astype(np.float32)            # gist mode: z == Wproj·gl2 == raw gist (W=I up to rot)
        # Actually: Wproj is din x din (1152x1152) random projection -> equivalent to rotated gist space
        # For EXACT identity gist, override: zc = spL raw, zf = gl2 raw
        zc = (sp_g).astype(np.float32); zf = gl2.astype(np.float32)       # identity: raw L2 gist
        zc_t = torch.from_numpy(zc).to(dev)
    TR = np.array([p for m in tasks_meta for p in m["tr"]]); rng.shuffle(TR)
    VA = np.array([p for m in tasks_meta for p in m["va"]])

    # ---------- main training: proto teacher + conditioned MDN + AdaLN generator ----------
    extra = (args.task_emb_dim if "task" in conds else 0) + (args.code_dim if "prevz" in conds else 0)
    fwd = MilestoneGenerator(din, args.code_dim).to(dev)
    predm = MilestonePredictor(din + extra, args.code_dim, args.K).to(dev)
    temb = nn.Embedding(len(datasets), args.task_emb_dim).to(dev) if "task" in conds else None
    o1 = torch.optim.AdamW(fwd.parameters(), lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(list(predm.parameters()) + (list(temb.parameters()) if temb else []), lr=2e-4, weight_decay=1e-5)

    def pred_in(fr_idx, cm_global):
        xs = [gist_all[fr_idx].to(dev)]
        if temb is not None:
            xs.append(temb(torch.from_numpy(frame_task[fr_idx]).long().to(dev)))
        if "prevz" in conds:
            xs.append(zc_t[torch.from_numpy(cm_global).long().to(dev)])     # teacher-forced prev-milestone code
        return torch.cat(xs, 1)

    for step in range(args.steps):
        b = TR[torch.randint(0, len(TR), (64,)).numpy()]
        ca, cb_, gcm, gnm = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        Gc = GZ[ca].float().to(dev); Gf = GZ[cb_].float().to(dev)
        z = zc_t[torch.from_numpy(gnm).long().to(dev)]
        gh = fwd(Gc, z)
        lift = torch.relu(cosr(gh.flatten(1), Gc.flatten(1)) - cosr(gh.flatten(1), Gf.flatten(1))).mean()
        l1 = F.smooth_l1_loss(gh, Gf) + args.lift_w * lift
        o1.zero_grad(); l1.backward(); o1.step()
        l2 = predm.nll(pred_in(ca, gcm), z); o2.zero_grad(); l2.backward(); o2.step()
    fwd.eval(); predm.eval()

    # ---------- P1 probe: progress direction u (temporal-ranking; CRAVE-free loss) + P2 module rivals ----------
    zf_t = torch.from_numpy(zf).to(dev); gn_t = gist_all.to(dev)
    u = nn.Parameter(torch.randn(args.code_dim, device=dev) / np.sqrt(args.code_dim))
    wg = nn.Parameter(torch.randn(din, device=dev) / np.sqrt(din))           # rival 1: linear on gist
    vmlp = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, 1)).to(dev)  # rival 2: MLP on gist
    op = torch.optim.AdamW([u, wg] + list(vmlp.parameters()), lr=1e-3)
    for step in range(args.probe_steps):
        b = TR[torch.randint(0, len(TR), (256,)).numpy()]
        ca, cb_ = b[:, 0], b[:, 1]
        dz = (zf_t[cb_] @ u) - (zf_t[ca] @ u)
        dg = (gn_t[cb_] @ wg) - (gn_t[ca] @ wg)
        dm = (vmlp(gn_t[cb_]) - vmlp(gn_t[ca])).squeeze(-1)
        loss = F.softplus(-dz).mean() + F.softplus(-dg).mean() + F.softplus(-dm).mean()
        op.zero_grad(); loss.backward(); op.step()

    # reach-head rival + external verifier rival
    rhead = nn.Sequential(nn.Linear(4 * args.code_dim, 256), nn.GELU(), nn.Linear(256, 1)).to(dev)
    vrf = nn.Sequential(nn.Linear(din + args.code_dim, 512), nn.GELU(), nn.Linear(512, 1)).to(dev)
    orv = torch.optim.AdamW(list(rhead.parameters()) + list(vrf.parameters()), lr=1e-3)
    ncm = torch.from_numpy(np.array([m["msoff"] for m in tasks_meta])).to(dev)

    def rin(zq, zcand):
        return torch.cat([zq, zcand, zq - zcand, (zq - zcand).abs()], 1)

    for step in range(args.probe_steps):
        b = TR[torch.randint(0, len(TR), (256,)).numpy()]
        ca, cb_, gcm, gnm = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        keep = gcm != gnm
        ca, cb_, gcm, gnm = ca[keep], cb_[keep], gcm[keep], gnm[keep]
        zc_nm = zc_t[torch.from_numpy(gnm).long().to(dev)]
        # reach-head: pos=(target frame, next-center) neg=(current frame, next-center)
        lp = rhead(rin(zf_t[cb_], zc_nm)).squeeze(-1); ln = rhead(rin(zf_t[ca], zc_nm)).squeeze(-1)
        lr_ = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + \
              F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
        # verifier: pos=(gist_c, true next center) negs=backward center (same task) + wrong-task center
        back = np.array([rng.choice(np.where((task_of_ms == task_of_ms[m]) & (progn_g <= progn_g[c]))[0])
                         for m, c in zip(gnm, gcm)])
        wrng = np.array([rng.choice(np.where(task_of_ms != task_of_ms[m])[0]) for m in gnm]) \
            if len(datasets) > 1 else back
        gp = gn_t[ca]
        vp = vrf(torch.cat([gp, zc_nm], 1)).squeeze(-1)
        vb = vrf(torch.cat([gp, zc_t[torch.from_numpy(back).long().to(dev)]], 1)).squeeze(-1)
        vw = vrf(torch.cat([gp, zc_t[torch.from_numpy(wrng).long().to(dev)]], 1)).squeeze(-1)
        lv = F.binary_cross_entropy_with_logits(vp, torch.ones_like(vp)) + \
             0.5 * (F.binary_cross_entropy_with_logits(vb, torch.zeros_like(vb)) +
                    F.binary_cross_entropy_with_logits(vw, torch.zeros_like(vw)))
        orv.zero_grad(); (lr_ + lv).backward(); orv.step()

    # ---------- EXAM (val; CRAVE labels used ONLY here) ----------
    res = {"tag": args.tag, "datasets": datasets, "cond": conds, "total_M": total_M, "steps": args.steps}
    with torch.no_grad():
        vaa, vab, gcm, gnm = VA[:, 0], VA[:, 1], VA[:, 2], VA[:, 3]
        keep = gcm != gnm
        # -- progress monotonicity (temporal order accuracy) --
        un = (u / u.norm()).detach()
        res["mono_u_code"] = round(float(((zf_t[vab] @ un) > (zf_t[vaa] @ un)).float().mean()), 4)
        res["mono_lin_gist"] = round(float(((gn_t[vab] @ wg) > (gn_t[vaa] @ wg)).float().mean()), 4)
        res["mono_mlp_gist"] = round(float((vmlp(gn_t[vab]) > vmlp(gn_t[vaa])).float().mean()), 4)
        # CRAVE-value agreement of u (exam only): corr(u·zc, progn) per task
        cors = [float(np.corrcoef((zc[m["msoff"]:m["msoff"] + m["M"]] @ un.cpu().numpy()), m["progn"])[0, 1])
                for m in tasks_meta]
        res["u_vs_crave_corr"] = {m["name"]: round(c, 3) for m, c in zip(tasks_meta, cors)}
        # -- forward cone --
        fc_pos = ((zc_t[gnm[keep]] - zf_t[vaa[keep]]) @ un).cpu().numpy()
        back_v = np.array([rng.choice(np.where((task_of_ms == task_of_ms[m]) & (progn_g <= progn_g[c]))[0])
                           for m, c in zip(gnm[keep], gcm[keep])])
        fc_neg = ((zc_t[back_v] - zf_t[vaa[keep]]) @ un).cpu().numpy()
        res["forward_cone_pass_gt"] = round(float((fc_pos > 0).mean()), 4)
        res["forward_cone_auc"] = round(auc(fc_pos, fc_neg), 4)
        # -- reach: code distance vs reach-head --
        pos_d = (zf_t[vab[keep]] - zc_t[gnm[keep]]).norm(dim=1).cpu().numpy()
        neg_d = (zf_t[vaa[keep]] - zc_t[gnm[keep]]).norm(dim=1).cpu().numpy()
        res["reach_dist_f1"] = best_f1(pos_d, neg_d); res["reach_dist_auc"] = round(auc(-pos_d, -neg_d), 4)
        rp = rhead(rin(zf_t[vab[keep]], zc_t[gnm[keep]])).squeeze(-1).cpu().numpy()
        rn = rhead(rin(zf_t[vaa[keep]], zc_t[gnm[keep]])).squeeze(-1).cpu().numpy()
        res["reach_head_f1"] = best_f1(-rp, -rn); res["reach_head_auc"] = round(auc(rp, rn), 4)
        # nearest-center identity of frame codes (is code space a valid tracker?)
        d_all = torch.cdist(zf_t, zc_t); ncid = d_all.argmin(1).cpu().numpy()
        res["frame_nearest_center_top1"] = round(float((ncid == frame_lms_g).mean()), 4)
        # -- MDN density vs external verifier (verify task AUC) --
        pin = pred_in(vaa[keep], gcm[keep])
        dp = per_sample_nll(predm, pin, zc_t[gnm[keep]]).cpu().numpy()
        db = per_sample_nll(predm, pin, zc_t[back_v]).cpu().numpy()
        wr_v = np.array([rng.choice(np.where(task_of_ms != task_of_ms[m])[0]) for m in gnm[keep]]) \
            if len(datasets) > 1 else back_v
        dw = per_sample_nll(predm, pin, zc_t[wr_v]).cpu().numpy()
        res["verify_density_auc_back"] = round(auc(dp, db), 4)
        res["verify_density_auc_wrongtask"] = round(auc(dp, dw), 4)
        vps = vrf(torch.cat([gn_t[vaa[keep]], zc_t[gnm[keep]]], 1)).squeeze(-1).cpu().numpy()
        vbs = vrf(torch.cat([gn_t[vaa[keep]], zc_t[back_v]], 1)).squeeze(-1).cpu().numpy()
        vws = vrf(torch.cat([gn_t[vaa[keep]], zc_t[wr_v]], 1)).squeeze(-1).cpu().numpy()
        res["verify_head_auc_back"] = round(auc(vps, vbs), 4)
        res["verify_head_auc_wrongtask"] = round(auc(vps, vws), 4)
        # -- calibration + entropy gate: MDN confidence vs code-space identity correctness --
        logit, mu, _ = predm(pred_in(vaa, gcm))
        pi = F.softmax(logit, -1); conf = pi.max(1).values.cpu().numpy()
        ent = (-(pi * (pi + 1e-9).log()).sum(1) / np.log(args.K)).cpu().numpy()
        zdep = mu[torch.arange(len(vaa)), logit.argmax(-1)]
        pred_id = torch.cdist(zdep, zc_t).argmin(1).cpu().numpy()
        correct = (pred_id == gnm).astype(np.float32)
        res["deploy_id_code_top1"] = round(float(correct.mean()), 4)
        bins = np.clip((conf * 10).astype(int), 0, 9); ece = 0.0
        for bi in range(10):
            mset = bins == bi
            if mset.any():
                ece += mset.mean() * abs(correct[mset].mean() - conf[mset].mean())
        res["mdn_ece"] = round(float(ece), 4)
        res["entropy_mean"] = round(float(ent.mean()), 4)
        res["entropy_gate_frac@0.9"] = round(float((ent > 0.9).mean()), 4)
        res["acc_gated_vs_all"] = {"all": round(float(correct.mean()), 4),
                                   "kept@ent<0.9": round(float(correct[ent <= 0.9].mean()), 4) if (ent <= 0.9).any() else None}
        # -- density-based ABSTENTION (replacement for dead entropy gate): deploy-code log-density vs correctness --
        dep_dens = per_sample_nll(predm, pred_in(vaa, gcm), zdep).cpu().numpy()
        res["abstain_density_auc"] = round(auc(dep_dens[correct == 1], dep_dens[correct == 0]), 4)  # >0.5 => density separates right/wrong
        for q in (0.3, 0.5):                                               # keep top-(1-q) by density; is kept-acc higher?
            th = np.quantile(dep_dens, q); kept = dep_dens >= th
            res[f"acc_density_kept@drop{int(q*100)}"] = round(float(correct[kept].mean()), 4)
        # -- identity-CHANGE reach event (rival to code-distance): at target frame, is nearest center the next-ms? --
        res["reach_idchange_next@target"] = round(float((ncid[vab] == gnm)[keep].mean()), 4)
        res["reach_idchange_cur@current"] = round(float((ncid[vaa] == gcm)[keep].mean()), 4)  # should be high if code tracks id
        # -- standard per-task eval (comparability with train_multitask runs) --
        f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
        def cn(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
        per_task = {}
        for m in tasks_meta:
            va = np.array(m["va"]); spL = m["spL"]
            a, b_, cm_, nm_, lcm, lnm = va[:, 0], va[:, 1], va[:, 2], va[:, 3], va[:, 4], va[:, 5]
            cds, cps, idp = [], [], []
            for s in range(0, len(a), 128):
                Gc = GZ[a[s:s + 128]].float().to(dev); Gf = GZ[b_[s:s + 128]].float().to(dev)
                zdep_b = predm.deploy_mean(pred_in(a[s:s + 128], cm_[s:s + 128]))
                gh = fwd(Gc, zdep_b)
                cds.append(cn(f(gh), f(Gf))); cps.append(cn(f(Gc), f(Gf)))
                idp.append(gh.mean((2, 3)).cpu().numpy())
            idp = np.concatenate(idp); idp /= (np.linalg.norm(idp, axis=1, keepdims=True) + 1e-8)
            per_task[m["name"]] = {"deploy": round(float(np.concatenate(cds).mean()), 4),
                                   "persistence": round(float(np.concatenate(cps).mean()), 4),
                                   "identity_topN": topn_hit(idp @ spL.T, lnm), "n_val": len(va)}
        res["per_task"] = per_task
        res["train_deploy_mean"] = round(float(np.mean([v["deploy"] for v in per_task.values()])), 4)
        res["train_id_top3_mean"] = round(float(np.mean([v["identity_topN"]["top3"] for v in per_task.values()])), 4)

    outp = REPO / f"lmwm/outputs/lmwm2/{args.tag}.json"; outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    if args.save_ckpt:
        ck = {"fwd": fwd.state_dict(), "predm": predm.state_dict(), "u": un.cpu().numpy(), "Wproj": Wproj,
              "temb": temb.state_dict() if temb else None, "cond": conds, "gmu": gmu, "gsd": gsd,
              "din": din, "code_dim": args.code_dim, "K": args.K, "datasets": datasets, "total_M": total_M}
        cp = REPO / f"lmwm/checkpoints/{args.tag}.pt"; torch.save(ck, cp)
        print(f"saved -> {cp}", flush=True)


if __name__ == "__main__":
    main()
