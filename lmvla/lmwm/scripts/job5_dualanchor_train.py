"""Job5: CRAVE 最新「双锚」标签 vs 现用单锚标签 —— 同 recipe 重训 + 同口径评测(libero10)。

Arm A (base)  : build_pairs_abl 现状(hard_start + 最终段 self-loop)
Arm B (anchor): + CRAVE final_architecture §1 的双锚, 但锚按 LIBERO task 分别算
                (全局锚在 10 任务混合数据上是跨任务糊平均 -> 实测只吸走 0.4% 帧, 见 job4)

共同判据 = 与标签无关的「全帧 -> 该 ep 真实末帧」deploy vs persist(job3 口径),
两臂可直接相减; 另附各自 val pair 上的 per-task deploy/lift。
用法: CUDA_VISIBLE_DEVICES=0 python job5_dualanchor_train.py --arm base
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/tim/workspace/deepdive_kai0/lmvla")
sys.path.insert(0, str(REPO / "lmwm/scripts")); sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import load_index  # noqa: E402
from train_multitask import _read_libero  # noqa: E402
from train_twomodel_v2 import MilestoneGenerator, MilestonePredictor, cosr  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from job4_dualanchor_seg import viterbi_fwd2  # noqa: E402

ROOT = "/vePFS/tim/workspace/LIBERO_fastwam/libero_10_no_noops_lerobot"
FDIR = REPO / "crave/data/libero10_dinov3base"
GRAPH = REPO / "lmwm/data/recurrence_graphs/libero10_dinov3base/recurrence_graph.npz"
CAM = "observation.images.image"


def build_pairs(E, FR, Fn, protoL, pord, val_eps, seed, ep2t=None, anch=None):
    """(cur, tgt, cur_ms, next_ms) —— anch=None 走现状; 否则 per-task 双锚。"""
    tr, va = [], []
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
        Fq = Fn[order]
        if anch is None:
            P, PO, hend, gid = protoL, pord, False, None
        else:
            t = ep2t[int(ep)]; a_s, a_e = anch[t]
            P = np.concatenate([protoL, a_s[None], a_e[None]], 0)
            PO = np.concatenate([pord, [0.0], [1.0]]).astype(np.float32); hend = True
            # 锚 id 按 task 分开(否则 10 个任务的锚共用一个 teacher code = 又糊回去了)
            gid = {len(protoL): len(protoL) + 2 * t, len(protoL) + 1: len(protoL) + 2 * t + 1}
        ms = viterbi_fwd2(np.linalg.norm(Fq[:, None] - P[None], axis=2), PO,
                          up=3.0, down=25.0, hard_start=True, hard_end=hend)
        ch = np.where(np.diff(ms) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(ms)]])
        seg_med, seg_m, spans = [], [], []
        for s, e in zip(st, en):
            m = int(ms[s]); seg_med.append(int(order[s + int((Fq[s:e] @ P[m]).argmax())]))
            seg_m.append(m); spans.append((s, e))
        gm = (lambda m: gid.get(m, m)) if gid else (lambda m: m)
        dst = va if ep in val_eps else tr
        for i in range(len(seg_m) - 1):
            dst.append((int(order[spans[i][1] - 1]), seg_med[i + 1], gm(seg_m[i]), gm(seg_m[i + 1])))
        li = len(seg_m) - 1                                    # 最终段 self-loop(两臂一致)
        dst.append((int(order[spans[li][1] - 1]), seg_med[li], gm(seg_m[li]), gm(seg_m[li])))
    rng = np.random.default_rng(seed); rng.shuffle(tr); rng.shuffle(va)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "anchor"])
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--code_dim", type=int, default=128)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lift_w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    dev = "cuda"; cd = a.code_dim

    E, FR, Fn = load_index(FDIR)
    g = np.load(GRAPH); proto = g["prototype_table"].astype(np.float32); pord = g["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    tname = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{ROOT}/meta/tasks.jsonl")}
    n2i = {v: k for k, v in tname.items()}
    ep2t = {json.loads(l)["episode_index"]: n2i[json.loads(l)["tasks"][0]] for l in open(f"{ROOT}/meta/episodes.jsonl")}

    obe = {}
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]; obe[int(ep)] = loc[np.argsort(FR[loc])]
    anch = None
    if a.arm == "anchor":
        anch = {}
        for t in sorted(set(ep2t.values())):
            os_ = [o for e_, o in obe.items() if ep2t.get(e_) == t]
            s = np.mean([Fn[o[:3]].mean(0) for o in os_], 0); e_ = np.mean([Fn[o[-3:]].mean(0) for o in os_], 0)
            anch[t] = (s / (np.linalg.norm(s) + 1e-8), e_ / (np.linalg.norm(e_) + 1e-8))

    rng = np.random.default_rng(a.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    tr, va = build_pairs(E, FR, Fn, protoL, pord, val_eps, a.seed, ep2t, anch)
    if len(tr) > 8000:
        tr = [tr[i] for i in rng.choice(len(tr), 8000, replace=False)]
    if len(va) > 4000:
        va = [va[i] for i in rng.choice(len(va), 4000, replace=False)]
    M = len(protoL) + (2 * len(anch) if anch else 0)
    print(f"[{a.arm}] M={M} pairs tr={len(tr)} va={len(va)}", flush=True)

    # 共同判据用帧: 每个 val ep 均匀采 12 帧 + 该 ep 末帧
    common = []
    for ep in sorted(val_eps):
        o = obe[int(ep)]; n = len(o)
        for k in np.linspace(0, n - 2, 12).astype(int):
            common.append((int(o[k]), int(o[-1]), float(k) / (n - 1), int(ep)))
    print(f"[{a.arm}] common-yardstick frames: {len(common)}", flush=True)

    uniq = sorted(set([p[0] for p in tr + va] + [p[1] for p in tr + va]
                      + [c[0] for c in common] + [c[1] for c in common]))
    u2k = {gi: k for k, gi in enumerate(uniq)}
    print(f"[{a.arm}] encoding {len(uniq)} frames ...", flush=True)
    ie, _ = _read_libero(Path(ROOT), CAM, E, FR, np.array(uniq), 224, 128)
    enc = load_encoder("dinov3-base", device=dev)
    grids = enc.encode_grid(ie, bs=32).astype(np.float32)
    grids = grids / (np.linalg.norm(grids, axis=1, keepdims=True) + 1e-8)   # per-patch L2(终版约定)
    din = grids.shape[1]
    gmu, gsd = float(grids.mean()), float(grids.std() + 1e-6)
    GZ = torch.from_numpy(((grids - gmu) / gsd).astype(np.float32))
    gist = GZ.mean((2, 3))

    # proto teacher(shared PCA128), 与 libero10_lmwm_sharedpca 一致
    from sklearn.decomposition import PCA
    # Pall 行序必须与 build_pairs 的 gid 一致: protoL, 然后 [task0_start, task0_end, task1_start, ...]
    Pall = np.concatenate([protoL] + [anch[t][i][None] for t in sorted(anch) for i in (0, 1)], 0) if anch else protoL
    msid = (Fn[np.array(uniq)] @ Pall.T).argmax(1)
    gnp = grids.mean((2, 3))
    sp = np.stack([gnp[msid == m].mean(0) if (msid == m).any() else np.zeros(din, np.float32)
                   for m in range(len(Pall))])
    spL = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)
    gl2 = gnp / (np.linalg.norm(gnp, axis=1, keepdims=True) + 1e-8)
    pca = PCA(n_components=cd, random_state=a.seed).fit(gl2)
    zt = pca.transform(spL).astype(np.float32)
    zt = zt / (np.linalg.norm(zt, axis=1).mean() + 1e-8)
    zteach = torch.from_numpy(zt).to(dev)                       # id 已在 build_pairs 内映射到 Pall 行序

    fwd = MilestoneGenerator(din, cd).to(dev)
    predm = MilestonePredictor(din, cd, a.K).to(dev)
    o1 = torch.optim.AdamW(fwd.parameters(), lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(predm.parameters(), lr=2e-4, weight_decay=1e-5)
    TR = np.array([(u2k[c], u2k[t], cm, nm) for (c, t, cm, nm) in tr])
    for step in range(a.steps):
        b = TR[np.random.randint(0, len(TR), 64)]
        Gc = GZ[b[:, 0]].to(dev); Gf = GZ[b[:, 1]].to(dev)
        z = zteach[torch.from_numpy(b[:, 3]).long().to(dev)]
        gh = fwd(Gc, z)
        lift = torch.relu(cosr(gh.flatten(1), Gc.flatten(1)) - cosr(gh.flatten(1), Gf.flatten(1))).mean()
        l1 = F.smooth_l1_loss(gh, Gf) + a.lift_w * lift
        o1.zero_grad(); l1.backward(); o1.step()
        l2 = predm.nll(gist[b[:, 0]].to(dev), z.detach()); o2.zero_grad(); l2.backward(); o2.step()
        if step % 1500 == 0:
            print(f"  step {step}: rec={l1.item():.4f} nll={l2.item():.3f}", flush=True)
    fwd.eval(); predm.eval()

    def cn(x, y): return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-8)
    f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    res = {"arm": a.arm, "M": int(M), "n_tr": len(tr), "n_va": len(va)}

    # ---- 判据1: 共同口径(全帧 -> 该 ep 真实末帧), 与标签无关 ----
    bins = {}; per_t_c = {}
    with torch.no_grad():
        for s in range(0, len(common), 128):
            blk = common[s:s + 128]
            ca = np.array([u2k[c[0]] for c in blk]); cb = np.array([u2k[c[1]] for c in blk])
            Gc = GZ[ca].to(dev); Gf = GZ[cb].to(dev); gtr = f(Gf)
            d = cn(f(fwd(Gc, predm.deploy_mean(gist[ca].to(dev)))), gtr); p = cn(f(Gc), gtr)
            for k, c in enumerate(blk):
                bk = min(int(c[2] * 10), 9)
                bins.setdefault(bk, []).append(d[k] - p[k])
                t = ep2t[c[3]]; per_t_c.setdefault(t, {"d": [], "p": []})
                per_t_c[t]["d"].append(d[k]); per_t_c[t]["p"].append(p[k])
    res["common_bins"] = {f"{k*10}-{k*10+10}%": round(float(np.mean(v)), 4) for k, v in sorted(bins.items())}
    res["common_end_lift"] = round(float(np.mean(bins[9])), 4)
    res["common_all_lift"] = round(float(np.mean([x for v in bins.values() for x in v])), 4)
    res["common_per_task"] = {int(t): dict(deploy=round(float(np.mean(v["d"])), 4),
                                           persist=round(float(np.mean(v["p"])), 4),
                                           lift=round(float(np.mean(v["d"]) - np.mean(v["p"])), 4))
                              for t, v in sorted(per_t_c.items())}

    # ---- 判据2: 各自 val pair 上的 per-task deploy/lift ----
    per = {}
    with torch.no_grad():
        for s in range(0, len(va), 128):
            blk = va[s:s + 128]
            ca = np.array([u2k[p[0]] for p in blk]); cb = np.array([u2k[p[1]] for p in blk])
            Gc = GZ[ca].to(dev); Gf = GZ[cb].to(dev); gtr = f(Gf)
            d = cn(f(fwd(Gc, predm.deploy_mean(gist[ca].to(dev)))), gtr); p = cn(f(Gc), gtr)
            for k, pr in enumerate(blk):
                t = ep2t.get(int(E[pr[0]]), -1)
                per.setdefault(t, {"d": [], "p": []})
                per[t]["d"].append(d[k]); per[t]["p"].append(p[k])
    res["own_per_task"] = {int(t): dict(n=len(v["d"]), deploy=round(float(np.mean(v["d"])), 4),
                                        persist=round(float(np.mean(v["p"])), 4),
                                        lift=round(float(np.mean(v["d"]) - np.mean(v["p"])), 4))
                           for t, v in sorted(per.items())}
    ad = np.concatenate([per[t]["d"] for t in per]); apx = np.concatenate([per[t]["p"] for t in per])
    res["own_all"] = dict(deploy=round(float(ad.mean()), 4), persist=round(float(apx.mean()), 4),
                          lift=round(float(ad.mean() - apx.mean()), 4))

    sfx = "" if a.seed == 2026 else f"_s{a.seed}"
    out = REPO / f"lmwm/outputs/job5_dualanchor_{a.arm}{sfx}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()
