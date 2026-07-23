"""Job4-A(CPU): CRAVE 最新「双锚」读出 vs LMWM 现用「单锚(hard_start)」的分割结构对比。

CRAVE final_architecture §1/§2.10 最新方案 = 起点锚(→0) + 终点锚(→1) 的 Viterbi;
LMWM build_pairs_abl 目前只有 hard_start,没有终点锚 —— 这是唯一未对齐的一环。
本脚本只比分割结构(不训练),回答:加终点锚后最终段是否变小/是否落到真完成态。
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/tim/workspace/deepdive_kai0/lmvla")
sys.path.insert(0, str(REPO / "lmwm/scripts")); sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import load_index  # noqa: E402
from crave.utils.dp import forward_penalty  # noqa: E402


def viterbi_fwd2(emit, values, up=3.0, down=25.0, hard_start=True, hard_end=False):
    """viterbi_forward + 可选 hard_end(强制末帧落在最高 value 状态) —— CRAVE 双锚。"""
    NF, S = emit.shape
    pen = forward_penalty(values, up, down)
    cost = np.full(S, 1e9)
    if hard_start:
        s0 = int(values.argmin()); cost[s0] = emit[0, s0]
    else:
        cost = emit[0].copy()
    bp = np.zeros((NF, S), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen
        k = tr.argmin(1)
        cost = emit[j] + tr[np.arange(S), k]
        bp[j] = k
    st = np.zeros(NF, int)
    st[-1] = int(values.argmax()) if hard_end else int(cost.argmin())
    for j in range(NF - 2, -1, -1):
        st[j] = bp[j + 1, st[j + 1]]
    return st


def segs(ms):
    ch = np.where(np.diff(ms) != 0)[0] + 1
    return np.concatenate([[0], ch]), np.concatenate([ch, [len(ms)]])


def main():
    fdir = REPO / "crave/data/libero10_dinov3base"
    graph = REPO / "lmwm/data/recurrence_graphs/libero10_dinov3base/recurrence_graph.npz"
    E, FR, Fn = load_index(fdir)
    g = np.load(graph)
    proto = g["prototype_table"].astype(np.float32); pord = g["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    print(f"[data] {len(Fn)} frames, {len(np.unique(E))} eps, M={len(protoL)} "
          f"pord=[{pord.min():.3f},{pord.max():.3f}]", flush=True)

    # ---- CRAVE 双锚:起点锚=全ep首3帧均值→0, 终点锚=全ep末3帧均值→1 ----
    order_by_ep = {}
    for ep in np.unique(E):
        loc = np.where(E == ep)[0]
        order_by_ep[int(ep)] = loc[np.argsort(FR[loc])]
    a_st = np.mean([Fn[o[:3]].mean(0) for o in order_by_ep.values()], axis=0)
    a_en = np.mean([Fn[o[-3:]].mean(0) for o in order_by_ep.values()], axis=0)
    a_st /= np.linalg.norm(a_st) + 1e-8; a_en /= np.linalg.norm(a_en) + 1e-8
    print(f"[anchor] cos(start,end)={float(a_st @ a_en):.4f}   "
          f"(高=首末别名严重, CRAVE 靠全局路径+末锚消解)", flush=True)
    d2 = np.linalg.norm(protoL - a_en, axis=1)
    print(f"[anchor] 终点锚最近的 milestone: id={int(d2.argmin())} pord={pord[d2.argmin()]:.3f} "
          f"dist={d2.min():.3f}", flush=True)

    protoD = np.concatenate([protoL, a_st[None], a_en[None]], 0)
    pordD = np.concatenate([pord, [0.0], [1.0]]).astype(np.float32)
    AS, AE = len(protoL), len(protoL) + 1

    # ---- 变体 C:per-task 锚(libero10 是 10 任务混合, 全局末帧均值是跨任务糊平均) ----
    import json as _j
    R = "/vePFS/tim/workspace/LIBERO_fastwam/libero_10_no_noops_lerobot"
    tname = {_j.loads(l)["task_index"]: _j.loads(l)["task"] for l in open(f"{R}/meta/tasks.jsonl")}
    n2i = {v: k for k, v in tname.items()}
    ep2t = {_j.loads(l)["episode_index"]: n2i[_j.loads(l)["tasks"][0]] for l in open(f"{R}/meta/episodes.jsonl")}
    tids = sorted(set(ep2t.values()))
    ast_t, aen_t = {}, {}
    for t in tids:
        os_ = [o for e_, o in order_by_ep.items() if ep2t.get(e_) == t]
        s = np.mean([Fn[o[:3]].mean(0) for o in os_], 0); e_ = np.mean([Fn[o[-3:]].mean(0) for o in os_], 0)
        ast_t[t] = s / (np.linalg.norm(s) + 1e-8); aen_t[t] = e_ / (np.linalg.norm(e_) + 1e-8)
    print(f"[anchorC] per-task 锚 {len(tids)} 组; "
          f"任务间末锚 cos mean="
          f"{np.mean([aen_t[a] @ aen_t[b] for a in tids for b in tids if a < b]):.4f}", flush=True)

    rows = []
    for ep, o in order_by_ep.items():
        Fq = Fn[o]
        e1 = np.linalg.norm(Fq[:, None] - protoL[None], axis=2)
        e2 = np.linalg.norm(Fq[:, None] - protoD[None], axis=2)
        m1 = viterbi_fwd2(e1, pord, hard_start=True, hard_end=False)              # 现状
        m2 = viterbi_fwd2(e2, pordD, hard_start=True, hard_end=True)              # CRAVE 双锚(全局)
        t = ep2t.get(int(ep))
        protoC = np.concatenate([protoL, ast_t[t][None], aen_t[t][None]], 0)      # 变体C: per-task 锚
        e3 = np.linalg.norm(Fq[:, None] - protoC[None], axis=2)
        m3 = viterbi_fwd2(e3, pordD, hard_start=True, hard_end=True)
        n = len(m1)
        s1, en1 = segs(m1); s2, en2 = segs(m2); s3, en3 = segs(m3)
        # 真尾巴 = 最后一个【非锚】段(锚段只吃首/末几帧, 不能算"尾巴被治好")
        def last_real(s, e, ms, bad):
            for i in range(len(s) - 1, -1, -1):
                if ms[s[i]] not in bad:
                    return (e[i] - s[i]) / n
            return 0.0
        rows.append(dict(
            ep=int(ep), n=n,
            nseg1=len(s1), nseg2=len(s2),
            tail1=(en1[-1] - s1[-1]) / n, tail2=(en2[-1] - s2[-1]) / n,
            rtail1=last_real(s1, en1, m1, set()), rtail2=last_real(s2, en2, m2, {AS, AE}),
            nseg3=len(s3), rtail3=last_real(s3, en3, m3, {AS, AE}),
            endanchor_frac3=float((m3 == AE).mean()), task=int(t),
            lastp1=float(pord[m1[-1]]), lastp2=float(pordD[m2[-1]]),
            endanchor_frac=float((m2 == AE).mean()), startanchor_frac=float((m2 == AS).mean()),
        ))
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== 分割结构对比 (libero10, %d ep) ===" % len(df))
    print(f"{'指标':<28}{'现状(单锚)':>14}{'CRAVE双锚':>14}")
    print("-" * 58)
    print(f"{'段数 mean':<28}{df.nseg1.mean():>14.2f}{df.nseg2.mean():>14.2f}")
    print(f"{'最终段占比 mean':<28}{df.tail1.mean():>14.3f}{df.tail2.mean():>14.3f}")
    print(f"{'最终段占比 median':<28}{df.tail1.median():>14.3f}{df.tail2.median():>14.3f}")
    print(f"{'末帧 pord mean':<28}{df.lastp1.mean():>14.3f}{df.lastp2.mean():>14.3f}")
    print(f"{'尾巴>30% 的 ep 占比':<28}{(df.tail1 > .3).mean():>14.3f}{(df.tail2 > .3).mean():>14.3f}")
    print("-- 真尾巴(最后一个非锚段) --")
    print(f"{'真尾巴 mean':<28}{df.rtail1.mean():>14.3f}{df.rtail2.mean():>14.3f}")
    print(f"{'真尾巴 median':<28}{df.rtail1.median():>14.3f}{df.rtail2.median():>14.3f}")
    print(f"{'真尾巴>30% 占比':<28}{(df.rtail1 > .3).mean():>14.3f}{(df.rtail2 > .3).mean():>14.3f}")
    print(f"\n终点锚吸走帧占比: 全局锚 mean={df.endanchor_frac.mean():.3f}  "
          f"per-task 锚 mean={df.endanchor_frac3.mean():.3f}  (起点锚 {df.startanchor_frac.mean():.3f})")
    print("\n=== 变体C: per-task 锚 ===")
    print(f"{'段数 mean':<28}{df.nseg3.mean():>14.2f}")
    print(f"{'真尾巴 mean':<28}{df.rtail3.mean():>14.3f}")
    print(f"{'真尾巴>30% 占比':<28}{(df.rtail3 > .3).mean():>14.3f}")
    print("\n-- per-task 真尾巴: 现状 / 全局双锚 / per-task双锚 --")
    for t, sub in df.groupby("task"):
        print(f"  task {t:>2} (n={len(sub):>3}): {sub.rtail1.mean():.3f} / "
              f"{sub.rtail2.mean():.3f} / {sub.rtail3.mean():.3f}   末锚吸帧 {sub.endanchor_frac3.mean():.3f}")
    df.to_json(REPO / "lmwm/outputs/job4_dualanchor_seg.json", orient="records")
    print(f"[save] {REPO}/lmwm/outputs/job4_dualanchor_seg.json", flush=True)


if __name__ == "__main__":
    main()
