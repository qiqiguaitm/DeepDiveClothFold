"""Job6: v2 r 场的【末段终端目标】跨本体检验 —— LIBERO 成立 ≠ kai0 成立。

v2(recurrence_field_architecture_v2 §2 ④ / p1_libero_rvalley_pairs.py:61)末段目标 = **末帧**。
LIBERO 末帧≈完成态所以没问题;但 kai0 末帧被机械臂回 home 污染,
而 home 姿态是 r 最高的(每条 ep 首尾都经过)-> 末段 r-脊可能塌到 home。
本脚本在两个本体上同口径量:末段 r-脊落在哪、末帧 vs 脊 谁更接近"真完成态"。

参照系 = CRAVE v1 双锚 Viterbi 进度标签(kai0 上 corr 0.943 vs 监督 GT, 是可信的进度尺)。
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

REPO = Path("/home/tim/workspace/deepdive_kai0/lmvla")
sys.path.insert(0, str(REPO / "lmwm/scripts")); sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import load_index  # noqa: E402

THR = 0.03


def r_and_segments(gd):
    """与 p1_libero_rvalley_pairs.r_and_segments 逐字同算法。"""
    eps = list(gd)
    F = np.concatenate([gd[e] for e in eps]).astype(np.float32)
    F = F / (np.linalg.norm(F, axis=-1, keepdims=True) + 1e-9)
    ep = np.concatenate([np.full(len(gd[e]), i) for i, e in enumerate(eps)]); ne = len(eps)
    lens = [len(gd[e]) for e in eps]; offs = np.cumsum([0] + lens)
    D = cdist(F, F); dmin = np.full((len(F), ne), 1e9, np.float32)
    for j in range(ne):
        dmin[:, j] = D[:, ep == j].min(1)
    other = ep[:, None] != np.arange(ne)[None]
    sig = np.median(dmin[other])
    r = (np.exp(-dmin ** 2 / (2 * sig * sig)) * other).sum(1) / (ne - 1)
    res = {}
    for i, e in enumerate(eps):
        s, en = offs[i], offs[i + 1]; n = en - s; rr = r[s:en]
        v, _ = find_peaks(-gaussian_filter1d(rr, 1.4), prominence=THR, distance=max(2, n // 12))
        seg = [0] + list(v) + [n]
        ridge = [a + int(np.argmax(rr[a:b])) for a, b in zip(seg[:-1], seg[1:])]
        res[e] = (seg, ridge, n, rr)
    return res


def report(name, res, prog=None):
    """prog[e] = 该 ep 逐帧进度参照(可选)。"""
    rid, lastf, rid_prog, last_prog, rid_r, last_r, start_sim = [], [], [], [], [], [], []
    for e, (seg, ridge, n, rr) in res.items():
        t = ridge[-1]                                   # 末段 r-脊
        rid.append(t / (n - 1)); lastf.append(1.0)
        rid_r.append(float(rr[t])); last_r.append(float(rr[n - 1]))
        if prog is not None and e in prog:
            p = prog[e]
            rid_prog.append(float(p[min(t, len(p) - 1)])); last_prog.append(float(p[-1]))
    print(f"\n=== {name} ({len(res)} ep) ===")
    print(f"末段 r-脊的相对位置 mean={np.mean(rid):.3f} median={np.median(rid):.3f} "
          f"(1.0=就是末帧)  脊==末帧的 ep 占比={np.mean(np.array(rid) > 0.995):.3f}")
    print(f"r 值: 脊 {np.mean(rid_r):.4f} vs 末帧 {np.mean(last_r):.4f}")
    if rid_prog:
        print(f"进度参照(CRAVE v1 双锚标签): 脊处 {np.mean(rid_prog):.3f} vs 末帧 {np.mean(last_prog):.3f}")
        print(f"  -> 末帧进度 < 脊进度 的 ep 占比 = "
              f"{np.mean(np.array(last_prog) < np.array(rid_prog)):.3f}  (高=末帧倒退,锚末帧有害)")
    return dict(ridge_pos=float(np.mean(rid)), ridge_is_last=float(np.mean(np.array(rid) > 0.995)),
                ridge_prog=float(np.mean(rid_prog)) if rid_prog else None,
                last_prog=float(np.mean(last_prog)) if last_prog else None)


def main():
    out = {}

    # ---- LIBERO10 (v2 的验证场, 作对照) ----
    E, FR, Fn = load_index(REPO / "crave/data/libero10_dinov3base")
    import json
    R = "/vePFS/tim/workspace/LIBERO_fastwam/libero_10_no_noops_lerobot"
    tn = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{R}/meta/tasks.jsonl")}
    n2i = {v: k for k, v in tn.items()}
    ep2t = {json.loads(l)["episode_index"]: n2i[json.loads(l)["tasks"][0]] for l in open(f"{R}/meta/episodes.jsonl")}
    eps = sorted({int(e) for e in np.unique(E) if ep2t.get(int(e)) == 0})[:30]     # 单任务(r场按任务算)
    gd = {}
    for e in eps:
        loc = np.where(E == e)[0]; gd[e] = Fn[loc[np.argsort(FR[loc])]]
    out["libero10_task0"] = report("LIBERO10 task0 (v2 原生验证场)", r_and_segments(gd))

    # ---- kai0 (单一折叠任务, 有回 home) ----
    E2, FR2, Fn2 = load_index(REPO / "crave/data/kai_dinov3base")
    eps2 = sorted({int(e) for e in np.unique(E2)})[:30]
    gd2, prog = {}, {}
    lab = REPO.parent / "temp/crave_ae_labels/final"
    for e in eps2:
        loc = np.where(E2 == e)[0]; o = loc[np.argsort(FR2[loc])]
        sub = o[::5]                                                  # 30Hz -> 6Hz, 控 cdist O(n²)
        gd2[e] = Fn2[sub]
        f = lab / f"ep{e}.npy"
        if f.exists():
            p = np.load(f)
            idx = np.minimum((FR2[sub]).astype(int), len(p) - 1)
            prog[e] = p[idx]
    print(f"\n[kai0] {len(gd2)} ep, 进度参照可用 {len(prog)} ep", flush=True)
    out["kai0"] = report("kai0 折叠 (有回 home)", r_and_segments(gd2), prog if prog else None)

    import json as _j
    (REPO / "lmwm/outputs/job6_rfield_terminal.json").write_text(_j.dumps(out, indent=1))
    print(f"\n[save] {REPO}/lmwm/outputs/job6_rfield_terminal.json")


if __name__ == "__main__":
    main()
