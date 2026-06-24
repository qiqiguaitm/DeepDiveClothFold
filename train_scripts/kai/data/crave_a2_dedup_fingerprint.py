"""A2-dedup 收尾: 用 milestone 段时长指纹替代朴素 reached-set 签名(后者 1109/1117 同签名,无区分度)。

指纹 = 每 ep 到达 20 个 milestone 的归一化时刻向量 (cross_frames / n_frames) ∈ [0,1]^20(单调)。
两条 demo 时间结构越像 → 指纹 L2 越小。验证:
  ① 指纹有区分度(pairwise 距离有展开,非全 0)→ 修复 reached-set 的退化;
  ② 能找到真近重复对(最小距离对的时序几乎重合);
  ③ 最近邻距离分布 → 可设阈值做 dedup。

输入: temp/crave_a1a2/segments.json(已由 crave_a1a2_validate.py 产出)
输出: temp/crave_a1a2/dedup_fingerprint.json + docs/.../crave_a2_dedup_fingerprint.png
Run: kai0/.venv/bin/python train_scripts/kai/data/crave_a2_dedup_fingerprint.py
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
SEG = REPO / "temp/crave_a1a2/segments.json"
OUT = REPO / "temp/crave_a1a2"
FIG = REPO / "docs/visualization/cross_episode_recurrence_value"


def main():
    seg = json.load(open(SEG))
    eps = sorted(int(e) for e in seg)
    nM = 20
    # 归一化到达时刻指纹 (cross_frame<0 的未到达 → 置 1.0=末尾)
    F = []
    for e in eps:
        d = seg[str(e)]; n = d["n_frames_3hz"]; cf = d["cross_frames_3hz"]
        v = np.array([(c / max(1, n - 1)) if c >= 0 else 1.0 for c in cf], float)
        F.append(v)
    F = np.array(F)                                   # (Nep, 20)
    # 朴素 reached-set 签名(对照)
    naive = [tuple(d["milestones_reached"]) for d in (seg[str(e)] for e in eps)]
    from collections import Counter
    naive_top = Counter(naive).most_common(1)[0][1]

    # pairwise 距离(指纹)
    D = np.linalg.norm(F[:, None] - F[None], axis=2)   # (Nep,Nep)
    iu = np.triu_indices(len(eps), 1); pd_ = D[iu]
    np.fill_diagonal(D, np.inf); nn = D.min(1); nn_idx = D.argmin(1)

    # 最近重复对
    k = int(np.argmin(pd_)); i, j = iu[0][k], iu[1][k]
    closest = {"ep_a": eps[i], "ep_b": eps[j], "dist": float(pd_[k]),
               "fp_a": [round(x, 3) for x in F[i]], "fp_b": [round(x, 3) for x in F[j]]}
    # dedup 阈值演示: NN 距离 < q5 视作近重复候选
    thr = float(np.percentile(nn, 5))
    n_near_dup = int((nn < thr).sum())

    out = {
        "n_eps": len(eps), "fingerprint_dim": nM,
        "naive_reached_set_top_group_size": naive_top,
        "naive_reached_set_note": f"{naive_top}/{len(eps)} 同签名 → 无区分度(被本指纹替代)",
        "fingerprint_pairwise_dist": {"mean": float(pd_.mean()), "std": float(pd_.std()),
                                       "min": float(pd_.min()), "p50": float(np.percentile(pd_, 50)),
                                       "max": float(pd_.max())},
        "fingerprint_nn_dist": {"mean": float(nn.mean()), "p5_thr": thr, "min": float(nn.min())},
        "n_near_dup_below_p5": n_near_dup,
        "closest_pair": closest,
        "discriminative": bool(pd_.std() > 0.02),
    }
    json.dump(out, open(OUT / "dedup_fingerprint.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(pd_, bins=60, color="C0", alpha=.8); ax[0].axvline(thr, color="C3", ls="--", label=f"p5 dedup thr={thr:.3f}")
    ax[0].set_title(f"segment-timing fingerprint: pairwise L2\nmean={pd_.mean():.2f} std={pd_.std():.2f} (discriminative)")
    ax[0].set_xlabel("fingerprint L2 distance"); ax[0].legend()
    ax[1].plot(F[i], "o-", label=f"ep{eps[i]}"); ax[1].plot(F[j], "s--", label=f"ep{eps[j]}")
    ax[1].set_title(f"closest pair (dist={pd_[k]:.3f}) - near-duplicate timing"); ax[1].set_xlabel("milestone idx"); ax[1].set_ylabel("norm. reach time"); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIG / "crave_a2_dedup_fingerprint.png", dpi=110); plt.close()
    print(f"-> {OUT}/dedup_fingerprint.json + {FIG}/crave_a2_dedup_fingerprint.png", flush=True)


if __name__ == "__main__":
    main()
