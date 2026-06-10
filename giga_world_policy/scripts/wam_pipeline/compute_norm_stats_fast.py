"""WAM 归一化统计的快速直算实现(绕开 lerobot DataLoader,直接读 parquet 向量化)。

为什么:慢管线(compute_norm_stats.py 走 giga LeRobotDataset)每取一帧都调 HF
datasets.select() 去 gather 48 帧 action 窗口,而 select() 每次深拷贝整个 DatasetInfo
(features 嵌套字典)→ 纯 CPU 串行开销 + 文件级 fingerprint 锁竞争,~200 it/s 封顶,
全量要数小时,且加 worker 会因锁竞争/超订反而卡死。

本脚本直接复刻同一套变换(已对慢管线 bit 级验证):
  - state 统计 = 原始 observation.state(14 维)。
  - action 统计 = 对每帧 t 取 episode 内 clamp 的 48 帧窗口 action[t:t+48](末尾重复最后帧),
    再做 DeltaActions:delta[i,d] = action_win[i,d] - (state_t[d] if mask[d] else 0)。
    (PadStatesAndActions 在 14→14 时为 no-op。)
mean/std 与慢管线 bit 级一致;q01/q99 为 5000-bin 直方图近似,因 update 分块不同会有
~1e-3 级差异(与"批量 vs 逐帧"同量级,均同等接近真值)。直算全量(默认 sample_rate=1.0)
还消除了采样噪声,分位数更准。全量 vis+kai 仅需几分钟。

用法:
  python -m scripts.wam_pipeline.compute_norm_stats_fast <emb_dir> <embodiment_id> <out_short> \
      [--delta-mask ...] [--sample-rate 1.0] [--action-chunk 48] [--action-dim 14]
默认 delta-mask = 14 维 piper(关节 delta、夹爪 index 6/13 绝对)。
"""

import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq

from scripts.compute_norm_stats import NormStats, RunningStats, serialize_json_with_mask  # 复用同一统计实现

PIPER_MASK = [True] * 6 + [False] + [True] * 6 + [False]


def episode_arrays(pqf, mask, chunk):
    """返回 (state (n,14), delta_action (n*chunk,14))。"""
    d = pq.read_table(pqf, columns=["observation.state", "action"]).to_pydict()
    S = np.asarray([np.asarray(x) for x in d["observation.state"]], dtype=np.float64)
    A = np.asarray([np.asarray(x) for x in d["action"]], dtype=np.float64)
    n, dims = A.shape
    idx = np.clip(np.arange(n)[:, None] + np.arange(chunk)[None, :], 0, n - 1)  # (n,chunk) 集内 clamp
    win = A[idx]                                                                 # (n,chunk,dims)
    masked_state = np.where(mask[:dims], S, 0.0)                                 # (n,dims)
    delta = win - masked_state[:, None, :]                                       # (n,chunk,dims)
    return S, delta.reshape(n * chunk, dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dir")
    ap.add_argument("embodiment_id", type=int)
    ap.add_argument("out_short")
    ap.add_argument("--data-base", default="../kai0/data/wam_fold_v1")
    ap.add_argument("--out-dir", default="./assets_visrobot01")
    ap.add_argument("--delta-mask", nargs="+", type=lambda s: s.lower() == "true", default=PIPER_MASK)
    ap.add_argument("--sample-rate", type=float, default=1.0)
    ap.add_argument("--action-chunk", type=int, default=48)
    ap.add_argument("--action-dim", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.join(args.data_base, args.emb_dir)
    info = json.load(open(f"{root}/meta/info.json"))
    cs, tmpl = info["chunks_size"], info["data_path"]
    eps = [json.loads(l)["episode_index"] for l in open(f"{root}/meta/episodes.jsonl") if l.strip()]
    mask = np.array(args.delta_mask, dtype=bool)
    assert len(mask) == args.action_dim, f"delta_mask 长度 {len(mask)} != action_dim {args.action_dim}"

    rng = np.random.default_rng(args.seed)
    stats = {k: RunningStats() for k in ("observation.state", "action")}
    n_state_rows = 0
    for ep in eps:
        pqf = os.path.join(root, tmpl.format(episode_chunk=ep // cs, episode_index=ep))
        S, dact = episode_arrays(pqf, mask, args.action_chunk)
        if args.action_dim > S.shape[1]:  # PadStatesAndActions 等价:零填充到 action_dim
            S = np.pad(S, ((0, 0), (0, args.action_dim - S.shape[1])))
            dact = np.pad(dact, ((0, 0), (0, args.action_dim - dact.shape[1])))
        if args.sample_rate < 1.0:
            # 按帧无放回采样(state 与对应 action 窗口一起取,保持与慢管线 sample_rate 语义一致)
            n = len(S)
            take = max(2, int(round(args.sample_rate * n)))
            sel = rng.choice(n, size=take, replace=False)
            S = S[sel]
            chunk = args.action_chunk
            arows = (sel[:, None] * chunk + np.arange(chunk)[None, :]).reshape(-1)
            dact = dact[arows]
        stats["observation.state"].update(S)
        stats["action"].update(dact)
        n_state_rows += len(S)

    out = {k: stats[k].get_statistics() for k in stats}
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"norm_stats_{args.out_short}.json")
    with open(out_path, "w") as f:
        f.write(serialize_json_with_mask(out, mask.tolist()))
    print(f"FAST_NORM_DONE {args.emb_dir} -> {out_path} (sample_rate={args.sample_rate}, state_rows={n_state_rows})")


if __name__ == "__main__":
    main()
