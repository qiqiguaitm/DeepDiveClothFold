#!/usr/bin/env python3
"""X-VLA 真机 trace: 模型输出 (预测 chunk) vs 真正下发的 PosCmd — 过滤量 + 一致性。

模型每次 infer 出 H=30 帧 (pose14), 但 PosCmd 按 publish_rate(~30Hz) 抽帧下发, 且
infer ~3Hz → 新 chunk 到来时 StreamActionBuffer 替换掉旧 chunk 未消费的帧。本脚本量化:
  (1) 过滤量: 模型预测总帧 vs 实发 vs jump-guard 丢 vs 被新 chunk 取代(未消费)
  (2) 每 chunk 实际消费几帧 + 落在 chunk 的哪些 horizon 索引 (RTC 只用前缀?)
  (3) 一致性: 每条实发 PosCmd 对到它"活跃 chunk"内最近的模型帧, 看值是否被改
      (firmware 分支应跳过 EMA/jump-protect → 逐字透传, |Δ|≈0; >0 说明有平滑/混合)

只用 client 侧 trace (client_trace.jsonl + client_arrays/), 不依赖 rosbag。
poscmd 事件的 t_mono 与 client_infer 的 t_ws_recv 同为 node 进程 time.monotonic() → 可对齐。

用法: python xvla/analyze_cmd_vs_output.py <trace_dir>
"""
from __future__ import annotations
import sys, os, json, glob
import numpy as np

CYAN, GREEN, YELLOW, RED, NC = "\033[0;36m", "\033[0;32m", "\033[1;33m", "\033[0;31m", "\033[0m"


def _load(d):
    infers, pos = [], []
    with open(os.path.join(d, "client_trace.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("stage") == "client_infer":
                infers.append(r)
            elif r.get("stage") == "poscmd":
                pos.append(r)
    return infers, pos


def main():
    if len(sys.argv) < 2:
        print("用法: python xvla/analyze_cmd_vs_output.py <trace_dir>", file=sys.stderr); sys.exit(1)
    d = sys.argv[1]
    infers, pos = _load(d)
    if not infers or not pos:
        print(f"{RED}缺 client_infer / poscmd 记录{NC}", file=sys.stderr); sys.exit(2)

    # 每个 infer 的 (t_ws_recv, seq, pose14[30,14])
    chunks = []
    for r in infers:
        seq = r.get("seq")
        npz = os.path.join(d, "client_arrays", f"{int(seq):06d}.npz")
        if not os.path.isfile(npz):
            continue
        p = np.load(npz).get("pose14")
        if p is None:
            continue
        t = r.get("t_ws_recv") or r.get("t_mono")
        chunks.append((float(t), int(seq), np.asarray(p, dtype=float)))
    chunks.sort()
    ct = np.array([c[0] for c in chunks])
    H = chunks[0][2].shape[0] if chunks else 0

    published = [r for r in pos if not r.get("dropped")]
    dropped = [r for r in pos if r.get("dropped")]
    n_pred = len(chunks) * H
    n_pub, n_drop = len(published), len(dropped)
    n_superseded = n_pred - n_pub - n_drop

    print(f"\n{CYAN}══════════ 模型输出 → 实发 PosCmd: 过滤 + 一致性 ══════════{NC}")
    print(f"trace: {d}")
    print(f"\n{CYAN}── 过滤量 ──{NC}")
    print(f"  推理 chunk: {len(chunks)} × H={H}  → 模型共预测 {n_pred} 帧")
    print(f"  实发 PosCmd:            {n_pub} 帧 ({n_pub/max(1,n_pred)*100:.0f}% of 预测)")
    print(f"  jump-guard 丢弃:        {n_drop} 帧 ({n_drop/max(1,n_pred)*100:.1f}%)")
    print(f"  被新 chunk 取代(未消费): {n_superseded} 帧 ({n_superseded/max(1,n_pred)*100:.0f}%)  ← 主要'过滤'")
    print(f"  注: 取代是 RTC/StreamBuffer 正常行为 (infer~3Hz×30帧 ≫ 发布30Hz), 非异常丢失。")

    # 消费量按【时间】归属 (robust): 每条实发 cmd 落到 t_ws_recv≤t 的最近 chunk。
    # 一致性按【到该 chunk 任一模型帧的最近距离】= 偏离"预测流形"的下界 (blend 后实发不等于任一原帧)。
    try:
        from scipy.spatial.transform import Rotation
        _have_scipy = True
    except Exception:
        _have_scipy = False
    per_chunk = {}
    cons_xyz, cons_rot = [], []
    unmatched = 0
    for r in published:
        t = r.get("t_mono")
        L, R = r.get("L"), r.get("R")
        if t is None or L is None or R is None:
            continue
        cmd14 = np.array(list(L) + list(R), dtype=float)
        idx = int(np.searchsorted(ct, t, side="right") - 1)
        if idx < 0:
            unmatched += 1; continue
        per_chunk[chunks[idx][1]] = per_chunk.get(chunks[idx][1], 0) + 1
        # 一致性: 取活跃 chunk(+前一个) 内 xyz 最近帧, 报该帧的 xyz/rot 偏差 (下界)
        best = None
        for _t, _seq, p in ([chunks[idx]] + ([chunks[idx-1]] if idx-1 >= 0 else [])):
            dxyz = np.linalg.norm(p[:, [0, 1, 2, 7, 8, 9]] - cmd14[[0, 1, 2, 7, 8, 9]], axis=1)
            h = int(np.argmin(dxyz))
            if best is None or dxyz[h] < best[0]:
                best = (dxyz[h], p[h])
        cons_xyz.append(best[0] * 1000.0)
        if _have_scipy:
            rc = Rotation.from_euler('xyz', cmd14[[3, 4, 5]])
            ra = Rotation.from_euler('xyz', best[1][[3, 4, 5]])
            lc = Rotation.from_euler('xyz', cmd14[[10, 11, 12]])
            la = Rotation.from_euler('xyz', best[1][[10, 11, 12]])
            cons_rot.append(max((rc*ra.inv()).magnitude(), (lc*la.inv()).magnitude()) * 180/np.pi)

    cons_xyz = np.asarray(cons_xyz)
    print(f"\n{CYAN}── 每 chunk 消费 (按时间归属) ──{NC}")
    consumed = np.array(list(per_chunk.values()))
    if consumed.size:
        print(f"  实际消费帧数/chunk: p50={np.median(consumed):.0f} min={consumed.min()} max={consumed.max()} "
              f"(≈publish_rate/infer_rate; 其余被新 chunk 取代)")

    print(f"\n{CYAN}── 一致性 (实发 PosCmd vs 模型预测帧) ──{NC}")
    def q(a, u): return f"p50={np.median(a):.2f}{u} p95={np.percentile(a,95):.2f}{u} max={a.max():.2f}{u}"
    if cons_xyz.size:
        print(f"  xyz 偏离最近预测帧: {q(cons_xyz,'mm')}")
        if cons_rot:
            print(f"  rot 偏离最近预测帧(测地): {q(np.asarray(cons_rot),'°')}")
        verbatim = np.mean(cons_xyz < 0.5) * 100
        print(f"  逐字透传比例 (xyz|Δ|<0.5mm): {verbatim:.0f}%  (其余被 chunk 边界 blend 改写)")
        print(f"  机制: StreamActionBuffer.integrate_new_chunk 对相邻 chunk 重叠段做 min-jerk 加权 blend")
        print(f"        (w_old·old + w_new·new) → 边界帧是两次预测的混合, 非任一原帧。位置 blend OK;")
        print(f"        {YELLOW}注: 它对 pose14 整体线性 blend, 含 euler rpy → 姿态非正确旋转插值 (应 slerp){NC}")
    if unmatched:
        print(f"  {YELLOW}注: {unmatched} 条实发早于首个 chunk, 未匹配{NC}")
    print()


if __name__ == "__main__":
    main()
