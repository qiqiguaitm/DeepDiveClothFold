#!/usr/bin/env python3
"""X-VLA 真机 pipeline trace 离线核验器。

读 `XVLA_TRACE_DIR` 下 server + client 两侧落盘的 trace, 按 seq join, 逐段对账
pipeline 是否有问题, 出一份 PASS/WARN/FAIL 报告。

用法:
    python xvla/analyze_pipeline_trace.py /tmp/xvla_stack/trace_YYYYmmdd_HHMMSS
    python xvla/analyze_pipeline_trace.py <dir> --dump-seq 42   # 打印某帧全字段

落盘布局 (由 serve_policy_xvla.py + policy_inference_node.py 写, 见各自 _PipeTrace):
    <dir>/
    ├── meta.json                          run 元信息 (ckpt / ts / git / execute / topics)
    ├── server_trace.jsonl                 server 每次 infer 一条 (stage="server_infer")
    ├── server_arrays/<seq>.npz            state14, state20, raw20(H,20), world16(H,16)
    ├── server_images/<seq>_<slot>.jpg     模型实际输入图 (resize_pad 后 256/256/224)
    ├── client_trace.jsonl                 stage="client_infer" (每次 infer) + "poscmd" (每帧下发)
    ├── client_arrays/<seq>.npz            state14_sent, ee_chunk_recv(H,16), pose14(H,14)
    ├── client_images/<seq>_<slot>.jpg     node 发出的 obs 图 (resize_with_pad 224 CHW)
    └── rosbag/                            ros2 bag (控制+状态 topic, 无相机)

核验的不变量:
    1. obs 完整性     client.state14_sent == server.state14            (websocket 上行无损)
    2. action 完整性  server.world16     == client.ee_chunk_recv       (websocket 下行无损)
    3. quat 单位模长  ‖quat‖ ≈ 1 (server world16 + client recv)
    4. chunk 形状一致 H 恒定 (sidecar action_chunk)
    5. pose14 合理域  xyz∈工作空间, rpy∈[-π,π], grip≥0
    6. 时序           infer 频率 / ws 往返开销 / server infer_ms 分布
    7. PosCmd 下发    发出 vs jump-guard 丢帧率 / 下发频率
    8. proprio 来源   sensed vs pred 分布 (firmware 模式应恒为 sensed)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

GREEN, YELLOW, RED, CYAN, NC = "\033[0;32m", "\033[1;33m", "\033[0;31m", "\033[0;36m", "\033[0m"


def _load_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _npz(arr_dir, seq):
    p = os.path.join(arr_dir, f"{int(seq):06d}.npz")
    return np.load(p) if os.path.isfile(p) else None


def _verdict(ok, warn=False):
    return f"{GREEN}PASS{NC}" if ok else (f"{YELLOW}WARN{NC}" if warn else f"{RED}FAIL{NC}")


def _stat(label, vals, unit="", fmt="{:.2f}"):
    if not vals:
        print(f"    {label}: (无数据)")
        return
    a = np.asarray(vals, dtype=float)
    print(f"    {label}: n={len(a)} "
          f"min={fmt.format(a.min())}{unit} "
          f"p50={fmt.format(np.median(a))}{unit} "
          f"p95={fmt.format(np.percentile(a, 95))}{unit} "
          f"max={fmt.format(a.max())}{unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--dump-seq", type=int, default=None, help="打印某 seq 的 server+client 全字段")
    ap.add_argument("--pos-workspace", type=float, default=1.2,
                    help="pose14 xyz 合理域 |coord| 上限 m (默认 1.2)")
    ap.add_argument("--grip-lo", type=float, default=-0.02,
                    help="pose14 gripper 物理下限 m (默认 -0.02; close_value≈-0.0055 是正常闭合, 非越界)")
    ap.add_argument("--grip-hi", type=float, default=0.10,
                    help="pose14 gripper 物理上限 m (默认 0.10; open_value≈0.066)")
    args = ap.parse_args()

    d = args.trace_dir
    if not os.path.isdir(d):
        print(f"{RED}trace dir 不存在: {d}{NC}", file=sys.stderr)
        sys.exit(1)

    meta = {}
    mp = os.path.join(d, "meta.json")
    if os.path.isfile(mp):
        meta = json.load(open(mp))

    srv = {r["seq"]: r for r in _load_jsonl(os.path.join(d, "server_trace.jsonl"))
           if r.get("stage") == "server_infer" and "seq" in r}
    cli_rows = _load_jsonl(os.path.join(d, "client_trace.jsonl"))
    cli = {r["seq"]: r for r in cli_rows if r.get("stage") == "client_infer" and "seq" in r}
    pos = [r for r in cli_rows if r.get("stage") == "poscmd"]

    srv_arr = os.path.join(d, "server_arrays")
    cli_arr = os.path.join(d, "client_arrays")

    print(f"\n{CYAN}══════════ X-VLA pipeline trace 核验 ══════════{NC}")
    print(f"trace_dir : {d}")
    if meta:
        print(f"ckpt      : {meta.get('ckpt_name')}  step={meta.get('step')}  "
              f"execute={meta.get('execute')}  dtype={meta.get('dtype')}")
        print(f"prompt    : {meta.get('prompt')!r}  domain_id={meta.get('domain_id')}  "
              f"chunk={meta.get('action_chunk')}")
        print(f"start     : {meta.get('start_wall_iso')}  git={meta.get('git_sha')}  host={meta.get('host')}")
    print(f"records   : server_infer={len(srv)}  client_infer={len(cli)}  poscmd={len(pos)}")

    common = sorted(set(srv) & set(cli))
    only_srv = sorted(set(srv) - set(cli))
    only_cli = sorted(set(cli) - set(srv))
    print(f"join(seq) : matched={len(common)}  only_server={len(only_srv)}  only_client={len(only_cli)}")

    if args.dump_seq is not None:
        s = args.dump_seq
        print(f"\n{CYAN}── dump seq={s} ──{NC}")
        print(f"  server: {json.dumps(srv.get(s, {}), ensure_ascii=False, indent=2)}")
        print(f"  client: {json.dumps(cli.get(s, {}), ensure_ascii=False, indent=2)}")
        sa, ca = _npz(srv_arr, s), _npz(cli_arr, s)
        if sa is not None:
            print(f"  server arrays: {[ (k, sa[k].shape) for k in sa.files ]}")
        if ca is not None:
            print(f"  client arrays: {[ (k, ca[k].shape) for k in ca.files ]}")
        return

    fails, warns = [], []

    # ── 1+2+3: 跨进程数组完整性 (obs 上行 / action 下行 / quat) ──
    print(f"\n{CYAN}── 1. 数组完整性 (websocket 上/下行 + quat) ──{NC}")
    obs_diffs, act_diffs, qn_all = [], [], []
    checked = 0
    for seq in common:
        sa, ca = _npz(srv_arr, seq), _npz(cli_arr, seq)
        if sa is None or ca is None:
            continue
        checked += 1
        if "state14" in sa and "state14_sent" in ca:
            n = min(sa["state14"].size, ca["state14_sent"].size)
            obs_diffs.append(float(np.max(np.abs(
                np.ravel(sa["state14"])[:n] - np.ravel(ca["state14_sent"])[:n]))))
        if "world16" in sa and "ee_chunk_recv" in ca:
            w, e = np.asarray(sa["world16"]), np.asarray(ca["ee_chunk_recv"])
            if w.shape == e.shape:
                act_diffs.append(float(np.max(np.abs(w - e))))
            for h in range(w.shape[0]):
                qn_all += [float(np.linalg.norm(w[h, 3:7])), float(np.linalg.norm(w[h, 11:15]))]
    obs_ok = (not obs_diffs) or max(obs_diffs) < 1e-3
    act_ok = (not act_diffs) or max(act_diffs) < 1e-4
    qn_ok = (not qn_all) or max(abs(np.asarray(qn_all) - 1.0)) < 1e-2
    print(f"  [{_verdict(obs_ok, warn=not obs_diffs)}] obs 上行  state14 client==server  "
          f"(checked={checked}, max|Δ|={max(obs_diffs) if obs_diffs else float('nan'):.2e})")
    print(f"  [{_verdict(act_ok, warn=not act_diffs)}] action 下行 world16==ee_recv      "
          f"(max|Δ|={max(act_diffs) if act_diffs else float('nan'):.2e})")
    print(f"  [{_verdict(qn_ok, warn=not qn_all)}] quat 模长 ‖q‖≈1                  "
          f"(max|‖q‖-1|={max(abs(np.asarray(qn_all)-1.0)) if qn_all else float('nan'):.2e})")
    if not obs_ok:
        fails.append("obs 上行 state14 不一致 — 序列化/对齐问题")
    if not act_ok:
        fails.append("action 下行 world16!=ee_recv — websocket 序列化问题")
    if not qn_ok:
        fails.append("quat 非单位模长 — rot6d→R 或四元数转换问题")
    if not checked:
        warns.append("无 server+client 配对数组 (npz) — 无法做跨进程完整性核验")

    # ── 4: chunk 形状 ──
    print(f"\n{CYAN}── 2. chunk 形状一致性 ──{NC}")
    hs = Counter(r.get("chunk_h") for r in cli.values() if r.get("chunk_h"))
    exp_h = meta.get("action_chunk")
    h_ok = len(hs) <= 1 and (exp_h is None or (hs and next(iter(hs)) == exp_h))
    print(f"  [{_verdict(h_ok, warn=not hs)}] H 分布={dict(hs)} (sidecar 期望={exp_h})")
    if not h_ok and hs:
        warns.append(f"chunk H 非恒定/不符 sidecar: {dict(hs)}")

    # ── 5: pose14 合理域 ──
    print(f"\n{CYAN}── 3. pose14 合理域 (base xyz / rpy / grip) ──{NC}")
    bad_xyz = bad_rpy = bad_grip = 0
    xyz_max = rpy_max = 0.0
    pchecked = 0
    for seq in cli:
        ca = _npz(cli_arr, seq)
        if ca is None or "pose14" not in ca:
            continue
        p = np.asarray(ca["pose14"])
        pchecked += 1
        xyzL, xyzR = p[:, 0:3], p[:, 7:10]
        rpyL, rpyR = p[:, 3:6], p[:, 10:13]
        gripLR = p[:, [6, 13]]
        xyz_max = max(xyz_max, float(np.abs(np.concatenate([xyzL, xyzR], 1)).max()))
        rpy_max = max(rpy_max, float(np.abs(np.concatenate([rpyL, rpyR], 1)).max()))
        bad_xyz += int(np.any(np.abs(np.concatenate([xyzL, xyzR], 1)) > args.pos_workspace))
        bad_rpy += int(np.any(np.abs(np.concatenate([rpyL, rpyR], 1)) > np.pi + 1e-3))
        # gripper 是命令值 (m): binarize 闭合值 ≈ -0.0055 是正常负数, 只有超物理包络才算坏
        bad_grip += int(np.any((gripLR < args.grip_lo) | (gripLR > args.grip_hi)))
    pose_ok = (bad_xyz == 0 and bad_rpy == 0 and bad_grip == 0)
    print(f"  [{_verdict(pose_ok, warn=not pchecked)}] chunks={pchecked}  "
          f"|xyz|max={xyz_max:.3f}m (>{args.pos_workspace} 帧={bad_xyz})  "
          f"|rpy|max={rpy_max:.3f}rad (>π 帧={bad_rpy})  "
          f"grip∉[{args.grip_lo},{args.grip_hi}] 帧={bad_grip}")
    if not pose_ok:
        warns.append(f"pose14 越界: xyz坏={bad_xyz} rpy坏={bad_rpy} grip坏={bad_grip}")

    # ── 6: 时序 ──
    print(f"\n{CYAN}── 4. 时序 (infer 频率 / 延迟) ──{NC}")
    cli_wall = sorted(r["t_wall"] for r in cli.values() if "t_wall" in r)
    if len(cli_wall) >= 2:
        dts = np.diff(cli_wall)
        hz = 1.0 / dts[dts > 1e-3]   # 忽略 <1ms 间隔 (避免抖动放大成天文 Hz)
        if hz.size:
            print(f"    infer 频率: p50={np.median(hz):.2f}Hz  min={hz.min():.2f}  max={hz.max():.2f}")
    _stat("client infer_ms (含 ws 往返)", [r.get("infer_ms") for r in cli.values() if r.get("infer_ms")], "ms")
    _stat("server infer_ms (纯模型)", [r.get("infer_ms") for r in srv.values() if r.get("infer_ms")], "ms")
    _stat("server total_ms", [r.get("total_ms") for r in srv.values() if r.get("total_ms")], "ms")
    ws_overhead = []
    for seq in common:
        c, s = cli[seq].get("infer_ms"), srv[seq].get("total_ms")
        if c and s:
            ws_overhead.append(c - s)
    _stat("ws+(de)序列化开销 = client.infer_ms - server.total_ms", ws_overhead, "ms")

    # ── 7: PosCmd 下发 ──
    print(f"\n{CYAN}── 5. PosCmd 下发 / jump-guard 丢帧 ──{NC}")
    published = [r for r in pos if not r.get("dropped")]
    dropped = [r for r in pos if r.get("dropped")]
    drop_rate = len(dropped) / max(1, len(pos))
    drop_ok = drop_rate < 0.02
    print(f"  [{_verdict(drop_ok, warn=True)}] 发出={len(published)}  丢帧(jump>8cm)={len(dropped)}  "
          f"丢帧率={drop_rate*100:.1f}%")
    pos_wall = sorted(r["t_wall"] for r in published if "t_wall" in r)
    if len(pos_wall) >= 2:
        d2 = np.diff(pos_wall)
        h2 = 1.0 / d2[d2 > 1e-3]
        if h2.size:
            print(f"    下发频率: p50={np.median(h2):.2f}Hz  max={h2.max():.2f}")
    if dropped:
        jm = [r.get("jump_mm") for r in dropped if r.get("jump_mm")]
        _stat("    丢帧 jump 幅度", jm, "mm", "{:.0f}")
    if not drop_ok and pos:
        warns.append(f"PosCmd jump-guard 丢帧率 {drop_rate*100:.1f}% 偏高 (目标输出跳变大)")

    # ── 8: proprio 来源 ──
    print(f"\n{CYAN}── 6. proprio 来源 (firmware 应恒 sensed) ──{NC}")
    psrc = Counter(r.get("proprio_source") for r in srv.values() if r.get("proprio_source"))
    print(f"    分布: {dict(psrc)}")
    if meta.get("ee_ctrl") == "firmware" and psrc.get("pred", 0) > 0:
        warns.append("firmware 模式出现 pred proprio (预期恒 sensed)")

    # ── 总结 ──
    print(f"\n{CYAN}══════════ 总结 ══════════{NC}")
    if not fails and not warns:
        print(f"{GREEN}✓ pipeline 全部核验通过{NC}")
    else:
        for x in fails:
            print(f"{RED}✗ FAIL: {x}{NC}")
        for x in warns:
            print(f"{YELLOW}! WARN: {x}{NC}")
    print(f"\n提示: 单帧详查  python {os.path.basename(__file__)} {d} --dump-seq <seq>")
    bag = os.path.join(d, "rosbag")
    if os.path.isdir(bag) and not os.path.isfile(os.path.join(bag, "metadata.yaml")):
        print(f"      {YELLOW}rosbag 缺 metadata.yaml (收尾未完成) → 先补: ros2 bag reindex {bag}{NC}")
    print(f"      rosbag    ros2 bag info {bag}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
