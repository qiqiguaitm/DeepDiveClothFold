#!/usr/bin/env python3
"""真机在线对比采集器:gwp_ans vs kai0(A_smooth800)叠衣服任务。

操作者驱动:每个 trial 回车开始计时、任务结束回车停止,录入成功/接管次数/备注;
追加到 jsonl,并实时打印每个模型的 成功率 / 平均完成时间 / 接管率。在线闭环 SR 是
离线 MAE 无法替代的真指标(且免疫任何 train/val 泄漏)。

延迟可选:若 policy 节点开了 KAI0_LATENCY_PROFILE=1,用 --latency_csv 指向其 csv,
脚本会在每个 trial 结束时记下该文件当前的 p50/p95 推理延迟。

用法:
  # 终端A: 起某个模型的栈 (见下方两条命令), 用 /policy/execute 控制执行
  # 终端B:
  python train_scripts/kai/eval/eval_online_compare.py --model gwp_ans --n 20 \
      --out /data2/gwp_eval/out/online_compare.jsonl
  python train_scripts/kai/eval/eval_online_compare.py --model kai0_A_smooth800 --n 20 \
      --out /data2/gwp_eval/out/online_compare.jsonl
  # 汇总:
  python train_scripts/kai/eval/eval_online_compare.py --summary \
      --out /data2/gwp_eval/out/online_compare.jsonl
"""
import argparse, json, os, time, glob
from collections import defaultdict


def _latency_snapshot(csv_path):
    """读 KAI0_LATENCY_PROFILE csv 的最后一列(总推理 ms),返回 p50/p95。容错。"""
    try:
        import numpy as np
        # 取最新的一个 latency csv(若给的是目录或 glob)
        files = sorted(glob.glob(csv_path)) if any(c in csv_path for c in "*?[") else [csv_path]
        path = files[-1] if files else csv_path
        rows = [l.strip().split(",") for l in open(path) if l.strip()]
        if len(rows) < 2:
            return None
        hdr = rows[0]
        # 找“总”列:优先含 total/infer/e2e 的列,否则用最后一列
        ci = next((i for i, h in enumerate(hdr) if any(k in h.lower() for k in ("total", "e2e", "infer_ms"))), len(hdr) - 1)
        vals = []
        for r in rows[1:]:
            try: vals.append(float(r[ci]))
            except Exception: pass
        if not vals:
            return None
        a = np.asarray(vals)
        return {"p50_ms": round(float(np.percentile(a, 50)), 1),
                "p95_ms": round(float(np.percentile(a, 95)), 1), "n": len(a)}
    except Exception:
        return None


def summarize(out_path):
    if not os.path.exists(out_path):
        print("(no data yet)"); return
    recs = [json.loads(l) for l in open(out_path) if l.strip()]
    by = defaultdict(list)
    for r in recs:
        by[r["model"]].append(r)
    print(f"\n=== online comparison ({len(recs)} trials) ===")
    print(f"{'model':>20} | {'N':>3} | {'success%':>8} | {'avg_time_s':>10} | {'interv/trial':>12} | {'lat p95':>8}")
    for m, rs in sorted(by.items()):
        n = len(rs); succ = sum(1 for r in rs if r["success"])
        succ_ok = [r["duration_s"] for r in rs if r["success"] and r.get("duration_s")]
        avg_t = sum(succ_ok) / len(succ_ok) if succ_ok else float("nan")
        interv = sum(r.get("interventions", 0) for r in rs) / n
        p95 = next((r["latency"]["p95_ms"] for r in reversed(rs) if r.get("latency")), None)
        print(f"{m:>20} | {n:>3} | {100*succ/n:>7.1f}% | {avg_t:>10.1f} | {interv:>12.2f} | {str(p95):>8}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="gwp_ans | kai0_A_smooth800 | <任意标签>")
    ap.add_argument("--n", type=int, default=20, help="trial 数")
    ap.add_argument("--out", default="/data2/gwp_eval/out/online_compare.jsonl")
    ap.add_argument("--latency_csv", default="", help="KAI0_LATENCY_PROFILE csv 路径或 glob(可选)")
    ap.add_argument("--summary", action="store_true", help="只打印汇总,不采集")
    args = ap.parse_args()

    if args.summary:
        summarize(args.out); return
    if not args.model:
        ap.error("--model required unless --summary")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"=== 采集 model={args.model}, 计划 {args.n} trials ===")
    print("每个 trial: [Enter]开始计时 → 执行任务 → [Enter]结束 → 录入 成功/接管/备注。Ctrl-C 提前结束。")
    done = sum(1 for l in open(args.out) if l.strip() and json.loads(l).get("model") == args.model) if os.path.exists(args.out) else 0
    try:
        for i in range(done, args.n):
            input(f"\n[trial {i+1}/{args.n}] 摆好布、确认 /policy/execute=true,按 [Enter] 开始...")
            t0 = time.time()
            input("  ...执行中,任务结束/失败后按 [Enter] 停止计时...")
            dur = round(time.time() - t0, 1)
            s = input(f"  成功? [y/N] (用时 {dur}s): ").strip().lower()
            success = s in ("y", "yes", "1")
            try:
                interv = int(input("  接管/纠正次数 [0]: ").strip() or "0")
            except ValueError:
                interv = 0
            note = input("  备注(可空): ").strip()
            rec = {"model": args.model, "trial": i + 1, "success": success,
                   "duration_s": dur, "interventions": interv, "note": note,
                   "ts": int(t0)}
            if args.latency_csv:
                lat = _latency_snapshot(args.latency_csv)
                if lat: rec["latency"] = lat
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  recorded: success={success} dur={dur}s interv={interv}")
    except KeyboardInterrupt:
        print("\n[中断] 已保存已完成的 trials。")
    summarize(args.out)


if __name__ == "__main__":
    main()
