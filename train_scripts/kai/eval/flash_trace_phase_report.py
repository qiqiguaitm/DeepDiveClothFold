#!/usr/bin/env python3
"""R6 决策前置: 把 v2 (FLASH-on-eager) 真机 trace 按**相位**拆 accept/fallback/radius。

回答的核心问题 (决定要不要投 spec-on-v1 工程, 见 flash_impl_log.md §10/§11):
  FLASH 在**抓取段** (夹爪开合事件附近) 到底是「自信地投机」还是「自动退回全量」?
  - 若抓取段 fallback% ≫ smooth 段 → FLASH 已在精度关键事件处**自动退回全量**, 那么 spec-on-v1
    的 **prefill-skip 也不会在抓取段生效** (因为退回全量会刷新 KV) → 视觉陈旧风险**低**, 可投。
  - 若抓取段 accept 仍高 (≈ smooth) → draft 在抓取段也自信 → 投机 (含 prefill-skip 复用旧 KV)
    **会在抓取段生效** → 视觉陈旧会正中开环病 → spec-on-v1 必须按相位门控 full 刷新 (N=1 at grasp)。

输入 = `serve_policy_flash.py --trace-out` 或 `start_autonomy_from_ckpt_v2.sh --trace` 产的 JSONL
(每行一帧: accept/rad_mean/rad_max/fb/gl_rng/gr_rng/gl_net/gr_net/gl_obs/gr_obs/...)。

相位判定: grasp = 该帧输出 chunk 内任一臂夹爪净位移 |net| > --grasp-thr (归一化动作空间)。
(不依赖 FLASH 内部 gripper-verify — 部署 shim 未串 last_gripper, g_stop/g_cut 多为 0。)

Run:
  python train_scripts/kai/eval/flash_trace_phase_report.py /tmp/flash_trace_8001_*.jsonl
  python train_scripts/kai/eval/flash_trace_phase_report.py <trace.jsonl> --grasp-thr 0.25
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np


def _load(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for pat in paths:
        for fp in sorted(glob.glob(pat)):
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                            r.setdefault("_file", fp)
                            rows.append(r)
                        except json.JSONDecodeError:
                            pass
    return rows


def _ab_compare(rows: list[dict]) -> None:
    """When >1 trace file is given (e.g. spec vs --no-spec), print per-file arm-motion +
    accept summary so the real-machine under-actuation A/B is read off directly."""
    import os

    files = sorted({r.get("_file", "?") for r in rows})
    if len(files) < 2:
        return
    print("\n  -- 按文件 A/B (arm_motion = 归一化动作空间逐维 max-min 之和, 越大动得越多) --")
    base = None
    for fp in files:
        fr = [r for r in rows if r.get("_file") == fp]
        am = np.array([r.get("arm_motion", np.nan) for r in fr], dtype=np.float64)
        acc = np.array([r.get("accept", np.nan) for r in fr], dtype=np.float64)
        mode = fr[0].get("mode", "?") if fr else "?"
        am_med = float(np.nanmedian(am))
        if base is None:
            base = am_med
        rel = f" ({am_med / base:.2f}x vs first)" if base and base > 1e-9 else ""
        print(f"     {os.path.basename(fp):42s} mode={mode:8s} n={len(fr):4d} | "
              f"arm_motion med={am_med:.3f} mean={np.nanmean(am):.3f}{rel} | accept med={np.nanmedian(acc):.0f}")
    print("     ⇒ 若 spec 的 arm_motion 明显 < no_spec → 真机 OOD 帧上 draft 欠驱动 (offline in-dist probe 测不出)。")


def _col(rows, key, default=np.nan):
    return np.array([r.get(key, default) for r in rows], dtype=np.float64)


def _bucket_stats(name, mask, accept, fb, rad_mean, rad_max, arm_motion, ah):
    n = int(mask.sum())
    if n == 0:
        print(f"  {name:7s}: (无帧)")
        return None
    a = accept[mask]
    f = fb[mask]
    rm = rad_mean[mask]
    rx = rad_max[mask]
    am = arm_motion[mask]
    accept_frac = float(np.nanmean(a)) / ah
    fb_pct = 100.0 * float(np.nanmean(f))
    print(f"  {name:7s}: n={n:5d} | accept={np.nanmean(a):5.1f}/{ah} ({accept_frac:4.0%}) | "
          f"fallback={fb_pct:5.1f}% | rad_mean={np.nanmean(rm):.4f} | rad_max(p90)={np.nanpercentile(rx,90):.4f} | "
          f"arm_motion={np.nanmedian(am):.3f}")
    return {"n": n, "accept_frac": accept_frac, "fb_pct": fb_pct}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="+", help="JSONL trace 路径 (支持 glob)")
    ap.add_argument("--grasp-thr", type=float, default=0.20,
                    help="grasp 判定: 任一臂夹爪 chunk 内净位移 |net| 超过此值即算抓取段 (归一化空间)")
    args = ap.parse_args()

    rows = _load(args.trace)
    if not rows:
        print("no trace rows found")
        return 1
    n = len(rows)
    ah = int(rows[0].get("H", 50))

    accept = _col(rows, "accept")
    fb = _col(rows, "fb", 0.0)
    rad_mean = _col(rows, "rad_mean")
    rad_max = _col(rows, "rad_max")
    arm_motion = _col(rows, "arm_motion")
    t = _col(rows, "t", 0.0)
    gl_net = np.abs(_col(rows, "gl_net", 0.0))
    gr_net = np.abs(_col(rows, "gr_net", 0.0))
    gl_rng = np.abs(_col(rows, "gl_rng", 0.0))
    gr_rng = np.abs(_col(rows, "gr_rng", 0.0))
    grip_travel = np.maximum(np.nanmax(np.vstack([gl_net, gr_net]), axis=0),
                             0.0)  # 主判据: 净开合
    grip_rng = np.nanmax(np.vstack([gl_rng, gr_rng]), axis=0)

    has_grip = np.isfinite(gl_net).any() or np.isfinite(gr_net).any()
    grasp = grip_travel > args.grasp_thr if has_grip else np.zeros(n, dtype=bool)
    smooth = ~grasp

    span = float(np.nanmax(t) - np.nanmin(t)) if np.isfinite(t).any() else float("nan")
    print("\n========== FLASH acceptxphase report ==========")
    print(f"  trace rows = {n}  | eval_h = {ah} | span = {span:.1f}s "
          f"(~{n/max(span,1e-9):.1f} infer/s)")
    print(f"  overall: accept={np.nanmean(accept):.1f}/{ah} ({np.nanmean(accept)/ah:.0%}) | "
          f"fallback={100*np.nanmean(fb):.1f}% | rad_mean={np.nanmean(rad_mean):.4f} | "
          f"arm_motion med={np.nanmedian(arm_motion):.3f}")
    _ab_compare(rows)
    if has_grip:
        print(f"  夹爪净位移 |net| (用于相位判定): p50={np.nanpercentile(grip_travel,50):.3f} "
              f"p90={np.nanpercentile(grip_travel,90):.3f} max={np.nanmax(grip_travel):.3f}  "
              f"(chunk-内 range p90={np.nanpercentile(grip_rng,90):.3f})")
    else:
        print("  ⚠️ trace 无夹爪字段 (gl_net/gr_net) → 无法做相位拆分; 升级 serve_policy_flash.py 后重录。")

    print(f"\n  -- 相位拆分 (grasp = max臂|net| > {args.grasp_thr}) --")
    s_sm = _bucket_stats("smooth", smooth, accept, fb, rad_mean, rad_max, arm_motion, ah)
    s_gr = _bucket_stats("grasp", grasp, accept, fb, rad_mean, rad_max, arm_motion, ah)

    print("\n  -- verdict --")
    if not has_grip or s_gr is None:
        print("     无抓取段帧 (或无夹爪字段) → 此 trace 未覆盖抓取; 跑一段含抓/放的真机 rollout 再判。")
        return 0
    d_fb = s_gr["fb_pct"] - s_sm["fb_pct"] if s_sm else s_gr["fb_pct"]
    if d_fb > 20.0:
        print(f"     抓取段 fallback ({s_gr['fb_pct']:.0f}%) ≫ smooth ({s_sm['fb_pct']:.0f}%) → FLASH 已在抓取段"
              "**自动退回全量** (radius/夹爪门生效)。")
        print("     ⇒ spec-on-v1 的 prefill-skip 不会在抓取段生效 (退回全量会刷新 KV) → **视觉陈旧风险低, 可投**。")
    elif s_gr["accept_frac"] > 0.7 and abs(d_fb) <= 20.0:
        print(f"     抓取段 accept 仍高 ({s_gr['accept_frac']:.0%}, fallback {s_gr['fb_pct']:.0f}% ≈ smooth) → "
              "draft 在抓取段也自信。")
        print("     ⚠️ ⇒ 投机 (含 prefill-skip 复用旧 KV) **会在抓取段生效** → 视觉陈旧正中开环病。")
        print("     spec-on-v1 必须**按相位门控**: 抓取段强制 N=1 (每帧 full 刷新 KV), 仅 smooth 段放大 N。")
    else:
        print(f"     抓取段 accept={s_gr['accept_frac']:.0%} fallback={s_gr['fb_pct']:.0f}%, smooth "
              f"accept={s_sm['accept_frac']:.0%} fallback={s_sm['fb_pct']:.0f}% — 居中。")
        print("     建议: 调 --grasp-thr 看稳定性; 并在真机标注实际抓取成败与 trace 对齐再定 N 策略。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
