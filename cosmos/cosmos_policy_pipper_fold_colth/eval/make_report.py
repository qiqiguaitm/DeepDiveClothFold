#!/usr/bin/env python3
"""Generate the cosmos_policy_pipper_fold_colth evaluation report (Markdown + PNG plots).

Reads:
  - reports/mae_curve.jsonl   (per-checkpoint val action-MAE, written by auto_eval_loop.sh)
  - logs/full_train.log       (training iter/loss from iter_speed callback)
  - reports/cluster_status.txt (optional, latest AIHC job status)
Writes:
  - reports/REPORT.md
  - reports/curves.png        (train loss + val MAE vs iteration)
Selects best checkpoint by lowest mae_overall.
"""
import glob
import json
import os
import re

R = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/cosmos_policy_pipper_fold_colth_runs"
REPORTS = f"{R}/reports"
CURVE = f"{REPORTS}/mae_curve.jsonl"
CURVE_CLUSTER = f"{REPORTS}/mae_curve_cluster.jsonl"
TRAINLOG = f"{R}/logs/full_train.log"
OUT_MD = f"{REPORTS}/REPORT.md"
OUT_PNG = f"{REPORTS}/curves.png"


def load_mae(path=CURVE):
    rows = []
    if os.path.exists(path):
        for ln in open(path):
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
    # dedup by iter keep last
    by = {}
    for r in rows:
        by[r["iter"]] = r
    return [by[k] for k in sorted(by)]


def load_loss():
    pts = []
    if os.path.exists(TRAINLOG):
        for m in re.finditer(r"(\d+) : iter_speed [\d.]+ seconds per iteration \| Loss: ([\d.]+)", open(TRAINLOG, errors="ignore").read()):
            pts.append((int(m.group(1)), float(m.group(2))))
    return pts


def plot(mae, loss):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        if loss:
            xs, ys = zip(*loss)
            ax[0].plot(xs, ys, lw=0.8)
            ax[0].set_title("train flow-matching loss"); ax[0].set_xlabel("iter"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=.3)
        if mae:
            xs = [m["iter"] for m in mae]
            for k in ["mae_overall", "mae@1", "mae@10"]:
                ax[1].plot(xs, [m[k] for m in mae], marker="o", label=k)
            ax[1].set_title("val action-MAE (rad)"); ax[1].set_xlabel("iter"); ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT_PNG, dpi=90); plt.close(fig)
        return True
    except Exception as e:
        print("plot skipped:", e)
        return False


def main():
    os.makedirs(REPORTS, exist_ok=True)
    mae = load_mae()
    loss = load_loss()
    cluster = ""
    if os.path.exists(f"{REPORTS}/cluster_status.txt"):
        cluster = open(f"{REPORTS}/cluster_status.txt").read().strip()
    best = min(mae, key=lambda r: r["mae_overall"]) if mae else None
    has_png = plot(mae, loss)

    cur_iter = loss[-1][0] if loss else 0
    cur_loss = loss[-1][1] if loss else None
    train_alive = os.system("pgrep -f cosmos_policy.scripts.train >/dev/null 2>&1") == 0

    L = []
    L.append("# cosmos_policy_pipper_fold_colth — Evaluation Report\n")
    L.append("Warm-started `nvidia/Cosmos-Policy-ALOHA-Predict2-2B` → Agilex Piper dual-arm cloth-fold "
             "(wam_fold_v3), via NVlabs/cosmos-policy. 14-D absolute joint actions, 50-step chunk, "
             "3 cameras @224, batch 16, lr 1e-4, warm-start.\n")
    L.append(f"**Training:** {'RUNNING' if train_alive else 'stopped'} at iter ~{cur_iter}"
             + (f", loss {cur_loss:.3f}" if cur_loss is not None else "") + " (max_iter 6000, ckpt/500).")
    if cluster:
        L.append(f"\n**5-node cluster job:** {cluster}")
    L.append("\n## Val action-MAE (lower=better, raw joint radians; 100 val episodes, stride 50, 10 denoise steps)\n")
    if mae:
        L.append("| iter | MAE | mae@1 | mae@10 | mae@25 | mae@50 |")
        L.append("|---:|---:|---:|---:|---:|---:|")
        for m in mae:
            mark = " ⬅ best" if best and m["iter"] == best["iter"] else ""
            L.append(f"| {m['iter']} | **{m['mae_overall']:.4f}**{mark} | {m['mae@1']:.4f} | {m['mae@10']:.4f} | {m['mae@25']:.4f} | {m['mae@50']:.4f} |")
        L.append("")
        if best:
            L.append(f"**Best checkpoint:** `iter_{best['iter']:06d}` — MAE **{best['mae_overall']:.4f}** rad "
                     f"(mae@1 {best['mae@1']:.4f}).\n")
            L.append(f"  path: `train_out/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_pipper_fold_colth/"
                     f"checkpoints/iter_{best['iter']:06d}/model.pt`\n")
    else:
        L.append("_(no eval points yet)_\n")
    if loss:
        ys = [y for _, y in loss]
        L.append(f"## Training loss\nstart {ys[0]:.3f} → latest {ys[-1]:.3f} (min {min(ys):.3f}), {len(loss)} logged points.\n")
    if has_png:
        L.append("![curves](curves.png)\n")
    # ---- cluster (5-node) run section ----
    mae_c = load_mae(CURVE_CLUSTER)
    if mae_c:
        best_c = min(mae_c, key=lambda r: r["mae_overall"])
        L.append("## 5-node cluster run (train_out_aihc, global batch 640) — val action-MAE\n")
        L.append("| iter | MAE | mae@1 | mae@10 |")
        L.append("|---:|---:|---:|---:|")
        for m in mae_c:
            mark = " ⬅ best" if m["iter"] == best_c["iter"] else ""
            L.append(f"| {m['iter']} | **{m['mae_overall']:.4f}**{mark} | {m['mae@1']:.4f} | {m['mae@10']:.4f} |")
        L.append(f"\n**Cluster best:** `iter_{best_c['iter']:06d}` MAE **{best_c['mae_overall']:.4f}** rad.\n")
    L.append("## Artifacts\n"
             "- checkpoints: `runs/train_out/.../checkpoints/iter_*` (DCP; `model.pt` = consolidated)\n"
             "- cluster run: `runs/train_out_aihc/...`\n"
             "- per-iter MAE json: `runs/reports/mae_iter_*.json`; curve: `runs/reports/mae_curve.jsonl`\n"
             "- eval: `eval/offline_eval.py` (+ `run_eval.sh`); DCP→pt: `eval/dcp_to_pt.py`\n")
    open(OUT_MD, "w").write("\n".join(L))
    print(f"wrote {OUT_MD}" + (f" + {OUT_PNG}" if has_png else ""))
    if best:
        print(f"best: iter_{best['iter']} MAE {best['mae_overall']:.4f}")


if __name__ == "__main__":
    main()
