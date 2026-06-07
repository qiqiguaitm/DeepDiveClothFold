"""Aggregate eval jsonl -> comparison tables + report.html (with side-by-side videos).

Reads all *.jsonl in results/, groups by (model,horizon), reports mean PSNR/SSIM/LPIPS/
temporal + count, builds metric-vs-horizon, and embeds the saved pred/gt mp4 pairs.
Robust to partial runs (reports whatever is present).
"""
import json, glob, os, html
from collections import defaultdict
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ASSETS = "report_assets"
HORIZONS = ["1s", "3s", "7s"]
METRICS = [("psnr", "PSNR↑"), ("ssim", "SSIM↑"), ("lpips", "LPIPS↓"), ("temporal_absdiff_ratio", "Temporal→1")]

def load():
    rows = []
    for f in glob.glob(os.path.join(RES, "*.jsonl")):
        for l in open(f):
            l = l.strip()
            if l:
                try: rows.append(json.loads(l))
                except Exception: pass
    return rows

def agg(rows):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for mk, _ in METRICS:
            if mk in r: g[(r["model"], r["horizon"])][mk].append(r[mk])
        g[(r["model"], r["horizon"])]["gen_s"].append(r.get("gen_s", 0))
        g[(r["model"], r["horizon"])]["_n"].append(1)
    return g

def mean(xs): return st.mean(xs) if xs else float("nan")

def main():
    rows = load()
    models = sorted({r["model"] for r in rows})
    g = agg(rows)
    print(f"loaded {len(rows)} records; models={models}")

    # ----- console summary -----
    for m in models:
        for hz in HORIZONS:
            d = g.get((m, hz))
            if not d: continue
            print(f"{m:28s} {hz}: n={len(d['_n']):3d} "
                  + " ".join(f"{mk}={mean(d[mk]):.3f}" for mk, _ in METRICS if d.get(mk))
                  + f" gen_s={mean(d['gen_s']):.0f}")

    # ----- HTML -----
    def cell(m, hz, mk):
        d = g.get((m, hz))
        return f"{mean(d[mk]):.3f}" if d and d.get(mk) else "—"
    parts = ["<html><head><meta charset='utf-8'><title>Cosmos3 三模型 I2V 世界预测评测</title>",
             "<style>body{font-family:system-ui,Arial;margin:24px;color:#1a1a1a}"
             "h1{font-size:22px}h2{margin-top:32px}table{border-collapse:collapse;margin:8px 0}"
             "td,th{border:1px solid #ccc;padding:6px 12px;text-align:center;font-size:14px}"
             "th{background:#f3f4f6}.best{background:#dcfce7;font-weight:600}"
             "video{width:300px;border:1px solid #ddd;border-radius:4px}.pair{display:inline-block;margin:6px 14px 6px 0;vertical-align:top}"
             ".cap{font-size:12px;color:#555}</style></head><body>"]
    parts.append("<h1>Cosmos3 三模型 I2V 世界预测评测 — 叠衣 fold val (visrobot01_val)</h1>")
    parts.append(f"<p>任务: Image→Video 世界预测 (teacher-forced 滑窗). 模型: {', '.join(models)}. "
                 f"指标在公共 256² 网格、24fps 对齐后计算. 记录数: {len(rows)}.</p>")

    # comparison table per metric
    for mk, label in METRICS:
        parts.append(f"<h2>{label}</h2><table><tr><th>model \\ horizon</th>"
                     + "".join(f"<th>{hz}</th>" for hz in HORIZONS) + "</tr>")
        # best per column
        best = {}
        for hz in HORIZONS:
            vals = [(m, mean(g[(m, hz)][mk])) for m in models if g.get((m, hz)) and g[(m, hz)].get(mk)]
            if vals:
                best[hz] = (min if mk == "lpips" else max)(vals, key=lambda x: x[1])[0]
        for m in models:
            parts.append(f"<tr><td style='text-align:left'>{html.escape(m)}</td>")
            for hz in HORIZONS:
                cls = " class='best'" if best.get(hz) == m else ""
                parts.append(f"<td{cls}>{cell(m, hz, mk)}</td>")
            parts.append("</tr>")
        parts.append("</table>")

    # gallery
    parts.append("<h2>并排画廊 (GT | 预测, cam_high)</h2>")
    preds = sorted(glob.glob(os.path.join(HERE, ASSETS, "*_pred.mp4")))
    for p in preds:
        stem = os.path.basename(p)[:-9]
        gt = os.path.join(HERE, ASSETS, stem + "_gt.mp4")
        if not os.path.exists(gt): continue
        parts.append(f"<div class='pair'><div class='cap'>{html.escape(stem)}</div>"
                     f"<video controls muted loop src='{ASSETS}/{stem}_gt.mp4'></video> "
                     f"<video controls muted loop src='{ASSETS}/{stem}_pred.mp4'></video></div>")
    parts.append("</body></html>")
    out = os.path.join(HERE, "report.html")
    open(out, "w").write("".join(parts))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
