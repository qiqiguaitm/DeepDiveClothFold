"""Generate per-(model,camera) HTML reports + a master report.html.

- report_<model>_<cam>.html : episodes x horizons grid, each cell = merged side-by-side
  (GT|pred) video + mean metrics for that (ep,cam,horizon).
- report.html (master)       : model x horizon comparison tables (means over all units)
  + links to every per-(model,camera) report. Robust to partial data.
"""
import json, glob, os, html
from collections import defaultdict
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MERGED = "report_assets/merged"
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
HZS = ["1s", "3s", "7s"]
METRICS = [("psnr", "PSNR↑"), ("ssim", "SSIM↑"), ("temporal_absdiff_ratio", "Temporal→1"), ("lpips", "LPIPS↓")]
CSS = ("<style>body{font-family:system-ui,Arial;margin:24px;color:#161616}h1{font-size:22px}"
       "h2{margin-top:28px}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;"
       "padding:5px 10px;text-align:center;font-size:13px}th{background:#f3f4f6}.best{background:#dcfce7;font-weight:600}"
       "video{width:430px;border:1px solid #ddd;border-radius:4px}.cell{display:inline-block;margin:8px 16px 8px 0;vertical-align:top}"
       ".cap{font-size:12px;color:#444;margin-bottom:3px}a{color:#1d4ed8}</style>")

def load():
    rows = []
    for f in glob.glob(os.path.join(RES, "*.jsonl")):
        for l in open(f):
            l = l.strip()
            if l:
                try: rows.append(json.loads(l))
                except Exception: pass
    return rows

def mean(xs): return st.mean(xs) if xs else None
def fmt(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

def per_cam_report(rows, model, cam):
    sub = [r for r in rows if r["model"] == model and r["cam"] == cam]
    if not sub: return None
    eps = sorted({r["ep"] for r in sub})
    agg = defaultdict(lambda: defaultdict(list))  # (ep,hz) -> metric -> list
    for r in sub:
        for mk, _ in METRICS:
            if mk in r: agg[(r["ep"], r["horizon"])][mk].append(r[mk])
    p = [f"<html><head><meta charset='utf-8'><title>{model} / {cam}</title>{CSS}</head><body>",
         f"<h1>{html.escape(model)} — {cam}</h1>",
         "<p><a href='report.html'>← 总报告</a> ｜ 每格: GT(左) | 预测(右)，按锚点拼接的合并视频。</p>"]
    # metrics table
    p.append("<table><tr><th>episode \\ horizon</th>" + "".join(f"<th>{hz} (PSNR/SSIM/Temp)</th>" for hz in HZS) + "</tr>")
    for ep in eps:
        p.append(f"<tr><td>ep{ep}</td>")
        for hz in HZS:
            d = agg.get((ep, hz), {})
            cell = "/".join(fmt(mean(d.get(mk, []))) for mk in ("psnr", "ssim", "temporal_absdiff_ratio"))
            p.append(f"<td>{cell}</td>")
        p.append("</tr>")
    p.append("</table>")
    # videos grid
    for ep in eps:
        p.append(f"<h2>episode {ep}</h2>")
        for hz in HZS:
            mp4 = f"{MERGED}/{model}__{ep}__{cam}__{hz}__merged.mp4"
            d = agg.get((ep, hz), {})
            cap = f"{hz} | PSNR {fmt(mean(d.get('psnr',[])))} SSIM {fmt(mean(d.get('ssim',[])))} Temp {fmt(mean(d.get('temporal_absdiff_ratio',[])))}"
            if os.path.exists(os.path.join(HERE, mp4)):
                p.append(f"<div class='cell'><div class='cap'>{cap}</div><video controls muted loop src='{mp4}'></video></div>")
            else:
                p.append(f"<div class='cell'><div class='cap'>{cap} (video pending)</div></div>")
    p.append("</body></html>")
    out = os.path.join(HERE, f"report_{model}_{cam}.html")
    open(out, "w").write("".join(p))
    return out

def master(rows):
    models = sorted({r["model"] for r in rows})
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for mk, _ in METRICS:
            if mk in r: g[(r["model"], r["horizon"])][mk].append(r[mk])
    p = [f"<html><head><meta charset='utf-8'><title>Cosmos3 三模型 I2V 世界预测评测</title>{CSS}</head><body>",
         "<h1>Cosmos3 三模型 I2V 世界预测评测 — 叠衣 fold val (visrobot01_val)</h1>",
         f"<p>任务: Image→Video 世界预测 (teacher-forced 滑窗, 锚点喂 GT 帧). 模型: {', '.join(models)}. "
         f"3 episodes (102/126/168) × 3 cameras × horizon(1s/3s/7s). 指标在 256² / 24fps 对齐后计算. 记录数: {len(rows)}.</p>"]
    for mk, label in METRICS:
        cols = HZS
        p.append(f"<h2>{label}</h2><table><tr><th>model \\ horizon</th>" + "".join(f"<th>{h}</th>" for h in cols) + "</tr>")
        best = {}
        for h in cols:
            vals = [(m, mean(g[(m, h)].get(mk, []))) for m in models if mean(g[(m, h)].get(mk, [])) is not None]
            if vals: best[h] = (min if mk == "lpips" else max)(vals, key=lambda x: x[1])[0]
        for m in models:
            p.append(f"<tr><td style='text-align:left'>{html.escape(m)}</td>")
            for h in cols:
                v = mean(g[(m, h)].get(mk, [])); cls = " class='best'" if best.get(h) == m else ""
                p.append(f"<td{cls}>{fmt(v)}</td>")
            p.append("</tr>")
        p.append("</table>")
    p.append("<h2>分相机详细报告 (含合并视频)</h2><ul>")
    for m in models:
        for cam in CAMS:
            f = f"report_{m}_{cam}.html"
            if os.path.exists(os.path.join(HERE, f)):
                p.append(f"<li><a href='{f}'>{html.escape(m)} / {cam}</a></li>")
    p.append("</ul></body></html>")
    open(os.path.join(HERE, "report.html"), "w").write("".join(p))

def main():
    rows = load()
    models = sorted({r["model"] for r in rows})
    n = 0
    for m in models:
        for cam in CAMS:
            if per_cam_report(rows, m, cam): n += 1
    master(rows)
    print(f"generated {n} per-cam reports + master report.html ({len(rows)} records, models={models})")

if __name__ == "__main__":
    main()
