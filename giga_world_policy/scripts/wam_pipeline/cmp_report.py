"""两个 run 的 MAE 趋势对比报告 report_cmp.html(自包含,内联 SVG,无外网依赖)。

读两个 run 各自的 report_step<N>/summary.json(由 episode_report/watcher 产出,含 raw_mae{1,10,24,48}
+ pi05 基线),叠加 mae@h-vs-step 曲线 + 终点/最优对标表。点稀疏也能画。

两个系列槽位是【中性】的:baseline(绿)与 target(红)—— 与模型用 delta 还是 abs 表示无关,
谁是基线谁是被测由调用方决定,标签用 --label_baseline / --label_target 自定义。
(旧参数名 --delta_run/--abs_run/--label_delta/--label_abs 仍作为别名保留,向后兼容。)

用法:
  python -m scripts.wam_pipeline.cmp_report \
    --baseline_run runs/gwp_abs_v4 --target_run <fastwam-v5 run> \
    --label_baseline "gwp_abs_v4 (共享transformer@v3)" --label_target "fastwam-v5 (独立ActionDiT@v3)" \
    --title "fastwam-v5 vs gwp_abs_v4 @ v3" --out runs/report_cmp.html
"""
import argparse
import glob
import json
import os

HORIZONS = ["1", "10", "24", "48"]
COL = {"baseline": "#2a7", "target": "#e35", "pi05": "#88a"}


def collect(run):
    pts = {}
    for f in glob.glob(os.path.join(run, "report_step*", "summary.json")):
        try:
            n = int(os.path.basename(os.path.dirname(f)).replace("report_step", ""))
            d = json.load(open(f))
            rm = d.get("raw_mae", {})
            if rm:
                pts[n] = {h: float(rm[h]) for h in HORIZONS if h in rm}
                pts[n]["_pi05"] = d.get("pi05", {})
        except Exception:
            pass
    return dict(sorted(pts.items()))


def svg_chart(h, baseline, target, pi05, w=520, ht=260, pad=44):
    """一个 horizon 的折线图:x=step, y=mae@h。baseline/target 两条线 + pi05 水平基线。"""
    xs = sorted(set(baseline) | set(target))
    if not xs:
        return f"<svg width={w} height={ht}></svg>"
    allv = [baseline[s][h] for s in baseline if h in baseline[s]] + [target[s][h] for s in target if h in target[s]]
    if pi05:
        allv.append(pi05)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0.0, max(allv) * 1.1 + 1e-6
    def X(s): return pad + (s - xmin) / max(1, xmax - xmin) * (w - 2 * pad)
    def Y(v): return ht - pad - (v - ymin) / (ymax - ymin) * (ht - 2 * pad)
    def line(pts, color, dash=""):
        if not pts:
            return ""
        d = "M" + " L".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in pts)
        dots = "".join(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="2.5" fill="{color}"/>' for s, v in pts)
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" {dash}/>{dots}'
    bl = [(s, baseline[s][h]) for s in sorted(baseline) if h in baseline[s]]
    tl = [(s, target[s][h]) for s in sorted(target) if h in target[s]]
    grid = "".join(f'<line x1="{pad}" y1="{Y(ymax*i/4):.0f}" x2="{w-pad}" y2="{Y(ymax*i/4):.0f}" stroke="#eee"/>'
                   f'<text x="6" y="{Y(ymax*i/4)+4:.0f}" font-size="10" fill="#999">{ymax*i/4:.3f}</text>' for i in range(5))
    pi = (f'<line x1="{pad}" y1="{Y(pi05):.1f}" x2="{w-pad}" y2="{Y(pi05):.1f}" stroke="{COL["pi05"]}" '
          f'stroke-width="1.5" stroke-dasharray="5 4"/><text x="{w-pad-90}" y="{Y(pi05)-4:.0f}" font-size="10" '
          f'fill="{COL["pi05"]}">pi0.5={pi05:.4f}</text>') if pi05 else ""
    xlab = "".join(f'<text x="{X(s):.0f}" y="{ht-pad+16:.0f}" font-size="9" fill="#999" text-anchor="middle">{s//1000}k</text>'
                   for s in xs[::max(1, len(xs)//8)])
    return (f'<svg width="{w}" height="{ht}" style="background:#fff;border:1px solid #eee">'
            f'<text x="{w/2:.0f}" y="16" font-size="13" font-weight="600" text-anchor="middle">mae@{h} vs step</text>'
            f'{grid}{pi}{line(bl, COL["baseline"])}{line(tl, COL["target"])}{xlab}</svg>')


def main():
    ap = argparse.ArgumentParser()
    # 中性参数名 baseline/target;保留旧名 delta/abs 作别名(向后兼容)
    ap.add_argument("--baseline_run", "--delta_run", dest="baseline_run", default="runs/visrobot01_fold_aihc_latent_5x")
    ap.add_argument("--target_run", "--abs_run", dest="target_run", default="runs/visrobot01_fold_abs_50k")
    ap.add_argument("--out", default="runs/report_cmp.html")
    ap.add_argument("--label_baseline", "--label_delta", dest="label_baseline", default="baseline")
    ap.add_argument("--label_target", "--label_abs", dest="label_target", default="target")
    ap.add_argument("--title", default="WAM 叠衣服:MAE 趋势对比(visrobot01_val)")
    ap.add_argument("--desc", default="↓ 越小越好。")
    args = ap.parse_args()
    label_b, label_t = args.label_baseline, args.label_target
    baseline, target = collect(args.baseline_run), collect(args.target_run)
    pi05 = {}
    for src in (target, baseline):
        for s in src.values():
            if s.get("_pi05"):
                pi05 = s["_pi05"]; break
        if pi05:
            break

    charts = "".join(svg_chart(h, baseline, target, float(pi05.get(h)) if pi05.get(h) else None) for h in HORIZONS)

    def last(src):
        return src[max(src)] if src else None
    bl, tl = last(baseline), last(target)
    def best(src, h):
        vs = [src[s][h] for s in src if h in src[s]]
        return min(vs) if vs else None
    rows = ""
    for h in HORIZONS:
        b = bl.get(h) if bl else None; t = tl.get(h) if tl else None
        tb = best(target, h); bb = best(baseline, h); p = pi05.get(h)
        ratio = f"{t/b:.2f}×" if (t and b) else "—"
        rows += (f"<tr><td>mae@{h}</td><td>{b:.4f}</td><td>{t:.4f}</td><td>{ratio}</td>"
                 f"<td>{bb:.4f}</td><td>{tb:.4f}</td><td style='color:#88a'>{p}</td></tr>"
                 if (b and t) else
                 f"<tr><td>mae@{h}</td><td>{b if b else '—'}</td><td>{t if t else '—'}</td><td>—</td>"
                 f"<td>{bb if bb else '—'}</td><td>{tb if tb else '—'}</td><td style='color:#88a'>{p}</td></tr>")
    bmax = max(baseline) if baseline else 0; tmax = max(target) if target else 0
    html = f"""<!doctype html><meta charset=utf-8><title>{args.title}</title>
<style>body{{font-family:system-ui,Arial;margin:24px;color:#222}}
h1{{font-size:20px}} .leg span{{display:inline-block;margin-right:18px;font-size:13px}}
table{{border-collapse:collapse;margin:14px 0}} td,th{{border:1px solid #ddd;padding:5px 12px;font-size:13px;text-align:right}}
th{{background:#f6f6f6}} td:first-child{{text-align:left;font-weight:600}} .charts svg{{margin:6px}}</style>
<h1>{args.title}</h1>
<p class=leg><span style="color:{COL['baseline']}">● {label_b}</span>
<span style="color:{COL['target']}">● {label_t}</span>
<span style="color:{COL['pi05']}">--- pi0.5 基线</span></p>
<p style="font-size:13px;color:#555">{label_b} 点 {len(baseline)} 个(最新 {bmax//1000}k),{label_t} 点 {len(target)} 个(最新 {tmax//1000}k)。
{args.desc}</p>
<div class=charts>{charts}</div>
<h3>对标表(最新 step 终值 + 全程最优 best)</h3>
<table><tr><th>指标</th><th>{label_b} 终值</th><th>{label_t} 终值</th><th>{label_t}/{label_b}</th>
<th>{label_b} best</th><th>{label_t} best</th><th>pi0.5</th></tr>{rows}</table>
<p style="font-size:12px;color:#888">注:ratio=target/baseline;mae@1 含锚定红利,长 horizon(mae@48)更能反映真实策略质量。</p>"""
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(html)
    print(f"[cmp] baseline {len(baseline)} pts, target {len(target)} pts -> {args.out}")
    if bl and tl:
        print("[cmp] latest: " + " ".join(f"@{h} {label_b}={bl.get(h)} {label_t}={tl.get(h)}" for h in HORIZONS))


if __name__ == "__main__":
    main()
