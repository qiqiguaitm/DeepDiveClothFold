"""delta vs abs 对比报告 report_cmp.html(自包含,内联 SVG,无外网依赖)。

读两个 run 各自的 report_step<N>/summary.json(由 episode_report/watcher 产出,含 raw_mae{1,10,24,48}
+ pi05 基线),叠加 mae@h-vs-step 曲线 + 终点/最优对标表。delta 点稀疏也能画。

用法:
  python -m scripts.wam_pipeline.cmp_report \
    --delta_run runs/visrobot01_fold_aihc_latent_5x --abs_run runs/visrobot01_fold_abs_50k \
    --out runs/report_cmp.html
"""
import argparse
import glob
import json
import os

HORIZONS = ["1", "10", "24", "48"]
COL = {"delta": "#2a7", "abs": "#e35", "pi05": "#88a"}


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


def svg_chart(h, delta, abs_, pi05, w=520, ht=260, pad=44):
    """一个 horizon 的折线图:x=step, y=mae@h。delta/abs 两条线 + pi05 水平基线。"""
    xs = sorted(set(delta) | set(abs_))
    if not xs:
        return f"<svg width={w} height={ht}></svg>"
    allv = [delta[s][h] for s in delta if h in delta[s]] + [abs_[s][h] for s in abs_ if h in abs_[s]]
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
    dl = [(s, delta[s][h]) for s in sorted(delta) if h in delta[s]]
    al = [(s, abs_[s][h]) for s in sorted(abs_) if h in abs_[s]]
    grid = "".join(f'<line x1="{pad}" y1="{Y(ymax*i/4):.0f}" x2="{w-pad}" y2="{Y(ymax*i/4):.0f}" stroke="#eee"/>'
                   f'<text x="6" y="{Y(ymax*i/4)+4:.0f}" font-size="10" fill="#999">{ymax*i/4:.3f}</text>' for i in range(5))
    pi = (f'<line x1="{pad}" y1="{Y(pi05):.1f}" x2="{w-pad}" y2="{Y(pi05):.1f}" stroke="{COL["pi05"]}" '
          f'stroke-width="1.5" stroke-dasharray="5 4"/><text x="{w-pad-90}" y="{Y(pi05)-4:.0f}" font-size="10" '
          f'fill="{COL["pi05"]}">pi0.5={pi05:.4f}</text>') if pi05 else ""
    xlab = "".join(f'<text x="{X(s):.0f}" y="{ht-pad+16:.0f}" font-size="9" fill="#999" text-anchor="middle">{s//1000}k</text>'
                   for s in xs[::max(1, len(xs)//8)])
    return (f'<svg width="{w}" height="{ht}" style="background:#fff;border:1px solid #eee">'
            f'<text x="{w/2:.0f}" y="16" font-size="13" font-weight="600" text-anchor="middle">mae@{h} vs step</text>'
            f'{grid}{pi}{line(dl, COL["delta"])}{line(al, COL["abs"])}{xlab}</svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_run", default="runs/visrobot01_fold_aihc_latent_5x")
    ap.add_argument("--abs_run", default="runs/visrobot01_fold_abs_50k")
    ap.add_argument("--out", default="runs/report_cmp.html")
    args = ap.parse_args()
    delta, abs_ = collect(args.delta_run), collect(args.abs_run)
    pi05 = {}
    for src in (abs_, delta):
        for s in src.values():
            if s.get("_pi05"):
                pi05 = s["_pi05"]; break
        if pi05:
            break

    charts = "".join(svg_chart(h, delta, abs_, float(pi05.get(h)) if pi05.get(h) else None) for h in HORIZONS)

    def last(src):
        return src[max(src)] if src else None
    dl, al = last(delta), last(abs_)
    def best(src, h):
        vs = [src[s][h] for s in src if h in src[s]]
        return min(vs) if vs else None
    rows = ""
    for h in HORIZONS:
        d = dl.get(h) if dl else None; a = al.get(h) if al else None
        ab = best(abs_, h); db = best(delta, h); p = pi05.get(h)
        ratio = f"{a/d:.2f}×" if (a and d) else "—"
        rows += (f"<tr><td>mae@{h}</td><td>{d:.4f}</td><td>{a:.4f}</td><td>{ratio}</td>"
                 f"<td>{db:.4f}</td><td>{ab:.4f}</td><td style='color:#88a'>{p}</td></tr>"
                 if (d and a) else
                 f"<tr><td>mae@{h}</td><td>{d if d else '—'}</td><td>{a if a else '—'}</td><td>—</td>"
                 f"<td>{db if db else '—'}</td><td>{ab if ab else '—'}</td><td style='color:#88a'>{p}</td></tr>")
    dmax = max(delta) if delta else 0; amax = max(abs_) if abs_ else 0
    html = f"""<!doctype html><meta charset=utf-8><title>delta vs abs 对比</title>
<style>body{{font-family:system-ui,Arial;margin:24px;color:#222}}
h1{{font-size:20px}} .leg span{{display:inline-block;margin-right:18px;font-size:13px}}
table{{border-collapse:collapse;margin:14px 0}} td,th{{border:1px solid #ddd;padding:5px 12px;font-size:13px;text-align:right}}
th{{background:#f6f6f6}} td:first-child{{text-align:left;font-weight:600}} .charts svg{{margin:6px}}</style>
<h1>WAM 叠衣服:delta vs abs 动作表示对比(visrobot01_val)</h1>
<p class=leg><span style="color:{COL['delta']}">● delta (5x, batch64)</span>
<span style="color:{COL['abs']}">● abs (batch64, 同配方)</span>
<span style="color:{COL['pi05']}">--- pi0.5 基线</span></p>
<p style="font-size:13px;color:#555">delta 点 {len(delta)} 个(最新 {dmax//1000}k),abs 点 {len(abs_)} 个(最新 {amax//1000}k)。
唯一差别=动作表示(delta 关节减 state vs abs 绝对关节);batch/LR/配方/数据完全一致。↓ 越小越好。</p>
<div class=charts>{charts}</div>
<h3>对标表(最新 step 终值 + 全程最优 best)</h3>
<table><tr><th>指标</th><th>delta 终值</th><th>abs 终值</th><th>abs/delta</th>
<th>delta best</th><th>abs best</th><th>pi0.5</th></tr>{rows}</table>
<p style="font-size:12px;color:#888">注:abs 越接近 delta(ratio→1)或越接近 pi0.5,说明 absolute 表示在充分训练下能对标。
mae@1 的 delta 优势含锚定红利,长 horizon(mae@48)更能反映真实策略质量。</p>"""
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(html)
    print(f"[cmp] delta {len(delta)} pts, abs {len(abs_)} pts -> {args.out}")
    if dl and al:
        print("[cmp] latest: " + " ".join(f"@{h} delta={dl.get(h)} abs={al.get(h)}" for h in HORIZONS))


if __name__ == "__main__":
    main()
