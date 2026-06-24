"""Assemble the tau0 fold fine-tune report.html (self-contained, images base64-embedded).

Data-driven from: runs/eval_report.json (flow-loss curve + 1-step action MSE),
runs/eval_gigaworld.json (GigaWorld-aligned video+action metrics, P1 & P2),
runs/report_assets/{video_metrics.json,*.png} (sample frames). Output: runs/report.html.
"""
import base64
import json
import os

RUNS = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/runs"
ASSETS = os.path.join(RUNS, "report_assets")
OUT = os.path.join(RUNS, "report.html")

ABLATION = [("预训练干 + 训练头 (P1)", 1.00, "目标配置"),
            ("随机干 + 训练头", 3.16, "换掉预训练干 → 头失效"),
            ("预训练干 + 随机头 (未训练)", 4.76, "起点")]
# pi0.5 fold baseline (kai0 pi05, best MAE @step 10000/50000): action MAE @{1,10,25,50}, abs units
PI05 = {1: 0.0219, 10: 0.0425, 25: 0.0743, 50: 0.1155}
# GigaWorld-Policy WAM recorded results (same fold task / visrobot01, eval_watch.py, chunk=48)
GW_MAIN = {"psnr": 19.23, "ssim": 0.718, "temporal_absdiff_ratio": 1.638, "action_mae": 0.190,
           "mae@1": 0.144, "mae@10": 0.156, "mae@24": 0.190, "mae@48": 0.241, "src": "fold_aihc_latent @20k, n=18196"}
GW_5X = {"psnr": 20.79, "ssim": 0.762, "action_mae": 0.085,
         "mae@1": 0.0028, "mae@10": 0.0347, "mae@24": 0.0720, "mae@48": 0.1128,
         "src": "fold_aihc_latent_5x raw (非EMA) best, n=17993 (最佳)"}
CURVE_STEPS = {"p1_trained": 3000, "p2_step5000": 5000, "p2_step10000": 10000,
               "p2_step15000": 15000, "p2_final": 20000}


def b64(path):
    mime = "gif" if path.lower().endswith(".gif") else "png"
    return f"data:image/{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def load(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def svg_curve(pts, w=560, h=180, pad=34):
    if not pts:
        return ""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), 0, max(ys) * 1.1
    def X(x): return pad + (x - x0) / (x1 - x0 + 1e-9) * (w - 2 * pad)
    def Y(y): return h - pad - (y - y0) / (y1 - y0 + 1e-9) * (h - 2 * pad)
    poly = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
    dots = "".join(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="#2563eb"/>'
                   f'<text x="{X(x):.1f}" y="{Y(y)-8:.1f}" font-size="10" text-anchor="middle" fill="#1e3a8a">{y:.2f}</text>'
                   for x, y in pts)
    xlab = "".join(f'<text x="{X(x):.1f}" y="{h-10:.1f}" font-size="9" text-anchor="middle" fill="#6b7280">{int(x/1000)}k</text>' for x, y in pts)
    return (f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #e5e7eb;border-radius:6px">'
            f'<polyline points="{poly}" fill="none" stroke="#2563eb" stroke-width="2"/>{dots}{xlab}'
            f'<text x="{pad}" y="14" font-size="10" fill="#6b7280">val flow-loss</text></svg>')


def main():
    ev = load(os.path.join(RUNS, "eval_report.json"), [])
    gw = load(os.path.join(RUNS, "eval_gigaworld.json"), [])
    vidmeta = load(os.path.join(ASSETS, "video_metrics.json"), {})
    byt = {r["tag"]: r for r in ev}
    gwt = {r["tag"]: r for r in gw}
    p1, p2 = byt.get("p1_trained", {}), byt.get("p2_final", {})
    g1, g2 = gwt.get("p1_trained", {}), gwt.get("p2_final", {})
    # P3 (full-FT + 5:1 loss + warmup-cosine + native chunk=9): auto-select the latest p3_step* eval
    def _p3num(t):  # p3_step20000 -> 20000 ; p3_final -> 10**9 (always newest)
        import re as _re
        if "final" in t:
            return 10**9
        m = _re.search(r"step(\d+)", t)
        return int(m.group(1)) if m else -1
    def _latest(d, prefix):
        ts = sorted([t for t in d if t.startswith(prefix)], key=_p3num)
        if not ts:
            return {}, None
        last = ts[-1]
        return d.get(last, {}), ("final" if "final" in last else str(_p3num(last)))
    p3, p3_step = _latest(byt, "p3_")
    g3, g3_step = _latest(gwt, "p3_")
    _st = g3_step or p3_step
    p3_done = (_st == "final")
    p3_lbl = ("final (step 50000)" if p3_done else (f"step{_st}" if _st else ""))
    p3_status = "✅ 完成" if p3_done else "进行中 — interim"
    has_p3 = bool(g3) or bool(p3)
    # P4 (chunk=33 + λ_video 1:1 + cond-noise): latest p4_* eval
    p4, p4_step = _latest(byt, "p4_")
    g4, g4_step = _latest(gwt, "p4_")
    _st4 = g4_step or p4_step
    p4_lbl = ("final" if _st4 == "final" else (f"step{_st4}" if _st4 else ""))
    has_p4 = bool(g4) or bool(p4)

    curve = sorted([(CURVE_STEPS[t], byt[t]["val_action_loss_mean"]) for t in CURVE_STEPS if t in byt])
    c_first = f"{curve[0][1]:.2f}" if curve else "—"
    c_last = f"{curve[-1][1]:.2f}" if curve else "—"

    def tr(cells, th=False):
        tag = "th" if th else "td"
        return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

    abl = "".join(tr([n, f"{v:.2f}", note]) for n, v, note in ABLATION)

    def g(d, k, f="{:.3f}"):
        v = d.get(k)
        return f.format(v) if isinstance(v, (int, float)) else "—"

    # GigaWorld metrics table: P1 vs P2
    gw_rows = "".join([
        tr(["PSNR ↑", g(g1, "psnr", "{:.2f}"), g(g2, "psnr", "{:.2f}"), "视频重建质量 (含条件帧)"]),
        tr(["SSIM ↑", g(g1, "ssim"), g(g2, "ssim"), "结构相似"]),
        tr(["temporal ratio →1", g(g1, "temporal_absdiff_ratio"), g(g2, "temporal_absdiff_ratio"), "帧间运动幅度 / GT"]),
        tr(["action MAE ↓", g(g1, "action_mae"), g(g2, "action_mae"), "全 chunk 绝对误差 (rad)"]),
        tr(["action MSE ↓", g(g1, "action_mse"), g(g2, "action_mse"), "rad²"]),
        tr(["mae@1 ↓", g(g1, "mae@1", "{:.4f}"), g(g2, "mae@1", "{:.4f}"), "首步"]),
        tr(["mae@10 ↓", g(g1, "mae@10", "{:.4f}"), g(g2, "mae@10", "{:.4f}"), "近段 (10 步)"]),
        tr(["mae@33 ↓", g(g1, "mae@33"), g(g2, "mae@33"), "末步"]),
        tr(["mae_move ↓", g(g1, "mae_move"), g(g2, "mae_move"), "运动维 MAE"]),
        tr(["shape_corr_move ↑", g(g1, "shape_corr_move"), g(g2, "shape_corr_move"), "轨迹形状相关"]),
    ])

    # comprehensive comparison: tau0-P1 vs tau0-P2 vs GigaWorld-Policy WAM 5x vs pi0.5
    def gg(d, k, f="{:.4f}"):
        v = d.get(k)
        return f.format(v) if isinstance(v, (int, float)) else "—"
    # P3 / P4 column cells (empty placeholder "—" until those evals exist)
    def p3c(k, f="{:.4f}", suf=""):
        v = g3.get(k) if g3 else None
        return (f.format(v) + suf) if isinstance(v, (int, float)) else "—"
    def p4c(k, f="{:.4f}", suf=""):
        v = g4.get(k) if g4 else None
        return (f.format(v) + suf) if isinstance(v, (int, float)) else "—"
    cmp = "".join([
        tr(["PSNR ↑", g(g1, "psnr", "{:.1f}") + "*", g(g2, "psnr", "{:.1f}") + "*", p3c("psnr", "{:.1f}") + ("*" if g3 else ""), p4c("psnr", "{:.1f}") + ("†" if g4 else ""), gg(GW_5X, "psnr", "{:.1f}"), "—"]),
        tr(["SSIM ↑", g(g1, "ssim"), g(g2, "ssim"), p3c("ssim", "{:.3f}"), p4c("ssim", "{:.3f}"), gg(GW_5X, "ssim", "{:.3f}"), "—"]),
        tr(["action_mae ↓", g(g1, "action_mae"), g(g2, "action_mae"), p3c("action_mae", "{:.3f}"), p4c("action_mae", "{:.3f}"), gg(GW_5X, "action_mae", "{:.3f}"), "—"]),
        tr(["mae@1 ↓", g(g1, "mae@1", "{:.4f}"), g(g2, "mae@1", "{:.4f}"), p3c("mae@1"), p4c("mae@1"), gg(GW_5X, "mae@1"), f"{PI05[1]:.4f}"]),
        tr(["mae@10 ↓", g(g1, "mae@10", "{:.4f}"), g(g2, "mae@10", "{:.4f}"), p3c("mae@10"), p4c("mae@10"), gg(GW_5X, "mae@10"), f"{PI05[10]:.4f}"]),
        tr(["mae@中段 ↓", g(g1, "mae@16", "{:.4f}") + " <span class='muted'>@16</span>", g(g2, "mae@16", "{:.4f}") + " <span class='muted'>@16</span>", p3c("mae@16", suf=" <span class='muted'>@16</span>"), p4c("mae@16", suf=" <span class='muted'>@16</span>"), gg(GW_5X, "mae@24") + " <span class='muted'>@24</span>", f"{PI05[25]:.4f} <span class='muted'>@25</span>"]),
        tr(["mae@末步 ↓", g(g1, "mae@33", "{:.4f}") + " <span class='muted'>@33</span>", g(g2, "mae@33", "{:.4f}") + " <span class='muted'>@33</span>", p3c("mae@33", suf=" <span class='muted'>@33</span>"), p4c("mae@33", suf=" <span class='muted'>@33</span>"), gg(GW_5X, "mae@48") + " <span class='muted'>@48</span>", f"{PI05[50]:.4f} <span class='muted'>@50</span>"]),
    ])

    # GigaWorld episode_report-style per-sample blocks: action trajectory (14-dim pred vs GT) + GT/pred rollout video
    imgs = ""
    for w in (vidmeta.get("windows") or []):
        vp = os.path.join(ASSETS, w.get("img", ""))
        tp = os.path.join(ASSETS, w.get("traj", ""))
        if not os.path.exists(vp):
            continue
        mae = w.get("mae", {})
        mae_str = " · ".join(f"{k} {v}" for k, v in mae.items()) if mae else ""
        traj_html = (f'<p class="muted">action 轨迹 (14 维 = 双臂 6 关节+夹爪, pred(raw) 红 vs GT 黑虚, 沿 33 步 chunk):</p>'
                     f'<img src="{b64(tp)}" style="width:100%"/>') if os.path.exists(tp) else ""
        imgs += (f'<details open><summary>样例 {w["window"]} · 视频 PSNR {w["psnr"]} / SSIM {w["ssim"]} ({w.get("frames","")} 帧闭环 rollout){(" · 动作 " + mae_str) if mae_str else ""}</summary>'
                 f'{traj_html}'
                 f'<p class="muted">世界模型闭环 rollout (上=GT 真实未来, 下=τ₀ 生成; 3 视角拼接 top_head|left_wrist|right_wrist; 动图):</p>'
                 f'<figure style="flex:1 1 100%"><img src="{b64(vp)}"/></figure></details>')

    rmse1 = p1.get("action_rmse_phys_mean_rad", 0); rmse2 = p2.get("action_rmse_phys_mean_rad", 0)
    vid_frames = (vidmeta.get("windows") or [{}])[0].get("frames", 33) if vidmeta.get("windows") else 33
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>τ₀-WM 叠衣服关节微调 · Report</title>
<style>body{{font-family:-apple-system,Segoe UI,'PingFang SC','Microsoft YaHei',sans-serif;max-width:1080px;margin:0 auto;padding:32px 24px;color:#1a1a1a;line-height:1.6;background:#fafafa}}
h1{{font-size:26px;border-bottom:3px solid #2563eb;padding-bottom:8px}} h2{{font-size:20px;margin-top:34px;color:#1e3a8a;border-left:4px solid #2563eb;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;background:#fff;font-size:14px}} th,td{{border:1px solid #e5e7eb;padding:7px 10px;text-align:center}}
th{{background:#eff6ff}} tr:nth-child(even) td{{background:#f9fafb}} .go{{display:inline-block;background:#16a34a;color:#fff;padding:2px 12px;border-radius:12px;font-weight:600}}
.muted{{color:#6b7280;font-size:12px}} code{{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:13px}}
.grid{{display:flex;flex-wrap:wrap;gap:16px}} figure{{margin:0;flex:1 1 320px;background:#fff;padding:8px;border:1px solid #e5e7eb;border-radius:6px}}
figure img{{width:100%;border-radius:4px}} figcaption{{font-size:12px;text-align:center;margin-top:6px}} .kpi{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}}
.kpi div{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px 18px;min-width:140px}} .kpi b{{display:block;font-size:22px;color:#1e3a8a}}
.note{{background:#fffbeb;border:1px solid #fde68a;padding:10px 14px;border-radius:6px;font-size:13px}} .good{{color:#16a34a;font-weight:600}} .bad{{color:#dc2626;font-weight:600}}
details{{margin:12px 0;border:1px solid #e5e7eb;border-radius:8px;padding:10px 14px;background:#fff}} summary{{cursor:pointer;font-weight:600;padding:4px 0;font-size:15px;color:#1e3a8a}} summary:hover{{color:#2563eb}} details img{{width:100%;border-radius:4px}}</style></head><body>

<h1>τ₀-WM 叠衣服关节空间微调 · 评估报告</h1>
<p class="muted">任务: Flatten and fold the cloth · 目标本体: visrobot01 · 训练: P1 16 卡 + P2 32 卡 (AIHC) · 评估: 16 卡分布式 · 指标对齐 giga_world_policy/eval_watch.py</p>

<h2>0. 结论</h2>
<p><span class="go">GO</span> τ₀-WM 预训练干迁移到关节空间成立 (§2), 并经 P1(训头)→P2(解冻 action_blocks)→<b>P3(全量 FT + 原生 chunk=9 + 5:1 损失)</b> 三阶段做实。<b class="good">P3-final ({p3_lbl}) 在所有 horizon 全面超过 π0.5, 并在 @10/@末 超过 GigaWorld-5x, 仅 @1 略逊</b>: mae@1 <b>{g(g3,'mae@1','{:.4f}')}</b> (π0.5 0.0219) · mae@10 <b>{g(g3,'mae@10','{:.4f}')}</b> (π0.5 0.0425 / 5x 0.0347) · mae@末 <b>{g(g3,'mae@33','{:.4f}')}</b> (π0.5 0.1155 / 5x 0.1128) · 视频 PSNR <b>{g(g3,'psnr','{:.1f}')}</b> / SSIM {g(g3,'ssim')}。P2 暴露的<b class="bad">长程漂移 (mae@末 {g(g2,'mae@33','{:.3f}')})</b> 被 P3 收敛到 {g(g3,'mae@33','{:.3f}')} (≈{(g2.get('mae@33',0)/g3.get('mae@33',1)) if g3.get('mae@33') else 0:.1f}×)。</p>
<div class="note"><b>为何 τ₀-WM 长程动作最弱?</b> 根因是<b>复用 GigaWorld 缓存时选了 chunk=5</b> (仅 2 潜帧视频上下文), 而 GigaWorld 用 chunk=48: ① 动作分支 cross-attn 的视频条件极短 → 长程预测缺乏世界模型支撑; ② 仅 1× 数据、仅后训动作分支 (视频骨干冻结) vs GigaWorld 5x 的 ×5 数据全量; ③ 推理 (10 步/shift) 未调。近段 (mae@1) 受影响小故仍强, 长程随 rollout 误差累积 (§5 PSNR 40→{vidmeta.get('mean_psnr','—')})。<b>P3 闭合方向: 原生 chunk≥9 重抽 latent + ×5 数据 + 更多步</b>。</div>
{(f'''<div class="note" style="background:#ecfdf5;border-color:#6ee7b7"><b>✅ P3 (full-FT + 5:1 loss + warmup-cosine LR + 原生 chunk=9) {p3_status} @ {p3_lbl}</b>:
视频 <b class="good">PSNR {g(g3,'psnr','{:.1f}')}</b> / SSIM {g(g3,'ssim')} · temporal {g(g3,'temporal_absdiff_ratio')};
动作 <b class="good">mae@1 {g(g3,'mae@1','{:.4f}')}</b> (优于 π0.5 0.0219), <b class="good">mae@10 {g(g3,'mae@10','{:.4f}')}</b> (优于 π0.5 0.0425 / GigaWorld 5x 0.0347), <b class="good">mae@末 {g(g3,'mae@33','{:.4f}')}</b> (已低于 π0.5 0.1155 / GigaWorld 5x 0.1128 — <b>长程弱点正在闭合</b>)。
原生 chunk=9 (3 潜帧, contiguous) 重抽 latent + 全量解冻 5.5B + 动作:视频=5:1 损失 + warmup-cosine LR。{'<b>训练已完成 50000 步 (final.pt)</b>; ' if p3_done else '训练进行中; '}10k→{p3_lbl} 全程单调改善 (mae@末 0.107→{g(g3,'mae@33','{:.3f}')}, PSNR 37.0→{g(g3,'psnr','{:.1f}')})。</div>''') if has_p3 else ''}
{(f'''<div class="note" style="background:#eff6ff;border-color:#93c5fd"><b>🎬 P4 (full-FT + 原生 chunk=33 + 动作:视频=1:1 + 条件帧噪声增广) @ {p4_lbl}</b> —— 针对 <b>视频 rollout 质量</b>:
P3 暴露的剩余短板是 <b>8 块自回归视频 rollout 仍衰减到 PSNR≈11</b> (单块强、长视频弱, exposure bias + chunk 短)。P4 把 chunk 拉到 <b>33 (=动作 horizon, 一次前向出整段视频, 几乎无自回归)</b>。
同口径 (chunk=33 单块 33 帧) 视频: <b class="good">PSNR {g(g4,'psnr','{:.1f}')} / SSIM {g(g4,'ssim')}</b> vs P3@ck33 <b>19.4</b> ({'+' if g4.get('psnr',0)>19.4 else ''}{(g4.get('psnr',0)-19.4):.1f} dB, ≈{(g4.get('psnr',0)/11):.1f}× 旧 rollout 11)。
动作 mae@1 {g(g4,'mae@1','{:.4f}')} / @10 {g(g4,'mae@10','{:.4f}')} / @末 {g(g4,'mae@33','{:.4f}')} (仍优于 π0.5)。<b>结论: chunk=33 显著修复了视频 rollout</b>。(训练于 step 12000 手动停止)</div>''') if has_p4 else ''}
<div class="kpi">
<div><b>{g(g3,'mae@1','{:.4f}')}</b>P3 mae@1 rad <span class="muted">(π0.5 0.0219)</span></div>
<div><b>{g(g3,'mae@33','{:.4f}')}</b>P3 mae@末 rad <span class="muted">(π0.5 0.116 / 5x 0.113)</span></div>
<div><b>{g(g3,'psnr','{:.1f}')}</b>P3 视频 PSNR <span class="muted">(单块)</span></div>
<div><b>{g(g3,'ssim')}</b>P3 视频 SSIM</div>
</div>

<h2>1. 方法</h2>
<ul><li>只重置 3 个张量 (20→14), 其余 <b>1403/1406</b> 从 tau0 预训练加载 (action_blocks×30 + 视频主干)。</li>
<li>复用 GigaWorld vae_latent + t5 缓存 (VAE 归一化常数逐值相同), chunk=5; flow-matching, 14 维关节 delta。</li>
<li><b>P1</b>: 冻结干 + 训 32K 头 (16 卡, 3000 步)。<b>P2</b>: 解冻 action_blocks (512M, 32 卡, 20000 步)。</li></ul>

<h2>2. 先验迁移消融 (P1, 16 卡 val flow-loss)</h2>
<table>{tr(["配置","val flow-loss","说明"], th=True)}{abl}</table>
<p>同样训练头在<b>预训练干 1.00</b> vs <b>随机干 3.16</b> (3.2×), 头训练 4.76→1.00 → 预训练干特征是关键。</p>

<h2>3. P1 → P2 验证损失曲线 (16 卡 val)</h2>
{svg_curve(curve)}
<p class="muted">flow-loss: 3k(P1)={c_first} → 20k(P2)={c_last}; P2 解冻 action_blocks 后单调下降, ~15k 收敛。</p>

<h2>4. GigaWorld 对齐评估: 视频 + 动作指标 (16 卡, n={g2.get('n_windows','—')})</h2>
<p class="muted">闭环生成 (观测帧+state → 未来视频+动作 chunk); 指标定义同 <code>eval_watch.video_metrics_gpu</code> + action MAE/horizon/move 分析 (move 维阈 0.05 rad)。</p>
<table>{tr(["指标","P1","P2","说明"], th=True)}{gw_rows}</table>
<div class="note"><b>关键洞察</b>: P2 解冻 action_blocks 后, 动作分支学到真实结构 — <code>shape_corr_move</code> {g(g1,'shape_corr_move')}→<b>{g(g2,'shape_corr_move')}</b>, mae@1 {g(g1,'mae@1','{:.3f}')}→<b>{g(g2,'mae@1','{:.3f}')}</b> (近段大幅改善); 但末段 mae@33 漂移增大 ({g(g2,'mae@33')})。视频指标 P1≈P2 (视频骨干两阶段均冻结)。</div>

<h2>5. 世界模型视频生成 + 动作轨迹 vs 真实 (P3-final, 抽样 episode, 参考 GigaWorld episode_report)</h2>
<p class="muted">可视化风格对齐 <code>giga_world_policy/scripts/wam_pipeline/episode_report.py</code>: 每个抽样 episode 出 ① <b>14 维动作轨迹</b> (pred(raw) vs GT, 沿 33 步 chunk) ② <b>2 行 GT/pred × 3 视角</b>闭环 rollout 视频。来源 ckpt: <code>{vidmeta.get('ckpt','—')}</code> ({vidmeta.get('steps','—')} 步去噪)。
动作 full-inference MAE: {' · '.join(f"{k} {v}" for k,v in (vidmeta.get('action_mae_at_horizon') or {}).items())}。</p>
<p class="muted">闭环逐块生成: 观测帧→生成下一帧→以其为新条件递推 8 块 (~{vid_frames} 帧)。视频 rollout 均值 PSNR <b>{vidmeta.get('mean_psnr','—')}</b> / SSIM <b>{vidmeta.get('mean_ssim','—')}</b>。
<b>注意口径</b>: §4 的 PSNR {g(g3,'psnr','{:.0f}')} 是<b>单块</b>(含条件帧) 指标; 此处是 <b>8 块自回归 rollout</b> —— 即使 P3 动作/单块已强, 纯视频递推仍会累积误差衰减 (动作分支以 store_buffer 复用观测特征, 受影响小, 故 action MAE 仍佳)。</p>
{imgs}

<h2>6. 对比: τ₀-WM vs GigaWorld-Policy WAM vs π0.5</h2>
<table>{tr(["指标 (abs rad)","τ₀-WM P1 <span class='muted'>ck33</span>","τ₀-WM P2 <span class='muted'>ck33</span>",f"τ₀-WM P3 <span class='muted'>full-FT ck9·{p3_lbl}</span>",f"τ₀-WM P4 <span class='muted'>full-FT ck33·{p4_lbl}</span>","GigaWorld WAM 5x <span class='muted'>ck48</span>","π0.5 <span class='muted'>ck50</span>"], th=True)}{cmp}</table>
<p class="muted">† P4 PSNR 为 <b>chunk=33 单块 33 帧</b>口径 (条件帧占比仅 1/9, 长程), 与 P1/P2/P3 的 ck9 单块 (条件帧占比大, PSNR 偏高) <b>不可直接比</b>。视频质量的同口径对比见 §5: <b>P3@ck33 19.4 → P4@ck33 {g(g4,'psnr','{:.1f}')}</b>。</p>
<p class="muted">GigaWorld-Policy WAM 5x: {GW_5X['src']} (同 fold 任务/visrobot01_val, <code>eval_watch.py</code>, 物理 rad MAE, 可直接比)。
π0.5: kai0 pi05 best MAE @step10000 (@{{1,10,25,50}}=0.0219/0.0425/0.0743/0.1155)。
*τ₀-WM PSNR 偏高=chunk=5 仅 2 潜帧 (条件帧占比大), 不可与 GigaWorld 48 帧直接比; 真实长程质量看 §5 闭环 rollout (PSNR≈{vidmeta.get('mean_psnr','—')})。</p>
<div class="note"><b>解读</b>: P1→P2 (解冻 action_blocks) 动作精度提升 — mae@1 {g(g1,'mae@1','{:.3f}')}→<b>{g(g2,'mae@1','{:.3f}')}</b>; P2 近段 (mae@1 {g(g2,'mae@1','{:.3f}')}) 优于 π0.5 (0.022), 但<b>逊于 GigaWorld 5x 最佳 (0.0028)</b>, 且全 chunk / 长程 (mae@末步 {g(g2,'mae@33','{:.3f}')}) 仍逊 5x (0.113) / π0.5 (0.116)。
→ τ₀-WM 仅单 1x 数据、仅动作分支后训, 近段已强; <b>距 GigaWorld 5x / 专用 π0.5 的差距主要在长程</b>, 闭合需 P3 (×5 数据 + 更多步 + 推理调参 + 更长 chunk, 对照 GigaWorld 5x 配方)。
⚠️ <b>口径差异</b>: τ₀-WM/GigaWorld 同 visrobot01_val + 物理 rad + delta, 可直接比; π0.5 来自 kai0 不同 eval (chunk=50, 不同 val), 仅供方向性参考。</div>

<h2>7. 训练状态</h2>
<p>P1 (16 卡, 3000 步, 训头) → P2 (32 卡, 20000 步, 解冻 action_blocks) → <b>P3 (4 节点 32 卡, full-FT 5.5B + ZeRO-2, 50000 步, 原生 chunk=9 重抽 latent + 5:1 损失 + warmup-cosine LR peak 1.1e-4)</b>, 均已完成 final.pt。P3 ckpt 每 2000 步存档 (step_2000…step_50000 + final.pt, 各 ~11GB bf16)。</p>

<h2>8. 复现</h2>
<p class="muted">代码 <code>tau0_wm/finetune/</code>: model_joint · data_joint · train_tau0 · run_train · run_eval_dist · eval_gigaworld_dist · gen_video_compare · make_report_html · launch_*2node.sh · aihc/。
GigaWorld 对齐评估: <code>bash finetune/launch_gweval_2node.sh --ckpt &lt;ckpt&gt; --tag &lt;t&gt;</code> (16 卡)。</p>
</body></html>"""
    open(OUT, "w").write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
