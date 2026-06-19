"""FastWAM 完整 report.html —— 版式/坐标轴/视频完全对齐 gwp episode_report.py。
- viz episode 用【全部 window】(同 gwp;非 max_win 子集)→ 全集长视频长度、traj 密度、GT 轴范围都与 gwp 一致。
- 视频:每帧从模型拼接图裁出 3 个 cam 区域【横排】(top_head | hand_left | hand_right),GT/pred 垂直 2 行,fps=5。
- action 曲线:内联 SVG(无 matplotlib),y 轴只由 GT 决定(与 gwp 同公式)→ 坐标轴逐格一致。
- 分卡:--shard_id/--num_shards 各处理一份 viz episode,写 blocks/ + 视频到共享 out_dir;--aggregate 合并 HTML。

worker:  CUDA_VISIBLE_DEVICES=g python report_fastwam.py --shard_id g --num_shards N <共同参数>
合并:    python report_fastwam.py --aggregate --num_shards N --out_dir <同上> --summary <dual/summary.json>
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR)); sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))
from eval_offline_fold import build_model, prep_image, VAL, VK, HOR, _REPO_DIR  # noqa: E402
from fastwam.utils.video_io import save_mp4  # noqa: E402

DIM = [f"L_j{i}" for i in range(6)] + ["L_grip"] + [f"R_j{i}" for i in range(6)] + ["R_grip"]
PI05 = {1: 0.0219, 10: 0.0425, 24: 0.0743, 48: 0.1155}
CSS = """<style>
body{font-family:-apple-system,Segoe UI,monospace;max-width:1180px;margin:24px auto;padding:0 18px;color:#222;line-height:1.5}
h2{border-bottom:2px solid #3a6;padding-bottom:6px} h3{margin-top:26px;color:#3a6}
table{border-collapse:collapse;margin:10px 0} th,td{border:1px solid #ccc;padding:6px 14px;text-align:center}
th{background:#eef5f0} tr:nth-child(even) td{background:#fafafa}
details{margin:12px 0;border:1px solid #ddd;border-radius:8px;padding:10px 14px;background:#fcfcfc}
summary{cursor:pointer;font-weight:600;padding:4px 0;font-size:15px} summary:hover{color:#3a6}
video{margin:6px 8px 6px 0;border:1px solid #ccc;border-radius:6px;vertical-align:top} img{border:1px solid #eee;border-radius:6px}
.vids{display:flex;flex-wrap:wrap;gap:8px} .note{color:#666;font-size:13px}
ul{line-height:1.7} code{background:#f0f0f0;padding:1px 5px;border-radius:3px}
</style>"""


def svg_series(gt, pred, dim, w=545, h=200, pad=26):
    """GT 黑虚线 / pred 红实线;y 轴只由 GT 决定([min-5%,max+5%],与 gwp 同公式)→ 两份报告坐标轴完全一致。"""
    r = float(gt.max() - gt.min()); py = 0.05 * r + 1e-6
    ymin = float(gt.min()) - py; ymax = float(gt.max()) + py
    n = len(gt)
    X = lambda i: pad + i / max(1, n - 1) * (w - 2 * pad)
    def Y(v):
        yy = h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad)
        return max(float(pad), min(float(h - pad), yy))
    def poly(arr, color, dash=""):
        pts = " ".join(f"{X(i):.1f},{Y(arr[i]):.1f}" for i in range(n))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.3"{da}/>'
    axis = (f'<line x1="{pad}" y1="{h-pad:.0f}" x2="{w-pad}" y2="{h-pad:.0f}" stroke="#ccc" stroke-width="0.6"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad:.0f}" stroke="#ccc" stroke-width="0.6"/>'
            f'<text x="{pad}" y="{pad-6}" font-size="9" fill="#444">[{ymin:.2f},{ymax:.2f}]</text>')
    leg = ('<text x="62" y="14" font-size="9" fill="#222">- - GT</text>'
           '<text x="122" y="14" font-size="9" fill="#d62728">— pred(raw)</text>') if dim == 0 else ""
    return (f'<svg width="{w}" height="{h}" style="margin:2px">'
            f'<text x="3" y="13" font-size="11" font-weight="600" fill="#333">{DIM[dim]}</text>{leg}'
            f'{axis}{poly(gt, "#222", "4,3")}{poly(pred, "#d62728")}</svg>')


def _chw_to_pil(t):  # [3,H,W] in [-1,1] -> PIL RGB
    a = ((t.clamp(-1, 1) + 1) * 0.5 * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def _pil_to_input(pil):  # 模型想象帧 PIL(stitched 320x384) -> 模型输入 [3,384,320] in [-1,1]
    return TF.to_tensor(pil) * 2.0 - 1.0


def to_3cam_row(stitched, rh=200):
    """从模型拼接图(PIL,size=(320,384):top 256 在上 / 左右腕 128 在下)裁出 3 个 cam 横排:
    top_head | hand_left | hand_right(与 gwp 3 视角横排一致)。"""
    top = stitched.crop((0, 0, 320, 256)); lf = stitched.crop((0, 256, 160, 384)); rt = stitched.crop((160, 256, 320, 384))
    rs = lambda im: im.resize((max(1, int(im.width * rh / im.height)), rh))
    cams = [rs(top), rs(lf), rs(rt)]
    W = sum(c.width for c in cams); row = Image.new("RGB", (W, rh), (0, 0, 0)); x = 0
    for c in cams:
        row.paste(c, (x, 0)); x += c.width
    return row


def _label(img, txt):
    img = img.copy(); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 92, 15], fill=(0, 0, 0)); d.text((3, 2), txt, fill=(255, 255, 0)); return img


def _save_2row(gt_rows, pred_rows, path, fps=5):
    """GT(上)/pred(下)垂直 2 行(各为 3-cam 横排),帧上标行名。"""
    T = min(len(gt_rows), len(pred_rows)); frames = []
    for i in range(T):
        g = _label(gt_rows[i], "GT"); p = _label(pred_rows[i], "pred(raw)")
        W = max(g.width, p.width); c = Image.new("RGB", (W, g.height + p.height), (0, 0, 0))
        c.paste(g, (0, 0)); c.paste(p, (0, g.height)); frames.append(c)
    save_mp4(frames, path, fps=fps)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--summary", default="")
    ap.add_argument("--n_viz_eps", type=int, default=20); ap.add_argument("--n_metric_eps", type=int, default=100)
    ap.add_argument("--nfe", type=int, default=10); ap.add_argument("--exec_horizon", type=int, default=16)
    ap.add_argument("--n_vid_per_ep", type=int, default=3); ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--shard_id", type=int, default=0); ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    return ap.parse_args()


def viz_ep_list(n_metric, n_viz):
    meta = [json.loads(l) for l in open(f"{VAL}/meta/episodes.jsonl")]
    all_eps = sorted(int(m["episode_index"]) for m in meta)[:n_metric]
    return [all_eps[i] for i in np.unique(np.linspace(0, len(all_eps) - 1, min(n_viz, len(all_eps))).astype(int))]


def worker(args):
    stats = json.load(open(args.stats))
    a_mean = np.array(stats["action"]["default"]["global_mean"]); a_std = np.array(stats["action"]["default"]["global_std"])
    s_mean = np.array(stats["state"]["default"]["global_mean"]); s_std = np.array(stats["state"]["default"]["global_std"])
    _te = os.environ.get("EVAL_TEXT_EMB", "visrobot01_fold")
    t5 = torch.load(list((_REPO_DIR / "data" / "text_embeds_cache" / _te).glob("*.pt"))[0], map_location="cpu", weights_only=False)
    ctx = t5["context"].clone(); cmask = t5["mask"].bool(); ctx[~cmask] = 0.0; cmask = torch.ones_like(cmask)
    if ctx.ndim == 2: ctx = ctx.unsqueeze(0)
    if cmask.ndim == 1: cmask = cmask.unsqueeze(0)
    model = build_model(args.weights)
    from torchcodec.decoders import VideoDecoder

    my = viz_ep_list(args.n_metric_eps, args.n_viz_eps)[args.shard_id::args.num_shards]
    print(f"[report shard {args.shard_id}/{args.num_shards}] eps={my}", flush=True)
    H = args.exec_horizon; deltas = np.linspace(0, 48, 5).astype(int)
    ctxd = ctx.to(model.device, model.torch_dtype); cmaskd = cmask.to(model.device)
    os.makedirs(f"{args.out_dir}/blocks", exist_ok=True)

    for ep in my:
        df = pd.read_parquet(f"{VAL}/data/chunk-000/episode_{ep:06d}.parquet")
        gt_all = np.stack(df["action"].to_numpy())[:, :14]; st_all = np.stack(df["observation.state"].to_numpy())[:, :14]
        decs = {k: VideoDecoder(f"{VAL}/videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4") for k in VK}
        L = len(df)
        wins = list(range(0, max(1, L - 48), args.exec_horizon))  # 全部 window(同 gwp viz)
        def stitch_t(ts):
            fr = {k: decs[k].get_frames_at([min(ts, decs[k].metadata.num_frames - 1)]).data[0].permute(1, 2, 0).numpy() for k in VK}
            return prep_image(fr)
        vid_wins = set(wins[:: max(1, len(wins) // max(1, args.n_vid_per_ep))][: args.n_vid_per_ep])
        tp, tg, aes = [], [], []; gt_rows, pred_rows = [], []; bvids = []; vms = []
        win_frames = {}
        for f in wins:
            img = stitch_t(f)
            prop = torch.from_numpy((st_all[f] - s_mean) / (s_std + 1e-8)).float()
            gt_act = torch.from_numpy((gt_all[f:f + 48] - a_mean) / (a_std + 1e-8)).float()
            t0 = time.time()
            with torch.no_grad():
                ov = model.infer(prompt=None, input_image=img, num_frames=5, action=gt_act, action_horizon=48,
                                 proprio=prop, text_cfg_scale=1.0, action_cfg_scale=1.0,
                                 num_inference_steps=args.nfe, seed=42, tiled=False, context=ctxd, context_mask=cmaskd)
            vms.append((time.time() - t0) * 1000)
            pa = ov["action"].float().cpu().numpy() * (a_std + 1e-8) + a_mean
            gt = gt_all[f:f + 48]; n = min(len(pa), len(gt)); aes.append(np.abs(pa[:n] - gt[:n]))
            tp.append(pa[:min(H, n)]); tg.append(gt[:min(H, n)])
            g_rows = [to_3cam_row(_chw_to_pil(stitch_t(min(f + int(d), L - 1)))) for d in deltas]
            p_rows = [to_3cam_row(p) for p in ov["video"]]
            gt_rows += g_rows; pred_rows += p_rows
            if f in vid_wins: win_frames[f] = (g_rows, p_rows)
        ae = np.concatenate(aes, 0); cum = {h: float(ae[:h].mean()) for h in HOR if h <= len(ae)}
        full = f"ep{ep}_full.mp4"; _save_2row(gt_rows, pred_rows, os.path.join(args.out_dir, full), args.fps)
        for f, (g_rows, p_rows) in win_frames.items():
            bp = f"ep{ep}_w{f}.mp4"; _save_2row(g_rows, p_rows, os.path.join(args.out_dir, bp), args.fps); bvids.append(bp)
        P = np.concatenate(tp, 0); G = np.concatenate(tg, 0)
        svgs = "".join(svg_series(G[:, d], P[:, d], d) for d in range(14))
        fullh = f'<p class=note><b>A 全 episode 长视频</b>(沿所有 window 拼,teacher-forced:每窗重锚 GT):</p><video src="{full}" controls width="720"></video>'
        bvs = "".join(f'<video src="{v}" controls width="460"></video>' for v in bvids)
        bsec = f'<p class=note><b>B 代表窗短视频</b>(各 1s):</p><div class=vids>{bvs}</div>' if bvs else ""

        # ---- C 开环 rollout(exec-jump=exec_horizon;仅喂第0帧,之后用模型想象帧+预测state自驱动)----
        EX = args.exec_horizon; near = int(np.argmin([abs(int(d) - EX) for d in deltas]))  # 最接近 exec 的关键帧
        cur_img = stitch_t(0)
        cur_state = torch.from_numpy((st_all[0] - s_mean) / (s_std + 1e-8)).float()
        ol_pa, ol_gt, ol_pf, ol_gf = [], [], [], []; t = 0
        while t + EX <= L - 1:
            with torch.no_grad():
                oo = model.infer(prompt=None, input_image=cur_img, num_frames=5, action=None, action_horizon=48,
                                 proprio=cur_state, text_cfg_scale=1.0, action_cfg_scale=1.0,
                                 num_inference_steps=args.nfe, seed=42, tiled=False, context=ctxd, context_mask=cmaskd)
            vms.append(0.0)  # 计入推理数(时延已在 A 段统计口径)
            pa_ol = oo["action"].float().cpu().numpy() * (a_std + 1e-8) + a_mean
            pv_ol = oo["video"]
            ol_pa.append(pa_ol[:EX]); ol_gt.append(gt_all[t:t + EX])
            ol_pf.append(to_3cam_row(pv_ol[near])); ol_gf.append(to_3cam_row(_chw_to_pil(stitch_t(min(t + EX, L - 1)))))
            cur_img = _pil_to_input(pv_ol[near])                               # 开环:下一步输入=模型想象帧
            cur_state = torch.from_numpy((pa_ol[EX - 1] - s_mean) / (s_std + 1e-8)).float()  # 预测 state(abs:动作=目标关节位)
            t += EX
        roll = f"ep{ep}_rollout.mp4"; _save_2row(ol_gf, ol_pf, os.path.join(args.out_dir, roll), args.fps)
        OP = np.concatenate(ol_pa, 0); OG = np.concatenate(ol_gt, 0)
        roll_mae = float(np.abs(OP - OG).mean())
        roll_svgs = "".join(svg_series(OG[:, d], OP[:, d], d) for d in range(14))
        rollh = (f'<p class=note><b>C 开环 rollout 长视频</b>(仅喂第0帧,之后喂模型自己想象帧+预测state 自驱动,exec-jump={EX};暴露误差累积):</p>'
                 f'<video src="{roll}" controls width="720"></video>'
                 f'<p class=note>开环 rollout action 曲线(raw vs GT,全程自驱动累积,rollout MAE={roll_mae:.4f}):</p>'
                 f'<div style="display:flex;flex-wrap:wrap;max-width:1130px">{roll_svgs}</div>')

        block = (f'<details><summary>ep {ep} &nbsp;|&nbsp; n_win={len(wins)} &nbsp; mae@1={cum.get(1,0):.4f} &nbsp; mae@48={cum.get(48,0):.4f} &nbsp; rollout_mae={roll_mae:.4f}</summary>'
                 f'<p class=note>action 曲线(raw vs GT,沿 exec_horizon 拼接,teacher-forced):</p>'
                 f'<div style="display:flex;flex-wrap:wrap;max-width:1130px">{svgs}</div>{fullh}{bsec}{rollh}</details>')
        json.dump({"ep": int(ep), "html": block, "vms_sum": float(np.sum(vms)), "vms_n": len(vms)},
                  open(f"{args.out_dir}/blocks/ep{ep:06d}.json", "w"))
        print(f"[report shard {args.shard_id}] ep{ep} n_win={len(wins)} cum@48={cum.get(48):.4f} rollout_mae={roll_mae:.4f}", flush=True)


def aggregate(args):
    blks = sorted(glob.glob(f"{args.out_dir}/blocks/ep*.json"))
    data = [json.load(open(b)) for b in blks]
    blocks = "".join(d["html"] for d in sorted(data, key=lambda r: r["ep"]))
    vms = sum(d["vms_sum"] for d in data); vn = sum(d["vms_n"] for d in data)
    rows_html = ""; n_metric = "?"; act_ms = 0
    if args.summary and os.path.isfile(args.summary):
        sm = json.load(open(args.summary)); cmm = sm.get("cum_mae", {}); pi = sm.get("pi05", PI05)
        rows_html = "".join(f"<tr><td>{h}</td><td><b>{float(cmm.get(str(h),0)):.4f}</b></td><td>{float(pi.get(str(h),PI05[h])):.4f}</td></tr>" for h in HOR)
        n_metric = sm.get("n_metric_eps", "?"); act_ms = sm.get("latency", {}).get("action_ms", 0)
    la = f"action-only <b>{act_ms:.0f} ms</b> · with-video {vms/max(1,vn):.0f} ms · 去噪 {args.nfe} 步"
    ckname = os.path.basename(os.path.dirname(os.path.dirname(args.weights))) + "/" + Path(args.weights).stem
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>Episode Report — FastWAM-v6</title>{CSS}</head><body>
<h2>Episode 测试报告 — FastWAM-v6 ({ckname})</h2>
<p class=note>held-out:指标 {n_metric} ep / 可视化 {len(data)} ep · exec_horizon={args.exec_horizon} · <b>开环(非闭环 SR)</b> · 与 gwp_abs_v5 报告同协议/同 episode/同坐标轴</p>
<h3>视频说明</h3><ul>
<li>每视频 <b>2 行</b>(原分辨率,帧上有行名):<b>行1 GT(真值) / 行2 pred(模型 raw 权重想象)</b></li>
<li>每行内 3 视角横排:<b>top_head(头) | hand_left(左腕) | hand_right(右腕)</b></li>
<li><b>A 全 episode 长视频</b>(<code>ep*_full.mp4</code>,teacher-forced):沿 exec_horizon 取该集所有 window 拼接,<b>每窗重锚 GT 当前帧</b>;时间轴=window 序列,每 window 5 帧=action chunk 的 delta[0,12,24,36,48]</li>
<li><b>B 代表窗短视频</b>(<code>ep*_w*.mp4</code>):该集 {args.n_vid_per_ep} 个代表 window,各 5 帧 1s</li>
<li><b>C 开环 rollout 长视频</b>(<code>ep*_rollout.mp4</code>):<b>仅喂第 0 帧真值</b>,之后喂模型自己想象的帧 + 预测 state 自驱动(exec-jump=exec_horizon)→ 暴露世界模型的<b>误差累积</b>;另附开环 action 曲线 + rollout MAE</li></ul>
<h3>推理性能</h3><p>{la}</p>
<h3>聚合指标(raw,全 {n_metric} ep · cumulative mae@h)</h3>
<table><tr><th>horizon</th><th>action MAE</th><th>π0.5 参考</th></tr>{rows_html}</table>
<h3>逐 episode(抽样可视化)</h3>{blocks}
</body></html>"""
    open(os.path.join(args.out_dir, "report.html"), "w").write(html)
    print(f"[aggregate] {len(data)} eps -> {args.out_dir}/report.html", flush=True)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.aggregate:
        aggregate(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
