"""Episode 测试报告(分片版):全量 episode 指标 + 抽样 episode 可视化。开环逐窗口,非闭环 SR。
可视化每 viz episode 产:
  - action 曲线(14维 raw vs GT,沿 exec_horizon 拼接的部署式轨迹)
  - 视频(2 行: GT / pred(raw); 每行 3 视角横排 cam_high|cam_left|cam_right; 帧上标行名):
      A 全 episode 长视频 ep<ID>_full.mp4  (沿所有 window 拼接,每 window 5 帧 delta[0,12,24,36,48])
      B 代表窗短视频     ep<ID>_w<f>.mp4   (N 个代表 window,各 5 帧 1s)
分布式:metric_eps[shard_id::num_shards] 分片,出 shards/shard_<id>.json + 共享 episodes/;--aggregate 合并 HTML。
orchestrator: run_report_dist.sh
"""
import argparse, os, sys, glob, json, time, base64, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import OrderedDict
from wam_pipeline.eval_watch import build_window_indices, EpisodeFrameCache, _hwc_to_chw01, _to_thwc_gpu
VK = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
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


def _label(frames, text):
    import cv2
    out = np.ascontiguousarray(frames.copy())
    for t in range(len(out)):
        cv2.putText(out[t], text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return out


def _save_2row(gt, raw, path, fps=5):
    """GT/raw 垂直 2 行(原分辨率,不缩),帧上标行名。各 [T,H,W,C] uint8。"""
    import torchvision
    T = min(len(gt), len(raw))
    cat = np.concatenate([_label(gt[:T], "GT"), _label(raw[:T], "pred(raw)")], axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.io.write_video(path, torch.from_numpy(cat), fps=fps)


def get_args():
    ap = argparse.ArgumentParser()
    for a in ["val_root", "out_dir"]:
        ap.add_argument("--" + a, required=True)
    for a in ["transformer_dir", "model_id", "stats_path", "t5_pkl", "ema_dir"]:
        ap.add_argument("--" + a, default=None)
    ap.add_argument("--n_metric_eps", type=int, default=200); ap.add_argument("--n_viz_eps", type=int, default=20)
    ap.add_argument("--n_vid_per_ep", type=int, default=3); ap.add_argument("--n_ema_eps", type=int, default=0)
    ap.add_argument("--max_win_per_ep", type=int, default=6); ap.add_argument("--full_video", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0); ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--exec_horizon", type=int, default=16); ap.add_argument("--action_chunk", type=int, default=48)
    ap.add_argument("--steps_inf", type=int, default=10); ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--steps_act", type=int, default=0)
    ap.add_argument("--frame_cache", type=int, default=4)
    ap.add_argument("--engine", default="stock", choices=["stock", "opt"])  # opt=prefix-KV/compile 优化引擎
    ap.add_argument("--opt_tier", default="exact", choices=["eager", "exact", "fp8"])
    ap.add_argument("--opt_bac", type=int, default=0)  # BAC 跳过中段 block 数(0=off)  # 每进程缓存解码 episode 数;62G 级主机(如 jpsz)用 1  # ANS 动作步数 T_a;0=自动(ANS ckpt→5,其余同步)
    ap.add_argument("--delta_mask", default="")  # 空=从 --stats_path 内嵌 delta_mask 取(默认);传 "1,1,..,0" 覆盖
    ap.add_argument("--width", type=int, default=768); ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--view_keys", default="", help="逗号分隔的相机键名;空=用默认 v1 键名")
    return ap.parse_args()


def plan(args):
    idx, _, info = build_window_indices(args.val_root, "exec", args.exec_horizon, args.action_chunk, args.exec_horizon)
    ep2win = OrderedDict()
    for gi in idx:
        ep2win.setdefault(info[gi][0], []).append(gi)
    eps = list(ep2win.keys()); metric = eps[:args.n_metric_eps]
    viz = [metric[i] for i in np.unique(np.linspace(0, len(metric) - 1, min(args.n_viz_eps, len(metric))).astype(int))]
    return info, ep2win, metric, viz


def main():
    args = get_args()
    HOR = sorted({h for h in (1, 10, args.action_chunk // 2, args.action_chunk) if h <= args.action_chunk})
    info, ep2win, metric_eps, viz_eps = plan(args)
    if args.aggregate:
        return aggregate(args, viz_eps, HOR)
    os.makedirs(os.path.join(args.out_dir, "shards"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "episodes"), exist_ok=True)
    dev, dt = "cuda", torch.bfloat16
    from giga_datasets import load_dataset
    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
    from world_action_model.pipeline.wa_pipeline import WAPipeline
    from world_action_model.pipeline.utils import (extract_normalization_tensors, load_stats,
        load_t5_embedding_from_pkl, denormalize_action, add_state_to_action, normalize_state, build_ref_image)
    from diffusers.models import AutoencoderKLWan
    stats = load_stats(args.stats_path); norm = extract_normalization_tensors(stats, device=dev, state_dim=14, action_dim=14)
    t5 = load_t5_embedding_from_pkl(args.t5_pkl, target_len=64).to(dev, torch.float32)
    if args.delta_mask.strip():
        _dm = [c == "1" for c in args.delta_mask.split(",")]
    else:
        from world_action_model.pipeline.utils import resolve_delta_mask
        _dm = resolve_delta_mask(stats, 14).tolist()
    dm = torch.tensor(_dm, device=dev, dtype=torch.bool)
    ve = dict(_class_name="LeRobotDataset", data_path=args.val_root, delta_info={"action": args.action_chunk},
              skip_video_decoding=True, embodiment="visrobot01", tolerance_s=1e-3)
    vk = [k.strip() for k in args.view_keys.split(",")] if args.view_keys.strip() else VK
    ds = load_dataset([ve]); fc = EpisodeFrameCache(args.val_root, vk, args.frame_cache)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dt)
    sid, n = args.shard_id, args.num_shards
    my_metric = metric_eps[sid::n]; vset = set(viz_eps)
    print(f"[report shard {sid}/{n}] metric {len(my_metric)} viz {len(vset & set(my_metric))}", flush=True)

    tf = CasualWorldActionTransformer.from_pretrained(args.transformer_dir).to(dt)
    raw_pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=tf, torch_dtype=dt).to(dev)
    # world-model-lookahead ckpts (action_attends_video=True) must denoise the video latents the
    # action tokens attend to -> the action_only fast path is invalid; force full denoising for all eps.
    LOOKAHEAD = bool(getattr(tf.config, "action_attends_video", False))
    if LOOKAHEAD:
        print(f"[report shard {sid}/{n}] action_attends_video=True -> full denoise (action_only disabled)", flush=True)
    # ANS ckpt:动作 T_a 步先出(默认 5),metric eps 不需要视频时提前返回(finish_video=False)
    ANS = bool(getattr(tf.config, "async_noise", False))
    STEPS_ACT = (args.steps_act or None) if not ANS else (args.steps_act or 5)
    if ANS:
        print(f"[report shard {sid}/{n}] async_noise=True -> T_a={STEPS_ACT}/T_O={args.steps_inf}", flush=True)
    OPT = args.engine == "opt"
    if OPT:
        from scripts.opt_ans import AnsPrefixRunner, opt_call
        from scripts.prefix_cache import PrefixCachedRunner
        if args.opt_tier == "fp8":
            from scripts.fp8_linear import swap_linears_to_fp8
            print(f"[report shard {sid}/{n}] fp8 swapped {swap_linears_to_fp8(tf.blocks)} linears", flush=True)
        if args.opt_tier in ("exact", "fp8"):
            for _mod in tf.modules():
                if hasattr(_mod, "fuse_projections") and hasattr(_mod, "set_processor"):
                    try: _mod.fuse_projections()
                    except Exception: pass
        _runner = AnsPrefixRunner(tf) if LOOKAHEAD else PrefixCachedRunner(tf)
        if args.opt_tier in ("exact", "fp8"):
            _runner.compile_prepare("reduce-overhead")
            (_runner.compile_step_ans if LOOKAHEAD else _runner.compile_step)("reduce-overhead")
        if args.opt_bac:
            _runner.init_bac(len(tf.blocks))
            if args.opt_tier in ("exact", "fp8"):
                (_runner.compile_bac_ans if LOOKAHEAD else _runner.compile_bac)()
        print(f"[report shard {sid}/{n}] engine=opt tier={args.opt_tier} runner={type(_runner).__name__}", flush=True)

    def infer(gi, want_video):
        d = ds[int(gi)]; ep, f = info[int(gi)]; fr = fc.get(ep)
        ref = build_ref_image(images={k: _hwc_to_chw01(fr[k][f]) for k in vk}, dst_size=(args.width, args.height), crop_mode="center")
        st = d["observation.state"].float().unsqueeze(0).to(dev); ns = normalize_state(st, norm, mode="zscore").to(dev, dt)
        with torch.no_grad():
            if OPT and not want_video:
                act = opt_call(raw_pipe, _runner, image=ref, state=ns,
                               prompt_embeds=t5.unsqueeze(0).to(dev, torch.float32),
                               height=args.height, width=args.width, num_frames=5, action_chunk=args.action_chunk,
                               num_inference_steps=args.steps_inf, action_num_inference_steps=STEPS_ACT,
                               is_ans=LOOKAHEAD, bac_skip=args.opt_bac)
                out = (None, act)
            else:
                out = raw_pipe(height=args.height, width=args.width, action_chunk=args.action_chunk, state=ns, num_frames=5,
                           guidance_scale=0.0, num_inference_steps=args.steps_inf, image=ref,
                           action_only=(not want_video) and not LOOKAHEAD,
                           action_num_inference_steps=STEPS_ACT, finish_video=want_video or STEPS_ACT is None,
                           return_dict=False, prompt_embeds=t5.unsqueeze(0).to(dev, torch.float32))
        imgs, act = out[0], out[1]
        pa = add_state_to_action(denormalize_action(act[0].float(), norm, mode="zscore"), st[0].float().to(act.device),
                                 action_chunk=args.action_chunk, mask=dm).cpu().numpy()
        gt = d["action"].float().numpy()[:, :14]
        pv = _to_thwc_gpu(imgs[0], dev).round().clamp(0, 255).byte().cpu().numpy() if want_video else None
        gv = None
        if want_video:
            offs = [0, args.action_chunk // 4, args.action_chunk // 2, 3 * args.action_chunk // 4, args.action_chunk]
            Lf = fr[vk[0]].shape[0]
            gv = np.stack([np.array(build_ref_image(images={k: _hwc_to_chw01(fr[k][min(f + o, Lf - 1)]) for k in vk},
                          dst_size=(args.width, args.height), crop_mode="center")) for o in offs])
        return ep, f, pa, gt, pv, gv

    def metrics(pa, gt):
        L = min(len(pa), len(gt)); ae = np.abs(pa[:L] - gt[:L]); m = {"action_mae": float(ae.mean())}
        for h in HOR:
            if h <= L: m[f"mae@{h}"] = float(ae[:h].mean())
        return m

    latency = {}
    if sid == 0 and my_metric:
        g0 = ep2win[my_metric[0]][0]
        for _ in range(2): infer(g0, False)
        torch.cuda.synchronize(); t = time.time(); [infer(g0, False) for _ in range(5)]; torch.cuda.synchronize()
        latency["action_ms"] = (time.time() - t) / 5 * 1000
        if not OPT:  # opt 引擎只出动作,跳过带视频路径的延迟测量(stock 前向与 read 模式 processor 不兼容)
            torch.cuda.synchronize(); t = time.time(); [infer(g0, True) for _ in range(3)]; torch.cuda.synchronize()
            latency["video_ms"] = (time.time() - t) / 3 * 1000

    rows = {}
    for k, ep in enumerate(my_metric):
        if k + 1 < len(my_metric):
            fc.prefetch(my_metric[k + 1])  # 藏解码:后台预取下一个 episode
        wins = ep2win[ep]; is_v = ep in vset
        em = {kk: [] for kk in ["action_mae"] + [f"mae@{h}" for h in HOR]}
        if not is_v:
            ws = wins if len(wins) <= args.max_win_per_ep else [wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, args.max_win_per_ep).astype(int))]
            for gi in ws:
                _, _, pa, gt, _, _ = infer(gi, False)
                for kk, v in metrics(pa, gt).items(): em[kk].append(v)
            rows[int(ep)] = {"ep": int(ep), "n_win": len(wins), **{kk: float(np.mean(em[kk])) for kk in em if em[kk]}}
        else:
            tp, tg = {}, {}; gt_ep, raw_ep = [], []; bvids = []
            vid_wins = set(wins[:: max(1, len(wins) // args.n_vid_per_ep)][:args.n_vid_per_ep])
            for gi in wins:
                _, f, pa, gt, pv, gv = infer(gi, True)
                for kk, v in metrics(pa, gt).items(): em[kk].append(v)
                h = args.exec_horizon; tp[f] = pa[:h].tolist(); tg[f] = gt[:h].tolist()
                gt_ep.append(gv); raw_ep.append(pv)
                if gi in vid_wins:
                    bp = f"episodes/ep{ep}_w{f}.mp4"; _save_2row(gv, pv, os.path.join(args.out_dir, bp), args.fps); bvids.append(bp)
            full = None
            if args.full_video and gt_ep:
                full = f"episodes/ep{ep}_full.mp4"
                _save_2row(np.concatenate(gt_ep), np.concatenate(raw_ep), os.path.join(args.out_dir, full), args.fps)
            fs = sorted(tp.keys()); P = np.concatenate([np.array(tp[f]) for f in fs]); G = np.concatenate([np.array(tg[f]) for f in fs])
            x = np.arange(len(P)); fig, axes = plt.subplots(7, 2, figsize=(13, 15)); axes = axes.flatten()
            for dd in range(14):
                axes[dd].plot(x, G[:, dd], "k--", lw=1.5, label="GT"); axes[dd].plot(x, P[:, dd], "r-", lw=1.2, label="pred(raw)")
                axes[dd].set_title(DIM[dd], fontsize=8)
                if dd == 0: axes[dd].legend(fontsize=7)
            fig.suptitle(f"episode {ep} — deploy-style action traj (raw, exec_h={args.exec_horizon})")
            tpng = f"episodes/ep{ep}_traj.png"; fig.savefig(os.path.join(args.out_dir, tpng), dpi=70, bbox_inches="tight"); plt.close(fig)
            rows[int(ep)] = {"ep": int(ep), "n_win": len(wins), **{kk: float(np.mean(em[kk])) for kk in em if em[kk]},
                             "traj_png": tpng, "vids": bvids, "full_video": full}
        if k % 3 == 0: print(f"[shard {sid}] {k+1}/{len(my_metric)} ep{ep}{' viz' if is_v else ''}", flush=True)
    del raw_pipe, tf; torch.cuda.empty_cache()
    json.dump({"metric": rows, "latency": latency}, open(os.path.join(args.out_dir, "shards", f"shard_{sid}.json"), "w"))
    print(f"[report shard {sid}] done -> shards/shard_{sid}.json", flush=True)


def aggregate(args, viz_eps, HOR):
    sh = sorted(glob.glob(os.path.join(args.out_dir, "shards", "shard_*.json")))
    metric, lat = {}, {}
    for f in sh:
        d = json.load(open(f)); metric.update({int(k): v for k, v in d["metric"].items()}); lat.update(d.get("latency", {}))
    print(f"[aggregate] merged {len(sh)} shards, {len(metric)} ep", flush=True)
    agg = {f"mae@{h}": float(np.mean([r[f"mae@{h}"] for r in metric.values() if r.get(f"mae@{h}") is not None])) for h in HOR}

    def b64(p):
        fp = os.path.join(args.out_dir, p)
        return "data:image/png;base64," + base64.b64encode(open(fp, "rb").read()).decode() if os.path.isfile(fp) else ""
    blocks = []
    for ep in viz_eps:
        r = metric.get(ep)
        if not r or not r.get("traj_png"): continue
        full = f'<p class=note><b>A 全 episode 长视频</b>(沿所有 window 拼):</p><video src="{r["full_video"]}" controls width="820"></video>' if r.get("full_video") else ""
        bvs = "".join(f'<video src="{v}" controls width="400"></video>' for v in r.get("vids", []))
        bsec = f'<p class=note><b>B 代表窗短视频</b>(各 1s):</p><div class=vids>{bvs}</div>' if bvs else ""
        blocks.append(f'<details><summary>ep {ep} &nbsp;|&nbsp; n_win={r["n_win"]} &nbsp; mae@1={r["mae@1"]:.4f} &nbsp; mae@48={r["mae@48"]:.4f}</summary>'
                      f'<p class=note>action 曲线(raw vs GT,沿 exec_horizon 拼接):</p><img src="{b64(r["traj_png"])}" width="920">{full}{bsec}</details>')
    rows_html = "".join(f"<tr><td>{h}</td><td><b>{agg[f'mae@{h}']:.4f}</b></td><td>{PI05[h]:.4f}</td></tr>" for h in HOR)
    la = f"action-only <b>{lat.get('action_ms',0):.0f} ms</b> · with-video {lat.get('video_ms',0):.0f} ms · 去噪 {args.steps_inf} 步"
    ckname = os.path.basename(os.path.dirname(args.transformer_dir)) if args.transformer_dir else "ckpt"
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>Episode Report — {ckname}</title>{CSS}</head><body>
<h2>Episode 测试报告 — {ckname}</h2>
<p class=note>held-out:指标 {len(metric)} ep / 可视化 {len([e for e in viz_eps if e in metric])} ep · exec_horizon={args.exec_horizon} · <b>开环(非闭环 SR)</b></p>
<h3>视频说明</h3><ul>
<li>每视频 <b>2 行</b>(原分辨率,帧上有行名):<b>行1 GT(真值) / 行2 pred(模型 raw 权重预测)</b></li>
<li>每行内 3 视角横排:<b>cam_high(头) | cam_left(左腕) | cam_right(右腕)</b></li>
<li><b>A 全 episode 长视频</b>(<code>ep*_full.mp4</code>):沿 exec_horizon 取该集所有 window 拼接,时间轴=window 序列;每 window 5 帧=action chunk 的 delta[0,12,24,36,48]</li>
<li><b>B 代表窗短视频</b>(<code>ep*_w*.mp4</code>):该集 {args.n_vid_per_ep} 个代表 window,各 5 帧 1s(模型 num_frames=5 固定的稀疏长跨度关键帧)</li></ul>
<h3>推理性能</h3><p>{la}</p>
<h3>聚合指标(raw,全 {len(metric)} ep)</h3>
<table><tr><th>horizon</th><th>action MAE</th><th>π0.5 参考</th></tr>{rows_html}</table>
<h3>逐 episode(抽样可视化)</h3>{''.join(blocks)}
</body></html>"""
    open(os.path.join(args.out_dir, "report.html"), "w").write(html)
    json.dump({"n_metric_eps": len(metric), "latency": lat, "raw_mae": {h: agg[f"mae@{h}"] for h in HOR}, "pi05": PI05},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print("[aggregate] raw mae@: " + " ".join(f"@{h} {agg[f'mae@{h}']:.4f}" for h in HOR) +
          f" | act-lat {lat.get('action_ms',0):.0f}ms -> {args.out_dir}/report.html", flush=True)


if __name__ == "__main__":
    main()
