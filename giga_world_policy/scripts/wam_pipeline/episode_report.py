"""Episode 测试报告:全量 episode 指标 + 抽样 episode 可视化(action 曲线 + raw|ema|GT 三联视频)
+ raw/ema 对比 + 推理性能,汇总成自包含 HTML。开环逐窗口(exec_horizon 重规划节奏),非闭环 SR。
设计见对话/docs。复用 eval_watch 组件。

用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/wam_pipeline/episode_report.py \
    --transformer_dir runs/visrobot01_fold_aihc_latent_5x/models/checkpoint_epoch_1_step_18000/transformer \
    --ema_dir   runs/visrobot01_fold_aihc_latent_5x/models/checkpoint_epoch_1_step_18000/transformer_ema \
    --model_id "$WAN_DIFFUSERS" --stats_path assets_visrobot01/norm_stats_vis.json \
    --val_root "$GWP_DATA/visrobot01_val" --t5_pkl "$GWP_DATA/visrobot01_val/t5_embedding/episode_000000.pt" \
    --n_metric_eps 200 --n_viz_eps 20 --n_vid_per_ep 3 --n_ema_eps 8 --out_dir runs/visrobot01_fold_aihc_latent_5x/report_step18000
"""
import argparse, os, sys, glob, json, time, base64, io, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from wam_pipeline.eval_watch import build_window_indices, EpisodeFrameCache, _hwc_to_chw01, _to_thwc_gpu, video_metrics_gpu, _save_side_by_side
VK = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
DIM = [f"L_j{i}" for i in range(6)] + ["L_grip"] + [f"R_j{i}" for i in range(6)] + ["R_grip"]
STAY = {1: 0.000, 10: 0.042, 24: 0.099, 48: 0.175}; PI05 = {1: 0.0219, 10: 0.0425, 24: 0.0743, 48: 0.1155}


def _png_b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=70, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    for a in ["transformer_dir", "model_id", "stats_path", "val_root", "t5_pkl", "out_dir"]:
        ap.add_argument("--" + a, required=True)
    ap.add_argument("--ema_dir", default=None)
    ap.add_argument("--n_metric_eps", type=int, default=200); ap.add_argument("--n_viz_eps", type=int, default=20)
    ap.add_argument("--n_vid_per_ep", type=int, default=3); ap.add_argument("--n_ema_eps", type=int, default=8)
    ap.add_argument("--exec_horizon", type=int, default=16); ap.add_argument("--action_chunk", type=int, default=48)
    ap.add_argument("--steps_inf", type=int, default=10); ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--delta_mask", default="1,1,1,1,1,1,0,1,1,1,1,1,1,0")
    ap.add_argument("--width", type=int, default=768); ap.add_argument("--height", type=int, default=192)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); dev, dt = "cuda", torch.bfloat16
    from giga_datasets import load_dataset
    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
    from world_action_model.pipeline.wa_pipeline import WAPipeline
    from world_action_model.pipeline.utils import (extract_normalization_tensors, load_stats,
        load_t5_embedding_from_pkl, denormalize_action, add_state_to_action, normalize_state, build_ref_image)
    from diffusers.models import AutoencoderKLWan
    stats = load_stats(args.stats_path); norm = extract_normalization_tensors(stats, device=dev, state_dim=14, action_dim=14)
    t5 = load_t5_embedding_from_pkl(args.t5_pkl, target_len=64).to(dev, torch.float32)
    dm = torch.tensor([c == "1" for c in args.delta_mask.split(",")], device=dev, dtype=torch.bool)
    ve = dict(_class_name="LeRobotDataset", data_path=args.val_root, delta_info={"action": args.action_chunk},
              skip_video_decoding=True, embodiment="visrobot01", tolerance_s=1e-3)
    ds = load_dataset([ve]); idx, _, info = build_window_indices(args.val_root, "exec", args.exec_horizon, args.action_chunk, args.exec_horizon)
    fc = EpisodeFrameCache(args.val_root, VK, 4)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dt)
    # episode -> 该集窗口(全局idx) 列表(按帧序)
    from collections import OrderedDict
    ep2win = OrderedDict()
    for gi in idx:
        ep2win.setdefault(info[gi][0], []).append(gi)
    eps = list(ep2win.keys())
    metric_eps = eps[:args.n_metric_eps]
    viz_eps = [eps[i] for i in np.unique(np.linspace(0, len(metric_eps) - 1, min(args.n_viz_eps, len(metric_eps))).astype(int))]
    ema_eps = set(viz_eps[:args.n_ema_eps])
    HOR = sorted({h for h in (1, 10, args.action_chunk // 2, args.action_chunk) if h <= args.action_chunk})

    def load_pipe(sub):
        tf = CasualWorldActionTransformer.from_pretrained(sub).to(dt)
        return WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=tf, torch_dtype=dt).to(dev), tf

    def infer(pipe, gi, want_video):
        d = ds[int(gi)]; ep, f = info[int(gi)]; fr = fc.get(ep)
        ref = build_ref_image(images={k: _hwc_to_chw01(fr[k][f]) for k in VK}, dst_size=(args.width, args.height), crop_mode="center")
        st = d["observation.state"].float().unsqueeze(0).to(dev); ns = normalize_state(st, norm, mode="zscore").to(dev, dt)
        with torch.no_grad():
            out = pipe(height=args.height, width=args.width, action_chunk=args.action_chunk, state=ns, num_frames=5,
                       guidance_scale=0.0, num_inference_steps=args.steps_inf, image=ref, action_only=not want_video,
                       return_dict=False, prompt_embeds=t5.unsqueeze(0).to(dev, torch.float32))
        imgs, act = out[0], out[1]   # pipe 始终返回 (imgs_or_None, action);action_only 时 imgs=None
        pa = add_state_to_action(denormalize_action(act[0].float(), norm, mode="zscore"), st[0].float().to(act.device),
                                 action_chunk=args.action_chunk, mask=dm).cpu().numpy()
        gt = d["action"].float().numpy()[:, :14]
        pv = _to_thwc_gpu(imgs[0], dev) if want_video else None
        # GT 视频(window 内 5 帧, 与 pred 同空间)
        gv = None
        if want_video:
            offs = [0, args.action_chunk // 4, args.action_chunk // 2, 3 * args.action_chunk // 4, args.action_chunk]
            Lf = fr[VK[0]].shape[0]
            gv = torch.from_numpy(np.stack([np.array(build_ref_image(
                images={k: _hwc_to_chw01(fr[k][min(f + o, Lf - 1)]) for k in VK},
                dst_size=(args.width, args.height), crop_mode="center")) for o in offs])).to(dev, torch.float32)
        return ep, f, pa, gt, pv, gv

    def metrics(pa, gt):
        L = min(len(pa), len(gt)); ae = np.abs(pa[:L] - gt[:L]); m = {}
        for h in HOR:
            if h <= L: m[f"mae@{h}"] = float(ae[h - 1].mean())
        m["action_mae"] = float(ae.mean())
        return m

    # ---------- RAW: 全量 metric eps ----------
    pipe, tf = load_pipe(args.transformer_dir)
    # 性能:action-only / with-video latency
    g0 = list(ep2win[metric_eps[0]])[0]
    for _ in range(2): infer(pipe, g0, False)  # warmup
    torch.cuda.synchronize(); t = time.time(); [infer(pipe, g0, False) for _ in range(5)]; torch.cuda.synchronize()
    lat_act = (time.time() - t) / 5 * 1000
    torch.cuda.synchronize(); t = time.time(); [infer(pipe, g0, True) for _ in range(3)]; torch.cuda.synchronize()
    lat_vid = (time.time() - t) / 3 * 1000

    ep_rows = []; agg = {f"mae@{h}": [] for h in HOR}; agg["action_mae"] = []
    for k, ep in enumerate(metric_eps):
        wins = ep2win[ep]; em = {f"mae@{h}": [] for h in HOR}; em["action_mae"] = []
        want_v = ep in viz_eps; traj_p = {}; traj_g = {}; vids = []
        vid_wins = set(wins[:: max(1, len(wins) // args.n_vid_per_ep)][:args.n_vid_per_ep]) if want_v else set()
        for gi in wins:
            ep_, f, pa, gt, pv, gv = infer(pipe, gi, want_v and gi in vid_wins)
            for kk, v in metrics(pa, gt).items(): em[kk].append(v)
            if want_v:
                h = args.exec_horizon
                traj_p[f] = pa[:h]; traj_g[f] = gt[:h]
                if gi in vid_wins and pv is not None:
                    vids.append((f, pv.round().clamp(0, 255).byte().cpu().numpy(), gv.round().clamp(0, 255).byte().cpu().numpy()))
        row = {"ep": int(ep), "n_win": len(wins)}
        for kk in em: row[kk] = float(np.mean(em[kk])); agg[kk].append(row[kk])
        if want_v: row["_traj"] = (traj_p, traj_g); row["_vids"] = vids
        ep_rows.append(row)
        if k % 20 == 0: print(f"[raw] ep {k+1}/{len(metric_eps)}", flush=True)
    del pipe, tf; torch.cuda.empty_cache()

    # ---------- EMA: 抽样 eps ----------
    ema_rows = {}
    if args.ema_dir and os.path.isdir(args.ema_dir):
        pipe, tf = load_pipe(args.ema_dir)
        for ep in ema_eps:
            em = {f"mae@{h}": [] for h in HOR}; em["action_mae"] = []
            for gi in ep2win[ep]:
                _, f, pa, gt, _, _ = infer(pipe, gi, False)
                for kk, v in metrics(pa, gt).items(): em[kk].append(v)
            ema_rows[int(ep)] = {kk: float(np.mean(em[kk])) for kk in em}
        del pipe, tf; torch.cuda.empty_cache()

    # ---------- 写可视化 + HTML ----------
    os.makedirs(os.path.join(args.out_dir, "episodes"), exist_ok=True)
    blocks = []
    for row in ep_rows:
        if "_traj" not in row: continue
        ep = row["ep"]; tp, tg = row["_traj"]; fs = sorted(tp.keys())
        P = np.concatenate([tp[f] for f in fs]); G = np.concatenate([tg[f] for f in fs]); x = np.arange(len(P))
        fig, axes = plt.subplots(7, 2, figsize=(13, 15)); axes = axes.flatten()
        for dd in range(14):
            ax = axes[dd]; ax.plot(x, G[:, dd], "k--", lw=1.5, label="GT"); ax.plot(x, P[:, dd], "r-", lw=1.2, label="pred(raw)")
            ax.set_title(DIM[dd], fontsize=8)
            if dd == 0: ax.legend(fontsize=7)
        fig.suptitle(f"episode {ep} — deploy-style action traj (exec_h={args.exec_horizon})")
        img = _png_b64(fig)
        vtags = ""
        for (f, pv, gv) in row.get("_vids", []):
            p = os.path.join(args.out_dir, "episodes", f"ep{ep}_w{f}.mp4")
            _save_side_by_side(pv, gv, p, fps=args.fps)
            vtags += f'<video src="episodes/ep{ep}_w{f}.mp4" controls width="500"></video> '
        em = ema_rows.get(ep, {})
        emt = f" | ema@48 {em.get('mae@48', float('nan')):.4f}" if em else ""
        blocks.append(f"<details><summary><b>ep {ep}</b> (n_win={row['n_win']}) mae@1 {row['mae@1']:.4f} mae@48 {row['mae@48']:.4f}{emt}</summary>"
                      f'<img src="{img}" width="900"><br>{vtags}</details>')

    def avg(k): return float(np.mean(agg[k]))
    rows = "".join(f"<tr><td>{h}</td><td>{avg(f'mae@{h}'):.4f}</td><td>{STAY[h]:.3f}</td><td>{PI05[h]:.4f}</td>"
                   f"<td>{'✅' if avg(f'mae@{h}')<STAY[h] else '❌'}</td></tr>" for h in HOR)
    ema_cmp = ""
    if ema_rows:
        common = [e for e in ema_rows]
        r48 = np.mean([next(x['mae@48'] for x in ep_rows if x['ep'] == e) for e in common])
        e48 = np.mean([ema_rows[e]['mae@48'] for e in common])
        ema_cmp = f"<p>raw vs ema (抽样 {len(common)} ep): raw mae@48 {r48:.4f} vs ema mae@48 {e48:.4f} (Δ {abs(r48-e48):.4f})</p>"
    html = f"""<html><head><meta charset=utf-8><title>Episode Report</title></head><body style="font-family:monospace">
<h2>Episode 测试报告 — {os.path.basename(os.path.dirname(args.transformer_dir))}</h2>
<p>held-out episodes: 指标 {len(metric_eps)} / 可视化 {len(viz_eps)} / ema 抽样 {len(ema_rows)} · exec_horizon={args.exec_horizon} · 开环(非闭环SR)</p>
<h3>性能</h3><p>action-only latency: <b>{lat_act:.0f} ms</b> · with-video latency: {lat_vid:.0f} ms · 去噪步数 {args.steps_inf}</p>
<h3>聚合指标(raw, 全{len(metric_eps)} ep)</h3>
<table border=1 cellpadding=4><tr><th>horizon</th><th>raw MAE</th><th>stay-put</th><th>π0.5</th><th>优于stay?</th></tr>{rows}</table>
{ema_cmp}
<h3>逐 episode(抽样可视化)</h3>{''.join(blocks)}
</body></html>"""
    open(os.path.join(args.out_dir, "report.html"), "w").write(html)
    summ = {"transformer_dir": args.transformer_dir, "n_metric_eps": len(metric_eps),
            "latency_action_ms": lat_act, "latency_video_ms": lat_vid,
            "raw_mae": {h: avg(f"mae@{h}") for h in HOR}, "stay_put": STAY, "pi05": PI05,
            "ema_sample": ema_rows}
    json.dump(summ, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(f"[report] wrote {args.out_dir}/report.html  + summary.json")
    print(f"[report] raw mae@: " + " ".join(f"@{h} {avg(f'mae@{h}'):.4f}" for h in HOR) + f" | act-lat {lat_act:.0f}ms")


if __name__ == "__main__":
    main()
