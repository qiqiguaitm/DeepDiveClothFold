"""Episode 测试报告(分片版):全量 episode 指标 + 抽样 episode 可视化(action 曲线 + raw|GT 视频)
+ raw/ema 对比 + 推理性能,汇总 HTML。开环逐窗口(exec_horizon 重规划),非闭环 SR。

分布式:每 shard 处理 metric_eps[shard_id::num_shards],出 shards/shard_<id>.json + 自己 viz ep 的
图/视频(写共享 episodes/);--aggregate 读所有 shard 合并写 report.html。orchestrator: run_report_dist.sh。

单 ckpt 用法(worker):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/wam_pipeline/episode_report.py \
    --transformer_dir .../transformer --ema_dir .../transformer_ema --model_id "$WAN_DIFFUSERS" \
    --stats_path ... --val_root ... --t5_pkl ... --out_dir ... --shard_id 0 --num_shards 16
聚合:  python ... episode_report.py --out_dir ... --aggregate  (+ 同样的 val_root/transformer_dir 等用于一致划分)
"""
import argparse, os, sys, glob, json, time, base64, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import OrderedDict
from wam_pipeline.eval_watch import build_window_indices, EpisodeFrameCache, _hwc_to_chw01, _to_thwc_gpu, _save_side_by_side
VK = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
DIM = [f"L_j{i}" for i in range(6)] + ["L_grip"] + [f"R_j{i}" for i in range(6)] + ["R_grip"]
STAY = {1: 0.000, 10: 0.042, 24: 0.099, 48: 0.175}; PI05 = {1: 0.0219, 10: 0.0425, 24: 0.0743, 48: 0.1155}


def get_args():
    ap = argparse.ArgumentParser()
    for a in ["val_root", "out_dir"]:
        ap.add_argument("--" + a, required=True)
    for a in ["transformer_dir", "model_id", "stats_path", "t5_pkl", "ema_dir"]:
        ap.add_argument("--" + a, default=None)
    ap.add_argument("--n_metric_eps", type=int, default=200); ap.add_argument("--n_viz_eps", type=int, default=20)
    ap.add_argument("--n_vid_per_ep", type=int, default=3); ap.add_argument("--n_ema_eps", type=int, default=8)
    ap.add_argument("--max_win_per_ep", type=int, default=6)
    ap.add_argument("--shard_id", type=int, default=0); ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--exec_horizon", type=int, default=16); ap.add_argument("--action_chunk", type=int, default=48)
    ap.add_argument("--steps_inf", type=int, default=10); ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--delta_mask", default="1,1,1,1,1,1,0,1,1,1,1,1,1,0")
    ap.add_argument("--width", type=int, default=768); ap.add_argument("--height", type=int, default=192)
    return ap.parse_args()


def plan(args):
    """确定性 episode 划分(所有 shard + aggregate 必须一致)。"""
    idx, _, info = build_window_indices(args.val_root, "exec", args.exec_horizon, args.action_chunk, args.exec_horizon)
    ep2win = OrderedDict()
    for gi in idx:
        ep2win.setdefault(info[gi][0], []).append(gi)
    eps = list(ep2win.keys()); metric = eps[:args.n_metric_eps]
    viz = [metric[i] for i in np.unique(np.linspace(0, len(metric) - 1, min(args.n_viz_eps, len(metric))).astype(int))]
    ema = viz[:args.n_ema_eps]
    return info, ep2win, metric, viz, ema


def main():
    args = get_args()
    HOR = sorted({h for h in (1, 10, args.action_chunk // 2, args.action_chunk) if h <= args.action_chunk})
    info, ep2win, metric_eps, viz_eps, ema_eps = plan(args)
    if args.aggregate:
        return aggregate(args, metric_eps, viz_eps, ema_eps, HOR)
    # ---------------- worker ----------------
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
    dm = torch.tensor([c == "1" for c in args.delta_mask.split(",")], device=dev, dtype=torch.bool)
    ve = dict(_class_name="LeRobotDataset", data_path=args.val_root, delta_info={"action": args.action_chunk},
              skip_video_decoding=True, embodiment="visrobot01", tolerance_s=1e-3)
    ds = load_dataset([ve]); fc = EpisodeFrameCache(args.val_root, VK, 4)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dt)
    sid, n = args.shard_id, args.num_shards
    my_metric = metric_eps[sid::n]; vset = set(viz_eps); eset = set(ema_eps); mset = set(my_metric)
    print(f"[report shard {sid}/{n}] metric {len(my_metric)}ep viz {len(vset&mset)} ema {len(eset&mset)}", flush=True)

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
        imgs, act = out[0], out[1]
        pa = add_state_to_action(denormalize_action(act[0].float(), norm, mode="zscore"), st[0].float().to(act.device),
                                 action_chunk=args.action_chunk, mask=dm).cpu().numpy()
        gt = d["action"].float().numpy()[:, :14]
        pv = _to_thwc_gpu(imgs[0], dev).round().clamp(0, 255).byte().cpu().numpy() if want_video else None
        gv = None
        if want_video:
            offs = [0, args.action_chunk // 4, args.action_chunk // 2, 3 * args.action_chunk // 4, args.action_chunk]
            Lf = fr[VK[0]].shape[0]
            gv = np.stack([np.array(build_ref_image(images={k: _hwc_to_chw01(fr[k][min(f + o, Lf - 1)]) for k in VK},
                          dst_size=(args.width, args.height), crop_mode="center")) for o in offs])
        return ep, f, pa, gt, pv, gv

    def metrics(pa, gt):
        L = min(len(pa), len(gt)); ae = np.abs(pa[:L] - gt[:L]); m = {"action_mae": float(ae.mean())}
        for h in HOR:
            if h <= L: m[f"mae@{h}"] = float(ae[h - 1].mean())
        return m

    pipe, tf = load_pipe(args.transformer_dir)
    latency = {}
    if sid == 0:
        g0 = my_metric and ep2win[my_metric[0]][0]
        for _ in range(2): infer(pipe, g0, False)
        torch.cuda.synchronize(); t = time.time(); [infer(pipe, g0, False) for _ in range(5)]; torch.cuda.synchronize()
        latency["action_ms"] = (time.time() - t) / 5 * 1000
        torch.cuda.synchronize(); t = time.time(); [infer(pipe, g0, True) for _ in range(3)]; torch.cuda.synchronize()
        latency["video_ms"] = (time.time() - t) / 3 * 1000

    rows = {}
    for k, ep in enumerate(my_metric):
        wins = ep2win[ep]; is_v = ep in vset
        if not is_v and len(wins) > args.max_win_per_ep:
            wins = [wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, args.max_win_per_ep).astype(int))]
        em = {kk: [] for kk in ["action_mae"] + [f"mae@{h}" for h in HOR]}
        tp, tg = {}, {}; vid_wins = set(wins[:: max(1, len(wins) // args.n_vid_per_ep)][:args.n_vid_per_ep]) if is_v else set()
        vids = []
        for gi in wins:
            _, f, pa, gt, pv, gv = infer(pipe, gi, is_v and gi in vid_wins)
            for kk, v in metrics(pa, gt).items(): em[kk].append(v)
            if is_v:
                h = args.exec_horizon; tp[f] = pa[:h].tolist(); tg[f] = gt[:h].tolist()
                if pv is not None:
                    vp = f"episodes/ep{ep}_w{f}.mp4"; _save_side_by_side(pv, gv, os.path.join(args.out_dir, vp), fps=args.fps); vids.append(vp)
        row = {"ep": int(ep), "n_win": len(wins), "is_viz": is_v}
        for kk in em: row[kk] = float(np.mean(em[kk])) if em[kk] else None
        if is_v:  # 画 raw vs GT action 曲线(拼接 exec_horizon 段)存 png
            fs = sorted(tp.keys()); P = np.concatenate([np.array(tp[f]) for f in fs]); G = np.concatenate([np.array(tg[f]) for f in fs])
            x = np.arange(len(P)); fig, axes = plt.subplots(7, 2, figsize=(13, 15)); axes = axes.flatten()
            for dd in range(14):
                axes[dd].plot(x, G[:, dd], "k--", lw=1.5, label="GT"); axes[dd].plot(x, P[:, dd], "r-", lw=1.2, label="pred(raw)")
                axes[dd].set_title(DIM[dd], fontsize=8)
                if dd == 0: axes[dd].legend(fontsize=7)
            fig.suptitle(f"episode {ep} — deploy-style action traj (exec_h={args.exec_horizon})")
            tpng = f"episodes/ep{ep}_traj.png"; fig.savefig(os.path.join(args.out_dir, tpng), dpi=70, bbox_inches="tight"); plt.close(fig)
            row["traj_png"] = tpng; row["vids"] = vids
        rows[int(ep)] = row
        if k % 5 == 0: print(f"[shard {sid}] {k+1}/{len(my_metric)}", flush=True)
    del pipe, tf; torch.cuda.empty_cache()

    ema_rows = {}
    if args.ema_dir and os.path.isdir(args.ema_dir):
        pipe, tf = load_pipe(args.ema_dir)
        for ep in [e for e in my_metric if e in eset]:
            em = {kk: [] for kk in ["action_mae"] + [f"mae@{h}" for h in HOR]}
            wins = ep2win[ep]
            if len(wins) > args.max_win_per_ep:
                wins = [wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, args.max_win_per_ep).astype(int))]
            for gi in wins:
                _, _, pa, gt, _, _ = infer(pipe, gi, False)
                for kk, v in metrics(pa, gt).items(): em[kk].append(v)
            ema_rows[int(ep)] = {kk: float(np.mean(em[kk])) for kk in em if em[kk]}
        del pipe, tf; torch.cuda.empty_cache()
    json.dump({"metric": rows, "ema": ema_rows, "latency": latency},
              open(os.path.join(args.out_dir, "shards", f"shard_{sid}.json"), "w"))
    print(f"[report shard {sid}] done -> shards/shard_{sid}.json", flush=True)


def aggregate(args, metric_eps, viz_eps, ema_eps, HOR):
    sh = sorted(glob.glob(os.path.join(args.out_dir, "shards", "shard_*.json")))
    metric, ema, lat = {}, {}, {}
    for f in sh:
        d = json.load(open(f)); metric.update({int(k): v for k, v in d["metric"].items()})
        ema.update({int(k): v for k, v in d["ema"].items()}); lat.update(d.get("latency", {}))
    print(f"[aggregate] merged {len(sh)} shards, {len(metric)} episodes, {len(ema)} ema", flush=True)
    agg = {f"mae@{h}": np.mean([r[f"mae@{h}"] for r in metric.values() if r.get(f"mae@{h}") is not None]) for h in HOR}

    def b64(p):
        fp = os.path.join(args.out_dir, p)
        return "data:image/png;base64," + base64.b64encode(open(fp, "rb").read()).decode() if os.path.isfile(fp) else ""
    blocks = []
    for ep in viz_eps:
        r = metric.get(ep)
        if not r or not r.get("traj_png"): continue
        vtags = "".join(f'<video src="{v}" controls width="500"></video> ' for v in r.get("vids", []))
        emt = f" | ema@48 {ema[ep].get('mae@48', float('nan')):.4f}" if ep in ema else ""
        blocks.append(f"<details><summary><b>ep {ep}</b> (n_win={r['n_win']}) mae@1 {r['mae@1']:.4f} mae@48 {r['mae@48']:.4f}{emt}</summary>"
                      f'<img src="{b64(r["traj_png"])}" width="900"><br>{vtags}</details>')
    rows_html = "".join(f"<tr><td>{h}</td><td>{agg[f'mae@{h}']:.4f}</td><td>{STAY[h]:.3f}</td><td>{PI05[h]:.4f}</td>"
                        f"<td>{'✅' if agg[f'mae@{h}']<STAY[h] else '❌'}</td></tr>" for h in HOR)
    ema_cmp = ""
    if ema:
        common = [e for e in ema if e in metric]
        r48 = np.mean([metric[e]["mae@48"] for e in common]); e48 = np.mean([ema[e]["mae@48"] for e in common])
        ema_cmp = f"<p>raw vs ema (抽样 {len(common)} ep): raw mae@48 <b>{r48:.4f}</b> vs ema mae@48 {e48:.4f} (Δ {abs(r48-e48):.4f})</p>"
    la = f"action-only <b>{lat.get('action_ms',0):.0f} ms</b> · with-video {lat.get('video_ms',0):.0f} ms · 去噪 {args.steps_inf} 步"
    html = f"""<html><head><meta charset=utf-8><title>Episode Report</title></head><body style="font-family:monospace">
<h2>Episode 测试报告 — {os.path.basename(os.path.dirname(args.transformer_dir or 'ckpt'))}</h2>
<p>held-out: 指标 {len(metric)} ep / 可视化 {len([e for e in viz_eps if e in metric])} / ema 抽样 {len(ema)} · exec_horizon={args.exec_horizon} · 开环(非闭环SR)</p>
<h3>性能</h3><p>{la}</p>
<h3>聚合指标(raw, 全 {len(metric)} ep)</h3>
<table border=1 cellpadding=4><tr><th>horizon</th><th>raw MAE</th><th>stay-put</th><th>π0.5</th><th>优于stay?</th></tr>{rows_html}</table>
{ema_cmp}
<h3>逐 episode(抽样可视化:raw 曲线 + raw|GT 视频)</h3>{''.join(blocks)}
</body></html>"""
    open(os.path.join(args.out_dir, "report.html"), "w").write(html)
    json.dump({"n_metric_eps": len(metric), "latency": lat, "raw_mae": {h: float(agg[f"mae@{h}"]) for h in HOR},
               "stay_put": STAY, "pi05": PI05, "ema_sample": ema}, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print("[aggregate] raw mae@: " + " ".join(f"@{h} {agg[f'mae@{h}']:.4f}" for h in HOR) +
          f" | act-lat {lat.get('action_ms',0):.0f}ms -> {args.out_dir}/report.html", flush=True)


if __name__ == "__main__":
    main()
