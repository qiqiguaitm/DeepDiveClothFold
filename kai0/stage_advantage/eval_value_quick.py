#!/usr/bin/env python3
"""Quick eval: run crave_poly_mono / crave_poly / AE-C on N episodes, plot value curves.
Usage (from kai0/):
  CKPT=checkpoints/ADVANTAGE_TORCH_CRAVE_POLY_MONO/crave_poly_mono/50000 \
  CONFIG=ADVANTAGE_TORCH_CRAVE_POLY_MONO \
  DATA=data/Task_A/self_built/crave_stage_poly_mono \
  .venv/bin/python stage_advantage/eval_value_quick.py --n 12 --out eval_mono_v1
"""
import argparse, json, os, sys, pickle, random
from pathlib import Path
import numpy as np, pandas as pd, cv2
import torch, safetensors.torch, dataclasses
from types import SimpleNamespace
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/vePFS/tim/workspace/lerobot")

from openpi.training import config as _config
from openpi.models_pytorch.pi0_pytorch import AdvantageEstimator
from openpi.shared import image_tools
import openpi.models.tokenizer as _tokenizer


def load_episodes(data_root: Path) -> list:
    files = sorted(data_root.glob("data/**/*.parquet"))
    eps = []
    for f in files:
        ep = int(f.stem.split("_")[1])
        n = len(pd.read_parquet(f))
        eps.append((ep, n))
    return eps


def eval_episode(ep_idx: int, data_root: Path, model, tokenizer, device,
                 frame_interval: int = 15, relative_interval: int = 50):
    chunk_size = 1000
    chunk = ep_idx // chunk_size
    pf = data_root / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(pf)
    gt_cols = [c for c in ["stage_progress_gt", "task_index"] if c in df.columns]
    gt = df[["frame_index"] + gt_cols].set_index("frame_index")

    cam_keys = ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
    vids = {}
    for cam in cam_keys:
        vp = data_root / f"videos/chunk-{chunk:03d}/{cam}/episode_{ep_idx:06d}.mp4"
        caps = cv2.VideoCapture(str(vp))
        frames, idx = [], 0
        min_fi = int(gt.index.min())
        max_fi = int(gt.index.max())
        while True:
            ret, frame = caps.read()
            if not ret or idx > max_fi:
                break
            if idx >= min_fi and (idx - min_fi) % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        caps.release()
        vids[cam] = frames

    N = len(vids[cam_keys[0]])
    if N < 2:
        return []

    def proc_img(rgb):
        t = torch.from_numpy(rgb).float() / 255.0 * 2.0 - 1.0
        t = image_tools.resize_with_pad_torch(t, 224, 224)
        return t.permute(2, 0, 1)

    tokens, token_masks = tokenizer.tokenize("Flatten and fold the cloth.", state=None)
    sampled_rel = max(1, relative_interval // frame_interval)
    max_idx = N - 1

    init_top = proc_img(vids[cam_keys[0]][0]).unsqueeze(0).to(device)
    init_left = proc_img(vids[cam_keys[1]][0]).unsqueeze(0).to(device)
    init_right = proc_img(vids[cam_keys[2]][0]).unsqueeze(0).to(device)

    all_results = []
    eff_bs = 160
    start = 0

    while start < N:
        end = min(start + eff_bs, N)
        bs = end - start
        cur_idx = list(range(start, end))
        fut_idx = [min(i + sampled_rel, max_idx) for i in cur_idx]

        def batch_imgs(cam, indices):
            return torch.stack([proc_img(vids[cam][i]) for i in indices], dim=0).to(device)

        cur_t, cur_l, cur_r = batch_imgs(cam_keys[0], cur_idx), batch_imgs(cam_keys[1], cur_idx), batch_imgs(cam_keys[2], cur_idx)
        fut_t, fut_l, fut_r = batch_imgs(cam_keys[0], fut_idx), batch_imgs(cam_keys[1], fut_idx), batch_imgs(cam_keys[2], fut_idx)

        tok_b = torch.from_numpy(np.tile(tokens[None], (bs, 1))).to(device)
        mask_b = torch.from_numpy(np.tile(token_masks[None], (bs, 1))).to(device)
        state_b = torch.zeros((bs, 32), dtype=torch.float32, device=device)

        def make_obs(base, his):
            return SimpleNamespace(state=state_b,
                images={"base_-100_rgb": his[0], "left_wrist_-100_rgb": his[1], "right_wrist_-100_rgb": his[2],
                        "base_0_rgb": base[0], "left_wrist_0_rgb": base[1], "right_wrist_0_rgb": base[2]},
                image_masks={}, tokenized_prompt=tok_b, tokenized_prompt_mask=mask_b)

        rel_obs = make_obs((fut_t, fut_l, fut_r), (cur_t, cur_l, cur_r))
        abs_obs = make_obs((cur_t, cur_l, cur_r),
                           (init_top.expand(bs, -1, -1, -1), init_left.expand(bs, -1, -1, -1), init_right.expand(bs, -1, -1, -1)))

        with torch.no_grad():
            rel_pred = model.sample_values(device, rel_obs).cpu().numpy()
            abs_pred = model.sample_values(device, abs_obs).cpu().numpy()

        min_fi_actual = int(gt.index.min())
        for j in range(bs):
            fi = cur_idx[j]; fi_fut = fut_idx[j]
            gap = fi_fut - fi
            rel_val = float(rel_pred[j, 0])
            if gap == 0: rel_val = 0.0
            elif gap != sampled_rel: rel_val = rel_val / gap * sampled_rel
            abs_val = 0.0 if fi == 0 else float(abs_pred[j, 0])

            actual_fi = fi * frame_interval + min_fi_actual
            actual_fi_fut = fi_fut * frame_interval + min_fi_actual

            stage_progress_gt = float(gt.loc[actual_fi, "stage_progress_gt"]) if actual_fi in gt.index and "stage_progress_gt" in gt.columns else float("nan")
            task_index = int(gt.loc[actual_fi, "task_index"]) if actual_fi in gt.index and "task_index" in gt.columns else -1

            all_results.append(dict(frame_idx=actual_fi, future_frame_idx=actual_fi_fut,
                                    relative_advantage=float(np.clip(rel_val, -1, 1)),
                                    absolute_value=float(np.clip(abs_val, -1, 1)),
                                    stage_progress_gt=stage_progress_gt, task_index=task_index))
        start = end

    # Compute absolute_advantage from absolute_value differences
    by_fi = {r["frame_idx"]: r for r in all_results}
    for r in all_results:
        fi, fi_fut = r["frame_idx"], r["future_frame_idx"]
        if fi == fi_fut:
            r["absolute_advantage"] = 0.0
        else:
            delta = by_fi[fi_fut]["absolute_value"] - r["absolute_value"]
            real_gap = fi_fut - fi
            r["absolute_advantage"] = float(np.clip(delta / real_gap * relative_interval if real_gap != relative_interval else delta, -1, 1))

    return all_results


def plot_curves(all_results, out_dir: Path, title_prefix: str = ""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    xs = np.linspace(0, 1, 100)

    # ---- absolute_value curves coloured by Spearman vs GT ----
    fig, ax = plt.subplots(figsize=(20, 8))
    rhos = []
    for ep, results in sorted(all_results.items()):
        if not results: continue
        fi = np.array([r["frame_idx"] for r in results])
        av = np.array([r["absolute_value"] for r in results])
        sp_gt = np.array([r["stage_progress_gt"] for r in results])
        x_norm = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
        valid = ~np.isnan(sp_gt)
        rho = float(spearmanr(av[valid], sp_gt[valid])[0]) if valid.sum() > 2 else 0.0
        rhos.append(rho)
        color = plt.cm.RdYlGn((rho + 1) / 2)
        ax.plot(x_norm, av, color=color, alpha=0.4, linewidth=0.8, label=f"ep{ep}" if len(all_results) <= 12 else "")

    curves = []
    for results in all_results.values():
        if not results: continue
        fi = np.array([r["frame_idx"] for r in results])
        av = np.array([r["absolute_value"] for r in results])
        x_norm = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
        curves.append(np.interp(xs, x_norm, av))
    if curves:
        ax.plot(xs, np.mean(curves, axis=0), color="black", linewidth=3, label="mean ± IQR")
        ax.fill_between(xs, np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0), color="gray", alpha=0.15)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(-1, 1))
    plt.colorbar(sm, ax=ax, label="Spearman(abs_val, stage_progress_gt)")
    title = f"{title_prefix} absolute_value (n={len([v for v in all_results.values() if v])}, mean Spearman={np.mean(rhos):.3f})"
    ax.set_title(title)
    ax.set_xlabel("normalized time"); ax.set_ylabel("absolute_value")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3, label="ideal ceiling")
    if len(all_results) <= 12: ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "curves_absolute_value.png", dpi=150)
    plt.close(fig)
    print(f"[plot] {out_dir / 'curves_absolute_value.png'}")

    # ---- Per-episode detail: absolute_value + stage_progress_gt side-by-side ----
    ep_list = sorted(all_results.items())
    for ep, results in ep_list[:6]:
        if not results: continue
        fi = np.array([r["frame_idx"] for r in results])
        av = np.array([r["absolute_value"] for r in results])
        sp_gt = np.array([r["stage_progress_gt"] for r in results])
        valid = ~np.isnan(sp_gt)
        rho = float(spearmanr(av[valid], sp_gt[valid])[0]) if valid.sum() > 2 else 0.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        t = (fi - fi[0]) / 30.0
        ax1.plot(t, av, linewidth=1.2, color="#2166ac")
        ax1.set_title(f"ep {ep} — absolute_value (predicted)")
        ax1.set_xlabel("time (s)"); ax1.set_ylabel("absolute_value")
        ax1.axhline(y=1.0, color="green", linestyle="--", alpha=0.3)
        ax1.set_ylim(-0.3, 1.3)

        ax2.plot(t, sp_gt, linewidth=1.2, color="#b2182b")
        ax2.set_title(f"ep {ep} — stage_progress_gt (ground truth / teacher label)")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("stage_progress_gt")
        ax2.set_ylim(-0.1, 1.1)
        fig.suptitle(f"ep {ep}  |  Spearman ρ = {rho:.3f}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / f"detail_ep{ep:06d}.png", dpi=120)
        plt.close(fig)
    print(f"[plot] {len(ep_list[:6])} per-ep detail plots")


def compute_metrics(all_results):
    from scipy.stats import spearmanr
    sp_list, disc_list, mono_list, mean_av_list = [], [], [], []
    for results in all_results.values():
        if not results: continue
        av = np.array([r["absolute_value"] for r in results])
        sp_gt = np.array([r["stage_progress_gt"] for r in results])
        ti = np.array([r["task_index"] for r in results])
        valid = ~np.isnan(sp_gt)
        if valid.sum() > 2: sp_list.append(float(spearmanr(av[valid], sp_gt[valid])[0]))
        pos = ti == 1; neg = ti == 0
        if pos.any() and neg.any(): disc_list.append(float(np.mean(av[pos]) - np.mean(av[neg])))
        if len(av) > 1: mono_list.append(float(np.mean(np.diff(av) >= 0)))
        mean_av_list.append(float(np.mean(av)))

    def _s(arr):
        return {"mean": float(np.mean(arr)) if arr else float("nan"), "std": float(np.std(arr)) if arr else float("nan")}

    return {"n_episodes": len([v for v in all_results.values() if v]),
            "spearman_vs_gt": _s(sp_list), "frame_discrimination": _s(disc_list),
            "monotonicity": _s(mono_list), "mean_absolute_value": _s(mean_av_list)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=str, default="eval_value_quick_out")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frame-interval", type=int, default=15)
    a = ap.parse_args()

    ckpt_dir = a.ckpt or os.environ.get("ADV_EST_CKPT_REL", "checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1")
    config_name = a.config or os.environ.get("ADV_EST_CONFIG", "ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD")
    data_root = Path(a.data) if a.data else Path(os.environ.get("ADV_EST_DATA", "data/Task_A/self_built/crave_stage_poly_mono"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    out_dir = ROOT / a.out; out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{a.gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cfg = _config.get_config(config_name)
    model_cfg = dataclasses.replace(cfg.model)
    model = AdvantageEstimator(model_cfg).to(device)
    model.eval()
    safetensors.torch.load_model(model, str(ROOT / ckpt_dir / "model.safetensors"), strict=True)
    tokenizer = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)
    print(f"[model] {ckpt_dir} loaded | {sum(p.numel() for p in model.parameters())/1e9:.2f}B params", flush=True)

    eps = load_episodes(data_root)
    random.seed(42); sampled = sorted(random.sample([e[0] for e in eps], min(a.n, len(eps))))
    print(f"[data] {len(eps)} ep total | sampled {len(sampled)}: {sampled}", flush=True)

    all_results = {}
    for ep in tqdm(sampled):
        try:
            res = eval_episode(ep, data_root, model, tokenizer, device, frame_interval=a.frame_interval)
            if res: all_results[ep] = res
        except Exception as e:
            print(f"  skip ep {ep}: {e}", flush=True)
        torch.cuda.empty_cache()

    with open(out_dir / "inference_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    metrics = compute_metrics(all_results)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    title = f"{config_name} ({Path(ckpt_dir).name})"
    plot_curves(all_results, out_dir, title)

    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k}: mean={v['mean']:.4f} std={v['std']:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n[done] {out_dir}")

if __name__ == "__main__":
    main()
