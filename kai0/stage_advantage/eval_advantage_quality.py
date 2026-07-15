#!/usr/bin/env python3
"""Advantage quality eval: run AE checkpoint on held-out episodes, compute advantage metrics.

Measures what matters for AWBC downstream:
  1. P/N flip count per episode (old AE-C: 256 flips → target <<256)
  2. Zero-crossing rate (fraction of frames where |advantage| < eps)
  3. GT-advantage Spearman (predicted sign vs GT sign alignment)
  4. Signal-to-noise ratio
  5. Per-episode advantage curves plot

Usage (from kai0/):
  .venv/bin/python stage_advantage/eval_advantage_quality.py \
    --ckpt checkpoints/ADVANTAGE_TORCH_CRAVE_POLY_MONO/crave_poly_mono/50000 \
    --config ADVANTAGE_TORCH_CRAVE_POLY_MONO \
    --data data/Task_A/self_built/crave_stage_poly_mono \
    --n 15 --horizon 50 --out eval_adv_mono
"""
import argparse, json, os, sys, pickle, random
from pathlib import Path
import numpy as np
import torch, safetensors.torch, dataclasses
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/vePFS/tim/workspace/lerobot")

from openpi.training import config as _config
from openpi.models_pytorch.pi0_pytorch import AdvantageEstimator
from openpi.shared import image_tools
import openpi.models.tokenizer as _tokenizer


# ── helpers ────────────────────────────────────────────────
def sign(x, eps=1e-4):
    """Ternary sign: +1 / 0 / -1 with a dead zone for near-zero."""
    s = np.sign(x)
    s[np.abs(x) < eps] = 0
    return s


def pn_flips(a):
    """Count sign flips in an advantage sequence, ignoring dead-zone frames."""
    s = sign(a)
    nonzero = s != 0
    if nonzero.sum() < 2:
        return 0, len(a), 0.0
    s_nz = s[nonzero]
    flips = int((np.diff(s_nz) != 0).sum())
    neg_frac = float((s == -1).mean())
    return flips, int(nonzero.sum()), neg_frac


def zero_cross_rate(a, eps=1e-3):
    """Proportion of frames where |advantage| < eps."""
    return float((np.abs(a) < eps).mean())


def snr(a):
    """Signal-to-noise: std(advantage) / max(mean(|advantage|), 1e-9)."""
    return float(np.std(a) / max(np.mean(np.abs(a)), 1e-9))


GT_SPG_COL = "stage_progress_gt"


# ── inference ──────────────────────────────────────────────
def load_episodes(data_root: Path):
    files = sorted(data_root.glob("data/**/*.parquet"))
    return [(int(f.stem.split("_")[1]), len(pd.read_parquet(f))) for f in files]


def eval_episode_adv(ep_idx: int, data_root: Path, model, tokenizer, device,
                     frame_interval: int = 15, horizon: int = 50):
    """Run inference on one episode, return per-frame absolute_value + compute advantage."""
    import pandas as pd, cv2
    chunk_size = 1000; chunk = ep_idx // chunk_size
    pf = data_root / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(pf)
    gt_cols = [c for c in [GT_SPG_COL, "task_index"] if c in df.columns]
    gt = df[["frame_index"] + gt_cols].set_index("frame_index")
    state_cols = [c for c in df.columns if "observation.state" in c or c == "state"]
    if state_cols:
        state_arr = np.stack(df[state_cols[0]].to_numpy()).astype(np.float32)
    else:
        state_arr = np.zeros((len(df), 14), dtype=np.float32)

    cam_keys = ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
    vids = {}
    for cam in cam_keys:
        vp = data_root / f"videos/chunk-{chunk:03d}/{cam}/episode_{ep_idx:06d}.mp4"
        caps = cv2.VideoCapture(str(vp))
        frames, idx = [], 0
        min_fi, max_fi = int(gt.index.min()), int(gt.index.max())
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
    h_step = max(1, horizon // frame_interval)

    def proc_img(rgb):
        t = torch.from_numpy(rgb).float() / 255.0 * 2.0 - 1.0
        t = image_tools.resize_with_pad_torch(t, 224, 224)
        return t.permute(2, 0, 1)

    tokens, token_masks = tokenizer.tokenize("Flatten and fold the cloth.", state=None)
    init_top = proc_img(vids[cam_keys[0]][0]).unsqueeze(0).to(device)
    init_left = proc_img(vids[cam_keys[1]][0]).unsqueeze(0).to(device)
    init_right = proc_img(vids[cam_keys[2]][0]).unsqueeze(0).to(device)

    all_results = []; eff_bs, start = 160, 0; max_idx = N - 1
    while start < N:
        end = min(start + eff_bs, N); bs = end - start
        cur_idx = list(range(start, end))

        def batch_imgs(cam, indices):
            return torch.stack([proc_img(vids[cam][i]) for i in indices], dim=0).to(device)

        cur_t, cur_l, cur_r = batch_imgs(cam_keys[0], cur_idx), batch_imgs(cam_keys[1], cur_idx), batch_imgs(cam_keys[2], cur_idx)
        tok_b = torch.from_numpy(np.tile(tokens[None], (bs, 1))).to(device)
        mask_b = torch.from_numpy(np.tile(token_masks[None], (bs, 1))).to(device)
        state_b = torch.zeros((bs, 32), dtype=torch.float32, device=device)

        abs_obs = SimpleNamespace(state=state_b,
            images={"base_-100_rgb": init_top.expand(bs, -1, -1, -1),
                    "left_wrist_-100_rgb": init_left.expand(bs, -1, -1, -1),
                    "right_wrist_-100_rgb": init_right.expand(bs, -1, -1, -1),
                    "base_0_rgb": cur_t, "left_wrist_0_rgb": cur_l, "right_wrist_0_rgb": cur_r},
            image_masks={}, tokenized_prompt=tok_b, tokenized_prompt_mask=mask_b)

        with torch.no_grad():
            abs_pred = model.sample_values(device, abs_obs).cpu().numpy()

        min_fi_actual = int(gt.index.min())
        for j in range(bs):
            fi = cur_idx[j]
            abs_val = 0.0 if fi == 0 else float(abs_pred[j, 0])
            actual_fi = fi * frame_interval + min_fi_actual
            spg = float(gt.loc[actual_fi, GT_SPG_COL]) if actual_fi in gt.index and GT_SPG_COL in gt.columns else float("nan")
            all_results.append(dict(frame_idx=actual_fi, absolute_value=float(np.clip(abs_val, -1, 1)),
                                    stage_progress_gt=spg))
        start = end

    # Compute advantage = g(s_{t+horizon}) - g(s_t)  from absolute_value
    by_fi = {r["frame_idx"]: r for r in all_results}
    fi_list = sorted(by_fi.keys())
    fi_step = frame_interval
    for r in all_results:
        fi, fi_fut = r["frame_idx"], r["frame_idx"] + horizon
        if fi_fut in by_fi:
            r["pred_advantage"] = float(np.clip(by_fi[fi_fut]["absolute_value"] - r["absolute_value"], -1, 1))
        else:
            # Use nearest neighbour at tail
            max_fi = max(fi_list)
            alt = min(fi_fut, max_fi)
            r["pred_advantage"] = float(np.clip(by_fi[alt]["absolute_value"] - r["absolute_value"], -1, 1))
        # GT advantage from polyline
        spg_i = r.get("stage_progress_gt", np.nan)
        spg_f = by_fi.get(fi_fut, {}).get("stage_progress_gt", np.nan)
        r["gt_advantage"] = float(spg_f - spg_i) if not np.isnan(spg_i) and not np.isnan(spg_f) else float("nan")

    return all_results


# ── metrics ────────────────────────────────────────────────
def compute_adv_metrics(all_results):
    from scipy.stats import spearmanr
    rows = []
    for ep, results in all_results.items():
        if not results: continue
        pa = np.array([r["pred_advantage"] for r in results])
        ga = np.array([r["gt_advantage"] for r in results])
        av = np.array([r["absolute_value"] for r in results])
        valid = ~np.isnan(ga)
        if valid.sum() < 20: continue
        flips, nz, neg_frac = pn_flips(pa[valid])
        zcr = zero_cross_rate(pa[valid])
        s = snr(pa[valid])
        # GT sign alignment: does pred sign match GT sign where GT is decisive?
        gt_sure = np.abs(ga[valid]) > 0.01
        if gt_sure.sum() > 5:
            sign_acc = float((sign(pa[valid][gt_sure]) == sign(ga[valid][gt_sure])).mean())
        else:
            sign_acc = float("nan")
        rho = float(spearmanr(pa[valid], ga[valid])[0]) if valid.sum() > 5 else float("nan")
        mean_av = float(np.mean(av))
        rows.append(dict(ep=ep, n_frames=int(valid.sum()),
                         pn_flips=flips, neg_frac=neg_frac, zero_cross_rate=zcr,
                         snr=s, sign_acc=sign_acc, spearman_adv=rho, mean_abs_value=mean_av))
    return rows


# ── plot ───────────────────────────────────────────────────
def plot_adv_curves(all_results, rows, out_dir: Path, title_prefix: str = ""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Advantage overlay: GT (dashed red) vs pred (solid blue) per ep
    ep_list = sorted(all_results.items())
    n_cols = 3; n_rows = (len(ep_list) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 3.5))
    axes = np.atleast_1d(axes).flatten()
    for ax, (ep, results) in zip(axes, ep_list):
        if not results: continue
        pa = np.array([r["pred_advantage"] for r in results])
        ga = np.array([r["gt_advantage"] for r in results])
        fi = np.array([r["frame_idx"] for r in results])
        t = (fi - fi[0]) / 30.0
        ax.plot(t, ga, color="#b2182b", lw=1.0, alpha=0.6, label="GT adv (polyline)")
        ax.plot(t, pa, color="#2166ac", lw=1.4, label="pred adv")
        flips, _, _ = pn_flips(pa); zcr = zero_cross_rate(pa)
        ax.set_title(f"ep{ep}  flips={flips} zcr={zcr:.2f}", fontsize=9)
        ax.axhline(y=0, color="grey", lw=0.4, alpha=0.5); ax.grid(alpha=0.2)
        ax.set_ylim(-0.6, 0.6)
        if ax == axes[0]: ax.legend(fontsize=7)
    for ax in axes[len(ep_list):]: ax.set_visible(False)
    fig.suptitle(f"{title_prefix} — predicted advantage vs GT advantage (red=GT, blue=pred)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "advantage_curves.png", dpi=130)
    plt.close(fig)
    print(f"[plot] {out_dir / 'advantage_curves.png'}")

    # 2) Summary scatter: flips vs sign_acc per ep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    flips_list, sign_acc_list, zcr_list, snr_list = [], [], [], []
    for r in rows:
        flips_list.append(r["pn_flips"]); sign_acc_list.append(r.get("sign_acc", np.nan) or 0)
        zcr_list.append(r["zero_cross_rate"]); snr_list.append(r["snr"])
    ax1.scatter(flips_list, sign_acc_list, c=snr_list, cmap="viridis", s=40, alpha=0.8, edgecolors="grey", linewidth=0.3)
    ax1.set_xlabel("P/N flips per episode"); ax1.set_ylabel("sign accuracy vs GT")
    ax1.set_title(f"per-ep advantage quality (n={len(rows)} ep, color=SNR)"); ax1.grid(alpha=0.2)
    ax2.hist(flips_list, bins=20, color="#2166ac", alpha=0.7, edgecolor="white")
    ax2.axvline(x=256, color="red", linestyle="--", lw=1.5, label="old AE-C (256 flips)")
    ax2.set_xlabel("P/N flips per episode"); ax2.set_ylabel("count")
    ax2.set_title("flip count distribution"); ax2.legend(); ax2.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "advantage_summary.png", dpi=130)
    plt.close(fig)
    print(f"[plot] {out_dir / 'advantage_summary.png'}")

    # 3) absolute_value overlay (same as before, sanity check)
    fig, ax = plt.subplots(figsize=(20, 8))
    from scipy.stats import spearmanr
    xs = np.linspace(0, 1, 100); curves = []
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
        ax.plot(x_norm, av, color=color, alpha=0.4, lw=0.8)
        curves.append(np.interp(xs, x_norm, av))
    if curves:
        ax.plot(xs, np.mean(curves, axis=0), color="black", lw=3, label="mean ± IQR")
        ax.fill_between(xs, np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0), color="gray", alpha=0.15)
    ax.set_title(f"{title_prefix} absolute_value (n={len(rhos)} ep, mean Spearman={np.mean(rhos):.3f})")
    ax.set_xlabel("normalized time"); ax.set_ylabel("absolute_value"); ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(-1, 1))
    plt.colorbar(sm, ax=ax, label="Spearman(abs_val, stage_progress_gt)")
    fig.tight_layout(); fig.savefig(out_dir / "curves_absolute_value.png", dpi=130); plt.close(fig)


# ── main ───────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", type=str, default="eval_adv_quality")
    ap.add_argument("--ckpt", type=str, default="checkpoints/ADVANTAGE_TORCH_CRAVE_POLY_MONO/crave_poly_mono/50000")
    ap.add_argument("--config", type=str, default="ADVANTAGE_TORCH_CRAVE_POLY_MONO")
    ap.add_argument("--data", type=str, default="data/Task_A/self_built/crave_stage_poly_mono")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=50)
    a = ap.parse_args()

    ckpt_dir = Path(a.ckpt) if a.ckpt.startswith("/") else ROOT / a.ckpt
    data_root = Path(a.data) if a.data.startswith("/") else ROOT / a.data
    out_dir = ROOT / a.out; out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{a.gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cfg = _config.get_config(a.config)
    model_cfg = dataclasses.replace(cfg.model)
    model = AdvantageEstimator(model_cfg).to(device)
    model.eval()
    st_path = ckpt_dir / "model.safetensors"
    if st_path.exists():
        safetensors.torch.load_model(model, str(st_path), strict=True)
    else:
        # Try params dir (JAX ckpt)
        raise FileNotFoundError(f"no model.safetensors at {st_path}")
    tokenizer = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)
    print(f"[model] {ckpt_dir} loaded | {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    eps = load_episodes(data_root)
    random.seed(42); sampled = sorted(random.sample([e[0] for e in eps], min(a.n, len(eps))))
    print(f"[data] {len(eps)} ep total | sampled {len(sampled)}: {sampled}", flush=True)

    all_results = {}
    from tqdm import tqdm
    for ep in tqdm(sampled):
        try:
            res = eval_episode_adv(ep, data_root, model, tokenizer, device, horizon=a.horizon)
            if res: all_results[ep] = res
        except Exception as e:
            print(f"  skip ep {ep}: {e}", flush=True)
        torch.cuda.empty_cache()

    # Save raw results
    with open(out_dir / "inference_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    rows = compute_adv_metrics(all_results)

    # Aggregate
    flips_all = [r["pn_flips"] for r in rows]
    sign_acc_all = [r.get("sign_acc", 0) or 0 for r in rows if not np.isnan(r.get("sign_acc", np.nan))]
    zcr_all = [r["zero_cross_rate"] for r in rows]
    sp_all = [r["spearman_adv"] for r in rows if not np.isnan(r.get("spearman_adv", np.nan))]

    metrics = {
        "n_episodes_evaluated": len(rows),
        "pn_flips_per_ep": {"mean": float(np.mean(flips_all)), "median": float(np.median(flips_all)),
                            "min": int(np.min(flips_all)), "max": int(np.max(flips_all))},
        "sign_accuracy_vs_gt": {"mean": float(np.mean(sign_acc_all)), "std": float(np.std(sign_acc_all))},
        "zero_cross_rate": {"mean": float(np.mean(zcr_all)), "std": float(np.std(zcr_all))},
        "spearman_adv_vs_gt": {"mean": float(np.mean(sp_all)), "std": float(np.std(sp_all))} if sp_all else {},
        "comparison_old_ae_c_256_flips": f"mean flips = {np.mean(flips_all):.1f} vs AE-C 256",
    }
    with open(out_dir / "advantage_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    title = f"{a.config} (step={ckpt_dir.name})"
    plot_adv_curves(all_results, rows, out_dir, title)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  n_episodes: {len(rows)}")
    print(f"  P/N flips/ep:  mean={np.mean(flips_all):.1f}  median={np.median(flips_all):.0f}  min={np.min(flips_all)}  max={np.max(flips_all)}")
    print(f"  Sign acc vs GT: mean={np.mean(sign_acc_all):.3f}")
    print(f"  Zero-cross rate: mean={np.mean(zcr_all):.3f}")
    if sp_all: print(f"  Spearman adv:   mean={np.mean(sp_all):.3f}  std={np.std(sp_all):.3f}")
    print(f"  ══ vs old AE-C (256 flips): {np.mean(flips_all):.0f}x better ══")
    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
