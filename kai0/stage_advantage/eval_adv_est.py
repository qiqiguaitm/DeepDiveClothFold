"""
Evaluate adv_est_v1 on N randomly sampled episodes using 8 GPUs.

Correct evaluation logic:
  - task_index is a PER-FRAME label (not per-episode).
  - Every episode contains both positive (task_index=1) and negative
    (task_index=0) frames. There is no meaningful "positive episode" concept.
  - Quality metrics measured:
      1. Spearman(absolute_value, stage_progress_gt) per episode
         → does the model track ground-truth task progress?
      2. Frame discrimination Δ = mean(abs_val | task_index=1)
                                 - mean(abs_val | task_index=0) per episode
         → does the model rank high-quality frames above low-quality ones?
      3. Monotonicity of absolute_value curve

Each GPU independently loads the model and processes its episode shard.
Results are merged, metrics computed, and plots saved.

Usage (from kai0/):
    uv run python stage_advantage/eval_adv_est.py [--n 200] [--steps 100000] [--out eval_adv_est_out]
    uv run python stage_advantage/eval_adv_est.py --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import dataclasses
import json
import os
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/vePFS/tim/workspace/lerobot")

CKPT_BASE = ROOT / "checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1"
CONFIG_NAME = "ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD"
CHUNKS_SIZE = 1000
FRAME_INTERVAL = 10
RELATIVE_INTERVAL = 50

# Set at runtime from --data argument
DATA_ROOT: Path = None  # type: ignore


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def parquet_path(ep_idx: int) -> Path:
    chunk = ep_idx // CHUNKS_SIZE
    return DATA_ROOT / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"


def video_paths(ep_idx: int) -> tuple:
    chunk = ep_idx // CHUNKS_SIZE
    base = DATA_ROOT / f"videos/chunk-{chunk:03d}"
    return (
        base / f"observation.images.top_head/episode_{ep_idx:06d}.mp4",
        base / f"observation.images.hand_left/episode_{ep_idx:06d}.mp4",
        base / f"observation.images.hand_right/episode_{ep_idx:06d}.mp4",
    )


def load_episode_meta() -> pd.DataFrame:
    """Load per-episode metadata. GT columns are optional."""
    files = sorted(DATA_ROOT.glob("data/**/*.parquet"))
    records = []
    for f in files:
        sample = pd.read_parquet(f, columns=["episode_index"])
        has_gt = "stage_progress_gt" in pd.read_parquet(f).columns
        if has_gt:
            df = pd.read_parquet(f, columns=["episode_index", "frame_index",
                                              "progress_gt", "stage_progress_gt"])
            ep = df.groupby("episode_index").agg(
                n_frames=("frame_index", "count"),
                max_progress=("progress_gt", "max"),
                max_stage_progress=("stage_progress_gt", "max"),
            ).reset_index()
        else:
            df = pd.read_parquet(f, columns=["episode_index", "frame_index"])
            ep = df.groupby("episode_index").agg(
                n_frames=("frame_index", "count"),
            ).reset_index()
        records.append(ep)
    return pd.concat(records).reset_index(drop=True)


def sample_episodes(meta: pd.DataFrame, n: int, seed: int = 42) -> list:
    """Sample n episodes randomly from all available episodes."""
    rng = random.Random(seed)
    all_eps = meta["episode_index"].tolist()
    return sorted(rng.sample(all_eps, min(n, len(all_eps))))


# ---------------------------------------------------------------------------
# Per-GPU worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def worker(gpu_id: int, episode_list: list, ckpt_dir: str, out_pkl: str,
           batch_size: int, frame_interval: int, data_root: str):
    """Load model on gpu_id, run inference on episode_list, save to out_pkl.

    Each result dict now includes GT columns (stage_progress_gt, task_index)
    looked up from the parquet for the sampled frame index.
    """
    global DATA_ROOT
    DATA_ROOT = Path(data_root)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, "/vePFS/tim/workspace/lerobot")

    import dataclasses
    import numpy as np
    import pandas as pd
    import pickle
    import torch
    import safetensors.torch
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import cv2

    from openpi.training import config as _config
    from openpi.models_pytorch.pi0_pytorch import AdvantageEstimator
    import openpi.models.tokenizer as _tokenizer
    from openpi.shared import image_tools

    device = torch.device("cuda:0")

    cfg = _config.get_config(CONFIG_NAME)
    model_cfg = dataclasses.replace(cfg.model)
    model = AdvantageEstimator(model_cfg).to(device)
    model.eval()

    model_file = Path(ckpt_dir) / "model.safetensors"
    safetensors.torch.load_model(model, str(model_file), strict=True)
    print(f"[GPU {gpu_id}] model loaded, processing {len(episode_list)} episodes", flush=True)

    tokenizer = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)
    pool = ThreadPoolExecutor(max_workers=8)

    def load_video(path, min_fi, max_fi, interval):
        cap = cv2.VideoCapture(str(path))
        frames, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if min_fi <= idx <= max_fi and (idx - min_fi) % interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif idx > max_fi:
                break
            idx += 1
        cap.release()
        return frames

    def proc_img(rgb):
        t = torch.from_numpy(rgb).float() / 255.0 * 2.0 - 1.0
        t = image_tools.resize_with_pad_torch(t, 224, 224)
        return t.permute(2, 0, 1)

    def to_tensor(imgs):
        futs = [pool.submit(proc_img, img) for img in imgs]
        return torch.stack([f.result() for f in futs], dim=0)

    def eval_episode(ep_idx):
        top_p, left_p, right_p = video_paths(ep_idx)
        raw = pd.read_parquet(parquet_path(ep_idx))
        gt_cols = [c for c in ["stage_progress_gt", "task_index"] if c in raw.columns]
        df = raw[["frame_index"] + gt_cols].set_index("frame_index")
        min_fi = int(df.index.min())
        max_fi = int(df.index.max())

        futs = [pool.submit(load_video, p, min_fi, max_fi, frame_interval)
                for p in (top_p, left_p, right_p)]
        top_f, left_f, right_f = [f.result() for f in futs]
        N = len(top_f)
        if N < 2:
            return []

        tokens, token_masks = tokenizer.tokenize("Flatten and fold the cloth.", state=None)
        sampled_rel = max(1, RELATIVE_INTERVAL // frame_interval)
        max_idx = N - 1

        init_top = proc_img(top_f[0]).unsqueeze(0).to(device)
        init_left = proc_img(left_f[0]).unsqueeze(0).to(device)
        init_right = proc_img(right_f[0]).unsqueeze(0).to(device)

        all_results = []
        eff_bs = batch_size
        start = 0

        while start < N:
            end = min(start + eff_bs, N)
            bs = end - start
            cur_idx = list(range(start, end))
            fut_idx = [min(i + sampled_rel, max_idx) for i in cur_idx]

            cur_top = to_tensor([top_f[i] for i in cur_idx]).to(device)
            cur_left = to_tensor([left_f[i] for i in cur_idx]).to(device)
            cur_right = to_tensor([right_f[i] for i in cur_idx]).to(device)
            fut_top = to_tensor([top_f[i] for i in fut_idx]).to(device)
            fut_left = to_tensor([left_f[i] for i in fut_idx]).to(device)
            fut_right = to_tensor([right_f[i] for i in fut_idx]).to(device)

            tok_b = torch.from_numpy(np.tile(tokens[None], (bs, 1))).to(device)
            mask_b = torch.from_numpy(np.tile(token_masks[None], (bs, 1))).to(device)
            state_b = torch.zeros((bs, 32), dtype=torch.float32, device=device)

            def make_obs(base_imgs, his_imgs):
                return SimpleNamespace(
                    state=state_b,
                    images={
                        "base_-100_rgb": his_imgs[0],
                        "left_wrist_-100_rgb": his_imgs[1],
                        "right_wrist_-100_rgb": his_imgs[2],
                        "base_0_rgb": base_imgs[0],
                        "left_wrist_0_rgb": base_imgs[1],
                        "right_wrist_0_rgb": base_imgs[2],
                    },
                    image_masks={},
                    tokenized_prompt=tok_b,
                    tokenized_prompt_mask=mask_b,
                )

            try:
                rel_obs = make_obs((fut_top, fut_left, fut_right), (cur_top, cur_left, cur_right))
                abs_obs = make_obs(
                    (cur_top, cur_left, cur_right),
                    (init_top.expand(bs, -1, -1, -1),
                     init_left.expand(bs, -1, -1, -1),
                     init_right.expand(bs, -1, -1, -1)),
                )
                with torch.no_grad():
                    rel_pred = model.sample_values(device, rel_obs).cpu().numpy()
                    abs_pred = model.sample_values(device, abs_obs).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if eff_bs <= 4:
                    raise
                eff_bs = max(4, eff_bs // 2)
                print(f"[GPU {gpu_id}] OOM, retry bs={eff_bs}", flush=True)
                continue

            for j in range(bs):
                fi = cur_idx[j]
                fi_fut = fut_idx[j]
                gap = fi_fut - fi
                rel_val = float(rel_pred[j, 0])
                if gap == 0:
                    rel_val = 0.0
                elif gap != sampled_rel:
                    rel_val = rel_val / gap * sampled_rel
                abs_val = 0.0 if fi == 0 else float(abs_pred[j, 0])

                # Actual parquet frame index
                actual_fi = fi * frame_interval + min_fi

                # Look up GT values from parquet (columns may not exist)
                if actual_fi in df.index:
                    gt_row = df.loc[actual_fi]
                    stage_progress_gt = float(gt_row["stage_progress_gt"]) \
                        if "stage_progress_gt" in df.columns else float("nan")
                    task_index = int(gt_row["task_index"]) \
                        if "task_index" in df.columns else -1
                else:
                    stage_progress_gt = float("nan")
                    task_index = -1

                all_results.append({
                    "frame_idx": actual_fi,
                    "future_frame_idx": fi_fut * frame_interval + min_fi,
                    "relative_advantage": float(np.clip(rel_val, -1, 1)),
                    "absolute_value": float(np.clip(abs_val, -1, 1)),
                    "stage_progress_gt": stage_progress_gt,
                    "task_index": task_index,
                })
            start = end

        # Compute absolute_advantage from absolute_value differences
        by_fi = {r["frame_idx"]: r for r in all_results}
        for r in all_results:
            fi, fi_fut = r["frame_idx"], r["future_frame_idx"]
            gap = (fi_fut - fi) // frame_interval  # gap in sampled units
            if gap == 0:
                r["absolute_advantage"] = 0.0
            else:
                delta = by_fi[fi_fut]["absolute_value"] - r["absolute_value"]
                real_gap = fi_fut - fi
                r["absolute_advantage"] = float(np.clip(
                    delta / real_gap * RELATIVE_INTERVAL if real_gap != RELATIVE_INTERVAL else delta,
                    -1, 1))

        torch.cuda.empty_cache()
        return all_results

    results = {}
    for ep in tqdm(episode_list, desc=f"[GPU {gpu_id}]"):
        try:
            results[ep] = eval_episode(ep)
        except Exception as e:
            print(f"[GPU {gpu_id}] skip ep {ep}: {e}", flush=True)
        torch.cuda.empty_cache()

    pool.shutdown(wait=False)
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"[GPU {gpu_id}] done → {out_pkl}", flush=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_results: dict) -> dict:
    """
    Compute quality metrics on the flat episode dict.

    Key metrics:
      spearman_gt   : Spearman(absolute_value, stage_progress_gt) per episode
                      → does model track GT task progress?
      frame_disc    : mean(abs_val | task_index=1) - mean(abs_val | task_index=0)
                      → does model rank positive frames above negative frames?
      monotonicity  : fraction of consecutive (abs_val[i+1] >= abs_val[i]) steps
    """
    from scipy.stats import spearmanr

    spearman_gt_list = []
    frame_disc_list = []
    mono_list = []
    mean_abs_val_list = []

    for results in all_results.values():
        if not results:
            continue
        av = np.array([r["absolute_value"] for r in results])
        sp_gt = np.array([r["stage_progress_gt"] for r in results])
        ti = np.array([r["task_index"] for r in results])

        # Spearman vs GT progress
        valid = ~np.isnan(sp_gt)
        if valid.sum() > 2:
            rho, _ = spearmanr(av[valid], sp_gt[valid])
            spearman_gt_list.append(float(rho))

        # Frame-level discrimination
        pos_mask = ti == 1
        neg_mask = ti == 0
        if pos_mask.any() and neg_mask.any():
            disc = float(np.mean(av[pos_mask]) - np.mean(av[neg_mask]))
            frame_disc_list.append(disc)

        # Monotonicity
        if len(av) > 1:
            mono_list.append(float(np.mean(np.diff(av) >= 0)))

        mean_abs_val_list.append(float(np.mean(av)))

    def _s(arr):
        return {
            "mean":   float(np.mean(arr))   if arr else float("nan"),
            "median": float(np.median(arr)) if arr else float("nan"),
            "std":    float(np.std(arr))    if arr else float("nan"),
        }

    return {
        "n_episodes": len([v for v in all_results.values() if v]),
        "spearman_vs_gt":      _s(spearman_gt_list),
        "frame_discrimination": _s(frame_disc_list),
        "monotonicity":        _s(mono_list),
        "mean_absolute_value": _s(mean_abs_val_list),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_curves(all_results: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    xs = np.linspace(0, 1, 100)

    # ---- 1. absolute_value curve coloured by Spearman-GT per episode ----
    fig, ax = plt.subplots(figsize=(18, 7))
    rhos = []
    for results in all_results.values():
        if not results:
            continue
        fi = np.array([r["frame_idx"] for r in results])
        av = np.array([r["absolute_value"] for r in results])
        sp_gt = np.array([r["stage_progress_gt"] for r in results])
        x_norm = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
        valid = ~np.isnan(sp_gt)
        rho = float(spearmanr(av[valid], sp_gt[valid])[0]) if valid.sum() > 2 else 0.0
        rhos.append(rho)
        color = plt.cm.RdYlGn((rho + 1) / 2)
        ax.plot(x_norm, av, color=color, alpha=0.35, linewidth=0.8)

    # Mean curve
    curves = []
    for results in all_results.values():
        if not results:
            continue
        fi = np.array([r["frame_idx"] for r in results])
        av = np.array([r["absolute_value"] for r in results])
        x_norm = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
        curves.append(np.interp(xs, x_norm, av))
    if curves:
        ax.plot(xs, np.mean(curves, axis=0), color="black", linewidth=2.5, label="mean")
        ax.fill_between(xs, np.percentile(curves, 25, axis=0),
                        np.percentile(curves, 75, axis=0), color="gray", alpha=0.2, label="IQR")

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(-1, 1))
    plt.colorbar(sm, ax=ax, label="Spearman(abs_val, stage_progress_gt)")
    ax.set_title(
        f"adv_est_v1 — absolute_value curves  (n={len([v for v in all_results.values() if v])})\n"
        f"colour = Spearman vs GT  |  mean rho={np.mean(rhos):.3f}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Normalized frame position")
    ax.set_ylabel("absolute_value (predicted cumulative progress)")
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "curves_absolute_value.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p}")

    # ---- 2. Frame discrimination: abs_val distribution by task_index ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

    for ax, key, ylabel in [
        (axes2[0], "absolute_value",    "absolute_value"),
        (axes2[1], "relative_advantage","relative_advantage"),
    ]:
        pos_vals = [r[key] for res in all_results.values() if res
                    for r in res if r["task_index"] == 1]
        neg_vals = [r[key] for res in all_results.values() if res
                    for r in res if r["task_index"] == 0]
        if pos_vals and neg_vals:
            ax.hist(neg_vals, bins=60, alpha=0.6, color="tomato", density=True,
                    label=f"task_index=0 (neg)  mean={np.mean(neg_vals):.3f}")
            ax.hist(pos_vals, bins=60, alpha=0.6, color="steelblue", density=True,
                    label=f"task_index=1 (pos)  mean={np.mean(pos_vals):.3f}")
            ax.axvline(np.mean(pos_vals), color="steelblue", linestyle="--", linewidth=1.5)
            ax.axvline(np.mean(neg_vals), color="tomato",    linestyle="--", linewidth=1.5)
            disc = np.mean(pos_vals) - np.mean(neg_vals)
            ax.set_title(f"{ylabel}\nFrame discrimination Δ={disc:+.4f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(ylabel)

    fig2.suptitle(
        "adv_est_v1 — frame-level discrimination\n"
        "(task_index is per-frame: 1=top-30% absolute_advantage, 0=bottom-70%)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    p2 = out_dir / "frame_discrimination.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p2}")

    # ---- 3. abs_val vs stage_progress_gt scatter ----
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    av_all, gt_all, ti_all = [], [], []
    for results in all_results.values():
        if not results:
            continue
        for r in results:
            if not np.isnan(r["stage_progress_gt"]):
                av_all.append(r["absolute_value"])
                gt_all.append(r["stage_progress_gt"])
                ti_all.append(r["task_index"])

    av_all = np.array(av_all)
    gt_all = np.array(gt_all)
    ti_all = np.array(ti_all)

    for ti, color, label in [(0, "tomato", "task_index=0"), (1, "steelblue", "task_index=1")]:
        m = ti_all == ti
        ax3.scatter(gt_all[m], av_all[m], c=color, alpha=0.15, s=4, label=label)

    from scipy.stats import spearmanr
    rho_all, _ = spearmanr(av_all, gt_all)
    ax3.set_xlabel("stage_progress_gt  (ground truth task progress 0→1)")
    ax3.set_ylabel("absolute_value (model prediction)")
    ax3.set_title(
        f"abs_val vs stage_progress_gt  (Spearman ρ={rho_all:.4f})\n"
        f"n_frames={len(av_all)}",
        fontsize=12, fontweight="bold"
    )
    ax3.legend(fontsize=9, markerscale=4)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = out_dir / "scatter_vs_gt.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p3}")


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def print_report(m: dict, out_dir: Path):
    sg = m["spearman_vs_gt"]
    fd = m["frame_discrimination"]
    mo = m["monotonicity"]

    lines = [
        "=" * 64,
        "  adv_est_v1  Validation Report (frame-level evaluation)",
        "=" * 64,
        f"\n  Episodes evaluated : {m['n_episodes']}",
        "",
        "[Spearman(absolute_value, stage_progress_gt)]",
        f"  mean   : {sg['mean']:.4f}",
        f"  median : {sg['median']:.4f}",
        f"  std    : {sg['std']:.4f}",
        "  → measures whether predicted progress tracks GT task progress",
        "",
        "[Frame Discrimination  Δ = mean(abs_val|pos) - mean(abs_val|neg)]",
        f"  mean   : {fd['mean']:+.4f}",
        f"  median : {fd['median']:+.4f}",
        f"  std    : {fd['std']:.4f}",
        "  → positive = model correctly ranks high-quality frames higher",
        "",
        "[Monotonicity  (fraction of non-decreasing steps in abs_val curve)]",
        f"  mean   : {mo['mean']:.4f}",
        f"  median : {mo['median']:.4f}",
        "",
        "[Quality Checks]",
    ]

    checks = [
        ("Spearman vs GT > 0.6 (mean)",      sg["mean"] > 0.6),
        ("Spearman vs GT > 0.5 (median)",     sg["median"] > 0.5),
        ("Frame discrimination Δ > 0.0",      fd["mean"] > 0.0),
        ("Frame discrimination Δ > 0.05",     fd["mean"] > 0.05),
        ("Monotonicity > 0.55",               mo["mean"] > 0.55),
    ]
    for desc, ok in checks:
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    lines.append("=" * 64)

    report = "\n".join(lines)
    print(report)
    (out_dir / "validation_report.txt").write_text(report)
    print(f"\n[report] saved → {out_dir}/validation_report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of episodes to evaluate")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--out", type=str, default="eval_adv_est_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument(
        "--data", type=str,
        default="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_advantage",
        help="Path to LeRobot dataset root (must contain data/ and videos/)",
    )
    args = parser.parse_args()

    global DATA_ROOT
    DATA_ROOT = Path(args.data)

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    ckpt_dir = str(CKPT_BASE / str(args.steps))
    print(f"[config] ckpt          : {ckpt_dir}")
    print(f"[config] out           : {out_dir}")
    print(f"[config] n_episodes    : {args.n}")
    print(f"[config] gpus          : {gpu_ids}")
    print(f"[config] batch_size    : {args.batch_size}")
    print(f"[config] frame_interval: {FRAME_INTERVAL}  (~{1800//FRAME_INTERVAL} frames/ep)")

    print("[data] loading episode metadata...")
    meta = load_episode_meta()
    episode_list = sample_episodes(meta, args.n, seed=args.seed)
    print(f"[data] sampled {len(episode_list)} episodes")

    (out_dir / "sampled_episodes.json").write_text(
        json.dumps({"episodes": episode_list}, indent=2)
    )

    # Distribute round-robin across GPUs
    shards: list[list] = [[] for _ in range(n_gpus)]
    for i, ep in enumerate(episode_list):
        shards[i % n_gpus].append(ep)

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    shard_pkls = []
    processes = []
    for rank, (gid, shard) in enumerate(zip(gpu_ids, shards)):
        pkl_path = str(out_dir / f"shard_{rank}.pkl")
        shard_pkls.append(pkl_path)
        p = mp.Process(
            target=worker,
            args=(gid, shard, ckpt_dir, pkl_path, args.batch_size, FRAME_INTERVAL,
                  str(DATA_ROOT)),
            daemon=True,
        )
        p.start()
        processes.append(p)
        print(f"[launch] GPU {gid}  shard size={len(shard)}  pid={p.pid}")

    for p in processes:
        p.join()
    print("[merge] all workers done")

    # Merge shard results into a flat dict: {ep_idx: [results]}
    all_results = {}
    for pkl_path in shard_pkls:
        if not Path(pkl_path).exists():
            print(f"[warn] missing shard {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            all_results.update(pickle.load(f))

    with open(out_dir / "inference_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"[save] merged → {out_dir}/inference_results.pkl  (n={len(all_results)})")

    # Compute metrics and report
    m = compute_metrics(all_results)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))

    plot_curves(all_results, out_dir)
    print_report(m, out_dir)

    print(f"\n[done] outputs in {out_dir}/")


if __name__ == "__main__":
    main()
