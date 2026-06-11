#!/usr/bin/env python
"""TCC 最小复现/适配: 冻结 DINOv2 特征 + MLP embedder, 用 XIRL 官方 compute_tcc_loss
(deterministic regression-MSE cycle-back) 跨 episode 对齐训练。
(cross_episode_recurrence_value_plan.md Phase C; XIRL arXiv:2106.03911, TCC arXiv:1904.07846)

产物 (--out 下):
  feat_cache/ep{N}.npz       每 episode DINOv2 特征缓存 (3Hz)
  tcc_head.pt                训练好的 MLP embedder
  curves_train.png           loss 曲线
  value_curves.png           XIRL value(=−‖emb−goal‖) 随机 episode 曲线
  eval.json                  val episode 上 phase-vs-GT Kendall tau (若有 stage_progress_gt)

用法 (gf0 本地):
  HF_HUB_OFFLINE=1 kai0/.venv/bin/python train_scripts/kai/data/tcc_train_features.py \
    --dataset kai0/data/Task_A/kai0_advantage --n-train 120 --n-val 20 --steps 2000 \
    --out temp/tcc_kai0
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

XIRL_DIR = "/vePFS/tim/workspace/recurrence_research/google-research/xirl"
sys.path.insert(0, XIRL_DIR)
from xirl.losses import compute_tcc_loss  # noqa  (官方实现, PyTorch)

CAM_CANDIDATES = ("observation.images.top_head", "top_head", "observation.images.cam_high")


def cam_dir(ds: Path) -> str:
    base = ds / "videos" / "chunk-000"
    for c in CAM_CANDIDATES:
        if (base / c).is_dir():
            return c
    raise FileNotFoundError(base)


def extract_episode(ds: Path, cam: str, ep: int, chunks_size: int, stride: int,
                    proc, model, dev) -> np.ndarray:
    import av
    mp4 = ds / "videos" / f"chunk-{ep // chunks_size:03d}" / cam / f"episode_{ep:06d}.mp4"
    imgs = []
    c = av.open(str(mp4))
    for i, f in enumerate(c.decode(video=0)):
        if i % stride:
            continue
        h, w = f.height, f.width
        s = 224 / min(h, w)
        g = f.reformat(width=round(w * s), height=round(h * s), format="rgb24")
        img = g.to_ndarray(format="rgb24")
        hh, ww = img.shape[:2]
        y, x = (hh - 224) // 2, (ww - 224) // 2
        imgs.append(img[y:y + 224, x:x + 224])
    c.close()
    feats = []
    with torch.no_grad():
        for b in range(0, len(imgs), 64):
            px = proc(images=imgs[b:b + 64], return_tensors="pt").to(dev)
            cls = model(**px).last_hidden_state[:, 0]
            feats.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.concatenate(feats)


class Head(nn.Module):
    def __init__(self, din=384, dh=256, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.GELU(),
                                 nn.Linear(dh, dh), nn.GELU(), nn.Linear(dh, dout))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kai0/data/Task_A/kai0_advantage")
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-eps", type=int, default=8, help="每 batch episode 数")
    ap.add_argument("--frames-per-ep", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="temp/tcc_kai0")
    ap.add_argument("--model", default="facebook/dinov2-small")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    ds, out = Path(args.dataset), Path(args.out)
    (out / "feat_cache").mkdir(parents=True, exist_ok=True)
    cam = cam_dir(ds)
    chunks_size = json.load(open(ds / "meta" / "info.json")).get("chunks_size", 1000)
    eps_all = [json.loads(l)["episode_index"] for l in open(ds / "meta" / "episodes.jsonl")]
    random.Random(args.seed).shuffle(eps_all)
    train_eps = sorted(eps_all[: args.n_train])
    val_eps = sorted(eps_all[args.n_train: args.n_train + args.n_val])
    print(f"[tcc] train {len(train_eps)} eps, val {len(val_eps)} eps, cam={cam}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(args.model)
    enc = AutoModel.from_pretrained(args.model).to(dev).eval()

    def get_feats(ep: int) -> np.ndarray:
        p = out / "feat_cache" / f"ep{ep}.npz"
        if p.exists():
            return np.load(p)["f"]
        f = extract_episode(ds, cam, ep, chunks_size, args.stride, proc, enc, dev)
        np.savez_compressed(p, f=f)
        return f

    print("[tcc] extracting/caching features ...")
    feats = {}
    for n, ep in enumerate(train_eps + val_eps):
        feats[ep] = get_feats(ep)
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(train_eps)+len(val_eps)}")

    head = Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)
    losses = []
    T = args.frames_per_ep
    for step in range(args.steps):
        batch_eps = random.sample(train_eps, args.batch_eps)
        embs, steps_l, lens = [], [], []
        for ep in batch_eps:
            f = feats[ep]; n = len(f)
            idx = np.sort(np.random.choice(n, size=min(T, n), replace=n < T))
            x = torch.from_numpy(f[idx]).float().to(dev)
            embs.append(head(x))
            steps_l.append(torch.from_numpy(idx).long())
            lens.append(n)
        embs = torch.stack(embs)                                  # (B,T,D)
        steps_t = torch.stack(steps_l).to(dev)
        lens_t = torch.tensor(lens).to(dev)
        loss = compute_tcc_loss(
            embs=embs, idxs=steps_t, seq_lens=lens_t,
            stochastic_matching=False, normalize_embeddings=False,
            loss_type="regression_mse", similarity_type="l2",
            num_cycles=20, cycle_length=2, temperature=0.1,
            label_smoothing=0.1, variance_lambda=0.001,
            huber_delta=0.1, normalize_indices=True)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
        if (step + 1) % 200 == 0:
            print(f"[tcc] step {step+1}/{args.steps} loss {np.mean(losses[-200:]):.4f}")

    torch.save(head.state_dict(), out / "tcc_head.pt")

    # ---- eval: phase proxy 与 GT 的 Kendall tau (val eps; 需 stage_progress_gt) ----
    import pandas as pd
    from scipy.stats import kendalltau
    head.eval()
    # goal embedding = train eps 末帧均值 (XIRL 协议)
    with torch.no_grad():
        goals = [head(torch.from_numpy(feats[ep][-3:]).float().to(dev)).mean(0) for ep in train_eps]
        goal = torch.stack(goals).mean(0)
    taus_raw, taus_tcc, taus_val = [], [], []
    for ep in val_eps:
        f = feats[ep]
        with torch.no_grad():
            e = head(torch.from_numpy(f).float().to(dev))
            v = -torch.norm(e - goal, dim=-1).cpu().numpy()       # XIRL value
        pq = ds / "data" / f"chunk-{ep // chunks_size:03d}" / f"episode_{ep:06d}.parquet"
        try:
            gt = pd.read_parquet(pq, columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy()
        except Exception:
            continue
        idx = np.arange(len(f)) * args.stride
        idx = idx[idx < len(gt)]
        g = gt[idx]; v = v[: len(idx)]
        if g.std() < 1e-6:
            continue
        # 对照: 原始 DINOv2 特征到 goal 的负距离 (不训 TCC)
        rawgoal = np.stack([feats[tep][-3:].mean(0) for tep in train_eps]).mean(0)
        vr = -np.linalg.norm(f[: len(idx)] - rawgoal, axis=-1)
        taus_tcc.append(kendalltau(v, g)[0])
        taus_raw.append(kendalltau(vr, g)[0])
    res = {"tau_tcc_value": float(np.nanmean(taus_tcc)) if taus_tcc else None,
           "tau_raw_dino_value": float(np.nanmean(taus_raw)) if taus_raw else None,
           "n_val": len(taus_tcc), "steps": args.steps, "n_train": len(train_eps)}
    json.dump(res, open(out / "eval.json", "w"), indent=2)
    print("\n========== TCC EVAL ==========")
    print(json.dumps(res, indent=2))

    # ---- 可视化 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.convolve(losses, np.ones(50) / 50, mode="valid"))
    ax.set_title("TCC loss (50-step MA)"); ax.set_xlabel("step"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "curves_train.png", dpi=110)

    sample = val_eps[:4] if val_eps else train_eps[:4]
    fig, axes = plt.subplots(1, len(sample), figsize=(4.2 * len(sample), 3.2))
    for ax, ep in zip(np.atleast_1d(axes), sample):
        f = feats[ep]
        with torch.no_grad():
            e = head(torch.from_numpy(f).float().to(dev))
            v = -torch.norm(e - goal, dim=-1).cpu().numpy()
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        ax.plot(np.linspace(0, 1, len(v)), v, "g-", lw=1.5, label="TCC value (norm)")
        ax.plot([0, 1], [0, 1], "k--", alpha=.4, label="linear ref")
        ax.set_title(f"ep{ep}", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle("XIRL-style value = -||emb - goal|| (val episodes)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "value_curves.png", dpi=110)
    print(f"outputs -> {out}/")


if __name__ == "__main__":
    main()
