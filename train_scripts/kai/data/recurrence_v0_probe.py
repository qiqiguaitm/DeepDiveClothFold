#!/usr/bin/env python
"""V0 探针: 跨 episode 重复度 → 自动 milestone 假说验证 (cross_episode_recurrence_value_plan.md §3.1).

抽 N 个 episode 的 top_head 帧 (3Hz) → DINOv2 特征 → KMeans → 每簇 episode 覆盖率
(first-visit, McGovern&Barto 教训: 不用帧频). 输出 4 件套到 --out:
  coverage_curve.png        簇覆盖率 vs 簇平均时间位置 (峰=候选 milestone)
  milestone_clusters.png    高覆盖簇代表帧网格 (肉眼判语义; 每簇取不同 episode 的帧防 nuisance)
  low_coverage_segments.md  低覆盖段清单+缩略图 (人工审计: 真错误 vs regrasp 恢复)
  per_episode_timeline.png  episode×时间 热图, 色=帧所属簇覆盖率
  embeddings.npz            特征缓存 (复跑聚类免重抽)

用法 (kai0/.venv, gf0):
  HF_ENDPOINT=https://hf-mirror.com kai0/.venv/bin/python \
    train_scripts/kai/data/recurrence_v0_probe.py \
    --dataset kai0/data/Task_A/self_built/A_new_smooth_800/base \
    --n-episodes 50 --out temp/recurrence_v0
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path

import av
import numpy as np
import torch

CAM_CANDIDATES = ("observation.images.top_head", "top_head", "observation.images.cam_high", "cam_high")


def find_cam_dir(ds: Path) -> str:
    base = ds / "videos" / "chunk-000"
    for c in CAM_CANDIDATES:
        if (base / c).is_dir():
            return c
    raise FileNotFoundError(f"no top camera dir under {base}")


def video_path(ds: Path, cam: str, ep: int, chunks_size: int) -> Path:
    return ds / "videos" / f"chunk-{ep // chunks_size:03d}" / cam / f"episode_{ep:06d}.mp4"


def decode_strided(mp4: Path, stride: int, size: int = 224):
    """Decode every `stride`-th frame, center-crop-resize to size."""
    out, idxs = [], []
    c = av.open(str(mp4))
    for i, f in enumerate(c.decode(video=0)):
        if i % stride:
            continue
        h, w = f.height, f.width
        s = size / min(h, w)
        g = f.reformat(width=round(w * s), height=round(h * s), format="rgb24")
        img = g.to_ndarray(format="rgb24")
        hh, ww = img.shape[:2]
        y, x = (hh - size) // 2, (ww - size) // 2
        out.append(img[y:y + size, x:x + size])
        idxs.append(i)
    c.close()
    return np.stack(out), np.array(idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kai0/data/Task_A/self_built/A_new_smooth_800/base")
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--stride", type=int, default=10, help="30fps/10=3Hz")
    ap.add_argument("--k", type=int, default=48, help="KMeans clusters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="temp/recurrence_v0")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--top-m", type=int, default=10, help="milestone 候选簇数 (高覆盖)")
    args = ap.parse_args()

    ds, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thumbs").mkdir(exist_ok=True)
    cam = find_cam_dir(ds)
    print(f"[v0] dataset={ds} cam={cam}")

    chunks_size = json.load(open(ds / "meta" / "info.json")).get("chunks_size", 1000)
    eps_meta = [json.loads(l) for l in open(ds / "meta" / "episodes.jsonl")]
    all_eps = [e["episode_index"] for e in eps_meta]
    random.Random(args.seed).shuffle(all_eps)
    eps = sorted(all_eps[: args.n_episodes])
    print(f"[v0] sampled {len(eps)} episodes: {eps[:8]}...")

    # ---- DINOv2 特征 ----
    from transformers import AutoImageProcessor, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(dev).eval()

    cache = out / "embeddings.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        feats, ep_ids, fr_idx, tnorm = z["feats"], z["ep_ids"], z["fr_idx"], z["tnorm"]
        print(f"[v0] loaded cache: {feats.shape}")
    else:
        feats, ep_ids, fr_idx, tnorm = [], [], [], []
        for n, ep in enumerate(eps):
            mp4 = video_path(ds, cam, ep, chunks_size)
            imgs, idxs = decode_strided(mp4, args.stride)
            with torch.no_grad():
                for b in range(0, len(imgs), 64):
                    px = proc(images=list(imgs[b:b + 64]), return_tensors="pt").to(dev)
                    cls = model(**px).last_hidden_state[:, 0]          # CLS token
                    feats.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
            ep_ids += [ep] * len(idxs)
            fr_idx += list(idxs)
            tnorm += list(idxs / max(1, idxs[-1]))
            print(f"[v0] {n+1}/{len(eps)} ep{ep}: {len(idxs)} frames")
        feats = np.concatenate(feats)
        ep_ids, fr_idx, tnorm = np.array(ep_ids), np.array(fr_idx), np.array(tnorm)
        np.savez_compressed(cache, feats=feats, ep_ids=ep_ids, fr_idx=fr_idx, tnorm=tnorm)
        print(f"[v0] features {feats.shape} cached -> {cache}")

    # ---- KMeans + first-visit 覆盖率 ----
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.k, n_init=4, random_state=args.seed).fit(feats)
    lab = km.labels_
    n_ep = len(set(ep_ids.tolist()))
    cov = np.array([len(set(ep_ids[lab == c].tolist())) / n_ep for c in range(args.k)])  # first-visit!
    tpos = np.array([tnorm[lab == c].mean() for c in range(args.k)])
    # nuisance 检查: 簇内帧的 episode 集中度 (top-1 episode 占比; 高=可能是布颜色/个体特征而非阶段)
    dom = np.array([np.bincount(ep_ids[lab == c]).max() / (lab == c).sum() for c in range(args.k)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) coverage vs time
    order = np.argsort(tpos)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(tpos[order], cov[order], "o-", ms=5)
    for c in np.argsort(cov)[-args.top_m:]:
        ax.annotate(str(c), (tpos[c], cov[c]), fontsize=7)
    ax.set_xlabel("cluster mean normalized time"); ax.set_ylabel("episode coverage (first-visit)")
    ax.set_title(f"cluster coverage vs time  (k={args.k}, {n_ep} eps)  — peaks = candidate milestones")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "coverage_curve.png", dpi=120)

    # (2) 高覆盖簇代表帧 (每簇取 4 个不同 episode、最近质心的帧)
    top = sorted(np.argsort(cov)[-args.top_m:], key=lambda c: tpos[c])
    fig, axes = plt.subplots(len(top), 4, figsize=(10, 2.4 * len(top)))
    need = {}  # (ep, frame) -> (row, col)
    for r, c in enumerate(top):
        m = np.where(lab == c)[0]
        d = np.linalg.norm(feats[m] - km.cluster_centers_[c], axis=1)
        seen, picks = set(), []
        for i in m[np.argsort(d)]:
            if ep_ids[i] not in seen:
                seen.add(ep_ids[i]); picks.append(i)
            if len(picks) == 4:
                break
        for col, i in enumerate(picks):
            need[(int(ep_ids[i]), int(fr_idx[i]))] = (r, col)
        axes[r, 0].set_ylabel(f"c{c}\ncov={cov[c]:.0%}\nt={tpos[c]:.2f}\ndom={dom[c]:.0%}", fontsize=7)
    by_ep = {}
    for (ep, fr), pos in need.items():
        by_ep.setdefault(ep, []).append((fr, pos))
    for ep, items in by_ep.items():
        mp4 = video_path(ds, cam, ep, chunks_size)
        want = {fr: pos for fr, pos in items}
        cont = av.open(str(mp4))
        for i, f in enumerate(cont.decode(video=0)):
            if i in want:
                r, col = want[i]
                axes[r, col].imshow(f.to_ndarray(format="rgb24"))
                axes[r, col].set_title(f"ep{ep} f{i}", fontsize=6)
        cont.close()
    for ax_ in axes.flat:
        ax_.set_xticks([]); ax_.set_yticks([])
    fig.suptitle("top-coverage clusters (candidate milestones) — 每行一簇, 4 个不同 episode", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "milestone_clusters.png", dpi=120)

    # (3) per-episode timeline 热图
    fig, ax = plt.subplots(figsize=(12, 0.16 * n_ep + 2))
    for row, ep in enumerate(sorted(set(ep_ids.tolist()))):
        m = ep_ids == ep
        ax.scatter(tnorm[m], np.full(m.sum(), row), c=cov[lab[m]], cmap="RdYlGn",
                   vmin=0, vmax=1, s=6, marker="s")
    ax.set_xlabel("normalized time"); ax.set_ylabel("episode (row)")
    ax.set_title("per-frame cluster coverage (red=low → 候选 detour/error/recovery)")
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn"), ax=ax, label="coverage")
    fig.tight_layout(); fig.savefig(out / "per_episode_timeline.png", dpi=120)

    # (4) 低覆盖段清单 (bottom-decile 簇的连续段, 给人工审计)
    low_clusters = set(np.argsort(cov)[: max(1, args.k // 10)].tolist())
    lines = ["# 低覆盖段审计清单 (bottom-decile clusters)\n",
             f"clusters: {sorted(low_clusters)} (coverage: " +
             ", ".join(f"c{c}={cov[c]:.0%}" for c in sorted(low_clusters)) + ")\n",
             "\n| episode | 帧段(原始30fps) | 时长s | 簇 |  审计(人工填: error/recovery/nuisance) |",
             "|---|---|---|---|---|"]
    thumbs = []
    for ep in sorted(set(ep_ids.tolist())):
        m = np.where(ep_ids == ep)[0]
        runs, cur = [], None
        for i in m:
            if lab[i] in low_clusters:
                if cur is None:
                    cur = [fr_idx[i], fr_idx[i], lab[i]]
                else:
                    cur[1] = fr_idx[i]
            elif cur is not None:
                runs.append(cur); cur = None
        if cur is not None:
            runs.append(cur)
        for s, e, c in runs:
            if e - s < args.stride:  # 跳过孤立单帧
                continue
            lines.append(f"| {ep} | {s}-{e} | {(e-s)/30:.1f} | c{c} |  |")
            thumbs.append((int(ep), int((s + e) // 2)))
    # 缩略图 (每段中点帧)
    for ep, fr in thumbs[:60]:
        mp4 = video_path(ds, cam, ep, chunks_size)
        cont = av.open(str(mp4))
        for i, f in enumerate(cont.decode(video=0)):
            if i == fr:
                from PIL import Image
                Image.fromarray(f.to_ndarray(format="rgb24")).save(out / "thumbs" / f"ep{ep}_f{fr}.jpg")
                break
        cont.close()
    lines.append(f"\n缩略图: thumbs/ ({min(len(thumbs),60)} 张, 每段中点帧)")
    (out / "low_coverage_segments.md").write_text("\n".join(lines))

    # 摘要
    print("\n========== V0 SUMMARY ==========")
    print(f"eps={n_ep} frames={len(feats)} k={args.k}")
    print(f"coverage: min={cov.min():.0%} median={np.median(cov):.0%} max={cov.max():.0%}")
    print(f"top milestones (cov, t, ep-dominance): " +
          ", ".join(f"c{c}({cov[c]:.0%},t={tpos[c]:.2f},dom={dom[c]:.0%})" for c in top))
    print(f"⚠️ dom 高(>30%)的簇可能是 nuisance(单 episode/布个体特征)而非阶段")
    print(f"low-coverage segments for audit: {len(thumbs)} (see low_coverage_segments.md)")
    print(f"outputs -> {out}/")


if __name__ == "__main__":
    main()
