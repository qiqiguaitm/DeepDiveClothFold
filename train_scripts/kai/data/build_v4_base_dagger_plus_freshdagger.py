#!/usr/bin/env python3
"""任务② (Mode B plan §8): A_v4_base_dagger_plus_freshdagger = A_v4_base_dagger 全量(2017ep, 已标)
+ 额外 fresh dagger 06-29~07-03(506ep, 复用 A_v4_freshdagger_ft 里 group=='dagger' 的已标 ep) → ~2523ep.

思路(plan §8): 保住已证不冻的 A_v4_base_dagger 配方不动, 只【加】20% 新 dagger 修夹爪. 全程 no-DCT.
复用已有 adv_est_v1 advantage 标签(免 2523ep 重跑 AE 打标 ~36h) → 全集重算 discretize top-30% + norm.

统一列(防 lerobot schema 不一致 CastError, 见 ep104 教训): 只留标准 7 列 + task_index + absolute_advantage.
视频 deref-symlink 到真实 mp4(避免断链). 之后调 discretize_advantage.py + compute_norm_stats.
Run: kai0/.venv/bin/python train_scripts/kai/data/build_v4_base_dagger_plus_freshdagger.py
"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from build_no_release import per_episode_stats   # lerobot episodes_stats 'stats' dict

ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built")
BASE = ROOT / "A_v4_base_dagger"            # 2017ep, labeled (absolute_advantage/task_index)
FRESH = ROOT / "A_v4_freshdagger_ft"        # 973ep; dagger(group=='dagger')=新 dagger 506
OUT = ROOT / "A_v4_base_dagger_plus_freshdagger"
CAMS = ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
# 统一列: 标准 lerobot 7 + task_index. 丢 advantage 列(base_dagger 213ep 本就无 advantage 列).
# 复用各源现成 task_index(base_dagger proven 标签原样保住 + fresh dagger 现成标签), 不重 discretize
# → 忠实 plan §8「保住 A_v4_base_dagger 配方不动, 只做加法」. tasks.jsonl 沿用 pos/neg.
KEEP = ["observation.state", "action", "timestamp", "frame_index", "episode_index",
        "index", "task_index"]
CHUNK = 0

def parquet_path(root, ep):
    return root / "data" / f"chunk-{CHUNK:03d}" / f"episode_{ep:06d}.parquet"

# 1) 收集源 ep: base_dagger 全部 2017 + freshdagger 的 dagger(group=='dagger')
base_eps = [json.loads(l)["episode_index"] for l in (BASE/"meta"/"episodes.jsonl").read_text().splitlines()]
fresh_meta = [json.loads(l) for l in (FRESH/"meta"/"episodes.jsonl").read_text().splitlines()]
fresh_dagger = [e["episode_index"] for e in fresh_meta if e.get("group") == "dagger"]
print(f"base_dagger={len(base_eps)}ep + fresh_dagger={len(fresh_dagger)}ep -> {len(base_eps)+len(fresh_dagger)}ep", flush=True)
assert len(fresh_dagger) == 506, f"期望 506 新 dagger, 实际 {len(fresh_dagger)}"

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "data" / f"chunk-{CHUNK:03d}").mkdir(parents=True)
(OUT / "meta").mkdir()

sources = [(BASE, e) for e in base_eps] + [(FRESH, e) for e in fresh_dagger]

def videos_ok(src_root, old_ep):
    for cam in CAMS:
        sv = (src_root / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{old_ep:06d}.mp4").resolve()
        if not sv.is_file():
            return False
    return True

new_eps_meta = []; eps_stats = []
gidx = 0; total_frames = 0; new_ep = 0; skipped = []
for src_root, old_ep in sources:
    if not videos_ok(src_root, old_ep):          # 跳过断链 ep(cnsh 源视频被清理, 见 v4 源视频侵蚀教训)
        skipped.append((src_root.name, old_ep)); continue
    df = pq.read_table(parquet_path(src_root, old_ep)).to_pandas()
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        raise SystemExit(f"FATAL ep {old_ep}@{src_root.name} 缺列 {missing} (无法复用标签)")
    df = df[KEEP].copy()
    n = len(df)
    df["episode_index"] = np.int64(new_ep)
    df["index"] = np.arange(gidx, gidx + n, dtype=np.int64)
    gidx += n; total_frames += n
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path(OUT, new_ep))
    for cam in CAMS:
        sv = (src_root / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{old_ep:06d}.mp4").resolve()
        dv = OUT / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{new_ep:06d}.mp4"
        dv.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(sv), dv)
    new_eps_meta.append({"episode_index": new_ep, "tasks": ["Flatten and fold the cloth."], "length": n})
    eps_stats.append({"episode_index": new_ep, "stats": per_episode_stats(df)})
    new_ep += 1
    if new_ep % 500 == 0:
        print(f"  {new_ep} written", flush=True)
print(f"skipped {len(skipped)} 断链 ep: {skipped[:15]}", flush=True)

# meta: info.json(以 BASE 为模板改计数)+ episodes.jsonl
info = json.loads((BASE / "meta" / "info.json").read_text())
info["total_episodes"] = new_ep
info["total_frames"] = total_frames
info["total_chunks"] = 1
info["chunks_size"] = new_ep
(OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2))
with (OUT / "meta" / "episodes.jsonl").open("w") as f:
    for e in new_eps_meta:
        f.write(json.dumps(e) + "\n")
with (OUT / "meta" / "episodes_stats.jsonl").open("w") as f:   # lerobot 必需, 否则 FileNotFound→OfflineModeIsEnabled 崩
    for s in eps_stats:
        f.write(json.dumps(s) + "\n")
# tasks.jsonl: 沿用 base_dagger 的 pos/neg(不重 discretize)
shutil.copy(BASE / "meta" / "tasks.jsonl", OUT / "meta" / "tasks.jsonl")
print(f"merged {new_ep}ep / {total_frames}frames (skipped {len(skipped)}) -> {OUT}", flush=True)
print("DONE_BUILD", flush=True)
