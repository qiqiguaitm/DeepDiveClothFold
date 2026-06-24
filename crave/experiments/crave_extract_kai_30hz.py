#!/usr/bin/env python
"""为 CRAVE 30Hz 挖矿扩样: 从 kai0_base + kai0_dagger 抽 episode, 逐帧(stride1=30Hz)提 raw⊕armmask⊕state。
- raw = DINOv2-small patch tokens mean-pool(L2-norm); armmask = 剔除臂/橙缆 patch 后均值(同 lerobot_v3_extract_features)。
- 只用 kai 数据(kai0_base / kai0_dagger), 复用已有 83 个 30Hz 特征(crave_30hz_mine), 增量补到 ~250。
- 多数据源: cache 索引 = kai0_base→ep{i}, kai0_dagger→ep{3000000+i}; sources.json 记 {cache_idx:[dataset_rel, real_ep]}。
用法:
  python crave_extract_kai_30hz.py --prep         # 采样+复用83+写sources.json(非GPU)
  CUDA_VISIBLE_DEVICES=0 python ... --shard 0 2    # GPU0 处理一半
  CUDA_VISIBLE_DEVICES=1 python ... --shard 1 2    # GPU1 处理另一半
"""
import argparse, colorsys, json, shutil
import numpy as np, pandas as pd

from crave.config import REPO
from crave.data import kai0

# TODO(crave-lib): bespoke raw+armmask dinov2-small encoder (arm-prototype + HSV
# orange-cable masking) is not exposed by crave.encoders.load_encoder (no armmask path);
# kept inline to preserve the exact cache (raw/armmask/state) the 3-path loadep reads.
PROTO = np.load(REPO / "temp/armmask/arm_prototypes.npz")["proto"]; THR = 0.6; P = 16
OUT = REPO / "temp/crave_30hz_kaimix"; FC = OUT / "feat_cache"; SRC = OUT / "sources.json"
EXIST = REPO / "temp/crave_30hz_mine/feat_cache"   # 已有 83 个 kai0_base 30Hz 特征(复用)
DAGGER_OFFSET = 3_000_000
# TODO(crave-lib): kai0_dagger is not in the dataset registry; kept as explicit paths.
DATASETS = {"kai0_base": REPO / "kai0/data/Task_A/kai0_base",
            "kai0_dagger": REPO / "kai0/data/Task_A/kai0_dagger"}
N_BASE_NEW = 90    # 新增 kai0_base(不含已有83)
N_DAGGER = 80      # 新增 kai0_dagger
SEED = 20260617


def eps_of(dskey):
    ds = DATASETS[dskey]
    return sorted(int(json.loads(l).get("episode_index")) for l in open(ds / "meta/episodes.jsonl"))


def build_tasks():
    """返回 [(cache_idx, dskey, real_ep)] 全量(确定性采样) + sources dict。"""
    reuse = sorted(int(p.stem[2:]) for p in EXIST.glob("ep*.npz"))   # 83 个 kai0_base, cache_idx=real
    rng = np.random.RandomState(SEED)
    base_all = [e for e in eps_of("kai0_base") if e not in set(reuse)]
    base_new = sorted(rng.choice(base_all, min(N_BASE_NEW, len(base_all)), replace=False).tolist())
    dag_all = eps_of("kai0_dagger")
    dag_new = sorted(rng.choice(dag_all, min(N_DAGGER, len(dag_all)), replace=False).tolist())
    tasks, sources = [], {}
    for e in reuse:    tasks.append((e, "kai0_base", e)); sources[str(e)] = ["kai0_base", e]
    for e in base_new: tasks.append((e, "kai0_base", e)); sources[str(e)] = ["kai0_base", e]
    for e in dag_new:  ci = DAGGER_OFFSET + e; tasks.append((ci, "kai0_dagger", e)); sources[str(ci)] = ["kai0_dagger", e]
    return tasks, sources, reuse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--shard", type=int, nargs=2, default=None)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    FC.mkdir(parents=True, exist_ok=True)
    tasks, sources, reuse = build_tasks()

    if a.prep:
        for e in reuse:   # 复用已有 83(复制, 保持纯 kai0_base)
            dst = FC / f"ep{e}.npz"
            if not dst.exists(): shutil.copy(EXIST / f"ep{e}.npz", dst)
        SRC.write_text(json.dumps(sources, indent=0))
        nb = sum(1 for _, k, _ in tasks if k == "kai0_base"); nd = sum(1 for _, k, _ in tasks if k == "kai0_dagger")
        print(f"[prep] 复用 {len(reuse)} kai0_base; 总任务 {len(tasks)} = {nb} kai0_base + {nd} kai0_dagger; sources.json 已写", flush=True)
        return

    todo = [t for t in tasks if not (FC / f"ep{t[0]}.npz").exists()]
    if a.shard: i, n = a.shard; todo = todo[i::n]
    print(f"[extract] 本shard 待提 {len(todo)} eps (stride={a.stride})", flush=True)
    import torch, av
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    enc = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    proto_t = torch.from_numpy(PROTO).float().to(dev)
    csz = {k: json.load(open(DATASETS[k] / "meta/info.json"))["chunks_size"] for k in DATASETS}

    def feats(imgs):
        raw, arm = [], []
        with torch.no_grad():
            for b in range(0, len(imgs), 32):
                batch = imgs[b:b + 32]
                px = proc(images=batch, return_tensors="pt").to(dev)
                toks = enc(**px).last_hidden_state[:, 1:]
                raw.append(torch.nn.functional.normalize(toks.mean(1), dim=-1).cpu().numpy())
                tn = torch.nn.functional.normalize(toks, dim=-1); sim = (tn @ proto_t.T).max(-1).values
                om = []
                for im in batch:
                    rgb = im.reshape(P, 14, P, 14, 3).mean((1, 3)) / 255.0
                    hsv = np.array([[colorsys.rgb_to_hsv(*rgb[i, j]) for j in range(P)] for i in range(P)])
                    om.append(((hsv[..., 0] > 0.02) & (hsv[..., 0] < 0.12) & (hsv[..., 1] > 0.4) & (hsv[..., 2] > 0.25)).reshape(-1))
                om = torch.from_numpy(np.stack(om)).to(dev)
                keep = (~((sim > THR) | om)).float().unsqueeze(-1)
                emb = (toks * keep).sum(1) / keep.sum(1).clamp(min=8)
                arm.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(raw), np.concatenate(arm)

    done = 0
    for ci, dskey, ep in todo:
        ds = DATASETS[dskey]; cs = csz[dskey]
        mp4 = ds / f"videos/chunk-{ep // cs:03d}/observation.images.top_head/episode_{ep:06d}.mp4"
        pq = ds / f"data/chunk-{ep // cs:03d}/episode_{ep:06d}.parquet"
        try:
            imgs = []; c = av.open(str(mp4))
            for j, f in enumerate(c.decode(video=0)):
                if j % a.stride: continue
                imgs.append(kai0.crop224(f.to_ndarray(format="rgb24")))
            c.close()
            st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy())[::a.stride]
            r, m = feats(imgs); k = min(len(r), len(st))
            np.savez_compressed(FC / f"ep{ci}.npz", raw=r[:k].astype(np.float32),
                                armmask=m[:k].astype(np.float32), state=st[:k].astype(np.float32))
        except Exception as e:
            print(f"  [skip] {dskey} ep{ep}: {type(e).__name__} {str(e)[:60]}", flush=True); continue
        done += 1
        if done % 10 == 0: print(f"  {done}/{len(todo)}", flush=True)
    print(f"[extract] shard done, {done} extracted", flush=True); print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
