"""校验:我的 fresh encode(load_encoder dinov3-h + crop224 + L2)能否复现 kai0 缓存特征。
若 cosine≈1 → 流水线一致, v4 低置信是真 domain shift; 若不一致 → encode bug。
顺带:同 v4 帧 vs kai0_base 同进度帧 的视觉对比图。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crave.config import REPO, resolve_dataset
from crave.encoders import load_encoder
from crave.data import kai0
from crave.utils import L2

OUTD = REPO / "temp/crave_full_dinov3h"
idx = np.load(OUTD / "index.npz", allow_pickle=True); E, FR, T = idx["E"], idx["FR"], idx["T"].astype(float)
s0 = np.load(OUTD / "shard_0.npz"); s1 = np.load(OUTD / "shard_1.npz")
N = len(E); feat = np.zeros((N, 1280), np.float16)
for s in (s0, s1): feat[s["gidx"]] = s["feat"]

cfg = resolve_dataset("kai0_base")
enc = load_encoder("dinov3-h")
# 取几个缓存帧重编码比对
rng_gi = [100, 50000, 150000, 250000, 334000]
imgs = []; cached = []
for gi in rng_gi:
    fm = kai0.grab_ep(cfg, int(E[gi]), [int(FR[gi])])
    if int(FR[gi]) in fm:
        imgs.append(fm[int(FR[gi])]); cached.append(L2(feat[gi].astype(np.float32)[None])[0])
fresh = L2(enc.encode_pooled(imgs))
cached = np.array(cached)
cos = np.sum(fresh * cached, 1)
print("缓存帧 fresh-encode vs cached cosine:", np.round(cos, 4).tolist(), flush=True)
print(f"  → {'流水线一致(domain shift 为真)' if cos.mean()>0.95 else 'encode 不一致(有 bug)'}", flush=True)

# kai0_base 帧内部 dmin 基线(同一批帧到自己 milestone)对照已知 tau0.44; 这里给 kai0 帧两两距离尺度
print(f"\nkai0 缓存帧间随机距离(尺度参考): ", flush=True)
fa = L2(feat[np.random.default_rng(0).choice(N, 200, replace=False)].astype(np.float32))
dd = np.linalg.norm(fa[:100] - fa[100:200], axis=1)
print(f"  kai0 随机帧对 L2 距离 p50={np.percentile(dd,50):.3f}", flush=True)
