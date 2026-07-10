#!/usr/bin/env python
"""补齐 xvla 全 168 ep 的 DINOv3-H pooled 特征(多卡并行 worker).

cam_high(JPEG bytes)→ cv2.imdecode → resize256 → DINOv3-H encode_pooled → 1280D fp16。
稳态 ~111 fps/GPU;2×A100 交错分 84 ep/卡 ≈ 26min。

用法(2 卡并行):
  EPS=$(ls xvla/data/xvla_soft_fold/<dir>/episode_*.hdf5 | sed 's/.*episode_//;s/.hdf5//' | sort -n)
  E0=$(echo "$EPS"|awk 'NR%2==1'|paste -sd,); E1=$(echo "$EPS"|awk 'NR%2==0'|paste -sd,)
  CUDA_VISIBLE_DEVICES=0 python extract_xvla_dinov3h.py 0 "$E0" &
  CUDA_VISIBLE_DEVICES=1 python extract_xvla_dinov3h.py 1 "$E1" &
  # 完成后合并见文件末 __main__ 之后的 merge 片段(或 lmvla/lmwm 侧的 bank writer)
输出: temp/xvla_extract/part_{gpu}.npz(E/FR/feat)→ 合并成 temp/xvla_dinov3h/{index,shard_0}.npz(168ep)。
"""
import sys, os, numpy as np, cv2, h5py, time
from pathlib import Path
sys.path.insert(0, "/home/tim/workspace/deepdive_kai0/lmvla/crave/src")
from crave.encoders import load_encoder

REPO = Path("/home/tim/workspace/deepdive_kai0")
XDIR = REPO / "xvla/data/xvla_soft_fold/0707_11pm_stage_1_stage2new_new_cam_very_slow"


def main(g, eps):
    enc = load_encoder("dinov3-h", device="cuda"); t0 = time.time()
    Es = []; FRs = []; FE = []
    for ci, ep in enumerate(eps):
        fp = XDIR / f"episode_{ep}.hdf5"
        if not fp.exists():
            print(f"[g{g}] MISSING ep{ep}", flush=True); continue
        f = h5py.File(fp, "r"); cam = f["observations/images/cam_high"]; nfr = len(cam); feats = []
        for k in range(0, nfr, 256):
            raw = cam[k:min(k + 256, nfr)]
            imgs = [cv2.resize(np.ascontiguousarray(cv2.imdecode(np.frombuffer(raw[i], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]), (256, 256))
                    for i in range(len(raw))]
            feats.append(np.asarray(enc.encode_pooled(np.stack(imgs))).astype(np.float16))
        fe = np.concatenate(feats); Es += [ep] * nfr; FRs += list(range(nfr)); FE.append(fe); f.close()
        if ci % 10 == 0:
            print(f"[g{g}] {ci+1}/{len(eps)} ep{ep} {nfr}fr · {(time.time()-t0)/60:.1f}min", flush=True)
    out = REPO / "temp/xvla_extract"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"part_{g}.npz", E=np.array(Es, np.int64), FR=np.array(FRs, np.int64), feat=np.concatenate(FE))
    print(f"[g{g}] DONE {len(Es)} frames ({(time.time()-t0)/60:.1f}min)", flush=True)


def merge():
    """合并两卡分片 → temp/xvla_dinov3h/(按 (ep,帧) 排序;旧 bank 备份 *_80ep_bak.npz)."""
    R = REPO / "temp"
    p0 = np.load(R / "xvla_extract/part_0.npz"); p1 = np.load(R / "xvla_extract/part_1.npz")
    E = np.concatenate([p0["E"], p1["E"]]); FR = np.concatenate([p0["FR"], p1["FR"]]); feat = np.concatenate([p0["feat"], p1["feat"]])
    o = np.lexsort((FR, E)); E, FR, feat = E[o], FR[o], feat[o]; n = len(E)
    out = R / "xvla_dinov3h"; import shutil
    if (out / "index.npz").exists():
        shutil.copy(out / "index.npz", out / "index_80ep_bak.npz"); shutil.copy(out / "shard_0.npz", out / "shard_0_80ep_bak.npz")
    np.savez(out / "index.npz", E=E.astype(np.int64), FR=FR.astype(np.int64), T=(FR / 30.0).astype(np.float32), n=np.int64(n))
    np.savez(out / "shard_0.npz", gidx=np.arange(n, dtype=np.int64), feat=feat, valid=np.ones(n, bool))
    print(f"merged {len(np.unique(E))} eps / {n} frames -> temp/xvla_dinov3h/")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge()
    else:
        main(int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")])
