#!/usr/bin/env python
"""构建去重紧凑 milestone-target 存储(供 gf3, 免同步 39GB 全特征)。
unique(ep,tgt_fi) 的目标特征 → target_compact.npz (ep, tgt_fi, feat[Nu,256,768] fp16)。
用法: srpo python p1_build_target_compact.py
"""
import os, argparse, numpy as np

_D = "/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data"
PAIRS = f"{_D}/libero_milestone/pairs.npz"
FEAT  = f"{_D}/libero_dinov3base"
OUT   = f"{_D}/libero_milestone/target_compact.npz"

def main():
    global PAIRS, FEAT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=PAIRS); ap.add_argument("--feat", default=FEAT); ap.add_argument("--out", default=OUT)
    a = ap.parse_args(); PAIRS, FEAT, OUT = a.pairs, a.feat, a.out
    P = np.load(PAIRS)
    uniq = sorted(set(zip(P["cur_ep"].tolist(), P["tgt_fi"].tolist())))
    print(f"[uniq] {len(uniq)} 个目标 (ep,tgt_fi)", flush=True)
    # 按 episode 分组, 每 ep 只加载一次特征
    from collections import defaultdict
    by_ep = defaultdict(list)
    for ep, tfi in uniq:
        by_ep[ep].append(tfi)
    eps_arr, tfi_arr, feats = [], [], []
    for i, (ep, tfis) in enumerate(sorted(by_ep.items())):
        g = np.load(f"{FEAT}/ep{ep}.npz")["grid"]          # [N,256,768] fp16
        for tfi in tfis:
            eps_arr.append(ep); tfi_arr.append(tfi); feats.append(g[tfi].astype(np.float16))
        if i % 200 == 0:
            print(f"  {i}/{len(by_ep)} eps", flush=True)
    feat = np.stack(feats, 0)                              # [Nu,256,768] fp16
    np.savez(OUT, ep=np.array(eps_arr, np.int32), tgt_fi=np.array(tfi_arr, np.int32), feat=feat)
    print(f"[save] {OUT}  feat={feat.shape} {feat.nbytes/1e9:.2f}GB", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
