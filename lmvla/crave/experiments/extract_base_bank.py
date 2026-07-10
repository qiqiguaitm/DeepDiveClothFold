#!/usr/bin/env python
"""抽取 DINOv3-base(768D pooled)特征 bank(对齐最终 kai0 方案的编码器).

复用 crave.data loader(resolve_dataset/list_eps/load_ep → 224 RGB + state)+ load_encoder('dinov3-base')。
输出 temp/<ds>_dinov3base/{index.npz(E/FR/T/n), shard_0.npz(gidx/feat/valid)}, 与 *_dinov3h 同格式。
用法(单卡/分片): python extract_base_bank.py <ds> [gpu] [ep_csv]
  ds ∈ {kai0,vis,coffee,xvla};给 ep_csv 则只抽该子集(多卡分片用),否则全量。
"""
import sys, os, numpy as np, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from crave.config import resolve_dataset
from crave.data.loaders import list_eps, load_ep_native
from crave.encoders import load_encoder
REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[3]))
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def main(ds, gpu=0, ep_csv=None):
    cfg = resolve_dataset(ds)
    import os as _os
    if _os.environ.get("ROOT_OVERRIDE"): cfg = type(cfg)(**{**cfg.__dict__, "root": _os.environ["ROOT_OVERRIDE"]}) if hasattr(cfg,"__dict__") else cfg
    enc = load_encoder("dinov3-base", device="cuda")
    eps = [int(x) for x in ep_csv.split(",")] if ep_csv else list_eps(cfg)
    t0 = time.time(); E=[]; FR=[]; FE=[]
    for k, e in enumerate(eps):
        try:
            _o = load_ep_native(cfg, e); f224, state = _o[0], _o[1]
        except Exception as ex:
            print(f"[g{gpu}] ep{e} skip ({ex})", flush=True); continue
        if len(f224) < 5: continue
        feat = np.asarray(enc.encode_pooled(f224)).astype(np.float16); n = len(feat)
        E += [e]*n; FR += list(range(n)); FE.append(feat)
        if (k+1) % 25 == 0: print(f"[g{gpu}] {k+1}/{len(eps)} ep{e} n{n} · {(time.time()-t0)/60:.1f}min", flush=True)
    out = REPO / "temp/xvla_extract_base" if ep_csv else REPO / f"temp/{ds}_dinov3base"
    out.mkdir(parents=True, exist_ok=True)
    if ep_csv:
        np.savez(out / f"part_{gpu}.npz", E=np.array(E,np.int64), FR=np.array(FR,np.int64), feat=np.concatenate(FE))
    else:
        Ea=np.array(E,np.int64); FRa=np.array(FR,np.int64); feat=np.concatenate(FE)
        np.savez(out/"index.npz", E=Ea, FR=FRa, T=(FRa/30.0).astype(np.float32), n=np.int64(len(Ea)))
        np.savez(out/"shard_0.npz", gidx=np.arange(len(Ea),dtype=np.int64), feat=feat, valid=np.ones(len(Ea),bool))
    print(f"[g{gpu}] DONE {ds} {len(E)} frames ({(time.time()-t0)/60:.1f}min) -> {out}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 0, sys.argv[3] if len(sys.argv)>3 else None)
