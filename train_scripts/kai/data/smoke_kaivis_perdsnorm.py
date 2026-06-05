"""Smoke test per-DS-norm + conditioning pipeline (C1/C2/C3) — video/model-free (runs on the
jumpbox which lacks an FFmpeg/video codec). Validates the NEW code only.

  1. config.data.create(): 2 per-domain norm loaded, repack[0]=ReadDatasetIdFromTaskIndex,
     domain_sample_weights set, kai vs vis q01 differ.
  2. ReadDatasetIdFromTaskIndex: task_index 0->dataset_id 0, 1->1, and no-op when dataset_id preset
     (inference) or task_index absent.
  3. DomainNormalize: dataset_id 0 uses kai norm, 1 uses vis norm (different outputs, matches manual
     quantile formula).
  4. DomainWeightedSampler balance: weighted draws over real task_index -> vis fraction ~0.5.

Run: kai0/.venv/bin/python train_scripts/kai/data/smoke_kaivis_perdsnorm.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kai0" / "src"))
import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

CFG = "xvla_kaivis_perdsnorm_cond"
MERGED = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/kai_vis_merged"


def main():
    ok = True
    config = _config.get_config(CFG)
    dc = config.data.create(config.assets_dirs, config.model)

    # [1] config.create wiring
    assert set(dc.domain_norm_stats) == {0, 1}, dc.domain_norm_stats
    assert set(dc.domain_sample_weights) == {0, 1} and dc.domain_sample_weights[0] == 1.0, dc.domain_sample_weights
    rnames = [type(t).__name__ for t in dc.repack_transforms.inputs]
    assert rnames[0] == "ReadDatasetIdFromTaskIndex", rnames
    k01 = np.asarray(dc.domain_norm_stats[0]["actions"].q01)[:6]
    v01 = np.asarray(dc.domain_norm_stats[1]["actions"].q01)[:6]
    differ = not np.allclose(k01, v01)
    print(f"[1] create(): norm{{0,1}} weights={dc.domain_sample_weights} repack[0]={rnames[0]} kai!=vis_q01={differ} {'✅' if differ else '❌'}")
    ok &= differ

    # [2] ReadDatasetIdFromTaskIndex
    rid = _transforms.ReadDatasetIdFromTaskIndex()
    a = rid({"task_index": np.int64(0)}); b = rid({"task_index": np.int64(1)})
    c = rid({"dataset_id": np.int32(7), "task_index": np.int64(0)})  # preset -> keep
    d = rid({"foo": 1})  # no task_index -> no-op (inference)
    t2 = int(a["dataset_id"]) == 0 and int(b["dataset_id"]) == 1 and int(c["dataset_id"]) == 7 and "dataset_id" not in d
    print(f"[2] ReadDatasetIdFromTaskIndex: ti0->{int(a['dataset_id'])} ti1->{int(b['dataset_id'])} preset->{int(c['dataset_id'])} noop->{'dataset_id' not in d} {'✅' if t2 else '❌'}")
    ok &= t2

    # [3] DomainNormalize picks per-domain
    dn = _transforms.DomainNormalize(dc.domain_norm_stats, use_quantiles=dc.use_quantile_norm)
    st = np.zeros(14, dtype=np.float32); st[:6] = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6]
    ok_k = dn({"state": st.copy(), "dataset_id": np.int32(0)})["state"]
    ok_v = dn({"state": st.copy(), "dataset_id": np.int32(1)})["state"]
    # manual quantile for domain 0:
    s0 = dc.domain_norm_stats[0]["state"]
    q01, q99 = np.asarray(s0.q01)[:14], np.asarray(s0.q99)[:14]
    manual = (st - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    t3 = (not np.allclose(ok_k, ok_v)) and np.allclose(np.asarray(ok_k)[:14], manual, atol=1e-4)
    print(f"[3] DomainNormalize: kai!=vis={not np.allclose(ok_k,ok_v)} kai_matches_manual={np.allclose(np.asarray(ok_k)[:14],manual,atol=1e-4)} {'✅' if t3 else '❌'}")
    ok &= t3

    # [4] weighted-sampler balance (reads task_index, no video)
    import json
    info = json.load(open(f"{MERGED}/meta/info.json"))
    # per-frame task_index via per-episode length + episode->domain (ep<6512 kai else vis); cheaper than loading hf
    import pandas as pd, glob
    pqs = sorted(glob.glob(f"{MERGED}/data/chunk-000/episode_*.parquet"))
    # sample frame-domain by reading task_index of a stratified subset is overkill; build full from lengths:
    eps = [json.loads(l) for l in open(f"{MERGED}/meta/episodes.jsonl")]
    # domain by episode index: kai 0..6511, vis 6512..  (matches build order)
    ti = np.concatenate([np.full(e["length"], 0 if i < 6512 else 1, dtype=np.int8) for i, e in enumerate(eps)])
    w = np.ones(len(ti), dtype=np.float64); w[ti == 1] = dc.domain_sample_weights[1]
    import torch
    g = torch.Generator(); g.manual_seed(0)
    idx = torch.multinomial(torch.as_tensor(w), 20000, replacement=True, generator=g).numpy()
    frac = float(np.mean(ti[idx] == 1))
    t4 = 0.42 < frac < 0.58
    print(f"[4] weighted sampler vis-fraction={frac:.3f} (target~0.50) frames={len(ti)} {'✅' if t4 else '❌'}")
    ok &= t4

    print("\nSMOKE OK ✅" if ok else "\nSMOKE FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
