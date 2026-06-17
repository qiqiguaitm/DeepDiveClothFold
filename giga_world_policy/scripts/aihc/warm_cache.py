#!/usr/bin/env python3
"""Pre-build (warm) the HuggingFace datasets arrow cache for a training config, SINGLE-PROCESS.

Why: after parquet edits the HF arrow cache is invalid. When a distributed job launches,
all N ranks (across nodes, shared PFS) race to rebuild it -> partial/incomplete arrow files
-> FileNotFoundError(...incomplete...) / IndexError in check_timestamps_sync -> job dies.

This warms the cache via the EXACT trainer load path (giga LeRobotDataset.load(entry).open()),
so the produced cache hash matches what the job computes, and the job just reads it.

Used by run_train_aihc.sh on NODE_RANK 0 before `accelerate launch` (other nodes wait on a
sentinel). Also runnable standalone:
  python -m scripts.aihc.warm_cache --config world_action_model.configs.visrobot01_gwp_abs_v4.config
"""
import argparse
import importlib
import sys


def _resolve_config(dotted):
    # dotted like "world_action_model.configs.visrobot01_gwp_abs_v4.config"
    mod_path, _, attr = dotted.rpartition(".")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="dotted path to config dict (…module.config)")
    args = ap.parse_args()

    cfg = _resolve_config(args.config)
    from giga_datasets.datasets.lerobot_dataset import LeRobotDataset

    # gather train (and any test) data entries
    entries = []
    dl = cfg.get("dataloaders", {})
    for split in ("train", "test"):
        s = dl.get(split, {}) or {}
        doc = s.get("data_or_config")
        if isinstance(doc, list):
            entries += doc
        elif doc is not None:
            entries.append(doc)

    seen = set()
    for ent in entries:
        # dedupe by data_path (the 4× same-path copies share one cache)
        dp = ent.get("data_path") if isinstance(ent, dict) else None
        if dp in seen:
            continue
        seen.add(dp)
        print(f"[warm] opening {dp} ...", flush=True)
        ds = LeRobotDataset.load(ent)
        ds.open()
        n = len(ds)
        print(f"[warm] OK {dp}: {n} rows", flush=True)
        ds.close()
    print("[warm] done", flush=True)


if __name__ == "__main__":
    main()
