#!/usr/bin/env python3
"""Consolidate a cosmos-policy DCP checkpoint (iter_XXXX/model/) into a single .pt
whose flat {net.*: tensor} dict matches the released Cosmos-Policy-ALOHA-Predict2-2B.pt,
so the eval/deploy loader (which expects a .pt) can read it.

Usage: python dcp_to_pt.py <iter_dir>  ->  writes <iter_dir>/model.pt
"""
import argparse, os, torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("iter_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model_dcp = os.path.join(args.iter_dir, "model")
    assert os.path.isdir(model_dcp), f"no model/ DCP dir under {args.iter_dir}"
    out = args.out or os.path.join(args.iter_dir, "model.pt")
    print(f"consolidating {model_dcp} -> {out}")
    dcp_to_torch_save(model_dcp, out)
    d = torch.load(out, map_location="cpu", weights_only=False)
    print(f"done: {len(d)} keys, e.g. {list(d.keys())[:3]}")


if __name__ == "__main__":
    main()
