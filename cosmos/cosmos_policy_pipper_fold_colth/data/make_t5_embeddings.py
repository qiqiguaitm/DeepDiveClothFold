#!/usr/bin/env python3
"""
Precompute the T5 text embedding for the single cloth-fold command and save it as
<data_dir>/t5_embeddings.pkl in the exact format ALOHADataset expects
({command_str: torch.Tensor (1,512,1024) bf16}).

We bypass save_aloha_t5_text_embeddings.py (which would load the whole video dataset just
to collect unique_commands) and embed the command string directly. The command MUST equal
what aloha_dataset.py parses from the task dir name "fold_cloth" -> "fold cloth".
"""
import argparse
import os

from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings, save_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="ALOHADataset data_dir; t5_embeddings.pkl written here")
    ap.add_argument("--commands", nargs="+", default=["fold cloth"])
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    emb = generate_t5_embeddings(args.commands)
    for k, v in emb.items():
        print(f"  '{k}': {tuple(v.shape)} {v.dtype}")
    save_embeddings(emb, args.data_dir)


if __name__ == "__main__":
    main()
