"""Offline eval for the P1 go/no-go ablation: mean validation action flow-loss.

Runs forward_step (no backward) over N held-out windows and reports mean action
velocity-MSE. Compare a P1 checkpoint (pretrained frozen trunk + trained joint heads)
against a random-trunk control: markedly lower val a_loss => the tau0 prior transfers
to joint space => proceed to P2; on par => fall back to GigaWorld joint-14.

Eval data needs cached latents. visrobot01_val has only t5 (no vae_latent); precompute
with GigaWorld's encoder first:
  cd giga_world_policy && python -m scripts.wam_pipeline.compute_latents --emb visrobot01_val --stride 4
(or point --val_path at any LatentJointDataset dir that has vae_latent/).
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finetune.model_joint import build_joint_wanmodel  # noqa: E402
from finetune.train_tau0 import TauFlowTrainer, ACTION_CHUNK  # noqa: E402
from finetune.data_joint import LatentJointDataset  # noqa: E402

CKPT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/tau0_wm/checkpoints/tau-0-wm"
ASSETS = os.path.join(ROOT, "finetune", "assets")


@torch.no_grad()
def evaluate(val_path, stats, ckpt=None, load_pretrained=True, n=200, seeds=4, device="cuda"):
    ds = LatentJointDataset(val_path, stats, ACTION_CHUNK, embed_id=0)
    model, _ = build_joint_wanmodel(action_in_dim=14, load_pretrained=load_pretrained,
                                    dtype=torch.bfloat16, device=device, verbose=False)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[eval] loaded ckpt {ckpt}: applied {len(sd)} tensors (missing {len(miss)})")
    model.eval()
    tr = TauFlowTrainer(model, torch.device(device))
    losses = []
    rng = np.random.RandomState(0)
    idxs = rng.randint(0, len(ds), size=min(n, len(ds)))
    for j, i in enumerate(idxs):
        b = ds[i]
        z0 = b["video_latent"].to(device, torch.bfloat16)
        ref = b["ref"].to(device, torch.bfloat16)
        a0 = b["action"].unsqueeze(0).to(device, torch.bfloat16)
        state = b["state"].unsqueeze(0).to(device, torch.bfloat16)
        ctx = b["t5"].to(device, torch.bfloat16)
        # average over several noise/sigma draws for a stable estimate
        per = []
        for _ in range(seeds):
            _, parts = tr.forward_step(z0, a0, state, ctx, ref=ref)
            per.append(parts["a_loss"])
        losses.append(float(np.mean(per)))
    m = float(np.mean(losses))
    print(f"[eval] windows={len(idxs)} seeds={seeds}  mean val action-loss = {m:.5f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_path", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_val")
    ap.add_argument("--stats", default=f"{ASSETS}/statistics_visrobot01.json")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--random_trunk", action="store_true", help="control: random-init trunk")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    evaluate(args.val_path, args.stats, ckpt=args.ckpt or None,
             load_pretrained=not args.random_trunk, n=args.n)
