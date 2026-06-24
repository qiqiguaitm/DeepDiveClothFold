#!/usr/bin/env python3
"""R1-d falsification control: does sample_from_prefix REJECT a garbage draft?

The R1-d eval reported 50/50 acceptance on every holdout frame. Before trusting
that, prove the verify path actually discriminates: feed an UNTRAINED DraftChunkHead
through the exact same `sample_from_prefix` on the exact same cached holdout shards.
If acceptance collapses (high radius, fallback), the trained 50/50 is real signal,
not a trivial "x0_hat == x0_draft" path bug.

Run:
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090/bin/python \
    train_scripts/kai/eval/spec_draft_r1d_control.py \
    --config pi05_pytorch_a_new_pure_200 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_others/pytorch_pure200_step50000 \
    --asset-id a_new_pure_200 \
    --cache-dir /data1/tmp/spec_cache_r1d_pure200 \
    --trained /tmp/draft_r1d_pure200.pt --n 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file as _load_safetensors
import torch


def _eval_draft(sampler, draft, cache_dir, hold_man, n, mdtype, device, ah, action_dim):
    sampler.draft = draft.to(mdtype)
    torch.manual_seed(0)
    accs, dists, falls = [], [], 0
    seen = 0
    with torch.no_grad():
        for sh in hold_man["shards"]:
            t = _load_safetensors(str(Path(cache_dir) / sh["path"]))
            for i in range(t["prefix_embs"].shape[0]):
                if seen >= n:
                    break
                pe = t["prefix_embs"][i : i + 1].to(device, mdtype)
                ppad = t["prefix_pad"][i : i + 1].to(device)
                patt = t["prefix_att"][i : i + 1].to(device)
                st = t["robot_state"][i : i + 1].to(device, mdtype)
                out = sampler.sample_from_prefix(pe, ppad, patt, st, noise=None, last_gripper=None)
                accs.append(int(out["accepted_prefix_len"].item()))
                dists.append(float(out["radius_dist"].min(dim=1).values.mean().item()))
                falls += int(bool(out["used_full_fallback"]))
                seen += 1
            if seen >= n:
                break
    accs = np.asarray(accs)
    return accs, float(np.mean(dists)), falls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.3)
    args = ap.parse_args()

    from openpi.models_pytorch.draft import DraftChunkHead
    from openpi.models_pytorch.spec_pi0_pytorch import SpecArgs
    from openpi.models_pytorch.spec_pi0_pytorch import SpeculativeSampler
    from openpi.policies import policy_config as pc
    from openpi.training import checkpoints as ck
    from openpi.training import config as tc

    ckpt = Path(args.ckpt).resolve()
    train_cfg = tc.get_config(args.config)
    norm_stats = ck.load_norm_stats(ckpt / "assets", args.asset_id)
    policy = pc.create_trained_policy(train_cfg, ckpt, norm_stats=norm_stats)
    model = policy._model  # noqa: SLF001
    device = next(model.parameters()).device
    mdtype = next(model.parameters()).dtype
    ah, ad = int(model.config.action_horizon), int(model.config.action_dim)

    spec_args = SpecArgs(chunk_m=ah, tau_radius=args.tau, max_exec_steps=ah, full_num_steps=10)
    sampler = SpeculativeSampler(model, None, spec_args)

    manifest = json.loads((Path(args.cache_dir) / "manifest.json").read_text())
    hold_man = manifest["splits"]["holdout"]
    img_dim = int(manifest["feature_dim"])
    vlm_lm = model.paligemma_with_expert.paligemma.language_model

    # --- A: untrained draft (random action_head + queries; VLM-layer0 warm-start only) ---
    untr = DraftChunkHead(img_dim=img_dim, chunk_m=ah, out_dim=ad, use_state_token=False,
                          gemma_config=vlm_lm.config).to(device=device, dtype=torch.float32)
    untr.init_from_vlm_layer(vlm_lm.layers[0])
    untr.eval()
    a_u, d_u, f_u = _eval_draft(sampler, untr, args.cache_dir, hold_man, args.n, mdtype, device, ah, ad)

    # --- B: trained draft (the R1-d artifact) ---
    blob = torch.load(args.trained, map_location=device)
    tr = DraftChunkHead(img_dim=img_dim, chunk_m=ah, out_dim=ad, use_state_token=False,
                        gemma_config=vlm_lm.config).to(device=device, dtype=torch.float32)
    tr.load_state_dict(blob["state_dict"])
    tr.eval()
    a_t, d_t, f_t = _eval_draft(sampler, tr, args.cache_dir, hold_man, args.n, mdtype, device, ah, ad)

    print("\n========== R1-d CONTROL (same cache, same sample_from_prefix path) ==========")
    print(f"  n={len(a_u)} frames  tau={args.tau}")
    print(f"  UNTRAINED draft : mean_accept={a_u.mean():.1f}/{ah}  mean_radius={d_u:.4f}  "
          f"zero-accept={int((a_u <= 0).sum())}/{len(a_u)}  fallback={f_u}/{len(a_u)}")
    print(f"  TRAINED   draft : mean_accept={a_t.mean():.1f}/{ah}  mean_radius={d_t:.4f}  "
          f"zero-accept={int((a_t <= 0).sum())}/{len(a_t)}  fallback={f_t}/{len(a_t)}")
    verdict = "DISCRIMINATES (50/50 is real)" if (a_u.mean() < a_t.mean() - 5 and d_u > d_t) \
        else "SUSPECT: path may not discriminate"
    print(f"  => verify path {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
