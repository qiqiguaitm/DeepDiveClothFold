#!/usr/bin/env python3
"""Mechanics validation of SpeculativeSampler on a real kai0 pi05 model (GPU).

Proves the full speculative round runs end-to-end on a loaded checkpoint and that
each branch behaves correctly -- WITHOUT needing a distilled draft yet (R1-c):

  A. UNTRAINED draft  -> draft chunk is garbage -> radius rejects -> accepted~0
     -> full fallback fires, output shape (1,H,action_dim) finite.
  B. ORACLE draft (x0_draft := the model's own full-denoise chunk) -> verify
     accepts a long prefix -> accepted_prefix_len high, NO fallback. This isolates
     and validates the accept/stitch/gripper path on real model outputs.

Also reports the speculative *signals* (accepted_prefix_len, radius distance,
gripper flags, draft/verify ms) that R3/R5 will consume.

Run (GPU, patched venv):
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090/bin/python \
    train_scripts/kai/eval/spec_sampler_mechanics_probe.py \
    --config pi05_pytorch_a_new_pure_200 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_others/pytorch_pure200_step50000 \
    --asset-id a_new_pure_200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _synth_obs(cams=("top_head", "hand_left", "hand_right"), h=480, w=640, state_dim=14):
    rng = np.random.default_rng(0)
    images = {c: rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for c in cams}
    state = rng.standard_normal(state_dim).astype(np.float32) * 0.1
    return {"images": images, "state": state, "prompt": "Flatten and fold the cloth."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--tau", type=float, default=0.3)
    args = ap.parse_args()

    import jax

    from openpi.models import model as _model
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
    device = policy._pytorch_device  # noqa: SLF001
    ah, ad = int(model.config.action_horizon), int(model.config.action_dim)
    print(f"[model] pi05={getattr(model, 'pi05', '?')} H={ah} action_dim={ad} device={device}")

    # build observation through the real transform pipeline
    inputs = policy._input_transform(_synth_obs())  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    observation = _model.Observation.from_dict(inputs)

    # draft head matching the VLM (warm-started)
    vlm_lm = model.paligemma_with_expert.paligemma.language_model
    draft = DraftChunkHead(
        img_dim=int(vlm_lm.config.hidden_size), chunk_m=ah, out_dim=ad,
        use_state_token=False, gemma_config=vlm_lm.config,
    ).to(device=device, dtype=next(model.parameters()).dtype)
    draft.init_from_vlm_layer(vlm_lm.layers[0])
    draft.eval()

    spec_args = SpecArgs(chunk_m=ah, tau_radius=args.tau, max_exec_steps=ah)
    sampler = SpeculativeSampler(model, draft, spec_args)

    fixed_noise = model.sample_noise((1, ah, ad), device)

    # ---- A: untrained draft -> expect ~0 accepted, fallback fires ----
    print("\n== A. untrained draft (expect reject + fallback) ==")
    rA = sampler.sample(observation, noise=fixed_noise)
    accA = int(rA["accepted_prefix_len"].item())
    print(f"  actions={tuple(rA['actions'].shape)} finite={bool(torch.isfinite(rA['actions']).all())}")
    print(f"  accepted_prefix_len={accA}  used_full_fallback={rA['used_full_fallback']}")
    print(f"  radius_dist mean={rA['radius_dist'].mean().item():.3f}  "
          f"draft_ms={rA['draft_ms']:.2f} verify_ms={rA['verify_ms']:.2f}")
    okA = (tuple(rA["actions"].shape) == (1, ah, ad)) and bool(torch.isfinite(rA["actions"]).all())
    okA = okA and rA["used_full_fallback"] and accA <= 2

    # ---- B: oracle draft = model's own full denoise -> expect high acceptance ----
    print("\n== B. oracle draft = full-denoise chunk (expect high acceptance, no fallback) ==")
    _pe, ppad, _patt, state, pkv = sampler._prefill(observation)  # noqa: SLF001
    x0_full = sampler._full_denoise(state, ppad, pkv, fixed_noise)  # noqa: SLF001
    rB = sampler.sample(observation, noise=fixed_noise, x0_draft_override=x0_full)
    accB = int(rB["accepted_prefix_len"].item())
    evalh = int(rB["eval_h"])
    print(f"  accepted_prefix_len={accB} / eval_h={evalh}  used_full_fallback={rB['used_full_fallback']}")
    print(f"  radius_dist mean={rB['radius_dist'].mean().item():.4f}  "
          f"gripper_verify_stop={bool(rB['gripper_verify_stop'].item())} "
          f"gripper_switch_cut={bool(rB['gripper_switch_cut'].item())}")
    # oracle draft should match verify closely -> accept much more than the untrained draft
    okB = accB > max(accA, 5)

    print("\n==== mechanics: "
          f"A(reject+fallback)={'OK' if okA else 'FAIL'}  "
          f"B(oracle-accept)={'OK' if okB else 'FAIL'} ====")
    return 0 if (okA and okB) else 1


if __name__ == "__main__":
    raise SystemExit(main())
