#!/usr/bin/env python3
"""Validate the FLASH draft-head attachment seam on a REAL kai0 pi05 model + GPU.

This is the R1-b de-risking step before writing the full speculative state machine:
it proves the draft head attaches to the actual VLM prefix embeddings of a loaded
pi05 checkpoint and measures the headline FLASH quantity -- draft-forward latency
vs a full flow-matching sample_actions call (the potential speedup ceiling).

It does NOT touch any existing inference path: it loads the model via the normal
`create_trained_policy`, then *hooks* `embed_prefix` to capture a real prefix
during one ordinary `policy.infer()` call (so we reuse the real transform/
preprocess pipeline instead of reimplementing it). The draft head is brand-new
(random or VLM-layer0 warm-started) -- we are testing the seam + latency, not
accuracy (accuracy needs distillation = R1-c).

Run (GPU, patched-transformers venv):
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090/bin/python \
    train_scripts/kai/eval/spec_draft_attach_probe.py \
    --config pi05_pytorch_a_new_pure_200 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_others/pytorch_pure200_step50000 \
    --asset-id a_new_pure_200
"""

from __future__ import annotations

import argparse
import time
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
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-warmstart", action="store_true", help="skip VLM-layer0 warm-start of draft block")
    args = ap.parse_args()

    import jax
    from openpi.models import model as _model
    from openpi.policies import policy_config as pc
    from openpi.training import checkpoints as ck
    from openpi.training import config as tc
    from openpi.models_pytorch.draft import DraftChunkHead

    ckpt = Path(args.ckpt).resolve()
    train_cfg = tc.get_config(args.config)
    norm_stats = ck.load_norm_stats(ckpt / "assets", args.asset_id)
    print(f"[load] config={args.config} ckpt={ckpt.name} asset={args.asset_id}", flush=True)
    t0 = time.time()
    policy = pc.create_trained_policy(train_cfg, ckpt, norm_stats=norm_stats)
    print(f"[load] policy ready {time.time() - t0:.1f}s", flush=True)

    model = policy._model  # PI0Pytorch  (noqa: SLF001 - probe)
    device = policy._pytorch_device  # noqa: SLF001
    is_pt = policy._is_pytorch_model  # noqa: SLF001
    if not is_pt:
        print("[err] this probe requires a PyTorch checkpoint (model.safetensors)")
        return 2
    ah, ad = int(model.config.action_horizon), int(model.config.action_dim)
    print(f"[model] pi05={getattr(model, 'pi05', '?')} action_horizon={ah} action_dim={ad} device={device}")

    # ---- build a real Observation through the normal transform pipeline ----
    # (replicates Policy.infer front-half; avoids the torch.compile'd sample_actions
    #  wrapper which CUDA-graph-crashes on this venv, and lets our embed_prefix hook
    #  fire by calling the EAGER class method type(model).sample_actions.)
    obs = _synth_obs(state_dim=14)
    inputs = policy._input_transform(obs)  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    observation = _model.Observation.from_dict(inputs)

    cap = {}
    orig_embed_prefix = model.embed_prefix

    def hooked(images, img_masks, lang_tokens, lang_masks):
        out = orig_embed_prefix(images, img_masks, lang_tokens, lang_masks)
        cap["prefix_embs"], cap["prefix_pad_masks"], cap["prefix_att_masks"] = out
        return out

    model.embed_prefix = hooked

    # full-pipeline latency via the EAGER sample_actions (uncompiled class method).
    full_ms = []
    full_actions = None
    with torch.no_grad():
        for i in range(args.warmup + args.iters):
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            t = time.time()
            acts = type(model).sample_actions(model, device, observation, num_steps=10)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            if i >= args.warmup:
                full_ms.append((time.time() - t) * 1000.0)
            full_actions = acts
    model.embed_prefix = orig_embed_prefix  # restore immediately

    full_actions = full_actions.float().cpu().numpy()[0]
    pe = cap["prefix_embs"]
    pad = cap["prefix_pad_masks"]
    att = cap["prefix_att_masks"]
    img_dim = int(pe.shape[-1])
    print(f"[prefix] embs={tuple(pe.shape)} dtype={pe.dtype} pad={tuple(pad.shape)} img_dim={img_dim}")
    print(f"[full]   actions={full_actions.shape}  latency P50={np.median(full_ms):.1f}ms "
          f"min={np.min(full_ms):.1f} max={np.max(full_ms):.1f}")

    # ---- build draft head matching the VLM, optionally warm-start from layer0 ----
    vlm_lm = model.paligemma_with_expert.paligemma.language_model
    gemma_cfg = vlm_lm.config
    layer0 = vlm_lm.layers[0]
    draft = DraftChunkHead(
        img_dim=img_dim, chunk_m=ah, out_dim=ad, use_state_token=False, gemma_config=gemma_cfg
    ).to(device=device, dtype=pe.dtype)
    draft.eval()
    warm_ok = "skipped"
    if not args.no_warmstart:
        try:
            draft.init_from_vlm_layer(layer0)
            warm_ok = "OK"
        except Exception as ex:  # noqa: BLE001
            warm_ok = f"FAILED: {type(ex).__name__}: {ex}"
    print(f"[draft] hidden={draft.hidden_size} heads={draft.num_heads} kv={draft.num_kv_heads} "
          f"head_dim={draft.head_dim} warm_start={warm_ok}")

    # ---- draft forward latency on the captured real prefix ----
    draft_ms = []
    with torch.no_grad():
        for i in range(args.warmup + args.iters):
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            t = time.time()
            da = draft(prefix_embs=pe, prefix_pad_masks=pad, prefix_att_masks=att)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            if i >= args.warmup:
                draft_ms.append((time.time() - t) * 1000.0)
    da_np = da.float().cpu().numpy()
    print(f"[draft]  out={tuple(da.shape)} finite={bool(np.isfinite(da_np).all())} "
          f"latency P50={np.median(draft_ms):.2f}ms min={np.min(draft_ms):.2f} max={np.max(draft_ms):.2f}")
    print(f"[speed]  full P50={np.median(full_ms):.1f}ms  draft P50={np.median(draft_ms):.2f}ms  "
          f"=> draft is {np.median(full_ms) / max(np.median(draft_ms), 1e-6):.1f}x faster (single-call, untrained)")
    print(f"[note]   draft output is UNTRAINED -> values not meaningful yet (distill = R1-c). "
          f"draft|mean|={np.mean(np.abs(da_np)):.3f} full|mean|={np.mean(np.abs(full_actions)):.3f}")
    print("[OK] draft-head attaches to the real pi05 prefix and runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
