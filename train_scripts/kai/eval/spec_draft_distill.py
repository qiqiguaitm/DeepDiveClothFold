#!/usr/bin/env python3
"""Distill a DraftChunkHead for a kai0 pi05 checkpoint (R1-c), kai0-native.

Same idea as FLASH's enc_cache + spec_draft_train, but collapsed into one
self-contained, smaller script (no sharded safetensors cache / manifest machinery
— we just hold the prefix cache in CPU RAM, which is plenty for a few hundred
frames). It is ADDITIVE: trains only the new draft head, freezes/never-touches
the loaded pi05 model, writes the draft to its own file.

Pipeline:
  1. CACHE  — for sampled frames of val episodes: build the real observation,
              run model.embed_prefix -> prefix tensors (CPU), and a TEACHER target
              = the model's own full flow-matching chunk x0 under a FIXED noise
              (so the draft learns to mimic the Action Expert -> high acceptance).
  2. TRAIN  — regress DraftChunkHead(prefix) -> teacher chunk (step-weighted Huber),
              backbone frozen, Adam, warm-started from VLM layer 0.
  3. EVAL   — load the trained draft into SpeculativeSampler and report the REAL
              accepted_prefix_len / radius distance on held-out frames (this is the
              quantity R1-d / R5 need).

Run (GPU, patched venv):
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090/bin/python \
    train_scripts/kai/eval/spec_draft_distill.py \
    --config pi05_pytorch_a_new_pure_200 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_others/pytorch_pure200_step50000 \
    --asset-id a_new_pure_200 \
    --val kai0/data/Task_A/self_built/A_new_pure_200_val \
    --out /tmp/draft_pure200.pt --frames-per-ep 8 --epochs 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read_video(path: Path, n: int) -> np.ndarray:
    import av

    c = av.open(str(path))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    out = []
    for fr in c.decode(s):
        out.append(fr.to_ndarray(format="rgb24"))
        if len(out) >= n:
            break
    c.close()
    a = np.stack(out[:n], 0)
    if a.shape[0] < n:
        a = np.concatenate([a, np.repeat(a[-1:], n - a.shape[0], 0)], 0)
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="/tmp/draft_distilled.pt")
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--frames-per-ep", type=int, default=8)
    ap.add_argument("--max-eps", type=int, default=16)
    ap.add_argument("--holdout-eps", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--teacher-steps", type=int, default=10)
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
    val = Path(args.val).resolve()
    train_cfg = tc.get_config(args.config)
    norm_stats = ck.load_norm_stats(ckpt / "assets", args.asset_id)
    policy = pc.create_trained_policy(train_cfg, ckpt, norm_stats=norm_stats)
    model = policy._model  # noqa: SLF001
    device = policy._pytorch_device  # noqa: SLF001
    mdtype = next(model.parameters()).dtype
    ah, ad = int(model.config.action_horizon), int(model.config.action_dim)
    print(f"[model] pi05={getattr(model, 'pi05', '?')} H={ah} action_dim={ad} device={device} dtype={mdtype}")

    spec_args = SpecArgs(chunk_m=ah, tau_radius=args.tau, max_exec_steps=ah, full_num_steps=args.teacher_steps)
    sampler = SpeculativeSampler(model, None, spec_args)  # draft attached later

    eps = [json.loads(l) for l in (val / "meta" / "episodes.jsonl").read_text().splitlines()][: args.max_eps]
    cams = ("top_head", "hand_left", "hand_right")
    fixed_noise = model.sample_noise((1, ah, ad), device)  # same noise for teacher + verify => fair

    def build_obs(ei, k, vid, state):
        imgs = {c: vid[c][k] for c in cams}
        obs = {"images": imgs, "state": state[k], "prompt": args.prompt}
        inputs = policy._input_transform(obs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    # ---- 1. CACHE prefix + teacher chunk ----
    cache = {"train": [], "holdout": []}
    with torch.no_grad():
        for ix, ep in enumerate(eps):
            ei, L = ep["episode_index"], ep["length"]
            df = __import__("pyarrow.parquet", fromlist=["x"]).read_table(
                val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet"
            ).to_pandas()
            state = np.stack([np.asarray(x) for x in df["observation.state"]]).astype(np.float32)
            vid = {c: read_video(val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", L)
                   for c in cams}
            ks = np.linspace(0, L - 1, args.frames_per_ep).astype(int)
            split = "holdout" if ix < args.holdout_eps else "train"
            for k in ks:
                obs = build_obs(ei, k, vid, state)
                pe, ppad, patt, st, pkv = sampler._prefill(obs)  # noqa: SLF001
                teacher = sampler._full_denoise(st, ppad, pkv, fixed_noise)  # (1,H,ad)  noqa: SLF001
                cache[split].append({
                    "prefix_embs": pe.squeeze(0).cpu(), "prefix_pad": ppad.squeeze(0).cpu(),
                    "prefix_att": patt.squeeze(0).cpu(), "target": teacher.squeeze(0).float().cpu(),
                })
            print(f"  cached ep{ei} ({split}) [{ix + 1}/{len(eps)}]", flush=True)
    ntr, nho = len(cache["train"]), len(cache["holdout"])
    img_dim = int(cache["train"][0]["prefix_embs"].shape[-1])
    print(f"[cache] train={ntr} holdout={nho} frames; img_dim={img_dim}")

    # ---- 2. TRAIN draft head ----
    vlm_lm = model.paligemma_with_expert.paligemma.language_model
    draft = DraftChunkHead(
        img_dim=img_dim, chunk_m=ah, out_dim=ad, use_state_token=False, gemma_config=vlm_lm.config,
    ).to(device=device, dtype=torch.float32)
    draft.init_from_vlm_layer(vlm_lm.layers[0])
    draft.train()
    opt = torch.optim.Adam(draft.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # step weights: emphasize the executed prefix (exp decay over horizon)
    step_w = torch.tensor([0.97 ** i for i in range(ah)], device=device).view(1, ah, 1)

    tr = cache["train"]
    idx = np.arange(ntr)
    best_loss, best_state = float("inf"), None
    for ep_i in range(args.epochs):
        np.random.default_rng(ep_i).shuffle(idx)
        tot = 0.0
        for b0 in range(0, ntr, args.batch):
            bi = idx[b0 : b0 + args.batch]
            pe = torch.stack([tr[i]["prefix_embs"] for i in bi]).to(device, torch.float32)
            ppad = torch.stack([tr[i]["prefix_pad"] for i in bi]).to(device)
            patt = torch.stack([tr[i]["prefix_att"] for i in bi]).to(device)
            tgt = torch.stack([tr[i]["target"] for i in bi]).to(device, torch.float32)
            pred = draft(prefix_embs=pe, prefix_pad_masks=ppad, prefix_att_masks=patt)
            loss = (torch.nn.functional.huber_loss(pred, tgt, reduction="none", delta=0.1) * step_w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), max_norm=1.0)
            opt.step()
            tot += float(loss) * len(bi)
        sched.step()
        epoch_loss = tot / ntr
        if epoch_loss < best_loss:  # save BEST, not last (final epochs can diverge)
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in draft.state_dict().items()}
        if ep_i % 25 == 0 or ep_i == args.epochs - 1:
            print(f"  epoch {ep_i:4d}  train_huber={epoch_loss:.5f}  (best={best_loss:.5f})", flush=True)

    if best_state is not None:
        draft.load_state_dict(best_state)
        print(f"[best] restored draft @ train_huber={best_loss:.5f}")
    draft.eval()
    torch.save({"state_dict": draft.state_dict(), "img_dim": img_dim, "chunk_m": ah, "out_dim": ad,
                "config": args.config, "ckpt": str(ckpt)}, args.out)
    print(f"[save] draft -> {args.out}")

    # ---- 3. EVAL real acceptance on holdout ----
    sampler.draft = draft.to(mdtype)
    accs, dists, falls = [], [], 0
    with torch.no_grad():
        # rebuild holdout obs is costly; reuse cached prefix by injecting via override-free path:
        # run a fresh draft+verify directly on cached prefix tensors.
        for item in cache["holdout"]:
            pe = item["prefix_embs"][None].to(device, mdtype)
            ppad = item["prefix_pad"][None].to(device)
            patt = item["prefix_att"][None].to(device)
            x0_draft = sampler._draft_x0(pe, ppad, patt, None, fixed_noise)  # noqa: SLF001
            # verify-from-draft directly (mirror sample() core) using teacher as the verify target proxy:
            tgt = item["target"][None].to(device, torch.float32)
            # radius of draft vs teacher (this is exactly what acceptance would compute vs verify)
            from openpi.models_pytorch.spec_pi0_pytorch import _compute_radius_prefix_acceptance
            acc, dist = _compute_radius_prefix_acceptance(
                x0_draft=x0_draft, x0_hat=tgt[:, None], tau_radius=args.tau, dist_dims=12, eval_h=ah,
            )
            accs.append(int(acc.item()))
            dists.append(float(dist.mean().item()))
            if int(acc.item()) <= 0:
                falls += 1
    print("\n========== R1-c DISTILL RESULT (holdout) ==========")
    print(f"  frames={nho}  mean accepted_prefix_len={np.mean(accs):.1f}/{ah}  "
          f"median={np.median(accs):.0f}  max={np.max(accs)}")
    print(f"  mean radius(draft vs teacher)={np.mean(dists):.4f}  (tau={args.tau})  zero-accept frames={falls}/{nho}")
    print(f"  => draft is {'USEFUL (nonzero acceptance)' if np.mean(accs) > 1 else 'still weak (more data/epochs)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
