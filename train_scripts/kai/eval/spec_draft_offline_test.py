#!/usr/bin/env python3
"""Offline (CPU, no-checkpoint) tests for the FLASH speculative port to kai0.

Validates the two additive modules in isolation, so we can trust the building
blocks before wiring them to a real pi05 checkpoint on GPU (R1.4):

  1. DraftChunkHead forward -> correct chunk shape (B, chunk_m, out_dim), with
     and without the explicit robot-state token.
  2. _compute_radius_prefix_acceptance numerics: identical draft==verify accepts
     the whole eval window; a perturbation at step k truncates acceptance to k;
     a perturbation only in a GRIPPER dim is ignored by the radius (grippers are
     excluded from the distance and handled by the phase gate instead).
  3. Dual-gripper switch detection + truncation: an L-gripper flip at step 5 and
     an R-gripper flip at step 8 are both detected, and the accepted prefix is
     cut at the EARLIEST switch.
  4. _stitch_radius_prefix_output: draft prefix + verified tail compose correctly.

Run (from repo root):
  kai0/.venv/bin/python train_scripts/kai/eval/spec_draft_offline_test.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import sys

import torch

from openpi.models_pytorch.draft import DraftChunkHead
from openpi.models_pytorch.spec_pi0_pytorch import (
    KAI0_GRIPPER_DIMS,
    _compute_radius_prefix_acceptance,
    _detect_verify_gripper_switch_any_k,
    _stitch_radius_prefix_output,
    _truncate_accepted_prefix_on_gripper_switch,
)

torch.manual_seed(0)
torch.set_num_threads(4)
PASS, FAIL = 0, 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {extra}")


def test_draft_shapes() -> None:
    print("\n== 1. DraftChunkHead forward shapes ==")
    B, S, H, M, OUT = 2, 40, 2048, 50, 14
    prefix = torch.randn(B, S, H)
    pad = torch.ones(B, S, dtype=torch.bool)
    att = torch.zeros(B, S, dtype=torch.bool)

    # with state token (kai0 raw joint state = 14-D)
    head = DraftChunkHead(img_dim=H, chunk_m=M, out_dim=OUT, state_dim=14, use_state_token=True).eval()
    with torch.no_grad():
        a = head(
            prefix_embs=prefix, prefix_pad_masks=pad, prefix_att_masks=att, robot_state=torch.randn(B, 14)
        )
    check("with-state output shape (B,M,OUT)", tuple(a.shape) == (B, M, OUT), str(tuple(a.shape)))
    check("output is finite", bool(torch.isfinite(a).all()))
    check("output dtype float32", a.dtype == torch.float32)

    # without state token (pi05 folds state into language prefix)
    head2 = DraftChunkHead(img_dim=H, chunk_m=M, out_dim=OUT, use_state_token=False).eval()
    with torch.no_grad():
        a2 = head2(prefix_embs=prefix, prefix_pad_masks=pad, prefix_att_masks=att)
    check("no-state output shape (B,M,OUT)", tuple(a2.shape) == (B, M, OUT), str(tuple(a2.shape)))

    # wrong state dim should raise
    raised = False
    try:
        head(prefix_embs=prefix, prefix_pad_masks=pad, prefix_att_masks=att, robot_state=torch.randn(B, 32))
    except ValueError:
        raised = True
    check("wrong state_dim raises ValueError", raised)


def test_radius_acceptance() -> None:
    print("\n== 2. radius prefix acceptance numerics ==")
    B, H, D, K = 1, 12, 14, 2
    draft = torch.randn(B, H, D)
    # identical verify -> accept whole eval window
    hat = draft[:, None].repeat(1, K, 1, 1)
    acc, dist = _compute_radius_prefix_acceptance(
        x0_draft=draft, x0_hat=hat, tau_radius=0.1, dist_dims=12, eval_h=H, gripper_dims=KAI0_GRIPPER_DIMS
    )
    check("identical draft/verify accepts full window", int(acc.item()) == H, f"acc={int(acc.item())}")

    # perturb arm dim at step 6 beyond tau -> accept exactly 6
    hat2 = draft[:, None].repeat(1, K, 1, 1)
    hat2[0, 0, 6, 0] += 10.0  # arm dim 0 (non-gripper), step 6, member 0
    acc2, _ = _compute_radius_prefix_acceptance(
        x0_draft=draft, x0_hat=hat2, tau_radius=0.1, dist_dims=12, eval_h=H, gripper_dims=KAI0_GRIPPER_DIMS
    )
    check("arm perturbation at step6 -> accept 6", int(acc2.item()) == 6, f"acc={int(acc2.item())}")

    # perturb ONLY a gripper dim -> radius ignores it (grippers excluded)
    hat3 = draft[:, None].repeat(1, K, 1, 1)
    hat3[0, 0, 3, 6] += 10.0  # gripper dim 6, step 3
    hat3[0, 1, 4, 13] += 10.0  # gripper dim 13, step 4
    acc3, _ = _compute_radius_prefix_acceptance(
        x0_draft=draft, x0_hat=hat3, tau_radius=0.1, dist_dims=12, eval_h=H, gripper_dims=KAI0_GRIPPER_DIMS
    )
    check("gripper-only perturbation ignored by radius", int(acc3.item()) == H, f"acc={int(acc3.item())}")

    # min-over-K: member 1 fails earlier -> overall = the stricter (earlier)
    hat4 = draft[:, None].repeat(1, K, 1, 1)
    hat4[0, 0, 9, 1] += 10.0  # member 0 fails at 9
    hat4[0, 1, 4, 1] += 10.0  # member 1 fails at 4
    acc4, _ = _compute_radius_prefix_acceptance(
        x0_draft=draft, x0_hat=hat4, tau_radius=0.1, dist_dims=12, eval_h=H, gripper_dims=KAI0_GRIPPER_DIMS
    )
    check("min-over-K acceptance = stricter member (4)", int(acc4.item()) == 4, f"acc={int(acc4.item())}")


def test_dual_gripper_phase_gate() -> None:
    print("\n== 3. dual-gripper switch detect + truncate ==")
    B, H, D = 1, 12, 14
    thr = 0.5
    # build a clean chunk both grippers start "closed" (0.0) and stay, then flips
    chunk = torch.zeros(B, H, D)
    chunk[0, 5:, 6] = 1.0  # L gripper opens at step 5
    chunk[0, 8:, 13] = 1.0  # R gripper opens at step 8
    gprev = torch.zeros(B, len(KAI0_GRIPPER_DIMS))  # both previously closed

    # detection on a K=1 verify view
    trig = _detect_verify_gripper_switch_any_k(
        x0_hat=chunk[:, None], gripper_prev=gprev, gripper_switch_threshold=thr, eval_h=H,
        gripper_dims=KAI0_GRIPPER_DIMS,
    )
    check("dual-gripper switch detected", bool(trig.item()))

    # truncation: accepted prefix initially full -> cut at earliest switch (step 5)
    acc = torch.tensor([H], dtype=torch.int64)
    trunc, cut = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=chunk, accepted_prefix_len=acc, gripper_prev=gprev, gripper_switch_threshold=thr,
        gripper_dims=KAI0_GRIPPER_DIMS,
    )
    check("truncate to earliest switch (step 5)", int(trunc.item()) == 5, f"trunc={int(trunc.item())}")
    check("cut_mask set", bool(cut.item()))

    # no switch (both grippers already open and stay open) -> no cut
    chunk_open = torch.ones(B, H, D)
    gprev_open = torch.ones(B, len(KAI0_GRIPPER_DIMS))
    trunc2, cut2 = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=chunk_open, accepted_prefix_len=acc, gripper_prev=gprev_open, gripper_switch_threshold=thr,
        gripper_dims=KAI0_GRIPPER_DIMS,
    )
    check("no switch -> prefix unchanged", int(trunc2.item()) == H and not bool(cut2.item()))

    # only R-arm flips at step 3 -> truncate to 3
    chunk_r = torch.zeros(B, H, D)
    chunk_r[0, 3:, 13] = 1.0
    trunc3, _ = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=chunk_r, accepted_prefix_len=acc, gripper_prev=gprev, gripper_switch_threshold=thr,
        gripper_dims=KAI0_GRIPPER_DIMS,
    )
    check("single-arm (R) switch truncates to 3", int(trunc3.item()) == 3, f"trunc={int(trunc3.item())}")


def test_stitch() -> None:
    print("\n== 4. prefix stitch ==")
    B, H, D = 2, 12, 14
    draft = torch.full((B, H, D), 1.0)
    tail = torch.full((B, H, D), 2.0)
    acc = torch.tensor([4, 0], dtype=torch.int64)
    out = _stitch_radius_prefix_output(x0_draft=draft, x0_tail=tail, accepted_prefix_len=acc)
    ok0 = bool((out[0, :4] == 1.0).all() and (out[0, 4:] == 2.0).all())
    ok1 = bool((out[1] == 2.0).all())  # zero accepted -> all tail
    check("row0 first4 draft, rest tail", ok0)
    check("row1 zero-accept -> all tail", ok1)


def main() -> int:
    test_draft_shapes()
    test_radius_acceptance()
    test_dual_gripper_phase_gate()
    test_stitch()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
