"""Speculative-inference acceptance logic for kai0 (ported from Realtime-VLA FLASH).

This module holds the *pure tensor* helpers of FLASH's speculative loop:
  * radius prefix acceptance        -- accept the longest draft prefix whose
                                       distance to the verified action stays <= tau
  * gripper-switch detection         -- never speculate across a gripper open/close
                                       event (the precision-critical phase)
  * gripper-switch truncation        -- cut an accepted prefix at the first switch
  * prefix stitching                 -- draft prefix + verified tail -> final chunk
plus the `SpecArgs` config dataclass.

It is ADDITIVE: importing it touches no existing kai0 inference code, and the
functions are side-effect-free. The heavier `SpecPI0Pytorch(PI0Pytorch)` inference
state machine is added in a later increment (R1.4) at the bottom of this file.

Key generalization vs upstream FLASH (LIBERO single-arm, gripper hardcoded at
dim 6):  kai0 is **dual-arm 14-D** with grippers at dims **(6, 13)**. Every
gripper-aware function takes an explicit `gripper_dims` tuple and OR-combines the
switch logic across all of them, and the radius distance is computed over the
*non-gripper* dims rather than LIBERO's "first 6" shortcut. This dual-gripper
phase gate is also the foundation for research thread R2 (gripper-phase strong
vision verification) in flash_future_research.md.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch

# kai0 dual-arm joint layout: arm-L joints 0-5, gripper 6; arm-R joints 7-12, gripper 13.
KAI0_GRIPPER_DIMS: tuple[int, ...] = (6, 13)
KAI0_ARM_DIMS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


@dataclasses.dataclass
class SpecArgs:
    """Inference-time speculative config. Edit fields here, not env vars."""

    # ---- chunking ----
    chunk_m: int = 50
    max_exec_steps: int = 12

    # ---- verify ----
    # Verify timesteps: near-terminal / low-noise (smaller t => closer to x0 in
    # x_t = t*x1 + (1-t)*x0). Multiple entries => K-way verify ensemble.
    t_list: tuple[float, ...] = (0.10, 0.05)
    tau_radius: float = 0.3
    dist_dims: int = 12  # kai0 has 12 arm joints (LIBERO used 6); radius excludes grippers
    dist_dim_idx: tuple[int, ...] | None = None  # explicit non-gripper dims; None => auto
    verify_mode: Literal["radius", "random"] = "radius"
    random_accept_prob: float = 0.5
    random_seed: int = 0

    # ---- draft ----
    draft_history_len: int = 6

    # ---- gripper phase gate (dual-arm aware) ----
    gripper_dims: tuple[int, ...] = KAI0_GRIPPER_DIMS
    gripper_switch_threshold: float = 0.0
    enable_gripper_verify: bool = True
    enable_gripper_post_verify: bool = True
    gripper_full_window: int = 1

    # ---- full-pipeline fallback ----
    full_fallback: bool = True
    full_num_steps: int = 10
    force_full_each_round: bool = False
    periodic_full_every_n_draft_rounds: int = 0


def _resolve_dist_dim_idx(
    *, action_dim: int, gripper_dims: tuple[int, ...], dist_dims: int, dist_dim_idx: tuple[int, ...] | None
) -> list[int]:
    """Indices over which the radius distance is computed (grippers excluded)."""
    if dist_dim_idx is not None:
        idx = [int(i) for i in dist_dim_idx if 0 <= int(i) < int(action_dim)]
    else:
        grip = {int(g) for g in gripper_dims}
        idx = [i for i in range(int(action_dim)) if i not in grip]
    if not idx:
        raise ValueError(f"no distance dims left after excluding grippers={gripper_dims} from action_dim={action_dim}")
    if dist_dims and dist_dims > 0:
        idx = idx[: int(dist_dims)]
    return idx


def _accepted_prefix_len_from_mask(accept_mask: torch.Tensor) -> torch.Tensor:
    """Per-step accept decisions (B,H) -> accepted prefix length (B,)."""
    if accept_mask.ndim != 2:
        raise ValueError(f"expected accept_mask to be (B,H), got shape={tuple(accept_mask.shape)}")
    prefix_ok = torch.cumprod(accept_mask.to(dtype=torch.int64), dim=1)
    return prefix_ok.sum(dim=1, dtype=torch.int64)


def _gripper_switch_mask_1d(
    *, prev_values: torch.Tensor, curr_values: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Boolean switch mask where the gripper crosses `threshold` between prev->curr."""
    t = float(threshold)
    return ((prev_values < t) & (curr_values >= t)) | ((prev_values >= t) & (curr_values < t))


def _truncate_accepted_prefix_on_gripper_switch(
    *,
    x0_out: torch.Tensor,
    accepted_prefix_len: torch.Tensor,
    gripper_prev: torch.Tensor | None,
    gripper_switch_threshold: float,
    gripper_dims: tuple[int, ...] = KAI0_GRIPPER_DIMS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut the accepted prefix at the first gripper open/close event (any arm).

    `gripper_prev` is (B, G) -- the last executed gripper value for each of the G
    gripper dims (or (B,) for single gripper, broadcast). Returns the truncated
    prefix length and a (B,) cut_mask flagging which samples were cut.
    """
    if x0_out.ndim != 3:
        raise ValueError(f"expected x0_out to be (B,H,D), got shape={tuple(x0_out.shape)}")
    b, h, d = x0_out.shape
    accepted_prefix_len = accepted_prefix_len.to(device=x0_out.device, dtype=torch.int64)
    cut_mask = torch.zeros((b,), device=x0_out.device, dtype=torch.bool)
    grip = [g for g in gripper_dims if 0 <= int(g) < int(d)]
    if not grip or gripper_prev is None:
        return accepted_prefix_len, cut_mask

    gp = gripper_prev.to(device=x0_out.device, dtype=torch.float32)
    if gp.ndim == 1:
        gp = gp[:, None].expand(b, len(grip))
    if int(gp.shape[0]) != b or int(gp.shape[1]) != len(grip):
        raise ValueError(f"gripper_prev must be (B,{len(grip)}) or (B,), got {tuple(gripper_prev.shape)}")

    step_idx = torch.arange(h, device=x0_out.device, dtype=torch.int64)[None, :]
    active_mask = step_idx < accepted_prefix_len[:, None]
    any_switch = torch.zeros((b, h), device=x0_out.device, dtype=torch.bool)
    for gi, gdim in enumerate(grip):
        prev_values = torch.cat([gp[:, gi : gi + 1], x0_out[:, :-1, gdim].to(dtype=torch.float32)], dim=1)
        curr_values = x0_out[:, :, gdim].to(dtype=torch.float32)
        sw = _gripper_switch_mask_1d(
            prev_values=prev_values, curr_values=curr_values, threshold=gripper_switch_threshold
        )
        any_switch = any_switch | (sw & active_mask)

    cut_mask = any_switch.any(dim=1)
    first_switch_idx = any_switch.to(dtype=torch.int64).argmax(dim=1)
    truncated_prefix_len = torch.where(cut_mask, first_switch_idx, accepted_prefix_len)
    return truncated_prefix_len.to(dtype=torch.int64), cut_mask


def _detect_verify_gripper_switch_any_k(
    *,
    x0_hat: torch.Tensor,
    gripper_prev: torch.Tensor | None,
    gripper_switch_threshold: float,
    eval_h: int,
    gripper_dims: tuple[int, ...] = KAI0_GRIPPER_DIMS,
) -> torch.Tensor:
    """(B,) trigger mask: does any verify member predict a gripper switch in the eval window?"""
    if x0_hat.ndim != 4:
        raise ValueError(f"expected x0_hat to be (B,K,H,D), got shape={tuple(x0_hat.shape)}")
    b, k, h, d = x0_hat.shape
    trigger_mask = torch.zeros((b,), device=x0_hat.device, dtype=torch.bool)
    grip = [g for g in gripper_dims if 0 <= int(g) < int(d)]
    if not grip or gripper_prev is None:
        return trigger_mask

    gp = gripper_prev.to(device=x0_hat.device, dtype=torch.float32)
    if gp.ndim == 1:
        gp = gp[:, None].expand(b, len(grip))
    if int(gp.shape[0]) != b or int(gp.shape[1]) != len(grip):
        raise ValueError(f"gripper_prev must be (B,{len(grip)}) or (B,), got {tuple(gripper_prev.shape)}")

    eval_h2 = int(min(h, max(1, int(eval_h))))
    for gi, gdim in enumerate(grip):
        prev_values = torch.cat(
            [
                gp[:, gi][:, None, None].expand(-1, k, -1),
                x0_hat[:, :, : max(0, eval_h2 - 1), gdim].to(dtype=torch.float32),
            ],
            dim=2,
        )
        curr_values = x0_hat[:, :, :eval_h2, gdim].to(dtype=torch.float32)
        sw = _gripper_switch_mask_1d(
            prev_values=prev_values, curr_values=curr_values, threshold=gripper_switch_threshold
        )
        trigger_mask = trigger_mask | sw.any(dim=2).any(dim=1)
    return trigger_mask


def _compute_radius_prefix_acceptance(
    *,
    x0_draft: torch.Tensor,
    x0_hat: torch.Tensor,
    tau_radius: float,
    dist_dims: int,
    eval_h: int,
    gripper_dims: tuple[int, ...] = KAI0_GRIPPER_DIMS,
    dist_dim_idx: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Longest draft prefix accepted under the radius rule, min over K verify members.

    Returns (accepted_prefix_len (B,), per-(B,K,H) normalized distance).
    Distance is over the non-gripper dims (grippers handled by the phase gate).
    """
    if x0_draft.ndim != 3:
        raise ValueError(f"expected x0_draft to be (B,H,D), got shape={tuple(x0_draft.shape)}")
    if x0_hat.ndim != 4:
        raise ValueError(f"expected x0_hat to be (B,K,H,D), got shape={tuple(x0_hat.shape)}")
    b, h, d = int(x0_draft.shape[0]), int(x0_draft.shape[1]), int(x0_draft.shape[2])
    if int(x0_hat.shape[0]) != b or int(x0_hat.shape[2]) != h or int(x0_hat.shape[3]) != d:
        raise ValueError(f"x0_hat must be (B,K,H,D)={(b, 'K', h, d)}, got shape={tuple(x0_hat.shape)}")

    eval_h2 = int(min(h, max(1, int(eval_h))))
    idx = _resolve_dist_dim_idx(
        action_dim=d, gripper_dims=gripper_dims, dist_dims=int(dist_dims), dist_dim_idx=dist_dim_idx
    )
    idx_t = torch.tensor(idx, device=x0_draft.device, dtype=torch.long)
    eval_d = len(idx)

    draft_sel = x0_draft[:, :eval_h2].index_select(2, idx_t)  # (B,Hc,eval_d)
    hat_sel = x0_hat[:, :, :eval_h2].index_select(3, idx_t)  # (B,K,Hc,eval_d)
    diff = hat_sel - draft_sel[:, None]
    norm_d = torch.tensor(float(eval_d), device=x0_draft.device, dtype=torch.float32).sqrt().clamp_min(1.0)
    dist = torch.linalg.vector_norm(diff, ord=2, dim=3).to(dtype=torch.float32) / norm_d  # (B,K,Hc)

    ok = dist <= float(tau_radius)
    prefix_mask = ok.to(dtype=torch.int64).cumprod(dim=2)
    prefix_len_k = prefix_mask.sum(dim=2)
    accepted_prefix_len = prefix_len_k.min(dim=1).values.to(dtype=torch.int64)
    return accepted_prefix_len, dist


def _stitch_radius_prefix_output(
    *,
    x0_draft: torch.Tensor,
    x0_tail: torch.Tensor,
    accepted_prefix_len: torch.Tensor,
) -> torch.Tensor:
    """Keep draft up to accepted_prefix_len, verified tail beyond."""
    if x0_draft.ndim != 3:
        raise ValueError(f"expected x0_draft to be (B,H,D), got shape={tuple(x0_draft.shape)}")
    if x0_tail.ndim != 3 or tuple(x0_tail.shape) != tuple(x0_draft.shape):
        raise ValueError(f"x0_tail must match x0_draft shape={tuple(x0_draft.shape)}, got {tuple(x0_tail.shape)}")
    if accepted_prefix_len.ndim != 1 or int(accepted_prefix_len.shape[0]) != int(x0_draft.shape[0]):
        raise ValueError(
            f"accepted_prefix_len must have shape (B,)={(int(x0_draft.shape[0]),)}, "
            f"got {tuple(accepted_prefix_len.shape)}"
        )
    accepted_prefix_len = accepted_prefix_len.to(device=x0_draft.device, dtype=torch.int64)
    idx = torch.arange(int(x0_draft.shape[1]), device=x0_draft.device, dtype=torch.int64)[None, :]
    accept_mask = (idx < accepted_prefix_len[:, None])[:, :, None]
    return torch.where(accept_mask, x0_draft, x0_tail)


def _should_schedule_full_fallback(
    *,
    full_fallback: bool,
    accepted_prefix_len: torch.Tensor,
    gripper_switch_cut_mask: torch.Tensor | None = None,
) -> bool:
    if not bool(full_fallback):
        return False
    zero_accept = bool((accepted_prefix_len <= 0).any().item())
    switch_cut = bool(gripper_switch_cut_mask is not None and gripper_switch_cut_mask.any().item())
    return zero_accept or switch_cut


class SpeculativeSampler:
    """Additive speculative-inference driver around an existing PI0Pytorch model.

    Wraps (does NOT subclass / mutate) a loaded `PI0Pytorch` plus a trained (or
    probe) `DraftChunkHead`. One `sample()` call performs the FLASH speculative
    round adapted to kai0 pi05:

        prefill VLM KV once  ->  draft proposes a full chunk x0_draft  ->  K-way
        "verify-from-draft" (build x_t=t*noise+(1-t)*x0_draft, one denoise_step
        per t, x0_hat = x_t - t*v_t)  ->  radius prefix acceptance  ->  dual-arm
        gripper-phase gate (force full-verify across any gripper switch)  ->
        stitch accepted draft prefix + verified tail.  Optional full fallback
        (parent's real multi-step denoise) when too little is accepted.

    Everything runs EAGER (the model's torch.compiled `sample_actions` wrapper
    CUDA-graph-crashes on the 5090 verify path; see flash_impl_log.md §2.2).

    The returned dict carries the speculative *signals* (accepted_prefix_len,
    radius distance, gripper flags) that research threads R3 (divergence as
    uncertainty / DAgger trigger) and R5 (acceptance rate as an open-loop probe)
    consume -- not just the action chunk.
    """

    def __init__(self, model, draft, spec_args: SpecArgs | None = None):
        self.model = model
        self.draft = draft
        self.args = spec_args or SpecArgs()
        self.action_horizon = int(model.config.action_horizon)
        self.action_dim = int(model.config.action_dim)
        self.chunk_m = int(self.args.chunk_m)

    # ---- prefill: replicate PI0Pytorch.sample_actions prefix-cache stage (eager) ----
    def _embed_prefix(self, observation):
        """Observation -> (prefix_embs, prefix_pad_masks, prefix_att_masks, state). No KV."""
        model = self.model
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(  # noqa: SLF001
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks, state

    def _prefill_kv(self, prefix_embs, prefix_pad_masks, prefix_att_masks):
        """Build the VLM KV cache from prefix embeddings (eager).

        Split out of `_prefill` so a cached prefix (e.g. R1-d disk shards) can be
        re-prefilled for the real verify path without re-running the vision encoder
        or re-decoding video. `prefix_embs` must already be in the model dtype.
        """
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        model = self.model
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_4d = model._prepare_attention_masks_4d(prefix_att_2d)  # noqa: SLF001
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_key_values

    def _prefill(self, observation):
        prefix_embs, prefix_pad_masks, prefix_att_masks, state = self._embed_prefix(observation)
        pkv = self._prefill_kv(prefix_embs, prefix_pad_masks, prefix_att_masks)
        return prefix_embs, prefix_pad_masks, prefix_att_masks, state, pkv

    def _draft_x0(self, prefix_embs, prefix_pad_masks, prefix_att_masks, state, noise):
        """Run the draft head, lay its chunk into the model's (H, action_dim) space."""
        b, h, d = noise.shape
        rs = state if self.draft.use_state_token else None
        chunk = self.draft(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            robot_state=rs,
        ).to(dtype=torch.float32)  # (B, chunk_m, out_dim)
        x0 = torch.zeros((b, h, d), device=noise.device, dtype=torch.float32)
        m = int(min(self.chunk_m, h))
        od = int(min(self.draft.out_dim, d))
        x0[:, :m, :od] = chunk[:, :m, :od]
        if h > m > 0:  # hold last drafted step for any tail beyond chunk_m
            x0[:, m:, :od] = x0[:, m - 1 : m, :od].expand(b, h - m, od)
        return x0

    @torch.no_grad()
    def sample(self, observation, noise=None, last_gripper=None, x0_draft_override=None):
        """Full path: observation -> prefill (encode+KV) -> speculative round."""
        prefix_embs, prefix_pad_masks, prefix_att_masks, state, pkv = self._prefill(observation)
        return self._spec_core(
            prefix_embs, prefix_pad_masks, prefix_att_masks, state, pkv,
            noise=noise, last_gripper=last_gripper, x0_draft_override=x0_draft_override,
        )

    @torch.no_grad()
    def full_denoise_from_observation(self, observation, noise=None):
        """Eager parent-equivalent multi-step denoise from a raw observation.

        Bypasses the draft/verify entirely: prefill (eager) -> `_full_denoise`. Used as a
        5090-safe catastrophic fallback by serve_policy_flash -- the model's *compiled*
        `sample_actions` CUDA-graph-crashes on the 5090 (flash_impl_log.md §2.2), so the
        server cannot fall back to it; this eager path produces the same clean chunk.
        Returns just the (B, H, action_dim) tensor (no speculative signals).
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks, state, pkv = self._prefill(observation)
        bsize = int(prefix_embs.shape[0])
        if noise is None:
            device = next(self.model.parameters()).device
            noise = self.model.sample_noise((bsize, self.action_horizon, self.action_dim), device)
        return self._full_denoise(state, prefix_pad_masks, pkv, noise)

    @torch.no_grad()
    def sample_from_prefix(
        self, prefix_embs, prefix_pad_masks, prefix_att_masks, state,
        noise=None, last_gripper=None, x0_draft_override=None,
    ):
        """Cache-only path: cached prefix tensors -> rebuild KV -> speculative round.

        This is the REAL verify-from-draft acceptance path used by R1-d eval: it runs
        the same draft + K-way denoise verify + radius/gripper/fallback logic as
        `sample`, but starts from disk-cached prefix embeddings (no vision encoder,
        no video decode). `prefix_embs` is cast to the model dtype for the KV build.
        """
        model = self.model
        mdtype = next(model.parameters()).dtype
        prefix_embs = prefix_embs.to(dtype=mdtype)
        pkv = self._prefill_kv(prefix_embs, prefix_pad_masks, prefix_att_masks)
        return self._spec_core(
            prefix_embs, prefix_pad_masks, prefix_att_masks, state, pkv,
            noise=noise, last_gripper=last_gripper, x0_draft_override=x0_draft_override,
        )

    def _spec_core(
        self, prefix_embs, prefix_pad_masks, prefix_att_masks, state, pkv,
        noise=None, last_gripper=None, x0_draft_override=None,
    ):
        import time as _time

        model = self.model
        device = next(model.parameters()).device
        bsize = int(prefix_embs.shape[0])
        h, d = self.action_horizon, self.action_dim
        if noise is None:
            noise = model.sample_noise((bsize, h, d), device)

        t0 = _time.time()
        if x0_draft_override is not None:
            # Oracle/eval path: caller supplies the draft chunk (e.g. for mechanics
            # validation before a real draft is distilled).
            x0_draft = x0_draft_override.to(device=device, dtype=torch.float32)
        else:
            x0_draft = self._draft_x0(prefix_embs, prefix_pad_masks, prefix_att_masks, state, noise)
        if device.type == "cuda":
            torch.cuda.synchronize()
        draft_ms = (_time.time() - t0) * 1000.0

        # ---- K-way verify-from-draft (sequential; K small) ----
        t1 = _time.time()
        tks = list(self.args.t_list)
        x0_hat_list = []
        for tk in tks:
            tk_t = torch.tensor(float(tk), device=device, dtype=noise.dtype)
            x_t = tk_t * noise + (1.0 - tk_t) * x0_draft.to(noise.dtype)
            v_t = model.denoise_step(state, prefix_pad_masks, pkv, x_t, tk_t.expand(bsize))
            x0_hat_list.append((x_t - tk_t * v_t).to(torch.float32))
        x0_hat = torch.stack(x0_hat_list, dim=1)  # (B, K, H, D)
        if device.type == "cuda":
            torch.cuda.synchronize()
        verify_ms = (_time.time() - t1) * 1000.0

        eval_h = int(min(h, max(1, int(self.args.max_exec_steps))))
        accepted_prefix_len, dist = _compute_radius_prefix_acceptance(
            x0_draft=x0_draft,
            x0_hat=x0_hat,
            tau_radius=float(self.args.tau_radius),
            dist_dims=int(self.args.dist_dims),
            eval_h=eval_h,
            gripper_dims=self.args.gripper_dims,
            dist_dim_idx=self.args.dist_dim_idx,
        )
        x0_tail = x0_hat.mean(dim=1)
        x0_out = _stitch_radius_prefix_output(
            x0_draft=x0_draft, x0_tail=x0_tail, accepted_prefix_len=accepted_prefix_len
        )

        gripper_verify_stop = torch.zeros((bsize,), device=device, dtype=torch.bool)
        gripper_switch_cut = torch.zeros((bsize,), device=device, dtype=torch.bool)
        if self.args.enable_gripper_verify:
            gripper_verify_stop = _detect_verify_gripper_switch_any_k(
                x0_hat=x0_hat,
                gripper_prev=last_gripper,
                gripper_switch_threshold=float(self.args.gripper_switch_threshold),
                eval_h=eval_h,
                gripper_dims=self.args.gripper_dims,
            )
            accepted_prefix_len = torch.where(
                gripper_verify_stop, torch.zeros_like(accepted_prefix_len), accepted_prefix_len
            )
            x0_out = torch.where(gripper_verify_stop[:, None, None], x0_tail, x0_out)
        if self.args.enable_gripper_post_verify:
            accepted_after_cut, gripper_switch_cut = _truncate_accepted_prefix_on_gripper_switch(
                x0_out=x0_out,
                accepted_prefix_len=accepted_prefix_len,
                gripper_prev=last_gripper,
                gripper_switch_threshold=float(self.args.gripper_switch_threshold),
                gripper_dims=self.args.gripper_dims,
            )
            accepted_prefix_len = torch.where(gripper_verify_stop, accepted_prefix_len, accepted_after_cut)
            gripper_switch_cut = gripper_switch_cut & (~gripper_verify_stop)

        used_full_fallback = False
        if _should_schedule_full_fallback(
            full_fallback=self.args.full_fallback,
            accepted_prefix_len=accepted_prefix_len,
            gripper_switch_cut_mask=(gripper_switch_cut | gripper_verify_stop),
        ):
            # Real multi-step denoise from the parent (eager) -> clean chunk.
            x0_out = self._full_denoise(state, prefix_pad_masks, pkv, noise)
            used_full_fallback = True

        return {
            "actions": x0_out,  # (B, H, action_dim) in the model's native space
            "accepted_prefix_len": accepted_prefix_len,
            "radius_dist": dist,  # (B, K, eval_h)
            "gripper_verify_stop": gripper_verify_stop,
            "gripper_switch_cut": gripper_switch_cut,
            "used_full_fallback": used_full_fallback,
            "draft_ms": draft_ms,
            "verify_ms": verify_ms,
            "eval_h": eval_h,
        }

    def _full_denoise(self, state, prefix_pad_masks, pkv, noise):
        """Parent-equivalent full flow-matching denoise (eager), used on fallback."""
        model = self.model
        num_steps = int(self.args.full_num_steps)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        bsize = int(noise.shape[0])
        while time >= -dt / 2:
            v_t = model.denoise_step(state, prefix_pad_masks, pkv, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t
