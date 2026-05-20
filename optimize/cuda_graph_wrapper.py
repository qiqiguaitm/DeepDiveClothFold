"""
CUDA Graph wrapper for PI0Pytorch.sample_actions inference.

Implements the "C" backend of the pi05 inference benchmark — manual CUDA Graph
capture-replay (no Inductor fusion). Pattern adapted from realtime-vla v1
`pi05_infer.py:746-802`.

Key constraints (must be satisfied for capture to work):
- All inputs/outputs in fixed pre-allocated buffers (in-place .copy_() to update)
- Fixed shapes / dtypes / device throughout
- No host-device sync (.item() / .cpu() / print()) inside captured region
- model in eval() + torch.inference_mode() during warm-up and capture
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass

import torch


logger = logging.getLogger(__name__)


@dataclass
class CUDAGraphedSampleActions:
    """Wrapper that captures PI0Pytorch.sample_actions into a CUDA Graph.

    Usage:
        runner = CUDAGraphedSampleActions.capture(model, sample_obs, num_steps=10)
        for _ in range(N):
            out = runner(new_obs)  # replays the graph, returns action chunk
    """
    model: torch.nn.Module
    static_obs: object               # observation with .images / .image_masks / .tokenized_prompt / .tokenized_prompt_mask / .state
    static_noise: torch.Tensor       # (B, action_horizon, action_dim) fp32
    static_output: torch.Tensor      # output buffer, action chunk
    graph: torch.cuda.CUDAGraph
    num_steps: int
    device: torch.device

    @classmethod
    def capture(
        cls,
        model: torch.nn.Module,
        sample_obs,
        num_steps: int = 10,
        n_warmup: int = 3,
    ) -> "CUDAGraphedSampleActions":
        """Capture sample_actions into a CUDA Graph.

        sample_obs: a fully-populated observation that will be cloned into static buffers.
        """
        device = next(model.parameters()).device
        assert device.type == "cuda", f"CUDA Graph requires CUDA device, got {device}"

        # 1. Pre-allocate static buffers
        static_obs = _clone_observation_to_static(sample_obs, device)
        bsize = static_obs.state.shape[0]

        # Determine action shape — match Pi0Config defaults
        action_horizon = model.config.action_horizon
        action_dim = model.config.action_dim
        # Match the model's compute dtype — pi05 in production is bf16
        model_dtype = next(model.parameters()).dtype
        static_noise = torch.zeros(bsize, action_horizon, action_dim, dtype=model_dtype, device=device)
        static_output = torch.zeros_like(static_noise)

        # 2. Use the (patched) sample_actions method bound to model.
        #    After build_model in benchmark_pi05_inference.py, model.sample_actions
        #    is the bf16-aware monkey-patched version (NOT torch.compile-wrapped).
        sample_fn = model.sample_actions

        # 3. Warm-up on a side stream (let cuDNN/cuBLAS heuristics pick kernels)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(n_warmup):
                out = sample_fn(device, static_obs, noise=static_noise, num_steps=num_steps)
                static_output.copy_(out)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # 4. Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            out = sample_fn(device, static_obs, noise=static_noise, num_steps=num_steps)
            static_output.copy_(out)

        logger.info(f"CUDA Graph captured for sample_actions (num_steps={num_steps})")

        return cls(
            model=model,
            static_obs=static_obs,
            static_noise=static_noise,
            static_output=static_output,
            graph=graph,
            num_steps=num_steps,
            device=device,
        )

    def __call__(self, new_obs=None, new_noise: torch.Tensor | None = None) -> torch.Tensor:
        """Replay the captured graph with new inputs.

        new_obs: if provided, copy its fields into static_obs (in-place).
        new_noise: if provided, copy into static_noise (in-place).
        Returns: static_output (same tensor object every call; clone if you want to keep).
        """
        if new_obs is not None:
            _copy_observation_inplace(self.static_obs, new_obs)
        if new_noise is not None:
            self.static_noise.copy_(new_noise)
        self.graph.replay()
        return self.static_output


def _resolve_eager_sample_actions(model: torch.nn.Module):
    """Get the un-compiled `sample_actions` method bound to `model`.

    PI0Pytorch.__init__ does:
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
    This replaces the instance attribute. To recover the eager class method, we
    re-bind from the class.
    """
    cls = type(model)
    eager_method = cls.sample_actions  # unbound class method (def in pi0_pytorch.py:376)
    return types.MethodType(eager_method, model)


def _clone_observation_to_static(obs, device: torch.device):
    """Create a deep-cloned observation on `device` with stable tensor addresses.

    The returned object has the same attributes as `obs` (.images, .image_masks,
    .tokenized_prompt, .tokenized_prompt_mask, .state) but with newly-allocated
    tensors that will be used as the capture-time static buffers.
    """
    class _StaticObs:
        pass

    static = _StaticObs()
    static.images = {k: v.detach().clone().to(device) for k, v in obs.images.items()}
    static.image_masks = {k: v.detach().clone().to(device) for k, v in obs.image_masks.items()}
    static.tokenized_prompt = obs.tokenized_prompt.detach().clone().to(device)
    static.tokenized_prompt_mask = obs.tokenized_prompt_mask.detach().clone().to(device)
    static.token_ar_mask = obs.token_ar_mask.detach().clone().to(device)
    static.token_loss_mask = obs.token_loss_mask.detach().clone().to(device)
    static.state = obs.state.detach().clone().to(device)
    return static


def _copy_observation_inplace(static_obs, new_obs):
    """In-place copy new_obs fields into static_obs buffers.

    All shapes / dtypes must match — caller's responsibility.
    """
    for k, v in new_obs.images.items():
        static_obs.images[k].copy_(v)
    for k, v in new_obs.image_masks.items():
        static_obs.image_masks[k].copy_(v)
    static_obs.tokenized_prompt.copy_(new_obs.tokenized_prompt)
    static_obs.tokenized_prompt_mask.copy_(new_obs.tokenized_prompt_mask)
    static_obs.token_ar_mask.copy_(new_obs.token_ar_mask)
    static_obs.token_loss_mask.copy_(new_obs.token_loss_mask)
    static_obs.state.copy_(new_obs.state)
