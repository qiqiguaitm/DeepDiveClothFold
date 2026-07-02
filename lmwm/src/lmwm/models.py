"""Neural model definitions for the Latent Milestone World Model.

Single source of truth for every ``nn.Module`` used across LMWM training,
inference, and the runtime API. Training scripts and the runtime predictor must
import from here so that architecture and checkpoint state-dict keys never drift
between the trainer that writes a checkpoint and the loader that reads it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Configurable GELU MLP.

    Args:
        in_dim: input feature dim.
        hidden_dim: width of every hidden block.
        depth: number of hidden blocks (>=1).
        out_dim: output dim of the final linear layer.
        norm: if True, each hidden block is ``Linear -> GELU -> LayerNorm``;
            if False, ``Linear -> GELU``. The Stage-1 LaWM-shaped model uses
            ``norm=False`` to preserve its original architecture; the graph /
            unified models use ``norm=True``.

    Note: the ``(in_dim, hidden_dim, depth, out_dim)`` argument order is the
    canonical one. The old Stage-1 script used ``(in_dim, out_dim, hidden_dim,
    depth)``; that call site has been migrated, so only this order exists now.
    """

    def __init__(self, in_dim: int, hidden_dim: int, depth: int, out_dim: int, *, norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnifiedLMWM(nn.Module):
    """Stage-3 unified LMWM (current best artifact).

    Input: current DINOv3-H frame feature.
    Outputs: transition logits, Greedy/Max-product milestone logits, and
    L2-normalized latent prototype subgoals for both heads.
    """

    def __init__(self, in_dim: int, latent_dim: int, num_milestones: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.trunk = MLP(in_dim, hidden_dim, depth, hidden_dim)
        self.transition_head = nn.Linear(hidden_dim, num_milestones)
        self.greedy_head = nn.Linear(hidden_dim, num_milestones)
        self.max_product_head = nn.Linear(hidden_dim, num_milestones)
        self.greedy_proto_head = MLP(hidden_dim, hidden_dim, 1, latent_dim)
        self.max_product_proto_head = MLP(hidden_dim, hidden_dim, 1, latent_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        greedy_proto = F.normalize(self.greedy_proto_head(h), dim=-1)
        max_product_proto = F.normalize(self.max_product_proto_head(h), dim=-1)
        return {
            "transition_logits": self.transition_head(h),
            "greedy_logits": self.greedy_head(h),
            "max_product_logits": self.max_product_head(h),
            "greedy_proto": greedy_proto,
            "max_product_proto": max_product_proto,
        }


class GraphSupervisedLMWM(nn.Module):
    """Stage-2 graph-supervised policy: frame feature -> transition/greedy/max-product logits."""

    def __init__(self, in_dim: int, num_milestones: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.trunk = MLP(in_dim, hidden_dim, depth, hidden_dim)
        self.transition_head = nn.Linear(hidden_dim, num_milestones)
        self.greedy_head = nn.Linear(hidden_dim, num_milestones)
        self.max_product_head = nn.Linear(hidden_dim, num_milestones)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "transition_logits": self.transition_head(h),
            "greedy_logits": self.greedy_head(h),
            "max_product_logits": self.max_product_head(h),
        }


class LaWMShapedLMWM(nn.Module):
    """Stage-1 LaWM-shaped transition model.

    ``(r_t, r_future) -> inverse code u_t``; ``(r_t, u_t) -> r_hat_future``;
    ``r_hat_future -> milestone logits``. Uses ``norm=False`` MLPs to preserve
    the original Stage-1 architecture and keep old checkpoints loadable.
    """

    def __init__(
        self,
        raw_dim: int,
        num_milestones: int,
        code_dim: int,
        transition_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.projector = MLP(raw_dim, hidden_dim, 1, code_dim, norm=False)
        self.inverse = MLP(code_dim * 2, hidden_dim, depth, transition_dim, norm=False)
        self.forward_decoder = MLP(code_dim + transition_dim, hidden_dim, depth, code_dim, norm=False)
        self.classifier = nn.Linear(code_dim, num_milestones)

    def forward(self, current_raw: torch.Tensor, future_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        r_t = self.projector(current_raw)
        r_future = self.projector(future_raw)
        u_t = self.inverse(torch.cat([r_t, r_future], dim=-1))
        r_hat = self.forward_decoder(torch.cat([r_t, u_t], dim=-1))
        logits = self.classifier(r_hat)
        return {"r_t": r_t, "r_future": r_future, "u_t": u_t, "r_hat": r_hat, "logits": logits}


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
