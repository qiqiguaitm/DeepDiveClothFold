"""VLA-facing LMWM interface (Phase D).

Packages the honest, real-future-validated recipe from Phases A/B/C into a single
online predictor for use as a VLA planning prior:

    p_cal   = softmax(greedy_logits / T)                  # temperature-calibrated
    p_prior = transition_probs[current_milestone]         # empirical milestone prior
    p_fused ∝ p_cal^(1-λ) · p_prior^λ                      # log-linear pool

Defaults `T=1.30`, `λ=0.30` are the held-out optima from Phase B. Outputs a
frame-conditional next-milestone distribution, top-k candidates, a latent
prototype subgoal, and confidence/entropy — all validated against the *real*
observed next milestone, not the circular graph-lookup metric.

This is intentionally separate from `runtime.UnifiedLMWMPredictor` (the older
graph-hybrid/fallback API): the VLA interface exposes the calibrated distribution
and a soft prior instead of hard graph fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lmwm.models import UnifiedLMWM


@dataclass(frozen=True)
class VLAInterfaceConfig:
    checkpoint: Path
    graph_npz: Path
    device: str = "cuda:0"
    temperature: float = 1.30
    prior_weight: float = 0.30
    topk: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VLAInterfaceConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            checkpoint=Path(raw["checkpoint"]),
            graph_npz=Path(raw["graph_npz"]),
            device=str(raw.get("device", "cuda:0")),
            temperature=float(raw.get("temperature", 1.30)),
            prior_weight=float(raw.get("prior_weight", 0.30)),
            topk=int(raw.get("topk", 5)),
        )


class VLALMWMPredictor:
    """Online next-milestone predictor for VLA conditioning."""

    def __init__(self, config: VLAInterfaceConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        g = np.load(config.graph_npz)
        self.prototype_table_np = g["prototype_table"].astype(np.float32)
        self.prototype_table = torch.from_numpy(self.prototype_table_np).to(self.device)
        transition = g["transition_probs"].astype(np.float64)
        self.transition_probs = transition / transition.sum(axis=1, keepdims=True).clip(1e-12)
        num_m = int(self.prototype_table_np.shape[0])

        ck = torch.load(config.checkpoint, map_location="cpu")
        mc = ck["config"].get("model", {})
        meta = ck.get("meta", {})
        in_dim = int(meta.get("input_dim", self.prototype_table_np.shape[1]))
        latent_dim = int(meta.get("latent_dim", self.prototype_table_np.shape[1]))
        self.model = UnifiedLMWM(in_dim, latent_dim, num_m, int(mc.get("hidden_dim", 512)), int(mc.get("depth", 2))).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.label_source = meta.get("label_source", "graph_lookup")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VLALMWMPredictor":
        return cls(VLAInterfaceConfig.from_yaml(path))

    @property
    def input_dim(self) -> int:
        return int(self.model.trunk.net[0].in_features)

    def _current_milestone(self, features: np.ndarray, given: np.ndarray | None) -> np.ndarray:
        if given is not None:
            m = np.asarray(given, dtype=np.int64)
            return m[None] if m.ndim == 0 else m
        # nearest prototype by cosine on the current-frame slice (last frame_dim dims)
        d = self.prototype_table_np.shape[1]
        cur = features[:, -d:]
        fn = cur / (np.linalg.norm(cur, axis=1, keepdims=True) + 1e-8)
        pn = self.prototype_table_np / (np.linalg.norm(self.prototype_table_np, axis=1, keepdims=True) + 1e-8)
        return (fn @ pn.T).argmax(axis=1).astype(np.int64)

    def predict(self, current_features: np.ndarray, current_milestones: np.ndarray | None = None) -> dict[str, np.ndarray]:
        features = np.asarray(current_features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.shape[1] != self.input_dim:
            raise ValueError(f"expected feature dim {self.input_dim}, got {features.shape[1]}")
        cur_m = self._current_milestone(features, current_milestones)

        with torch.no_grad():
            out = self.model(torch.from_numpy(features).to(self.device))
            logits = out["greedy_logits"] / self.config.temperature
            p_cal = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
            subgoal = out["greedy_proto"].cpu().numpy().astype(np.float32)

        p_prior = self.transition_probs[cur_m]
        lam = self.config.prior_weight
        logp = (1.0 - lam) * np.log(np.clip(p_cal, 1e-12, 1.0)) + lam * np.log(np.clip(p_prior, 1e-12, 1.0))
        logp -= logp.max(axis=1, keepdims=True)
        p_fused = np.exp(logp)
        p_fused /= p_fused.sum(axis=1, keepdims=True)

        k = min(self.config.topk, p_fused.shape[1])
        topk_idx = np.argpartition(-p_fused, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(p_fused, topk_idx, axis=1), axis=1)
        topk_idx = np.take_along_axis(topk_idx, order, axis=1)
        topk_p = np.take_along_axis(p_fused, topk_idx, axis=1)
        entropy = -(p_fused * np.log(np.clip(p_fused, 1e-12, 1.0))).sum(axis=1)

        return {
            "current_milestone": cur_m.astype(np.int64),
            "next_milestone_probs": p_fused.astype(np.float32),   # frame-conditional + soft graph prior
            "calibrated_probs": p_cal.astype(np.float32),          # neural only, temperature-scaled
            "topk_milestones": topk_idx.astype(np.int64),
            "topk_probs": topk_p.astype(np.float32),
            "next_milestone": topk_idx[:, 0].astype(np.int64),
            "confidence": p_fused.max(axis=1).astype(np.float32),
            "entropy": entropy.astype(np.float32),
            "subgoal_latent": subgoal,                             # L2-normalized prototype subgoal
        }

    def predict_one(self, current_feature: np.ndarray, current_milestone: int | None = None) -> dict[str, Any]:
        m = None if current_milestone is None else np.array([current_milestone], dtype=np.int64)
        batch = self.predict(np.asarray(current_feature, dtype=np.float32)[None, :], m)
        return {kk: vv[0] for kk, vv in batch.items()}
