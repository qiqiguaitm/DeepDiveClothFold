"""Runtime API for VLA-facing LMWM inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lmwm.models import UnifiedLMWM


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


@dataclass(frozen=True)
class HybridGateConfig:
    greedy_conf_threshold: float = 0.90
    max_product_conf_threshold: float = 0.92
    weak_milestones: tuple[int, ...] = ()


@dataclass(frozen=True)
class PredictorConfig:
    checkpoint: Path
    graph_npz: Path
    device: str = "cuda:0"
    gate: HybridGateConfig = HybridGateConfig()
    uncertainty_policy_npz: Path | None = None
    greedy_error_threshold: float | None = None
    max_product_error_threshold: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PredictorConfig":
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        weak = raw.get("weak_milestones", [])
        if isinstance(weak, str):
            weak_tuple = tuple(int(x) for x in weak.split(",") if x.strip())
        else:
            weak_tuple = tuple(int(x) for x in weak)
        return cls(
            checkpoint=Path(raw["checkpoint"]),
            graph_npz=Path(raw["graph_npz"]),
            device=str(raw.get("device", "cuda:0")),
            gate=HybridGateConfig(
                greedy_conf_threshold=float(raw.get("greedy_conf_threshold", 0.90)),
                max_product_conf_threshold=float(raw.get("max_product_conf_threshold", 0.92)),
                weak_milestones=weak_tuple,
            ),
            uncertainty_policy_npz=Path(raw["uncertainty_policy_npz"]) if raw.get("uncertainty_policy_npz") else None,
            greedy_error_threshold=float(raw["greedy_error_threshold"]) if raw.get("greedy_error_threshold") is not None else None,
            max_product_error_threshold=(
                float(raw["max_product_error_threshold"]) if raw.get("max_product_error_threshold") is not None else None
            ),
        )


class UnifiedLMWMPredictor:
    """Load a Stage-3 LMWM checkpoint and expose online prediction methods."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.graph = np.load(config.graph_npz)
        self.prototype_table_np = self.graph["prototype_table"].astype(np.float32)
        self.prototype_table = torch.from_numpy(self.prototype_table_np).to(self.device)
        self.greedy_table_np = self.graph["greedy_next"].astype(np.int64)
        self.max_product_table_np = self.graph["max_product_next"].astype(np.int64)
        self.transition_probs_np = self.graph["transition_probs"].astype(np.float32)
        self.uncertainty_policy = None
        if config.uncertainty_policy_npz is not None:
            self.uncertainty_policy = np.load(config.uncertainty_policy_npz, allow_pickle=True)

        ckpt = torch.load(config.checkpoint, map_location="cpu")
        model_cfg = ckpt["config"].get("model", {})
        input_dim = int(ckpt["meta"].get("input_dim", self.prototype_table_np.shape[1]))
        latent_dim = int(ckpt["meta"].get("latent_dim", self.prototype_table_np.shape[1]))
        num_milestones = int(self.prototype_table_np.shape[0])
        self.model = UnifiedLMWM(
            input_dim,
            latent_dim,
            num_milestones,
            int(model_cfg.get("hidden_dim", 512)),
            int(model_cfg.get("depth", 2)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UnifiedLMWMPredictor":
        return cls(PredictorConfig.from_yaml(path))

    @property
    def input_dim(self) -> int:
        return int(self.model.trunk.net[0].in_features)

    def predict(
        self,
        current_features: np.ndarray,
        current_milestones: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        features = np.asarray(current_features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.shape[1] != self.input_dim:
            raise ValueError(f"expected feature dim {self.input_dim}, got {features.shape[1]}")

        if current_milestones is None:
            feature_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            proto_norm = self.prototype_table_np / (np.linalg.norm(self.prototype_table_np, axis=1, keepdims=True) + 1e-8)
            current_milestones_np = (feature_norm @ proto_norm.T).argmax(axis=1).astype(np.int64)
        else:
            current_milestones_np = np.asarray(current_milestones, dtype=np.int64)
            if current_milestones_np.ndim == 0:
                current_milestones_np = current_milestones_np[None]
            if len(current_milestones_np) != len(features):
                raise ValueError("current_milestones length must match batch size")

        with torch.no_grad():
            x = torch.from_numpy(features).to(self.device)
            out = self.model(x)
            transition_probs = F.softmax(out["transition_logits"], dim=-1)
            greedy_probs = F.softmax(out["greedy_logits"], dim=-1)
            max_product_probs = F.softmax(out["max_product_logits"], dim=-1)
            greedy_pred = greedy_probs.argmax(dim=-1)
            max_product_pred = max_product_probs.argmax(dim=-1)
            greedy_conf = greedy_probs.max(dim=-1).values
            max_product_conf = max_product_probs.max(dim=-1).values
            transition_conf = transition_probs.max(dim=-1).values
            transition_entropy = _entropy(transition_probs)

            graph_greedy_np = self.greedy_table_np[current_milestones_np]
            graph_max_np = self.max_product_table_np[current_milestones_np]
            graph_greedy = torch.from_numpy(graph_greedy_np).to(self.device)
            graph_max = torch.from_numpy(graph_max_np).to(self.device)
            weak_mask_np = np.isin(current_milestones_np, np.array(self.config.gate.weak_milestones, dtype=np.int64))
            weak_mask = torch.from_numpy(weak_mask_np).to(self.device)
            greedy_error_prob = None
            max_product_error_prob = None
            if self.uncertainty_policy is None:
                greedy_fallback = (greedy_conf < self.config.gate.greedy_conf_threshold) | weak_mask
                max_fallback = (max_product_conf < self.config.gate.max_product_conf_threshold) | weak_mask
            else:
                onehot = F.one_hot(
                    torch.from_numpy(current_milestones_np).to(self.device),
                    num_classes=self.prototype_table.shape[0],
                ).float()
                greedy_feat = torch.cat(
                    [greedy_conf[:, None], transition_entropy[:, None], transition_conf[:, None], onehot],
                    dim=1,
                )
                max_feat = torch.cat(
                    [max_product_conf[:, None], transition_entropy[:, None], transition_conf[:, None], onehot],
                    dim=1,
                )
                greedy_w = torch.from_numpy(self.uncertainty_policy["greedy_weights"].astype(np.float32)).to(self.device)
                max_w = torch.from_numpy(self.uncertainty_policy["max_product_weights"].astype(np.float32)).to(self.device)
                greedy_error_prob = torch.sigmoid(greedy_feat @ greedy_w + float(self.uncertainty_policy["greedy_bias"]))
                max_product_error_prob = torch.sigmoid(
                    max_feat @ max_w + float(self.uncertainty_policy["max_product_bias"])
                )
                greedy_thr = (
                    self.config.greedy_error_threshold
                    if self.config.greedy_error_threshold is not None
                    else float(self.uncertainty_policy["greedy_error_threshold"])
                )
                max_thr = (
                    self.config.max_product_error_threshold
                    if self.config.max_product_error_threshold is not None
                    else float(self.uncertainty_policy["max_product_error_threshold"])
                )
                greedy_fallback = (greedy_error_prob >= greedy_thr) | weak_mask
                max_fallback = (max_product_error_prob >= max_thr) | weak_mask
            hybrid_greedy = torch.where(greedy_fallback, graph_greedy, greedy_pred)
            hybrid_max = torch.where(max_fallback, graph_max, max_product_pred)
            graph_greedy_proto = self.prototype_table[graph_greedy]
            graph_max_proto = self.prototype_table[graph_max]
            hybrid_greedy_proto = torch.where(greedy_fallback[:, None], graph_greedy_proto, out["greedy_proto"])
            hybrid_max_proto = torch.where(max_fallback[:, None], graph_max_proto, out["max_product_proto"])

        result = {
            "current_milestone": current_milestones_np.astype(np.int64),
            "transition_probs": transition_probs.cpu().numpy().astype(np.float32),
            "neural_greedy": greedy_pred.cpu().numpy().astype(np.int64),
            "neural_max_product": max_product_pred.cpu().numpy().astype(np.int64),
            "graph_greedy": graph_greedy_np.astype(np.int64),
            "graph_max_product": graph_max_np.astype(np.int64),
            "hybrid_greedy": hybrid_greedy.cpu().numpy().astype(np.int64),
            "hybrid_max_product": hybrid_max.cpu().numpy().astype(np.int64),
            "neural_greedy_subgoal_latent": out["greedy_proto"].cpu().numpy().astype(np.float32),
            "neural_max_product_subgoal_latent": out["max_product_proto"].cpu().numpy().astype(np.float32),
            "hybrid_greedy_subgoal_latent": hybrid_greedy_proto.cpu().numpy().astype(np.float32),
            "hybrid_max_product_subgoal_latent": hybrid_max_proto.cpu().numpy().astype(np.float32),
            "greedy_confidence": greedy_conf.cpu().numpy().astype(np.float32),
            "max_product_confidence": max_product_conf.cpu().numpy().astype(np.float32),
            "transition_entropy": transition_entropy.cpu().numpy().astype(np.float32),
            "greedy_fallback_mask": greedy_fallback.cpu().numpy().astype(bool),
            "max_product_fallback_mask": max_fallback.cpu().numpy().astype(bool),
        }
        if greedy_error_prob is not None and max_product_error_prob is not None:
            result["greedy_error_probability"] = greedy_error_prob.cpu().numpy().astype(np.float32)
            result["max_product_error_probability"] = max_product_error_prob.cpu().numpy().astype(np.float32)
        return result

    def predict_one(self, current_feature: np.ndarray, current_milestone: int | None = None) -> dict[str, Any]:
        milestones = None if current_milestone is None else np.array([current_milestone], dtype=np.int64)
        batch = self.predict(np.asarray(current_feature, dtype=np.float32)[None, :], milestones)
        return {k: v[0] for k, v in batch.items()}
