"""Dataset, config, and split helpers shared by LMWM training scripts.

Cohesive home for everything that reads YAML configs and turns exported CRAVE
``.npz`` artifacts into GPU-resident tensors. Kept separate from model and
training-loop code so a future streaming loader can replace ``*_data`` builders
without touching the trainers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def split_indices(
    z: np.lib.npyio.NpzFile,
    n: int,
    val_ratio: float,
    seed: int,
    device: torch.device,
    split_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(train_idx, val_idx)`` as device tensors.

    ``split_mode="episode"`` holds out whole episodes (no frame-level leakage)
    when the dataset carries ``episode_id``; otherwise a random frame split.
    """
    if split_mode == "episode" and "episode_id" in z.files:
        ep = z["episode_id"].astype(np.int64)
        unique_ep = np.unique(ep)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_ep)
        n_val_ep = max(1, int(round(len(unique_ep) * val_ratio)))
        val_ep = set(unique_ep[:n_val_ep].tolist())
        val_np = np.array([e in val_ep for e in ep])
        return (
            torch.from_numpy(np.where(~val_np)[0].astype(np.int64)).to(device),
            torch.from_numpy(np.where(val_np)[0].astype(np.int64)).to(device),
        )
    perm = torch.randperm(n, device=device)
    n_val = max(1, int(round(n * val_ratio)))
    return perm[n_val:], perm[:n_val]


def load_state_pair_data(cfg: dict, device: torch.device) -> tuple[dict[str, torch.Tensor], np.lib.npyio.NpzFile]:
    """Load LaWM-shaped current/future prototype pairs (Stage-1)."""
    z = np.load(cfg["dataset_npz"])
    data = {
        "current": torch.from_numpy(z["current"].astype(np.float32)).to(device),
        "future": torch.from_numpy(z["future"].astype(np.float32)).to(device),
        "future_milestone": torch.from_numpy(z["future_milestone"].astype(np.int64)).to(device),
    }
    return data, z


def load_graph_policy_data(
    cfg: dict,
    device: torch.device,
    include_proto: bool,
    label_source: str = "graph_lookup",
    proto_target_source: str = "centroid",
) -> tuple[dict[str, torch.Tensor], np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    """Load frame features + next-milestone supervision targets (Stage-2/3).

    ``label_source`` selects what the greedy / max-product / prototype heads are
    trained against:

    - ``"graph_lookup"`` (default, backward compatible): deterministic graph-table
      lookups indexed by the current milestone id (``greedy_next[current_m]`` /
      ``max_product_next[current_m]``). This is table-like: the target is a
      function of the discretized current state, not of the observed future.
    - ``"real_future"``: the actually observed next-unique milestone recorded in
      the dataset (``future_milestone``). Both point heads target this single
      real future; the prototype heads target ``proto[future_milestone]``.

    The transition head always targets the empirical milestone-level distribution
    ``transition_probs[current_m]`` (the honest multimodal distribution), so its
    real-future NLL is comparable across label sources.

    When ``include_proto`` is True, also attach the prototype-latent subgoal
    targets used by the unified Stage-3 model.
    """
    if label_source not in ("graph_lookup", "real_future"):
        raise ValueError(f"unknown label_source {label_source!r}")
    z = np.load(cfg["dataset_npz"])
    g = np.load(cfg["graph_npz"])
    current_m = z["current_milestone"].astype(np.int64)
    transition_probs = g["transition_probs"].astype(np.float32)
    if label_source == "real_future":
        future_m = z["future_milestone"].astype(np.int64)
        greedy_target = future_m
        max_product_target = future_m
    else:
        greedy_target = g["greedy_next"].astype(np.int64)[current_m]
        max_product_target = g["max_product_next"].astype(np.int64)[current_m]
    data: dict[str, torch.Tensor] = {
        "current": torch.from_numpy(z["current"].astype(np.float32)).to(device),
        "transition_target": torch.from_numpy(transition_probs[current_m]).to(device),
        "greedy_target": torch.from_numpy(greedy_target).to(device),
        "max_product_target": torch.from_numpy(max_product_target).to(device),
    }
    if include_proto:
        if proto_target_source == "episode_medoid":
            # Continuous, episode-real target: the next stage's medoid latent
            # (real frame closest to its centroid), L2-normalized. Both point
            # heads share it since it is the single observed next stage.
            if "next_medoid" not in z.files:
                raise KeyError("proto_target_source=episode_medoid needs `next_medoid` in the dataset")
            med = z["next_medoid"].astype(np.float32)
            med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
            med_t = torch.from_numpy(med).to(device)
            data["greedy_proto_target"] = med_t
            data["max_product_proto_target"] = med_t
        elif proto_target_source == "centroid":
            proto = g["prototype_table"].astype(np.float32)
            data["greedy_proto_target"] = torch.from_numpy(proto[greedy_target]).to(device)
            data["max_product_proto_target"] = torch.from_numpy(proto[max_product_target]).to(device)
        else:
            raise ValueError(f"unknown proto_target_source {proto_target_source!r}")
    return data, z, g
