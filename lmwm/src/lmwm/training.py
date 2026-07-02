"""Training-loop scaffolding shared by LMWM training scripts.

Seeding, device resolution, run/checkpoint directory management, minibatch
sampling, and checkpoint writing. Task-specific loss and evaluation logic stays
in the individual trainer scripts; only the stage-agnostic plumbing lives here.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: dict) -> torch.device:
    return torch.device(str(cfg.get("device", "cuda:0")) if torch.cuda.is_available() else "cpu")


def make_run_dir(cfg: dict, config_path: Path, default_run_root: str, default_run_id: str) -> Path:
    """Create ``<run_root>/<timestamp>+<run_id>`` and snapshot the config into it."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = str(cfg.get("run_id", default_run_id))
    run_root = Path(cfg.get("run_root_dir", default_run_root))
    run_dir = run_root / f"{ts}+{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / config_path.name)
    return run_dir


def make_ckpt_dir(cfg: dict, run_dir: Path, default_ckpt_root: str) -> Path:
    ckpt_dir = Path(cfg.get("checkpoint_root_dir", default_ckpt_root)) / run_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    return ckpt_dir


def sample_batch(train_idx: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Uniform with-replacement minibatch of indices (matches prior trainers)."""
    return train_idx[torch.randint(0, len(train_idx), (batch_size,), device=device)]


def save_checkpoint(path: Path, model: nn.Module, cfg: dict, meta: dict, metrics: dict | None) -> None:
    torch.save({"model": model.state_dict(), "config": cfg, "meta": meta, "metrics": metrics}, path)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
