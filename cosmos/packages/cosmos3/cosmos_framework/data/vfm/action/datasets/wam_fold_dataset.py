# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WAM fold (dual-arm Agilex Piper) LeRobot dataset for Cosmos3 Action policy SFT.

This mirrors :class:`~cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset.DROIDLeRobotDataset`
but adapts to the ``wam_fold_v1`` LeRobot v2.1 layout, which differs from DROID's
*file-grouped* layout in two important ways:

1. **Per-episode files.** Both parquet and mp4 are stored one-file-per-episode at
   ``data/chunk-{NNN}/episode_{NNNNNN}.parquet`` and
   ``videos/chunk-{NNN}/<video_key>/episode_{NNNNNN}.mp4`` (DROID groups many
   episodes into ``file-*.parquet``).  We therefore enumerate parquet files,
   read each episode's frames, and key videos by ``(episode_chunk, episode_index)``.

2. **14-D joint-space action.** The action / state vectors are 14-D dual-arm joint
   commands (per arm: 6 joints + 1 gripper).  We use an *absolute* 14-D Joint
   action spec and quantile-normalize against pre-computed per-dim stats — there is
   no pose-delta / rot6d conversion as in DROID.

The ``__getitem__`` return contract matches ``DROIDLeRobotDataset`` exactly so the
same ``ActionTransformPipeline`` / DataPacker can consume it:
``ai_caption``, ``video`` (uint8 ``[T, C, H, W]`` with ``T = chunk + 1``),
``action`` (normalized ``[chunk, 14]``), ``conditioning_fps``, ``mode``,
``domain_id``, ``viewpoint``, ``idle_frames``, plus ``additional_view_description``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.action_normalization import normalize_action
from cosmos_framework.data.vfm.action.action_spec import Joint, build_action_spec
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.pose_utils import compute_idle_frames

Viewpoint = Literal["concat_view"]

# wam_fold_v1 LeRobot video keys (see meta/info.json features).
_IMAGE_FEATURES = {
    "high": "observation.images.cam_high",
    "left": "observation.images.cam_left_wrist",
    "right": "observation.images.cam_right_wrist",
}

# Single fixed task for this dataset (meta/tasks.jsonl).
_TASK_TEXT = "Flatten and fold the cloth."

# Modes randomly sampled when constructed with mode="joint".
_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")

_ACTION_DIM = 14
# Per-arm joint layout: 6 arm joints + 1 gripper, x2 arms = 14.
# Idle detection treats all 14 columns as JOINT (frame-diff based), which is
# correct for absolute joint commands.
_ACTION_SPEC = build_action_spec(Joint(n=_ACTION_DIM, label="arm"))

# Per-rig defaults for GWP-style cross-embodiment joint training. Both rigs run
# the SAME 14-D fold task but with different camera extrinsics + workspace, so
# they are DISTINCT embodiment domains with per-rig quantile normalization (see
# giga_world_policy/world_action_model/configs/visrobot01_fold_aihc_latent.py:
# robotype_to_embed_id={"visrobot01":0,"kairobot01":1}, norm_path=[vis,kai]).
_DATA_ROOT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1"
_STATS_ROOT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/data/stats"
_RIG_DEFAULTS: dict[str, dict[str, str]] = {
    # visrobot01 → domain 16 ("wam_fold"); split applies (visrobot01_train/_val).
    "visrobot01": {
        "root": f"{_DATA_ROOT}/visrobot01_train",
        "domain_name": "wam_fold",
        "stats_path": f"{_STATS_ROOT}/visrobot01.json",
    },
    # kairobot01 → domain 17 ("kairobot01"); single root, no train/val split.
    "kairobot01": {
        "root": f"{_DATA_ROOT}/kairobot01",
        "domain_name": "kairobot01",
        "stats_path": f"{_STATS_ROOT}/kairobot01.json",
    },
}


class WamFoldLeRobotDataset(Dataset):
    """WAM fold dual-arm Action dataset (14-D joint policy)."""

    def __init__(
        self,
        rig: str = "visrobot01",
        root: str | None = None,
        split: str | None = None,
        domain_name: str | None = None,
        fps: float = 30.0,
        chunk_length: int = 16,
        mode: str = "policy",
        stats_path: str | None = None,
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
    ) -> None:
        super().__init__()
        if viewpoint != "concat_view":
            raise NotImplementedError("WamFoldLeRobotDataset only supports concat_view.")

        # Rig-derived defaults (GWP-style cross-embodiment): root / domain_name /
        # stats_path fall back to the per-rig entry in ``_RIG_DEFAULTS`` unless the
        # caller passes them explicitly. With no args at all this reproduces the
        # original behavior (visrobot01_train, domain 16="wam_fold", vis stats) so
        # ``eval_report.py`` and existing configs are unaffected.
        if rig not in _RIG_DEFAULTS:
            raise KeyError(f"Unknown rig {rig!r}. Available rigs: {sorted(_RIG_DEFAULTS)}")
        rig_defaults = _RIG_DEFAULTS[rig]
        if root is None:
            root = rig_defaults["root"]
        if domain_name is None:
            domain_name = rig_defaults["domain_name"]
        if stats_path is None:
            stats_path = rig_defaults["stats_path"]

        # ``split`` is accepted so configs can pass split="train"/"val"; it maps the
        # default root's trailing ``visrobot01_train`` to ``visrobot01_val`` etc.
        # (kairobot01 has no train/val split, so configs leave split=None for it.)
        if split is not None:
            root = self._apply_split(root, split)

        self._rig = rig
        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        # Rig-correct domain id (visrobot01="wam_fold"→16, kairobot01→17); each
        # per-item dict carries this so the collate's per-sample domain_id list
        # selects the right per-domain action bank.
        self._domain_name = domain_name
        self._domain_id = get_domain_id(domain_name)
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self._stats_path = stats_path

        self._root = Path(root)
        self._info = json.loads((self._root / "meta" / "info.json").read_text())

        # Per-episode parquet layout: enumerate every episode file, then build a
        # flat list of (episode_index, episode_chunk, local_row_index) start
        # positions. Each start can seed a (chunk_length + 1)-frame window that
        # stays within the same episode.
        self._episode_files = sorted(
            (self._root / "data").glob("chunk-*/episode_*.parquet"),
            key=lambda p: (p.parent.name, p.name),
        )
        if not self._episode_files:
            raise FileNotFoundError(f"No episode parquet files found under {self._root / 'data'}")

        # Index = list of (file_idx, start_row) windows. Cache per-file length and
        # the episode_chunk/episode_index parsed from the path so video lookup is
        # O(1) without reloading parquet.
        self._windows: list[tuple[int, int]] = []
        self._file_meta: list[dict[str, int]] = []
        for file_idx, path in enumerate(self._episode_files):
            episode_chunk = int(path.parent.name.split("-")[-1])
            episode_index = int(path.stem.split("_")[-1])
            num_rows = pq.read_metadata(path).num_rows
            self._file_meta.append(
                {
                    "episode_chunk": episode_chunk,
                    "episode_index": episode_index,
                    "num_rows": int(num_rows),
                }
            )
            n_windows = max(0, int(num_rows) - self._chunk_length)
            for start in range(n_windows):
                self._windows.append((file_idx, start))

    @staticmethod
    def _apply_split(root: str, split: str) -> str:
        """Swap the trailing ``visrobot01_<split>`` segment of the default root."""
        p = Path(root)
        name = p.name
        if "_" in name:
            prefix = name.rsplit("_", 1)[0]
            return str(p.with_name(f"{prefix}_{split}"))
        return str(p.with_name(split))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value

    @property
    def rig(self) -> str:
        return self._rig

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    @property
    def action_names(self) -> list[str]:
        return list(_ACTION_SPEC.names)

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(_MODE_CHOICES)
        return self._mode

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        file_idx, start = self._windows[idx]
        path = self._episode_files[file_idx]
        meta = self._file_meta[file_idx]

        table = pq.read_table(path)
        # Observation window has chunk_length + 1 frames; actions are the first
        # chunk_length of those (frame-aligned, matching DROID's slicing).
        obs_slice = slice(start, start + self._chunk_length + 1)
        timestamps = table.column("timestamp").to_pylist()[obs_slice]
        timestamps = [float(t[0]) if isinstance(t, (list, tuple)) else float(t) for t in timestamps]

        action_np = np.asarray(
            table.column("action").to_pylist()[start : start + self._chunk_length],
            dtype=np.float32,
        )  # [chunk, 14]

        video = self._load_concat_video(meta, timestamps)
        raw_action = torch.from_numpy(action_np).float()

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=_TASK_TEXT,
            additional_view_description=(
                "The top row is from the head-mounted camera. "
                "The bottom row contains two horizontally concatenated wrist-camera views, "
                "one from the left arm and one from the right arm."
            ),
        )

    def _load_concat_video(
        self,
        meta: dict[str, int],
        timestamps: list[float],
    ) -> torch.Tensor:
        frames_by_view = {
            name: decode_video_frames(
                self._video_path(meta, video_key),
                timestamps,
                self._tolerance_s,
            )
            for name, video_key in _IMAGE_FEATURES.items()
        }

        high = frames_by_view["high"]
        left = frames_by_view["left"]
        right = frames_by_view["right"]
        _, _, h_h, w_h = high.shape
        half_h, half_w = h_h // 2, w_h // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([high, bottom], dim=-2)

    def _video_path(self, meta: dict[str, int], video_key: str) -> Path:
        rel = self._info["video_path"].format(
            video_key=video_key,
            episode_chunk=meta["episode_chunk"],
            chunk_index=meta["episode_chunk"],
            episode_index=meta["episode_index"],
            file_index=meta["episode_index"],
        )
        return self._root / rel

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor,
        action: torch.Tensor,
        ai_caption: str,
        **extras: Any,
    ) -> dict[str, Any]:
        idle_frames = compute_idle_frames(
            action,
            _ACTION_SPEC,
            joint_threshold=5e-3,
            min_streak=3,
        )
        normalized_action = normalize_action(action, "quantile", self._load_norm_stats())
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            **extras,
        }

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is not None:
            return self._norm_stats
        # Stats file format: {"global": {"action": {mean,std,min,max,q01,q99 each
        # len-14}, "observation.state": {...}}}. We only need the "action" block.
        raw = json.loads(Path(self._stats_path).read_text())
        action_block = raw["global"]["action"]
        stat_keys = ("mean", "std", "min", "max", "q01", "q99")
        self._norm_stats = {
            key: torch.tensor(action_block[key], dtype=torch.float32)
            for key in stat_keys
            if key in action_block
        }
        return self._norm_stats
