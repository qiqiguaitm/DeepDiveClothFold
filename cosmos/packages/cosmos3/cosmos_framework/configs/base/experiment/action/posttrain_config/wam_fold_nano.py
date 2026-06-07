# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``wam_fold_nano`` — Cosmos3-Nano-Policy posttrain on the WAM dual-arm fold dataset.

Fine-tunes the Nano (Qwen3-VL-8B / Cosmos3-Nano) MoT generator on ``wam_fold_v1``
(14-D dual-arm joint policy, 3 cameras concatenated into one view).  The action
head (``action2llm`` / ``llm2action`` / ``action_modality_embed`` /
``action_pos_embed``) is reset (skipped at load time) so it is re-learned for the
new embodiment; the model keeps ``max_action_dim=64`` and masks the unused 50
channels via the per-sample ``raw_action_dim=14`` flag emitted by
``ActionTransformPipeline``.

Mirrors the structure of ``configs/base/experiment/sft/vision_sft_nano.py`` but:

  * swaps the dataloader to ``DataPackerDataLoader`` driven by
    :class:`WamFoldLeRobotDataset` + a minimal action ``DataPacker`` that runs
    ``ActionTransformPipeline`` and produces a ``custom_collate_fn``-compatible
    batch (see ``cosmos_framework/data/vfm/joint_dataloader.py::custom_collate_fn``);
  * uses non-fused AdamW (``fused=False``) at ``lr=2e-5`` to avoid the
    transformer_engine FusedAdam path;
  * resets the action head via ``checkpoint.keys_to_skip_loading`` +
    ``strict_resume=False``.

Usage::

    BASE_CKPT_DCP=<dcp_path> torchrun --nproc_per_node=8 \\
        -m cosmos_framework.scripts.train \\
        --config=cosmos_framework/configs/base/config.py -- \\
        experiment=wam_fold_nano
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.utils.data
from hydra.core.config_store import ConfigStore
from torch.utils.data.dataloader import default_collate

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.action.datasets.wam_fold_dataset import WamFoldLeRobotDataset
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.data.vfm.data_packer import DataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader

cs = ConfigStore.instance()

# Keep the model's action head width and mask zero-padded channels per-sample.
# Must match NANO_MODEL_CONFIG["max_action_dim"].
_MAX_ACTION_DIM = 64

# Token-cost constants for compute_num_tokens (mirror
# joint_dataloader._compute_num_tokens_per_sample). Nano video tokenizer:
# spatial_compression=16, patch_spatial=2 → spatial downsample 32; temporal=4.
_SPATIAL_COMPRESSION = 16
_PATCH_SPATIAL = 2
_TEMPORAL_COMPRESSION = 4


def build_cross_rig_data_source(chunk_length: int = 16, fps: float = 30.0) -> torch.utils.data.ConcatDataset:
    """GWP-style cross-embodiment (cross-rig) joint training data_source.

    Concatenates two DISTINCT-embodiment rigs of the SAME 14-D dual-arm fold task
    (different camera extrinsics + workspace → separate domain ids, per-rig
    quantile normalization):

      * visrobot01_train (domain 16="wam_fold", vis stats), UPSAMPLED ×3
      * kairobot01        (domain 17="kairobot01", kai stats), ×1

    The ×3 upsample balances 2098×3 ≈ 6512 (≈3:1), exactly mirroring the GWP
    reference config
    ``giga_world_policy/world_action_model/configs/visrobot01_fold_aihc_latent.py``
    (``data = visrobot01_train ×3 + kairobot01``, ``robotype_to_embed_id=
    {"visrobot01":0,"kairobot01":1}``, ``norm_path=[vis,kai]``).

    Each underlying ``WamFoldLeRobotDataset`` yields its own ``domain_id`` (16/17)
    and rig-correct per-rig normalization per item, so the ``ActionDataPacker``
    collate (which lists ``domain_id`` per-sample) handles the mix transparently.
    ``shuffle=True`` on ``DataPackerDataLoader`` interleaves the two rigs.
    ``torch.utils.data.ConcatDataset`` is a map-style ``Dataset`` (not an
    ``IterableDataset``), so ``DataPackerDataLoader`` routes it through
    ``_ShuffledMapIterableDataset`` (data_packer_dataloader.py:287-308) and shuffles
    it like any single map-style dataset.
    """
    vis = WamFoldLeRobotDataset(
        rig="visrobot01", split="train", mode="policy", chunk_length=chunk_length, fps=fps
    )
    kai = WamFoldLeRobotDataset(
        rig="kairobot01", mode="policy", chunk_length=chunk_length, fps=fps
    )
    # visrobot01 ×3 upsample + kairobot01 ×1 ≈ 3:1 balance (GWP recipe).
    return torch.utils.data.ConcatDataset([vis, vis, vis, kai])


class ActionDataPacker(DataPacker):
    """Runs ``ActionTransformPipeline`` per sample and collates a VFM action batch.

    The collated batch matches ``custom_collate_fn`` in
    ``cosmos_framework/data/vfm/joint_dataloader.py``: list-valued for the
    multi-item / per-sequence keys (``video``, ``action``, ``domain_id``,
    ``sequence_plan``, ``raw_action_dim``, ``image_size``, ``text_token_ids``)
    and ``default_collate``-stacked for scalar keys (``conditioning_fps``,
    ``mode``, ``viewpoint``, ``idle_frames``, ...).
    """

    # Same set as custom_collate_fn's ``list_collate_keys``.
    _LIST_COLLATE_KEYS = {
        "text_token_ids",
        "images",
        "video",
        "action",
        "domain_id",
        "sequence_plan",
        "sound",
        "raw_action_dim",
        "image_size",
    }
    # Optional metadata keys that may not survive the pipeline / vary per sample.
    _DROP_IF_NONE_KEYS = {"additional_view_description"}

    def __init__(
        self,
        tokenizer_config: Any = None,
        resolution: str | None = "480",
        max_action_dim: int = _MAX_ACTION_DIM,
        cfg_dropout_rate: float = 0.1,
        append_idle_frames: bool = True,
        format_prompt_as_json: bool = False,
        spatial_compression: int = _SPATIAL_COMPRESSION,
        patch_spatial: int = _PATCH_SPATIAL,
        temporal_compression: int = _TEMPORAL_COMPRESSION,
    ) -> None:
        self._resolution = resolution
        self._spatial_compression = spatial_compression
        self._patch_spatial = patch_spatial
        self._temporal_compression = temporal_compression
        self._pipeline = ActionTransformPipeline(
            pad_keys=["video"],
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            max_action_dim=max_action_dim,
            action_channel_masking=True,  # emit raw_action_dim → mask padded channels
            append_viewpoint_info=True,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
            append_idle_frames=append_idle_frames,
            format_prompt_as_json=format_prompt_as_json,
            video_temporal_downsample=temporal_compression,
        )

    def sft_process_sample(self, item: dict) -> dict:
        # ActionTransformPipeline mutates and returns the dict; it resizes/pads
        # the video, tokenizes the caption (if tokenizer_config given), builds a
        # SequencePlan, and pads/masks the action to max_action_dim.
        return self._pipeline(item, resolution=self._resolution)

    def compute_num_tokens(self, sample: dict) -> int:
        # Mirror joint_dataloader._compute_num_tokens_per_sample:
        # text + 1 (eos) + (vision latent tokens + 2) + action steps.
        num_tokens = 1
        text_token_ids = sample.get("text_token_ids")
        if text_token_ids is not None:
            num_tokens += int(text_token_ids.shape[0])

        video = sample.get("video")  # [C, T, H, W]
        if isinstance(video, torch.Tensor):
            _, T, H, W = video.shape
            spatial_ds = self._spatial_compression * self._patch_spatial
            latent_h = H // spatial_ds
            latent_w = W // spatial_ds
            latent_t = 1 + (T - 1) // self._temporal_compression
            num_tokens += latent_h * latent_w * latent_t + 2

        action = sample.get("action")
        if isinstance(action, torch.Tensor):
            num_tokens += int(action.shape[0])

        return int(num_tokens)

    def sft_collate_fn(
        self,
        samples: list[dict],
        max_len: int,
        ignore_label_id: int = -100,
    ) -> dict:
        result: dict = {}
        keys: set = set().union(*(s.keys() for s in samples))
        for key in keys:
            values = [s.get(key) for s in samples]
            if any(v is None for v in values):
                # Drop optional metadata keys that are absent on some samples.
                # (Action samples are always full, so this only trims things like
                # additional_view_description if they ever go missing.)
                continue
            if key == "text_token_ids":
                # The model's ``_load_and_tokenize_text_data`` decodes the list
                # form via ``[tokens.tolist() for x in batch for tokens in x]``
                # (omni_mot_model.py:1089), i.e. it expects ``list[list[Tensor]]``:
                # one entry per sample, each a single-element list holding that
                # sample's 1-D id tensor. ``ActionTransformPipeline`` emits a bare
                # 1-D ``[N_tokens]`` tensor per sample (transforms.py:50 via
                # TextTokenizerTransform), so a flat ``list[Tensor]`` would make
                # ``for tokens in x`` iterate the tensor element-wise and yield
                # scalar ints -> ``text_ids`` reaching ``_pack_text_tokens`` is an
                # int. Wrap each per-sample tensor in a one-element list to match
                # the ``_MULTI_ITEM_KEYS`` contract produced by JointDataLoader's
                # ``_get_next_sample`` (joint_dataloader.py:475-486).
                result[key] = [[v] for v in values]
            elif key in self._LIST_COLLATE_KEYS:
                result[key] = values
            else:
                result[key] = default_collate(values)
        return result


wam_fold_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action",
            name="wam_fold_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            # Keep max_action_dim=64; raw_action_dim=14 (per-sample) masks padding.
            config=copy.deepcopy(NANO_MODEL_CONFIG),
        ),
        optimizer=dict(
            betas=[0.9, 0.95],
            eps=1.0e-08,
            fused=True,   # TE installed -> framework requires fused
            keys_to_select=[],  # train all params (incl. the reset action head)
            lr=2.0e-05,
            lr_multipliers={},
            optimizer_type="AdamW",
            weight_decay=0,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaCosine",
            cycle_lengths=[300],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[30],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=300,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                grad_clip=dict(clip_norm=0.1, force_finite=True),
                iter_speed=dict(every_n=1, hit_thres=100000, save_s3=False, save_s3_every_log_n=500),  # log every step (perf tuning visibility)
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                skip_nan_step=dict(max_consecutive_nan=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Reset the action head + EMA copy so the 14-D fold policy is learned
            # fresh on the new embodiment domain.
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm.",
                "llm2action.",
                "action_modality_embed",
                "action_pos_embed.",
            ],
            load_ema_to_reg=False,
            load_path="${oc.env:BASE_CKPT_DCP}",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,  # skipped action-head keys are absent → non-strict
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(DataPackerDataLoader)(
            # GWP-style cross-rig joint training: visrobot01_train ×3 + kairobot01
            # as a ConcatDataset. Each rig yields its own domain_id (16/17) +
            # per-rig normalization per sample; shuffle=True below mixes them.
            # See visrobot01_fold_aihc_latent.py (data = visrobot01_train ×3 + kairobot01).
            data_source=L(build_cross_rig_data_source)(
                chunk_length=16,
                fps=30.0,
            ),
            data_packer=L(ActionDataPacker)(
                # Inherit the model's VLM tokenizer so caption text is tokenized
                # consistently (matches vision_sft_nano's interpolation pattern).
                tokenizer_config="${model.config.vlm_config.tokenizer}",
                resolution="480",
                max_action_dim=_MAX_ACTION_DIM,
                cfg_dropout_rate=0.1,
                append_idle_frames=True,
            ),
            max_tokens=45056,
            max_batch_size=8,
            pool_size=16,
            shuffle=True,
            seed=42,
            num_workers=8,
            prefetch_factor=6,
            persistent_workers=True,
            pin_memory=True,
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [wam_fold_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
