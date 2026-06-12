import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        latent_cache_dir: Optional[str] = None,      # VAE latent 缓存目录(compute_latents.py 产出)
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        # ── VAE latent 缓存(compute_latents.py 产出)──
        # 设置后:数据集只暴露"已缓存的窗口"(索引重映射,等价 GWP LatentEpisodeSampler stride=4),
        # __getitem__ 走 fast path:parquet(action/state)+ 缓存 latent,完全不解码视频。
        self.latent_cache_dir = latent_cache_dir
        self._dataset_dirs = [str(d) for d in dataset_dirs]
        self._cache_index = None   # [(ep, global_start, win_idx_in_ep), ...]
        self._ep_lru = {}          # ep -> dict(payload mmap + ep_data),容量 _ep_lru_cap
        self._ep_lru_cap = 2
        self._task_str = None
        if latent_cache_dir is not None:
            # 缓存按全集 episode 编号(mp4 文件名)建;split 会重排编号导致画面对错集 → 禁止
            if float(val_set_proportion) != 0.0:
                raise ValueError(
                    f"latent_cache_dir requires val_set_proportion=0 (got {val_set_proportion}): "
                    "non-zero split re-indexes episodes and mismatches the cache built on full-set numbering."
                )
            import glob as _glob
            files = sorted(_glob.glob(os.path.join(latent_cache_dir, "episode_*.pt")))
            self._cache_index = []
            for f in files:
                ep = int(os.path.basename(f).split("_")[1].split(".")[0])
                payload = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
                for wi, gs in enumerate(payload["starts"]):
                    self._cache_index.append((ep, int(gs), wi))
            logger.info(f"latent cache: {len(files)} episodes, {len(self._cache_index)} windows from {latent_cache_dir}")

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        if self._cache_index is not None:
            return len(self._cache_index)
        return len(self.lerobot_dataset)

    def _ep_cache(self, ep):
        """per-worker LRU:{latents(mmap), starts, act_win[T,48,14] raw, state[T,14] raw}。"""
        if ep in self._ep_lru:
            return self._ep_lru[ep]
        payload = torch.load(os.path.join(self.latent_cache_dir, f"episode_{ep:06d}.pt"),
                             map_location="cpu", weights_only=True, mmap=True)
        ep_data = self.lerobot_dataset._get_episode_data(ep)  # 纯 parquet,无视频
        entry = {
            "latents": payload["latents"], "starts": payload["starts"],
            "act_win": ep_data["action"],   # {key: [T,48,dim]} raw(滑窗,末端复制)
            "state": ep_data["state"],      # {key: [T,1,dim]} raw
        }
        if len(self._ep_lru) >= self._ep_lru_cap:
            self._ep_lru.pop(next(iter(self._ep_lru)))
        self._ep_lru[ep] = entry
        return entry

    def _get_cached(self, idx):
        """fast path:零视频解码。归一化与原路径同链:
        action_state_transform → normalizer.forward → action_state_merger.forward(同一 processor 实例)。"""
        ep, gstart, wi = self._cache_index[idx]
        ent = self._ep_cache(ep)
        ep_from = self.lerobot_dataset.episode_data_index["from"]
        lstart = gstart - int(ep_from[ep])
        proc = self.lerobot_dataset.processor

        T = self.num_frames                      # 49
        A = T - 1                                # 48
        batch = {
            "action": {k: v[lstart].clone() for k, v in ent["act_win"].items()},          # [48,dim]
            "state": {k: v[lstart:lstart + T, 0].clone() for k, v in ent["state"].items()},  # [49,dim]
            "action_is_pad": torch.zeros(A, dtype=torch.bool),
            "state_is_pad": torch.zeros(T, dtype=torch.bool),
            "idx": idx,
        }
        batch = proc.action_state_transform(batch)
        batch = proc.normalizer.forward(batch)
        batch = proc.action_state_merger.forward(batch)

        if self._task_str is None:
            if self.override_instruction is not None:
                self._task_str = self.override_instruction
            else:
                import json as _json
                p = os.path.join(self._dataset_dirs[0], "meta", "tasks.jsonl")
                if not os.path.exists(p):
                    raise RuntimeError(f"latent fast path: missing {p}")
                self._task_str = _json.loads(open(p).readline())["task"]
        instruction = DEFAULT_PROMPT.format(task=self._task_str)
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        n_vid = len(self.video_sample_indices)   # 13
        return {
            "video_latents": ent["latents"][wi].clone(),     # [C,Tl,H,W] — 不留批维,collate 后 [B,C,Tl,H,W]
            "action": batch["action"],                       # [48,14] normalized
            "proprio": batch["state"][:-1, :],               # [48,14] normalized(对齐原路径 [:-1])
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": torch.zeros(n_vid, dtype=torch.bool),
            "action_is_pad": batch["action_is_pad"],
            "proprio_is_pad": batch["state_is_pad"][:-1],
        }

    def _get(self, idx):
        if self._cache_index is not None:
            return self._get_cached(idx)
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
