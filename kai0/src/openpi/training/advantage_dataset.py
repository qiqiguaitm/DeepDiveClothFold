from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch


class AdvantageLerobotDataset(LeRobotDataset):
    """LeRobot dataset for advantage estimator training: history frames, timestep-difference mode, stage progress."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = 'pyav',
    ):
        """Init same as LeRobotDataset; adds episode_index -> array index mapping."""
        super().__init__(
            repo_id,
            root,
            episodes,
            image_transforms,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync,
            download_videos,
            video_backend
        )
        # Rebuild episode_data_index from the ACTUAL hf_dataset row order.
        # The released Task_A/kai0_advantage dataset has inconsistent meta <-> data episode_index:
        # meta.episodes lists ep_idx 0..3054, but the parquet rows contain ep_idx values in
        # a different range (0..3365 with gaps). The default LeRobotDataset.episode_data_index
        # is built from meta (wrong), so resampling a "same-episode" frame routinely lands
        # on the wrong episode and loops forever. Reindex from real data here.
        ep_col = np.asarray(self.hf_dataset.data.column("episode_index").to_numpy(zero_copy_only=False))
        n = len(ep_col)
        # Find contiguous runs; each unique episode occupies exactly one contiguous block.
        ep_to_range: dict[int, tuple[int, int]] = {}
        if n > 0:
            run_start = 0
            for i in range(1, n):
                if ep_col[i] != ep_col[run_start]:
                    ep_to_range[int(ep_col[run_start])] = (run_start, i)
                    run_start = i
            ep_to_range[int(ep_col[run_start])] = (run_start, n)

        # Use only episodes that actually exist in the data (ignore missing meta entries).
        eps_list = sorted(ep_to_range.keys())
        self.data_episode_indices = eps_list
        self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(eps_list)}

        from_arr = np.zeros(len(eps_list), dtype=np.int64)
        to_arr = np.zeros(len(eps_list), dtype=np.int64)
        for arr_idx, ep_idx in enumerate(eps_list):
            from_arr[arr_idx], to_arr[arr_idx] = ep_to_range[ep_idx]
        self.episode_data_index = {
            "from": torch.from_numpy(from_arr),
            "to": torch.from_numpy(to_arr),
        }

    def get_sample_with_imgs_from_idx(self, idx: int) -> dict:
        """Return one sample with decoded video frames at index idx."""
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        
        query_indices = None
        if self.delta_indices is not None:
            arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
            query_indices, padding = self._get_query_indices(idx, arr_idx)
        
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)

            try:
                video_frames = self._query_videos(query_timestamps, ep_idx)
            except Exception as e:
                # Bad video/timestamp — raise IndexError so DataLoader skips this sample.
                raise IndexError(f"Video decode failed for ep_idx {ep_idx} at idx {idx}: {e}") from e

            item = {**video_frames, **item}
        
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        
        return item

    def __getitem__(self, idx: int) -> dict:
        """Return sample with delta indices, random timestep comparison frame, and progress label."""
        episode_level_dict = {}
        item_sequence = []
        final_item = {}
        
        # Get main sample
        item = self.get_sample_with_imgs_from_idx(idx)
        
        ep_idx = item["episode_index"].item()
        cur_timestamp = item["timestamp"].item()
        
        _EP_IDX = ep_idx
        _CUR_TIMESTAMP = cur_timestamp
        
        # Compute length from actual data (meta.episodes[ep_idx] may not exist for all ep_idx in hf_dataset).
        arr_idx_for_len = self.ep_idx_to_arr_idx[ep_idx]
        episode_level_dict['episode_length'] = int(
            self.episode_data_index["to"][arr_idx_for_len].item()
            - self.episode_data_index["from"][arr_idx_for_len].item()
        )
        
        # Handle delta indices for action sequences
        if self.delta_indices is not None:
            episode_level_dict = self.handle_delta_indices(idx, ep_idx, episode_level_dict)
        
        # Add task as a string
        task_idx = item["task_index"].item()
        episode_level_dict["task"] = self.meta.tasks[task_idx]
        
        item_sequence.append(item.copy())
        final_item = self.handle_timestep_difference_mode(idx, ep_idx, final_item, _EP_IDX, _CUR_TIMESTAMP)
        final_item = {**final_item, **item_sequence[-1], **episode_level_dict}
        stage_progress_gt_random = final_item[f"his_-100_stage_progress_gt"].item()
        stage_progress_gt = final_item[f"stage_progress_gt"].item()
        final_item['progress'] = stage_progress_gt - stage_progress_gt_random

        return final_item

    def handle_delta_indices(self, idx, ep_idx, episode_level_dict) -> dict:
        """Fill episode_level_dict with action sequence from delta indices."""
        query_indices = None
        arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
        query_indices, padding = self._get_query_indices(idx, arr_idx)
        query_result = self._query_hf_dataset(query_indices)
        
        episode_level_dict = {**episode_level_dict, **padding}
        for key, val in query_result.items():
            episode_level_dict[key] = val
        return episode_level_dict

    def handle_timestep_difference_mode(self, idx, ep_idx, final_item, _EP_IDX, _CUR_TIMESTAMP) -> dict:
        """Add a random same-episode timestep sample with keys prefixed by his_-100_."""
        random_timestep_name = -100
        arr_idx = self.ep_idx_to_arr_idx[ep_idx]
        ep_start_idx = self.episode_data_index["from"][arr_idx].item()
        ep_end_idx = self.episode_data_index["to"][arr_idx].item()
        # Single-episode edge case: if the episode has only 1 frame, we can't pick a different one.
        if ep_end_idx - ep_start_idx <= 1:
            raise IndexError(f"Episode {ep_idx} too short for timestep difference")
        for _retry in range(16):
            random_idx = random.randint(ep_start_idx, ep_end_idx - 1)
            if random_idx == idx:
                continue
            random_item = self.get_sample_with_imgs_from_idx(random_idx)
            cur_timestamp_check = random_item["timestamp"].item()
            if cur_timestamp_check == _CUR_TIMESTAMP:
                continue
            break
        else:
            raise IndexError(f"Could not find valid timestep comparison for ep_idx {ep_idx}")
        
        _keys = list(random_item.keys())
        for key in _keys:
            new_key = f"his_{random_timestep_name}_{key}"
            random_item[new_key] = random_item.pop(key)

        final_item = {**final_item, **random_item}
        return final_item
