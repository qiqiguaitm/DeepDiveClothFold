"""Multi-domain XVLA dataset wrapper for kai0 / vis (EE6D parquet + mp4).

Each sample yields:
  observation.images.image    (3,256,256) — top_head
  observation.images.image2   (3,256,256) — right_wrist (or left_wrist if right missing)
  observation.images.image3   (3,224,224) — left_wrist (or 2nd available)
  observation.state           (20,)        — EE6D current state
  action                       (30,20)      — EE6D chunk of next 30 actions
  task                         str           — language instruction
  domain_id                    int           — domain id (19=kai, 20=vis, 21=xvla)
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional, List
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import av  # PyAV for mp4 decoding

# P0 (2026-06-01): ImageNet normalization to match lerobot/xvla-base pretrain domain.
# The lerobot XVLAPolicy.forward does NOT normalize (XVLAImageNetNormalizeProcessorStep
# lives in the processor, which our train/serve path bypasses). Training on raw [0,1]
# left the pretrained Florence2 visual frontend mis-fed -> real-robot oscillation
# (see docs/training/analysis/xvla_vs_official_gap_rootcause.md R1). Apply ImageNet
# (image - mean) / std here; serve_policy_xvla.py applies the IDENTICAL transform.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def imagenet_normalize_chw(t: torch.Tensor) -> torch.Tensor:
    """(3,H,W) float in [0,1] -> ImageNet-normalized. Mirrors serve_policy_xvla._imagenet_normalize."""
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


def decode_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Decode 1 frame from mp4, return RGB (H,W,3) uint8."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    n_frames = stream.frames if stream.frames > 0 else None
    if n_frames and frame_idx >= n_frames:
        frame_idx = n_frames - 1
    # seek to keyframe before target then decode forward
    avg_rate = float(stream.average_rate) if stream.average_rate else 30.0
    tb = float(stream.time_base) if stream.time_base else (1.0 / avg_rate)
    target_pts = int(frame_idx / avg_rate / tb) if avg_rate > 0 and tb > 0 else 0
    try:
        container.seek(target_pts, stream=stream)
    except Exception:
        pass
    last = None
    for frame in container.decode(video=0):
        last = frame
        # PyAV VideoFrame has no .index in some versions — derive frame index from pts.
        cur_idx = int(round(float(frame.pts) * tb * avg_rate)) if frame.pts is not None else frame_idx
        if cur_idx >= frame_idx:
            img = frame.to_ndarray(format="rgb24")
            container.close()
            return img
    container.close()
    return last.to_ndarray(format="rgb24") if last is not None else np.zeros((480, 640, 3), dtype=np.uint8)


def resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Resize with padding, preserve aspect, returns (size,size,3) uint8."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    import cv2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_h = size - new_h
    pad_w = size - new_w
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    pt, pl = pad_h // 2, pad_w // 2
    padded[pt:pt+new_h, pl:pl+new_w] = resized
    return padded


class LeRobotEE6DDataset(Dataset):
    """Single-domain LeRobot v2.1 EE6D parquet dataset."""

    def __init__(
        self,
        root: str | Path,
        domain_id: int,
        task_prompt: str,
        action_chunk: int = 30,
        cam_keys: Optional[List[str]] = None,
        image_size_main: int = 256,
        image_size_wrist: int = 224,
        image_aug: bool = False,
        action_qdur: Optional[float] = None,
        static_skip: bool = False,
    ):
        self.root = Path(root)
        self.domain_id = int(domain_id)
        self.task_prompt = task_prompt
        self.action_chunk = action_chunk
        # D5 fix (2026-06-07): intention-abstraction action representation.
        # None  → legacy: 30 consecutive frames (dense, ~1s @30Hz) — 与 xvla-base 预训练表示不符。
        # 2.0   → 官方对齐: action_chunk 个 anchor 均匀铺在 qdur 秒上 (linspace 时间下采样),
        #         对齐 X-VLA base.py:152 (linspace(cur,cur+qdur,N+1)) + real_world.py:40 (qdur=2.0)。
        #         见 docs/training/analysis/xvla_vs_official_gap_rootcause.md §7。
        self.action_qdur = action_qdur
        self.image_size_main = image_size_main
        self.image_size_wrist = image_size_wrist
        # P0: ColorJitter(0.2) 对齐官方 X-VLA dataset.py (训练增强, 真机光照/颜色泛化)
        self.image_aug = bool(image_aug)
        self._jitter = None
        if self.image_aug:
            from torchvision.transforms import ColorJitter
            self._jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)

        info = json.load(open(self.root / "meta" / "info.json"))
        self.fps = info.get("fps", 30)
        # Detect camera keys
        all_cam_keys = [k for k in info["features"] if k.startswith("observation.images.")]
        if cam_keys is None:
            # Preferred order: top_head (main scene) -> hand_right (R wrist) -> hand_left (L wrist)
            preferred = ["observation.images.top_head", "observation.images.hand_right", "observation.images.hand_left"]
            self.cam_keys = [k for k in preferred if k in all_cam_keys][:3]
            for k in all_cam_keys:
                if k not in self.cam_keys and len(self.cam_keys) < 3:
                    self.cam_keys.append(k)
        else:
            self.cam_keys = cam_keys

        # Episodes manifest
        ep_path = self.root / "meta" / "episodes.jsonl"
        self.episodes = []
        with open(ep_path) as f:
            for line in f:
                ep = json.loads(line)
                self.episodes.append(ep)
        # Parquet path template + video path template (needed before sample build for existence check)
        self.parquet_tpl = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        self.video_tpl = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
        self.chunks_size = info.get("chunks_size", 1000)

        # 官方 static-frame skip (X-VLA domain_handler/base.py): 丢弃"未来首步双臂 EE 位姿几乎不动"
        # 的样本 (|seq[1]−seq[0]|<1e-5), 专剔 action≈proprio 的退化帧 (copycat 捷径温床)。
        # 仅 static_skip=True 时启用 (E0); 需逐 ep 读 action 列 → 略增 init 耗时, 故默认 off。
        self.static_skip = bool(static_skip)
        _POSE_IDX = [i for i in range(20) if i not in (9, 19)]  # 排除夹爪, 比 EE 位姿(xyz+rot6d)

        # Build (ep_index, frame_index) index
        # Each ep has "length" — total frames. Frame valid for sampling: 0 .. length-action_chunk
        self.samples = []
        n_skipped = 0      # episodes dropped (missing parquet)
        n_static = 0       # frames dropped (static, official skip)
        for ep in self.episodes:
            # LeRobot v2.1 uses "episode_index"; kai0-native (e.g. Task_A vis_base/v1) uses
            # "episode_id" with parquet/mp4 filenames numbered by that id → accept both.
            ep_idx = ep["episode_index"] if "episode_index" in ep else ep["episode_id"]
            length = ep["length"]
            # Defensive: skip stale manifest entries whose parquet was cleanup-deleted
            # (kai0 datasets sometimes leave episodes.jsonl ahead of the deleted files).
            pq_path = self._parquet_path(ep_idx)
            if not pq_path.exists():
                n_skipped += 1
                continue
            acts = None
            if self.static_skip:
                acts = np.stack([np.asarray(a, dtype=np.float32)
                                 for a in pq.read_table(pq_path, columns=["action"]).to_pandas()["action"]])
            for f_idx in range(max(0, length - action_chunk + 1)):
                if acts is not None:
                    # mirror official lseq[1]-lseq[0]: first future anchor vs current, both from action traj
                    if self.action_qdur is not None:
                        a1 = int(np.clip(round(f_idx + self.action_qdur * float(self.fps) / action_chunk),
                                         0, len(acts) - 1))
                    else:
                        a1 = min(f_idx + 1, len(acts) - 1)
                    if np.abs(acts[a1] - acts[f_idx])[_POSE_IDX].max() < 1e-5:
                        n_static += 1
                        continue
                self.samples.append((ep_idx, f_idx))
        if n_skipped:
            print(f"[LeRobotEE6DDataset] {self.root.name}: skipped {n_skipped} episode(s) "
                  f"with missing parquet (stale manifest)")
        if self.static_skip:
            print(f"[LeRobotEE6DDataset] {self.root.name}: static-skip dropped {n_static} frame(s)")

    def __len__(self):
        return len(self.samples)

    def _parquet_path(self, ep_idx: int) -> Path:
        chunk = ep_idx // self.chunks_size
        return self.root / self.parquet_tpl.format(episode_chunk=chunk, episode_index=ep_idx)

    def _video_path(self, ep_idx: int, cam_key: str) -> Path:
        chunk = ep_idx // self.chunks_size
        return self.root / self.video_tpl.format(episode_chunk=chunk, video_key=cam_key, episode_index=ep_idx)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, f_idx = self.samples[idx]
        pq_path = self._parquet_path(ep_idx)
        df = pq.read_table(pq_path).to_pandas()

        # State (current frame)
        state = np.array(df["observation.state"][f_idx], dtype=np.float32)
        n = len(df)
        if self.action_qdur is None:
            # legacy: next action_chunk consecutive frames (~1s @30Hz)
            max_f = min(n, f_idx + self.action_chunk)
            action_chunk = np.stack([np.array(df["action"][i], dtype=np.float32) for i in range(f_idx, max_f)])
            if action_chunk.shape[0] < self.action_chunk:
                pad = np.tile(action_chunk[-1:], (self.action_chunk - action_chunk.shape[0], 1))
                action_chunk = np.concatenate([action_chunk, pad], axis=0)
        else:
            # D5: intention-abstraction — action_chunk anchors over qdur seconds (linspace 时间下采样).
            # 对齐官方 base.py:152 `linspace(cur, cur+qdur, N+1)` 取后 N 个; 末端 clamp 到最后一帧.
            horizon = self.action_qdur * float(self.fps)          # e.g. 2.0s × 30 = 60 frames
            anchors = np.linspace(f_idx, f_idx + horizon, self.action_chunk + 1)[1:]  # N future anchors
            anchors = np.clip(np.rint(anchors).astype(int), 0, n - 1)
            action_chunk = np.stack([np.array(df["action"][i], dtype=np.float32) for i in anchors])

        # Decode camera frames
        img_dict = {}
        for i, cam_key in enumerate(self.cam_keys[:3]):
            try:
                video_path = self._video_path(ep_idx, cam_key)
                frame = decode_frame(video_path, f_idx)
                size = self.image_size_main if i < 2 else self.image_size_wrist
                frame = resize_pad(frame, size)
                # (H,W,3) → (3,H,W) [0,1] → [ColorJitter] → ImageNet-normalized (P0)
                # 顺序对齐官方: jitter 在 [0,1], 再归一化 (X-VLA dataset.py: ColorJitter→ToTensor→Normalize)
                t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                if self._jitter is not None:
                    t = self._jitter(t)
                t = imagenet_normalize_chw(t)
                img_dict["observation.images.image" + (str(i+1) if i > 0 else "")] = t
            except Exception as e:
                # Skip this sample on decode failure
                size = self.image_size_main if i < 2 else self.image_size_wrist
                img_dict["observation.images.image" + (str(i+1) if i > 0 else "")] = torch.zeros((3, size, size), dtype=torch.float32)

        return {
            **img_dict,
            "observation.state": torch.from_numpy(state),
            "action": torch.from_numpy(action_chunk),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
        }


class MultiDomainDataset(Dataset):
    """Concatenated multi-domain dataset for X-VLA training."""

    def __init__(self, datasets: List[LeRobotEE6DDataset]):
        self.datasets = datasets
        self.cum_lengths = np.cumsum([len(d) for d in datasets])
        self.total = int(self.cum_lengths[-1])

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int) -> dict:
        ds_idx = int(np.searchsorted(self.cum_lengths, idx, side="right"))
        local_idx = idx - (self.cum_lengths[ds_idx-1] if ds_idx > 0 else 0)
        return self.datasets[ds_idx][int(local_idx)]


def build_weighted_sampler(dataset: MultiDomainDataset, per_domain_weights: dict) -> torch.utils.data.WeightedRandomSampler:
    """Build WeightedRandomSampler from per-domain weights (domain_id -> weight)."""
    weights = []
    for ds in dataset.datasets:
        w = per_domain_weights.get(ds.domain_id, 1.0) / len(ds)  # per-sample weight
        weights.extend([w] * len(ds))
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


# ==================== XVLA-Soft-Fold hdf5 Dataset ====================
import h5py
import cv2


class XVLAHdf5Dataset(Dataset):
    """XVLA-Soft-Fold hdf5 dataset.

    Each hdf5 file = 1 episode. Yields the same sample structure as LeRobotEE6DDataset:
      observation.images.image    (3, 256, 256) from cam_high
      observation.images.image2   (3, 256, 256) from cam_right_wrist
      observation.images.image3   (3, 224, 224) from cam_left_wrist
      observation.state           (20,)         from observations/eef_6d
      action                       (30, 20)      from cached action_ee6d_cache/*.npy
      task                         str           "fold the cloth"
      domain_id                    int           21 (xvla)
    """

    def __init__(
        self,
        root: str | Path,
        action_cache_dir: str | Path,
        domain_id: int = 21,
        task_prompt: str = "Flatten and fold the cloth.",
        action_chunk: int = 30,
        image_size_main: int = 256,
        image_size_wrist: int = 224,
        action_qdur: Optional[float] = None,
        image_aug: bool = False,
        fps: int = 30,
    ):
        self.root = Path(root)
        self.action_cache_dir = Path(action_cache_dir)
        self.domain_id = int(domain_id)
        self.task_prompt = task_prompt
        self.action_chunk = action_chunk
        self.image_size_main = image_size_main
        self.image_size_wrist = image_size_wrist
        # D5 anchor (match LeRobotEE6DDataset): action_chunk anchors over qdur seconds (linspace downsample).
        self.action_qdur = action_qdur
        self.fps = fps
        # ColorJitter(0.2) train aug, identical to LeRobotEE6DDataset (jitter on [0,1] → ImageNet normalize).
        self.image_aug = bool(image_aug)
        self._jitter = None
        if self.image_aug:
            from torchvision.transforms import ColorJitter
            self._jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)

        # Find all hdf5 episodes
        self.hdf5_files = sorted(self.root.rglob("episode_*.hdf5"))

        # Build (file, frame_idx) sample index
        self.samples = []
        for hp in self.hdf5_files:
            cache_name = hp.parent.name + "__" + hp.stem + ".npy"
            cache_path = self.action_cache_dir / cache_name
            if not cache_path.exists():
                continue  # skip if no action cache
            # Get episode length from cache
            T = np.load(cache_path, mmap_mode="r").shape[0]
            for f_idx in range(max(0, T - action_chunk + 1)):
                self.samples.append((hp, cache_path, f_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        hp, cache_path, f_idx = self.samples[idx]

        # Load action cache (mmap)
        action_cache = np.load(cache_path, mmap_mode="r")
        T = action_cache.shape[0]
        if self.action_qdur is None:
            # legacy: next action_chunk consecutive frames (~1s @30Hz)
            max_f = min(T, f_idx + self.action_chunk)
            action_chunk = action_cache[f_idx:max_f].copy()
            if action_chunk.shape[0] < self.action_chunk:
                pad = np.tile(action_chunk[-1:], (self.action_chunk - action_chunk.shape[0], 1))
                action_chunk = np.concatenate([action_chunk, pad], axis=0)
        else:
            # D5: intention-abstraction — action_chunk anchors over qdur seconds (linspace 时间下采样).
            # Identical to LeRobotEE6DDataset anchor path (官方 base.py:152 linspace + 末端 clamp).
            horizon = self.action_qdur * float(self.fps)          # 2.0s × 30 = 60 frames
            anchors = np.linspace(f_idx, f_idx + horizon, self.action_chunk + 1)[1:]
            anchors = np.clip(np.rint(anchors).astype(int), 0, T - 1)
            action_chunk = action_cache[anchors].copy()

        # Load state + images from hdf5
        with h5py.File(hp, "r") as f:
            state = f["observations/eef_6d"][f_idx].astype(np.float32)
            # 3 cameras: cam_high (image), cam_right_wrist (image2), cam_left_wrist (image3)
            cam_order = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
            sizes = [self.image_size_main, self.image_size_main, self.image_size_wrist]
            imgs = {}
            for i, (cam, size) in enumerate(zip(cam_order, sizes)):
                jpg_bytes = f[f"observations/images/{cam}"][f_idx]
                arr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    arr = np.zeros((size, size, 3), dtype=np.uint8)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    arr = resize_pad(arr, size)
                key = "observation.images.image" + (str(i+1) if i > 0 else "")
                t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                if self._jitter is not None:
                    t = self._jitter(t)
                imgs[key] = imagenet_normalize_chw(t)

        return {
            **imgs,
            "observation.state": torch.from_numpy(state),
            "action": torch.from_numpy(action_chunk),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
        }
