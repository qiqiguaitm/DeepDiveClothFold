"""Policy transforms for the Agilex robot."""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class AgilexInputs(transforms.DataTransformFn):
    """Inputs for the Agilex policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. For normal pi05
      training, names must be exactly the keys of required_rename_map. For advantage
      estimator, optional_rename_map keys may be included as well.
    - state: [14]
    - actions: [action_horizon, 14]

    Optional extended inputs (gated by enable_depth / enable_ee_pose, default off
    so existing pi05 ckpts are unaffected):
    - depth_top_head: [H, W] float32 (meters) — D435 aligned depth, fed when
      enable_depth=True. Emitted as inputs["depth"]["base_0_depth"].
    - ee_pose_left / ee_pose_right: [7] xyz+quat_wxyz (m / unit quat) in world
      frame, fed when enable_ee_pose=True. Emitted as inputs["ee_pose"]["left"|"right"].
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.

    required_rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }
    # Optional cameras for advantage-estimator training (history frames).
    optional_rename_map = {
        "his_-100_top_head": "base_-100_rgb",
        "his_-100_hand_left": "left_wrist_-100_rgb",
        "his_-100_hand_right": "right_wrist_-100_rgb",
    }

    all_rename_map = {**required_rename_map, **optional_rename_map}

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(required_rename_map.keys())
    EXTRA_CAMERAS: ClassVar[tuple[str, ...]] = tuple(optional_rename_map.keys())

    # if set all state to zeros
    mask_state: bool = False

    # Extended modality toggles (default off → 14-dim joint-only behavior unchanged).
    # Set True for models that need depth and/or EE pose proprioception.
    enable_depth: bool = False
    enable_ee_pose: bool = False

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0/pi0_rtc model, not pi05/pi05_rtc or pi0-FAST
        mask_padding = self.model_type in (_model.ModelType.PI0, _model.ModelType.PI0_RTC)

        in_images = data["images"]

        if set(in_images) - set(self.EXPECTED_CAMERAS) - set(self.EXTRA_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        # Ensure state has correct shape [batch_size, state_dim]
        state = state.squeeze()

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        image_masks = {}
        for camera in self.EXPECTED_CAMERAS + self.EXTRA_CAMERAS:
            if camera in in_images:
                img = in_images[camera]
                # Convert torch tensor to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # Ensure image is in uint8 format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                # Convert from [C,H,W] to [H,W,C] if needed
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.all_rename_map[camera]] = img
                image_masks[self.all_rename_map[camera]] = np.True_

            elif camera not in in_images and camera in self.EXTRA_CAMERAS:
                continue  # optional camera can be skipped
            else:
                raise ValueError(f"Camera {camera} not found in data")


        # filter unnormal state / action value, set to 0
        state = np.where(state > np.pi, 0, state)
        state = np.where(state < -np.pi, 0, state)

        # Prepare inputs dictionary
        masked_state = np.zeros_like(state) if self.mask_state else state
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": masked_state,
        }

        # Optional: depth top_head (H, W) float32 meters → inputs["depth"]["base_0_depth"]
        if self.enable_depth:
            if "depth_top_head" not in data:
                raise ValueError("enable_depth=True but obs missing key 'depth_top_head'")
            depth = data["depth_top_head"]
            if isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()
            inputs["depth"] = {"base_0_depth": np.asarray(depth, dtype=np.float32)}

        # Optional: dual-arm EE pose, 7-dim xyz+quat_wxyz in world frame.
        # Kept as a separate top-level key so model code can opt-in without
        # disturbing the existing `state` shape (still 14-dim joint).
        if self.enable_ee_pose:
            for side in ("left", "right"):
                k = f"ee_pose_{side}"
                if k not in data:
                    raise ValueError(f"enable_ee_pose=True but obs missing key '{k}'")
            def _to_np(x):
                return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            inputs["ee_pose"] = {
                "left":  _to_np(data["ee_pose_left"]).astype(np.float32),
                "right": _to_np(data["ee_pose_right"]).astype(np.float32),
            }

        # Add actions if present
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            actions = np.where(actions > np.pi, 0, actions)
            actions = np.where(actions < -np.pi, 0, actions)
            if mask_padding:
                # Create action mask for padding
                action_mask = np.ones_like(actions, dtype=bool)
                action_mask[:, self.action_dim:] = False
                inputs["action_mask"] = action_mask

            inputs["actions"] = actions.squeeze()

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        # Advantage-estimator optional fields + X-VLA soft prompt dataset_id passthrough.
        # dataset_id must propagate to Pi0.embed_prefix to enable the soft_prompt_hub branch;
        # without it grad_norm(soft_prompt_hub) stays zero across the whole run.
        for key in ("frame_index", "episode_length", "progress", "image_original", "episode_index", "dataset_id"):
            if key in data:
                inputs[key] = data[key]
        
        def _to_tensor(x, default=None):
            if x is None and default is not None:
                return default
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            if isinstance(x, torch.Tensor):
                return x.detach().clone()
            raise NotImplementedError(f"Unsupported type: {type(x)}")

        if "action_advantage" in data:
            inputs["action_advantage"] = _to_tensor(data["action_advantage"], default=torch.tensor(1.0))
        if "action_advantage_original" in data:
            inputs["action_advantage_original"] = _to_tensor(data["action_advantage_original"])
        return inputs


@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    """Outputs for the Agilex policy.

    The model emits actions of shape [H, action_dim] where action_dim is the
    padded model action width (e.g. 32 for pi05). This wrapper slices to the
    meaningful prefix and tags `action_kind` so the client (policy_inference_node)
    can branch into joint-mode (drive arms directly) vs EE-mode (run IK first).

    `action_kind` MUST be configured at policy-creation time — the data shape
    alone cannot disambiguate (action_dim is always padded). It's a property
    of how the model was trained, not the runtime data.

    Slices:
      - action_kind="joint": actions[..., :14]
            14 = left_joint[6] + left_gripper + right_joint[6] + right_gripper
      - action_kind="ee":    actions[..., :16]
            16 = left_ee[7 = xyz + quat_wxyz] + left_gripper
               + right_ee[7 = xyz + quat_wxyz] + right_gripper
    """

    # "joint" (legacy, current default) or "ee" (Cartesian-trained model).
    # Set explicitly at policy creation time; do NOT infer from runtime shape.
    action_kind: str = "joint"

    def __call__(self, data: dict) -> dict:
        raw = np.asarray(data["actions"])
        if self.action_kind == "ee":
            return {"actions": raw[..., :16], "action_kind": "ee"}
        if self.action_kind == "joint":
            return {"actions": raw[..., :14], "action_kind": "joint"}
        raise ValueError(f"AgilexOutputs.action_kind must be 'joint' or 'ee', got {self.action_kind!r}")
