"""Pydantic types for the dagger_manager API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CameraHealth(BaseModel):
    """Per-camera liveness for the preview tiles (mirrors data_manager)."""
    fps: float = 0.0
    target_fps: int = 30
    dropped: int = 0
    latency_ms: float = 0.0


class DaggerStatus(BaseModel):
    """Snapshot of the dagger ROS2 stack state.

    Fields stay flat (no nested objects) so the React UI can render any
    subset without complex unwrapping. Missing/unknown values are reported
    as None / sensible defaults rather than omitted, so clients don't need
    to handle "key absent" + "key None" as separate cases.
    """
    # Infra stack = CAN + cameras + arms + dagger_recorder + dagger_pedal,
    # forked by start_dagger_collect.sh (enable_policy:=false).
    stack_running: bool = False
    stack_pid: Optional[int] = None
    stack_log_path: Optional[str] = None
    # Session = policy_inference_node, forked by start_dagger_session.sh.
    # Independent lifecycle from infra; many sessions per infra possible.
    session_running: bool = False
    session_pid: Optional[int] = None
    session_log_path: Optional[str] = None
    session_started_at: Optional[float] = None
    # State machine snapshot (latched on /dagger/state); None until the
    # dagger_recorder node first publishes.
    state: Optional[str] = None
    # Writer flag — independent of state. True iff pedal opened a dagger
    # episode and it hasn't been closed yet. None if dagger_recorder hasn't
    # been seen yet (stack not started).
    recording: Optional[bool] = None
    # Per-arm freedrive switch (latched on /master_button_left,right)
    button_left: bool = False
    button_right: bool = False
    # /policy/execute latest known value
    policy_execute: Optional[bool] = None
    # Latest pedal event monotonic timestamp (for "recently fired" UI)
    last_pedal_ts: Optional[float] = None
    # Episode counts on disk under <KAI0_DATA_ROOT>/<task>/{inference,dagger}/<date-v2>/
    inference_episodes: int = 0
    dagger_episodes: int = 0
    # ckpt selected for the (next) stack start
    ckpt: Optional[str] = None
    task: Optional[str] = None
    # Live camera preview health, keyed by tile name (top_head/hand_left/hand_right)
    cameras: dict[str, CameraHealth] = {}


class CkptEntry(BaseModel):
    """A discovered checkpoint directory + key metadata for the picker UI."""
    path: str
    name: str  # last path component for compact display
    group: str  # parent dir name (ckpt_v0 / ckpt_v1 / dagger / ...)
    # Inference path: 'v1' (Triton serve + websocket) for ckpt_v1/*, else 'v0'
    # (JAX in-process). Drives which start_dagger_session.sh branch runs.
    variant: str = "v0"
    has_sidecar: bool  # train_config.json present (required by start_dagger_collect.sh)
    has_norm_stats: bool  # assets/<asset_id>/norm_stats.json reachable
    # v1 only: a v1_p200.pkl resolvable (self-contained or optimize/results).
    # Required for the v1 serve to load; surfaced so the UI can warn early.
    has_v1_pkl: bool = False
    config_name: Optional[str] = None  # from sidecar base_config_name
    task_hint: Optional[str] = None  # inferred Task_A/B/...


class StartStackReq(BaseModel):
    # ckpt is now optional — infra mode doesn't load it (policy is deferred
    # to session start). Web UI passes it anyway so /api/dagger/stack/start
    # validates the sidecar early.
    ckpt: Optional[str] = Field(None, description="Optional ckpt for sidecar pre-validation; not loaded by infra")
    task: Optional[str] = Field(None, description="Task_A/B/...; empty = infer")
    subset: str = "dagger"
    prompt: Optional[str] = None


class StartSessionReq(BaseModel):
    ckpt: str = Field(..., description="Absolute path to packed checkpoint dir")
    gpu_id: Optional[str] = Field(None, description="CUDA_VISIBLE_DEVICES (default: 0)")
    prompt: Optional[str] = None
    variant: Optional[str] = Field(
        None, description="v0|v1|auto inference path; None → auto-detect from ckpt path"
    )


class ExecuteReq(BaseModel):
    enable: bool


class TakeoverReq(BaseModel):
    enable: bool  # True = enter dagger, False = handback
