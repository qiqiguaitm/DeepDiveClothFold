#!/usr/bin/env python3
"""DAgger session orchestrator + recorder (Form C: dual-dataset).

Form C — records BOTH policy rollouts AND human corrections to separate
datasets, binary-compatible with upstream kai0_dagger (HDF5 ≠ parquet but
schema-aligned). This is the open-source RECAP / AWBC advantage pipeline
prerequisite (see docs/deployment/strategy/awbc_implementation_plan.md).

Datasets:
  <task>/inference/<date-v2>/    ← policy rollouts (intervention=0)
  <task>/dagger/<date-v2>/       ← human corrections (intervention=1)

Lifecycle (4 states; pedal does NOT change state):

  POLICY_RUN   : policy publishes /master/joint_* → slave follows.
                 *Records to inference dataset, intervention=0.*
                 ↓ ANY freedrive switch rising edge (after slave-moved grace)
  ALIGNING     : (1) halt policy + finalize inference episode
                 (2) master into drag mode (encoder publish, slave follows)
                 ↓ both freedrive switches ON (handled inside _do_takeover)
  HUMAN_RECORD : drag mode active for as long as both switches stay ON.
                 Pedal toggles a SEPARATE _recording flag that drives the
                 dagger writer open/close — state stays HUMAN_RECORD.
                 *Records to dagger dataset (intervention=1) WHEN _recording.*
                 ↓ any switch falling
  RETURNING    : (1) close dagger writer if open
                 (2) re-enable masters (EnableArm + CAN_CTRL)
                 (3) /policy/execute=true → policy resumes
                 (4) open new inference episode
                 ↓ done
  POLICY_RUN   : back to start

Pedal toggle (KAI0 official Space ↔ s key equivalent):
  - In HUMAN_RECORD + _recording=False → open writer, _recording=True.
    Frames flow into a new dagger episode.
  - In HUMAN_RECORD + _recording=True → close writer, _recording=False.
    Episode is finalized; state stays HUMAN_RECORD.
  - In any other state → ignored (logged).

Multiple toggles within one (1,1) window produce multiple dagger episodes
— useful when one freedrive grip yields several distinct correction
segments. State machine cares about switches; pedal cares about which
frames are intervention=1.

Two-step button gate solves the "static prelude" problem: user opens
freedrive switches one at a time, drag only engages after BOTH are on
(meaning hands are physically on the masters and ready to drag). See
docs/deployment/strategy/dagger_implementation_plan.md §4.5.

state/action convention (KAI0 official, KAI0_ACTION_EQ_STATE=1):
  state  = puppet  left[7] + puppet  right[7]  (slave joint feedback)
  action = state (same — official kai0_dagger uses this)
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import threading
import time
from enum import Enum
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Empty, String


# ── reuse the data_manager writer (same on-disk bytes as teleop/autonomy) ──
def _bootstrap_backend_path() -> None:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "web" / "data_manager" / "backend" / "app" / "dataset_writer.py"
        if candidate.is_file():
            sys.path.insert(0, str(candidate.parent.parent))
            return
    raise RuntimeError("could not locate web/data_manager/backend")


os.environ["KAI0_DATA_ROOT"] = os.environ.get("KAI0_RECORDING_ROOT", "/data1/DATA_IMP/KAI0")
_bootstrap_backend_path()
from app.dataset_writer import (  # noqa: E402
    CAMERAS,
    DEPTH_CAMERAS,
    EpisodeWriter,
    FPS,
    next_episode_id,
    update_info_json,
    write_episode_meta,
)


CAM_RGB_TOPIC = {
    "top_head":   "/camera_f/camera/color/image_raw",
    "hand_left":  "/camera_l/camera/color/image_raw",
    "hand_right": "/camera_r/camera/color/image_raw",
}
CAM_DEPTH_TOPIC = {
    "top_head":   "/camera_f/camera/aligned_depth_to_color/image_raw",
    "hand_left":  "/camera_l/camera/aligned_depth_to_color/image_raw",
    "hand_right": "/camera_r/camera/aligned_depth_to_color/image_raw",
}

# Slave readback (state) + master readback (action publisher when 0xFA).
SLAVE_LEFT_TOPIC  = "/puppet/joint_left"
SLAVE_RIGHT_TOPIC = "/puppet/joint_right"
MASTER_LEFT_TOPIC  = "/master/joint_left"
MASTER_RIGHT_TOPIC = "/master/joint_right"

# Control surfaces — masters in 0xFC mode subscribe to /master_controled/joint_*
# (arm_teleop_node line 135). teleop_launch remaps the per-arm names below.
MASTER_DRIVE_LEFT  = "/master_controled/joint_left"
MASTER_DRIVE_RIGHT = "/master_controled/joint_right"
MASTER_CONFIG_LEFT  = "/teach/master_config_left"
MASTER_CONFIG_RIGHT = "/teach/master_config_right"
MASTER_ENABLE_LEFT  = "/teach/master_enable_left"
MASTER_ENABLE_RIGHT = "/teach/master_enable_right"
MASTER_TEACH_LEFT   = "/teach/teach_mode_left"
MASTER_TEACH_RIGHT  = "/teach/teach_mode_right"

# Safety: master should never blindly track slave from home position. Always go
# through a known-safe intermediate pose first (matches kai0 upstream agilex
# DAgger script). Empirically chosen — both arms inside their reachable
# workspace with no risk of self-collision.
SAFE_POSE = [0.0, 0.32, -0.36, 0.0, 0.24, 0.0, 0.07]

ALIGN_TOL_RAD = 0.02       # ~1.1° per joint
ALIGN_TIMEOUT_S = 5.0
ALIGN_PUBLISH_HZ = 10.0    # 10 Hz matches upstream; 50 Hz tended to overshoot
ALIGN_DURATION_S = 3.0     # how long to publish each target before moving on
MIN_EPISODE_SEC = 3.0      # drop accidental tap-toggles


class State(Enum):
    POLICY_RUN = "POLICY_RUN"
    ALIGNING = "ALIGNING"
    HUMAN_RECORD = "HUMAN_RECORD"
    RETURNING = "RETURNING"


def _decode_image_rgb(msg: Image) -> Optional[np.ndarray]:
    w, h, enc = msg.width, msg.height, msg.encoding
    data = bytes(msg.data)
    if enc == "rgb8":
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()
    if enc == "bgr8":
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(arr[:, :, ::-1])
    return None


def _decode_image_depth(msg: Image) -> Optional[np.ndarray]:
    if msg.encoding not in ("16UC1", "mono16"):
        return None
    w, h = msg.width, msg.height
    return np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w).copy()


def _to_7dim(msg: JointState) -> list[float]:
    pos = list(msg.position)[:7]
    pos += [0.0] * (7 - len(pos))
    return [float(x) for x in pos]


def _infer_task_from_ckpt(ckpt_dir: str) -> str:
    if not ckpt_dir:
        return "Task_A"
    s = ckpt_dir.lower()
    for letter in ("a", "b", "c", "d", "e"):
        if re.search(rf"\btask[_-]?{letter}\b", s) or f"/task_{letter}/" in s:
            return f"Task_{letter.upper()}"
    return "Task_A"


def _infer_prompt_from_ckpt(ckpt_dir: str) -> str:
    if not ckpt_dir:
        return "dagger correction"
    cfg_path = pathlib.Path(ckpt_dir) / "train_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
            p = cfg.get("prompt") or cfg.get("task_prompt") or cfg.get("default_prompt")
            if p:
                return str(p)
        except Exception:
            pass
    return f"dagger correction for {pathlib.Path(ckpt_dir).name}"


class DaggerRecorder(Node):
    def __init__(self) -> None:
        super().__init__("dagger_recorder")

        self.declare_parameter("task_name", "")
        self.declare_parameter("prompt", "")
        self.declare_parameter("subset", "dagger")
        self.declare_parameter("operator", "dagger")
        self.declare_parameter("checkpoint_dir", "")
        self.declare_parameter("align_tol_rad", ALIGN_TOL_RAD)
        self.declare_parameter("align_timeout_s", ALIGN_TIMEOUT_S)
        self.declare_parameter("min_episode_sec", MIN_EPISODE_SEC)

        ckpt_dir = self.get_parameter("checkpoint_dir").value or ""
        task_p = self.get_parameter("task_name").value or ""
        prompt_p = self.get_parameter("prompt").value or ""
        self._task: str = task_p or _infer_task_from_ckpt(ckpt_dir)
        self._prompt: str = prompt_p or _infer_prompt_from_ckpt(ckpt_dir)
        self._subset: str = self.get_parameter("subset").value or "dagger"
        self._operator: str = self.get_parameter("operator").value or "dagger"
        self._align_tol: float = float(self.get_parameter("align_tol_rad").value)
        self._align_timeout: float = float(self.get_parameter("align_timeout_s").value)
        self._min_ep_sec: float = float(self.get_parameter("min_episode_sec").value)

        self._lock = threading.Lock()
        self._state: State = State.POLICY_RUN

        # Dagger writer — opened/closed by pedal toggles inside HUMAN_RECORD.
        # Multiple pedal cycles within one (1,1) window produce multiple
        # episodes. Counts intervention=1 frames per episode.
        self._writer: Optional[EpisodeWriter] = None
        self._started_at: float = 0.0
        self._wrote_frames = 0
        # Pedal flag, independent of state machine. _on_pedal_toggle is the
        # only writer to this; _on_record_tick / _on_pedal_toggle read it.
        # Frames are written to dagger writer only when state=HUMAN_RECORD
        # AND _recording=True.
        self._recording: bool = False

        # Inference writer (Form C dual-dataset) — auto-opens in POLICY_RUN
        # once slave has moved + RGB ready, finalized on takeover. Counts
        # the policy rollout frames (intervention=0).
        self._inference_writer: Optional[EpisodeWriter] = None
        self._inf_started_at: float = 0.0
        self._inf_wrote_frames = 0

        self._rgb: dict[str, Optional[np.ndarray]] = {c: None for c in CAMERAS}
        self._depth: dict[str, Optional[np.ndarray]] = {c: None for c in DEPTH_CAMERAS}
        self._q_slave_left:  list[float] = [0.0] * 7
        self._q_slave_right: list[float] = [0.0] * 7
        self._q_master_left:  list[float] = [0.0] * 7
        self._q_master_right: list[float] = [0.0] * 7
        self._align_target_left:  Optional[list[float]] = None
        self._align_target_right: Optional[list[float]] = None

        # Physical-button raw state (5 Hz polled by arm_master_servo_node).
        self._button_left = False
        self._button_right = False
        # Previous level — used for edge detection. Without this, level-
        # triggered _on_button fires repeatedly while a switch is held ON,
        # spamming takeover attempts (one per 200 ms poll). See Phase D1
        # post-mortem in dagger_implementation_plan.md.
        self._prev_any_pressed = False

        # Grace period: don't accept takeover until slave has moved from its
        # boot zero pose. Some freedrive switches default to ON at power-up,
        # which would otherwise spawn a takeover before policy starts driving
        # the slave (slave at zero → _do_takeover aborts → loop). Cleared
        # once any slave joint exceeds 0.01 rad from zero.
        self._slave_seen_nonzero = False

        # Startup gate: if a freedrive switch is already ON when dagger_recorder
        # boots, the FIRST button message arrives with msg.data=True. Without
        # this gate, that gets treated as a rising edge → premature takeover
        # before policy has even finished loading (policy was loading JAX,
        # received execute=False mid-load, then went to OBSERVE on init —
        # observed in /tmp/dagger_step2_log.txt). We require seeing at least
        # one "all-OFF" message first, proving the switches are actually being
        # toggled by the operator after startup, not just held over from a
        # previous run / left high at power-up.
        self._seen_off_after_boot = False

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        for cam, topic in CAM_RGB_TOPIC.items():
            self.create_subscription(
                Image, topic,
                lambda msg, k=cam: self._on_rgb(k, msg),
                sensor_qos,
            )
        for cam in DEPTH_CAMERAS:
            topic = CAM_DEPTH_TOPIC.get(cam)
            if topic:
                self.create_subscription(
                    Image, topic,
                    lambda msg, k=cam: self._on_depth(k, msg),
                    sensor_qos,
                )

        self.create_subscription(JointState, SLAVE_LEFT_TOPIC,
                                 lambda m: self._on_slave("L", m), 10)
        self.create_subscription(JointState, SLAVE_RIGHT_TOPIC,
                                 lambda m: self._on_slave("R", m), 10)
        self.create_subscription(JointState, MASTER_LEFT_TOPIC,
                                 lambda m: self._on_master("L", m), 10)
        self.create_subscription(JointState, MASTER_RIGHT_TOPIC,
                                 lambda m: self._on_master("R", m), 10)

        self.create_subscription(Bool, "/dagger/takeover", self._on_takeover, 1)
        # Per-arm physical-button state (published by arm_master_servo_node).
        # dagger_launch remaps these to /master_button_left and /master_button_right.
        self.create_subscription(Bool, "/master_button_left",
                                  lambda m: self._on_button("L", m), 5)
        self.create_subscription(Bool, "/master_button_right",
                                  lambda m: self._on_button("R", m), 5)
        # USB pedal toggle (published by dagger_pedal_node on F3 release).
        # Each Empty event flips the _recording flag inside HUMAN_RECORD.
        self.create_subscription(Empty, "/dagger/pedal_toggled",
                                  self._on_pedal_toggle, 5)
        # Explicit start/save/discard commands (web/dagger_manager 三按钮),
        # mirroring start_data_collect.sh's recorder. String in {start,save,
        # discard}. Pedal stays a start↔save toggle; discard is web-only.
        self.create_subscription(String, "/dagger/record_cmd",
                                  self._on_record_cmd, 5)

        self.pub_execute = self.create_publisher(Bool, "/policy/execute", 1)
        # State machine snapshot for web/dagger_manager (latched, so a late
        # subscriber gets the current state immediately). One String message
        # per transition; consumers compare against State enum names.
        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_state = self.create_publisher(String, "/dagger/state", latched)
        self.pub_state.publish(String(data=self._state.value))
        # Recording flag: separate from state so consumers (web UI, logging)
        # can render "in dagger mode but not actively writing" distinctly
        # from "actively writing a dagger episode".
        self.pub_recording = self.create_publisher(Bool, "/dagger/recording", latched)
        self.pub_recording.publish(Bool(data=self._recording))
        self.pub_drive_left  = self.create_publisher(JointState, MASTER_DRIVE_LEFT, 10)
        self.pub_drive_right = self.create_publisher(JointState, MASTER_DRIVE_RIGHT, 10)
        self.pub_cfg_left  = self.create_publisher(String, MASTER_CONFIG_LEFT, 1)
        self.pub_cfg_right = self.create_publisher(String, MASTER_CONFIG_RIGHT, 1)
        # Match upstream agilex DAgger script: explicit re-enable + teach_mode pubs
        from std_msgs.msg import Int32  # local import — Int32 only used here
        self.pub_enable_left  = self.create_publisher(Bool, MASTER_ENABLE_LEFT, 10)
        self.pub_enable_right = self.create_publisher(Bool, MASTER_ENABLE_RIGHT, 10)
        self.pub_teach_left   = self.create_publisher(Int32, MASTER_TEACH_LEFT, 10)
        self.pub_teach_right  = self.create_publisher(Int32, MASTER_TEACH_RIGHT, 10)
        self._Int32 = Int32

        self.create_timer(1.0 / FPS, self._on_record_tick)

        # No mirror loop — master_servo subscribes /master/joint_* directly,
        # so master physically tracks whatever drives the slave (policy or
        # master's own encoder publish) automatically.

        self.get_logger().info(
            f"dagger_recorder ready: task={self._task} subset={self._subset} "
            f"prompt={self._prompt!r} fps={FPS}\n"
            f"  state={self._state.value} — waiting for /dagger/takeover"
        )

    def _publish_state(self) -> None:
        """Latch-publish current state to /dagger/state for web UI consumers.
        Safe to call inside or outside self._lock — only reads self._state."""
        try:
            self.pub_state.publish(String(data=self._state.value))
        except Exception:
            pass

    def _publish_recording(self) -> None:
        """Latch-publish current recording flag to /dagger/recording."""
        try:
            self.pub_recording.publish(Bool(data=self._recording))
        except Exception:
            pass

    # ── sensor callbacks ──
    def _on_rgb(self, cam: str, msg: Image) -> None:
        arr = _decode_image_rgb(msg)
        if arr is None:
            return
        with self._lock:
            self._rgb[cam] = arr

    def _on_depth(self, cam: str, msg: Image) -> None:
        arr = _decode_image_depth(msg)
        if arr is None:
            return
        with self._lock:
            self._depth[cam] = arr

    def _on_slave(self, side: str, msg: JointState) -> None:
        pos = _to_7dim(msg)
        with self._lock:
            if side == "L":
                self._q_slave_left = pos
            else:
                self._q_slave_right = pos
            # Grace period: once any slave joint has clearly moved off zero,
            # button presses can trigger takeover. Until then, freedrive
            # switches held ON since boot are ignored.
            if not self._slave_seen_nonzero:
                if (any(abs(x) > 0.01 for x in self._q_slave_left[:6]) or
                        any(abs(x) > 0.01 for x in self._q_slave_right[:6])):
                    self._slave_seen_nonzero = True

    def _on_master(self, side: str, msg: JointState) -> None:
        pos = _to_7dim(msg)
        with self._lock:
            if side == "L":
                self._q_master_left = pos
            else:
                self._q_master_right = pos

    def _on_button(self, side: str, msg: Bool) -> None:
        """Per-arm freedrive-button state from arm_master_servo's teach_status poll.

        Edge-triggered (rising/falling) so that level-held switches don't spam
        callbacks. The 5 Hz polling in arm_master_servo means a held-ON switch
        publishes True every 200 ms — without edge detection that would spawn
        a takeover thread every tick.

        Aggregated rule (Form C two-step gate):
          POLICY_RUN  + ANY rising  → start takeover (open dagger episode in
                                       ALIGNING → HUMAN_RECORD).
          HUMAN_RECORD + ANY falling → start handback (finalize dagger,
                                       resume policy, open new inference ep).

        Grace period: takeover ignored until slave has moved off boot zero,
        protecting against boot-time switch-already-ON race.
        """
        pressed = bool(msg.data)
        with self._lock:
            if side == "L":
                self._button_left = pressed
            else:
                self._button_right = pressed
            any_pressed = self._button_left or self._button_right
            prev_any = self._prev_any_pressed
            rising = any_pressed and not prev_any
            falling = (not any_pressed) and prev_any
            self._prev_any_pressed = any_pressed
            cur = self._state
            grace = self._slave_seen_nonzero
            # First all-OFF message arms the rising-edge detector.
            if not any_pressed and not self._seen_off_after_boot:
                self._seen_off_after_boot = True
            seen_off = self._seen_off_after_boot

        if rising and cur == State.POLICY_RUN:
            if not seen_off:
                self.get_logger().warn(
                    f"[button] {side} rising → IGNORED (freedrive switches were "
                    "ALREADY ON at dagger_recorder boot; release both switches "
                    "first, then re-engage to trigger takeover)"
                )
                return
            if not grace:
                self.get_logger().warn(
                    f"[button] {side} rising → IGNORED (slave still at boot "
                    "zero pose; freedrive switches must come ON after policy "
                    "starts driving slave)"
                )
                return
            self.get_logger().info(
                f"[button] {side} rising → takeover "
                f"(L={self._button_left} R={self._button_right})"
            )
            threading.Thread(target=self._do_takeover, daemon=True).start()
        elif falling and cur == State.HUMAN_RECORD:
            self.get_logger().info(
                f"[button] {side} falling → handback "
                f"(L={self._button_left} R={self._button_right})"
            )
            threading.Thread(target=self._do_handback, daemon=True).start()

    # ── /dagger/takeover edge handler ──
    def _on_takeover(self, msg: Bool) -> None:
        want = bool(msg.data)
        with self._lock:
            cur = self._state
        if want and cur == State.POLICY_RUN:
            threading.Thread(target=self._do_takeover, daemon=True).start()
        elif (not want) and cur == State.HUMAN_RECORD:
            threading.Thread(target=self._do_handback, daemon=True).start()
        else:
            self.get_logger().warn(
                f"/dagger/takeover={want} ignored in state={cur.value}"
            )

    # ── /dagger/pedal_toggled handler ──
    def _on_pedal_toggle(self, _msg: Empty) -> None:
        """Pedal press (F3 release on default HID 0483:5750).

        Toggles the dagger writer open/closed WITHOUT touching the state
        machine. Each press in HUMAN_RECORD with _recording=False starts a
        new episode; press with _recording=True finalizes it. State stays
        HUMAN_RECORD across pedal cycles — only switches drive transitions.

        Ignored in POLICY_RUN / ALIGNING / RETURNING.
        """
        with self._lock:
            cur = self._state
            recording = self._recording
        if cur != State.HUMAN_RECORD:
            self.get_logger().info(
                f"[pedal] ignored in state={cur.value} "
                "(pedal only meaningful in HUMAN_RECORD)"
            )
            return
        # Pedal = start↔save toggle (no discard; discard is web-only).
        if not recording:
            self._start_recording(src="pedal")
        else:
            self._save_recording(src="pedal")

    # ── /dagger/record_cmd handler (web 三按钮: start / save / discard) ──
    def _on_record_cmd(self, msg: String) -> None:
        """Explicit recording command, mirroring start_data_collect.sh's
        recorder.start / save / discard. State machine is untouched — these
        only drive the dagger episode writer inside HUMAN_RECORD."""
        cmd = (msg.data or "").strip().lower()
        if cmd == "start":
            self._start_recording(src="cmd")
        elif cmd == "save":
            self._save_recording(src="cmd")
        elif cmd == "discard":
            self._discard_recording(src="cmd")
        else:
            self.get_logger().warn(
                f"[record] unknown cmd '{cmd}' (want start|save|discard)"
            )

    # ── recording helpers (shared by pedal + record_cmd) ──
    def _start_recording(self, src: str = "cmd") -> bool:
        """Open a dagger episode (only in HUMAN_RECORD, only if not already
        recording). Returns True on success."""
        with self._lock:
            cur = self._state
            recording = self._recording
        if cur != State.HUMAN_RECORD:
            self.get_logger().info(
                f"[{src}] start ignored in state={cur.value} (need HUMAN_RECORD)")
            return False
        if recording:
            self.get_logger().info(f"[{src}] start ignored — already recording")
            return False
        self.get_logger().info(f"[{src}] open dagger episode (recording=True)")
        self._open_episode()
        with self._lock:
            ok = self._writer is not None
            if ok:
                self._recording = True
            else:
                self.get_logger().warn(
                    f"[{src}] writer init failed; recording stays False")
        self._publish_recording()
        return ok

    def _save_recording(self, src: str = "cmd") -> bool:
        """Finalize + keep the current dagger episode."""
        with self._lock:
            recording = self._recording
        if not recording:
            self.get_logger().info(f"[{src}] save ignored — not recording")
            return False
        self.get_logger().info(f"[{src}] save dagger episode (finalize, recording=False)")
        self._close_episode()
        with self._lock:
            self._recording = False
        self._publish_recording()
        return True

    def _discard_recording(self, src: str = "cmd") -> bool:
        """Abort the current dagger episode — delete partial files, keep none."""
        with self._lock:
            recording = self._recording
        if not recording:
            self.get_logger().info(f"[{src}] discard ignored — not recording")
            return False
        self.get_logger().info(f"[{src}] discard dagger episode (abort, recording=False)")
        self._discard_episode()
        with self._lock:
            self._recording = False
        self._publish_recording()
        return True

    # ── mirror loop (policy phase): continuously publish slave's encoder pose
    # to /master_controled/joint_* so the master's arm_master_servo_node drives
    # the master arm via JointCtrl to mirror slave. Visual feedback: when
    # policy moves slave, master moves in sync.
    # mirror_loop removed: master_servo subscribes /master/joint_* directly,
    # so the action stream published by policy_inference drives both slave
    # (via arm_reader mode=1) and master (via arm_master_servo subscribe state).

    # ── state transitions ──
    def _do_takeover(self) -> None:
        """POLICY_RUN → HUMAN_RECORD.

        Architecture: master arms are flashed as 0xFC followers (CAN-controllable).
        arm_master_servo_node accepts /master/enable Bool: True=control state
        (motors hold, accept JointCtrl), False=drag state (motors free, publish
        encoder to /master/joint_*).

        During policy: mirror loop publishes slave_pose → master mirrors slave.
        Takeover sequence:
          1. halt policy (/policy/execute=false)
          2. validate slave pose is non-zero
          3. master_enable=False → arm_master_servo DisableArm + start publishing
             /master/joint_* from encoder. slave's arm_reader follows.
          4. open episode
        """
        with self._lock:
            self._state = State.ALIGNING
        self._publish_state()

        # 1) halt policy publishing /master/joint_* + finalize inference ep
        self.get_logger().info("[TAKEOVER] 1/4 halt policy + close inference episode")
        self.pub_execute.publish(Bool(data=False))
        time.sleep(0.5)
        self._close_inference_episode()

        # 2) validate slave pose
        with self._lock:
            tl = list(self._q_slave_left)
            tr = list(self._q_slave_right)
        zero_l = all(abs(x) < 1e-4 for x in tl[:6])
        zero_r = all(abs(x) < 1e-4 for x in tr[:6])
        if zero_l or zero_r:
            self.get_logger().error(
                f"[TAKEOVER] ABORT — slave pose empty (L_zero={zero_l} R_zero={zero_r})"
            )
            self.pub_execute.publish(Bool(data=True))
            with self._lock:
                self._state = State.POLICY_RUN
            self._publish_state()
            return
        self.get_logger().info(
            f"[TAKEOVER] 2/4 slave OK  L={[round(x,3) for x in tl[:6]]}  R={[round(x,3) for x in tr[:6]]}"
        )

        # Master should already be at slave's pose from the mirror loop. Now
        # we transition master from "control" state (motors hold + accept
        # JointCtrl) to "drag" state (motors free + encoder publishes).

        # 3) master_enable=False → DisableArm + start encoder publishing
        self.get_logger().info("[TAKEOVER] 3/4 switch master to drag state (DisableArm)")
        for _ in range(3):
            self.pub_enable_left.publish(Bool(data=False))
            self.pub_enable_right.publish(Bool(data=False))
            time.sleep(0.1)
        time.sleep(1.2)  # let DisableArm settle and encoder publisher start

        # 4) drag mode ready; pedal will toggle the writer independently
        self.get_logger().info(
            "[TAKEOVER] 4/4 drag mode active — pedal controls recording"
        )
        with self._lock:
            self._state = State.HUMAN_RECORD
            self._recording = False
        self._publish_state()
        self._publish_recording()
        self.get_logger().info(
            "[TAKEOVER] DONE — master is free to drag. "
            "Press pedal (F3) to start/stop recording; toggle switches OFF to handback."
        )

    def _do_handback(self) -> None:
        """HUMAN_RECORD → POLICY_RUN.

        1. finalize episode
        2. master_enable=True → arm_master_servo EnableArm + CAN_CTRL + stop encoder publish
        3. /policy/execute=true → policy resumes
        4. Mirror loop resumes (state=POLICY_RUN), master follows slave again
        """
        with self._lock:
            self._state = State.RETURNING
            had_writer = self._writer is not None
            self._recording = False
        self._publish_state()
        self._publish_recording()

        # Close writer only if pedal had opened one (idempotent if not).
        if had_writer:
            self.get_logger().info("[HANDBACK] 1/3 finalize dagger episode")
            self._close_episode()
        else:
            self.get_logger().info(
                "[HANDBACK] 1/3 skip episode close (pedal never opened a writer)"
            )

        self.get_logger().info("[HANDBACK] 2/3 master_enable=True (EnableArm + CAN_CTRL)")
        for _ in range(3):
            self.pub_enable_left.publish(Bool(data=True))
            self.pub_enable_right.publish(Bool(data=True))
            time.sleep(0.1)
        time.sleep(2.0)  # EnablePiper loop in arm_master_servo can take ~1s

        self.get_logger().info("[HANDBACK] 3/3 resume policy")
        self.pub_execute.publish(Bool(data=True))

        with self._lock:
            self._state = State.POLICY_RUN
        self._publish_state()
        # Open a fresh inference episode for the next policy run (Form C).
        # _on_record_tick will lazy-confirm slave+rgb readiness before the
        # first write, so opening here is safe even if cameras momentarily
        # drop or slave is still settling after handback.
        self._open_inference_episode()
        self.get_logger().info(
            "[HANDBACK] DONE — policy running, master mirroring slave. "
            "Toggle on to record next."
        )

    def _publish_drive(self, ql: list[float], qr: list[float]) -> None:
        now = self.get_clock().now().to_msg()
        msg_l = JointState()
        msg_l.header.stamp = now
        msg_l.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        msg_l.position = ql
        self.pub_drive_left.publish(msg_l)
        msg_r = JointState()
        msg_r.header.stamp = now
        msg_r.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        msg_r.position = qr
        self.pub_drive_right.publish(msg_r)

    # ── episode lifecycle (dagger = human-correction segments) ──
    def _open_episode(self) -> None:
        """Open the dagger episode writer (Form C: human side)."""
        try:
            ep = next_episode_id(self._task, self._subset)
            writer = EpisodeWriter(
                task=self._task, subset=self._subset, ep=ep,
                prompt=self._prompt, template_id="dagger", operator=self._operator,
            )
        except Exception as e:
            self.get_logger().error(f"dagger writer init failed: {e}")
            return
        with self._lock:
            self._writer = writer
            self._started_at = time.time()
            self._wrote_frames = 0
        self.get_logger().info(
            f"  dagger ep={ep} → {writer.root}"
        )

    def _close_episode(self) -> None:
        with self._lock:
            writer = self._writer
            started_at = self._started_at
            wrote = self._wrote_frames
            self._writer = None
        if writer is None:
            return
        duration = time.time() - started_at

        if duration < self._min_ep_sec or wrote < int(self._min_ep_sec * FPS):
            self.get_logger().warn(
                f"  dagger episode too short ({duration:.1f}s, {wrote} frames) — DROPPING"
            )
            try:
                writer.abort()
            except Exception as e:
                self.get_logger().error(f"abort failed: {e}")
            return

        try:
            writer.finalize()
            write_episode_meta(
                writer, duration, success=True,
                note=f"dagger correction ({wrote} frames @ {FPS} Hz)",
                scene_tags=[],
            )
            update_info_json(self._task, self._subset)
            self.get_logger().info(
                f"  saved dagger ep={writer.ep} frames={wrote} duration={duration:.1f}s "
                f"→ {writer.root}"
            )
        except Exception as e:
            self.get_logger().error(f"finalize failed ({e}); aborting episode")
            try:
                writer.abort()
            except Exception:
                pass

    def _discard_episode(self) -> None:
        """Abort the dagger writer WITHOUT finalizing — deletes the half-written
        parquet/mp4 (mirrors data_manager recorder.discard)."""
        with self._lock:
            writer = self._writer
            self._writer = None
        if writer is None:
            return
        try:
            writer.abort()
            self.get_logger().info(
                f"  discarded dagger ep={writer.ep} (partial files deleted)")
        except Exception as e:
            self.get_logger().error(f"discard abort failed: {e}")

    # ── inference episode lifecycle (Form C: policy rollout side) ──
    def _open_inference_episode(self) -> None:
        """Open an inference (policy-rollout) episode writer.

        Writes to <task>/inference/<date-v2>/ — used by the RECAP advantage
        estimator (see docs/deployment/strategy/awbc_implementation_plan.md
        Stage 1) as negative / low-advantage samples.
        """
        try:
            ep = next_episode_id(self._task, "inference")
            writer = EpisodeWriter(
                task=self._task, subset="inference", ep=ep,
                prompt=self._prompt, template_id="inference",
                operator=self._operator,
            )
        except Exception as e:
            self.get_logger().error(f"inference writer init failed: {e}")
            return
        with self._lock:
            self._inference_writer = writer
            self._inf_started_at = time.time()
            self._inf_wrote_frames = 0
        self.get_logger().info(
            f"  inference ep={ep} → {writer.root}"
        )

    def _close_inference_episode(self) -> None:
        with self._lock:
            writer = self._inference_writer
            started_at = self._inf_started_at
            wrote = self._inf_wrote_frames
            self._inference_writer = None
        if writer is None:
            return
        duration = time.time() - started_at

        if duration < self._min_ep_sec or wrote < int(self._min_ep_sec * FPS):
            self.get_logger().warn(
                f"  inference episode too short ({duration:.1f}s, {wrote} frames) — DROPPING"
            )
            try:
                writer.abort()
            except Exception as e:
                self.get_logger().error(f"abort failed: {e}")
            return

        try:
            writer.finalize()
            write_episode_meta(
                writer, duration, success=True,
                note=f"policy rollout ({wrote} frames @ {FPS} Hz)",
                scene_tags=[],
            )
            update_info_json(self._task, "inference")
            self.get_logger().info(
                f"  saved inference ep={writer.ep} frames={wrote} duration={duration:.1f}s "
                f"→ {writer.root}"
            )
        except Exception as e:
            self.get_logger().error(f"inference finalize failed ({e}); aborting episode")
            try:
                writer.abort()
            except Exception:
                pass

    # ── 30 Hz capture (Form C dual-writer dispatch) ──
    def _on_record_tick(self) -> None:
        """Capture one 30 Hz frame to whichever writer is active.

        Dual-writer routing (Form C, see docstring at top + §4.5):
          POLICY_RUN                 → inference_writer (intervention=0);
                                       lazy-opened on first tick with slave
                                       moved + RGB ready.
          HUMAN_RECORD + recording   → dagger writer (intervention=1);
                                       opened by pedal toggle.
          HUMAN_RECORD + !recording  → quiet (drag mode, no write).
          ALIGNING / RETURNING       → quiet (transition windows).

        Lazy-open guard: inference writer can't open during boot zero pose
        (would yield empty parquet) — wait for slave-moved and RGB frame
        before opening. Once opened, _close_inference_episode is the only
        path that resets the writer to None (called by _do_takeover
        step 1/4 or finalize()).
        """
        with self._lock:
            cur_state = self._state
            recording = self._recording
            dag_writer = self._writer
            inf_writer = self._inference_writer
            state = self._q_slave_left + self._q_slave_right
            # KAI0_ACTION_EQ_STATE=1 convention — official kai0_dagger format
            action = list(state)
            frames = {cam: self._rgb[cam] for cam in CAMERAS}
            depth_frames = {cam: self._depth.get(cam) for cam in DEPTH_CAMERAS}
            now = time.time()

        # ── HUMAN_RECORD + recording branch: write to dagger ──
        if cur_state == State.HUMAN_RECORD and recording and dag_writer is not None:
            try:
                dag_writer.write_tick(frames, state, action, now,
                                      depth_frames=depth_frames,
                                      intervention=1)
            except Exception as e:
                self.get_logger().error(
                    f"dagger write_tick failed (aborting recording): {e}"
                )
                with self._lock:
                    self._writer = None
                try:
                    dag_writer.abort()
                except Exception:
                    pass
                return
            with self._lock:
                self._wrote_frames += 1
                n = self._wrote_frames
            if n % (FPS * 5) == 0:
                self.get_logger().info(f"  dagger recording {n} frames ({n / FPS:.1f}s)")
            return

        # ── POLICY_RUN branch: write to inference (lazy-open if needed) ──
        if cur_state != State.POLICY_RUN:
            # ALIGNING / HUMAN_RECORD-not-recording / RETURNING — quiet
            return

        if inf_writer is None:
            # Lazy-open: need slave moved + RGB before opening (else empty parquet).
            with self._lock:
                slave_ready = (any(abs(x) > 1e-4 for x in self._q_slave_left[:6]) and
                               any(abs(x) > 1e-4 for x in self._q_slave_right[:6]))
                rgb_ready = self._rgb.get("top_head") is not None
            if not (slave_ready and rgb_ready):
                return
            self._open_inference_episode()
            with self._lock:
                inf_writer = self._inference_writer
            if inf_writer is None:
                return  # _open_inference_episode logged failure already

        try:
            inf_writer.write_tick(frames, state, action, now,
                                  depth_frames=depth_frames,
                                  intervention=0)
        except Exception as e:
            self.get_logger().error(
                f"inference write_tick failed (aborting): {e}"
            )
            with self._lock:
                self._inference_writer = None
            try:
                inf_writer.abort()
            except Exception:
                pass
            return
        with self._lock:
            self._inf_wrote_frames += 1
            n = self._inf_wrote_frames
        if n % (FPS * 10) == 0:
            self.get_logger().info(f"  inference recording {n} frames ({n / FPS:.1f}s)")
        return

    def finalize(self) -> None:
        """Best-effort cleanup on shutdown — close both writers if active."""
        with self._lock:
            cur = self._state
            dag_open = self._writer is not None
            inf_open = self._inference_writer is not None
        if dag_open or cur == State.HUMAN_RECORD:
            self.get_logger().warn("shutdown during dagger recording — finalizing")
            self._close_episode()
        if inf_open:
            self.get_logger().warn("shutdown during inference recording — finalizing")
            self._close_inference_episode()


def main(args=None):
    rclpy.init(args=args)
    node = DaggerRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
