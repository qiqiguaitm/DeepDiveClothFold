"""rclpy bridge for dagger_manager.

Owns one ROS2 node + a background spin thread. Web threads (FastAPI handlers)
read/write Python attributes — no rclpy calls from the request thread, so
GIL contention with the spin loop is negligible.

Topics consumed (state mirror, all latched / heartbeat from publishers):
  /dagger/state              std_msgs/String        state-machine snapshot
  /master_button_left        std_msgs/Bool          left freedrive switch
  /master_button_right       std_msgs/Bool          right freedrive switch
  /policy/execute            std_msgs/Bool          policy is publishing /master/joint_*
  /dagger/pedal_toggled      std_msgs/Empty         pedal press event

Topics published (driver controls):
  /dagger/takeover           std_msgs/Bool          True=takeover, False=handback
  /dagger/pedal_toggled      std_msgs/Empty         soft pedal (web button fallback)
  /policy/execute            std_msgs/Bool          enable/halt policy publishing

Topics consumed for the live preview (same as start_data_collect.sh's UI):
  /camera_{f,l,r}/camera/color/image_raw  sensor_msgs/Image  → JPEG MJPEG stream
  /puppet/joint_{left,right}              sensor_msgs/JointState → 14-d arm state

If rclpy is not importable (e.g. running this backend without sourced ROS),
the bridge degrades to no-op so endpoints still return cleanly with state=None.
"""
from __future__ import annotations

import io
import os
import threading
import time
from collections import deque
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from std_msgs.msg import Bool, Empty, String
    from sensor_msgs.msg import Image, JointState
    _ROS_OK = True
except Exception as _e:  # noqa: BLE001
    print(f"[ros_bridge] rclpy import failed ({_e}); running in mock mode")
    _ROS_OK = False

# Imaging deps are independent of rclpy — if missing, the state mirror still
# works; only the camera MJPEG stream is disabled.
try:
    import numpy as _np
    from PIL import Image as _PILImage
    _IMG_OK = True
except Exception as _e:  # noqa: BLE001
    print(f"[ros_bridge] numpy/PIL import failed ({_e}); camera MJPEG disabled")
    _IMG_OK = False

# Fixed 3-camera layout (matches autonomy_launch.py / session_launch.py topic
# defaults). Key = UI tile name; value = ROS2 color image topic.
CAMERA_TOPICS = {
    "top_head":   "/camera_f/camera/color/image_raw",
    "hand_left":  "/camera_l/camera/color/image_raw",
    "hand_right": "/camera_r/camera/color/image_raw",
}
PUPPET_TOPICS = {
    "left":  "/puppet/joint_left",
    "right": "/puppet/joint_right",
}
_TARGET_FPS = 30.0


class DaggerRosBridge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Optional[str] = None
        self._rollout_paused: Optional[bool] = None
        self._recording: Optional[bool] = None
        self._button_left: bool = False
        self._button_right: bool = False
        self._policy_execute: Optional[bool] = None
        self._last_pedal_ts: Optional[float] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._node = None
        self._running = False
        # ── camera preview state (per tile) ──
        self._cam_jpeg: dict[str, bytes] = {k: b"" for k in CAMERA_TOPICS}
        self._cam_event: dict[str, threading.Event] = {
            k: threading.Event() for k in CAMERA_TOPICS
        }
        self._cam_stamps: dict[str, deque] = {
            k: deque(maxlen=60) for k in CAMERA_TOPICS
        }
        self._cam_latency_ms: dict[str, float] = {k: 0.0 for k in CAMERA_TOPICS}
        self._jpeg_quality = int(os.environ.get("KAI0_JPEG_QUALITY", "60"))
        self._jpeg_stride = max(1, int(os.environ.get("KAI0_JPEG_STRIDE", "2")))
        # ── joint state (left/right, 7 dof incl. gripper) ──
        self._joints: dict[str, dict] = {}  # "left"/"right" -> {pos, ts}
        if not _ROS_OK:
            return
        self._start()

    # ── lifecycle ──
    def _start(self) -> None:
        try:
            rclpy.init()
        except Exception:
            # Already initialized by another module — fine.
            pass
        self._node = Node("dagger_manager_bridge")
        # latched / transient-local for state + button (publisher side already
        # configured this way; we mirror so a late subscriber gets the value
        # without waiting for the next refresh)
        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._node.create_subscription(String, "/dagger/state", self._on_state, latched)
        self._node.create_subscription(Bool, "/dagger/rollout_paused", self._on_rollout_paused, latched)
        self._node.create_subscription(Bool, "/dagger/recording", self._on_recording, latched)
        self._node.create_subscription(Bool, "/master_button_left",
                                       lambda m: self._on_button("L", m), latched)
        self._node.create_subscription(Bool, "/master_button_right",
                                       lambda m: self._on_button("R", m), latched)
        self._node.create_subscription(Bool, "/policy/execute", self._on_execute, 5)
        self._node.create_subscription(Empty, "/dagger/pedal_toggled", self._on_pedal, 5)

        # ── Live preview: cameras (BEST_EFFORT sensor QoS) + puppet joints ──
        if _IMG_OK:
            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            for cam, topic in CAMERA_TOPICS.items():
                self._node.create_subscription(
                    Image, topic,
                    lambda msg, k=cam: self._on_cam_image(k, msg),
                    sensor_qos,
                )
        for side, topic in PUPPET_TOPICS.items():
            self._node.create_subscription(
                JointState, topic,
                lambda msg, k=side: self._on_joint(k, msg), 10,
            )

        self._pub_takeover = self._node.create_publisher(Bool, "/dagger/takeover", 1)
        self._pub_pedal = self._node.create_publisher(Empty, "/dagger/pedal_toggled", 5)
        self._pub_rollout_next = self._node.create_publisher(Empty, "/dagger/rollout_next", 1)
        self._pub_execute = self._node.create_publisher(Bool, "/policy/execute", 1)
        self._pub_record_cmd = self._node.create_publisher(String, "/dagger/record_cmd", 5)

        self._running = True
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def shutdown(self) -> None:
        self._running = False
        if self._spin_thread:
            self._spin_thread.join(timeout=1.0)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        # We don't rclpy.shutdown() — uvicorn lifespan may want it left up
        # if anyone else is sharing the context.

    def _spin(self) -> None:
        while self._running and rclpy.ok():
            try:
                rclpy.spin_once(self._node, timeout_sec=0.1)
            except Exception:
                time.sleep(0.05)

    # ── callbacks ──
    def _on_state(self, msg) -> None:
        with self._lock:
            self._state = msg.data

    def _on_rollout_paused(self, msg) -> None:
        with self._lock:
            self._rollout_paused = bool(msg.data)

    def _on_recording(self, msg) -> None:
        with self._lock:
            self._recording = bool(msg.data)

    def _on_button(self, side: str, msg) -> None:
        pressed = bool(msg.data)
        with self._lock:
            if side == "L":
                self._button_left = pressed
            else:
                self._button_right = pressed

    def _on_execute(self, msg) -> None:
        with self._lock:
            self._policy_execute = bool(msg.data)

    def _on_pedal(self, _msg) -> None:
        with self._lock:
            self._last_pedal_ts = time.monotonic()

    def _on_cam_image(self, cam: str, msg) -> None:
        """Encode sensor_msgs/Image → JPEG, store latest + wake MJPEG waiters."""
        now = time.time()
        self._cam_stamps[cam].append(now)
        stamp = msg.header.stamp
        msg_t = stamp.sec + stamp.nanosec * 1e-9
        self._cam_latency_ms[cam] = max(0.0, (now - msg_t) * 1000.0) if msg_t > 0 else 0.0
        try:
            w, h, enc = msg.width, msg.height, msg.encoding
            arr = _np.frombuffer(bytes(msg.data), dtype=_np.uint8)
            if enc in ("rgb8", "bgr8"):
                arr = arr.reshape(h, w, 3)
                if enc == "bgr8":
                    arr = arr[:, :, ::-1]
            elif enc in ("mono8", "8UC1"):
                arr = arr.reshape(h, w)
            else:
                return  # unsupported encoding
            s = self._jpeg_stride
            if s > 1:
                arr = arr[::s, ::s]
            buf = io.BytesIO()
            _PILImage.fromarray(arr).save(buf, format="JPEG", quality=self._jpeg_quality)
            jpeg = buf.getvalue()
            with self._lock:
                self._cam_jpeg[cam] = jpeg
            self._cam_event[cam].set()
        except Exception as e:  # noqa: BLE001
            print(f"[ros_bridge] encode {cam} failed: {e}")

    def _on_joint(self, side: str, msg) -> None:
        with self._lock:
            self._joints[side] = {"pos": list(msg.position), "ts": time.time()}

    # ── camera + joint getters (called from FastAPI request threads) ──
    def get_latest_jpeg(self, cam: str, wait_timeout: float = 2.0) -> Optional[bytes]:
        """Block until a fresh JPEG for `cam`; None on timeout/unknown cam."""
        ev = self._cam_event.get(cam)
        if ev is None:
            return None
        got = ev.wait(timeout=wait_timeout)
        ev.clear()
        if not got:
            return None
        with self._lock:
            return self._cam_jpeg.get(cam) or None

    def get_joint_state(self) -> dict:
        """Latest puppet (slave) joints as the 14-d obs split into L/R + grippers."""
        def _pick(side: str) -> list[float]:
            with self._lock:
                j = self._joints.get(side)
            pos = list(j["pos"]) if j else []
            return (pos + [0.0] * 7)[:7]
        left, right = _pick("left"), _pick("right")
        return {
            "left_joints": left[:6],
            "right_joints": right[:6],
            "left_gripper": left[6],
            "right_gripper": right[6],
        }

    def get_camera_health(self) -> dict[str, dict]:
        now = time.time()
        out: dict[str, dict] = {}
        with self._lock:
            for cam in CAMERA_TOPICS:
                recent = [t for t in self._cam_stamps[cam] if now - t < 1.0]
                fps = round(float(len(recent)), 1)
                out[cam] = {
                    "fps": fps,
                    "target_fps": int(_TARGET_FPS),
                    "dropped": max(0, int(round(_TARGET_FPS - fps))),
                    "latency_ms": round(self._cam_latency_ms[cam], 1),
                }
        return out

    # ── snapshot for status_hub / REST ──
    def snapshot(self) -> dict:
        cams = self.get_camera_health()
        with self._lock:
            return {
                "ros_alive": _ROS_OK and self._running,
                "state": self._state,
                "rollout_paused": self._rollout_paused,
                "recording": self._recording,
                "button_left": self._button_left,
                "button_right": self._button_right,
                "policy_execute": self._policy_execute,
                "last_pedal_ts": self._last_pedal_ts,
                "cameras": cams,
            }

    # ── driver actions (publish) ──
    def publish_takeover(self, enable: bool) -> bool:
        if not self._running or self._node is None:
            return False
        self._pub_takeover.publish(Bool(data=bool(enable)))
        return True

    def publish_pedal(self) -> bool:
        if not self._running or self._node is None:
            return False
        self._pub_pedal.publish(Empty())
        return True

    def publish_execute(self, enable: bool) -> bool:
        if not self._running or self._node is None:
            return False
        self._pub_execute.publish(Bool(data=bool(enable)))
        return True

    def publish_rollout_next(self) -> bool:
        """Single-button per-fold boundary toggle (end fold ↔ start next fold)."""
        if not self._running or self._node is None:
            return False
        self._pub_rollout_next.publish(Empty())
        return True

    def publish_record_cmd(self, cmd: str) -> bool:
        """Send an explicit start/save/discard command to dagger_recorder."""
        if not self._running or self._node is None:
            return False
        self._pub_record_cmd.publish(String(data=str(cmd)))
        return True


# Module-level singleton — same pattern as data_manager/ros_bridge.py.
bridge = DaggerRosBridge()
