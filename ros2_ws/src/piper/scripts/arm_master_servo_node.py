#!/usr/bin/env python3
"""Servo master arm ROS2 node — replaces arm_teleop_node mode=0 in DAgger flow.

Requires the master arm's firmware role flashed as 0xFC (follower), via
piper_tools/setup_can.sh --setup-roles or start_scripts/flash_master_to_follower.py.
After flash + power-cycle, master accepts CAN_CTRL + JointCtrl.

Two runtime states:
  control state (default after init):
    - Motors enabled, ctrl_mode = CAN_CTRL, MOVE_J
    - Subscribes /master_controled/joint_states → drives master via JointCtrl
    - Does NOT publish /master/joint_states (avoids conflict with policy publisher)

  drag state (entered when /master/enable=False received):
    - DisableArm → motors free → user can drag the master arm
    - Publishes /master/joint_states from encoder readback at 30 Hz
      → slave's arm_reader (mode=1) follows this topic, so master drag → slave follow

Control surface (matches dagger_recorder's existing publishers; arms see these
per-arm via dagger_launch.py remappings):
  Sub /master/enable      (Bool)        True=control, False=drag
  Sub /master_controled/joint_states  (JointState)  drives JointCtrl in control state
  Sub /master/linkage_config (String)   accepted for backward compat ("master"=drag, "slave"=control)
  Pub /master/joint_states (JointState) only in drag state
  Pub /puppet/joint_states (JointState) always — this is the slave arm's feedback
                                         topic name, but we publish master encoder here
                                         too for monitoring (matches arm_teleop mode=0 layout)
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String, Int32

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError as e:
    raise RuntimeError("piper_sdk must be installed (pip install piper_sdk)") from e


# rad → 0.001 deg (firmware unit). 1000 * 180 / pi
JOINT_RAD_TO_MDEG = 57295.7795


class ArmMasterServoNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_master_servo")

        self.declare_parameter("can_port", "can_left_mas")
        self.declare_parameter("speed_percent", 30)  # 0-100, MotionCtrl_2 speed cap
        self.declare_parameter("publish_rate_hz", 30.0)  # encoder publish rate when in drag state
        self.declare_parameter("start_state", "control")  # control | drag
        self.declare_parameter("enable_timeout_s", 30.0)  # longer for fresh power-cycle boot
        self.can_port = self.get_parameter("can_port").value
        self.spd = max(0, min(100, int(self.get_parameter("speed_percent").value)))
        self.publish_rate = float(self.get_parameter("publish_rate_hz").value)
        self.enable_timeout = float(self.get_parameter("enable_timeout_s").value)
        start_state = str(self.get_parameter("start_state").value).lower()

        self.get_logger().info(
            f"arm_master_servo starting: can_port={self.can_port} speed={self.spd}% "
            f"publish_rate={self.publish_rate}Hz start_state={start_state}"
        )

        # Connect via v2 SDK
        self.piper = C_PiperInterface_V2(can_name=self.can_port, judge_flag=False)
        self.piper.ConnectPort()
        time.sleep(0.5)

        # PiperInit sends motor angle/accel/firmware queries — this WAKES UP
        # firmware after a fresh power cycle so it starts publishing status
        # frames. Without this, SDK's polling thread sees all zeros and
        # EnablePiper loop times out.
        try:
            self.piper.PiperInit()
        except Exception as e:
            self.get_logger().warn(f"PiperInit failed: {e}")
        time.sleep(0.5)

        # Wait up to 15s for CAN comms to populate encoder data
        t0 = time.time()
        while time.time() - t0 < 15.0:
            js = self.piper.GetArmJointMsgs().joint_state
            if any(abs(v) > 1 for v in [js.joint_1, js.joint_2, js.joint_3]):
                break
            # Re-issue PiperInit periodically in case first one didn't reach arm
            if int(time.time() - t0) % 3 == 0 and (time.time() - t0) % 1 < 0.3:
                try:
                    self.piper.PiperInit()
                except Exception:
                    pass
            time.sleep(0.3)
        self.get_logger().info(
            f"  init encoder = {[js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]}"
        )

        # State machine
        self._state_lock = threading.Lock()
        self._state = "control"  # set to start_state at end of init
        # First GripperCtrl after connect needs code=0x03 (enable+clear error);
        # subsequent uses code=0x01.
        self._gripper_initialized = False

        # Subscriptions
        # /master/enable toggles between subscribe/publish modes.
        self.create_subscription(Bool, "/master/enable", self._cb_enable, 1)
        # SIMPLIFIED architecture: master_servo subscribes the SAME topic that
        # policy_inference publishes to (/master/joint_states, remapped per arm).
        # In subscribe state, every incoming message → JointCtrl + GripperCtrl on
        # master arm. Slave's arm_reader subscribes the same topic, so master and
        # slave are driven by the same action stream → physically synchronized.
        self.create_subscription(JointState, "/master/joint_states",
                                  self._cb_joint_cmd, 10)
        # Legacy compat (no-op but kept so old dagger_recorder publishes don't error)
        self.create_subscription(JointState, "/master_controled/joint_states",
                                  self._cb_joint_cmd, 10)
        self.create_subscription(String, "/master/linkage_config",
                                  self._cb_linkage, 1)
        self.create_subscription(Int32, "/master/teach_mode", self._cb_teach_mode, 1)

        # Publishers
        # /master/joint_states — published only in drag state so slave follows
        self.pub_master_joints = self.create_publisher(JointState, "/master/joint_states", 1)
        # /puppet/joint_states — master arm's encoder, always published for monitoring
        # (matches arm_teleop_node mode=0 layout — slave nodes use this for /puppet/joint_left/right)
        # Note: in dagger_launch this gets remapped to /puppet_master/joint_left/right.
        self.pub_puppet_joints = self.create_publisher(JointState, "/puppet/joint_states", 1)

        # Initial transition to start_state
        if start_state == "drag":
            self._transition_to_drag()
        else:
            self._transition_to_control()

        # Periodic encoder publisher
        self.create_timer(1.0 / self.publish_rate, self._on_publish_tick)

        # Physical-button auto-detect: teach_status field tracks the arm's freedrive
        # button (LED). Button ON (LED bright) → teach_status=START_RECORDING(0x1)
        # → arm is firmware-compliant → switch to publish state (encoder publishing,
        # slave follows master drag). Button OFF (LED dark) → teach_status=
        # STOP_RECORDING(0x2) → arm is CAN-controllable → switch to subscribe state
        # (master tracks policy/slave via JointCtrl).
        self._auto_switch_lock = threading.Lock()
        self._auto_switch_in_progress = False
        self._last_teach_status = None

        # Debounce: physical switch must hold a new state for ≥ DEBOUNCE_S
        # before it is accepted. Prevents single-poll glitches from triggering
        # spurious state transitions in dagger_recorder (Form C state machine
        # in `dagger_implementation_plan.md` §4.5).
        self.DEBOUNCE_S = 0.3
        self._debounced_button_state: Optional[bool] = None
        self._pending_button_state: Optional[bool] = None
        self._pending_since: Optional[float] = None

        # Latch the latest state for late-joining subscribers (dagger_recorder
        # boot order is not guaranteed). transient_local QoS replays the last
        # message to any subscriber that connects after we publish.
        latched_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_button_state = self.create_publisher(Bool, "/master/button_pressed", latched_qos)
        self.create_timer(0.2, self._poll_button_state)  # 5 Hz

        self.get_logger().info(
            f"arm_master_servo ready: state={self._state} on {self.can_port}"
        )

    # ── state transitions ──
    def _transition_to_control(self) -> bool:
        """Enable motors + set CAN_CTRL. Master holds position, accepts JointCtrl.
        Retries PiperInit periodically — fresh power-cycle boot can take 10-20s
        before firmware responds to enable commands."""
        self.get_logger().info("→ control state (EnablePiper + CAN_CTRL)")
        t0 = time.time()
        enabled = False
        last_init = 0.0
        while time.time() - t0 < self.enable_timeout:
            try:
                if self.piper.EnablePiper():
                    enabled = True
                    break
            except Exception:
                pass
            # Every 3s, re-trigger PiperInit to wake firmware if first attempt missed
            now = time.time()
            if now - last_init >= 3.0:
                try:
                    self.piper.PiperInit()
                except Exception:
                    pass
                last_init = now
                es = list(self.piper.GetArmEnableStatus())
                self.get_logger().info(f"  enable retry @ t={now-t0:.0f}s status={es}")
            time.sleep(0.2)
        if not enabled:
            es = list(self.piper.GetArmEnableStatus())
            self.get_logger().error(
                f"  enable timeout after {self.enable_timeout}s, status={es}\n"
                "  → master CANNOT be CAN-driven. Check: master arm 24V powered?\n"
                "    master firmware role = 0xFC (follower)? Run flash_master_to_follower.py + power-cycle."
            )
            return False

        # Sync firmware-internal desired_pos to current encoder pose BEFORE
        # entering CAN_CTRL. Without this, on a drag→control transition the
        # firmware would drive the arm to whatever desired_pos was last
        # programmed — typically the q_slave snapshot from the ALIGNING phase
        # of a takeover, which is the slave pose at takeover TIME (not the
        # pose the operator dragged to during HUMAN_RECORD). That stale
        # desired causes a visible jump-back, dragging the slave (which
        # mirrors the master via /master/joint_*) with it. The gripper is
        # the most noticeable victim because it has no gravity term to mask
        # the jump. See dagger handback bug 2026-05-26.
        try:
            js = self.piper.GetArmJointMsgs().joint_state
            j_mdeg = (int(js.joint_1), int(js.joint_2), int(js.joint_3),
                      int(js.joint_4), int(js.joint_5), int(js.joint_6))
            self.piper.JointCtrl(*j_mdeg)
            g_um = int(self.piper.GetArmGripperMsgs().gripper_state.grippers_angle)
            g_um = max(0, min(80000, g_um))
            code = 0x01 if self._gripper_initialized else 0x03
            self.piper.GripperCtrl(g_um, 1000, code, 0x00)
            self._gripper_initialized = True
            self.get_logger().info(
                f"  pre-CAN_CTRL desired sync: joint_mdeg={j_mdeg} gripper_um={g_um}"
            )
        except Exception as e:
            self.get_logger().warn(f"  pre-CAN_CTRL desired sync failed: {e}")

        # CAN ctrl mode + MOVE_J + horizontal install (×3 for robustness)
        for _ in range(3):
            self.piper.MotionCtrl_2(0x01, 0x01, self.spd, 0x00, 0, 0x01)
            time.sleep(0.05)
        time.sleep(0.3)
        with self._state_lock:
            self._state = "control"
        s = self.piper.GetArmStatus().arm_status
        self.get_logger().info(
            f"  ✓ control: ctrl_mode={s.ctrl_mode} enable={list(self.piper.GetArmEnableStatus())}"
        )
        return True

    def _transition_to_drag(self) -> None:
        """Drag-teach mode: motors stay ENABLED (gravity-compensated, holds against
        gravity) but compliant to external force — user can drag freely. NOT a
        full motor disable (DisableArm would let the arm fall under gravity).

        Piper achieves this via MotionCtrl_1(grag_teach_ctrl=0x01) = "Start
        teaching record / enter drag teach mode" — firmware switches into a
        zero-impedance controller while keeping motors energized.
        """
        self.get_logger().info("→ drag state (MotionCtrl_1 grag_teach=0x01, motors STAY enabled)")
        # Ensure motors are enabled FIRST (so gravity comp is active)
        try:
            self.piper.EnablePiper()
        except Exception:
            pass
        time.sleep(0.2)
        # Enter drag-teach mode
        try:
            for _ in range(3):
                self.piper.MotionCtrl_1(0x00, 0x00, 0x01)
                time.sleep(0.05)
        except Exception as e:
            self.get_logger().error(f"  MotionCtrl_1(drag-teach) failed: {e}")
        time.sleep(0.5)
        with self._state_lock:
            self._state = "drag"
        es = list(self.piper.GetArmEnableStatus())
        s = self.piper.GetArmStatus().arm_status
        self.get_logger().info(
            f"  ✓ drag: enable={es} ctrl_mode={s.ctrl_mode} teach={s.teach_status}"
        )

    # ── callbacks ──
    def _cb_enable(self, msg: Bool) -> None:
        want_control = bool(msg.data)
        with self._state_lock:
            cur = self._state
        if want_control and cur != "control":
            self._transition_to_control()
        elif (not want_control) and cur != "drag":
            self._transition_to_drag()

    def _poll_button_state(self) -> None:
        """Read teach_status @ 5 Hz, debounce, then publish + auto-toggle state.

          teach_status = START_RECORDING(0x1) = button ON  = LED bright = compliant (drag)
          teach_status = STOP_RECORDING (0x2) = button OFF = LED dark   = CAN-driven (control)

        Debounce logic (DEBOUNCE_S = 0.3s ≈ 1.5 poll cycles): a state change is
        only accepted after the raw reading holds the new value for ≥ DEBOUNCE_S.
        Prevents single-glitch reads (CAN packet loss, transient teach_status
        flicker during arm power-up) from triggering spurious state transitions.

        Publishes Bool(data=button_pressed) on /master/button_pressed (latched
        transient_local QoS, see __init__). dagger_recorder consumes this for
        the Form C state machine (see dagger_implementation_plan.md §4.5).
        """
        try:
            teach = self.piper.GetArmStatus().arm_status.teach_status
        except Exception:
            return
        raw_pressed = (teach == 0x01)
        now = time.monotonic()

        # First sample: accept immediately (no prior state to compare against).
        # If firmware teach state and our forced self._state from init disagree,
        # ALSO trigger an auto-transition so they align. Happens when previous
        # session was killed in drag mode (firmware teach_status=0x01 sticks
        # past process exit) and the new session's _transition_to_control()
        # set self._state="control" → without this fixup the button topic
        # publishes True forever while motors are locked, and downstream
        # consumers (dagger_recorder, web UI) see button=True but cannot
        # actually drag.
        if self._debounced_button_state is None:
            self._debounced_button_state = raw_pressed
            self._last_teach_status = teach
            self.pub_button_state.publish(Bool(data=raw_pressed))
            with self._state_lock:
                cur = self._state
            mismatched = (raw_pressed and cur == "control") or \
                         ((not raw_pressed) and cur == "drag")
            if mismatched:
                target_state = "drag" if raw_pressed else "control"
                self.get_logger().warn(
                    f"[button] first-sample mismatch: firmware teach_status="
                    f"{teach:#x} ({'ON' if raw_pressed else 'OFF'}) vs self._state="
                    f"{cur}; auto-transition to {target_state}"
                )
                with self._auto_switch_lock:
                    if not self._auto_switch_in_progress:
                        self._auto_switch_in_progress = True
                        threading.Thread(
                            target=self._do_auto_transition,
                            args=(target_state,), daemon=True,
                        ).start()
            return

        # Stable: re-publish heartbeat (cheap, keeps transient_local fresh).
        if raw_pressed == self._debounced_button_state:
            self._pending_button_state = None
            self._pending_since = None
            self.pub_button_state.publish(Bool(data=self._debounced_button_state))
            return

        # Change observed: start or continue debounce timer.
        if self._pending_button_state != raw_pressed:
            self._pending_button_state = raw_pressed
            self._pending_since = now
            return

        # Same pending state — check stability.
        if (now - (self._pending_since or now)) < self.DEBOUNCE_S:
            return

        # Debounce passed: commit the change.
        self._debounced_button_state = raw_pressed
        self._pending_button_state = None
        self._pending_since = None
        self._last_teach_status = teach
        self.pub_button_state.publish(Bool(data=raw_pressed))
        self.get_logger().info(
            f"[button] debounced teach_status={teach:#x} → button {'ON' if raw_pressed else 'OFF'}"
        )

        # Trigger auto-transition (unchanged from previous logic).
        target_state = "drag" if raw_pressed else "control"
        with self._state_lock:
            cur = self._state
        if cur == target_state:
            return
        with self._auto_switch_lock:
            if self._auto_switch_in_progress:
                return
            self._auto_switch_in_progress = True
        self.get_logger().info(f"[button] auto-switch to {target_state}")
        threading.Thread(target=self._do_auto_transition,
                         args=(target_state,), daemon=True).start()

    def _do_auto_transition(self, target_state: str) -> None:
        try:
            if target_state == "drag":
                self._transition_to_drag()
            else:
                self._transition_to_control()
        finally:
            with self._auto_switch_lock:
                self._auto_switch_in_progress = False

    def _cb_linkage(self, msg: String) -> None:
        """Backward-compat with upstream's /master/linkage_config.
        "master" → drag state (legacy: master=teach_handle in drag-teach mode)
        "slave"  → control state (legacy: master controlled by slave-side commands)
        """
        config = msg.data.lower().strip()
        if config in ("master", "0xfa", "fa"):
            with self._state_lock:
                cur = self._state
            if cur != "drag":
                self._transition_to_drag()
        elif config in ("slave", "0xfc", "fc"):
            with self._state_lock:
                cur = self._state
            if cur != "control":
                self._transition_to_control()
        else:
            self.get_logger().warn(f"unrecognised linkage_config: {msg.data}")

    def _cb_teach_mode(self, msg: Int32) -> None:
        """Backward-compat: Int32(1) → drag, Int32(0) → control."""
        if msg.data == 1:
            with self._state_lock:
                cur = self._state
            if cur != "drag":
                self._transition_to_drag()
        elif msg.data == 0:
            with self._state_lock:
                cur = self._state
            if cur != "control":
                self._transition_to_control()

    def _cb_joint_cmd(self, msg: JointState) -> None:
        """In control state, drive master via JointCtrl (joints 0-5) +
        GripperCtrl (joint 6 = gripper, separate CAN command per upstream clawvla).
        """
        with self._state_lock:
            if self._state != "control":
                return  # ignore in drag state
        if len(msg.position) < 6:
            return
        try:
            j = [int(round(msg.position[i] * JOINT_RAD_TO_MDEG)) for i in range(6)]
            self.piper.JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])
            # Gripper: position[6] is in METERS (slave publishes
            # grippers_angle/1e6 = meters). Send back as μm (×1e6).
            if len(msg.position) >= 7:
                g_um = int(round(abs(msg.position[6]) * 1_000_000.0))
                g_um = max(0, min(80000, g_um))  # clamp 0-80mm
                code = 0x01 if self._gripper_initialized else 0x03
                self.piper.GripperCtrl(g_um, 1000, code, 0x00)
                self._gripper_initialized = True
        except Exception as e:
            self.get_logger().warn(f"JointCtrl/GripperCtrl failed: {e}")
            return
        # Throttled debug log (every 3s)
        now = time.time()
        if not hasattr(self, "_last_cmd_log"):
            self._last_cmd_log = 0.0
            self._cmd_count = 0
        self._cmd_count += 1
        if now - self._last_cmd_log >= 3.0:
            g = msg.position[6] if len(msg.position) >= 7 else 0
            self.get_logger().info(
                f"[cmd] count={self._cmd_count} target_mdeg={j} gripper_m={g:.4f}"
            )
            self._last_cmd_log = now

    # ── periodic encoder publish ──
    def _on_publish_tick(self) -> None:
        try:
            js = self.piper.GetArmJointMsgs().joint_state
        except Exception:
            return
        # mdeg / 1000 / (180/pi) = rad. 0.001 deg → rad conversion factor:
        #   1 mdeg = 1e-3 deg = 1e-3 * pi/180 rad ≈ 1.7453e-5 rad
        rad = [
            js.joint_1 * 1e-3 * math.pi / 180.0,
            js.joint_2 * 1e-3 * math.pi / 180.0,
            js.joint_3 * 1e-3 * math.pi / 180.0,
            js.joint_4 * 1e-3 * math.pi / 180.0,
            js.joint_5 * 1e-3 * math.pi / 180.0,
            js.joint_6 * 1e-3 * math.pi / 180.0,
        ]
        # Gripper: grippers_angle is in 0.001mm (μm) units. Convert to meters
        # for consistency with /master/joint_*.position[6] convention used by
        # arm_reader (mode=1) slave: it does `joint_data.position[6]*1e6` → μm
        # for GripperCtrl.
        try:
            g_um = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        except Exception:
            g_um = 0
        rad.append(g_um * 1e-6)
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        out.position = rad

        # Always publish puppet_joints (master encoder for monitoring)
        self.pub_puppet_joints.publish(out)

        # Publish master_joints ONLY in drag state (so slave follows user-dragged master)
        with self._state_lock:
            in_drag = self._state == "drag"
        if in_drag:
            self.pub_master_joints.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ArmMasterServoNode()
    # SIGTERM handler: rclpy installs SIGINT but not SIGTERM. Without this,
    # ros2 launch escalation (SIGINT → SIGTERM → SIGKILL) hits SIGKILL before
    # we can reset teach mode / DisconnectPort, leaving stale CAN sockets +
    # firmware in drag mode for the next session.
    import signal
    def _term(_sig, _frm):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Do NOT touch firmware teach mode here. MotionCtrl_1(grag_teach=0x02)
        # was tried as a "reset" but empirically it puts ctrl_mode into
        # TEACHING_MODE(0x2) which sticks past the subsequent CAN_CTRL
        # request from the next session's MotionCtrl_2 (firmware ignores
        # ctrl_mode changes while in TEACHING_MODE). Instead, the next
        # session's _poll_button_state first-sample autocorrects via the
        # `seen_initial_mismatch` path below — if firmware was left in
        # teach_status=0x01 by a crash, polling fires an immediate
        # auto-transition so /master_button_* and self._state agree.
        # Release CAN socket + SDK internal ReadCan thread. Without
        # DisconnectPort, Python GC eventually closes the socket but the
        # background read thread may linger, holding the CAN interface and
        # blocking the next process from opening it cleanly.
        try:
            node.piper.DisconnectPort()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
