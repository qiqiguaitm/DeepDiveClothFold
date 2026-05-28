#!/usr/bin/env python3
"""USB foot-pedal → /dagger/pedal_toggled bridge.

Standalone ROS2 node that watches a USB HID device (default HID 0483:5750
emitting KEY_F3 — the same pedal wired up by
web/data_manager/backend/tools/pedal_listener.py for non-dagger collection)
and emits a single Empty message per pedal press on /dagger/pedal_toggled.

The dagger_recorder treats each Empty as a toggle in the PRE_RECORD ↔
HUMAN_RECORD pair, matching the official KAI0 'Space' key semantics from
train_deploy_alignment/dagger/agilex/agilex_openpi_dagger_collect.py.

Env overrides (compat with backend/tools/pedal_listener.py):
  PEDAL_VID=0483           USB VID (hex)
  PEDAL_PID=5750           USB PID (hex)
  PEDAL_KEY=KEY_F3         evdev key name
  PEDAL_EDGE=release       'release' (default) | 'press'
  PEDAL_DEBOUNCE_MS=500    same-direction min interval (ms)
  SKIP_PEDAL=1             exit immediately (handy in CI / no-hardware machines)
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

try:
    import evdev
except ImportError as e:
    print(f"[dagger_pedal] evdev not importable ({e}); pedal disabled. "
          "Install via the kai0 venv or set PYTHONPATH.", file=sys.stderr)
    evdev = None  # type: ignore[assignment]


VID = int(os.environ.get("PEDAL_VID", "0483"), 16)
PID = int(os.environ.get("PEDAL_PID", "5750"), 16)
KEY_NAME = os.environ.get("PEDAL_KEY", "KEY_F3")
EDGE = os.environ.get("PEDAL_EDGE", "release").lower()
DEBOUNCE_S = float(os.environ.get("PEDAL_DEBOUNCE_MS", "500")) / 1000.0
RETRY_FIND_S = 2.0
TRIGGER_VALUE = 0 if EDGE == "release" else 1  # evdev: 1=down, 0=up


class DaggerPedal(Node):
    """ROS2 wrapper around an evdev read-loop running in a background thread."""

    def __init__(self) -> None:
        super().__init__("dagger_pedal")
        self.pub = self.create_publisher(Empty, "/dagger/pedal_toggled", 5)
        self._running = True
        self._dev: Optional["evdev.InputDevice"] = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._warned_perm: set[str] = set()
        if evdev is None:
            self.get_logger().error(
                "evdev not available; pedal node idle (no toggles will fire)"
            )
            return
        if os.environ.get("SKIP_PEDAL", "0") == "1":
            self.get_logger().info("SKIP_PEDAL=1 — pedal listener disabled")
            return
        try:
            self._trigger_key = evdev.ecodes.ecodes[KEY_NAME]
        except KeyError:
            self.get_logger().error(f"unknown PEDAL_KEY={KEY_NAME!r}")
            return
        self.get_logger().info(
            f"pedal listener starting — VID:PID={VID:04x}:{PID:04x} "
            f"key={KEY_NAME} edge={EDGE} debounce={DEBOUNCE_S*1000:.0f}ms "
            f"topic=/dagger/pedal_toggled"
        )
        self._thread.start()

    def _find_pedal(self) -> Optional["evdev.InputDevice"]:
        if evdev is None:
            return None
        candidates: list[evdev.InputDevice] = []
        perm_denied: list[str] = []
        for path in sorted(Path("/dev/input").glob("event*")):
            try:
                d = evdev.InputDevice(str(path))
            except PermissionError:
                perm_denied.append(str(path))
                continue
            except OSError:
                continue
            if d.info.vendor == VID and d.info.product == PID:
                candidates.append(d)
            else:
                d.close()
        if not candidates and perm_denied:
            new_paths = [p for p in perm_denied if p not in self._warned_perm]
            if new_paths:
                self.get_logger().warn(
                    f"no /dev/input/event* readable ({len(perm_denied)} perm-denied). "
                    f"Add user to 'input' group: sudo gpasswd -a $USER input"
                )
                self._warned_perm.update(perm_denied)
        if not candidates:
            return None
        # Prefer an interface carrying the trigger key (pedals expose kbd + mouse HID)
        with_key: list[evdev.InputDevice] = []
        for d in candidates:
            caps = d.capabilities().get(evdev.ecodes.EV_KEY, [])
            if self._trigger_key in caps:
                with_key.append(d)
            else:
                d.close()
        if not with_key:
            first = candidates[0]
            for d in candidates[1:]:
                d.close()
            return first
        first = with_key[0]
        for d in with_key[1:]:
            d.close()
        return first

    def _run_session(self, dev: "evdev.InputDevice") -> None:
        try:
            dev.grab()
            grabbed = True
            self.get_logger().info(f"grabbed {dev.path} ({dev.name})")
        except OSError as e:
            grabbed = False
            self.get_logger().warn(f"grab failed on {dev.path}: {e} — continuing")
        last_fire = 0.0
        try:
            for ev in dev.read_loop():
                if not self._running:
                    break
                if ev.type != evdev.ecodes.EV_KEY or ev.code != self._trigger_key:
                    continue
                if ev.value != TRIGGER_VALUE:
                    continue
                now = time.monotonic()
                if now - last_fire < DEBOUNCE_S:
                    continue
                last_fire = now
                self.get_logger().info("pedal fired → /dagger/pedal_toggled")
                self.pub.publish(Empty())
        except OSError as e:
            self.get_logger().warn(f"device read error: {e} — reconnecting")
        finally:
            if grabbed:
                try:
                    dev.ungrab()
                except OSError:
                    pass
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass

    def _loop(self) -> None:
        waiting = False
        while self._running:
            dev = self._find_pedal()
            if dev is None:
                if not waiting:
                    self.get_logger().info(
                        f"waiting for pedal VID:PID={VID:04x}:{PID:04x} ..."
                    )
                    waiting = True
                time.sleep(RETRY_FIND_S)
                continue
            waiting = False
            self._run_session(dev)

    def shutdown(self) -> None:
        self._running = False


def main(args=None):
    rclpy.init(args=args)
    node = DaggerPedal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
