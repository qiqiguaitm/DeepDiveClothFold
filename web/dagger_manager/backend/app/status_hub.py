"""WebSocket hub for /ws/dagger snapshot push.

Maintains the set of connected clients, runs a 5 Hz aggregator that pulls
state from ros_bridge + stack + filesystem, then broadcasts to all clients.
Clients only need to consume snapshots — they never send anything.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Set

from fastapi import WebSocket

from .ros_bridge import bridge
from .stack import count_episodes, session, stack


class StatusHub:
    def __init__(self, push_hz: float = 5.0) -> None:
        self._clients: Set[WebSocket] = set()
        self._push_interval = 1.0 / max(0.1, push_hz)
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._last_episode_check = 0.0
        self._cached_episodes = {"inference": 0, "dagger": 0}

    async def attach(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        # Send an immediate snapshot so the client can render on connect.
        try:
            await ws.send_text(json.dumps(self.snapshot()))
        except Exception:
            pass

    async def detach(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    def snapshot(self) -> dict:
        ros = bridge.snapshot()
        st = stack.status()
        sess = session.status()
        # Episodes are filesystem-heavy; cache 2s. WS push @5Hz would
        # otherwise re-scan 5×/sec for no gain.
        now = time.monotonic()
        task = st.get("task") or "Task_A"
        if now - self._last_episode_check > 2.0:
            try:
                self._cached_episodes = count_episodes(task)
            except Exception:
                pass
            self._last_episode_check = now
        return {
            "ts": time.time(),
            "stack_running": st["running"],
            "stack_pid": st["pid"],
            "stack_log_path": st["log"],
            "session_running": sess["running"],
            "session_pid": sess["pid"],
            "session_log_path": sess["log"],
            "session_started_at": sess["started_at"],
            "state": ros.get("state"),
            "recording": ros.get("recording"),
            "button_left": ros.get("button_left", False),
            "button_right": ros.get("button_right", False),
            "policy_execute": ros.get("policy_execute"),
            "last_pedal_ts": ros.get("last_pedal_ts"),
            "ros_alive": ros.get("ros_alive", False),
            "inference_episodes": self._cached_episodes.get("inference", 0),
            "dagger_episodes": self._cached_episodes.get("dagger", 0),
            "ckpt": sess.get("ckpt") or st.get("ckpt"),
            "task": task,
            "cameras": ros.get("cameras", {}),
        }

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                snap = self.snapshot()
                payload = json.dumps(snap)
                # Copy set under lock to allow concurrent attach/detach.
                async with self._lock:
                    clients = list(self._clients)
                dead: list[WebSocket] = []
                for ws in clients:
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        dead.append(ws)
                if dead:
                    async with self._lock:
                        for ws in dead:
                            self._clients.discard(ws)
            except Exception:
                # Loop must never die — log-and-continue.
                pass
            await asyncio.sleep(self._push_interval)


hub = StatusHub()
