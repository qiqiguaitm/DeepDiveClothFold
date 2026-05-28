# dagger_manager — Web UI for DAgger sessions

Independent driver-mode web app for orchestrating DAgger data collection on
the dual-arm Piper rig. Runs alongside `web/data_manager/` on separate ports
(backend 8788 / frontend 5174 vs 8787 / 5173).

## What it does

- **Drive the dagger ROS2 stack**: pick a checkpoint, start
  `start_dagger_collect.sh` from the web, stop with one click. Each stack
  invocation logs to `logs/stack_<timestamp>.log` for triage.
- **Soft controls** (driver fallback when operator can't reach a switch):
  takeover (publish `/dagger/takeover True`), handback (False), record
  toggle (`/dagger/pedal_toggled`), policy execute (`/policy/execute`).
- **Live state**: 5 Hz WebSocket snapshot of state machine, both freedrive
  buttons, pedal age, policy execute, episode counts on disk.
- **Checkpoint picker**: enumerates `/data1/DATA_IMP/checkpoints/*/` and
  flags entries missing `train_config.json` or norm_stats.

## Quick start

```bash
# from anywhere:
cd web/dagger_manager
./run.sh start            # boots backend + frontend, opens ports
./run.sh logs backend     # tail
./run.sh restart
./run.sh stop
```

Then open `http://<ipc-ip>:5174/`. The UI doesn't auto-start the dagger
stack — click "Start stack" with a ckpt selected to fork
`start_dagger_collect.sh`.

## Architecture

```
backend (FastAPI :8788)
├── main.py        endpoints + WS /ws/dagger
├── ros_bridge.py  rclpy node — subs /dagger/state, /master_button_*,
│                  /policy/execute, /dagger/pedal_toggled
│                  pubs /dagger/takeover, /dagger/pedal_toggled,
│                  /policy/execute
├── stack.py       forks start_dagger_collect.sh in own session,
│                  kills via SIGINT→SIGTERM→SIGKILL escalation;
│                  also lists ckpts + counts episodes on disk
├── status_hub.py  5 Hz aggregator + WebSocket broadcast
└── models.py      pydantic schemas

frontend (Vite + React, :5174)
├── App.tsx        WS auto-reconnect, 4-card layout
├── components/
│   ├── StateCard       state badge + LEDs (button L/R + pedal age)
│   ├── ControlsCard    takeover/handback/record/execute buttons,
│   │                   disabled by current state
│   ├── CkptPicker      list + select + start/stop stack
│   └── EpisodesCard    inference + dagger counts on disk
└── api.ts         REST + WS client (vite proxies /api + /ws → :8788)
```

State machine values (from `dagger_recorder_node.py`):
`POLICY_RUN → ALIGNING → PRE_RECORD ↔ HUMAN_RECORD → RETURNING → POLICY_RUN`.

## Why a separate app vs. extending data_manager?

DAgger sessions have a fundamentally different lifecycle than passive
teleop recording — they bring up a policy server, run a state machine, and
own both `inference/` and `dagger/` datasets simultaneously. Folding those
into data_manager's "open / save / discard episode" model would obscure the
state machine and force every change to consider both modes. Independent
apps cost code duplication (camera streaming would have to be re-copied;
intentionally omitted from this MVP) but keep each UI focused.

## Endpoints (REST)

| Method | Path | Notes |
|---|---|---|
| GET  | `/api/health`               | liveness |
| GET  | `/api/dagger/status`        | full snapshot (same shape as WS) |
| GET  | `/api/dagger/ckpts`         | ckpt list under `/data1/DATA_IMP/checkpoints/` |
| POST | `/api/dagger/stack/start`   | `{ckpt, task?, subset?, prompt?}` |
| POST | `/api/dagger/stack/stop`    | SIGINT process group |
| POST | `/api/dagger/takeover`      | `{enable: bool}` → `/dagger/takeover` |
| POST | `/api/dagger/record/toggle` | publishes `/dagger/pedal_toggled` (Empty) |
| POST | `/api/dagger/execute`       | `{enable: bool}` → `/policy/execute` |
| WS   | `/ws/dagger`                | 5 Hz snapshot push (no client→server) |
