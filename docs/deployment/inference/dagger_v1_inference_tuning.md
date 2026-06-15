# DAgger V1/v0 inference efficiency tuning

**Date:** 2026-06-15 · **Scope:** deployment-only (no training/dataset-loader/norm_stats change)

## Problem

A V1 (Triton/RTC) ckpt behaved differently on the real machine under **dagger**
than under **standalone V1** (`start_autonomy_v1.sh`), even though
`start_dagger_session.sh` passes the *identical* RTC/timing params
(`inference_rate=20`, `latency_k=6`, `rtc_execute_horizon=12`, `publish_rate=80`,
`transport=shm`, `fast_obs_pipeline`, `pipelined_obs`) and the same
`start_serve_v1.sh`.

Root cause was **not** a missing param but **concurrent system load** unique to
the dagger stack:

- `dagger_recorder_node` — continuously decodes 3× color + depth callbacks, and
  during recording encodes 3× mp4 (PyAV libx264) + writes depth zarr + parquet
  every tick (bursty CPU + GIL/IO).
- `2× arm_master_servo_node` — steady CAN servo loops.
- **head (D435) depth ON** (infra default was `enable_head_depth=auto` →
  `camera_depth_flags.ENABLE_DEPTH_TOP_HEAD=True`) — extra USB3 bandwidth +
  per-frame depth grab in `multi_camera_node`, adding color-frame jitter.

On sim01 this contended with the V1 50 ms (20 Hz) loop + its `ObsPrefetchWorker`
thread, dropping the **achieved** inference rate below 20 Hz. Because RTC params
are rate-coupled (blend window = `latency_k × inference_period`, execute horizon,
publish-rate EMA extrapolation), a lower rate doesn't just slow motion — it
changes the trajectory. Same weights → different real-machine behavior.

## Mitigations (all opt-out)

| Knob | Default | Effect |
|------|---------|--------|
| `KAI0_HEAD_DEPTH` | `0` (dagger) | Head depth OFF for dagger: camera doesn't grab it AND the recorder drops `top_head` from its depth set (no zeros-filled zarr). `=1` re-enables. Teleop (`start_data_collect.sh`) leaves it unset → file default (ON). |
| `KAI0_CPU_PIN` | `1` | CPU-affinity isolation via launch `prefix=` (taskset). `=0` disables all pinning. |
| `KAI0_AFFINITY_INFERENCE` | `0-11,32-43` | Policy node + ObsPrefetchWorker + V1 serve. |
| `KAI0_AFFINITY_CAMERAS` | `12-15,44-47` | `multi_camera_node`. |
| `KAI0_AFFINITY_RECORDER` | `16-23,48-55` | `dagger_recorder_node`. |
| `KAI0_AFFINITY_SERVO` | `24-27,56-59` | 2× `arm_master_servo_node`. |
| `KAI0_RECORDER_NICE` | `10` | `nice` for the recorder (also `ionice -c2 -n7`). |
| `KAI0_VIDEO_CODEC` | `nvenc` (teleop) / `h264` (dagger) | GPU h264_nvenc when PyAV supports it. |
| `KAI0_NVENC_GPU` | `0` | Card the NVENC encode runs on. |

Core map targets the 64-thread EPYC 7543 (32 physical cores; SMT sibling =
`core+32`). Each role gets **whole physical cores** so inference and
recorder/servo never share a hyperthread. Sets are disjoint; 28-31,60-63 left
free. `prefix=''` (any role unset) performs to `shlex.split('')==[]` → no-op, so
non-dagger autonomy is bit-identical.

## GPU encode (NVENC)

`dataset_writer.pick_codec()` selects `h264_nvenc` when `KAI0_VIDEO_CODEC=nvenc`
**and** the linked PyAV exposes it:

- **Teleop recorder** runs under `web/data_manager/backend/.venv` (PyAV 17, NVENC ✓)
  → GPU encode, output is standard H.264/mp4 (decodes everywhere incl. LeRobot).
- **Dagger recorder** runs under `kai0/.venv` (PyAV **13**, pinned via
  `pyproject.toml` override-dependencies, no NVENC) → auto-falls back to libx264.
  Its encode is kept off the inference cores by the affinity isolation above, so
  GPU encode there is unnecessary. (To add it later: a sidecar encoder process
  under backend/.venv — deliberately deferred.)

## Verifying achieved rate

Standalone V1 sets `KAI0_LATENCY_PROFILE=1` (writes `/tmp/kai0_latency_<pid>.csv`,
11 cols incl. cycle ms / image_age). To compare under dagger, add
`export KAI0_LATENCY_PROFILE=1` before the `ros2 launch session_launch.py` in
`start_dagger_session.sh` and diff the `cycle` (≤50 ms ⇒ 20 Hz) and
`t_image_age_ms` columns against a standalone run. Also `ros2 topic hz
/master/joint_left` (~80 expected).

## Build note

Launch-file changes (`autonomy_launch.py`, `dagger_launch.py`,
`session_launch.py`) require `colcon build --packages-select piper` before they
take effect (the start scripts' "source changed" check covers this).
