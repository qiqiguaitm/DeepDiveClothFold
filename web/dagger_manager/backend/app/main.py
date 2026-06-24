"""FastAPI app for dagger_manager.

Endpoints (all under /api/dagger/* except /ws/dagger):
  GET  /api/health                       liveness
  GET  /api/dagger/status                full status snapshot (REST sibling of /ws)
  GET  /api/dagger/ckpts                 list packed checkpoints with metadata
  POST /api/dagger/stack/start           {ckpt, task?, subset?, prompt?}
  POST /api/dagger/stack/stop            kill -SIGINT process group
  POST /api/dagger/takeover              {enable: bool}  → publishes /dagger/takeover
  POST /api/dagger/record/toggle         soft-pedal: publishes /dagger/pedal_toggled
  POST /api/dagger/execute               {enable: bool}  → publishes /policy/execute
  WS   /ws/dagger                        5Hz snapshot push (same dict as /api/dagger/status)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .episodes import (
    delete_episode as ep_delete,
    episode_video_path as ep_video_path,
    list_episodes as ep_list,
    list_tasks as ep_tasks,
)

from .models import (
    CkptEntry,
    DaggerStatus,
    ExecuteReq,
    StartSessionReq,
    StartStackReq,
    TakeoverReq,
)
from .ros_bridge import bridge
from .stack import list_checkpoints, session, stack, system_start_async, system_stop
from .status_hub import hub


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.start()
    print("[startup] dagger_manager ready; status hub + ROS bridge online")
    yield
    await hub.stop()
    bridge.shutdown()


app = FastAPI(title="dagger_manager", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "dagger_manager"}


@app.get("/api/dagger/status", response_model=DaggerStatus)
def status() -> DaggerStatus:
    s = hub.snapshot()
    return DaggerStatus(**{k: s.get(k) for k in DaggerStatus.model_fields.keys()
                           if k in s})


@app.get("/api/dagger/ckpts", response_model=list[CkptEntry])
def ckpts(root: str = "/data1/DATA_IMP/checkpoints") -> list[CkptEntry]:
    return [CkptEntry(**c) for c in list_checkpoints(root)]


@app.post("/api/dagger/stack/start")
def stack_start(req: StartStackReq) -> dict:
    """Bring up infra (CAN/cameras/arms/dagger_recorder/dagger_pedal).
    Policy is NOT loaded — call /api/dagger/session/start with a ckpt
    after the operator clicks 'Start session' in the web UI."""
    # ckpt is now optional at infra level — pass empty string when omitted
    # so start_dagger_collect.sh falls back to its DEFAULT_CHECKPOINT_DIR
    # (only used for sidecar pre-validation; not actually loaded).
    try:
        return stack.start(ckpt=req.ckpt or "", task=req.task,
                           subset=req.subset, prompt=req.prompt)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/dagger/stack/stop")
def stack_stop() -> dict:
    return stack.stop()


@app.post("/api/dagger/session/start")
def session_start(req: StartSessionReq) -> dict:
    """Fork start_dagger_session.sh → loads JAX policy from req.ckpt.
    Requires infra to be up (start_dagger_collect.sh must have been
    started first) — policy_inference will spin on 'Waiting for sensor
    data…' if cameras/arms aren't publishing."""
    try:
        return session.start(ckpt=req.ckpt, gpu_id=req.gpu_id,
                             prompt=req.prompt, variant=req.variant)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/dagger/session/stop")
def session_stop() -> dict:
    return session.stop()


@app.post("/api/dagger/system/start")
def system_start_ep(req: StartSessionReq) -> dict:
    """One-click: brings up infra (if down) + session, in background.

    Returns immediately. The websocket stream surfaces progress: stack_running
    flips True first, then state goes from null → POLICY_RUN, then
    session_running flips True after JAX loads (~22s).
    """
    import threading
    threading.Thread(
        target=system_start_async,
        kwargs={"ckpt": req.ckpt, "gpu_id": req.gpu_id, "prompt": req.prompt,
                "variant": req.variant},
        daemon=True,
    ).start()
    return {"queued": True, "ckpt": req.ckpt}


@app.post("/api/dagger/system/stop")
def system_stop_ep() -> dict:
    """One-click: stop session + infra."""
    return system_stop()


@app.post("/api/dagger/takeover")
def takeover(req: TakeoverReq) -> dict:
    ok = bridge.publish_takeover(req.enable)
    if not ok:
        raise HTTPException(503, "ROS bridge not alive — start the stack first")
    return {"ok": True, "enable": req.enable}


@app.post("/api/dagger/record/toggle")
def record_toggle() -> dict:
    ok = bridge.publish_pedal()
    if not ok:
        raise HTTPException(503, "ROS bridge not alive")
    return {"ok": True}


@app.post("/api/dagger/rollout/next")
def rollout_next() -> dict:
    """Single-button rollout boundary (toggle). A rollout = one autonomous task
    attempt (any scene, not folding-specific). 1st press = attempt done → finalize
    inference (success) + pause; 2nd press = start next rollout (new inference ep +
    resume, flushes RTC). Only acts in POLICY_RUN; recorder ignores otherwise."""
    if not bridge.publish_rollout_next():
        raise HTTPException(503, "ROS bridge not alive")
    return {"ok": True}


@app.post("/api/dagger/record/start")
def record_start() -> dict:
    """开始: open a new dagger episode (only in HUMAN_RECORD, if not recording)."""
    if not bridge.publish_record_cmd("start"):
        raise HTTPException(503, "ROS bridge not alive")
    return {"ok": True, "cmd": "start"}


@app.post("/api/dagger/record/save")
def record_save() -> dict:
    """保存: finalize + keep the current dagger episode."""
    if not bridge.publish_record_cmd("save"):
        raise HTTPException(503, "ROS bridge not alive")
    return {"ok": True, "cmd": "save"}


@app.post("/api/dagger/record/discard")
def record_discard() -> dict:
    """丢弃: abort the current dagger episode, deleting partial files."""
    if not bridge.publish_record_cmd("discard"):
        raise HTTPException(503, "ROS bridge not alive")
    return {"ok": True, "cmd": "discard"}


@app.post("/api/dagger/execute")
def execute(req: ExecuteReq) -> dict:
    ok = bridge.publish_execute(req.enable)
    if not ok:
        raise HTTPException(503, "ROS bridge not alive")
    return {"ok": True, "enable": req.enable}


# ── Live preview (same as start_data_collect.sh's 3-cam + joints UI) ──

@app.get("/api/joints")
def joints() -> dict:
    """Latest puppet (slave) 14-d arm state for the ArmsPanel."""
    if not hasattr(bridge, "get_joint_state"):
        raise HTTPException(503, "joint stream not available")
    return bridge.get_joint_state()


@app.get("/api/camera/{cam}/mjpeg")
def camera_mjpeg(cam: str):
    """multipart/x-mixed-replace MJPEG stream consumed directly by <img src>."""
    if not hasattr(bridge, "get_latest_jpeg"):
        raise HTTPException(503, "camera stream not available (mock bridge)")

    def gen():
        boundary = b"--frame"
        while True:
            jpeg = bridge.get_latest_jpeg(cam, wait_timeout=2.0)
            if not jpeg:
                continue
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── history episode browse / replay ──
@app.get("/api/dagger/tasks")
def tasks_list() -> list[dict]:
    """All Task_* dirs under the data root (for the manual task picker)."""
    return ep_tasks()


@app.get("/api/dagger/episodes")
def episodes_list(task: str = "Task_A") -> list[dict]:
    """All dagger + inference episodes for a task, newest first."""
    return ep_list(task)


@app.delete("/api/dagger/episodes/{subset}/{date}/{episode_id}")
def episodes_delete(subset: str, date: str, episode_id: int,
                    task: str = "Task_A") -> dict:
    ep_delete(task, subset, date, episode_id)
    return {"deleted": True, "subset": subset, "date": date, "episode_id": episode_id}


@app.get("/api/dagger/episodes/{subset}/{date}/{episode_id}/video/{camera}")
def episodes_video(subset: str, date: str, episode_id: int, camera: str,
                   task: str = "Task_A", raw: bool = False):
    """Stream the episode mp4. Dagger videos are AV1, which Chrome plays but
    Safari/older browsers don't — transcode to H.264 on the fly unless raw=1
    or ffmpeg is unavailable (then serve as-is)."""
    p = ep_video_path(task, subset, date, episode_id, camera)
    if not p.exists():
        raise HTTPException(404, "video missing")
    if raw:
        return FileResponse(p, media_type="video/mp4", filename=p.name)

    import shutil as _sh
    import subprocess as _sp
    if not _sh.which("ffmpeg"):
        return FileResponse(p, media_type="video/mp4", filename=p.name)
    # Already H.264? serve directly (skip transcode cost).
    try:
        probe = _sp.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=nk=1:nw=1", str(p)],
            capture_output=True, text=True, timeout=3,
        )
        if probe.stdout.strip() == "h264":
            return FileResponse(p, media_type="video/mp4", filename=p.name)
    except Exception:
        pass
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(p),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov+faststart",
        "-f", "mp4", "pipe:1",
    ]
    proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL)

    def gen():
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                proc.kill()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="video/mp4")


@app.websocket("/ws/dagger")
async def ws_dagger(ws: WebSocket) -> None:
    await hub.attach(ws)
    try:
        while True:
            # We don't expect client → server messages; this keeps the
            # connection open + drains any pings.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.detach(ws)
