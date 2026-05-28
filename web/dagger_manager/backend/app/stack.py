"""Forks/kills start_dagger_collect.sh and tracks the resulting process group.

Stack manager design mirrors web/data_manager/run.sh's start_svc pattern at
the Python level: setsid + pidfile-written-by-child so we can kill -- -pgid
on stop and reach every ros2 launch grandchild.

Dataset roots are discovered relative to this file so PROJECT_ROOT works
whether the backend runs from a checkout or an installed venv.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

# Resolve PROJECT_ROOT by walking up to the directory containing `kai0/`.
_THIS = Path(__file__).resolve()
PROJECT_ROOT: Optional[Path] = None
for parent in _THIS.parents:
    if (parent / "kai0").is_dir():
        PROJECT_ROOT = parent
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path("/data1/tim/workspace/deepdive_kai0")

START_DAGGER_SH = PROJECT_ROOT / "start_scripts" / "start_dagger_collect.sh"
START_SESSION_SH = PROJECT_ROOT / "start_scripts" / "start_dagger_session.sh"
LOG_DIR = Path(os.environ.get("DAGGER_MANAGER_LOG_DIR",
                              str(PROJECT_ROOT / "web" / "dagger_manager" / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(os.environ.get("KAI0_DATA_ROOT", "/data1/DATA_IMP/KAI0"))
OPTIMIZE_RESULTS = PROJECT_ROOT / "optimize" / "results"


def resolve_v1_pkl(ckpt_dir: Path) -> Optional[Path]:
    """Locate the V1 Triton pickle for a ckpt dir, mirroring
    start_dagger_session.sh / start_autonomy_from_ckpt_v1.sh resolution:
    self-contained <ckpt>/v1_p200.pkl first, then legacy
    optimize/results/<name>_v1_p200.pkl. Returns None if neither exists."""
    self_pkl = ckpt_dir / "v1_p200.pkl"
    if self_pkl.is_file():
        return self_pkl
    legacy = OPTIMIZE_RESULTS / f"{ckpt_dir.name}_v1_p200.pkl"
    if legacy.is_file():
        return legacy
    return None


def variant_for(group: str, ckpt_path: str) -> str:
    """v1 for ckpt_v1/* (group dir name is authoritative; path check is a
    fallback for callers that only have the path). Everything else → v0."""
    if group == "ckpt_v1" or "/ckpt_v1/" in ckpt_path:
        return "v1"
    return "v0"


class StackManager:
    """One-stack-at-a-time manager. Concurrent dagger sessions are not safe
    (share CAN + cameras + GPU + dagger_recorder singleton state)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[Path] = None
        self._ckpt: Optional[str] = None
        self._task: Optional[str] = None

    # ── public ──
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            return {
                "running": running,
                "pid": self._proc.pid if running else None,
                "log": str(self._log_path) if self._log_path else None,
                "ckpt": self._ckpt,
                "task": self._task,
            }

    def start(self, ckpt: str, task: Optional[str], subset: str,
              prompt: Optional[str]) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError(
                    f"stack already running (pid={self._proc.pid}); stop first"
                )
            if not START_DAGGER_SH.is_file():
                raise FileNotFoundError(f"start_dagger_collect.sh not at {START_DAGGER_SH}")
            # ckpt is optional for infra start (policy isn't loaded here).
            # If provided, validate the sidecar so misconfigured ckpts surface
            # before fork. If empty, let the shell script's
            # DEFAULT_CHECKPOINT_DIR take over (it still needs a real path
            # because start_autonomy.sh's sidecar parser runs at infra
            # bring-up regardless).
            ckpt_str = (ckpt or "").strip()
            if ckpt_str:
                ckpt_p = Path(ckpt_str)
                if not ckpt_p.is_dir():
                    raise FileNotFoundError(f"ckpt dir not found: {ckpt_str}")
                sidecar = ckpt_p / "train_config.json"
                if not sidecar.is_file():
                    raise FileNotFoundError(
                        f"{sidecar} missing — pack_inference_ckpt.py wasn't run"
                    )
                ckpt_for_shell: Optional[Path] = ckpt_p
            else:
                ckpt_for_shell = None
            log_path = LOG_DIR / f"stack_{time.strftime('%Y%m%d_%H%M%S')}.log"
            args = ["bash", str(START_DAGGER_SH)]
            if ckpt_for_shell is not None:
                args += ["--ckpt", str(ckpt_for_shell)]
            if task:
                args += ["--task", task]
            if subset and subset != "dagger":
                args += ["--subset", subset]
            if prompt:
                args += ["--prompt", prompt]
            log_fh = open(log_path, "w")
            # setsid: new session/process group so kill -- -pgid reaches all
            # ros2 launch grandchildren. Without this the child only kills
            # the bash wrapper.
            self._proc = subprocess.Popen(
                args,
                stdout=log_fh, stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(PROJECT_ROOT),
            )
            self._log_path = log_path
            self._ckpt = str(ckpt_for_shell) if ckpt_for_shell else None
            self._task = task or ""
            return self.status_unlocked()

    def stop(self, timeout: float = 8.0) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                # Already dead — reset state and return.
                self._proc = None
                return {"running": False, "pid": None,
                        "log": str(self._log_path) if self._log_path else None,
                        "ckpt": None, "task": None}
            pid = self._proc.pid
            try:
                os.killpg(pid, signal.SIGINT)
            except Exception:
                pass
        # Wait outside the lock so status() requests can still query us.
        end = time.time() + timeout
        while time.time() < end:
            if self._proc.poll() is not None:
                break
            time.sleep(0.2)
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except Exception:
                    pass
                # Hard kill after another short grace period.
                time.sleep(2.0)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    pass
            self._proc = None
            return {"running": False, "pid": None,
                    "log": str(self._log_path) if self._log_path else None,
                    "ckpt": self._ckpt, "task": self._task}

    def status_unlocked(self) -> dict:
        running = self._proc is not None and self._proc.poll() is None
        return {
            "running": running,
            "pid": self._proc.pid if running else None,
            "log": str(self._log_path) if self._log_path else None,
            "ckpt": self._ckpt,
            "task": self._task,
        }


stack = StackManager()


class SessionManager:
    """policy_inference subprocess lifecycle, decoupled from infra.

    Forks start_dagger_session.sh which runs only policy_inference_node via
    session_launch.py. Assumes infra (cameras/arms/dagger_recorder) is up.
    Multiple sequential sessions are allowed but only one at a time; calling
    start() while one is running raises 409.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[Path] = None
        self._ckpt: Optional[str] = None
        self._started_at: Optional[float] = None

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            return {
                "running": running,
                "pid": self._proc.pid if running else None,
                "log": str(self._log_path) if self._log_path else None,
                "ckpt": self._ckpt,
                "started_at": self._started_at if running else None,
            }

    def start(self, ckpt: str, gpu_id: Optional[str] = None,
              prompt: Optional[str] = None,
              variant: Optional[str] = None) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError(
                    f"session already running (pid={self._proc.pid}); stop first"
                )
            if not START_SESSION_SH.is_file():
                raise FileNotFoundError(f"start_dagger_session.sh not at {START_SESSION_SH}")
            ckpt_p = Path(ckpt)
            if not ckpt_p.is_dir():
                raise FileNotFoundError(f"ckpt dir not found: {ckpt}")
            sidecar = ckpt_p / "train_config.json"
            if not sidecar.is_file():
                raise FileNotFoundError(f"{sidecar} missing (pack_inference_ckpt.py?)")
            # Resolve variant (None → auto from path). For v1, fail early if
            # the Triton pickle can't be located — the serve can't start
            # without it, and a 404 here is friendlier than a dead serve log.
            eff_variant = variant or variant_for(ckpt_p.parent.name, str(ckpt_p))
            if eff_variant == "v1" and resolve_v1_pkl(ckpt_p) is None:
                raise FileNotFoundError(
                    f"v1 ckpt {ckpt_p.name} has no v1_p200.pkl "
                    f"(checked self + optimize/results); convert it first"
                )
            log_path = LOG_DIR / f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"
            args = ["bash", str(START_SESSION_SH), "--ckpt", str(ckpt_p),
                    "--variant", eff_variant]
            if gpu_id is not None:
                args += ["--gpu", str(gpu_id)]
            if prompt:
                args += ["--prompt", prompt]
            log_fh = open(log_path, "w")
            self._proc = subprocess.Popen(
                args,
                stdout=log_fh, stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(PROJECT_ROOT),
            )
            self._log_path = log_path
            self._ckpt = str(ckpt_p)
            self._started_at = time.time()
            return self._status_unlocked()

    def stop(self, timeout: float = 6.0) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = None
                return {"running": False, "pid": None,
                        "log": str(self._log_path) if self._log_path else None,
                        "ckpt": None, "started_at": None}
            pid = self._proc.pid
            try:
                os.killpg(pid, signal.SIGINT)
            except Exception:
                pass
        end = time.time() + timeout
        while time.time() < end:
            if self._proc.poll() is not None:
                break
            time.sleep(0.2)
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except Exception:
                    pass
                time.sleep(1.5)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    pass
            prev_ckpt = self._ckpt
            self._proc = None
            self._ckpt = None
            self._started_at = None
            return {"running": False, "pid": None,
                    "log": str(self._log_path) if self._log_path else None,
                    "ckpt": prev_ckpt, "started_at": None}

    def _status_unlocked(self) -> dict:
        running = self._proc is not None and self._proc.poll() is None
        return {
            "running": running,
            "pid": self._proc.pid if running else None,
            "log": str(self._log_path) if self._log_path else None,
            "ckpt": self._ckpt,
            "started_at": self._started_at if running else None,
        }


session = SessionManager()


def system_start_async(ckpt: str, gpu_id: Optional[str] = None,
                       task: Optional[str] = None, prompt: Optional[str] = None,
                       variant: Optional[str] = None,
                       readiness_timeout: float = 30.0) -> None:
    """Start only the session (policy_inference). Infra is now bundled with
    start_dagger_collect.sh — if the web is up, infra is up, so all we do
    here is load the model with the picked ckpt.

    Runs in a background thread so the HTTP request returns immediately.
    Progress is observable via the WS snapshot (session_running flips True
    after JAX loads ~22s).
    """
    from .ros_bridge import bridge
    # Defensive readiness check: if /dagger/state isn't published yet
    # (browser opened web faster than dagger_recorder boot delay), wait
    # briefly so policy_inference doesn't spin on missing topics.
    if bridge.snapshot().get("state") is None:
        end = time.time() + readiness_timeout
        while time.time() < end:
            if bridge.snapshot().get("state") is not None:
                break
            time.sleep(0.5)
    if not session.is_running():
        try:
            session.start(ckpt=ckpt, gpu_id=gpu_id, prompt=prompt, variant=variant)
        except Exception as e:
            print(f"[system_start] session start failed: {e}")


def system_stop() -> dict:
    """Stop the session only — infra is managed by the shell wrapper
    (start_dagger_collect.sh: Ctrl-C the terminal to bring infra down)."""
    return {"session": session.stop()}


# ── ckpt discovery + episode counting (filesystem helpers; no ROS needed) ──

def list_checkpoints(root: str = "/data1/DATA_IMP/checkpoints") -> list[dict]:
    """Enumerate directories under <root>/*/ that look like packed ckpts.

    A "valid" ckpt is one with train_config.json (created by
    pack_inference_ckpt.py). Returns list sorted by group, then mtime desc.
    """
    out: list[dict] = []
    root_p = Path(root)
    if not root_p.is_dir():
        return out
    for group_dir in sorted(root_p.iterdir()):
        if not group_dir.is_dir():
            continue
        for ckpt_dir in sorted(group_dir.iterdir(),
                               key=lambda p: p.stat().st_mtime, reverse=True):
            if not ckpt_dir.is_dir():
                continue
            sidecar = ckpt_dir / "train_config.json"
            has_sidecar = sidecar.is_file()
            config_name: Optional[str] = None
            asset_id: Optional[str] = None
            if has_sidecar:
                try:
                    import json
                    cfg = json.loads(sidecar.read_text())
                    config_name = cfg.get("base_config_name")
                    asset_id = cfg.get("override_asset_id")
                except Exception:
                    pass
            has_norm_stats = False
            if asset_id:
                has_norm_stats = (ckpt_dir / "assets" / asset_id /
                                  "norm_stats.json").is_file()
            variant = variant_for(group_dir.name, str(ckpt_dir))
            # Only resolve the (multi-GB) v1 pickle for v1 ckpts; for v0 it's
            # irrelevant and we skip the stat to keep listing cheap.
            has_v1_pkl = (variant == "v1" and resolve_v1_pkl(ckpt_dir) is not None)
            task_hint: Optional[str] = None
            lower = ckpt_dir.name.lower()
            for letter in ("a", "b", "c", "d", "e"):
                if f"task_{letter}" in lower or f"task{letter}" in lower:
                    task_hint = f"Task_{letter.upper()}"
                    break
            out.append({
                "path": str(ckpt_dir),
                "name": ckpt_dir.name,
                "group": group_dir.name,
                "variant": variant,
                "has_sidecar": has_sidecar,
                "has_norm_stats": has_norm_stats,
                "has_v1_pkl": has_v1_pkl,
                "config_name": config_name,
                "task_hint": task_hint,
            })
    return out


def count_episodes(task: str) -> dict:
    """Count parquet files under <DATA_ROOT>/<task>/{inference,dagger}/<date>-v2/data/.

    Uses parquet count rather than episodes.jsonl line count so a still-running
    session (writer mid-finalize) doesn't get partially counted.
    """
    counts = {"inference": 0, "dagger": 0}
    task_root = DATA_ROOT / task
    if not task_root.is_dir():
        return counts
    for subset in ("inference", "dagger"):
        subset_root = task_root / subset
        if not subset_root.is_dir():
            continue
        total = 0
        for date_dir in subset_root.iterdir():
            if not date_dir.is_dir():
                continue
            data_dir = date_dir / "data"
            if not data_dir.is_dir():
                continue
            for chunk in data_dir.iterdir():
                if chunk.is_dir():
                    total += len(list(chunk.glob("episode_*.parquet")))
        counts[subset] = total
    return counts
