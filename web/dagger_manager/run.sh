#!/usr/bin/env bash
# dagger_manager — backend (8788) + frontend (5174)
#
# Companion to data_manager (8787 / 5173) — separate ports so both can run
# at the same time. This manager does NOT start CAN/cameras/teleop itself;
# those come up when the user clicks "Start stack" inside the web UI, which
# spawns start_dagger_collect.sh (which in turn pulls in start_autonomy.sh
# + dagger_launch.py).
#
# Usage:
#   ./run.sh start | stop | restart | status | logs [svc]
# Env overrides:
#   SKIP_DEPS=1     skip pip + npm install
#   PORT_BACKEND=8788 / PORT_FRONTEND=5174

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEB_DIR="$SCRIPT_DIR"
BACKEND_DIR="$WEB_DIR/backend"
FRONTEND_DIR="$WEB_DIR/frontend"
LOG_DIR="$WEB_DIR/logs"
PID_DIR="$WEB_DIR/.pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

ROS_SETUP="/opt/ros/jazzy/setup.bash"
ROS2_WS_SETUP="$REPO_ROOT/ros2_ws/install/setup.bash"
PORT_BACKEND="${PORT_BACKEND:-8788}"
PORT_FRONTEND="${PORT_FRONTEND:-5174}"

SERVICES=(backend frontend)
declare -A SVC_PORT=( [backend]="$PORT_BACKEND" [frontend]="$PORT_FRONTEND" )

# ── helpers ──
log()  { echo -e "\033[36m[$(date +%H:%M:%S)]\033[0m $*"; }
warn() { echo -e "\033[33m[warn]\033[0m $*" >&2; }
err()  { echo -e "\033[31m[err]\033[0m $*" >&2; }

pid_file() { echo "$PID_DIR/$1.pid"; }
log_file() { echo "$LOG_DIR/$1.log"; }

is_running() {
    local svc="$1" pf; pf="$(pid_file "$svc")"
    [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null
}

kill_port() {
    local port="$1" pids
    pids=$(ss -lntp 2>/dev/null | awk -v p=":$port\$" '$4 ~ p' | grep -oP 'pid=\K[0-9]+' | sort -u)
    [[ -z "$pids" ]] && return 0
    log "kill orphan(s) on :$port → $(echo $pids | tr '\n' ' ')"
    kill -TERM $pids 2>/dev/null || true
    sleep 0.5
    kill -KILL $pids 2>/dev/null || true
}

start_svc() {
    local svc="$1"; shift
    local cmd="$*"
    if is_running "$svc"; then
        warn "$svc already running (pid $(cat "$(pid_file "$svc")"))"
        return 0
    fi
    log "start $svc: $cmd"
    local pidf; pidf="$(pid_file "$svc")"
    rm -f "$pidf"
    setsid bash -c 'echo $$ > "$0"; eval "$1"' "$pidf" "$cmd" \
        >"$(log_file "$svc")" 2>&1 &
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        [[ -s "$pidf" ]] && break
        sleep 0.1
    done
    sleep 1
    if is_running "$svc"; then
        log "  -> $svc ok (pid $(cat "$(pid_file "$svc")"))  log: $(log_file "$svc")"
    else
        err "$svc failed to start; see $(log_file "$svc")"
    fi
}

stop_svc() {
    local svc="$1" pf pid
    pf="$(pid_file "$svc")"
    if [[ -f "$pf" ]]; then
        pid="$(cat "$pf")"
        if kill -0 "$pid" 2>/dev/null; then
            log "stop $svc (pid $pid, group)"
            kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
            for _ in 1 2 3 4 5; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            kill -KILL -- -"$pid" 2>/dev/null
        fi
        rm -f "$pf"
    fi
    local port="${SVC_PORT[$svc]:-}"
    [[ -n "$port" ]] && kill_port "$port"
}

# ── actions ──
do_start() {
    for svc in "${!SVC_PORT[@]}"; do
        kill_port "${SVC_PORT[$svc]}"
    done

    # Backend venv: use /usr/bin/python3.12 explicitly (matches ROS2 jazzy's
    # python). NOT --system-site-packages — that flag makes pip skip
    # installations whose package name happens to match something in
    # miniconda's site-packages (e.g. uvicorn from a stale env), leaving
    # .venv/bin/uvicorn missing. rclpy + std_msgs are pulled in at runtime
    # by sourcing ROS in start_svc (PYTHONPATH gets
    # /opt/ros/jazzy/lib/python3.12/site-packages), which works inside a
    # bare venv just fine.
    PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
    if [[ "${SKIP_DEPS:-0}" != "1" ]]; then
        if [[ ! -d "$BACKEND_DIR/.venv" ]]; then
            # Prefer uv (5× faster, no need for apt python3.12-venv which isn't
            # installed on this machine). Falls back to python -m venv which
            # requires python3.12-venv apt pkg to be installed.
            if command -v uv >/dev/null 2>&1; then
                log "create backend venv via uv ($PYTHON_BIN)"
                uv venv --python "$PYTHON_BIN" "$BACKEND_DIR/.venv"
            else
                log "create backend venv via python -m venv ($PYTHON_BIN)"
                "$PYTHON_BIN" -m venv "$BACKEND_DIR/.venv" \
                    || warn "venv creation failed — install python3.12-venv or uv"
            fi
        fi
        if [[ -x "$BACKEND_DIR/.venv/bin/python" ]]; then
            log "sync backend deps"
            if command -v uv >/dev/null 2>&1; then
                # uv pip install — works even when .venv has no pip executable
                # (uv venv skips pip by default). VIRTUAL_ENV directs uv at .venv.
                VIRTUAL_ENV="$BACKEND_DIR/.venv" uv pip install -q -r "$BACKEND_DIR/requirements.txt" \
                    || warn "uv pip install returned non-zero (continuing)"
            else
                "$BACKEND_DIR/.venv/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
                "$BACKEND_DIR/.venv/bin/python" -m pip install -q -r "$BACKEND_DIR/requirements.txt" \
                    || warn "pip install returned non-zero (continuing)"
            fi
        else
            warn "backend venv missing at $BACKEND_DIR/.venv"
        fi
        if [[ -d "$FRONTEND_DIR" ]] && [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
            log "npm install (first time)"
            (cd "$FRONTEND_DIR" && npm install --silent || warn "npm install failed (continuing)")
        fi
    fi

    start_svc backend "source '$ROS_SETUP' && source '$ROS2_WS_SETUP' && cd '$BACKEND_DIR' && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT_BACKEND"
    start_svc frontend "cd '$FRONTEND_DIR' && npm run dev -- --host --port $PORT_FRONTEND"

    echo
    log "all services launched."
    log "  前端:  http://localhost:$PORT_FRONTEND/"
    log "  后端:  http://localhost:$PORT_BACKEND/  (docs: /docs)"
}

do_stop() {
    for svc in frontend backend; do
        stop_svc "$svc"
    done
}

do_status() {
    printf "%-10s %-8s %-8s %s\n" "SERVICE" "STATE" "PID" "LOG"
    for svc in "${SERVICES[@]}"; do
        if is_running "$svc"; then
            printf "%-10s \033[32m%-8s\033[0m %-8s %s\n" "$svc" "running" "$(cat "$(pid_file "$svc")")" "$(log_file "$svc")"
        else
            printf "%-10s \033[31m%-8s\033[0m %-8s %s\n" "$svc" "stopped" "-" "$(log_file "$svc")"
        fi
    done
}

do_logs() {
    local svc="${1:-}"
    if [[ -z "$svc" ]]; then
        tail -n 20 -F "$LOG_DIR"/*.log
    else
        tail -n 50 -F "$(log_file "$svc")"
    fi
}

case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 1; do_start ;;
    status)  do_status ;;
    logs)    shift; do_logs "${1:-}" ;;
    *) echo "usage: $0 {start|stop|restart|status|logs [svc]}"; exit 1 ;;
esac
