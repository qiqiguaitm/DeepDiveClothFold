#!/bin/bash
###############################################################################
# DAgger 一键启动 (tmux 双窗格)
#
# 单脚本起 tmux session "kai0-dagger":
#   左窗格 : start_dagger_collect.sh --ckpt <path>           (launch 全栈 + 日志)
#   右窗格 : toggle_execute.sh                              (Space 接管/交回)
# 用户 attach 进 session 后:
#   - Ctrl+B → / ←     切换聚焦窗格
#   - Space (在右窗格) 触发 dagger takeover
#   - Ctrl+B d         detach (session 保留运行, 可 tmux attach -t kai0-dagger 回来)
#   - 在左窗格 Ctrl+C  停 launch → 整个 session 退出
#
# 用法:
#   ./start_scripts/start_dagger.sh --ckpt <path> [options...]
# 所有 options 透传给 start_dagger_collect.sh.
###############################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="kai0-dagger"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

# ── prerequisites ──
command -v tmux >/dev/null 2>&1 || fail "tmux not installed (apt install tmux)"

# Detect --ckpt for friendlier error vs deep failure inside start_dagger_collect
# Also verify the flag has a non-empty VALUE (not just the flag itself with nothing
# after it), otherwise start_dagger_collect.sh's `set -e` + shift 2 silently
# aborts and the tmux pane closes immediately with no log on disk.
HAVE_CKPT=false
CKPT_VALUE=""
_argv=("$@")
for ((_i=0; _i<${#_argv[@]}; _i++)); do
    case "${_argv[_i]}" in
        --ckpt|--checkpoint-dir)
            HAVE_CKPT=true
            CKPT_VALUE="${_argv[_i+1]:-}"
            ;;
    esac
done
if [[ "$HAVE_CKPT" = true && -z "$CKPT_VALUE" ]]; then
    echo "[FAIL] --ckpt was given but the next argument is empty/missing." >&2
    echo "        Did you copy/paste the command and lose the path?" >&2
    echo "Usage:  $0 --ckpt /data1/DATA_IMP/checkpoints/<exp>/<step> [other options...]" >&2
    exit 1
fi
if [[ "$HAVE_CKPT" = true && ! -d "$CKPT_VALUE" ]]; then
    echo "[FAIL] --ckpt value is not a directory: $CKPT_VALUE" >&2
    exit 1
fi
if [[ "$HAVE_CKPT" != true ]]; then
    cat >&2 <<EOF
[FAIL] missing --ckpt <path>

Usage:
  $0 --ckpt /data1/DATA_IMP/checkpoints/<exp>/<step> [other options...]

Common options forwarded to start_dagger_collect.sh:
  --task Task_A
  --prompt 'Flatten and fold the cloth.'
  --subset dagger
  --mode ros2|websocket|both
  --no-rerun
EOF
    exit 1
fi

# ── reuse existing session if alive ──
if tmux has-session -t "$SESSION" 2>/dev/null; then
    info "tmux session '$SESSION' already exists — attaching"
    exec tmux attach -t "$SESSION"
fi

# ── escape args for tmux send-keys ──
# We can't just splat $@ — tmux send-keys treats each arg as a keypress.
# Build a single shell-safe command line.
LAUNCH_ARGS=""
for a in "$@"; do
    LAUNCH_ARGS+=" $(printf %q "$a")"
done

LAUNCH_LOG="/tmp/kai0_dagger_session.log"
# tee launch output to a log file so failures survive even if tmux closes the pane.
# `set -o pipefail` ensures the launch's exit code propagates through the pipe.
# Trailing `read` keeps the pane open on failure so the user can scroll back.
LAUNCH_CMD="set -o pipefail; cd $(printf %q "$(dirname "$SCRIPT_DIR")") && \
  bash $(printf %q "$SCRIPT_DIR/start_dagger_collect.sh")$LAUNCH_ARGS 2>&1 | tee $(printf %q "$LAUNCH_LOG"); \
  ec=\$?; echo; echo \"[start_dagger] launch exited with code \$ec — log: $LAUNCH_LOG\"; \
  echo '[start_dagger] press Enter to close pane'; read"
# toggle_execute.sh detects /tmp/kai0_deployment_mode = dagger and switches UX.
# Wait for marker so toggle prints the correct mode banner.
TOGGLE_CMD="cd $(printf %q "$(dirname "$SCRIPT_DIR")") && \
  echo 'waiting for /tmp/kai0_deployment_mode=dagger...' && \
  for i in {1..60}; do [ \"\$(cat /tmp/kai0_deployment_mode 2>/dev/null)\" = dagger ] && break; sleep 1; done && \
  bash $(printf %q "$SCRIPT_DIR/toggle_execute.sh")"

info "spawning tmux session '$SESSION'"

# Left pane (launch). -d = detached create so we can configure before attach.
# -x/-y supply an initial size — without them tmux 3.4+ rejects split-window on
# detached sessions with "size missing". Numbers are a sane default; tmux
# auto-resizes on attach.
TMUX_COLS="${TMUX_COLS:-$(tput cols 2>/dev/null || echo 200)}"
TMUX_ROWS="${TMUX_ROWS:-$(tput lines 2>/dev/null || echo 50)}"
tmux new-session -d -s "$SESSION" -n dagger -x "$TMUX_COLS" -y "$TMUX_ROWS" "$LAUNCH_CMD"

# Right pane (toggle), 40% width on the right.
tmux split-window -h -t "$SESSION:0" -l 40% "$TOGGLE_CMD"

# Status bar reminder
tmux set-option -t "$SESSION" status-left "[#S] " >/dev/null 2>&1 || true
tmux set-option -t "$SESSION" status-right "Ctrl+B+arrow=switch pane | Ctrl+B+d=detach | Ctrl+C in left=stop" >/dev/null 2>&1 || true
tmux select-pane -t "$SESSION:0.1"  # focus toggle pane so Space goes to keyboard reader

# Helpful banner
cat <<EOF
============================================================
 DAgger session started in tmux: $SESSION

 Layout:
   ┌──────────────────────┬──────────────────────┐
   │  launch (logs)       │  toggle (keyboard)   │
   │  start_dagger_       │  Space = take over   │
   │  collect.sh          │  Space = hand back   │
   │                      │  q = baseline        │
   └──────────────────────┴──────────────────────┘

 Keys inside tmux:
   Ctrl+B then ← / →     switch pane focus
   Ctrl+B then d         detach (session stays alive)
   tmux attach -t $SESSION    reattach
   Ctrl+C in LEFT pane   stop entire stack

 Attaching now...
============================================================
EOF
sleep 1
exec tmux attach -t "$SESSION"
