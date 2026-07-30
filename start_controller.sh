#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
STATE_DIR="$ROOT_DIR/.launcher_state"
PID_FILE="$STATE_DIR/server.pid"
LOG_FILE="$STATE_DIR/server.log"
META_FILE="$STATE_DIR/runtime.env"
HOST="${MIGA_HOST:-0.0.0.0}"
PORT="${MIGA_PORT:-8000}"
RELOAD="${MIGA_RELOAD:-1}"
MODE="foreground"
UVICORN_EXTRA_ARGS=()

log() {
  echo "[launcher] $*"
}

ensure_state_dir() {
  mkdir -p "$STATE_DIR"
}

ensure_venv() {
  if [ -x "$PYTHON_BIN" ]; then
    return
  fi
  log "Creating project virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
}

check_requirements() {
  "$PYTHON_BIN" - "$REQUIREMENTS_FILE" <<'PY'
import importlib.metadata as md
import pathlib
import sys

requirements_path = pathlib.Path(sys.argv[1])
missing = []
mismatched = []

for raw_line in requirements_path.read_text(encoding='utf-8').splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#') or line.startswith('-'):
        continue
    name = line
    expected = None
    if '==' in line:
        name, expected = line.split('==', 1)
        name = name.strip()
        expected = expected.strip()
    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        missing.append(name)
        continue
    if expected and installed != expected:
        mismatched.append(f"{name}: installed {installed}, expected {expected}")

if missing or mismatched:
    if missing:
        print('Missing packages: ' + ', '.join(missing))
    if mismatched:
        print('Version mismatches: ' + '; '.join(mismatched))
    raise SystemExit(1)

try:
    import numpy as np  # noqa: F401
    import sklearn  # noqa: F401
    from ax.api.client import Client  # noqa: F401
except Exception as exc:
    print(f'Ax runtime check failed: {exc}')
    raise SystemExit(1)
PY
}

install_requirements() {
  log "Installing or updating Python packages in $VENV_DIR"
  "$PYTHON_BIN" -m pip install --upgrade pip
  if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
    log "Installing CPU PyTorch required by Ax"
    "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
  fi
  "$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"
}

run_preflight() {
  cd "$ROOT_DIR"
  ensure_state_dir
  ensure_venv
  if ! check_requirements; then
    install_requirements
    check_requirements
  fi
}

read_pid() {
  if [ ! -f "$PID_FILE" ]; then
    return 1
  fi
  tr -d '[:space:]' < "$PID_FILE"
}

server_running() {
  local pid
  if ! pid="$(read_pid)"; then
    return 1
  fi
  if [ -z "$pid" ]; then
    rm -f "$PID_FILE"
    return 1
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    return 0
  fi
  if [ -d "/proc/$pid" ]; then
    return 0
  fi
  rm -f "$PID_FILE"
  return 1
}

load_runtime_metadata() {
  if [ -f "$META_FILE" ]; then
    # shellcheck disable=SC1090
    . "$META_FILE"
  fi
}

write_runtime_metadata() {
  ensure_state_dir
  cat > "$META_FILE" <<EOF
HOST=$HOST
PORT=$PORT
RELOAD=$RELOAD
EOF
}

build_uvicorn_cmd() {
  UVICORN_CMD=("$PYTHON_BIN" -m uvicorn main:app --host "$HOST" --port "$PORT")
  if [ "$RELOAD" != "0" ]; then
    UVICORN_CMD+=(--reload)
  fi
  UVICORN_CMD+=("${UVICORN_EXTRA_ARGS[@]}")
}

print_status() {
  ensure_state_dir
  load_runtime_metadata
  local status="stopped"
  local pid=""
  if server_running; then
    status="running"
    pid="$(read_pid)"
  fi
  echo "STATUS=$status"
  echo "PID=$pid"
  echo "LOG_FILE=$LOG_FILE"
  echo "META_FILE=$META_FILE"
  echo "HOST=$HOST"
  echo "PORT=$PORT"
  echo "RELOAD=$RELOAD"
  echo "VENV_DIR=$VENV_DIR"
}

start_detached() {
  if [ "$(id -u)" -ne 0 ]; then
    exec sudo env MIGA_HOST="$HOST" MIGA_PORT="$PORT" MIGA_RELOAD="$RELOAD" /bin/bash "$0" --start-detached "${UVICORN_EXTRA_ARGS[@]}"
  fi

  run_preflight
  if server_running; then
    log "Controller already running (pid=$(read_pid))"
    print_status
    return 0
  fi

  build_uvicorn_cmd
  write_runtime_metadata
  cd "$ROOT_DIR"

  {
    echo
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') START host=$HOST port=$PORT reload=$RELOAD ==="
  } >> "$LOG_FILE"

  nohup "${UVICORN_CMD[@]}" >> "$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  sleep 1

  if kill -0 "$pid" >/dev/null 2>&1; then
    log "Controller started in background (pid=$pid)"
    print_status
    return 0
  fi

  log "Controller failed to stay running. Check $LOG_FILE"
  rm -f "$PID_FILE"
  tail -n 40 "$LOG_FILE" || true
  return 1
}

stop_server() {
  if [ "$(id -u)" -ne 0 ]; then
    exec sudo /bin/bash "$0" --stop
  fi

  if ! server_running; then
    log "Controller is not running"
    print_status
    return 0
  fi

  local pid
  pid="$(read_pid)"
  log "Stopping controller (pid=$pid)"
  kill "$pid" >/dev/null 2>&1 || true

  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if kill -0 "$pid" >/dev/null 2>&1; then
    log "Controller did not exit gracefully; sending SIGKILL"
    kill -9 "$pid" >/dev/null 2>&1 || true
  fi

  rm -f "$PID_FILE"
  print_status
}

launch_server_foreground() {
  run_preflight
  build_uvicorn_cmd
  write_runtime_metadata
  cd "$ROOT_DIR"

  log "Using Python: $PYTHON_BIN"
  log "Launching controller on ${HOST}:${PORT} (reload=${RELOAD})"

  if [ "$(id -u)" -eq 0 ]; then
    exec "${UVICORN_CMD[@]}"
  fi
  exec sudo env MIGA_HOST="$HOST" MIGA_PORT="$PORT" MIGA_RELOAD="$RELOAD" "${UVICORN_CMD[@]}"
}

parse_args() {
  for arg in "$@"; do
    case "$arg" in
      --check-only)
        MODE="check_only"
        ;;
      --start-detached)
        MODE="start_detached"
        ;;
      --stop)
        MODE="stop"
        ;;
      --status)
        MODE="status"
        ;;
      --print-log-path)
        MODE="print_log_path"
        ;;
      *)
        UVICORN_EXTRA_ARGS+=("$arg")
        ;;
    esac
  done
}

main() {
  parse_args "$@"

  case "$MODE" in
    check_only)
      run_preflight
      log "Environment check passed."
      ;;
    start_detached)
      start_detached
      ;;
    stop)
      stop_server
      ;;
    status)
      print_status
      ;;
    print_log_path)
      ensure_state_dir
      echo "$LOG_FILE"
      ;;
    *)
      launch_server_foreground
      ;;
  esac
}

main "$@"
