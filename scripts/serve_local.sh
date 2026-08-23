#!/usr/bin/env bash
# Operator launcher for exactly one loopback ZAP-IT service process.
#
# Usage: scripts/serve_local.sh {start|stop|status|logs|restart}
#
# Start requires the operator to export the live-verified GPU UUID. If no port
# is exported, the helper selects 17891, then 23654, then a verified free port
# in 20000..40000. The service repeats the ss+bind preflight immediately before
# opening its listener, so selection is never treated as a reservation.
#
# Runtime pid/log files are private, ephemeral operator artifacts below
# SLAIF_ZAP_IT_TMP_ROOT. The script only signals a PID whose command line is
# this repository's serve_local.py entrypoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="${SLAIF_ZAP_IT_TMP_ROOT:-/dev/shm/slaif-zap-it}"
RUNTIME_ROOT="$TMP_ROOT/runtime"
PIDFILE="$RUNTIME_ROOT/serve-local.pid"
LOGFILE="$RUNTIME_ROOT/serve-local.log"
HEALTH_TIMEOUT_S="${SLAIF_ZAP_IT_START_TIMEOUT:-600}"
PID_START_TIMEOUT_S="${SLAIF_ZAP_IT_PID_START_TIMEOUT:-10}"

python_bin() {
  printf '%s\n' "${SLAIF_ZAP_IT_PYTHON:-$REPO_ROOT/.venv-gpu/bin/python}"
}

run_python() {
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$(python_bin)" "$@"
}

pid_from_pidfile() {
  [ -f "$PIDFILE" ] || return 1
  local pid
  pid="$(<"$PIDFILE")"
  case "$pid" in
    ''|*[!0-9]*) rm -f -- "$PIDFILE"; return 1 ;;
  esac
  printf '%s\n' "$pid"
}

owned_pid() {
  local pid="$1"
  [ "$pid" -gt 1 ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [ -r "/proc/$pid/cmdline" ] || return 1
  local state script_arg
  state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
  [ "$state" != "Z" ] || return 1
  script_arg="$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n '2p')"
  [ "$script_arg" = "$REPO_ROOT/scripts/serve_local.py" ]
}

wait_for_owned_pid() {
  local pid="$1" waited=0
  while [ "$waited" -lt "$PID_START_TIMEOUT_S" ]; do
    if owned_pid "$pid"; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  return 1
}

is_running() {
  local pid
  pid="$(pid_from_pidfile)" || return 1
  if owned_pid "$pid"; then
    return 0
  fi
  rm -f -- "$PIDFILE"
  return 1
}

prepare_runtime() {
  local python
  python="$(python_bin)"
  [ -x "$python" ] || {
    echo "serve_local: Python interpreter is not executable: $python" >&2
    return 1
  }
  run_python - "$TMP_ROOT" <<'PY'
import sys
from src.runtime.shm import ensure_shm_root

ensure_shm_root(sys.argv[1], min_free_bytes=64 * 1024 * 1024)
PY
  umask 077
  mkdir -p -- "$RUNTIME_ROOT"
  chmod 700 -- "$RUNTIME_ROOT"
}

require_launch_env() {
  : "${SLAIF_ZAP_IT_EXPECTED_GPU_UUID:?SLAIF_ZAP_IT_EXPECTED_GPU_UUID must name the live-verified physical GPU1 UUID}"
  if [ "${SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX:-1}" != "1" ]; then
    echo "serve_local: physical GPU index is fixed at 1" >&2
    return 1
  fi
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1
  export SLAIF_ZAP_IT_HOST=127.0.0.1
  export SLAIF_ZAP_IT_TMP_ROOT="$TMP_ROOT"
}

select_port_if_needed() {
  if [ -n "${SLAIF_ZAP_IT_PORT:-}" ]; then
    return 0
  fi
  export SLAIF_ZAP_IT_PORT
  SLAIF_ZAP_IT_PORT="$(run_python - <<'PY'
from src.runtime.ports import select_candidate_port

print(select_candidate_port().port)
PY
)"
  [ -n "$SLAIF_ZAP_IT_PORT" ] || {
    echo "serve_local: no verified-unused loopback port was found" >&2
    return 1
  }
}

verify_port_free() {
  local port="$1"
  if ss -H -ltn 2>/dev/null \
    | awk -v wanted=":$port" '$4 ~ wanted"$" { found=1 } END { exit found ? 0 : 1 }'; then
    echo "serve_local: port $port already has a listener; refusing to start" >&2
    return 1
  fi
}

wait_healthy() {
  local pid="$1" port="$2" waited=0
  while [ "$waited" -lt "$HEALTH_TIMEOUT_S" ]; do
    if ! owned_pid "$pid"; then
      echo "serve_local: process exited during startup; see $LOGFILE" >&2
      return 1
    fi
    if "$(python_bin)" - "$port" <<'PY' 2>/dev/null
import sys
import urllib.request

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "serve_local: health check did not pass within ${HEALTH_TIMEOUT_S}s" >&2
  return 1
}

cmd_start() {
  require_launch_env
  prepare_runtime
  if is_running; then
    echo "serve_local: already running (pid $(<"$PIDFILE"))"
    return 0
  fi
  select_port_if_needed
  verify_port_free "$SLAIF_ZAP_IT_PORT"
  : > "$LOGFILE"
  chmod 600 -- "$LOGFILE"
  # The small exec wrapper makes setsid run from a non-leader child, so it
  # retains this PID while creating a detached session for the Python service.
  # wait_for_owned_pid below closes the exec/PID-file timing window.
  nohup bash -c 'exec setsid "$@"' zap-it-service "$(python_bin)" \
    "$REPO_ROOT/scripts/serve_local.py" </dev/null >>"$LOGFILE" 2>&1 &
  local pid=$!
  if ! wait_for_owned_pid "$pid"; then
    echo "serve_local: service process did not reach the owned entrypoint" >&2
    if owned_pid "$pid"; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
    return 1
  fi
  umask 077
  printf '%s\n' "$pid" > "$PIDFILE"
  chmod 600 -- "$PIDFILE"
  if wait_healthy "$pid" "$SLAIF_ZAP_IT_PORT"; then
    echo "serve_local: started pid $pid on 127.0.0.1:$SLAIF_ZAP_IT_PORT"
    echo "serve_local: log: $LOGFILE"
    return 0
  fi
  if owned_pid "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
  rm -f -- "$PIDFILE"
  return 1
}

cmd_stop() {
  local pid
  if ! pid="$(pid_from_pidfile)"; then
    echo "serve_local: not running (no valid pidfile)"
    return 0
  fi
  if owned_pid "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
    local waited=0
    while owned_pid "$pid" && [ "$waited" -lt 60 ]; do
      sleep 1
      waited=$((waited + 1))
    done
    if owned_pid "$pid"; then
      echo "serve_local: graceful stop timed out; sending SIGKILL to owned pid $pid" >&2
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f -- "$PIDFILE" "$LOGFILE"
  rmdir -- "$RUNTIME_ROOT" 2>/dev/null || true
  echo "serve_local: stopped"
}

cmd_status() {
  local pid
  if pid="$(pid_from_pidfile)" && owned_pid "$pid"; then
    echo "serve_local: running (pid $pid)"
    return 0
  fi
  echo "serve_local: stopped"
  return 1
}

cmd_logs() {
  if [ -f "$LOGFILE" ]; then
    tail -n "${SLAIF_ZAP_IT_LOG_LINES:-120}" -- "$LOGFILE"
  else
    echo "serve_local: no live log"
    return 1
  fi
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  restart) cmd_stop; cmd_start ;;
  *)
    echo "usage: $0 {start|stop|status|logs|restart}" >&2
    exit 64
    ;;
esac
