#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage (run on your Mac):
  ./clash-royale-mbrl/scripts/mac_machineb_run.sh \
    --vm 20.82.136.186 \
    --vm-port 50051 \
    --key ./ml-2-a10_key.pem \
    --capture-region 0,38,496,1078 \
    --capture-debug-dir ./capture_debug \
    --fps 30 --action-hz 2

What it does:
  1) Opens an SSH tunnel: localhost:<local-port> -> VM 127.0.0.1:<vm-port>
  2) Runs remote_client_loop.py sending frames and (optionally) requesting actions

Tip:
  - Use `python clash-royale-mbrl/src/utils/window_finder.py` to get capture-region.
EOF
}

VM=
VM_PORT=50051
LOCAL_PORT=50051
LOCAL_PORT_SET=0
KEY=./ml-2-a10_key.pem
CAPTURE_REGION=
CAPTURE_DEBUG_DIR=
FPS=30
ACTION_HZ=2
SCRCPY_TITLE=Android
PYTHON_BIN="${PYTHON_BIN:-python3}"

declare -a EXTRA_CLIENT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm) VM="$2"; shift 2;;
    --vm-port) VM_PORT="$2"; shift 2;;
    --local-port) LOCAL_PORT="$2"; LOCAL_PORT_SET=1; shift 2;;
    --key) KEY="$2"; shift 2;;
    --capture-region) CAPTURE_REGION="$2"; shift 2;;
    --capture-debug-dir) CAPTURE_DEBUG_DIR="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --action-hz) ACTION_HZ="$2"; shift 2;;
    --scrcpy-title) SCRCPY_TITLE="$2"; shift 2;;
    --) shift; EXTRA_CLIENT_ARGS+=("$@"); break;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${VM}" ]]; then
  echo "--vm is required" >&2
  usage
  exit 2
fi
if [[ -z "${CAPTURE_REGION}" ]]; then
  echo "--capture-region is required (left,top,width,height)" >&2
  usage
  exit 2
fi
if [[ ! -f "${KEY}" ]]; then
  echo "SSH key not found: ${KEY}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "${port}" >/dev/null 2>&1
    return $?
  fi
  return 1
}

_wait_for_local_listener() {
  local port="$1"
  local deadline_s="${2:-5}"
  local t0
  t0="$(date +%s)"
  while true; do
    if _port_in_use "${port}"; then
      return 0
    fi
    if [[ $(( $(date +%s) - t0 )) -ge "${deadline_s}" ]]; then
      return 1
    fi
    sleep 0.1
  done
}

cleanup() {
  if [[ -n "${TUNNEL_PID:-}" ]]; then
    kill "${TUNNEL_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if _port_in_use "${LOCAL_PORT}"; then
  if [[ "${LOCAL_PORT_SET}" -eq 1 ]]; then
    echo "[mac_machineb_run] Local port ${LOCAL_PORT} is already in use." >&2
    echo "Pick a different local port (example: --local-port 50052), or stop the existing SSH tunnel." >&2
    if command -v lsof >/dev/null 2>&1; then
      lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN || true
    fi
    exit 2
  fi
  BUSY_PORT="${LOCAL_PORT}"
  for p in $(seq 50051 50150); do
    if ! _port_in_use "${p}"; then
      LOCAL_PORT="${p}"
      break
    fi
  done
  if [[ "${LOCAL_PORT}" != "${BUSY_PORT}" ]]; then
    echo "[mac_machineb_run] Local port ${BUSY_PORT} is in use; using ${LOCAL_PORT} instead."
  fi
fi

echo "[mac_machineb_run] Opening SSH tunnel localhost:${LOCAL_PORT} -> ${VM} 127.0.0.1:${VM_PORT}"
ssh -i "${KEY}" -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L "${LOCAL_PORT}:127.0.0.1:${VM_PORT}" \
  "azureuser@${VM}" &
TUNNEL_PID=$!

if ! kill -0 "${TUNNEL_PID}" >/dev/null 2>&1; then
  echo "[mac_machineb_run] SSH tunnel process exited immediately; aborting." >&2
  exit 1
fi
if ! _wait_for_local_listener "${LOCAL_PORT}" 5; then
  echo "[mac_machineb_run] SSH tunnel did not open local port ${LOCAL_PORT}; aborting." >&2
  exit 1
fi

declare -a CLIENT_ARGS=(
  "${PYTHON_BIN}" clash-royale-mbrl/scripts/remote_client_loop.py
  --target "127.0.0.1:${LOCAL_PORT}"
  --want-action
  --fps "${FPS}"
  --action-hz "${ACTION_HZ}"
  --jpeg --jpeg-quality 70
  --scrcpy-title "${SCRCPY_TITLE}"
  --capture-region "${CAPTURE_REGION}"
)
if [[ -n "${CAPTURE_DEBUG_DIR}" ]]; then
  CLIENT_ARGS+=(--capture-debug-dir "${CAPTURE_DEBUG_DIR}")
fi

echo "[mac_machineb_run] Running remote client..."
if [[ ${#EXTRA_CLIENT_ARGS[@]} -gt 0 ]]; then
  exec "${CLIENT_ARGS[@]}" "${EXTRA_CLIENT_ARGS[@]}"
fi
exec "${CLIENT_ARGS[@]}"
