#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage (run on the Azure VM):
  ./clash-royale-mbrl/scripts/vm_machinea_up.sh \
    --port 50051 \
    --size size12m \
    --logdir /mnt/azureuser/logs_dreamerv3 \
    --dumpdir /mnt/azureuser/dumps

Notes:
  - Uses docker-compose file: docker/machineA/docker-compose.vm.yml
  - Writes debug dumps under <dumpdir>/frame_*/ (frame, boxes, state, obs, action_mask).
EOF
}

PORT=50051
SIZE=size12m
LOGDIR=/mnt/azureuser/logs_dreamerv3
DUMPDIR=/mnt/azureuser/dumps

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --logdir) LOGDIR="$2"; shift 2;;
    --dumpdir) DUMPDIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HOST_RPC_PORT="$PORT"
export HOST_LOGDIR="$LOGDIR"
export HOST_DUMPDIR="$DUMPDIR"
export CR_SIZE="$SIZE"

sudo mkdir -p "$HOST_LOGDIR" "$HOST_DUMPDIR"
sudo chown -R "$USER:$USER" "$HOST_LOGDIR" "$HOST_DUMPDIR"

sudo -E docker-compose -f docker/machineA/docker-compose.vm.yml up -d --build

echo
echo "Running containers:"
sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | sed -n '1,20p'
echo
echo "Tail logs:"
sudo docker logs --tail 50 -f machinea_machinea_1

