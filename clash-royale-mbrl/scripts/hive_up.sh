#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

sudo docker compose -f docker/hive/docker-compose.yml up -d

echo
echo "Running containers:"
sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | sed -n '1,20p'
echo
echo "ADB devices:"
adb devices
