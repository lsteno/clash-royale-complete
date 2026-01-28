#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <apk_path> [ports...]"
  echo "Example: $0 ./clash.apk 5555 5556 5557"
  exit 2
fi

APK_PATH="$1"
shift

if [[ ! -f "$APK_PATH" ]]; then
  echo "APK not found: $APK_PATH" >&2
  exit 2
fi

PORTS=("$@")
if [[ ${#PORTS[@]} -eq 0 ]]; then
  PORTS=(5555 5556)
fi

for port in "${PORTS[@]}"; do
  serial="localhost:${port}"
  echo "Installing on ${serial}..."
  adb connect "${serial}" || true
  adb -s "${serial}" install -r "$APK_PATH"
  adb -s "${serial}" shell pm grant com.supercell.clashroyale android.permission.WRITE_EXTERNAL_STORAGE || true
done
