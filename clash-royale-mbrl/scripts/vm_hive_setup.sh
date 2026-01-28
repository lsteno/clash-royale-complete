#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Clash Royale Hive Host Setup (Ubuntu 22.04)

This script installs Docker, NVIDIA Container Toolkit, and Redroid kernel
modules required for Android containers (binder/ashmem).
EOF

sudo apt-get update
sudo apt-get install -y docker.io git kmod python3-pip curl

distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L "https://nvidia.github.io/nvidia-docker/${distribution}/nvidia-docker.list" \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

if [[ ! -d "redroid-modules" ]]; then
  git clone https://github.com/remote-android/redroid-modules.git
fi
pushd redroid-modules >/dev/null
sudo make install
popd >/dev/null

echo
echo "Loaded modules (expect binder/ashmem):"
lsmod | grep -E "binder|ashmem" || true
echo
echo "Setup complete."
