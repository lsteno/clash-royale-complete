#!/usr/bin/env bash
set -euo pipefail

# Generate Python gRPC stubs from proto definitions.
# Usage: ./scripts/gen_protos.sh
# Requires: python -m pip install grpcio grpcio-tools

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="${ROOT_DIR}/proto"
TMP_OUT="${ROOT_DIR}/.proto_gen"
PKG_OUT="${ROOT_DIR}/src/cr/rpc/v1"

rm -rf "${TMP_OUT}"
mkdir -p "${TMP_OUT}" "${PKG_OUT}"

python -m grpc_tools.protoc \
  -I"${PROTO_DIR}" \
  --python_out="${TMP_OUT}" \
  --grpc_python_out="${TMP_OUT}" \
  --pyi_out="${TMP_OUT}" \
  "${PROTO_DIR}/frame_service.proto"

mv "${TMP_OUT}"/frame_service_pb2.py "${PKG_OUT}/"
mv "${TMP_OUT}"/frame_service_pb2_grpc.py "${PKG_OUT}/"
mv "${TMP_OUT}"/frame_service_pb2.pyi "${PKG_OUT}/"
rm -rf "${TMP_OUT}"
