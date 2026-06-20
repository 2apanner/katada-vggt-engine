#!/usr/bin/env bash
# Build Docker image for cloud GPU (Vast / RunPod / AWS).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${KATADA_RUNTIME_IMAGE:-katada/vggt-runtime:latest}"

docker build -t "$IMAGE" "$ENGINE_DIR"
echo ">> Built $IMAGE (lightweight — engine cloned from GitHub at runtime)"
echo ">> Run:   ./container/run_cloud_gpu.sh"
