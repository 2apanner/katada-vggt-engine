#!/usr/bin/env bash
# Launch Katada Docker on cloud GPU (Vast.ai / RunPod / AWS).
# Requires: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET, KATADA_STORAGE_PREFIX
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${KATADA_RUNTIME_IMAGE:-katada/vggt-runtime:latest}"

: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"
: "${S3_BUCKET:?Set S3_BUCKET}"
: "${KATADA_STORAGE_PREFIX:?Set KATADA_STORAGE_PREFIX}"

AWS_REGION="${AWS_REGION:-ap-southeast-1}"

echo ">> Image:   $IMAGE"
echo ">> Project: $KATADA_STORAGE_PREFIX"
echo ">> Bucket:  $S3_BUCKET"

docker run --gpus all --rm \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION \
  -e S3_BUCKET \
  -e KATADA_STORAGE_PREFIX \
  -e S3_ENDPOINT_URL \
  -e HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN \
  -e VGGT_CHECKPOINT_URL \
  -v katada-work:/workspace \
  "$IMAGE" \
  run
