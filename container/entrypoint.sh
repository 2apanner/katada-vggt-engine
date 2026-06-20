#!/usr/bin/env bash
# Katada VGGT runtime entrypoint — bootstrap engine from GitHub, then run S3 pipeline.
set -euo pipefail

# shellcheck source=/bootstrap.sh
source /bootstrap.sh

ENGINE_ROOT="${KATADA_ENGINE_ROOT:-/opt/katada/katada_vggt_engine}"
export PYTHONPATH="${ENGINE_ROOT}:${PYTHONPATH:-}"
export KATADA_WORK_DIR="${KATADA_WORK_DIR:-/workspace}"

cmd="${1:-run}"

case "$cmd" in
  help)
    cat <<EOF
Katada VGGT Docker — lightweight image (engine from GitHub at runtime)

  run [args]   Full S3 pipeline via katada/run_pilot.py

Connection (pick one):
  A) Mount file:  -v ./connection.json:/run/connection.json:ro
  B) Env vars:    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET,
                  KATADA_STORAGE_PREFIX  (+ optional AWS_REGION, HF_TOKEN)

First start clones ${KATADA_ENGINE_REPO:-GitHub} and pip-installs deps (~5-15 min).
Later starts reuse cached engine + deps.

Cloud GPU:
  ./container/run_cloud_gpu.sh

Other: poses | version | shell | bootstrap
EOF
    ;;
  bootstrap)
    echo ">> Bootstrap complete — engine at ${ENGINE_ROOT}"
    ;;
  version)
    python3 -c "from katada.version import ENGINE_VERSION; print(ENGINE_VERSION)"
    ;;
  run)
    shift
    exec python3 "${ENGINE_ROOT}/katada/run_pilot.py" "$@"
    ;;
  poses)
    shift
    exec python3 "${ENGINE_ROOT}/katada/pipeline.py" "$@"
    ;;
  shell)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac
