#!/usr/bin/env bash
# Katada VGGT runtime entrypoint — Docker / cloud GPU with built-in S3 pipeline.
set -euo pipefail

ENGINE_ROOT="${KATADA_ENGINE_ROOT:-/opt/katada/katada_vggt_engine}"
export PYTHONPATH="${ENGINE_ROOT}:${PYTHONPATH:-}"
export KATADA_WORK_DIR="${KATADA_WORK_DIR:-/workspace}"

cmd="${1:-run}"

case "$cmd" in
  help)
    cat <<EOF
Katada VGGT Docker — full S3 pipeline (download → reconstruct → upload)

  run [args]   Default command. Runs katada/run_pilot.py (S3 in + out).

Connection (pick one):
  A) Mount file:  -v ./connection.json:/run/connection.json:ro
  B) Env vars:    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET,
                  KATADA_STORAGE_PREFIX  (+ optional AWS_REGION, S3_ENDPOINT_URL, HF_TOKEN)
  C) Both:        env vars + auto-fetch pilot/connection.json from S3

Cloud GPU example (Vast / RunPod):
  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
         S3_BUCKET=katadas3 KATADA_STORAGE_PREFIX=pilot-sg-drone-360
  docker run --gpus all --rm \\
    -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \\
    -e S3_BUCKET -e KATADA_STORAGE_PREFIX -e AWS_REGION -e HF_TOKEN \\
    -v katada-work:/workspace \\
    katada/vggt-runtime:latest

Other commands: poses | version | shell
EOF
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
