#!/usr/bin/env bash
# Katada sealed runtime entrypoint — same image for Colab bundle extract, Vast, RunPod.
set -euo pipefail

ENGINE_ROOT="${KATADA_ENGINE_ROOT:-/opt/katada/katada_vggt_engine}"
export PYTHONPATH="${ENGINE_ROOT}:${PYTHONPATH:-}"

cmd="${1:-help}"

case "$cmd" in
  help)
    cat <<EOF
Katada VGGT runtime container
  poses   — run demo_colmap (args: --input-dir --scene-dir [--use-ba])
  shell   — bash inside container
  version — print engine version
EOF
    ;;
  version)
    python3 -c "from katada.version import ENGINE_VERSION; print(ENGINE_VERSION)"
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
