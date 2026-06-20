#!/usr/bin/env bash
# Clone Katada engine from GitHub + install Python deps on first container start.
set -euo pipefail

ENGINE_ROOT="${KATADA_ENGINE_ROOT:-/opt/katada/katada_vggt_engine}"
ENGINE_REPO="${KATADA_ENGINE_REPO:-https://github.com/2apanner/katada-vggt-engine.git}"
ENGINE_REF="${KATADA_ENGINE_REF:-main}"
DEPS_MARKER="${ENGINE_ROOT}/.katada_deps_installed"

engine_ready() {
  [[ -f "${ENGINE_ROOT}/demo_colmap.py" ]]
}

ensure_engine() {
  if engine_ready; then
    echo ">> Engine ready at ${ENGINE_ROOT}"
    return 0
  fi
  echo ">> Cloning engine from ${ENGINE_REPO} (${ENGINE_REF})"
  rm -rf "${ENGINE_ROOT}"
  mkdir -p "$(dirname "${ENGINE_ROOT}")"
  git clone --depth 1 --branch "${ENGINE_REF}" "${ENGINE_REPO}" "${ENGINE_ROOT}"
  if ! engine_ready; then
    echo "ERROR: engine clone missing demo_colmap.py" >&2
    exit 1
  fi
  echo ">> Engine cloned to ${ENGINE_ROOT}"
}

ensure_deps() {
  if [[ -f "${DEPS_MARKER}" ]]; then
    echo ">> Python deps already installed"
    return 0
  fi
  echo ">> Installing Python dependencies (first run — may take several minutes)..."
  python3 -m pip install --upgrade pip
  python3 -m pip install --no-cache-dir \
    -r "${ENGINE_ROOT}/requirements.txt" \
    -r "${ENGINE_ROOT}/requirements_demo.txt" \
    huggingface_hub boto3 pillow
  python3 -m pip install --no-cache-dir nerfstudio open3d gsplat opencv-python-headless
  touch "${DEPS_MARKER}"
  echo ">> Dependencies installed"
}

ensure_engine
ensure_deps

export KATADA_ENGINE_ROOT="${ENGINE_ROOT}"
export PYTHONPATH="${ENGINE_ROOT}:${PYTHONPATH:-}"
