#!/usr/bin/env bash
# Build sealed runtime tarball for S3 (Colab downloads + extracts — no git clone).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$ENGINE_DIR/dist}"
mkdir -p "$OUT_DIR"

VERSION="$(python3 -c "import sys; sys.path.insert(0, '$ENGINE_DIR'); from katada.version import ENGINE_VERSION; print(ENGINE_VERSION)")"
GIT_SHA="$(cd "$ENGINE_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
BUNDLE_NAME="katada_vggt_runtime_${VERSION}_${STAMP}.tar.gz"
BUNDLE_PATH="$OUT_DIR/$BUNDLE_NAME"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

STAGE="$WORK/katada_vggt_runtime"
mkdir -p "$STAGE/engine"
rsync -a --exclude '.git' --exclude 'dist' --exclude '__pycache__' \
  "$ENGINE_DIR/" "$STAGE/engine/"

cat > "$STAGE/manifest.json" <<EOF
{
  "format": "katada_vggt_runtime",
  "version": "$VERSION",
  "git_sha": "$GIT_SHA",
  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "engine_path": "engine",
  "entrypoint": "engine/container/entrypoint.sh"
}
EOF

cp "$ENGINE_DIR/container/entrypoint.sh" "$STAGE/entrypoint.sh"
chmod +x "$STAGE/entrypoint.sh"

tar -czf "$BUNDLE_PATH" -C "$WORK" katada_vggt_runtime
echo ">> Bundle: $BUNDLE_PATH ($(du -h "$BUNDLE_PATH" | cut -f1))"
ln -sf "$(basename "$BUNDLE_PATH")" "$OUT_DIR/katada_vggt_runtime_latest.tar.gz"
