#!/usr/bin/env bash
set -euo pipefail

# Minimal setup script for Debian ARM64 SBC
# - Creates a Python venv
# - Installs Python deps
# - Installs rtsp-simple-server (no apt upgrade)

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
RTSP_VERSION="${RTSP_VERSION:-v1.1.0}"

echo "[1/4] Checking python..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 not found; please install python3/pip before running." >&2
  exit 1
fi

echo "[2/4] Creating venv at $VENV_DIR..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install opencv-python onnxruntime numpy

echo "[4/4] Installing rtsp-simple-server $RTSP_VERSION..."
ARCHIVE="rtsp-simple-server_${RTSP_VERSION}_linux_arm64v8.tar.gz"
URL="https://github.com/aler9/rtsp-simple-server/releases/download/${RTSP_VERSION}/${ARCHIVE}"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
curl -fsSL "$URL" -o "$tmpdir/rtsp.tar.gz"
tar -xzf "$tmpdir/rtsp.tar.gz" -C "$tmpdir"
chmod +x "$tmpdir/rtsp-simple-server"
sudo mv "$tmpdir/rtsp-simple-server" /usr/local/bin/

echo "Done. Activate venv with: source $VENV_DIR/bin/activate"
echo "Start RTSP server: rtsp-simple-server"
