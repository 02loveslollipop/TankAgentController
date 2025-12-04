#!/usr/bin/env bash
set -euo pipefail

# Minimal setup script for Debian ARM64 SBC
# - Creates a Python venv
# - Installs Python deps
# - Installs streaming service (RTSP via MediaMTX or UDP via GStreamer)
#
# Usage:
#   ./setup_env.sh          # Interactive prompt
#   ./setup_env.sh --rtsp   # Install RTSP (MediaMTX) without prompt
#   ./setup_env.sh --udp    # Install UDP (GStreamer) without prompt

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
RTSP_VERSION="${RTSP_VERSION:-v1.9.3}"

# Parse command line arguments
STREAMING_MODE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --rtsp)
      STREAMING_MODE="rtsp"
      shift
      ;;
    --udp)
      STREAMING_MODE="udp"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--rtsp | --udp]"
      exit 1
      ;;
  esac
done

# If no flag provided, prompt the user
if [[ -z "$STREAMING_MODE" ]]; then
  echo ""
  echo "Select streaming service to install:"
  echo "  1) RTSP (MediaMTX) - Traditional RTSP server, good for VLC/media players"
  echo "  2) UDP (GStreamer) - Direct UDP streaming, no relay server needed"
  echo ""
  read -rp "Enter choice [1 or 2]: " choice
  case $choice in
    1)
      STREAMING_MODE="rtsp"
      ;;
    2)
      STREAMING_MODE="udp"
      ;;
    *)
      echo "Invalid choice. Please enter 1 or 2."
      exit 1
      ;;
  esac
fi

echo ""
echo "Selected streaming mode: $STREAMING_MODE"
echo ""

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
pip install opencv-python numpy

# Check if we're on a Rockchip platform
if [[ -f /proc/device-tree/compatible ]]; then
  COMPATIBLE=$(cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' ')
  if [[ "$COMPATIBLE" == *"rk3588"* ]]; then
    PLATFORM="rk3588"
  elif [[ "$COMPATIBLE" == *"rk3568"* ]] || [[ "$COMPATIBLE" == *"rk3566"* ]]; then
    PLATFORM="rk3566"
  fi
fi

if [[ -n "${PLATFORM:-}" ]]; then
  echo "[3.5/4] Detected Rockchip platform: $PLATFORM"
  echo "Installing RKNN Lite runtime..."
  pip install rknn-lite2 || echo "[WARN] rknn-lite2 not found in PyPI, install from Rockchip SDK"
  
  # Download pre-built RKNN model from GitHub releases
  MODEL_URL="https://github.com/02loveslollipop/TankAgentController/releases/download/v1.0.0/bisenetv2_${PLATFORM}.rknn"
  MODEL_DIR="models"
  MODEL_FILE="${MODEL_DIR}/bisenetv2_${PLATFORM}.rknn"
  
  mkdir -p "$MODEL_DIR"
  if [[ ! -f "$MODEL_FILE" ]]; then
    echo "Downloading RKNN model from GitHub releases..."
    curl -fsSL "$MODEL_URL" -o "$MODEL_FILE"
    echo "Model saved to: $MODEL_FILE"
  else
    echo "Model already exists: $MODEL_FILE"
  fi
else
  echo "[3.5/4] Not a Rockchip platform, installing ONNX runtime..."
  pip install onnxruntime
fi

if [[ "$STREAMING_MODE" == "rtsp" ]]; then
  echo "[4/4] Installing MediaMTX (RTSP server) $RTSP_VERSION..."
  ARCHIVE="mediamtx_${RTSP_VERSION}_linux_arm64v8.tar.gz"
  URL="https://github.com/bluenviron/mediamtx/releases/download/${RTSP_VERSION}/${ARCHIVE}"
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT
  curl -fsSL "$URL" -o "$tmpdir/mediamtx.tar.gz"
  tar -xzf "$tmpdir/mediamtx.tar.gz" -C "$tmpdir"
  chmod +x "$tmpdir/mediamtx"
  sudo mv "$tmpdir/mediamtx" /usr/local/bin/

  echo ""
  echo "Done. Activate venv with: source $VENV_DIR/bin/activate"
  echo ""
  echo "To start RTSP server:"
  echo "  mediamtx"
  echo ""
  if [[ -n "${PLATFORM:-}" ]]; then
    echo "To stream with RKNN (on SBC):"
    echo "  python stream_bisenet_rknn.py --model models/bisenetv2_${PLATFORM}.rknn --host <client-ip> --port 5000"
  else
    echo "To stream with ONNX (on SBC):"
    echo "  python stream_bisenet_rtsp.py --model model/bisenetv2.onnx --rtsp rtsp://0.0.0.0:8554/bisenet"
  fi
  echo ""
  echo "To receive (on client):"
  echo "  vlc rtsp://<sbc-ip>:8554/bisenet"
  echo "  ffplay rtsp://<sbc-ip>:8554/bisenet"

elif [[ "$STREAMING_MODE" == "udp" ]]; then
  echo "[4/4] Installing GStreamer dependencies..."
  sudo apt-get update
  sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev

  echo ""
  echo "Done. Activate venv with: source $VENV_DIR/bin/activate"
  echo ""
  if [[ -n "${PLATFORM:-}" ]]; then
    echo "To stream with RKNN (on SBC):"
    echo "  python stream_bisenet_rknn.py --model models/bisenetv2_${PLATFORM}.rknn --host <client-ip> --port 5000"
  else
    echo "To stream with ONNX (on SBC):"
    echo "  python stream_bisenet_udp.py --model model/bisenetv2.onnx --host 0.0.0.0 --port 5000"
  fi
  echo ""
  echo "To receive (on client PC with GStreamer):"
  echo "  gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H264 ! rtph264depay ! decodebin ! autovideosink"
  echo ""
  echo "Or with ffplay:"
  echo "  ffplay -fflags nobuffer -flags low_delay -framedrop udp://@:5000"
fi
