#!/usr/bin/env bash
set -euo pipefail

# Setup script for Rockchip ARM64 SBC (Radxa, Orange Pi, etc.)
# - Installs RKNN Lite2 runtime for NPU acceleration
# - Creates a Python venv with dependencies
# - Downloads pre-built RKNN model from GitHub releases
# - Installs streaming service (RTSP via MediaMTX or UDP via GStreamer)
#
# Prerequisites for RK356X (RK3566/RK3568):
#   Enable NPU via: sudo rsetup -> Overlays -> Manage overlays -> Enable NPU
#   Then restart the system.
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
  
  # Install RKNPU2 driver and rknnlite2 via apt (Radxa OS)
  echo "Installing RKNN Lite2 runtime via apt..."
  sudo apt-get update
  
  # Install python3-rknnlite2 (available on Radxa OS)
  if sudo apt-get install -y python3-rknnlite2 2>/dev/null; then
    echo "Installed python3-rknnlite2 via apt"
  else
    echo "[WARN] python3-rknnlite2 not available via apt"
  fi
  
  # Install platform-specific RKNPU2 driver if needed
  if [[ "$PLATFORM" == "rk3588" ]]; then
    sudo apt-get install -y rknpu2-rk3588 2>/dev/null || echo "[INFO] rknpu2-rk3588 already installed or not needed"
  else
    sudo apt-get install -y rknpu2-rk356x 2>/dev/null || echo "[INFO] rknpu2-rk356x already installed or not needed"
  fi
  
  # Check RKNPU driver version
  echo ""
  echo "Checking RKNPU2 driver..."
  if dmesg | grep -q "Initialized rknpu"; then
    RKNPU_VERSION=$(dmesg | grep "Initialized rknpu" | tail -1 | grep -oP 'rknpu \K[0-9.]+' || echo "unknown")
    echo "RKNPU2 driver version: $RKNPU_VERSION"
  else
    echo "[WARN] RKNPU2 driver not detected. For RK356X, enable NPU via:"
    echo "  sudo rsetup -> Overlays -> Manage overlays -> Enable NPU"
    echo "Then restart the system."
  fi
  
  # Link system rknnlite2 to venv (pip install doesn't work for this package)
  echo ""
  echo "Setting up rknnlite2 in virtual environment..."
  
  VENV_SITE_PACKAGES=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
  SYS_DIST_PACKAGES="/usr/lib/python3/dist-packages"
  
  # Link rknnlite package directory
  if [[ -d "$SYS_DIST_PACKAGES/rknnlite" ]]; then
    ln -sf "$SYS_DIST_PACKAGES/rknnlite" "$VENV_SITE_PACKAGES/" 2>/dev/null || true
    echo "Linked rknnlite module to venv"
  else
    echo "[ERROR] rknnlite not found at $SYS_DIST_PACKAGES/rknnlite"
    echo "Please install with: sudo apt install python3-rknnlite2"
    exit 1
  fi
  
  # Also link any .dist-info or .egg-info for proper package detection
  for info_dir in "$SYS_DIST_PACKAGES"/rknn*info "$SYS_DIST_PACKAGES"/RKNN*info; do
    if [[ -d "$info_dir" ]]; then
      ln -sf "$info_dir" "$VENV_SITE_PACKAGES/" 2>/dev/null || true
      echo "Linked $(basename "$info_dir")"
    fi
  done
  
  # Verify rknnlite is importable
  if .venv/bin/python -c "from rknnlite.api import RKNNLite; print('rknnlite OK')" 2>/dev/null; then
    echo "[OK] rknnlite is working in venv"
  else
    echo "[ERROR] rknnlite import failed. Check installation."
    echo "Try running outside venv: python3 -c 'from rknnlite.api import RKNNLite'"
    exit 1
  fi
  
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
  echo "[3.5/4] Not a Rockchip platform - RKNN not available"
  echo "This script is designed for Rockchip SBCs (RK3566/RK3568/RK3588)"
  exit 1
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
  echo "To stream (on SBC):"
  echo "  python stream_bisenet_rtsp.py --model models/bisenetv2_${PLATFORM}.rknn"
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
  echo "To stream (on SBC):"
  echo "  python stream_bisenet_udp.py --model models/bisenetv2_${PLATFORM}.rknn --host <client-ip> --port 5000"
  echo ""
  echo "To receive (on client PC with GStreamer):"
  echo "  gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H264 ! rtph264depay ! decodebin ! autovideosink"
  echo ""
  echo "Or with ffplay:"
  echo "  ffplay -fflags nobuffer -flags low_delay -framedrop udp://@:5000"
fi
