#!/usr/bin/env bash
set -euo pipefail

# Build script for BiSeNetV2 RKNN conversion
# Creates venv, installs dependencies, and runs the conversion
#
# Usage:
#   ./build.sh                    # Build for rk3588 (default)
#   ./build.sh --platform rk3566  # Build for specific platform
#   ./build.sh --release          # Build for rk3566 & rk3588, upload to GitHub
#   ./build.sh --release --tag v1.0.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "============================================================"
echo "BiSeNetV2 RKNN Build Script"
echo "============================================================"

# Check Python
echo "[1/4] Checking Python..."
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    echo "Error: Python3 not found. Please install Python 3.8+."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "x86_64" ]]; then
    echo "Warning: RKNN Toolkit requires x86_64, but detected: $ARCH"
    echo "RKNN conversion may fail. PyTorch to ONNX conversion will still work."
fi

# Create virtual environment
echo ""
echo "[2/4] Setting up virtual environment..."
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Using existing venv at $VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install dependencies
echo ""
echo "[3/4] Installing dependencies..."
pip install --upgrade pip -q

echo "Installing RKNN Toolkit 2..."
pip install rknn-toolkit2 -q 2>/dev/null || {
    echo "Warning: rknn-toolkit2 not available via pip."
    echo "You may need to install it manually from Rockchip's repository."
    echo "Continuing without RKNN toolkit (PTH->ONNX conversion will still work)."
}

echo "Installing PyTorch..."
pip install "torch<=2.4.0" "torchvision<=0.19.0" --index-url https://download.pytorch.org/whl/cpu -q

echo "Installing other dependencies..."
pip install numpy opencv-python onnx -q

# Run conversion
echo ""
echo "[4/4] Running conversion..."
cd "$SCRIPT_DIR"
python convert_bisenet.py "$@"

echo ""
echo "============================================================"
echo "Build complete!"
echo "============================================================"
