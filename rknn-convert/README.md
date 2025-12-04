# RKNN Conversion Tools

Scripts for converting neural network models to RKNN format for Rockchip NPU acceleration.

## Requirements

- **x86_64 Linux** (RKNN Toolkit does not run on ARM)
- Python 3.8+
- RKNN Toolkit 2 (for RK3588/RK3566/RK3568) or RKNN Toolkit (for RK1808/RV1109/RV1126)
- GitHub CLI (`gh`) for uploading to releases (optional)

## Setup

### 1. Create a Virtual Environment (Recommended)

```bash
cd rknn-convert
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install RKNN Toolkit

```bash
# Install PyTorch first (required for PTH to ONNX conversion)
pip install torch torchvision

# For newer chips (RK3588, RK3566, RK3568)
pip install rknn-toolkit2

# For older chips (RK1808, RV1109, RV1126)
pip install rknn-toolkit
```

> **Note:** RKNN Toolkit only works on x86_64 Linux. If installation fails, you may need to install system dependencies:
> ```bash
> sudo apt install libxslt1-dev zlib1g-dev libglib2.0-dev libsm6 libgl1-mesa-glx
> ```

### 3. Install GitHub CLI (Optional, for uploading models)

```bash
# Debian/Ubuntu
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate with GitHub
gh auth login
```

## Usage

### Convert BiSeNetV2 to RKNN

```bash
# Default: Convert for RK3588
python convert_bisenet.py

# Specify target platform
python convert_bisenet.py --platform rk3566
python convert_bisenet.py --platform rk3568

# Skip download if ONNX already exists
python convert_bisenet.py --skip-download
```

### Upload to GitHub Releases

```bash
# Convert and upload to GitHub Releases (auto-generated tag)
python convert_bisenet.py --upload

# Specify a custom tag
python convert_bisenet.py --upload --tag v1.0.0

# Use a different repository
python convert_bisenet.py --upload --tag v1.0.0 --repo owner/repo
```

### Release Build (Multiple Platforms)

```bash
# Build for both rk3566 and rk3588, then upload both to GitHub Releases
python convert_bisenet.py --release

# Release with custom tag
python convert_bisenet.py --release --tag v1.0.0
```

### Supported Platforms

- `rk3588` (default)
- `rk3566`
- `rk3568`
- `rk3562`
- `rk1808`
- `rv1109`
- `rv1126`

## Output

Converted models are saved to `./models/`:
- `model_final_v2_city.pth` - Downloaded PyTorch model
- `bisenetv2.onnx` - Converted ONNX model
- `bisenetv2_<platform>.rknn` - RKNN model (FP16)

## Workflow

1. Run this script on an x86_64 PC
2. (Optional) Upload to GitHub Releases with `--upload` or `--release`
3. Transfer the `.rknn` file to your Rockchip SBC
4. Use RKNN-Toolkit-Lite on the SBC for inference

## Notes

- Models are converted to FP16 (no INT8 quantization) for better accuracy
- FP16 is natively supported by Rockchip NPUs (RK3566, RK3588, etc.)
- No calibration dataset is required for FP16 conversion
