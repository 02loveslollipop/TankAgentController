# BiSeNetV2 RKNN Segmentation Streaming

Real-time semantic segmentation streaming using BiSeNetV2 on Rockchip NPU (RK3566/RK3568/RK3588).

## Features

- **NPU-accelerated inference** using RKNN Lite2
- **Auto-download** of pre-built RKNN models from GitHub releases
- **UDP streaming** via GStreamer (low latency, no server needed)
- **RTSP streaming** via MediaMTX (compatible with VLC, ffplay)
- **Platform auto-detection** (RK3566/RK3568/RK3588)

## Requirements

### Hardware

- Rockchip SBC with NPU:
  - RK3566 (e.g., Radxa Zero 3)
  - RK3568 (e.g., Radxa E25, Rock 3A)
  - RK3588/RK3588S (e.g., Radxa Rock 5, Orange Pi 5)
- USB/CSI camera

### Software

- Radxa OS or compatible Debian-based Linux
- Python 3.10+
- RKNPU2 driver (usually pre-installed)
- RKNN Toolkit Lite2

## Setup

### 1. Enable NPU (RK356X only) (Radxa systems)

For RK3566/RK3568 boards, you must enable the NPU overlay:

```bash
sudo rsetup
# Navigate to: Overlays -> Manage overlays -> Enable NPU
# Save and reboot
```

> **Note:** If "Enable NPU" is not available, update your system first:
> `sudo rsetup` -> System -> System Update, then reboot and try again.

### 2. Run Setup Script

```bash
cd "samples/RKNN test"
chmod +x setup_env.sh
./setup_env.sh --udp    # For UDP streaming (recommended)
# or
./setup_env.sh --rtsp   # For RTSP streaming
```

The setup script will:
1. Create a Python virtual environment
2. Install `python3-rknnlite2` and RKNPU2 drivers
3. Download the pre-built RKNN model from GitHub releases
4. Install GStreamer (UDP) or MediaMTX (RTSP)

### 3. Verify RKNPU Driver

```bash
sudo dmesg | grep "Initialized rknpu"
# Expected output:
# [   15.522298] [drm] Initialized rknpu 0.9.6 20240322 for fdab0000.npu on minor 1
```

> **Note:** RK356X systems have driver version 0.8.8, RK3588 has 0.9.6+

## Usage

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

### UDP Streaming (Recommended)

**On SBC (sender):**
```bash
# Auto-detect platform and download model
python stream_bisenet_udp.py --host <client-ip> --port 5000

# Or specify model explicitly
python stream_bisenet_udp.py --model models/bisenetv2_rk3566.rknn --host 192.168.1.100 --port 5000

# Multicast (multiple receivers)
python stream_bisenet_udp.py --host 239.0.0.1 --port 5000
```

**On Client (receiver):**
```bash
# GStreamer
gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H264 ! rtph264depay ! decodebin ! autovideosink

# ffplay
ffplay -fflags nobuffer -flags low_delay -framedrop udp://@:5000

# VLC
vlc udp://@:5000
```

### RTSP Streaming

**On SBC (sender):**
```bash
# Start RTSP server (in background or separate terminal)
mediamtx &

# Start streaming
python stream_bisenet_rtsp.py --model models/bisenetv2_rk3566.rknn
```

**On Client (receiver):**
```bash
vlc rtsp://<sbc-ip>:8554/bisenet
# or
ffplay rtsp://<sbc-ip>:8554/bisenet
```

## Command Line Options

### stream_bisenet_udp.py

| Option | Default | Description |
|--------|---------|-------------|
| `--camera` | 0 | Camera index |
| `--model` | auto | Path to RKNN model |
| `--download` | - | Force download model from GitHub |
| `--platform` | auto | Target platform (rk3566/rk3568/rk3588) |
| `--host` | 239.0.0.1 | Destination IP or multicast address |
| `--port` | 5000 | UDP port |
| `--ffmpeg` | - | Use FFmpeg instead of GStreamer |

### stream_bisenet_rtsp.py

| Option | Default | Description |
|--------|---------|-------------|
| `--camera` | 0 | Camera index |
| `--model` | auto | Path to RKNN model |
| `--download` | - | Force download model from GitHub |
| `--platform` | auto | Target platform (rk3566/rk3568/rk3588) |
| `--rtsp` | rtsp://0.0.0.0:8554/bisenet | RTSP output URL |
| `--host-rtsp` | true | Launch MediaMTX locally |

## Model Information

- **Model:** BiSeNetV2 (Cityscapes pretrained)
- **Input:** 1024x512 RGB image
- **Output:** 19-class semantic segmentation
- **Format:** FP16 (no quantization)
- **Download URLs:**
  - RK3566/RK3568: https://github.com/02loveslollipop/TankAgentController/releases/download/v1.0.0/bisenetv2_rk3566.rknn
  - RK3588: https://github.com/02loveslollipop/TankAgentController/releases/download/v1.0.0/bisenetv2_rk3588.rknn

## Cityscapes Classes

| ID | Class | Color |
|----|-------|-------|
| 0 | Road | Purple |
| 1 | Sidewalk | Pink |
| 2 | Building | Dark Gray |
| 3 | Wall | Blue-Gray |
| 4 | Fence | Beige |
| 5 | Pole | Gray |
| 6 | Traffic Light | Orange |
| 7 | Traffic Sign | Yellow |
| 8 | Vegetation | Green |
| 9 | Terrain | Light Green |
| 10 | Sky | Light Blue |
| 11 | Person | Red |
| 12 | Rider | Bright Red |
| 13 | Car | Dark Blue |
| 14 | Truck | Navy |
| 15 | Bus | Teal |
| 16 | Train | Dark Teal |
| 17 | Motorcycle | Blue |
| 18 | Bicycle | Brown |

## Troubleshooting

### "RKNPU2 driver not detected"

1. For RK356X, enable NPU via `sudo rsetup`
2. Reboot after enabling
3. Check with `dmesg | grep rknpu`

### "rknnlite not found"

```bash
# Install via apt (Radxa OS)
sudo apt update
sudo apt install python3-rknnlite2

# Or install wheel manually
pip install rknn_toolkit_lite2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.whl
```

### Low FPS

1. Check NPU is enabled: `cat /sys/kernel/debug/rknpu/load`
2. Reduce camera resolution
3. Use RK3588 for better performance (~30 FPS vs ~15 FPS on RK356X)

### Camera not found

```bash
# List cameras
v4l2-ctl --list-devices

# Try different index
python stream_bisenet_udp.py --camera 1 ...
```

## Performance

| Platform | NPU Cores | Expected FPS |
|----------|-----------|--------------|
| RK3566 | 1 (0.8 TOPS) | ~10-15 |
| RK3568 | 1 (1.0 TOPS) | ~12-18 |
| RK3588 | 3 (6.0 TOPS) | ~25-35 |

## License

MIT License - See repository root for details.

## Related

- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [BiSeNetV2 Paper](https://arxiv.org/abs/2004.02147)
