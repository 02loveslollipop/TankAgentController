"""
Capture UVC camera frames, run BiSeNetV2 segmentation using RKNN NPU,
and stream a side-by-side composite (left: original, right: segmentation overlay)
via GStreamer UDP (no relay server required).

Prereqs:
- Install dependencies: `pip install opencv-python numpy`
- Install RKNN Lite: `pip install rknn-lite2` (from Rockchip SDK)
- Install GStreamer: `sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-*`

Usage:
  # Auto-download model and stream to a specific client IP (unicast)
  python stream_bisenet_udp.py --download --host 192.168.1.100 --port 5000

  # Use local model file
  python stream_bisenet_udp.py --model /path/to/bisenetv2.rknn --host 192.168.1.100 --port 5000

  # Stream to multicast group (multiple clients can receive)
  python stream_bisenet_udp.py --download --host 239.0.0.1 --port 5000

Receiving the stream:
  # With GStreamer (H.264 RTP):
  gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H264 ! rtph264depay ! decodebin ! autovideosink

  # With ffplay (raw MJPEG):
  ffplay -fflags nobuffer -flags low_delay -framedrop udp://@:5000

  # With VLC:
  vlc udp://@:5000

Notes:
- This script auto-detects the Rockchip platform (RK3566/RK3568/RK3588).
- Model input: 512x1024 RGB, normalized with ImageNet mean/std.
- Model output: 19-class Cityscapes segmentation.
- UDP is connectionless - clients can join/leave at any time.
- For LAN streaming, use unicast (specific IP) or multicast (239.x.x.x range).
"""

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Model configuration
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
INPUT_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)  # (width, height) for cv2.resize
FPS = 15

# GitHub release URLs for RKNN models
GITHUB_RELEASE_BASE = "https://github.com/02loveslollipop/TankAgentController/releases/download/v1.0.0"
MODEL_URLS = {
    "rk3566": f"{GITHUB_RELEASE_BASE}/bisenetv2_rk3566.rknn",
    "rk3568": f"{GITHUB_RELEASE_BASE}/bisenetv2_rk3566.rknn",  # RK3568 uses same model as RK3566
    "rk3588": f"{GITHUB_RELEASE_BASE}/bisenetv2_rk3588.rknn",
}

# Cityscapes color palette (19 classes)
PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def detect_rockchip_platform() -> Optional[str]:
    """Detect the Rockchip platform from /proc/device-tree/compatible."""
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            compatible = f.read().decode("utf-8", errors="ignore").lower()
        
        if "rk3588" in compatible:
            return "rk3588"
        elif "rk3568" in compatible:
            return "rk3568"
        elif "rk3566" in compatible:
            return "rk3566"
    except Exception as e:
        print(f"[WARN] Could not detect platform: {e}")
    
    return None


def download_model(platform: str, output_dir: Path) -> Path:
    """Download RKNN model from GitHub releases."""
    if platform not in MODEL_URLS:
        raise ValueError(f"Unknown platform: {platform}. Supported: {list(MODEL_URLS.keys())}")
    
    url = MODEL_URLS[platform]
    filename = url.split("/")[-1]
    output_path = output_dir / filename
    
    if output_path.exists():
        print(f"[INFO] Model already exists: {output_path}")
        return output_path
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading model from {url}...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    
    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print()  # New line after progress
    print(f"[INFO] Model saved to: {output_path}")
    return output_path


def load_rknn(model_path: Path):
    """Load RKNN model and initialize runtime."""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("[ERROR] rknn-lite2 not installed!")
        print("  Install with: pip install rknn-lite2")
        print("  Or from Rockchip SDK: https://github.com/airockchip/rknn-toolkit2")
        sys.exit(1)
    
    rknn = RKNNLite()
    
    print(f"[INFO] Loading RKNN model: {model_path}")
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {ret}")
    
    print("[INFO] Initializing RKNN runtime...")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
    
    return rknn


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Preprocess frame for RKNN inference."""
    # Resize to model input size
    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize with ImageNet mean/std
    inp = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    
    # RKNN expects NHWC format for input
    inp = np.expand_dims(inp, axis=0).astype(np.float32)
    
    return inp


def postprocess(outputs: List[np.ndarray], orig_shape: Tuple[int, int]) -> np.ndarray:
    """Postprocess RKNN outputs to colored segmentation mask."""
    # outputs[0] shape: (1, 19, H, W) or similar
    logits = outputs[0]
    
    # Handle different output shapes
    if len(logits.shape) == 4:
        if logits.shape[1] == 19:  # NCHW format
            mask = logits.argmax(axis=1)[0].astype(np.uint8)
        else:  # NHWC format
            mask = logits.argmax(axis=-1)[0].astype(np.uint8)
    else:
        mask = logits.argmax(axis=-1).astype(np.uint8)
    
    # Apply color palette
    colored = PALETTE[mask % len(PALETTE)]
    
    # Resize to original frame size
    colored = cv2.resize(colored, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return colored


def composite(original: np.ndarray, colored: np.ndarray) -> np.ndarray:
    # Half original, half segmentation (horizontal stack)
    h, w, _ = original.shape
    half_w = w // 2
    original_half = cv2.resize(original, (half_w, h), interpolation=cv2.INTER_LINEAR)
    colored_half = cv2.resize(colored, (w - half_w, h), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([original_half, colored_half], axis=1)


def start_gstreamer_udp(host: str, port: int, width: int, height: int, use_h264: bool = True) -> subprocess.Popen:
    """
    Start a GStreamer pipeline that reads raw video from stdin and streams via UDP.
    
    Args:
        host: Destination IP (unicast) or multicast address (239.x.x.x)
        port: UDP port
        width: Frame width
        height: Frame height
        use_h264: If True, encode with H.264 (lower bandwidth). If False, use MJPEG (simpler).
    """
    if use_h264:
        # H.264 encoding - lower bandwidth, better quality
        # Uses x264enc for software encoding (works on ARM)
        # For hardware encoding on specific SBCs, replace with platform-specific encoder
        pipeline = (
            f"fdsrc ! "
            f"rawvideoparse width={width} height={height} format=bgr framerate={FPS}/1 ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={host} port={port}"
        )
    else:
        # MJPEG - simpler, higher bandwidth but easier to decode
        pipeline = (
            f"fdsrc ! "
            f"rawvideoparse width={width} height={height} format=bgr framerate={FPS}/1 ! "
            f"videoconvert ! jpegenc quality=80 ! "
            f"rtpjpegpay ! "
            f"udpsink host={host} port={port}"
        )
    
    cmd: List[str] = ["gst-launch-1.0", "-q"] + pipeline.split()
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def start_ffmpeg_udp(host: str, port: int, width: int, height: int) -> subprocess.Popen:
    """
    Alternative: Use FFmpeg for UDP streaming (fallback if GStreamer not available).
    """
    cmd: List[str] = [
        "ffmpeg",
        "-re",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-f", "mpegts",
        f"udp://{host}:{port}?pkt_size=1316",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def check_gstreamer() -> bool:
    """Check if GStreamer is available."""
    return shutil.which("gst-launch-1.0") is not None


def run(camera_index: int, model_path: Path, host: str, port: int, use_ffmpeg: bool = False):
    """Main inference and streaming loop."""
    # Open camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read initial frame from camera")
    height, width, _ = frame.shape

    # Load RKNN model
    rknn = load_rknn(model_path)

    # Choose streaming backend
    if use_ffmpeg or not check_gstreamer():
        if not use_ffmpeg:
            print("[WARN] GStreamer not found, falling back to FFmpeg")
        streamer = start_ffmpeg_udp(host, port, width, height)
        backend = "FFmpeg"
    else:
        streamer = start_gstreamer_udp(host, port, width, height, use_h264=True)
        backend = "GStreamer"

    if streamer.stdin is None:
        raise RuntimeError("Failed to start streaming process")

    print(f"[UDP] Streaming via {backend} to {host}:{port} at {width}x{height} @ {FPS}fps")
    print(f"[UDP] Receive with: gst-launch-1.0 udpsrc port={port} ! application/x-rtp,encoding-name=H264 ! rtph264depay ! decodebin ! autovideosink")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, retrying...")
                time.sleep(0.05)
                continue

            # RKNN inference
            inp = preprocess(frame)
            outputs = rknn.inference(inputs=[inp])
            colored = postprocess(outputs, frame.shape[:2])
            out = composite(frame, colored)
            
            try:
                streamer.stdin.write(out.tobytes())
            except BrokenPipeError:
                print("[WARN] Streamer pipe broken, restarting...")
                streamer.stdin.close()
                streamer.wait()
                if use_ffmpeg or not check_gstreamer():
                    streamer = start_ffmpeg_udp(host, port, width, height)
                else:
                    streamer = start_gstreamer_udp(host, port, width, height, use_h264=True)
            
            # FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"\r[INFO] FPS: {fps:.1f}", end="", flush=True)
                    
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        if streamer.stdin:
            streamer.stdin.close()
        streamer.wait(timeout=2)
        cap.release()
        rknn.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream BiSeNetV2 segmentation using RKNN NPU over UDP"
    )
    parser.add_argument("--camera", type=int, default=0, 
                        help="UVC camera index (default: 0)")
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to BiSeNetV2 RKNN model")
    parser.add_argument("--download", action="store_true",
                        help="Download model from GitHub releases")
    parser.add_argument("--platform", type=str, default=None,
                        choices=["rk3566", "rk3568", "rk3588"],
                        help="Target platform (auto-detected if not specified)")
    parser.add_argument("--host", type=str, default="239.0.0.1", 
                        help="Destination IP or multicast address (default: 239.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, 
                        help="UDP port (default: 5000)")
    parser.add_argument("--ffmpeg", action="store_true", 
                        help="Use FFmpeg instead of GStreamer for streaming")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Determine model path
    if args.model:
        model_path = args.model
        if not model_path.exists():
            sys.exit(f"Model not found: {model_path}")
    elif args.download:
        # Auto-detect platform if not specified
        platform_name = args.platform or detect_rockchip_platform()
        if not platform_name:
            sys.exit("[ERROR] Could not detect Rockchip platform. Please specify with --platform")
        
        print(f"[INFO] Detected platform: {platform_name}")
        model_dir = Path(__file__).parent / "models"
        model_path = download_model(platform_name, model_dir)
    else:
        # Try to auto-detect and download
        platform_name = detect_rockchip_platform()
        if platform_name:
            print(f"[INFO] Detected platform: {platform_name}")
            model_dir = Path(__file__).parent / "models"
            model_path = download_model(platform_name, model_dir)
        else:
            sys.exit("Please specify --model or use --download to fetch from GitHub releases")
    
    run(args.camera, model_path, args.host, args.port, args.ffmpeg)
