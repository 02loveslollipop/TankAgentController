"""
Capture UVC camera frames, run BiSeNetV2 segmentation (pretrained, no fine-tuning),
and stream a side-by-side composite (left: original, right: segmentation overlay)
via GStreamer UDP (no relay server required).

Prereqs:
- Install dependencies: `pip install opencv-python onnxruntime numpy`
- Install GStreamer: `sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-*`
- Provide a BiSeNetV2 ONNX checkpoint (e.g., bisenetv2_cityscapes.onnx) and set `--model`.

Usage:
  # Stream to a specific client IP (unicast)
  python stream_bisenet_udp.py --model /path/to/bisenetv2.onnx --host 192.168.1.100 --port 5000

  # Stream to multicast group (multiple clients can receive)
  python stream_bisenet_udp.py --model /path/to/bisenetv2.onnx --host 239.0.0.1 --port 5000

Receiving the stream:
  # With GStreamer (H.264 RTP):
  gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H264 ! rtph264depay ! decodebin ! autovideosink

  # With ffplay (raw MJPEG):
  ffplay -fflags nobuffer -flags low_delay -framedrop udp://@:5000

  # With VLC:
  vlc udp://@:5000

Notes:
- This script uses a generic BiSeNetV2 ONNX with RGB input. Adjust `INPUT_SIZE` and normalization
  if your checkpoint differs.
- UDP is connectionless - clients can join/leave at any time.
- For LAN streaming, use unicast (specific IP) or multicast (239.x.x.x range).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = (512, 512)  # (width, height) expected by generic BiSeNetV2 checkpoints
FPS = 15
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


def load_session(model_path: Path) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model_path.as_posix(), providers=providers)


def preprocess(frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
    h, w, _ = frame.shape
    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    inp = resized.astype(np.float32) / 255.0
    # Standard ImageNet normalization (adjust if your checkpoint differs)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)[None, ...]
    return inp, w / INPUT_SIZE[0], h / INPUT_SIZE[1]


def postprocess(logits: np.ndarray, scale_w: float, scale_h: float, orig_shape: Tuple[int, int]) -> np.ndarray:
    # logits: (1, C, H, W)
    mask = logits.argmax(axis=1)[0].astype(np.uint8)
    colored = PALETTE[mask % len(PALETTE)]
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
    import shutil
    return shutil.which("gst-launch-1.0") is not None


def run(camera_index: int, model_path: Path, host: str, port: int, use_ffmpeg: bool = False):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read initial frame from camera")
    height, width, _ = frame.shape

    session = load_session(model_path)
    input_name = session.get_inputs()[0].name

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
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, retrying...")
                time.sleep(0.05)
                continue

            inp, scale_w, scale_h = preprocess(frame)
            logits = session.run(None, {input_name: inp})[0]
            colored = postprocess(logits, scale_w, scale_h, frame.shape[:2])
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
                    
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        if streamer.stdin:
            streamer.stdin.close()
        streamer.wait(timeout=2)
        cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Stream BiSeNetV2 segmentation over UDP (GStreamer)")
    parser.add_argument("--camera", type=int, default=0, help="UVC camera index (default: 0)")
    parser.add_argument("--model", type=Path, required=True, help="Path to BiSeNetV2 ONNX model")
    parser.add_argument("--host", type=str, default="239.0.0.1", 
                        help="Destination IP or multicast address (default: 239.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="UDP port (default: 5000)")
    parser.add_argument("--ffmpeg", action="store_true", 
                        help="Use FFmpeg instead of GStreamer for streaming")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}")
    run(args.camera, args.model, args.host, args.port, args.ffmpeg)
