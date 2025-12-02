"""
Capture UVC camera frames, run BiSeNetV2 segmentation (pretrained, no fine-tuning),
and stream a side-by-side composite (left: original, right: segmentation overlay).

You can either:
- Point at an existing RTSP server URL, OR
- Let the SBC host its own RTSP server (via `rtsp-simple-server` if installed).

Prereqs:
- Install dependencies: `pip install opencv-python onnxruntime numpy`
- Provide a BiSeNetV2 ONNX checkpoint (e.g., bisenetv2_cityscapes.onnx) and set `--model`.
- If you want the SBC to host RTSP itself, install `rtsp-simple-server` (binary on PATH).

Usage:
  # Use an external RTSP server
  python stream_bisenet_rtsp.py --model /path/to/bisenetv2.onnx --rtsp rtsp://localhost:8554/bisenet

  # Host RTSP on the SBC (requires rtsp-simple-server in PATH)
  python stream_bisenet_rtsp.py --model /path/to/bisenetv2.onnx --rtsp rtsp://0.0.0.0:8554/bisenet

Notes:
- This script uses a generic BiSeNetV2 ONNX with RGB input. Adjust `INPUT_SIZE` and normalization
  if your checkpoint differs.
- RTSP output is produced by piping raw frames to ffmpeg which pushes to the RTSP server.
"""

import argparse
import shutil
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


def start_ffmpeg(rtsp_url: str, width: int, height: int) -> subprocess.Popen:
    cmd: List[str] = [
        "ffmpeg",
        "-re",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-f",
        "rtsp",
        rtsp_url,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def start_local_rtsp_server() -> subprocess.Popen | None:
    """
    Launch rtsp-simple-server in-process if available on PATH.
    Listens on 0.0.0.0:8554 by default.
    """
    binary = shutil.which("rtsp-simple-server")
    if not binary:
        print("[WARN] rtsp-simple-server not found on PATH; cannot self-host RTSP. Please install or supply external RTSP URL.")
        return None
    env = os.environ.copy()
    env.setdefault("RTSP_RTSPADDRESS", ":8554")  # listen on all interfaces
    proc = subprocess.Popen([binary], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("[INFO] Started local rtsp-simple-server on rtsp://0.0.0.0:8554/")
    return proc


def run(camera_index: int, model_path: Path, rtsp_url: str, host_rtsp: bool):
    rtsp_proc = start_local_rtsp_server() if host_rtsp else None
    time.sleep(0.5 if host_rtsp else 0.0)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read initial frame from camera")
    height, width, _ = frame.shape

    session = load_session(model_path)
    input_name = session.get_inputs()[0].name

    ffmpeg = start_ffmpeg(rtsp_url, width, height)
    if ffmpeg.stdin is None:
        raise RuntimeError("Failed to start ffmpeg process")

    print(f"[RTSP] Streaming to {rtsp_url} at {width}x{height} @ {FPS}fps")
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
            ffmpeg.stdin.write(out.tobytes())
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait(timeout=2)
        cap.release()
        if rtsp_proc:
            rtsp_proc.terminate()
            try:
                rtsp_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                rtsp_proc.kill()


def parse_args():
    parser = argparse.ArgumentParser(description="Stream BiSeNetV2 segmentation over RTSP")
    parser.add_argument("--camera", type=int, default=0, help="UVC camera index (default: 0)")
    parser.add_argument("--model", type=Path, required=True, help="Path to BiSeNetV2 ONNX model")
    parser.add_argument("--rtsp", type=str, default="rtsp://0.0.0.0:8554/bisenet", help="RTSP output URL")
    parser.add_argument(
        "--host-rtsp",
        action="store_true",
        default=True,
        help="Launch rtsp-simple-server locally on 0.0.0.0:8554 (default: true)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}")
    run(args.camera, args.model, args.rtsp, args.host_rtsp)
