#!/usr/bin/env python3
"""
Download BiSeNetV2 ONNX model and convert/quantize it to RKNN format.

This script:
1. Downloads a pretrained BiSeNetV2 ONNX model
2. (TODO) Fine-tunes/trains the model on custom dataset
3. Quantizes and converts to RKNN format for Rockchip NPU
4. Saves the result to ./models/
5. Uploads the RKNN model to GitHub Releases

Requirements:
- Must run on x86_64 Linux (RKNN Toolkit doesn't support ARM)
- pip install rknn-toolkit2  (for RK3588/RK3566/RK3568)
  or
- pip install rknn-toolkit   (for older RK1808/RV1109/RV1126)
- GitHub CLI (gh) for uploading to releases

Usage:
    python convert_bisenet.py
    python convert_bisenet.py --platform rk3588
    python convert_bisenet.py --platform rk3566
    python convert_bisenet.py --upload --tag v1.0.0
    python convert_bisenet.py --release --tag v1.0.0  # Build rk3566 & rk3588, upload both
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

# Monkey patch for onnx.mapping (removed in ONNX 1.13+)
if not hasattr(onnx, 'mapping'):
    class Mapping:
        TENSOR_TYPE_TO_NP_TYPE = {
            TensorProto.FLOAT: np.dtype('float32'),
            TensorProto.BOOL: np.dtype('bool'),
            TensorProto.INT32: np.dtype('int32'),
            TensorProto.INT64: np.dtype('int64'),
            TensorProto.INT8: np.dtype('int8'),
            TensorProto.UINT8: np.dtype('uint8'),
            TensorProto.FLOAT16: np.dtype('float16'),
            TensorProto.DOUBLE: np.dtype('float64'),
            TensorProto.UINT32: np.dtype('uint32'),
            TensorProto.UINT64: np.dtype('uint64'),
            TensorProto.STRING: np.dtype('O'),
        }
        NP_TYPE_TO_TENSOR_TYPE = {
            np.dtype('float32'): TensorProto.FLOAT,
            np.dtype('bool'): TensorProto.BOOL,
            np.dtype('int32'): TensorProto.INT32,
            np.dtype('int64'): TensorProto.INT64,
            np.dtype('int8'): TensorProto.INT8,
            np.dtype('uint8'): TensorProto.UINT8,
            np.dtype('float16'): TensorProto.FLOAT16,
            np.dtype('float64'): TensorProto.DOUBLE,
            np.dtype('uint32'): TensorProto.UINT32,
            np.dtype('uint64'): TensorProto.UINT64,
            np.dtype('O'): TensorProto.STRING,
            # Add scalar types as well
            np.float32: TensorProto.FLOAT,
            np.bool_: TensorProto.BOOL,
            np.int32: TensorProto.INT32,
            np.int64: TensorProto.INT64,
            np.int8: TensorProto.INT8,
            np.uint8: TensorProto.UINT8,
            np.float16: TensorProto.FLOAT16,
            np.float64: TensorProto.DOUBLE,
            np.uint32: TensorProto.UINT32,
            np.uint64: TensorProto.UINT64,
            np.object_: TensorProto.STRING,
        }
    onnx.mapping = Mapping()


# Model configuration
MODEL_URL = "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v2_city.pth"
INPUT_SIZE = (1, 3, 512, 1024)  # NCHW format - BiSeNetV2 Cityscapes default
SUPPORTED_PLATFORMS = ["rk3588", "rk3566", "rk3568", "rk3562", "rk1808", "rv1109", "rv1126"]
RELEASE_PLATFORMS = ["rk3566", "rk3588"]  # Platforms to build when using --release

# GitHub configuration
GITHUB_REPO = "02loveslollipop/TankAgentController"


def check_gh_cli() -> bool:
    """Check if GitHub CLI is installed and authenticated."""
    if not shutil.which("gh"):
        print("Error: GitHub CLI (gh) is not installed.")
        print("Install from: https://cli.github.com/")
        return False
    
    # Check authentication
    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: GitHub CLI is not authenticated.")
        print("Run: gh auth login")
        return False
    
    return True


def get_or_create_release(tag: str, repo: str) -> bool:
    """Get existing release or create a new one."""
    # Check if release exists
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"Using existing release: {tag}")
        return True
    
    # Create new release
    print(f"Creating new release: {tag}")
    result = subprocess.run(
        [
            "gh", "release", "create", tag,
            "--repo", repo,
            "--title", f"BiSeNetV2 RKNN Models {tag}",
            "--notes", f"Pre-converted BiSeNetV2 RKNN models for Rockchip NPU.\n\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Failed to create release: {result.stderr}")
        return False
    
    print(f"Created release: {tag}")
    return True


def upload_to_github_release(file_path: Path, tag: str, repo: str) -> bool:
    """Upload a file to GitHub release."""
    if not check_gh_cli():
        return False
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return False
    
    # Create or get release
    if not get_or_create_release(tag, repo):
        return False
    
    print(f"\nUploading {file_path.name} to release {tag}...")
    
    # Check if asset already exists and delete it
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        assets = json.loads(result.stdout).get("assets", [])
        for asset in assets:
            if asset["name"] == file_path.name:
                print(f"Deleting existing asset: {file_path.name}")
                subprocess.run(
                    ["gh", "release", "delete-asset", tag, file_path.name, "--repo", repo, "--yes"],
                    capture_output=True
                )
                break
    
    # Upload the file
    result = subprocess.run(
        ["gh", "release", "upload", tag, str(file_path), "--repo", repo, "--clobber"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Failed to upload: {result.stderr}")
        return False
    
    file_size = file_path.stat().st_size / (1024 * 1024)
    print(f"Successfully uploaded {file_path.name} ({file_size:.2f} MB)")
    print(f"Download URL: https://github.com/{repo}/releases/download/{tag}/{file_path.name}")
    return True


def check_architecture():
    """Ensure we're running on x86_64."""
    arch = platform.machine()
    if arch not in ("x86_64", "AMD64"):
        print(f"Error: RKNN Toolkit requires x86_64 architecture, but detected: {arch}")
        print("Please run this script on an x86_64 PC, then transfer the .rknn file to your SBC.")
        sys.exit(1)


def download_model(url: str, output_path: Path) -> bool:
    """Download model from URL with progress indicator."""
    if output_path.exists():
        print(f"Model already exists: {output_path}")
        return True
    
    try:
        print(f"Downloading model from {url}...")
        
        def progress_hook(count, block_size, total_size):
            percent = min(100, int(count * block_size * 100 / total_size))
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = '=' * filled + '-' * (bar_len - filled)
            sys.stdout.write(f"\r[{bar}] {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading model: {e}")
        return False


def convert_pth_to_onnx(pth_path: Path, onnx_path: Path) -> bool:
    """Convert PyTorch .pth checkpoint to ONNX format."""
    if onnx_path.exists():
        print(f"ONNX model already exists: {onnx_path}")
        return True
    
    try:
        import torch
        import torch.nn as nn
        
        print(f"Converting {pth_path.name} to ONNX format...")
        
        # BiSeNetV2 model definition (simplified version matching the checkpoint)
        # This is based on the CoinCheung/BiSeNet repository structure
        
        class ConvBNReLU(nn.Module):
            def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
                super().__init__()
                self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
                self.bn = nn.BatchNorm2d(out_chan)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                return self.relu(self.bn(self.conv(x)))
        
        class DetailBranch(nn.Module):
            def __init__(self):
                super().__init__()
                self.S1 = nn.Sequential(
                    ConvBNReLU(3, 64, 3, stride=2),
                    ConvBNReLU(64, 64, 3, stride=1),
                )
                self.S2 = nn.Sequential(
                    ConvBNReLU(64, 64, 3, stride=2),
                    ConvBNReLU(64, 64, 3, stride=1),
                    ConvBNReLU(64, 64, 3, stride=1),
                )
                self.S3 = nn.Sequential(
                    ConvBNReLU(64, 128, 3, stride=2),
                    ConvBNReLU(128, 128, 3, stride=1),
                    ConvBNReLU(128, 128, 3, stride=1),
                )
            
            def forward(self, x):
                x = self.S1(x)
                x = self.S2(x)
                x = self.S3(x)
                return x
        
        class StemBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = ConvBNReLU(3, 16, 3, stride=2)
                self.left = nn.Sequential(
                    ConvBNReLU(16, 8, 1, stride=1, padding=0),
                    ConvBNReLU(8, 16, 3, stride=2),
                )
                self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.fuse = ConvBNReLU(32, 16, 3, stride=1)
            
            def forward(self, x):
                x = self.conv(x)
                left = self.left(x)
                right = self.right(x)
                x = torch.cat([left, right], dim=1)
                return self.fuse(x)
        
        class CEBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(128)
                self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
                self.conv_last = ConvBNReLU(128, 128, 3, stride=1)
            
            def forward(self, x):
                feat = torch.mean(x, dim=(2, 3), keepdim=True)
                feat = self.bn(feat)
                feat = self.conv_gap(feat)
                return self.conv_last(x + feat)
        
        class GELayerS1(nn.Module):
            def __init__(self, in_chan, out_chan, exp_ratio=6):
                super().__init__()
                mid_chan = in_chan * exp_ratio
                self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
                self.dwconv = nn.Sequential(
                    nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False),
                    nn.BatchNorm2d(mid_chan),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_chan),
                )
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                feat = self.conv1(x)
                feat = self.dwconv(feat)
                feat = self.conv2(feat)
                return self.relu(feat + x)
        
        class GELayerS2(nn.Module):
            def __init__(self, in_chan, out_chan, exp_ratio=6):
                super().__init__()
                mid_chan = in_chan * exp_ratio
                self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
                self.dwconv1 = nn.Sequential(
                    nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False),
                    nn.BatchNorm2d(mid_chan),
                )
                self.dwconv2 = nn.Sequential(
                    nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=mid_chan, bias=False),
                    nn.BatchNorm2d(mid_chan),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_chan),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False),
                    nn.BatchNorm2d(in_chan),
                    nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_chan),
                )
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                feat = self.conv1(x)
                feat = self.dwconv1(feat)
                feat = self.dwconv2(feat)
                feat = self.conv2(feat)
                shortcut = self.shortcut(x)
                return self.relu(feat + shortcut)
        
        class SegmentBranch(nn.Module):
            def __init__(self):
                super().__init__()
                self.S1S2 = StemBlock()
                self.S3 = nn.Sequential(GELayerS2(16, 32), GELayerS1(32, 32))
                self.S4 = nn.Sequential(GELayerS2(32, 64), GELayerS1(64, 64))
                self.S5_4 = nn.Sequential(GELayerS2(64, 128), GELayerS1(128, 128), GELayerS1(128, 128), GELayerS1(128, 128))
                self.S5_5 = CEBlock()
            
            def forward(self, x):
                x2 = self.S1S2(x)
                x3 = self.S3(x2)
                x4 = self.S4(x3)
                x5_4 = self.S5_4(x4)
                x5_5 = self.S5_5(x5_4)
                return x2, x3, x4, x5_4, x5_5
        
        class BGALayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.left1 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
                )
                self.left2 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                )
                self.right1 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                )
                self.right2 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
                )
                self.conv = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
            
            def forward(self, x_d, x_s):
                left1 = self.left1(x_d)
                left2 = self.left2(x_d)
                right1 = self.right1(x_s)
                right2 = self.right2(x_s)
                right1 = nn.functional.interpolate(right1, size=x_d.shape[2:], mode='bilinear', align_corners=False)
                left = left1 * torch.sigmoid(right1)
                right = left2 * torch.sigmoid(right2)
                right = nn.functional.interpolate(right, size=x_d.shape[2:], mode='bilinear', align_corners=False)
                return self.conv(left + right)
        
        class SegmentHead(nn.Module):
            def __init__(self, in_chan, mid_chan, n_classes):
                super().__init__()
                self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
                self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
            
            def forward(self, x, size=None):
                x = self.conv(x)
                x = self.conv_out(x)
                if size is not None:
                    x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
                return x
        
        class BiSeNetV2(nn.Module):
            def __init__(self, n_classes=19):
                super().__init__()
                self.detail = DetailBranch()
                self.segment = SegmentBranch()
                self.bga = BGALayer()
                self.head = SegmentHead(128, 1024, n_classes)
            
            def forward(self, x):
                size = x.shape[2:]
                feat_d = self.detail(x)
                feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
                feat_head = self.bga(feat_d, feat_s)
                logits = self.head(feat_head, size)
                return logits
        
        # Create model and load weights
        print("Creating BiSeNetV2 model...")
        model = BiSeNetV2(n_classes=19)
        
        print(f"Loading weights from {pth_path}...")
        state_dict = torch.load(pth_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # Export to ONNX
        print("Exporting to ONNX...")
        dummy_input = torch.randn(INPUT_SIZE)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=['input'],
            output_names=['output'],
            opset_version=12,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX model saved to: {onnx_path}")
        return True
        
    except ImportError:
        print("\nError: PyTorch is not installed!")
        print("Install with: pip install torch torchvision")
        return False
    except Exception as e:
        print(f"\nError converting to ONNX: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_model(onnx_path: Path, dataset_path: Path = None):
    """
    Fine-tune or train the BiSeNetV2 model on custom dataset.
    
    TODO: Implement training pipeline
    - Load pretrained ONNX model
    - Setup PyTorch training loop
    - Train on custom dataset
    - Export back to ONNX
    """
    print("\n" + "=" * 60)
    print("TRAINING STEP (SKIPPED)")
    print("=" * 60)
    print("Training/fine-tuning is not yet implemented.")
    print("Using pretrained Cityscapes weights directly.")
    print("=" * 60 + "\n")
    
    # TODO: Implement training
    # from train import BiSeNetV2Trainer
    # trainer = BiSeNetV2Trainer(onnx_path, dataset_path)
    # trainer.train(epochs=100, lr=0.01)
    # trainer.export_onnx(onnx_path)


def generate_calibration_data(output_dir: Path, num_images: int = 20):
    """
    Generate dummy calibration images for quantization.
    
    In production, you should use real images from your dataset.
    """
    import numpy as np
    
    calib_dir = output_dir / "calibration"
    calib_dir.mkdir(exist_ok=True)
    
    dataset_file = output_dir / "dataset.txt"
    
    # Check if calibration data already exists
    existing_images = list(calib_dir.glob("*.npy"))
    if len(existing_images) >= num_images and dataset_file.exists():
        print(f"Using existing calibration data ({len(existing_images)} images)")
        return dataset_file
    
    print(f"Generating {num_images} calibration images...")
    print("NOTE: For better quantization accuracy, replace with real dataset images!")
    
    with open(dataset_file, 'w') as f:
        for i in range(num_images):
            # Generate random image data (512x512 RGB)
            # In production, use real images from your target domain
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            img_path = calib_dir / f"calib_{i:04d}.npy"
            np.save(img_path, img)
            f.write(f"{img_path}\n")
    
    print(f"Calibration dataset saved to: {dataset_file}")
    return dataset_file


def convert_to_rknn(onnx_path: Path, rknn_path: Path, target_platform: str) -> bool:
    """Convert ONNX model to RKNN format with FP16 (no quantization)."""
    try:
        # Try rknn-toolkit2 first (for newer chips)
        try:
            from rknn.api import RKNN
            print("Using RKNN Toolkit 2")
        except ImportError:
            # Fall back to rknn-toolkit (older chips)
            from rknn.api import RKNN
            print("Using RKNN Toolkit")
        
        print(f"\nConverting {onnx_path.name} to RKNN format (FP16)...")
        print(f"Target platform: {target_platform}")
        
        # Create RKNN object
        rknn = RKNN(verbose=True)
        
        # Pre-process config for BiSeNetV2
        # Normalize from 0-255 to 0-1 range
        print("\n[1/3] Configuring RKNN...")
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=target_platform,
            optimization_level=3,
        )
        if ret != 0:
            print("Config failed!")
            return False
        
        # Load ONNX model
        print("\n[2/3] Loading ONNX model...")
        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=['input'],
            input_size_list=[list(INPUT_SIZE)],
        )
        if ret != 0:
            print("Load ONNX model failed!")
            print("Trying without explicit input specification...")
            ret = rknn.load_onnx(model=str(onnx_path))
            if ret != 0:
                print("Load ONNX model failed again!")
                return False
        
        # Build model WITHOUT quantization (FP16 for NPU)
        print("\n[3/3] Building RKNN model (FP16, no quantization)...")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            print("Build RKNN model failed!")
            return False
        
        # Export RKNN model
        print(f"\nExporting RKNN model to: {rknn_path}")
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            print("Export RKNN model failed!")
            return False
        
        # Get model info
        print("\nModel info:")
        
        # Release resources
        rknn.release()
        
        # Print file sizes
        onnx_size = onnx_path.stat().st_size / (1024 * 1024)
        rknn_size = rknn_path.stat().st_size / (1024 * 1024)
        print(f"\nONNX size: {onnx_size:.2f} MB")
        print(f"RKNN size: {rknn_size:.2f} MB")
        
        return True
        
    except ImportError:
        print("\nError: RKNN Toolkit not installed!")
        print("\nInstall with one of:")
        print("  pip install rknn-toolkit2  # For RK3588/RK3566/RK3568")
        print("  pip install rknn-toolkit   # For RK1808/RV1109/RV1126")
        print("\nNote: RKNN Toolkit only works on x86_64 Linux!")
        return False
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert BiSeNetV2 to RKNN format")
    parser.add_argument(
        "--platform", "-p",
        type=str,
        default="rk3588",
        choices=SUPPORTED_PLATFORMS,
        help="Target Rockchip platform (default: rk3588)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, use existing ONNX model"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the RKNN model to GitHub Releases"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Build for rk3566 and rk3588, then upload both to GitHub Releases"
    )
    parser.add_argument(
        "--tag", "-t",
        type=str,
        default=None,
        help="GitHub release tag (default: auto-generated based on date)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=GITHUB_REPO,
        help=f"GitHub repository (default: {GITHUB_REPO})"
    )
    args = parser.parse_args()
    
    # Check architecture
    check_architecture()
    
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    pth_path = models_dir / "model_final_v2_city.pth"
    onnx_path = models_dir / "bisenetv2.onnx"
    
    # Determine which platforms to build
    if args.release:
        platforms = RELEASE_PLATFORMS
        do_upload = True
        print("=" * 60)
        print("BiSeNetV2 RKNN Release Build")
        print("=" * 60)
        print(f"Building for platforms: {', '.join(platforms)}")
        print(f"ONNX path: {onnx_path}")
        print("=" * 60)
    else:
        platforms = [args.platform]
        do_upload = args.upload
        print("=" * 60)
        print("BiSeNetV2 RKNN Conversion Tool")
        print("=" * 60)
        print(f"Target platform: {args.platform}")
        print(f"ONNX path: {onnx_path}")
        print(f"RKNN path: {models_dir / f'bisenetv2_{args.platform}.rknn'}")
        print("=" * 60)
    
    # Step 1: Download PyTorch model
    if not args.skip_download:
        print("\n>>> Step 1: Downloading pretrained PyTorch model...")
        if not download_model(MODEL_URL, pth_path):
            print("Failed to download model!")
            return 1
    else:
        if not pth_path.exists() and not onnx_path.exists():
            print(f"Error: Neither PTH nor ONNX model found")
            return 1
        print("\n>>> Step 1: Using existing model")
    
    # Step 2: Convert PyTorch to ONNX
    print("\n>>> Step 2: Converting PyTorch to ONNX...")
    if not convert_pth_to_onnx(pth_path, onnx_path):
        print("Failed to convert to ONNX!")
        return 1
    
    # Step 3: Training (skipped for now)
    print("\n>>> Step 3: Training...")
    train_model(onnx_path)
    
    # Step 4: Convert to RKNN FP16 (for each platform)
    rknn_paths = []
    for plat in platforms:
        rknn_path = models_dir / f"bisenetv2_{plat}.rknn"
        print(f"\n>>> Step 4: Converting to RKNN (FP16) for {plat.upper()}...")
        if not convert_to_rknn(onnx_path, rknn_path, plat):
            print(f"\nConversion failed for {plat}!")
            return 1
        rknn_paths.append(rknn_path)
    
    # Step 5: Upload to GitHub Releases (optional)
    tag = args.tag or f"models-{datetime.now().strftime('%Y%m%d')}"
    if do_upload:
        print("\n>>> Step 5: Uploading to GitHub Releases...")
        upload_success = True
        for rknn_path in rknn_paths:
            if not upload_to_github_release(rknn_path, tag, args.repo):
                print(f"\nUpload failed for {rknn_path.name}!")
                upload_success = False
            else:
                print(f"Uploaded: {rknn_path.name}")
        if not upload_success:
            print("\nSome uploads failed! Models saved locally.")
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    for rknn_path in rknn_paths:
        print(f"RKNN model saved to: {rknn_path}")
    if do_upload:
        print(f"\nGitHub Release: https://github.com/{args.repo}/releases/tag/{tag}")
    print(f"\nTransfer files to your SBC:")
    for rknn_path in rknn_paths:
        print(f"  scp {rknn_path} user@sbc-ip:/path/to/model/")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
