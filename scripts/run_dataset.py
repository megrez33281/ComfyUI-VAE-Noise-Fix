"""
Batch-run VAENoiseFixNode over the bundled dataset for visual inspection.

Loads dataset images, runs the detection + inpaint pipeline, and writes
mask-overlay and repaired outputs to ``test_output/`` without requiring
a running ComfyUI instance.  Automated assertions live in ``tests/``
(run with ``pytest``); this script is for eyeballing results.

Usage:
    python scripts/run_dataset.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from comfyui_node import VAENoiseFixNode  # noqa: E402
from core import PreviewMode  # noqa: E402


# ---------------------------------------------------------------------------
# I/O helpers (Windows non-ASCII path safe)
# ---------------------------------------------------------------------------

def _load_image_as_comfyui_tensor(path: str) -> torch.Tensor:
    """Read an image file and return a ComfyUI-compatible [1, H, W, 3] tensor."""
    img_array = np.fromfile(path, np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)  # [1, H, W, C]


def _save_tensor(tensor: torch.Tensor, path: str) -> None:
    """Save a ComfyUI [1, H, W, 3] tensor to disk."""
    frame = tensor[0].detach().cpu().numpy()
    rgb_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1]
    success, img_encoded = cv2.imencode(ext, bgr_u8)
    if success:
        img_encoded.tofile(path)
    else:
        raise IOError(f"Failed to encode image for saving: {path}")


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def run_tests() -> None:
    dataset_dir = REPO_ROOT / "dataset"
    output_dir = REPO_ROOT / "test_output"
    output_dir.mkdir(exist_ok=True)

    image_paths = sorted(glob.glob(str(dataset_dir / "**" / "*.png"), recursive=True))
    if not image_paths:
        print("No PNG images found in dataset/. Exiting.")
        return

    node = VAENoiseFixNode()
    base_params = {
        "gradient_sensitivity": 0.35,
        "max_noise_size": 6,
        "mask_dilate": 2,
    }

    print(f"Found {len(image_paths)} images. Processing...\n")

    for img_path in image_paths:
        rel = os.path.relpath(img_path, dataset_dir)
        name = rel.replace(os.sep, "_").replace(".png", "")
        print(f"  [{name}] ", end="")

        tensor = _load_image_as_comfyui_tensor(img_path)

        # Detection overlay preview.
        mask_image, _ = node.execute(
            image=tensor,
            preview_mode=PreviewMode.MASK_OVERLAY.value,
            gradient_sensitivity=base_params["gradient_sensitivity"],
            max_noise_size=base_params["max_noise_size"],
            mask_dilate=base_params["mask_dilate"],
        )
        _save_tensor(mask_image, str(output_dir / f"{name}_mask.png"))

        # Repaired output.
        fixed_image, _ = node.execute(
            image=tensor,
            preview_mode=PreviewMode.REPAIRED.value,
            gradient_sensitivity=base_params["gradient_sensitivity"],
            max_noise_size=base_params["max_noise_size"],
            mask_dilate=base_params["mask_dilate"],
        )
        _save_tensor(fixed_image, str(output_dir / f"{name}_fixed.png"))

        print("OK")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    run_tests()
