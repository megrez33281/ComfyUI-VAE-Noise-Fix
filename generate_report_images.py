"""
Report Image Generation Script for ComfyUI-VAE-Noise-Fix.

This script extracts and saves every intermediate step of the noise-fix
pipeline for dataset/GroupA/000.png, providing high-quality images
suitable for PowerPoint slides.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path to ensure core can be imported
sys.path.append(str(Path(__file__).resolve().parent))

from core.tensor_bridge import TensorBridge
from core.energy import LaplacianEnergyExtractor, MedianResidualExtractor
from core.thresholding import DualPathMaskGenerator
from core.structural_filter import StructuralFilter
from core.chromatic_filter import ChromaticFilter
from core.morphology import MaskDilator
from core.inpainter import TeleaInpainter


def _load_image_safely(path: str) -> np.ndarray:
    """Read an image file correctly on Windows with non-ASCII paths."""
    img_array = np.fromfile(path, np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return bgr


def _save_image_safely(img: np.ndarray, path: str) -> None:
    """Save an image file correctly on Windows with non-ASCII paths."""
    ext = os.path.splitext(path)[1]
    success, img_encoded = cv2.imencode(ext, img)
    if success:
        img_encoded.tofile(path)
    else:
        raise IOError(f"Failed to encode image for saving: {path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    src_image_path = script_dir / "dataset" / "GroupA" / "000.png"
    out_dir = script_dir / "ppt_images"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading source image from: {src_image_path}")
    bgr_u8 = _load_image_safely(str(src_image_path))
    h, w, c = bgr_u8.shape
    print(f"Image loaded successfully. Size: {w}x{h}")

    # Parameters matching default ComfyUI settings
    gradient_sensitivity = 0.35
    max_noise_size = 6
    mask_dilate = 2

    # Baseline calculations as done in GradientNoiseDetector
    baseline_pixels = 1024.0 * 1024.0
    current_pixels = float(h * w)
    scale_factor = max(1.0, current_pixels / baseline_pixels)
    scaled_max_noise_size = int(np.ceil(max_noise_size * scale_factor))
    print(f"Scaled Max Noise Size: {scaled_max_noise_size}")

    # Initialize components
    laplacian = LaplacianEnergyExtractor()
    median = MedianResidualExtractor(scaled_max_noise_size)
    mask_gen = DualPathMaskGenerator()
    structural = StructuralFilter(scaled_max_noise_size)
    chromatic = ChromaticFilter(
        gradient_sensitivity=gradient_sensitivity,
        scaled_max_noise_size=scaled_max_noise_size,
    )
    dilator = MaskDilator(mask_dilate)

    # 1. Gray Scale
    print("Generating: a. Gray Scale")
    gray = TensorBridge.grayscale_rec709(bgr_u8)
    _save_image_safely(gray, str(out_dir / "01_gray_scale.png"))

    # 2. Laplacian Energy Map and Laplacian Binary Map
    print("Generating: 2. Laplacian energy and binary")
    lap_energy = laplacian.extract(gray)
    t_lap = int(gradient_sensitivity * 255.0)
    _, lap_binary = cv2.threshold(lap_energy, t_lap, 255, cv2.THRESH_BINARY)
    _save_image_safely(lap_energy, str(out_dir / "02a_laplacian_energy.png"))
    _save_image_safely(lap_binary, str(out_dir / "02b_laplacian_binary_thresholded.png"))

    # 3. Median Blur
    print("Generating: 3. Median Blur")
    ksize = median.kernel_size
    median_blur = cv2.medianBlur(bgr_u8, ksize)
    _save_image_safely(median_blur, str(out_dir / "03_median_blur.png"))

    # 4. Residual Map (Original vs. Median Blur)
    print("Generating: 4. Residual (BGR and Max-channel absolute)")
    bgr_diff = cv2.absdiff(bgr_u8, median_blur)
    med_residual = np.max(bgr_diff, axis=2)
    t_med = int(20 + gradient_sensitivity * 80.0)
    _, med_binary = cv2.threshold(med_residual, t_med, 255, cv2.THRESH_BINARY)
    
    _save_image_safely(bgr_diff, str(out_dir / "04a_median_residual_bgr.png"))
    _save_image_safely(med_residual, str(out_dir / "04b_median_residual_gray.png"))
    _save_image_safely(med_binary, str(out_dir / "04c_median_residual_binary_thresholded.png"))

    # 5. Context Mask (Low threshold, 輪廓遮罩)
    print("Generating: 5. Context Mask")
    context_thresh_factor = 0.25
    context_thresh_floor = 0.05
    context_thresh = max(
        context_thresh_floor,
        gradient_sensitivity * context_thresh_factor,
    )
    context_mask = mask_gen.generate(lap_energy, med_residual, context_thresh)
    _save_image_safely(context_mask, str(out_dir / "05_context_mask.png"))

    # 6. Seed Mask (High threshold, 核心遮罩)
    print("Generating: 6. Seed Mask")
    seed_mask = mask_gen.generate(lap_energy, med_residual, gradient_sensitivity)
    _save_image_safely(seed_mask, str(out_dir / "06_seed_mask.png"))

    # 7. 合起來後的 candidate mask (Logical OR / Dual-Path combination)
    # The Seed Mask itself is already the "Combined Seed Mask" (Laplacian Binary OR Median Residual Binary)
    # Let's generate a stunning, color-coded dual-path candidate visualization for the PPT:
    # Context mask (green background / broad candidate region) and Seed mask (red foreground / core seeds)
    print("Generating: 7. Combined Candidate Mask Visualization")
    # Gray background image for context overlay
    h_vis, w_vis = bgr_u8.shape[:2]
    combined_vis = bgr_u8.copy()
    
    # Create color overlay
    overlay = np.zeros_like(bgr_u8)
    # Context mask in Green (where context_mask > 0 but seed_mask == 0)
    overlay[np.logical_and(context_mask > 0, seed_mask == 0)] = [0, 255, 0] # G
    # Seed mask in Red (where seed_mask > 0)
    overlay[seed_mask > 0] = [0, 0, 255] # R
    
    # Alpha blend overlay on original image
    has_mask = np.logical_or(context_mask > 0, seed_mask > 0)
    combined_vis[has_mask] = cv2.addWeighted(bgr_u8, 0.4, overlay, 0.6, 0)[has_mask]
    
    # Save raw OR'ed combination (the context mask is actually the full candidate mask enclosing the seeds)
    _save_image_safely(combined_vis, str(out_dir / "07a_candidate_color_overlay.png"))
    _save_image_safely(context_mask, str(out_dir / "07b_raw_candidate_mask.png"))

    # 8. 過濾後的 filtered mask (Shape and Isolation Filtering)
    print("Generating: 8. Filtered Mask")
    filtered_mask = structural.filter(context_mask, seed_mask)
    _save_image_safely(filtered_mask, str(out_dir / "08_filtered_mask.png"))

    # Bonus: 9. Verified Mask, 10. Dilated Mask, 11. Final Repaired Image
    print("Generating: 9-11 (Verified, Dilated/Final, Repaired)")
    verified_mask = chromatic.filter(filtered_mask, bgr_u8)
    final_mask = dilator.dilate(verified_mask)
    repaired = TeleaInpainter.inpaint(bgr_u8, final_mask, scaled_max_noise_size)

    _save_image_safely(verified_mask, str(out_dir / "09_verified_mask_lab.png"))
    _save_image_safely(final_mask, str(out_dir / "10_final_mask_dilated.png"))
    _save_image_safely(repaired, str(out_dir / "11_repaired_output.png"))

    # Visual side-by-side comparison for slide WOW factor
    comparison = np.hstack((bgr_u8, repaired))
    _save_image_safely(comparison, str(out_dir / "12_comparison_side_by_side.png"))

    print("\nAll intermediate report images successfully generated in 'ppt_images' directory!")


if __name__ == "__main__":
    main()
