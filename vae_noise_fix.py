"""
ComfyUI Custom Node: VAE High-Frequency Noise Auto-Detection and Repair

A high-performance, Zero-NN (Zero Neural Network) post-processing node designed 
to autonomously detect and repair isolated high-frequency artifacts (fireflies) 
produced during SDXL VAE decoding. 

The pipeline utilizes a lightweight traditional computer vision stack:
1. Dual-Path Detection: Laplacian energy extraction ∪ Median filter residual.
2. Geometric Filtering: Connected Component Analysis (CCA) for area thresholding.
3. CIE-LAB Verification: Chromaticity vs. Luminance variance check to preserve 
   natural highlights (stars, specular reflections).
4. Boundary expansion: Morphological mask dilation for edge coverage.
5. Reconstruction: Telea (Fast Marching Method) inpainting for seamless repair.

Architecture adheres to SOLID principles, optimized for batched [B, H, W, C] 
tensors with explicit GPU/CPU memory boundary management.
"""

from __future__ import annotations

import math
from typing import Tuple, List

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Module 1 – Tensor / Array Bridge
# ---------------------------------------------------------------------------

class TensorBridge:
    """Handles format conversions between ComfyUI PyTorch tensors and
    OpenCV-compatible NumPy arrays while minimising cross-device transfers.

    ComfyUI image layout : ``[B, H, W, C]``, float32, range [0, 1], RGB.
    OpenCV image layout  : ``[H, W, C]``, uint8, range [0, 255], BGR.
    """

    @staticmethod
    def comfyui_to_cv2(tensor: torch.Tensor) -> np.ndarray:
        """Convert a single ComfyUI frame ``[H, W, 3]`` to an OpenCV BGR uint8 array.

        The tensor is moved to CPU only once; the subsequent quantisation and
        channel swap operate entirely in NumPy to avoid redundant copies.
        """
        # Detach, move to CPU, convert to numpy in one contiguous block.
        rgb_f32: np.ndarray = tensor.detach().cpu().numpy()
        rgb_u8: np.ndarray = np.clip(rgb_f32 * 255.0, 0, 255).astype(np.uint8)
        bgr_u8: np.ndarray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        return bgr_u8

    @staticmethod
    def cv2_to_comfyui(bgr_u8: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert an OpenCV BGR uint8 array back to a ComfyUI ``[H, W, 3]`` tensor."""
        rgb_u8: np.ndarray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)
        rgb_f32: np.ndarray = rgb_u8.astype(np.float32) / 255.0
        return torch.from_numpy(rgb_f32).to(device)

    @staticmethod
    def mask_to_cv2(mask_f32: np.ndarray) -> np.ndarray:
        """Quantise a float32 binary mask ``{0.0, 1.0}`` to uint8 ``{0, 255}``."""
        return (mask_f32 * 255.0).astype(np.uint8)

    @staticmethod
    def grayscale_rec709(bgr_u8: np.ndarray) -> np.ndarray:
        """Compute perceptual luminance following ITU-R BT.709 weighting.

        Y = 0.2126·R + 0.7152·G + 0.0722·B

        Returns a single-channel uint8 image.
        """
        # OpenCV's cvtColor with COLOR_BGR2GRAY uses BT.601 weights.
        # BT.709 is preferred for SDXL content generated at sRGB gamut.
        b, g, r = cv2.split(bgr_u8)
        y = (0.2126 * r.astype(np.float32)
             + 0.7152 * g.astype(np.float32)
             + 0.0722 * b.astype(np.float32))
        return np.clip(y, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Module 2 – Gradient-Based Noise Detector
# ---------------------------------------------------------------------------

class GradientNoiseDetector:
    """Identifies isolated high-frequency artifacts via a multi-stage pipeline:

    1. Dual-threshold candidate extraction:
       - Context Mask (Low Threshold): Captures the full extent of structures.
       - Seed Mask (High Threshold): Identifies high-energy "spikes" (noise).
    2. Structural & Isolation Filtering:
       - Contextual Area: Rejects components that are part of larger structures.
       - Shape Analysis: Rejects elongated structures (e.g., hair) via aspect ratio.
       - Seed Verification: Ensures the component contains a high-energy peak.
    3. LAB chromaticity-based neighbourhood verification.
    4. Optional mask dilation.
    """

    # Laplacian kernel depth; CV_16S avoids uint8 overflow.
    _LAPLACIAN_DDEPTH: int = cv2.CV_16S
    _LAPLACIAN_KSIZE: int = 3

    # Fixed low threshold multiplier for context detection (sensitivity * multiplier).
    # 0.25 allows capturing the "body" of structures even when sensitivity is high.
    _CONTEXT_THRESHOLD_FACTOR: float = 0.25

    # Maximum aspect ratio (max(w,h)/min(w,h)) to be considered noise.
    # Higher values allow more elongated shapes; hair usually exceeds 3.0.
    _MAX_ASPECT_RATIO: float = 3.0

    # 8-connectivity for CCA.
    _CCA_CONNECTIVITY: int = 8

    # Baseline resolution for max_noise_size (1024x1024 = 1 Megapixel)
    _BASELINE_PIXELS: float = 1048576.0

    def __init__(
        self,
        gradient_sensitivity: float,
        max_noise_size: int,
        image_height: int,
        image_width: int,
        mask_dilate: int = 0,
    ) -> None:
        self._gradient_sensitivity = gradient_sensitivity
        self._image_height = image_height
        self._image_width = image_width
        self._mask_dilate = mask_dilate

        # 1. Automatic Resolution Scaling
        # The user's max_noise_size is treated as the ideal size at 1024x1024.
        # We scale it proportionally to the actual image area.
        current_pixels = float(image_height * image_width)
        scale_factor = max(1.0, current_pixels / self._BASELINE_PIXELS)
        
        # We use math.ceil to ensure even small inputs (like 1 or 2) scale up 
        # sufficiently at higher resolutions (e.g., 2K = 4x area).
        self._scaled_max_noise_size = int(math.ceil(max_noise_size * scale_factor))

        # 2. Dynamic Median Kernel
        # Must be physically larger than the max noise blob to effectively erase it.
        # Diameter = ceil(sqrt(area)).
        noise_diameter = int(math.ceil(math.sqrt(self._scaled_max_noise_size)))
        # Ensure it's an odd number and at least 5 (increased from 3 for better baseline suppression)
        self._dynamic_median_ksize = max(5, (noise_diameter + 2) | 1)

    # -- Public API ----------------------------------------------------------

    def detect(self, bgr_u8: np.ndarray) -> np.ndarray:
        """Return a binary mask identifying isolated noise."""
        gray: np.ndarray = TensorBridge.grayscale_rec709(bgr_u8)

        # 1. Generate Energy Maps.
        laplacian_energy: np.ndarray = self._compute_laplacian_energy(gray)
        median_residual: np.ndarray = self._compute_median_residual(bgr_u8)

        # 2. Extract Seed Mask (High Energy Peaks).
        # Uses user-defined sensitivity.
        seed_mask: np.ndarray = self._generate_binary_mask(
            laplacian_energy, median_residual, self._gradient_sensitivity
        )

        # 3. Extract Context Mask (Full Structural Extent).
        # Uses a lower threshold to see if "seeds" are part of a larger object.
        context_threshold = max(0.05, self._gradient_sensitivity * self._CONTEXT_THRESHOLD_FACTOR)
        context_mask: np.ndarray = self._generate_binary_mask(
            laplacian_energy, median_residual, context_threshold
        )

        # 4. Filter Context Components by shape, isolation, and seed presence.
        candidate_mask: np.ndarray = self._filter_context_components(
            context_mask, seed_mask
        )

        # 5. LAB Chromaticity Verification.
        # Pass the scaled max noise size for accurate neighbourhood ring calculation.
        verified: np.ndarray = self._filter_by_lab_chromaticity(
            candidate_mask, bgr_u8, self._scaled_max_noise_size
        )

        # 6. Optional dilation.
        return self._dilate_mask(verified)

    # -- Stage 1: Energy Computation -----------------------------------------

    def _compute_laplacian_energy(self, gray: np.ndarray) -> np.ndarray:
        """Extract 2nd-order gradient energy."""
        lap_16s = cv2.Laplacian(gray, self._LAPLACIAN_DDEPTH, ksize=self._LAPLACIAN_KSIZE)
        return cv2.convertScaleAbs(lap_16s)

    def _compute_median_residual(self, bgr_u8: np.ndarray) -> np.ndarray:
        """Extract impulse noise residual using a dynamically sized kernel."""
        median = cv2.medianBlur(bgr_u8, self._dynamic_median_ksize)
        diff = cv2.absdiff(bgr_u8, median)
        return np.max(diff, axis=2)

    def _generate_binary_mask(
        self,
        laplacian_energy: np.ndarray,
        median_residual: np.ndarray,
        sensitivity: float
    ) -> np.ndarray:
        """Combine dual-path energy into a single binary mask at given sensitivity."""
        # Laplacian path
        t_lap = int(sensitivity * 255.0)
        _, b_lap = cv2.threshold(laplacian_energy, t_lap, 255, cv2.THRESH_BINARY)

        # Median path (slightly higher base threshold to avoid floor noise)
        t_med = int(20 + sensitivity * 80.0)
        _, b_med = cv2.threshold(median_residual, t_med, 255, cv2.THRESH_BINARY)

        return cv2.bitwise_or(b_lap, b_med)

    # -- Stage 2: Structural Filtering ---------------------------------------

    def _filter_context_components(
        self,
        context_mask: np.ndarray,
        seed_mask: np.ndarray
    ) -> np.ndarray:
        """Identify components in context_mask that represent isolated noise."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            context_mask, connectivity=self._CCA_CONNECTIVITY
        )
        output = np.zeros_like(context_mask)

        # Pre-compute absolute bounds to avoid math in the loop.
        # Context area can be up to 5x the max noise size to allow for "halos"
        # around the core noise spike, but we cap it at a minimum of 50 pixels
        # to ensure small halos aren't prematurely rejected.
        max_context_area = max(self._scaled_max_noise_size * 5, 50)

        for label_id in range(1, num_labels):
            context_area = stats[label_id, cv2.CC_STAT_AREA]

            # Rule 1: Isolation Check (Context Area).
            if context_area > max_context_area:
                continue

            # Rule 2: Seed Verification & Exact Core Size Check.
            component_roi = (labels == label_id)
            seed_pixels = seed_mask[component_roi]
            seed_area = np.count_nonzero(seed_pixels)

            # The actual high-energy "core" noise must exist AND be smaller than
            # the user's requested limit (scaled for resolution).
            if seed_area == 0 or seed_area > self._scaled_max_noise_size:
                continue

            # Rule 3: Intrinsic Shape Check (Compactness via minAreaRect).
            # We calculate the rotated bounding box to find the TRUE aspect ratio,
            # avoiding the AABB blind spot for diagonal lines (e.g., 45-degree hair).
            # For tiny blobs (1-4 pixels), minAreaRect can be unstable or overkill,
            # and they are inherently compact enough, so we skip the expensive check.
            if context_area > 4:
                # Find the (y, x) coordinates of all pixels in this component
                # Note: np.where returns (y_coords, x_coords)
                coords = np.column_stack(np.where(component_roi))
                # OpenCV minAreaRect expects (x, y) coordinates
                coords_xy = coords[:, ::-1]
                
                # minAreaRect returns (center(x, y), (width, height), angle of rotation)
                _, (rect_w, rect_h), _ = cv2.minAreaRect(coords_xy.astype(np.float32))
                
                # Calculate true intrinsic aspect ratio
                max_dim = max(rect_w, rect_h)
                min_dim = max(1e-5, min(rect_w, rect_h)) # Prevent division by zero
                intrinsic_aspect_ratio = max_dim / min_dim
                
                if intrinsic_aspect_ratio > self._MAX_ASPECT_RATIO:
                    continue

            # If it passes all tests, we output the ENTIRE context blob.
            output[component_roi] = 255

        return output

    # -- Stage 3: LAB chromaticity verification ------------------------------

    def _filter_by_lab_chromaticity(
        self,
        candidate_mask: np.ndarray,
        bgr_u8: np.ndarray,
        effective_noise_size: int,
    ) -> np.ndarray:
        """Reject candidates whose chromaticity matches their neighbourhood,
        indicating a natural highlight or specular reflection rather than
        VAE noise.

        This uses a relative contrast (steepness) approach for luminance.
        It checks if the component's internal mean drops severely from its peak 
        relative to the background, forming a "cliff" (impulse noise) rather 
        than a smooth "hill" (star).
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_mask, connectivity=self._CCA_CONNECTIVITY
        )
        verified_mask = np.zeros_like(candidate_mask)

        # Neighbourhood ring width in pixels, using the effective (scaled) noise size.
        ring_width: int = max(3, int(math.sqrt(effective_noise_size)) + 2)

        # Chromaticity distance threshold (a*b* plane).
        # We still keep this to catch pure color bleeds (e.g. purple/green fireflies)
        # that might not have a huge luminance cliff.
        chroma_threshold: float = 3.0 + self._gradient_sensitivity * 15.0

        h, w = candidate_mask.shape[:2]
        lab_f32: np.ndarray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

        for label_id in range(1, num_labels):
            x0: int = stats[label_id, cv2.CC_STAT_LEFT]
            y0: int = stats[label_id, cv2.CC_STAT_TOP]
            bw: int = stats[label_id, cv2.CC_STAT_WIDTH]
            bh: int = stats[label_id, cv2.CC_STAT_HEIGHT]

            # Bounding box for the surrounding annular region.
            rx0 = max(0, x0 - ring_width)
            ry0 = max(0, y0 - ring_width)
            rx1 = min(w, x0 + bw + ring_width)
            ry1 = min(h, y0 + bh + ring_width)

            roi_labels = labels[ry0:ry1, rx0:rx1]
            roi_lab = lab_f32[ry0:ry1, rx0:rx1]

            component_pixels = roi_labels == label_id
            neighbourhood_pixels = ~component_pixels

            if not np.any(neighbourhood_pixels):
                continue

            # 1. Analyze the internal structure of the candidate (L* channel)
            comp_luma = roi_lab[component_pixels, 0]
            peak_luma = np.max(comp_luma)
            mean_comp_luma = np.mean(comp_luma)

            # 2. Analyze the background (L* channel)
            mean_bg_lab = roi_lab[neighbourhood_pixels].mean(axis=0)
            mean_bg_luma = mean_bg_lab[0]

            # --- Check 1: Steepness of the Cliff (Relative Contrast) ---
            total_drop = peak_luma - mean_bg_luma
            
            # If there's barely any contrast with the background, it's not noise.
            if total_drop <= 5.0:
                continue

            # How much does the internal mean drop compared to the peak?
            # Ratio near 1.0 = steep cliff (noise). Ratio near 0.5 = smooth hill (star).
            internal_drop_ratio = (peak_luma - mean_comp_luma) / total_drop
            
            # Sensitivity adjusts how steep the cliff needs to be.
            # E.g., sens 0.35 -> threshold 0.53. Requires a fairly steep drop.
            cliff_threshold = 0.6 - (self._gradient_sensitivity * 0.2)
            
            is_cliff = internal_drop_ratio > cliff_threshold

            # --- Check 2: Chromatic Aberration (Color Shift) ---
            mean_comp_chroma = roi_lab[component_pixels, 1:].mean(axis=0)
            chroma_dist = float(np.linalg.norm(mean_comp_chroma - mean_bg_lab[1:]))
            is_chromatic_noise = chroma_dist >= chroma_threshold

            # Accept if it's a steep cliff OR exhibits severe color bleed.
            if is_cliff or is_chromatic_noise:
                verified_mask[labels == label_id] = 255

        return verified_mask

    # -- Stage 4: Mask dilation ------------------------------------------------

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """Expand detected noise regions by *mask_dilate* pixels.

        Morphological dilation ensures the mask fully covers noise edges
        that may fall below the Laplacian energy threshold.  This prevents
        Telea inpainting from sampling corrupted boundary pixels as source
        data during reconstruction.

        A circular structuring element is used to produce isotropic growth
        without favouring axis-aligned directions.
        """
        if self._mask_dilate <= 0:
            return mask

        diameter: int = 2 * self._mask_dilate + 1
        kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (diameter, diameter)
        )
        return cv2.dilate(mask, kernel, iterations=1)


# ---------------------------------------------------------------------------
# Module 3 – Telea Inpainter
# ---------------------------------------------------------------------------

class TeleaInpainter:
    """Fills masked regions using the Fast Marching Method (Telea 2004).

    The inpaint radius is derived from *max_noise_size* so that larger
    permitted noise areas receive a proportionally wider reconstruction
    neighbourhood without excessive blurring.
    """

    _MIN_RADIUS: int = 2
    _MAX_RADIUS: int = 7

    @classmethod
    def inpaint(cls, bgr_u8: np.ndarray, mask_u8: np.ndarray, max_noise_size: int) -> np.ndarray:
        """Apply Telea inpainting where *mask_u8* is non-zero.

        Args:
            bgr_u8:         Source image in BGR uint8 format.
            mask_u8:        Binary mask (255 = region to repair).
            max_noise_size: Maximum expected defect area; controls inpaint radius.

        Returns:
            Repaired BGR uint8 image.
        """
        radius: int = cls._compute_radius(max_noise_size)
        return cv2.inpaint(bgr_u8, mask_u8, radius, cv2.INPAINT_TELEA)

    @classmethod
    def _compute_radius(cls, max_noise_size: int) -> int:
        """Derive inpaint radius from the maximum noise area.

        Radius = ceil(sqrt(area)) clamped to [_MIN_RADIUS, _MAX_RADIUS]
        to prevent excessive performance degradation on large masks.
        """
        r = int(math.ceil(math.sqrt(max_noise_size)))
        return max(cls._MIN_RADIUS, min(r, cls._MAX_RADIUS))


# ---------------------------------------------------------------------------
# Module 4 – Debug Overlay Renderer
# ---------------------------------------------------------------------------

class DebugOverlayRenderer:
    """Generates a semi-transparent red overlay on detected noise regions
    for visual parameter tuning."""

    _OVERLAY_COLOUR_BGR: Tuple[int, int, int] = (0, 0, 255)  # Red in BGR.
    _OVERLAY_ALPHA: float = 0.45

    @classmethod
    def render(cls, bgr_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Composite a semi-transparent red mask over the source image.

        Non-masked pixels are left unchanged.
        """
        overlay: np.ndarray = bgr_u8.copy()
        overlay[mask_u8 > 0] = cls._OVERLAY_COLOUR_BGR
        blended: np.ndarray = cv2.addWeighted(
            bgr_u8, 1.0 - cls._OVERLAY_ALPHA,
            overlay, cls._OVERLAY_ALPHA,
            0.0,
        )
        return blended


# ---------------------------------------------------------------------------
# Module 5 – ComfyUI Node Definition
# ---------------------------------------------------------------------------

class VAENoiseFixNode:
    """ComfyUI custom node that automatically detects and repairs SDXL VAE
    high-frequency noise artifacts using traditional computer vision.

    Processing pipeline per frame:
        1. Transfer single frame from GPU to CPU (as BGR uint8).
        2. Run three-stage gradient-based noise detection.
        3. Either render debug overlay or apply Telea inpainting.
        4. Transfer result back to GPU as ComfyUI tensor.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "gradient_sensitivity": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Laplacian energy threshold (normalised). "
                            "Lower = more sensitive; higher = only extreme gradients."
                        ),
                    },
                ),
                "max_noise_size": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "tooltip": (
                            "Maximum connected-component area (in pixels) "
                            "to classify as noise. Blobs exceeding this are ignored."
                        ),
                    },
                ),
                "mask_dilate": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "display": "slider",
                        "tooltip": (
                            "Dilation radius (pixels) applied to the detected mask "
                            "before inpainting. Expands each noise region so that "
                            "boundary pixels are fully covered, preventing the "
                            "inpainter from sampling corrupted edge data."
                        ),
                    },
                ),
                "preview_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When enabled, outputs a red semi-transparent overlay "
                            "on detected noise instead of the repaired image."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/postprocessing"

    def execute(
        self,
        image: torch.Tensor,
        gradient_sensitivity: float,
        max_noise_size: int,
        mask_dilate: int,
        preview_mask: bool,
    ) -> Tuple[torch.Tensor]:
        """Entry point invoked by ComfyUI runtime.

        Iterates over the batch dimension ``[B, H, W, C]`` so that
        multi-frame (video / batched) inputs are handled correctly.
        """
        device: torch.device = image.device
        batch_size: int = image.shape[0]
        results: List[torch.Tensor] = []

        for idx in range(batch_size):
            frame_tensor: torch.Tensor = image[idx]  # [H, W, C]
            processed: torch.Tensor = self._process_single_frame(
                frame_tensor, gradient_sensitivity, max_noise_size,
                mask_dilate, preview_mask, device,
            )
            results.append(processed)

        # Re-stack along batch dimension → [B, H, W, C].
        return (torch.stack(results, dim=0),)

    # -- Per-frame processing ------------------------------------------------

    @staticmethod
    def _process_single_frame(
        frame: torch.Tensor,
        gradient_sensitivity: float,
        max_noise_size: int,
        mask_dilate: int,
        preview_mask: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Run detection → inpaint / overlay for one frame."""
        bgr_u8: np.ndarray = TensorBridge.comfyui_to_cv2(frame)
        h, w = bgr_u8.shape[:2]

        detector = GradientNoiseDetector(
            gradient_sensitivity=gradient_sensitivity,
            max_noise_size=max_noise_size,
            image_height=h,
            image_width=w,
            mask_dilate=mask_dilate,
        )
        noise_mask: np.ndarray = detector.detect(bgr_u8)

        if preview_mask:
            output_bgr = DebugOverlayRenderer.render(bgr_u8, noise_mask)
        else:
            if np.any(noise_mask):
                output_bgr = TeleaInpainter.inpaint(bgr_u8, noise_mask, max_noise_size)
            else:
                output_bgr = bgr_u8

        return TensorBridge.cv2_to_comfyui(output_bgr, device)


# ---------------------------------------------------------------------------
# ComfyUI Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VAENoiseFix": VAENoiseFixNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAENoiseFix": "VAE Noise Fix (Traditional CV)",
}
