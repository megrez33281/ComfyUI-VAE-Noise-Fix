"""
ComfyUI Custom Node: VAE High-Frequency Noise Auto-Detection and Repair

Detects and inpaints isolated high-frequency artifacts produced by SDXL VAE
decoding, using a lightweight traditional computer vision pipeline:
Laplacian gradient extraction → Connected Component Analysis → Neighborhood
variance verification → Telea (Fast Marching Method) inpainting.

Architecture follows SOLID principles with explicit GPU/CPU transfer boundaries.
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
    """Identifies isolated high-frequency artifacts via a three-stage pipeline:

    1. Laplacian energy extraction on perceptual luminance.
    2. Binary thresholding + Connected Component Area filtering (CCA).
    3. Neighbourhood colour-variance verification to reject natural highlights.
    """

    # Laplacian kernel depth; CV_16S avoids uint8 overflow during convolution.
    _LAPLACIAN_DDEPTH: int = cv2.CV_16S
    _LAPLACIAN_KSIZE: int = 3

    # 8-connectivity groups diagonally adjacent pixels as one component.
    _CCA_CONNECTIVITY: int = 8

    def __init__(
        self,
        gradient_sensitivity: float,
        max_noise_size: int,
        image_height: int,
        image_width: int,
    ) -> None:
        self._gradient_sensitivity = gradient_sensitivity
        self._max_noise_size = max_noise_size
        self._image_height = image_height
        self._image_width = image_width

    # -- Public API ----------------------------------------------------------

    def detect(self, bgr_u8: np.ndarray) -> np.ndarray:
        """Return a binary mask ``[H, W]`` (uint8, 255 = noise) for one frame.

        Pipeline order:
            Laplacian energy map → adaptive threshold → CCA area filter →
            neighbourhood variance rejection.
        """
        gray: np.ndarray = TensorBridge.grayscale_rec709(bgr_u8)
        energy_map: np.ndarray = self._compute_laplacian_energy(gray)
        binary: np.ndarray = self._adaptive_threshold(energy_map)
        area_filtered: np.ndarray = self._filter_by_area(binary)
        variance_filtered: np.ndarray = self._filter_by_neighbourhood_variance(
            area_filtered, bgr_u8
        )
        return variance_filtered

    # -- Stage 1: Laplacian energy -------------------------------------------

    def _compute_laplacian_energy(self, gray: np.ndarray) -> np.ndarray:
        """Apply Laplacian filter to extract high-frequency gradient energy.

        The absolute value of the 2nd-order derivative highlights regions with
        abrupt intensity transitions regardless of sign (bright-on-dark or
        dark-on-bright).
        """
        laplacian_16s: np.ndarray = cv2.Laplacian(
            gray,
            self._LAPLACIAN_DDEPTH,
            ksize=self._LAPLACIAN_KSIZE,
        )
        return cv2.convertScaleAbs(laplacian_16s)

    # -- Stage 2: Threshold + CCA area filter --------------------------------

    def _adaptive_threshold(self, energy_map: np.ndarray) -> np.ndarray:
        """Binarise the energy map using *gradient_sensitivity* as the cut-off.

        ``gradient_sensitivity`` is expressed on a normalised [0, 1] scale and
        mapped to the uint8 range [0, 255].  Lower values yield higher
        sensitivity (more candidates); higher values restrict detection to only
        the most extreme gradients.
        """
        threshold_value: int = int(self._gradient_sensitivity * 255.0)
        _, binary = cv2.threshold(
            energy_map, threshold_value, 255, cv2.THRESH_BINARY
        )
        return binary

    def _filter_by_area(self, binary: np.ndarray) -> np.ndarray:
        """Discard connected components whose pixel area exceeds *max_noise_size*.

        Components with area ≤ max_noise_size are retained as noise candidates.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=self._CCA_CONNECTIVITY
        )
        mask = np.zeros_like(binary)

        # Label 0 is the background; skip it.
        for label_id in range(1, num_labels):
            area: int = stats[label_id, cv2.CC_STAT_AREA]
            if area <= self._max_noise_size:
                mask[labels == label_id] = 255

        return mask

    # -- Stage 3: Neighbourhood colour-variance verification -----------------

    def _filter_by_neighbourhood_variance(
        self,
        candidate_mask: np.ndarray,
        bgr_u8: np.ndarray,
    ) -> np.ndarray:
        """Reject candidates that exhibit smooth gradient transitions typical
        of natural highlights (stars, specular reflections).

        For each connected component, compare the mean colour of its pixels
        against the mean colour of its surrounding annular neighbourhood.
        Noise pixels produce a large Euclidean colour distance (impulse-like),
        whereas natural bright features show gradual falloff.

        The dynamic rejection threshold is derived from *gradient_sensitivity*:
        less sensitive settings require a larger colour jump to qualify as noise.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_mask, connectivity=self._CCA_CONNECTIVITY
        )
        verified_mask = np.zeros_like(candidate_mask)

        # Neighbourhood ring width in pixels.  Adaptive to resolution so that
        # the annulus always captures enough context.
        ring_width: int = max(3, int(math.sqrt(self._max_noise_size)) + 2)

        # Colour-distance threshold.  Mapped from sensitivity: higher
        # sensitivity (lower gradient_sensitivity) → lower distance threshold.
        colour_distance_threshold: float = 30.0 + self._gradient_sensitivity * 120.0

        h, w = candidate_mask.shape[:2]
        bgr_f32: np.ndarray = bgr_u8.astype(np.float32)

        for label_id in range(1, num_labels):
            x0: int = stats[label_id, cv2.CC_STAT_LEFT]
            y0: int = stats[label_id, cv2.CC_STAT_TOP]
            bw: int = stats[label_id, cv2.CC_STAT_WIDTH]
            bh: int = stats[label_id, cv2.CC_STAT_HEIGHT]

            # Compute bounding box for the surrounding annular region.
            rx0 = max(0, x0 - ring_width)
            ry0 = max(0, y0 - ring_width)
            rx1 = min(w, x0 + bw + ring_width)
            ry1 = min(h, y0 + bh + ring_width)

            roi_labels = labels[ry0:ry1, rx0:rx1]
            roi_bgr = bgr_f32[ry0:ry1, rx0:rx1]

            component_pixels = roi_labels == label_id
            neighbourhood_pixels = ~component_pixels

            if not np.any(neighbourhood_pixels):
                continue

            mean_component: np.ndarray = roi_bgr[component_pixels].mean(axis=0)
            mean_neighbour: np.ndarray = roi_bgr[neighbourhood_pixels].mean(axis=0)

            # Euclidean distance in BGR colour space.
            colour_dist: float = float(np.linalg.norm(mean_component - mean_neighbour))

            if colour_dist >= colour_distance_threshold:
                verified_mask[labels == label_id] = 255

        return verified_mask


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
                preview_mask, device,
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
