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

    1. Dual-path candidate detection:
       a) Laplacian energy extraction on perceptual luminance.
       b) Median filter residual to catch impulse noise missed by gradients.
       Candidates from both paths are merged (union).
    2. Connected Component Area filtering (CCA).
    3. LAB chromaticity-based neighbourhood verification:
       separates genuine noise (chromatic aberration) from natural
       highlights and specular reflections (luminance-only shift).
    4. Optional mask dilation for full boundary coverage.
    """

    # Laplacian kernel depth; CV_16S avoids uint8 overflow during convolution.
    _LAPLACIAN_DDEPTH: int = cv2.CV_16S
    _LAPLACIAN_KSIZE: int = 3

    # Median filter kernel size (must be odd).
    _MEDIAN_KSIZE: int = 3

    # 8-connectivity groups diagonally adjacent pixels as one component.
    _CCA_CONNECTIVITY: int = 8

    def __init__(
        self,
        gradient_sensitivity: float,
        max_noise_size: int,
        image_height: int,
        image_width: int,
        mask_dilate: int = 0,
    ) -> None:
        self._gradient_sensitivity = gradient_sensitivity
        self._max_noise_size = max_noise_size
        self._image_height = image_height
        self._image_width = image_width
        self._mask_dilate = mask_dilate

    # -- Public API ----------------------------------------------------------

    def detect(self, bgr_u8: np.ndarray) -> np.ndarray:
        """Return a binary mask ``[H, W]`` (uint8, 255 = noise) for one frame.

        Pipeline order:
            (Laplacian energy ∪ Median residual) → CCA area filter →
            LAB chromaticity verification → optional mask dilation.
        """
        gray: np.ndarray = TensorBridge.grayscale_rec709(bgr_u8)

        # Dual-path candidate detection.
        laplacian_candidates: np.ndarray = self._detect_laplacian(gray)
        median_candidates: np.ndarray = self._detect_median_residual(bgr_u8)
        combined: np.ndarray = cv2.bitwise_or(laplacian_candidates, median_candidates)

        area_filtered: np.ndarray = self._filter_by_area(combined)
        verified: np.ndarray = self._filter_by_lab_chromaticity(
            area_filtered, bgr_u8
        )
        dilated: np.ndarray = self._dilate_mask(verified)
        return dilated

    # -- Stage 1a: Laplacian energy ------------------------------------------

    def _detect_laplacian(self, gray: np.ndarray) -> np.ndarray:
        """Laplacian energy → adaptive threshold → binary candidates.

        The absolute value of the 2nd-order derivative highlights regions with
        abrupt intensity transitions regardless of sign (bright-on-dark or
        dark-on-bright).
        """
        laplacian_16s: np.ndarray = cv2.Laplacian(
            gray,
            self._LAPLACIAN_DDEPTH,
            ksize=self._LAPLACIAN_KSIZE,
        )
        energy_map: np.ndarray = cv2.convertScaleAbs(laplacian_16s)
        threshold_value: int = int(self._gradient_sensitivity * 255.0)
        _, binary = cv2.threshold(
            energy_map, threshold_value, 255, cv2.THRESH_BINARY
        )
        return binary

    # -- Stage 1b: Median filter residual ------------------------------------

    def _detect_median_residual(self, bgr_u8: np.ndarray) -> np.ndarray:
        """Detect impulse noise by comparing the image to its median-filtered
        version.

        Median filtering is the canonical impulse-noise suppressor: it
        replaces each pixel with the median of its local neighbourhood,
        effectively erasing salt-and-pepper spikes while preserving edges.
        The per-channel absolute difference between the original and the
        median-filtered image isolates exactly these spikes.

        The maximum residual across B, G, R channels is thresholded to
        produce a binary candidate mask.  The threshold is derived from
        *gradient_sensitivity* to stay consistent with the Laplacian path.
        """
        median_filtered: np.ndarray = cv2.medianBlur(bgr_u8, self._MEDIAN_KSIZE)

        # Per-channel absolute difference → take max across channels.
        diff: np.ndarray = cv2.absdiff(bgr_u8, median_filtered)
        max_diff: np.ndarray = np.max(diff, axis=2)

        # Threshold: lower sensitivity → lower threshold → catch more.
        threshold_value: int = int(20 + self._gradient_sensitivity * 80.0)
        _, binary = cv2.threshold(
            max_diff, threshold_value, 255, cv2.THRESH_BINARY
        )
        return binary

    # -- Stage 2: CCA area filter --------------------------------------------

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

    # -- Stage 3: LAB chromaticity verification ------------------------------

    def _filter_by_lab_chromaticity(
        self,
        candidate_mask: np.ndarray,
        bgr_u8: np.ndarray,
    ) -> np.ndarray:
        """Reject candidates whose chromaticity matches their neighbourhood,
        indicating a natural highlight or specular reflection rather than
        VAE noise.

        The verification operates in CIE-LAB colour space.  Channel L*
        encodes lightness; channels a* and b* encode chromaticity.  By
        measuring distance in the a*b* plane only, the test becomes
        invariant to brightness differences:

        - **Noise**: chromatic aberration → high a*b* distance from neighbours.
        - **Reflection**: same hue, just brighter → low a*b* distance.

        A secondary luminance-spike check catches monochromatic impulse
        noise (pure white/black spikes) that has near-zero chromaticity
        shift but an extreme lightness jump.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_mask, connectivity=self._CCA_CONNECTIVITY
        )
        verified_mask = np.zeros_like(candidate_mask)

        # Neighbourhood ring width in pixels.
        ring_width: int = max(3, int(math.sqrt(self._max_noise_size)) + 2)

        # Chromaticity distance threshold (a*b* plane).
        # Lower sensitivity → lower threshold → catch more.
        chroma_threshold: float = 3.0 + self._gradient_sensitivity * 20.0

        # Luminance spike threshold (L* channel).
        # Catches pure white/black impulse noise with no chromatic shift.
        luma_spike_threshold: float = 25.0 + self._gradient_sensitivity * 40.0

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

            mean_comp: np.ndarray = roi_lab[component_pixels].mean(axis=0)
            mean_neigh: np.ndarray = roi_lab[neighbourhood_pixels].mean(axis=0)

            # Chromaticity distance (a*b* channels only, ignoring L*).
            chroma_dist: float = float(
                np.linalg.norm(mean_comp[1:] - mean_neigh[1:])
            )

            # Luminance difference (L* channel).
            luma_diff: float = abs(float(mean_comp[0] - mean_neigh[0]))

            # Accept if chromaticity deviates (colour noise) OR if
            # luminance spikes extremely (monochromatic impulse noise).
            if chroma_dist >= chroma_threshold or luma_diff >= luma_spike_threshold:
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
