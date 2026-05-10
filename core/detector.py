"""
Detection orchestrator.

Wires together:

    energy → thresholding → structural filter → chromatic filter → morphology

Returns either the final mask alone (legacy API) or a richer
``DetectionResult`` containing every intermediate map (new API used by
both the GUI and the ComfyUI debug previews).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .tensor_bridge import TensorBridge
from .energy import LaplacianEnergyExtractor, MedianResidualExtractor
from .thresholding import DualPathMaskGenerator
from .structural_filter import StructuralFilter
from .chromatic_filter import ChromaticFilter
from .morphology import MaskDilator


# ---------------------------------------------------------------------------
# Result DTO
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntermediateMaps:
    """Every binary stage map produced during detection.

    Holding these in a single immutable DTO lets the GUI render any debug
    view, and lets the ComfyUI node expose any intermediate state, without
    forcing them to call private detector methods (Demeter / SRP).
    """
    laplacian_binary:   np.ndarray  # uint8 ─ Laplacian above threshold
    median_binary:      np.ndarray  # uint8 ─ Median residual above threshold
    seed_mask:          np.ndarray  # uint8 ─ dual-path OR @ user sensitivity
    context_mask:       np.ndarray  # uint8 ─ dual-path OR @ low sensitivity
    filtered_mask:      np.ndarray  # uint8 ─ after structural filter
    verified_mask:      np.ndarray  # uint8 ─ after chromatic filter
    final_mask:         np.ndarray  # uint8 ─ after dilation (== detect() output)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class GradientNoiseDetector:
    """Multi-stage gradient + chromatic noise detector.

    Construction is cheap (no image processing happens until ``detect`` /
    ``detect_with_intermediates`` is called).  This makes the detector
    safe to instantiate per-frame in batched workflows.
    """

    # Low-threshold multiplier for context map.
    _CONTEXT_THRESHOLD_FACTOR: float = 0.25
    _CONTEXT_THRESHOLD_FLOOR:  float = 0.05

    # Resolution baseline for max_noise_size auto-scaling.
    _BASELINE_PIXELS: float = 1024.0 * 1024.0

    def __init__(
        self,
        gradient_sensitivity: float,
        max_noise_size: int,
        image_height: int,
        image_width: int,
        mask_dilate: int = 0,
    ) -> None:
        self._gradient_sensitivity = float(gradient_sensitivity)
        self._image_height = int(image_height)
        self._image_width = int(image_width)
        self._mask_dilate = int(mask_dilate)

        # Resolution-aware scaling: treat max_noise_size as the ideal value
        # at 1 megapixel and scale up proportionally for larger images.
        current_pixels = float(image_height * image_width)
        scale_factor = max(1.0, current_pixels / self._BASELINE_PIXELS)
        self._scaled_max_noise_size = int(math.ceil(max_noise_size * scale_factor))

        # Compose stages.
        self._laplacian = LaplacianEnergyExtractor()
        self._median = MedianResidualExtractor(self._scaled_max_noise_size)
        self._mask_gen = DualPathMaskGenerator()
        self._structural = StructuralFilter(self._scaled_max_noise_size)
        self._chromatic = ChromaticFilter(
            gradient_sensitivity=self._gradient_sensitivity,
            scaled_max_noise_size=self._scaled_max_noise_size,
        )
        self._dilator = MaskDilator(self._mask_dilate)

    # -- Public properties (read-only) ---------------------------------------

    @property
    def scaled_max_noise_size(self) -> int:
        return self._scaled_max_noise_size

    @property
    def median_kernel_size(self) -> int:
        return self._median.kernel_size

    # -- Public API ----------------------------------------------------------

    def detect(self, bgr_u8: np.ndarray) -> np.ndarray:
        """Return the final binary mask only (legacy fast path)."""
        return self.detect_with_intermediates(bgr_u8).final_mask

    def detect_with_intermediates(self, bgr_u8: np.ndarray) -> IntermediateMaps:
        """Run the full pipeline and return every stage's mask.

        This is the canonical entry point. Callers that only need the final
        mask use ``detect``; callers that want to debug or visualise pick
        fields off the returned DTO.
        """
        gray = TensorBridge.grayscale_rec709(bgr_u8)
        sensitivity = self._gradient_sensitivity

        # Stage 1 — Energy maps.
        lap_energy = self._laplacian.extract(gray)
        med_residual = self._median.extract(bgr_u8)

        # Stage 1.5 — Per-path binary maps (debug-only; rebuild at user
        # sensitivity so the GUI can show what each contributes).
        t_lap = int(sensitivity * 255.0)
        _, lap_binary = cv2.threshold(lap_energy, t_lap, 255, cv2.THRESH_BINARY)
        t_med = int(20 + sensitivity * 80.0)
        _, med_binary = cv2.threshold(med_residual, t_med, 255, cv2.THRESH_BINARY)

        # Stage 2 — Seed (high-energy) and context (low-energy) masks.
        seed_mask = self._mask_gen.generate(lap_energy, med_residual, sensitivity)
        context_thresh = max(
            self._CONTEXT_THRESHOLD_FLOOR,
            sensitivity * self._CONTEXT_THRESHOLD_FACTOR,
        )
        context_mask = self._mask_gen.generate(lap_energy, med_residual, context_thresh)

        # Stage 3 — Structural filter.
        filtered_mask = self._structural.filter(context_mask, seed_mask)

        # Stage 4 — Chromatic verification.
        verified_mask = self._chromatic.filter(filtered_mask, bgr_u8)

        # Stage 5 — Dilation.
        final_mask = self._dilator.dilate(verified_mask)

        return IntermediateMaps(
            laplacian_binary=lap_binary,
            median_binary=med_binary,
            seed_mask=seed_mask,
            context_mask=context_mask,
            filtered_mask=filtered_mask,
            verified_mask=verified_mask,
            final_mask=final_mask,
        )
