"""
End-to-end pipeline + preview-mode dispatch.

Both the GUI and the ComfyUI node need the same fundamental capability:

    BGR image  ─┐
                 ▶ run detection ─▶ run inpaint ─▶ pick which view to return
    parameters ─┘                                          │
                                                           ▼
                                  Original / Overlay / Mask Solo / Repaired /
                                  Side-by-side / Laplacian / Median /
                                  Seed / Context / Filtered / Verified

The GUI always wants every view simultaneously (caches them all and lets
the user toggle).  The ComfyUI node wants exactly one view at a time
(driven by the ``preview_mode`` enum input).

This module unifies both flows by:

  1. Running detection + inpainting once.
  2. Storing all 11 visualisations in a ``DetectionResult`` DTO.
  3. Exposing a ``select_view(mode)`` helper that returns one of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from .detector import GradientNoiseDetector, IntermediateMaps
from .inpainter import TeleaInpainter
from .overlay import (
    DebugOverlayRenderer,
    MaskSoloRenderer,
    SideBySideRenderer,
)
from .statistics import DetectionStats, NoiseStatistics


# ---------------------------------------------------------------------------
# Preview-mode enum
# ---------------------------------------------------------------------------

class PreviewMode(str, Enum):
    """Output selection for ``NoiseFixPipeline``.

    String-valued so it serialises trivially to ComfyUI dropdown choices.
    """
    ORIGINAL          = "Original"
    MASK_OVERLAY      = "Mask Overlay"
    MASK_SOLO         = "Mask Only"
    REPAIRED          = "Repaired"
    SIDE_BY_SIDE      = "Side-by-Side"
    LAPLACIAN_BINARY  = "Laplacian Energy Map (Binary)"
    MEDIAN_BINARY     = "Median Residual Map (Binary)"
    SEED_MASK         = "Combined Seed Mask"
    CONTEXT_MASK      = "Context Mask (Low Thresh)"
    FILTERED_MASK     = "Filtered Candidates (Shape/Iso)"
    VERIFIED_MASK     = "Final Verified Mask (LAB)"

    @classmethod
    def choices(cls) -> list[str]:
        return [m.value for m in cls]


# ---------------------------------------------------------------------------
# Result DTO
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionResult:
    """Everything one detection + inpaint pass produces.

    All BGR images are uint8 ``[H, W, 3]``.
    All masks are uint8 ``[H, W]`` ∈ {0, 255}.
    """

    # Source
    original_bgr:   np.ndarray
    # Algorithm output
    final_mask:     np.ndarray   # post-dilation; the mask actually inpainted
    repaired_bgr:   np.ndarray   # final repaired image
    # Renders
    overlay_bgr:    np.ndarray
    mask_solo_bgr:  np.ndarray
    side_by_side:   np.ndarray
    # Intermediate stages (BGR-converted for direct display)
    laplacian_bgr:  np.ndarray
    median_bgr:     np.ndarray
    seed_bgr:       np.ndarray
    context_bgr:    np.ndarray
    filtered_bgr:   np.ndarray
    verified_bgr:   np.ndarray   # pre-dilation
    # Metadata
    stats:          DetectionStats

    # ----- View selector --------------------------------------------------

    def select_view(self, mode: PreviewMode) -> np.ndarray:
        """Return the BGR image for the requested preview mode."""
        return _VIEW_DISPATCH[mode](self)


_VIEW_DISPATCH = {
    PreviewMode.ORIGINAL:         lambda r: r.original_bgr,
    PreviewMode.MASK_OVERLAY:     lambda r: r.overlay_bgr,
    PreviewMode.MASK_SOLO:        lambda r: r.mask_solo_bgr,
    PreviewMode.REPAIRED:         lambda r: r.repaired_bgr,
    PreviewMode.SIDE_BY_SIDE:     lambda r: r.side_by_side,
    PreviewMode.LAPLACIAN_BINARY: lambda r: r.laplacian_bgr,
    PreviewMode.MEDIAN_BINARY:    lambda r: r.median_bgr,
    PreviewMode.SEED_MASK:        lambda r: r.seed_bgr,
    PreviewMode.CONTEXT_MASK:     lambda r: r.context_bgr,
    PreviewMode.FILTERED_MASK:    lambda r: r.filtered_bgr,
    PreviewMode.VERIFIED_MASK:    lambda r: r.verified_bgr,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class NoiseFixPipeline:
    """One-shot pipeline: BGR in → ``DetectionResult`` out.

    Stateless — construct once and reuse for any number of frames as long
    as the parameters are unchanged.  For per-frame parameter sweeps,
    construct per call (it is cheap; nothing is allocated until ``run``).
    """

    def __init__(
        self,
        gradient_sensitivity: float,
        max_noise_size: int,
        mask_dilate: int,
    ) -> None:
        self._gradient_sensitivity = gradient_sensitivity
        self._max_noise_size = max_noise_size
        self._mask_dilate = mask_dilate

    # -- Public API ----------------------------------------------------------

    def run(
        self,
        bgr_u8: np.ndarray,
        skip_inpaint: bool = False,
    ) -> DetectionResult:
        """Execute detection + inpainting + view rendering.

        Args:
            bgr_u8:       Source image, OpenCV BGR uint8.
            skip_inpaint: If True, skip the Telea call entirely;
                          ``repaired_bgr`` and ``side_by_side`` will hold
                          copies of the original.  Used by the standalone
                          Detector node where inpainting happens downstream
                          (potentially after manual mask editing).
        """
        h, w = bgr_u8.shape[:2]

        detector = GradientNoiseDetector(
            gradient_sensitivity=self._gradient_sensitivity,
            max_noise_size=self._max_noise_size,
            image_height=h,
            image_width=w,
            mask_dilate=self._mask_dilate,
        )

        t0 = time.perf_counter()
        maps: IntermediateMaps = detector.detect_with_intermediates(bgr_u8)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Inpaint (skip if requested, or if mask is empty).
        if skip_inpaint or not np.any(maps.final_mask):
            repaired = bgr_u8.copy()
        else:
            repaired = TeleaInpainter.inpaint(
                bgr_u8, maps.final_mask, self._max_noise_size
            )

        # Rendered views.
        overlay = DebugOverlayRenderer.render(bgr_u8, maps.final_mask)
        mask_solo = MaskSoloRenderer.render(maps.final_mask)
        side_by_side = SideBySideRenderer.compose(bgr_u8, repaired)

        # Stats
        stats = NoiseStatistics.compute(maps.final_mask, (h, w), elapsed_ms)

        return DetectionResult(
            original_bgr=bgr_u8,
            final_mask=maps.final_mask,
            repaired_bgr=repaired,
            overlay_bgr=overlay,
            mask_solo_bgr=mask_solo,
            side_by_side=side_by_side,
            laplacian_bgr=cv2.cvtColor(maps.laplacian_binary, cv2.COLOR_GRAY2BGR),
            median_bgr=cv2.cvtColor(maps.median_binary, cv2.COLOR_GRAY2BGR),
            seed_bgr=cv2.cvtColor(maps.seed_mask, cv2.COLOR_GRAY2BGR),
            context_bgr=cv2.cvtColor(maps.context_mask, cv2.COLOR_GRAY2BGR),
            filtered_bgr=cv2.cvtColor(maps.filtered_mask, cv2.COLOR_GRAY2BGR),
            verified_bgr=cv2.cvtColor(maps.verified_mask, cv2.COLOR_GRAY2BGR),
            stats=stats,
        )

    # -- Convenience ---------------------------------------------------------

    def run_view(
        self,
        bgr_u8: np.ndarray,
        mode: PreviewMode,
        skip_inpaint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, DetectionStats]:
        """Run the pipeline and return ``(view_bgr, final_mask, stats)``.

        Convenience for single-view consumers like the ComfyUI node.
        ``skip_inpaint=True`` avoids the Telea call when the requested
        view is detection-only (e.g. ``MASK_OVERLAY``, ``MASK_SOLO`` or
        any intermediate stage map).
        """
        result = self.run(bgr_u8, skip_inpaint=skip_inpaint)
        return result.select_view(mode), result.final_mask, result.stats
