"""
Visualization renderers.

These are pure functions that consume an image + mask and emit a new
image suitable for display. Keeping them out of the detector and
inpainter keeps both algorithms headless and trivially testable.

Each renderer is a stateless class with a ``render`` (or ``compose``)
classmethod so they can be referenced by name without instantiation.
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Translucent red overlay
# ---------------------------------------------------------------------------

class DebugOverlayRenderer:
    """Composite a translucent red overlay on the masked region."""

    _OVERLAY_COLOUR_BGR: Tuple[int, int, int] = (0, 0, 255)
    _OVERLAY_ALPHA: float = 0.45

    @classmethod
    def render(cls, bgr_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        overlay = bgr_u8.copy()
        overlay[mask_u8 > 0] = cls._OVERLAY_COLOUR_BGR
        return cv2.addWeighted(
            bgr_u8, 1.0 - cls._OVERLAY_ALPHA,
            overlay, cls._OVERLAY_ALPHA,
            0.0,
        )


# ---------------------------------------------------------------------------
# Mask-only diagnostic view
# ---------------------------------------------------------------------------

class MaskSoloRenderer:
    """Render mask blobs as white pixels on black, with green locator rings.

    The locator rings have radius proportional to ``√area`` so small
    fireflies remain visible at 100% zoom in the GUI.
    """

    _CCA_CONNECTIVITY: int = 8
    _RING_COLOUR_BGR: Tuple[int, int, int] = (0, 255, 0)
    _RING_THICKNESS: int = 2
    _MIN_RING_RADIUS: int = 10
    _RING_RADIUS_MULTIPLIER: float = 4.0

    @classmethod
    def render(cls, mask_u8: np.ndarray) -> np.ndarray:
        h, w = mask_u8.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[mask_u8 > 0] = (255, 255, 255)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8, cls._CCA_CONNECTIVITY
        )
        for i in range(1, num_labels):
            cx = int(centroids[i][0])
            cy = int(centroids[i][1])
            area = stats[i, cv2.CC_STAT_AREA]
            radius = max(
                cls._MIN_RING_RADIUS,
                int(math.sqrt(area) * cls._RING_RADIUS_MULTIPLIER),
            )
            cv2.circle(canvas, (cx, cy), radius,
                       cls._RING_COLOUR_BGR, cls._RING_THICKNESS, cv2.LINE_AA)
        return canvas


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

class SideBySideRenderer:
    """Concatenate ``original | repaired`` with labels and a divider line."""

    _DIVIDER_COLOUR_BGR: Tuple[int, int, int] = (0, 255, 200)
    _DIVIDER_THICKNESS: int = 2
    _LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
    _LABEL_SCALE = 0.7

    @classmethod
    def compose(
        cls,
        original_bgr: np.ndarray,
        repaired_bgr: np.ndarray,
    ) -> np.ndarray:
        left = original_bgr.copy()
        right = repaired_bgr.copy()

        cls._draw_label(left,  "Original", (255, 255, 255), (0, 0, 0))
        cls._draw_label(right, "Repaired", (255, 255, 255), (0, 200, 100))

        combined = np.hstack([left, right])
        mid_x = left.shape[1]
        cv2.line(combined, (mid_x, 0), (mid_x, combined.shape[0]),
                 cls._DIVIDER_COLOUR_BGR, cls._DIVIDER_THICKNESS)
        return combined

    @classmethod
    def _draw_label(
        cls,
        image: np.ndarray,
        text: str,
        outline_bgr: Tuple[int, int, int],
        fill_bgr:    Tuple[int, int, int],
    ) -> None:
        cv2.putText(image, text, (10, 28), cls._LABEL_FONT, cls._LABEL_SCALE,
                    outline_bgr, 2, cv2.LINE_AA)
        cv2.putText(image, text, (10, 28), cls._LABEL_FONT, cls._LABEL_SCALE,
                    fill_bgr, 1, cv2.LINE_AA)
