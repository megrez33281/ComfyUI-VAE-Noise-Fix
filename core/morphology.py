"""
Morphological mask post-processing.

Telea inpainting samples *outside* the masked region for source data.  If
the mask is one pixel too small, the inpainter samples partially-corrupt
boundary pixels and re-injects them as “clean” paint, producing visible
halos around fireflies.

This module dilates the verified mask with an isotropic elliptical
structuring element so the inpaint boundary is firmly inside the clean
neighbourhood.
"""

from __future__ import annotations

import cv2
import numpy as np


class MaskDilator:
    """Isotropic mask dilation with a circular structuring element."""

    def __init__(self, dilate_radius: int) -> None:
        self._radius = max(0, int(dilate_radius))

    @property
    def radius(self) -> int:
        return self._radius

    def dilate(self, mask_u8: np.ndarray) -> np.ndarray:
        """Return dilated mask, or the input unchanged if radius == 0."""
        if self._radius <= 0:
            return mask_u8
        diameter = 2 * self._radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        return cv2.dilate(mask_u8, kernel, iterations=1)
