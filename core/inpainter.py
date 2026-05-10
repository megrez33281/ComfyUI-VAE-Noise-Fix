"""
Telea Fast-Marching inpainter.

Wraps ``cv2.inpaint(..., INPAINT_TELEA)`` with a derived inpaint radius.

Per the original spec, Telea is preferred over PatchMatch for tiny VAE
fireflies (1–5 px) because:

  * Reconstruction time is linear in masked-area, dominated by the FMM
    front rather than randomised search.
  * The radius can be set very small (2–7 px) without artefacting,
    keeping the inpaint local and avoiding far-region texture pollution.

The inpaint radius is derived from ``max_noise_size``, clamped to
``[2, 7]``.  Below 2 the FMM front becomes degenerate; above 7 the
inpaint blurs detail without measurable quality gain on isolated VAE
spikes.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class TeleaInpainter:
    """Telea (FMM) inpainter with auto-derived radius."""

    _MIN_RADIUS: int = 2
    _MAX_RADIUS: int = 7

    @classmethod
    def inpaint(
        cls,
        bgr_u8: np.ndarray,
        mask_u8: np.ndarray,
        max_noise_size: int,
    ) -> np.ndarray:
        """Apply Telea inpainting wherever ``mask_u8`` is non-zero."""
        radius = cls._compute_radius(max_noise_size)
        return cv2.inpaint(bgr_u8, mask_u8, radius, cv2.INPAINT_TELEA)

    @classmethod
    def _compute_radius(cls, max_noise_size: int) -> int:
        r = int(math.ceil(math.sqrt(max(1, max_noise_size))))
        return max(cls._MIN_RADIUS, min(r, cls._MAX_RADIUS))
