"""
Structural / geometric filtering.

Connected-component analysis of the *context* binary mask, with three
rejection rules driven by the *seed* mask and a maximum-noise-size budget:

    1. Isolation:    ``context_area > 5 × max_noise_size`` ⇒ part of a
                     larger structure, not an isolated artifact.
    2. Seed presence + core size:   the context blob must contain at least
                                    one high-energy seed pixel, and the
                                    seed core itself must not exceed the
                                    user budget.
    3. Compactness:  rotated bounding-box aspect ratio > 3:1 ⇒ probably a
                     hair, line, fibre, etc., not a firefly.

The original spec only required area thresholding; the aspect-ratio
guard is an evolution to defend against false-positives on diagonal hair
or fine line work where an axis-aligned box would underestimate
elongation.
"""

from __future__ import annotations

import numpy as np
import cv2


class StructuralFilter:
    """CCA-driven shape, size and isolation filter."""

    _CCA_CONNECTIVITY: int = 8
    _MAX_ASPECT_RATIO: float = 3.0
    _CONTEXT_AREA_MULTIPLIER: int = 5
    _CONTEXT_AREA_FLOOR: int = 50
    _SHAPE_CHECK_MIN_AREA: int = 4   # below this, blob is trivially compact

    def __init__(self, scaled_max_noise_size: int) -> None:
        self._scaled_max_noise_size = scaled_max_noise_size
        self._max_context_area = max(
            scaled_max_noise_size * self._CONTEXT_AREA_MULTIPLIER,
            self._CONTEXT_AREA_FLOOR,
        )

    def filter(
        self,
        context_mask: np.ndarray,
        seed_mask: np.ndarray,
    ) -> np.ndarray:
        """Return a uint8 mask containing only blobs that survive every rule."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            context_mask, connectivity=self._CCA_CONNECTIVITY
        )
        output = np.zeros_like(context_mask)

        for label_id in range(1, num_labels):
            context_area = stats[label_id, cv2.CC_STAT_AREA]

            # Rule 1 — isolation
            if context_area > self._max_context_area:
                continue

            component_roi = (labels == label_id)
            seed_pixels = seed_mask[component_roi]
            seed_area = int(np.count_nonzero(seed_pixels))

            # Rule 2 — must contain a seed core, and the core must fit
            if seed_area == 0 or seed_area > self._scaled_max_noise_size:
                continue

            # Rule 3 — compactness via rotated bounding box
            if context_area > self._SHAPE_CHECK_MIN_AREA:
                if not self._is_compact(component_roi):
                    continue

            output[component_roi] = 255

        return output

    # -- Helpers -------------------------------------------------------------

    @classmethod
    def _is_compact(cls, component_roi: np.ndarray) -> bool:
        """True iff the rotated bounding-box aspect ratio is below the limit."""
        coords_yx = np.column_stack(np.where(component_roi))
        coords_xy = coords_yx[:, ::-1].astype(np.float32)
        _, (rect_w, rect_h), _ = cv2.minAreaRect(coords_xy)

        max_dim = max(rect_w, rect_h)
        min_dim = max(1e-5, min(rect_w, rect_h))
        return (max_dim / min_dim) <= cls._MAX_ASPECT_RATIO
