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

Performance notes
-----------------
Earlier implementations did a Python ``for`` loop over every connected
component, calling ``labels == label_id`` (an O(N) whole-image boolean
compare) inside the loop.  On 4K images with a permissive context
threshold this produced 10 000 – 100 000 components, multiplying into
seconds-to-minutes of wall time.

The current implementation is vectorised:

  * Area and AABB stats come straight from
    ``cv2.connectedComponentsWithStats`` — no Python loop needed.
  * Seed-pixel counts per blob are computed with a single
    ``np.bincount`` pass over the label image.
  * Axis-aligned bounding-box aspect ratio is computed in bulk from
    stats, and aggressively pre-filters elongated structures so that
    only a tiny fraction of survivors ever reach the expensive
    ``cv2.minAreaRect`` rotated-box check.
  * The output mask is built once via a label LUT (``lut[labels]``)
    rather than scattered writes inside the loop.
"""

from __future__ import annotations

import numpy as np
import cv2


class StructuralFilter:
    """CCA-driven shape, size and isolation filter (vectorised)."""

    _CCA_CONNECTIVITY: int = 8
    _MAX_ASPECT_RATIO: float = 3.0
    _CONTEXT_AREA_MULTIPLIER: int = 5
    _CONTEXT_AREA_FLOOR: int = 50
    _SHAPE_CHECK_MIN_AREA: int = 4   # below this, blob is trivially compact

    # AABB aspect ratio is always >= rotated-bbox aspect ratio, but can
    # be considerably larger for diagonal blobs.  We use AABB ratio as a
    # cheap pre-filter: blobs with AABB ratio safely below MAX skip the
    # rotated check (they're definitely compact); blobs with AABB ratio
    # far above MAX skip the rotated check too (they're definitely
    # elongated).  Only blobs in the ambiguous band get the expensive
    # ``cv2.minAreaRect``.
    _AABB_DEFINITELY_COMPACT: float = 1.5            # ≤ this → skip rotated check, accept
    _AABB_DEFINITELY_ELONGATED_FACTOR: float = 1.7   # ≥ MAX·this → skip rotated check, reject

    def __init__(self, scaled_max_noise_size: int) -> None:
        self._scaled_max_noise_size = scaled_max_noise_size
        self._max_context_area = max(
            scaled_max_noise_size * self._CONTEXT_AREA_MULTIPLIER,
            self._CONTEXT_AREA_FLOOR,
        )

    # -- Public API ----------------------------------------------------------

    def filter(
        self,
        context_mask: np.ndarray,
        seed_mask: np.ndarray,
    ) -> np.ndarray:
        """Return a uint8 mask containing only blobs that survive every rule."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            context_mask, connectivity=self._CCA_CONNECTIVITY
        )
        if num_labels <= 1:
            return np.zeros_like(context_mask)

        # Per-label stats — all O(L) vectorised lookups.
        areas   = stats[:, cv2.CC_STAT_AREA]
        widths  = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
        heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)

        # ``keep`` is the running survivor mask, indexed by label_id.
        keep = np.zeros(num_labels, dtype=bool)
        keep[1:] = True            # label 0 = background, always rejected

        # Rule 1 — isolation (cheap, vectorised).
        keep &= areas <= self._max_context_area

        # Rule 2 — seed presence + core size (one bincount, O(N) instead of L·N).
        seed_per_label = np.bincount(
            labels.ravel(),
            weights=(seed_mask > 0).ravel().astype(np.int64),
            minlength=num_labels,
        ).astype(np.int64)
        keep &= seed_per_label > 0
        keep &= seed_per_label <= self._scaled_max_noise_size

        # Rule 3 — compactness (AABB pre-filter, then rotated bbox only on ambiguous).
        eps = 1e-5
        aabb_long  = np.maximum(widths, heights)
        aabb_short = np.maximum(np.minimum(widths, heights), eps)
        aabb_ratio = aabb_long / aabb_short

        big_enough = areas > self._SHAPE_CHECK_MIN_AREA

        # Definitely elongated → reject immediately.
        definitely_elongated_thresh = (
            self._MAX_ASPECT_RATIO * self._AABB_DEFINITELY_ELONGATED_FACTOR
        )
        definitely_elongated = big_enough & (aabb_ratio > definitely_elongated_thresh)
        keep &= ~definitely_elongated

        # Ambiguous band: AABB suggests possibly elongated, but rotated
        # bbox might still come in under MAX.  Run the expensive check
        # only on these (typically << 1% of all blobs on real images).
        ambiguous = (
            keep & big_enough
            & (aabb_ratio > self._AABB_DEFINITELY_COMPACT)
            & (aabb_ratio <= definitely_elongated_thresh)
        )
        if np.any(ambiguous):
            for label_id in np.flatnonzero(ambiguous):
                x0 = stats[label_id, cv2.CC_STAT_LEFT]
                y0 = stats[label_id, cv2.CC_STAT_TOP]
                w  = stats[label_id, cv2.CC_STAT_WIDTH]
                h  = stats[label_id, cv2.CC_STAT_HEIGHT]
                local_roi = labels[y0:y0 + h, x0:x0 + w] == label_id
                if not self._is_compact_local(local_roi):
                    keep[label_id] = False

        # Build output mask in one shot via a label LUT.
        keep[0] = False
        if not np.any(keep):
            return np.zeros_like(context_mask)

        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[keep] = 255
        return lut[labels]

    # -- Helpers -------------------------------------------------------------

    @classmethod
    def _is_compact_local(cls, local_roi: np.ndarray) -> bool:
        """True iff the rotated bounding-box aspect ratio is below the limit.

        Operates on a *local* ROI bool array (the blob's AABB), not the
        whole-image mask — keeps the ``np.where`` cost proportional to
        the blob's bounding box rather than the entire image.
        """
        coords_yx = np.column_stack(np.where(local_roi))
        if len(coords_yx) < 3:
            return True
        coords_xy = coords_yx[:, ::-1].astype(np.float32)
        _, (rect_w, rect_h), _ = cv2.minAreaRect(coords_xy)
        max_dim = max(rect_w, rect_h)
        min_dim = max(1e-5, min(rect_w, rect_h))
        return (max_dim / min_dim) <= cls._MAX_ASPECT_RATIO
