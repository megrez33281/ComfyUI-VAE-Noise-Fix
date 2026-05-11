"""
CIE-LAB chromaticity & cliff-steepness verification.

Implements the third step of the original architecture spec — neighbourhood
colour-variance discrimination — but generalised:

  * The original spec checked Euclidean colour distance only. That misses
    grey/white cliff fireflies (no chromatic aberration but huge L* jump).
  * The original spec described qualitative “step vs cliff” behaviour. We
    formalise it as a *relative-contrast steepness* score:

        internal_drop_ratio = (peak_L  − mean_comp_L) / (peak_L − mean_bg_L)

    A value near 1.0 means the blob has a flat top sitting on a cliff
    (impulse noise). A value near 0.5 means the blob gradually decays into
    the background (soft star, specular highlight, bokeh).

  * A blob is accepted if **either** the cliff-steepness exceeds the
    sensitivity-tuned threshold, **or** the chromatic distance in the a*b*
    plane exceeds a colour-bleed threshold.  This catches both achromatic
    impulse spikes and pure-colour fireflies (purple/green VAE bleed).

Performance notes
-----------------
Per-blob *component* statistics (mean L, peak L, mean a, mean b) are
computed in bulk via ``np.bincount`` / ``np.maximum.at`` — one pass over
the image, regardless of blob count.

Per-blob *background ring* statistics still require a Python loop
because the ring window is position-dependent, but only blobs that
survived the upstream structural filter reach this stage, so the loop
typically runs over a handful of candidates.

Final output mask is built once with a label LUT
(``lut[labels]``) instead of one whole-image compare per accepted blob.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class ChromaticFilter:
    """LAB-space verification: rejects natural highlights, keeps fireflies."""

    _CCA_CONNECTIVITY: int = 8
    _MIN_LUMA_DROP: float = 5.0   # below this, blob is indistinguishable from BG

    def __init__(
        self,
        gradient_sensitivity: float,
        scaled_max_noise_size: int,
    ) -> None:
        self._gradient_sensitivity = gradient_sensitivity
        self._ring_width = max(3, int(math.sqrt(max(1, scaled_max_noise_size))) + 2)

        # Cliff steepness threshold: more sensitive runs allow gentler cliffs.
        self._cliff_threshold = 0.6 - (gradient_sensitivity * 0.2)

        # Chromaticity distance threshold (a*b* plane).
        self._chroma_threshold = 3.0 + gradient_sensitivity * 15.0

    # -- Public API ----------------------------------------------------------

    def filter(
        self,
        candidate_mask: np.ndarray,
        bgr_u8: np.ndarray,
    ) -> np.ndarray:
        """Return mask after LAB-space verification."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_mask, connectivity=self._CCA_CONNECTIVITY
        )
        if num_labels <= 1:
            return np.zeros_like(candidate_mask)

        h, w = candidate_mask.shape[:2]
        lab_f32 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab_f32[..., 0]
        A = lab_f32[..., 1]
        B = lab_f32[..., 2]

        # --- Vectorised per-blob component statistics ----------------------
        # Mean L, A, B per label via bincount (single pass over the image).
        labels_flat = labels.ravel()
        counts = np.bincount(labels_flat, minlength=num_labels).astype(np.float64)
        # Guard against zero division on labels with no pixels (shouldn't happen
        # for valid CCA output but cheap insurance).
        counts_safe = np.where(counts > 0, counts, 1.0)

        sum_L = np.bincount(labels_flat, weights=L.ravel(), minlength=num_labels)
        sum_A = np.bincount(labels_flat, weights=A.ravel(), minlength=num_labels)
        sum_B = np.bincount(labels_flat, weights=B.ravel(), minlength=num_labels)
        mean_L = sum_L / counts_safe
        mean_A = sum_A / counts_safe
        mean_B = sum_B / counts_safe

        # Peak L per label via np.maximum.at (also a single pass).
        peak_L = np.full(num_labels, -np.inf, dtype=np.float32)
        np.maximum.at(peak_L, labels_flat, L.ravel())

        # --- Per-blob background ring loop ---------------------------------
        # Only blobs that survived the structural filter reach this point,
        # so this loop is typically short even at 4K.
        keep = np.zeros(num_labels, dtype=bool)

        for label_id in range(1, num_labels):
            x0 = stats[label_id, cv2.CC_STAT_LEFT]
            y0 = stats[label_id, cv2.CC_STAT_TOP]
            bw = stats[label_id, cv2.CC_STAT_WIDTH]
            bh = stats[label_id, cv2.CC_STAT_HEIGHT]

            rx0 = max(0, x0 - self._ring_width)
            ry0 = max(0, y0 - self._ring_width)
            rx1 = min(w, x0 + bw + self._ring_width)
            ry1 = min(h, y0 + bh + self._ring_width)

            roi_labels = labels[ry0:ry1, rx0:rx1]
            bg_mask = roi_labels != label_id
            if not np.any(bg_mask):
                continue

            roi_lab = lab_f32[ry0:ry1, rx0:rx1]
            mean_bg_L = float(roi_lab[..., 0][bg_mask].mean())

            total_drop = float(peak_L[label_id]) - mean_bg_L
            if total_drop <= self._MIN_LUMA_DROP:
                continue

            internal_drop_ratio = (
                (float(peak_L[label_id]) - float(mean_L[label_id])) / total_drop
            )
            is_cliff = internal_drop_ratio > self._cliff_threshold

            mean_bg_A = float(roi_lab[..., 1][bg_mask].mean())
            mean_bg_B = float(roi_lab[..., 2][bg_mask].mean())
            chroma_dist = math.hypot(
                float(mean_A[label_id]) - mean_bg_A,
                float(mean_B[label_id]) - mean_bg_B,
            )
            is_chromatic_noise = chroma_dist >= self._chroma_threshold

            if is_cliff or is_chromatic_noise:
                keep[label_id] = True

        # --- Build output mask via label LUT (avoids per-blob full-image cmp).
        if not np.any(keep):
            return np.zeros_like(candidate_mask)
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[keep] = 255
        return lut[labels]
