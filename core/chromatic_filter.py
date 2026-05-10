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

    def filter(
        self,
        candidate_mask: np.ndarray,
        bgr_u8: np.ndarray,
    ) -> np.ndarray:
        """Return mask after LAB-space verification."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_mask, connectivity=self._CCA_CONNECTIVITY
        )
        verified_mask = np.zeros_like(candidate_mask)

        h, w = candidate_mask.shape[:2]
        lab_f32 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

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
            roi_lab = lab_f32[ry0:ry1, rx0:rx1]

            component_pixels = roi_labels == label_id
            neighbourhood_pixels = ~component_pixels
            if not np.any(neighbourhood_pixels):
                continue

            if self._passes_lab_check(roi_lab, component_pixels, neighbourhood_pixels):
                verified_mask[labels == label_id] = 255

        return verified_mask

    # -- Helpers -------------------------------------------------------------

    def _passes_lab_check(
        self,
        roi_lab: np.ndarray,
        component_pixels: np.ndarray,
        neighbourhood_pixels: np.ndarray,
    ) -> bool:
        """Return True iff the blob is either a luminance cliff or chromatic bleed."""
        comp_luma = roi_lab[component_pixels, 0]
        peak_luma = float(np.max(comp_luma))
        mean_comp_luma = float(np.mean(comp_luma))

        mean_bg_lab = roi_lab[neighbourhood_pixels].mean(axis=0)
        mean_bg_luma = float(mean_bg_lab[0])

        total_drop = peak_luma - mean_bg_luma
        if total_drop <= self._MIN_LUMA_DROP:
            return False

        internal_drop_ratio = (peak_luma - mean_comp_luma) / total_drop
        is_cliff = internal_drop_ratio > self._cliff_threshold

        mean_comp_chroma = roi_lab[component_pixels, 1:].mean(axis=0)
        chroma_dist = float(np.linalg.norm(mean_comp_chroma - mean_bg_lab[1:]))
        is_chromatic_noise = chroma_dist >= self._chroma_threshold

        return is_cliff or is_chromatic_noise
