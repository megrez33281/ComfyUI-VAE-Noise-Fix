"""
Binary mask generators.

Threshold the energy maps into binary candidate masks. Kept separate from
the energy extractors so different threshold strategies (fixed, Otsu,
adaptive) can be substituted without touching the energy stage.
"""

from __future__ import annotations

import cv2
import numpy as np


class DualPathMaskGenerator:
    """Combine Laplacian and median-residual binary masks with a logical OR.

    Each path uses a sensitivity-driven threshold:

        Laplacian path:  T_lap = ⌊sensitivity · 255⌋
        Median   path:   T_med = ⌊20 + sensitivity · 80⌋
                                                ↑ floor to suppress sensor floor

    The dual-path OR catches both gradient edges (Laplacian) and impulse
    spikes that survive median suppression (residual), which is the main
    failure mode of single-path detectors.
    """

    @staticmethod
    def generate(
        laplacian_energy: np.ndarray,
        median_residual: np.ndarray,
        sensitivity: float,
    ) -> np.ndarray:
        """Return a uint8 ``{0, 255}`` mask combining both paths."""
        t_lap = int(sensitivity * 255.0)
        _, b_lap = cv2.threshold(laplacian_energy, t_lap, 255, cv2.THRESH_BINARY)

        t_med = int(20 + sensitivity * 80.0)
        _, b_med = cv2.threshold(median_residual, t_med, 255, cv2.THRESH_BINARY)

        return cv2.bitwise_or(b_lap, b_med)
