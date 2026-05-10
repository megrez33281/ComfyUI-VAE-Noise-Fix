"""
Energy-map extractors.

The original architecture spec calls for a Laplacian/Sobel high-frequency
filter as Step 1 of the detection pipeline. Implementation evolved into a
*dual-path* design: the Laplacian captures sharp gradient discontinuities
while a median-residual path captures impulse spikes that are too small or
too smooth to register on the Laplacian. Both extractors live here as
single-responsibility classes so they can be swapped, unit-tested, or
chained independently.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class LaplacianEnergyExtractor:
    """Extract 2nd-order gradient energy via the Laplacian operator.

    The kernel size is fixed at 3 — this is the smallest non-degenerate
    Laplacian and gives the cleanest single-pixel response to VAE
    fireflies, which is the dominant noise topology we target.
    """

    _DDEPTH: int = cv2.CV_16S       # signed 16-bit avoids uint8 overflow
    _KSIZE: int = 3

    def __call__(self, gray_u8: np.ndarray) -> np.ndarray:
        return self.extract(gray_u8)

    def extract(self, gray_u8: np.ndarray) -> np.ndarray:
        """Return absolute Laplacian magnitude as a uint8 energy map."""
        lap_16s = cv2.Laplacian(gray_u8, self._DDEPTH, ksize=self._KSIZE)
        return cv2.convertScaleAbs(lap_16s)


class MedianResidualExtractor:
    """Extract impulse-noise residual: ``|src − median(src)|``.

    The median kernel must be physically larger than the largest blob we
    expect to erase. We size it dynamically off the (resolution-scaled)
    ``max_noise_size`` so the residual remains responsive whether the user
    is targeting tiny 1-pixel pops or chunkier 50-pixel splotches.
    """

    _MIN_KSIZE: int = 5

    def __init__(self, scaled_max_noise_size: int) -> None:
        self._ksize = self._compute_kernel_size(scaled_max_noise_size)

    @property
    def kernel_size(self) -> int:
        return self._ksize

    def __call__(self, bgr_u8: np.ndarray) -> np.ndarray:
        return self.extract(bgr_u8)

    def extract(self, bgr_u8: np.ndarray) -> np.ndarray:
        """Return per-pixel max-channel absolute residual."""
        median = cv2.medianBlur(bgr_u8, self._ksize)
        diff = cv2.absdiff(bgr_u8, median)
        return np.max(diff, axis=2)

    # -- Helpers -------------------------------------------------------------

    @classmethod
    def _compute_kernel_size(cls, scaled_max_noise_size: int) -> int:
        """Diameter ≈ ⌈√area⌉, forced odd and ≥ ``_MIN_KSIZE``."""
        diameter = int(math.ceil(math.sqrt(max(1, scaled_max_noise_size))))
        return max(cls._MIN_KSIZE, (diameter + 2) | 1)
