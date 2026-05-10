"""
Detection statistics calculator.

Pure data-aggregation utility.  Used by the GUI HUD and re-exported via
the pipeline DTO so the ComfyUI node can write the same numbers to its
console / log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionStats:
    resolution:   str
    noise_blobs:  int
    noise_pixels: int
    coverage_pct: float
    area_min:     int
    area_max:     int
    area_avg:     float
    elapsed_ms:   float

    def as_dict(self) -> dict:
        """Backward-compat dict interface for legacy consumers."""
        return {
            "resolution":   self.resolution,
            "noise_blobs":  self.noise_blobs,
            "noise_pixels": self.noise_pixels,
            "coverage_pct": self.coverage_pct,
            "area_min":     self.area_min,
            "area_max":     self.area_max,
            "area_avg":     self.area_avg,
            "elapsed_ms":   self.elapsed_ms,
        }


class NoiseStatistics:
    """Compute aggregate stats from a binary noise mask."""

    _CCA_CONNECTIVITY: int = 8

    @classmethod
    def compute(
        cls,
        mask_u8: np.ndarray,
        image_shape: Tuple[int, int],
        elapsed_ms: float,
    ) -> DetectionStats:
        h, w = image_shape
        total_pixels = h * w

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, cls._CCA_CONNECTIVITY
        )
        noise_count = num_labels - 1
        noise_pixels = int(cv2.countNonZero(mask_u8))
        coverage_pct = (noise_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        areas: List[int] = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]

        return DetectionStats(
            resolution=f"{w} x {h}",
            noise_blobs=noise_count,
            noise_pixels=noise_pixels,
            coverage_pct=coverage_pct,
            area_min=min(areas) if areas else 0,
            area_max=max(areas) if areas else 0,
            area_avg=sum(areas) / len(areas) if areas else 0.0,
            elapsed_ms=elapsed_ms,
        )
