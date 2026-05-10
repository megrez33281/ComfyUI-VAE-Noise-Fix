"""
Statistics HUD overlay.

Renders a translucent panel in the bottom-left of the canvas with the
current detection statistics + the active canvas zoom level.

The HUD is purely a *view* — it does not compute any statistics, it only
formats a ``DetectionStats`` (or legacy dict) for on-screen display.
"""

from __future__ import annotations

from typing import Mapping, Union

import cv2
import numpy as np

from core.statistics import DetectionStats


_StatsLike = Union[DetectionStats, Mapping[str, object]]


class StatisticsHUD:
    """Render a translucent stats panel on a canvas."""

    _FONT          = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE    = 0.5
    _FONT_THICKNESS = 1
    _LINE_HEIGHT   = 22
    _PADDING       = 10
    _PANEL_WIDTH   = 380
    _ALPHA         = 0.65
    _TEXT_BGR      = (0, 255, 200)

    @classmethod
    def draw(
        cls,
        canvas: np.ndarray,
        stats: _StatsLike,
        view_zoom: float,
    ) -> np.ndarray:
        """Render and return the canvas with the stats panel composited."""
        s = cls._to_dict(stats)
        lines = [
            f"Resolution:   {s['resolution']}",
            f"Noise blobs:  {s['noise_blobs']}",
            f"Noise pixels: {s['noise_pixels']}  ({s['coverage_pct']:.4f}%)",
            f"Area range:   {s['area_min']} ~ {s['area_max']}  (avg {s['area_avg']:.1f})",
            f"Detection:    {s['elapsed_ms']:.1f} ms",
            f"Canvas zoom:  {view_zoom:.1f}x",
        ]

        out = canvas.copy()
        panel_h = len(lines) * cls._LINE_HEIGHT + cls._PADDING * 2
        x0, y0 = 10, out.shape[0] - panel_h - 10

        overlay = out.copy()
        cv2.rectangle(overlay,
                      (x0, y0),
                      (x0 + cls._PANEL_WIDTH, y0 + panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, cls._ALPHA, out, 1.0 - cls._ALPHA, 0, out)

        for i, line in enumerate(lines):
            ty = y0 + cls._PADDING + (i + 1) * cls._LINE_HEIGHT - 4
            cv2.putText(out, line, (x0 + cls._PADDING, ty),
                        cls._FONT, cls._FONT_SCALE,
                        cls._TEXT_BGR, cls._FONT_THICKNESS, cv2.LINE_AA)
        return out

    # -- Internal ------------------------------------------------------------

    @staticmethod
    def _to_dict(stats: _StatsLike) -> Mapping[str, object]:
        if isinstance(stats, DetectionStats):
            return stats.as_dict()
        return stats
