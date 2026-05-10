"""
Mouse-following magnification inset (a.k.a. "zoom lens").

Stateful — tracks current magnification, lens size, mouse position, and
on/off toggle.  Renders an inset patch in the upper-right corner with a
crosshair on the source pixel.
"""

from __future__ import annotations

import cv2
import numpy as np


class ZoomLens:
    """Inset zoom box following the mouse cursor."""

    _MIN_MAG: int = 2
    _MAX_MAG: int = 16

    _OUTLINE_BGR = (0, 255, 200)

    def __init__(self, mag: int = 6, lens_size: int = 200) -> None:
        self._mag = mag
        self._lens_size = lens_size
        self._enabled = False
        self._mx = 0
        self._my = 0

    # -- State ---------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self) -> None:
        self._enabled = not self._enabled

    def update_position(self, x: int, y: int) -> None:
        self._mx = x
        self._my = y

    def adjust_magnification(self, delta: int) -> None:
        self._mag = max(self._MIN_MAG, min(self._mag + delta, self._MAX_MAG))

    # -- Render --------------------------------------------------------------

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Draw the lens on ``canvas`` (returns a new image)."""
        if not self._enabled:
            return canvas

        out = canvas.copy()
        h, w = out.shape[:2]
        half_src = self._lens_size // (2 * self._mag)

        sx0 = max(0, self._mx - half_src)
        sy0 = max(0, self._my - half_src)
        sx1 = min(w, self._mx + half_src)
        sy1 = min(h, self._my + half_src)
        if sx1 - sx0 < 2 or sy1 - sy0 < 2:
            return out

        crop = out[sy0:sy1, sx0:sx1]
        zoomed = cv2.resize(crop, (self._lens_size, self._lens_size),
                            interpolation=cv2.INTER_NEAREST)

        ix0 = w - self._lens_size - 12
        iy0 = 12

        cv2.rectangle(out, (ix0 - 2, iy0 - 2),
                      (ix0 + self._lens_size + 2, iy0 + self._lens_size + 2),
                      self._OUTLINE_BGR, 2)
        out[iy0:iy0 + self._lens_size, ix0:ix0 + self._lens_size] = zoomed

        cv2.putText(out, f"{self._mag}x",
                    (ix0 + 6, iy0 + self._lens_size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._OUTLINE_BGR, 1, cv2.LINE_AA)
        cv2.drawMarker(out, (self._mx, self._my), self._OUTLINE_BGR,
                       cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
        return out
