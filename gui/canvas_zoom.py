"""
Whole-canvas zoom & pan state machine.

State stored in *normalised* image coordinates so that switching between
a 1024² image and a 4096² image preserves the user's zoom focus.
"""

from __future__ import annotations

import cv2
import numpy as np


class CanvasZoom:
    """Zoom level + pan centre, plus a method that crops a frame to match."""

    _MIN_ZOOM: float = 1.0
    _MAX_ZOOM: float = 16.0
    _ZOOM_FACTOR: float = 1.25
    _NEAREST_THRESHOLD: float = 4.0  # at and above this zoom, use nearest-neighbour

    def __init__(self) -> None:
        self._zoom = 1.0
        self._cx = 0.5  # pan centre x in [0, 1]
        self._cy = 0.5

    # -- Read-only props -----------------------------------------------------

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def is_zoomed(self) -> bool:
        return self._zoom > 1.01

    @property
    def center(self) -> tuple[float, float]:
        """Current pan centre in normalised image coords ``(cx, cy) ∈ [0, 1]``."""
        return self._cx, self._cy

    # -- Mutators ------------------------------------------------------------

    def reset(self) -> None:
        self._zoom = 1.0
        self._cx = 0.5
        self._cy = 0.5

    def adjust(self, delta: int, mouse_nx: float, mouse_ny: float) -> None:
        """Zoom in/out, anchoring the point under the mouse.

        Args:
            delta:    +1 to zoom in, -1 to zoom out.
            mouse_nx: mouse x in normalised [0, 1] window space.
            mouse_ny: mouse y in normalised [0, 1] window space.
        """
        old_zoom = self._zoom
        factor = self._ZOOM_FACTOR if delta > 0 else 1.0 / self._ZOOM_FACTOR
        new_zoom = max(self._MIN_ZOOM, min(self._zoom * factor, self._MAX_ZOOM))
        if abs(new_zoom - old_zoom) < 0.001:
            return

        half_w_old = 0.5 / old_zoom
        half_h_old = 0.5 / old_zoom
        src_x = self._cx - half_w_old + mouse_nx / old_zoom
        src_y = self._cy - half_h_old + mouse_ny / old_zoom

        half_w_new = 0.5 / new_zoom
        half_h_new = 0.5 / new_zoom
        new_cx = src_x - mouse_nx / new_zoom + half_w_new
        new_cy = src_y - mouse_ny / new_zoom + half_h_new

        self._zoom = new_zoom
        self._cx = max(half_w_new, min(new_cx, 1.0 - half_w_new))
        self._cy = max(half_h_new, min(new_cy, 1.0 - half_h_new))

    # -- Apply ---------------------------------------------------------------

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Return a zoomed/cropped copy of ``img`` at the current state."""
        if not self.is_zoomed:
            return img

        h, w = img.shape[:2]
        vis_w = w / self._zoom
        vis_h = h / self._zoom

        x0 = int(self._cx * w - vis_w / 2)
        y0 = int(self._cy * h - vis_h / 2)

        x0 = max(0, min(x0, w - int(vis_w)))
        y0 = max(0, min(y0, h - int(vis_h)))
        x1 = x0 + int(vis_w)
        y1 = y0 + int(vis_h)

        crop = img[y0:y1, x0:x1]
        interp = (cv2.INTER_NEAREST
                  if self._zoom >= self._NEAREST_THRESHOLD
                  else cv2.INTER_LINEAR)
        return cv2.resize(crop, (w, h), interpolation=interp)
