"""
Brush-based mask edit overlay.

The detection pipeline's output mask is read-only — when the user wants
to manually add a missed noise pixel or remove a false positive, this
module records those edits as two binary delta layers (``add`` /
``erase``) and overlays them on top of the algorithm's mask at display
time.

The original image is never touched: only the mask layer is modified.
Conceptually:

    effective_mask = (algorithm_mask | add_layer) & ~erase_layer

The class is intentionally tiny and stateful — it owns nothing besides
the two delta buffers, a brush radius, and an on/off flag.  All
rendering happens elsewhere; this module only provides the bookkeeping
and the ``apply()`` helper.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class BrushEditor:
    """Track user-painted add/erase deltas applied on top of an algorithm mask.

    Mode semantics:
        * ``'add'``   — paint mask pixels (catch a noise spot the algorithm missed).
        * ``'erase'`` — remove mask pixels (rescue a false positive).

    Painting one layer automatically clears the corresponding region on
    the opposite layer, so erase-then-add and add-then-erase behave
    intuitively without leaving stale deltas behind.
    """

    _MIN_RADIUS: int = 1
    _MAX_RADIUS: int = 100

    _CURSOR_COLOR_BGR: Tuple[int, int, int] = (0, 255, 255)
    _CURSOR_OUTLINE_BGR: Tuple[int, int, int] = (0, 0, 0)

    def __init__(self, initial_radius: int = 8) -> None:
        self._add: Optional[np.ndarray] = None
        self._erase: Optional[np.ndarray] = None
        self._radius = initial_radius
        self._enabled = False
        self._last_pos: Optional[Tuple[int, int]] = None

    # -- State ---------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def radius(self) -> int:
        return self._radius

    def toggle(self) -> None:
        self._enabled = not self._enabled
        self._last_pos = None

    def adjust_radius(self, delta: int) -> None:
        self._radius = max(self._MIN_RADIUS, min(self._radius + delta, self._MAX_RADIUS))

    def attach_to(self, shape: Tuple[int, int]) -> None:
        """Allocate (or resize) the delta layers for an image of shape ``(H, W)``.

        Called whenever a new image is loaded.  Existing edits are
        preserved if the shape is unchanged; otherwise they are cleared
        because they no longer correspond to anything meaningful.
        """
        h, w = shape[:2]
        if self._add is None or self._add.shape != (h, w):
            self._add = np.zeros((h, w), dtype=np.uint8)
            self._erase = np.zeros((h, w), dtype=np.uint8)
            self._last_pos = None

    def has_edits(self) -> bool:
        if self._add is None:
            return False
        return bool(self._add.any() or self._erase.any())

    def clear(self) -> None:
        if self._add is not None:
            self._add[:] = 0
        if self._erase is not None:
            self._erase[:] = 0
        self._last_pos = None

    # -- Painting ------------------------------------------------------------

    def begin_stroke(self) -> None:
        """Mouse-button-down event — reset stroke state."""
        self._last_pos = None

    def end_stroke(self) -> None:
        """Mouse-button-up event — terminate the current stroke."""
        self._last_pos = None

    def stroke(self, x: int, y: int, mode: str) -> None:
        """Paint a circular dab at ``(x, y)`` on the selected layer.

        If a previous position was recorded by an earlier call within the
        same drag, a thick line is also painted between the two so that
        fast mouse drags don't leave gaps.
        """
        if self._add is None or self._erase is None:
            return

        if mode == "add":
            target, opposite = self._add, self._erase
        elif mode == "erase":
            target, opposite = self._erase, self._add
        else:
            return

        h, w = target.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        # Bridge the gap from previous position for smooth strokes.
        if self._last_pos is not None:
            px, py = self._last_pos
            cv2.line(target, (px, py), (x, y),
                     255, self._radius * 2, cv2.LINE_AA)
            cv2.line(opposite, (px, py), (x, y),
                     0, self._radius * 2, cv2.LINE_AA)

        cv2.circle(target, (x, y), self._radius, 255, -1, cv2.LINE_AA)
        cv2.circle(opposite, (x, y), self._radius, 0, -1, cv2.LINE_AA)

        self._last_pos = (x, y)

    # -- Mask composition ----------------------------------------------------

    def apply(self, algo_mask_u8: np.ndarray) -> np.ndarray:
        """Return ``algo_mask`` combined with the user's add/erase deltas."""
        if self._add is None or not self.has_edits():
            return algo_mask_u8
        out = algo_mask_u8.copy()
        out[self._add > 0] = 255
        out[self._erase > 0] = 0
        return out

    # -- Cursor rendering ----------------------------------------------------

    def draw_cursor(self, canvas: np.ndarray, x: int, y: int) -> np.ndarray:
        """Draw a brush-radius outline on ``canvas`` at ``(x, y)``.

        Returns a new image so callers can chain renders without
        mutating their cache.
        """
        if not self._enabled:
            return canvas
        out = canvas.copy()
        # Thin black halo for visibility on bright backgrounds.
        cv2.circle(out, (x, y), self._radius + 1,
                   self._CURSOR_OUTLINE_BGR, 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), self._radius,
                   self._CURSOR_COLOR_BGR, 1, cv2.LINE_AA)
        return out
