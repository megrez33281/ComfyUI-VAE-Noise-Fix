"""
Interactive parameter-tuning GUI.

Built on top of OpenCV HighGUI.  The actual algorithms live in ``core``;
this file is purely *view* + *controller*.

The view-mode list mirrors ``core.PreviewMode`` exactly so the GUI and
the ComfyUI node always offer the same options.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core import NoiseFixPipeline, PreviewMode
from core.overlay import DebugOverlayRenderer
from core.pipeline import DetectionResult

from .brush_editor import BrushEditor
from .canvas_zoom import CanvasZoom
from .image_io import imread_safe, imwrite_safe
from .statistics_hud import StatisticsHUD
from .zoom_lens import ZoomLens


# ---------------------------------------------------------------------------
# View-mode ↔ keyboard binding
# ---------------------------------------------------------------------------

# The GUI keeps the same ordering it always had: 1..5 = main views,
# 6..0,- = intermediate stages.  The list mirrors PreviewMode declaration
# order, so .value is the human label.
_VIEW_MODES: List[PreviewMode] = list(PreviewMode)

_VIEW_KEYS: List[int] = [
    ord("1"), ord("2"), ord("3"), ord("4"), ord("5"),
    ord("6"), ord("7"), ord("8"), ord("9"), ord("0"), ord("-"),
]

_SAVE_SUFFIXES: List[str] = [
    "_original",
    "_mask_overlay",
    "_mask_only",
    "_fixed",
    "_compare",
    "_laplacian",
    "_median_residual",
    "_seed",
    "_context",
    "_filtered",
    "_final",
]


# ---------------------------------------------------------------------------
# PreviewApp
# ---------------------------------------------------------------------------

class PreviewApp:
    """Interactive preview / parameter tuning application."""

    _WINDOW = "VAE Noise Fix - Preview"

    def __init__(self, image_paths: List[str]) -> None:
        self._paths = image_paths
        self._idx = 0

        # Parameters (trackbar values are ints; sensitivity is ×100).
        self._sensitivity_i = 35
        self._max_noise_size = 6
        self._mask_dilate = 2

        # View state.
        self._view_mode_idx = 1  # MASK_OVERLAY by default
        self._zoom_lens = ZoomLens()
        self._canvas_zoom = CanvasZoom()
        self._brush = BrushEditor()
        self._mouse_x = 0
        self._mouse_y = 0
        self._cursor_canvas_x = 0
        self._cursor_canvas_y = 0

        # Cached pipeline result.
        self._bgr: Optional[np.ndarray] = None
        self._result: Optional[DetectionResult] = None
        self._dirty = True

        # Last composed frame (before zoom lens) — for save.
        self._last_canvas: Optional[np.ndarray] = None

        # Letterbox mapping.
        self._lb_x_off = 0
        self._lb_y_off = 0
        self._lb_scale = 1.0

    # -- Lifecycle -----------------------------------------------------------

    def run(self) -> None:
        cv2.namedWindow(self._WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._WINDOW, 1280, 800)

        cv2.createTrackbar("Sensitivity (x100)", self._WINDOW,
                           self._sensitivity_i, 100, self._on_sensitivity)
        cv2.createTrackbar("Max Noise Size", self._WINDOW,
                           self._max_noise_size, 100, self._on_max_size)
        cv2.createTrackbar("Mask Dilate", self._WINDOW,
                           self._mask_dilate, 10, self._on_mask_dilate)

        cv2.setMouseCallback(self._WINDOW, self._on_mouse)

        self._load_current_image()

        while True:
            if self._dirty:
                self._process()
                self._dirty = False

            display = self._compose_display()
            cv2.imshow(self._WINDOW, display)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in _VIEW_KEYS:
                self._view_mode_idx = _VIEW_KEYS.index(key)
            elif key == ord("z"):
                self._zoom_lens.toggle()
            elif key == ord("s"):
                self._save_current_view()
            elif key == ord("a"):
                self._switch_image(-1)
            elif key == ord("d"):
                self._switch_image(1)
            elif key == ord("r"):
                self._canvas_zoom.reset()
            elif key == ord("e"):
                # Toggle brush edit mode; force Mask Overlay view when ON.
                self._brush.toggle()
                if self._brush.enabled:
                    self._view_mode_idx = _VIEW_MODES.index(PreviewMode.MASK_OVERLAY)
            elif key == ord("["):
                self._brush.adjust_radius(-2)
            elif key == ord("]"):
                self._brush.adjust_radius(+2)
            elif key == ord("c"):
                # Wipe all brush edits (back to algorithm-only mask).
                self._brush.clear()

        cv2.destroyAllWindows()

    # -- Trackbar callbacks --------------------------------------------------

    def _on_sensitivity(self, val: int) -> None:
        self._sensitivity_i = max(1, val)
        self._dirty = True

    def _on_max_size(self, val: int) -> None:
        self._max_noise_size = max(1, val)
        self._dirty = True

    def _on_mask_dilate(self, val: int) -> None:
        self._mask_dilate = val
        self._dirty = True

    # -- Mouse callback ------------------------------------------------------

    def _window_to_canvas(self, wx: int, wy: int) -> Tuple[int, int]:
        cx = int((wx - self._lb_x_off) / self._lb_scale) if self._lb_scale > 0 else wx
        cy = int((wy - self._lb_y_off) / self._lb_scale) if self._lb_scale > 0 else wy
        return cx, cy

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        self._mouse_x = x
        self._mouse_y = y
        cx, cy = self._window_to_canvas(x, y)
        self._cursor_canvas_x = cx
        self._cursor_canvas_y = cy

        # ---- Brush edit interception (must come first so paint works
        #      regardless of zoom-lens / scroll bindings) ----------------
        if self._brush.enabled:
            img_xy = self._canvas_to_image(cx, cy)
            if event == cv2.EVENT_LBUTTONDOWN:
                self._brush.begin_stroke()
                if img_xy is not None:
                    self._brush.stroke(*img_xy, mode="add")
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._brush.begin_stroke()
                if img_xy is not None:
                    self._brush.stroke(*img_xy, mode="erase")
            elif event == cv2.EVENT_MOUSEMOVE:
                if flags & cv2.EVENT_FLAG_LBUTTON and img_xy is not None:
                    self._brush.stroke(*img_xy, mode="add")
                elif flags & cv2.EVENT_FLAG_RBUTTON and img_xy is not None:
                    self._brush.stroke(*img_xy, mode="erase")
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                self._brush.end_stroke()
            # Scroll still adjusts brush radius in edit mode.
            elif event == cv2.EVENT_MOUSEWHEEL:
                self._brush.adjust_radius(+2 if flags > 0 else -2)
            return

        # ---- Non-edit-mode behaviour (unchanged) -----------------------
        if event == cv2.EVENT_MOUSEMOVE:
            self._zoom_lens.update_position(cx, cy)

        elif event == cv2.EVENT_MOUSEWHEEL:
            ctrl_held = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0
            if ctrl_held:
                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect(self._WINDOW)
                except Exception:
                    win_w, win_h = 1280, 800

                img_w = win_w - 2 * self._lb_x_off
                img_h = win_h - 2 * self._lb_y_off

                if img_w > 0 and img_h > 0:
                    nx = max(0.0, min((x - self._lb_x_off) / img_w, 1.0))
                    ny = max(0.0, min((y - self._lb_y_off) / img_h, 1.0))
                else:
                    nx, ny = 0.5, 0.5

                delta = 1 if flags > 0 else -1
                self._canvas_zoom.adjust(delta, nx, ny)
            else:
                delta = 1 if flags > 0 else -1
                self._zoom_lens.adjust_magnification(delta)

    # -- Canvas ↔ source-image coord conversion --------------------------

    def _canvas_to_image(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        """Map a canvas-space pixel (post canvas-zoom) back to source-image space.

        Returns ``None`` if the source image isn't loaded or the point
        lies outside the image bounds.
        """
        if self._bgr is None:
            return None
        h, w = self._bgr.shape[:2]
        zoom = self._canvas_zoom.zoom
        if zoom <= 1.01:
            ix, iy = cx, cy
        else:
            vis_w = w / zoom
            vis_h = h / zoom
            # Mirror the crop maths in CanvasZoom.apply.
            ncx, ncy = self._canvas_zoom.center
            x0 = max(0, min(int(ncx * w - vis_w / 2), w - int(vis_w)))
            y0 = max(0, min(int(ncy * h - vis_h / 2), h - int(vis_h)))
            ix = x0 + int(cx / zoom)
            iy = y0 + int(cy / zoom)
        if 0 <= ix < w and 0 <= iy < h:
            return ix, iy
        return None

    # -- Image loading -------------------------------------------------------

    def _load_current_image(self) -> None:
        path = self._paths[self._idx]
        bgr = imread_safe(path)
        if bgr is None:
            print(f"Cannot read: {path}")
            return
        self._bgr = bgr
        self._canvas_zoom.reset()
        # Resize brush delta buffers to match new image; clears edits when
        # the dimensions actually change (and preserves them when they don't).
        self._brush.attach_to(bgr.shape[:2])
        self._dirty = True

    def _switch_image(self, delta: int) -> None:
        if len(self._paths) <= 1:
            return
        self._idx = (self._idx + delta) % len(self._paths)
        self._load_current_image()

    # -- Core processing -----------------------------------------------------

    def _process(self) -> None:
        """Run the unified pipeline once and cache the result."""
        if self._bgr is None:
            return

        pipeline = NoiseFixPipeline(
            gradient_sensitivity=self._sensitivity_i / 100.0,
            max_noise_size=self._max_noise_size,
            mask_dilate=self._mask_dilate,
        )
        self._result = pipeline.run(self._bgr)

    # -- Display composition -------------------------------------------------

    def _compose_display(self) -> np.ndarray:
        if self._bgr is None or self._result is None:
            return np.zeros((400, 600, 3), dtype=np.uint8)

        # Compute the effective mask: algorithm output + brush deltas.
        # When the user has edits the overlay/mask-solo views must be
        # re-rendered live so the changes are visible.
        algo_mask = self._result.final_mask
        effective_mask = (
            self._brush.apply(algo_mask) if self._brush.has_edits() else algo_mask
        )

        # In edit mode we force the Mask Overlay view — that's the only
        # view where painting is meaningful.  Re-render it live against
        # the effective mask so brush strokes appear immediately.
        if self._brush.enabled:
            canvas = DebugOverlayRenderer.render(self._bgr, effective_mask)
        else:
            mode = _VIEW_MODES[self._view_mode_idx]
            if self._brush.has_edits() and mode == PreviewMode.MASK_OVERLAY:
                # User toggled brush off but kept edits — keep showing them.
                canvas = DebugOverlayRenderer.render(self._bgr, effective_mask)
            else:
                canvas = self._result.select_view(mode).copy()

        # Canvas zoom (crop + scale).
        canvas = self._canvas_zoom.apply(canvas)

        # HUD overlays.
        self._draw_top_bar(canvas)
        canvas = StatisticsHUD.draw(canvas, self._result.stats, self._canvas_zoom.zoom)

        # Cache for "save current view" before adding the lens / cursor.
        self._last_canvas = canvas.copy()

        # Brush cursor outline tracks the mouse in edit mode.
        if self._brush.enabled:
            canvas = self._brush.draw_cursor(
                canvas, self._cursor_canvas_x, self._cursor_canvas_y
            )

        canvas = self._zoom_lens.render(canvas)
        return self._fit_to_window(canvas)

    def _fit_to_window(self, canvas: np.ndarray) -> np.ndarray:
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(self._WINDOW)
        except Exception:
            return canvas
        if win_w <= 0 or win_h <= 0:
            return canvas

        img_h, img_w = canvas.shape[:2]
        scale = min(win_w / img_w, win_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(canvas, (new_w, new_h), interpolation=interp)

        display = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x_off = (win_w - new_w) // 2
        y_off = (win_h - new_h) // 2
        display[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        self._lb_x_off = x_off
        self._lb_y_off = y_off
        self._lb_scale = scale
        return display

    def _draw_top_bar(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        mode = _VIEW_MODES[self._view_mode_idx]

        mode_text = f"[{self._view_mode_idx + 1}] {mode.value}"
        cv2.putText(canvas, mode_text, (10, 26), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, mode_text, (10, 26), font, 0.6, (0, 255, 200), 1, cv2.LINE_AA)

        fname = os.path.basename(self._paths[self._idx])
        idx_text = f"{fname}  [{self._idx + 1}/{len(self._paths)}]"
        cv2.putText(canvas, idx_text, (10, 52), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, idx_text, (10, 52), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        param_text = (
            f"sens={self._sensitivity_i / 100:.2f}  "
            f"max_size={self._max_noise_size}  "
            f"dilate={self._mask_dilate}"
        )
        cv2.putText(canvas, param_text, (10, 76), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, param_text, (10, 76), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        right_lines: List[str] = []
        if self._brush.enabled:
            right_lines.append(
                f"[E] BRUSH EDIT  r={self._brush.radius}  L=add  R=erase"
            )
        elif self._brush.has_edits():
            right_lines.append("[E] Brush edits applied (overlay view)")
        if self._zoom_lens.enabled:
            right_lines.append("[Z] Zoom Lens ON")
        if self._canvas_zoom.is_zoomed:
            right_lines.append(f"[Ctrl+Scroll] {self._canvas_zoom.zoom:.1f}x  [R] Reset")

        for i, line in enumerate(right_lines):
            ty = 26 + i * 24
            cv2.putText(canvas, line, (w - 360, ty), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (w - 360, ty), font, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        hints = ("1-5: View | 6-0,-: Debug | E: Brush | [/]: Size | C: Clear edits | "
                 "Ctrl+Scroll: Zoom | Z: Lens | S: Save | A/D: Nav | R: Reset | Q: Quit")
        tw = cv2.getTextSize(hints, font, 0.4, 1)[0][0]
        cv2.putText(canvas, hints, (w - tw - 10, h - 12), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, hints, (w - tw - 10, h - 12), font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    # -- Save ----------------------------------------------------------------

    def _save_current_view(self) -> None:
        if self._last_canvas is None:
            return
        src_path = self._paths[self._idx]
        base, ext = os.path.splitext(src_path)

        # The view suffix reflects whether brush edits are baked in.
        if self._brush.enabled or self._brush.has_edits():
            view_suffix = "_mask_edited_overlay"
        else:
            view_suffix = (_SAVE_SUFFIXES[self._view_mode_idx]
                           if self._view_mode_idx < len(_SAVE_SUFFIXES)
                           else "_view")
        view_path = f"{base}{view_suffix}{ext}"
        if imwrite_safe(view_path, self._last_canvas):
            print(f"Saved view: {view_path}")
        else:
            print(f"Failed to save view: {view_path}")

        # When brush edits exist, additionally dump the raw binary mask
        # so it can be fed back into a downstream pipeline (e.g. ComfyUI
        # LoadMask + VAENoiseInpainter).
        if self._brush.has_edits() and self._result is not None:
            effective_mask = self._brush.apply(self._result.final_mask)
            mask_path = f"{base}_mask_edited.png"
            if imwrite_safe(mask_path, effective_mask):
                print(f"Saved edited binary mask: {mask_path}")
            else:
                print(f"Failed to save mask: {mask_path}")


# ---------------------------------------------------------------------------
# Path helpers (re-exported via __init__)
# ---------------------------------------------------------------------------

def collect_image_paths(target: str) -> List[str]:
    """Resolve a file / directory / glob to a sorted list of image paths."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    if os.path.isfile(target):
        return [target]

    if os.path.isdir(target):
        paths = []
        for root, _, files in os.walk(target):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(root, f))
        paths.sort()
        return paths

    paths = sorted(glob.glob(target, recursive=True))
    return [p for p in paths if os.path.splitext(p)[1].lower() in exts]


def open_file_dialog() -> Optional[str]:
    """Open a native file dialog to select an image."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select an image or folder",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except ImportError:
        return input("Enter image or folder path: ").strip().strip('"')
