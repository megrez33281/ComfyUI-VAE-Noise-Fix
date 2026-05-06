"""
Interactive GUI Preview for VAE Noise Fix

A standalone OpenCV HighGUI application for visually tuning detection
parameters, previewing mask overlays, comparing before/after results,
and inspecting noise details with zoom — without ComfyUI.

Usage:
    python gui_preview.py                         # file dialog
    python gui_preview.py path/to/image.png       # direct open
    python gui_preview.py path/to/folder/         # browse folder

Controls:
    Trackbars       - gradient_sensitivity, max_noise_size
    1               - Original image
    2               - Mask overlay (red)
    3               - Mask only (white on black, green circles)
    4               - Repaired image
    5               - Side-by-side (original | repaired)
    Ctrl + Scroll   - Zoom canvas in/out at mouse position
    Z               - Toggle zoom lens on/off
    Scroll          - Adjust zoom lens magnification
    S               - Save current view
    A / D           - Previous / Next image (folder mode)
    Q / ESC         - Quit
"""

from __future__ import annotations

import os
import sys
import glob
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vae_noise_fix import (
    TensorBridge,
    GradientNoiseDetector,
    DebugOverlayRenderer,
    TeleaInpainter,
)


# ---------------------------------------------------------------------------
# Image I/O helpers (Windows non-ASCII path safe)
# ---------------------------------------------------------------------------

def _imread_safe(path: str) -> Optional[np.ndarray]:
    """Read image supporting non-ASCII file paths on Windows."""
    buf = np.fromfile(path, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _imwrite_safe(path: str, img: np.ndarray) -> bool:
    """Write image supporting non-ASCII file paths on Windows."""
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, img)
    if ok:
        encoded.tofile(path)
    return ok


# ---------------------------------------------------------------------------
# Statistics calculator
# ---------------------------------------------------------------------------

class NoiseStatistics:
    """Compute and format detection statistics for HUD overlay."""

    @staticmethod
    def compute(
        mask: np.ndarray,
        image_shape: Tuple[int, int],
        elapsed_ms: float,
    ) -> dict:
        """Return a statistics dict from a binary noise mask."""
        h, w = image_shape
        total_pixels = h * w

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        noise_count = num_labels - 1
        noise_pixels = int(cv2.countNonZero(mask))
        coverage_pct = (noise_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        areas: List[int] = []
        for i in range(1, num_labels):
            areas.append(int(stats[i, cv2.CC_STAT_AREA]))

        return {
            "resolution": f"{w} x {h}",
            "noise_blobs": noise_count,
            "noise_pixels": noise_pixels,
            "coverage_pct": coverage_pct,
            "area_min": min(areas) if areas else 0,
            "area_max": max(areas) if areas else 0,
            "area_avg": sum(areas) / len(areas) if areas else 0.0,
            "elapsed_ms": elapsed_ms,
        }

    @staticmethod
    def draw_hud(canvas: np.ndarray, stats: dict, view_zoom: float) -> np.ndarray:
        """Render statistics as a semi-transparent HUD panel on the image."""
        out = canvas.copy()
        lines = [
            f"Resolution:   {stats['resolution']}",
            f"Noise blobs:  {stats['noise_blobs']}",
            f"Noise pixels: {stats['noise_pixels']}  ({stats['coverage_pct']:.4f}%)",
            f"Area range:   {stats['area_min']} ~ {stats['area_max']}  (avg {stats['area_avg']:.1f})",
            f"Detection:    {stats['elapsed_ms']:.1f} ms",
            f"Canvas zoom:  {view_zoom:.1f}x",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        line_height = 22
        padding = 10
        panel_w = 380
        panel_h = len(lines) * line_height + padding * 2

        x0, y0 = 10, out.shape[0] - panel_h - 10
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

        for i, line in enumerate(lines):
            ty = y0 + padding + (i + 1) * line_height - 4
            cv2.putText(out, line, (x0 + padding, ty), font, scale,
                        (0, 255, 200), thickness, cv2.LINE_AA)

        return out


# ---------------------------------------------------------------------------
# Zoom lens renderer
# ---------------------------------------------------------------------------

class ZoomLens:
    """Draws an inset magnification window following the mouse cursor."""

    def __init__(self, mag: int = 6, lens_size: int = 200) -> None:
        self._mag = mag
        self._lens_size = lens_size
        self._enabled = False
        self._mx = 0
        self._my = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self) -> None:
        self._enabled = not self._enabled

    def update_position(self, x: int, y: int) -> None:
        self._mx = x
        self._my = y

    def adjust_magnification(self, delta: int) -> None:
        self._mag = max(2, min(self._mag + delta, 16))

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Draw the zoom lens inset on the canvas."""
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
                       (0, 255, 200), 2)
        out[iy0:iy0 + self._lens_size, ix0:ix0 + self._lens_size] = zoomed

        label = f"{self._mag}x"
        cv2.putText(out, label, (ix0 + 6, iy0 + self._lens_size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        cv2.drawMarker(out, (self._mx, self._my), (0, 255, 200),
                       cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

        return out


# ---------------------------------------------------------------------------
# Canvas zoom/pan state
# ---------------------------------------------------------------------------

class CanvasZoom:
    """Manages zoom level and pan offset for the main canvas view.

    Coordinates are stored in normalised [0, 1] space relative to the
    source image so they remain valid across different display sizes.
    """

    _MIN_ZOOM = 1.0
    _MAX_ZOOM = 16.0

    def __init__(self) -> None:
        self._zoom = 1.0
        # Pan center in normalised source coords.
        self._cx = 0.5
        self._cy = 0.5

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def is_zoomed(self) -> bool:
        return self._zoom > 1.01

    def reset(self) -> None:
        self._zoom = 1.0
        self._cx = 0.5
        self._cy = 0.5

    def adjust(self, delta: int, mouse_nx: float, mouse_ny: float) -> None:
        """Zoom in/out keeping the point under the mouse fixed.

        Args:
            delta:    +1 to zoom in, -1 to zoom out.
            mouse_nx: mouse x in normalised [0, 1] display space.
            mouse_ny: mouse y in normalised [0, 1] display space.
        """
        old_zoom = self._zoom
        factor = 1.25 if delta > 0 else 1.0 / 1.25
        new_zoom = max(self._MIN_ZOOM, min(self._zoom * factor, self._MAX_ZOOM))

        if abs(new_zoom - old_zoom) < 0.001:
            return

        # Point under mouse in source normalised coords (before zoom change).
        half_w_old = 0.5 / old_zoom
        half_h_old = 0.5 / old_zoom
        src_x = self._cx - half_w_old + mouse_nx / old_zoom
        src_y = self._cy - half_h_old + mouse_ny / old_zoom

        # Solve for new center so src_x,src_y stays at mouse_nx,mouse_ny.
        half_w_new = 0.5 / new_zoom
        half_h_new = 0.5 / new_zoom
        new_cx = src_x - mouse_nx / new_zoom + half_w_new
        new_cy = src_y - mouse_ny / new_zoom + half_h_new

        self._zoom = new_zoom
        self._cx = max(half_w_new, min(new_cx, 1.0 - half_w_new))
        self._cy = max(half_h_new, min(new_cy, 1.0 - half_h_new))

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Crop and scale the source image according to current zoom/pan."""
        if not self.is_zoomed:
            return img

        h, w = img.shape[:2]
        vis_w = w / self._zoom
        vis_h = h / self._zoom

        x0 = int(self._cx * w - vis_w / 2)
        y0 = int(self._cy * h - vis_h / 2)

        # Clamp to image bounds.
        x0 = max(0, min(x0, w - int(vis_w)))
        y0 = max(0, min(y0, h - int(vis_h)))
        x1 = x0 + int(vis_w)
        y1 = y0 + int(vis_h)

        crop = img[y0:y1, x0:x1]
        interp = cv2.INTER_NEAREST if self._zoom >= 4.0 else cv2.INTER_LINEAR
        return cv2.resize(crop, (w, h), interpolation=interp)


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------

class PreviewApp:
    """Interactive parameter tuning and preview application."""

    _WINDOW = "VAE Noise Fix - Preview"
    _VIEW_NAMES = ["Original", "Mask Overlay", "Mask Only", "Repaired", "Side-by-Side"]
    _SAVE_SUFFIXES = ["_original", "_mask_overlay", "_mask_only", "_fixed", "_compare"]

    def __init__(self, image_paths: List[str]) -> None:
        self._paths = image_paths
        self._idx = 0

        # Parameters (trackbar values are integers; sensitivity is ×100).
        self._sensitivity_i = 35
        self._max_noise_size = 6
        self._mask_dilate = 2

        # View state.
        self._view_mode = 1
        self._zoom_lens = ZoomLens()
        self._canvas_zoom = CanvasZoom()
        self._mouse_x = 0
        self._mouse_y = 0

        # Cached processing results.
        self._bgr: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._overlay: Optional[np.ndarray] = None
        self._mask_solo: Optional[np.ndarray] = None
        self._repaired: Optional[np.ndarray] = None
        self._stats: Optional[dict] = None
        self._dirty = True

        # Last composed frame (before zoom lens) for saving.
        self._last_canvas: Optional[np.ndarray] = None

        # Letterbox mapping: updated each frame by _fit_to_window.
        self._lb_x_off = 0
        self._lb_y_off = 0
        self._lb_scale = 1.0

    # -- Lifecycle -----------------------------------------------------------

    def run(self) -> None:
        """Main event loop."""
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
            elif key == ord("1"):
                self._view_mode = 0
            elif key == ord("2"):
                self._view_mode = 1
            elif key == ord("3"):
                self._view_mode = 2
            elif key == ord("4"):
                self._view_mode = 3
            elif key == ord("5"):
                self._view_mode = 4
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
        """Map window (letterboxed) pixel coordinates to canvas pixel coords."""
        cx = int((wx - self._lb_x_off) / self._lb_scale) if self._lb_scale > 0 else wx
        cy = int((wy - self._lb_y_off) / self._lb_scale) if self._lb_scale > 0 else wy
        return cx, cy

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        self._mouse_x = x
        self._mouse_y = y

        # Translate window coords to canvas coords for zoom lens.
        cx, cy = self._window_to_canvas(x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            self._zoom_lens.update_position(cx, cy)

        elif event == cv2.EVENT_MOUSEWHEEL:
            ctrl_held = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0

            if ctrl_held:
                # Ctrl + Scroll → canvas zoom at mouse position.
                # Normalised position within the image area (excluding letterbox).
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
                # Scroll without Ctrl → zoom lens magnification.
                delta = 1 if flags > 0 else -1
                self._zoom_lens.adjust_magnification(delta)

    # -- Image loading -------------------------------------------------------

    def _load_current_image(self) -> None:
        path = self._paths[self._idx]
        bgr = _imread_safe(path)
        if bgr is None:
            print(f"Cannot read: {path}")
            return
        self._bgr = bgr
        self._canvas_zoom.reset()
        self._dirty = True

    def _switch_image(self, delta: int) -> None:
        if len(self._paths) <= 1:
            return
        self._idx = (self._idx + delta) % len(self._paths)
        self._load_current_image()

    # -- Core processing -----------------------------------------------------

    def _process(self) -> None:
        """Run detection + inpainting and cache all results."""
        if self._bgr is None:
            return

        sensitivity = self._sensitivity_i / 100.0
        h, w = self._bgr.shape[:2]

        detector = GradientNoiseDetector(
            gradient_sensitivity=sensitivity,
            max_noise_size=self._max_noise_size,
            image_height=h,
            image_width=w,
            mask_dilate=self._mask_dilate,
        )

        t0 = time.perf_counter()
        self._mask = detector.detect(self._bgr)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._overlay = DebugOverlayRenderer.render(self._bgr, self._mask)
        self._mask_solo = self._render_mask_solo(self._mask)

        if np.any(self._mask):
            self._repaired = TeleaInpainter.inpaint(
                self._bgr, self._mask, self._max_noise_size
            )
        else:
            self._repaired = self._bgr.copy()

        self._stats = NoiseStatistics.compute(self._mask, (h, w), elapsed_ms)

    @staticmethod
    def _render_mask_solo(mask: np.ndarray) -> np.ndarray:
        """Render mask as white blobs on black with green circle markers."""
        h, w = mask.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[mask > 0] = (255, 255, 255)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, num_labels):
            cx = int(centroids[i][0])
            cy = int(centroids[i][1])
            area = stats[i, cv2.CC_STAT_AREA]
            radius = max(10, int(math.sqrt(area) * 4))
            cv2.circle(canvas, (cx, cy), radius, (0, 255, 0), 2, cv2.LINE_AA)

        return canvas

    # -- Display composition -------------------------------------------------

    def _compose_display(self) -> np.ndarray:
        """Build the final display frame."""
        if self._bgr is None:
            return np.zeros((400, 600, 3), dtype=np.uint8)

        # Select base canvas by view mode.
        if self._view_mode == 0:
            canvas = self._bgr.copy()
        elif self._view_mode == 1:
            canvas = self._overlay.copy()
        elif self._view_mode == 2:
            canvas = self._mask_solo.copy()
        elif self._view_mode == 3:
            canvas = self._repaired.copy()
        elif self._view_mode == 4:
            canvas = self._make_side_by_side()
        else:
            canvas = self._bgr.copy()

        # Apply canvas zoom (crop + scale).
        canvas = self._canvas_zoom.apply(canvas)

        # HUD overlays.
        self._draw_top_bar(canvas)
        if self._stats is not None:
            canvas = NoiseStatistics.draw_hud(canvas, self._stats, self._canvas_zoom.zoom)

        # Cache for saving (before zoom lens overlay).
        self._last_canvas = canvas.copy()

        # Zoom lens on top of everything.
        canvas = self._zoom_lens.render(canvas)

        # Fit canvas to window with correct aspect ratio (letterbox).
        canvas = self._fit_to_window(canvas)

        return canvas

    def _fit_to_window(self, canvas: np.ndarray) -> np.ndarray:
        """Resize canvas to fill the window while maintaining aspect ratio.

        Adds black letterbox bars where the aspect ratios differ.
        Falls back to the raw canvas if window dimensions cannot be read.
        """
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

        # Place centred on a black background matching the window size.
        display = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x_off = (win_w - new_w) // 2
        y_off = (win_h - new_h) // 2
        display[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Cache mapping for mouse coordinate translation.
        self._lb_x_off = x_off
        self._lb_y_off = y_off
        self._lb_scale = scale

        return display

    def _make_side_by_side(self) -> np.ndarray:
        """Concatenate original and repaired images horizontally."""
        left = self._bgr.copy()
        right = self._repaired.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(left, "Original", (10, 28), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(left, "Original", (10, 28), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(right, "Repaired", (10, 28), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, "Repaired", (10, 28), font, 0.7, (0, 200, 100), 1, cv2.LINE_AA)

        combined = np.hstack([left, right])
        mid_x = left.shape[1]
        cv2.line(combined, (mid_x, 0), (mid_x, combined.shape[0]), (0, 255, 200), 2)
        return combined

    def _draw_top_bar(self, canvas: np.ndarray) -> None:
        """Draw mode indicator and filename at the top."""
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        mode_text = f"[{self._view_mode + 1}] {self._VIEW_NAMES[self._view_mode]}"
        cv2.putText(canvas, mode_text, (10, 26), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, mode_text, (10, 26), font, 0.6, (0, 255, 200), 1, cv2.LINE_AA)

        fname = os.path.basename(self._paths[self._idx])
        idx_text = f"{fname}  [{self._idx + 1}/{len(self._paths)}]"
        cv2.putText(canvas, idx_text, (10, 52), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, idx_text, (10, 52), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        param_text = f"sens={self._sensitivity_i / 100:.2f}  max_size={self._max_noise_size}  dilate={self._mask_dilate}"
        cv2.putText(canvas, param_text, (10, 76), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, param_text, (10, 76), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Right-side indicators.
        right_lines: List[str] = []
        if self._zoom_lens.enabled:
            right_lines.append("[Z] Zoom Lens ON")
        if self._canvas_zoom.is_zoomed:
            right_lines.append(f"[Ctrl+Scroll] {self._canvas_zoom.zoom:.1f}x  [R] Reset")

        for i, line in enumerate(right_lines):
            ty = 26 + i * 24
            cv2.putText(canvas, line, (w - 300, ty), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (w - 300, ty), font, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        hints = "1-5: View | Ctrl+Scroll: Zoom | Z: Lens | S: Save | A/D: Nav | R: Reset | Q: Quit"
        tw = cv2.getTextSize(hints, font, 0.4, 1)[0][0]
        cv2.putText(canvas, hints, (w - tw - 10, h - 12), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, hints, (w - tw - 10, h - 12), font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    # -- Save ----------------------------------------------------------------

    def _save_current_view(self) -> None:
        """Save the currently displayed view (whatever mode is active)."""
        if self._last_canvas is None:
            return

        src_path = self._paths[self._idx]
        base, ext = os.path.splitext(src_path)
        suffix = self._SAVE_SUFFIXES[self._view_mode] if self._view_mode < len(self._SAVE_SUFFIXES) else "_view"
        out_path = f"{base}{suffix}{ext}"

        if _imwrite_safe(out_path, self._last_canvas):
            print(f"Saved: {out_path}")
        else:
            print(f"Failed to save: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _collect_image_paths(target: str) -> List[str]:
    """Resolve a file or directory argument to a sorted list of image paths."""
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


def _open_file_dialog() -> Optional[str]:
    """Open a native file dialog to select an image or folder."""
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


def main() -> None:
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = _open_file_dialog()
        if not target:
            print("No file selected. Exiting.")
            return

    paths = _collect_image_paths(target)
    if not paths:
        print(f"No images found at: {target}")
        return

    print(f"Loaded {len(paths)} image(s). Launching preview...")
    print()
    print("  Controls:")
    print("    1-5          Switch view (Original / Overlay / Mask / Repaired / Compare)")
    print("    Ctrl+Scroll  Zoom canvas in/out at mouse position")
    print("    R            Reset canvas zoom")
    print("    Z            Toggle zoom lens")
    print("    Scroll       Adjust zoom lens magnification")
    print("    S            Save current view")
    print("    A / D        Previous / Next image")
    print("    Q / ESC      Quit")
    print()

    app = PreviewApp(paths)
    app.run()


if __name__ == "__main__":
    main()
