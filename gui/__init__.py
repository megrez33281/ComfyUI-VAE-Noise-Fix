"""
GUI sub-package for the standalone OpenCV HighGUI preview application.

Modules:
    image_io     — Windows-safe image read/write helpers.
    zoom_lens    — Mouse-following magnification inset.
    canvas_zoom  — Whole-canvas zoom & pan state machine.
    statistics_hud — HUD overlay drawing for detection statistics.
    preview_app  — The interactive ``PreviewApp`` itself.
"""

from .image_io import imread_safe, imwrite_safe
from .zoom_lens import ZoomLens
from .canvas_zoom import CanvasZoom
from .statistics_hud import StatisticsHUD
from .brush_editor import BrushEditor
from .preview_app import PreviewApp, collect_image_paths, open_file_dialog

__all__ = [
    "imread_safe",
    "imwrite_safe",
    "ZoomLens",
    "CanvasZoom",
    "StatisticsHUD",
    "BrushEditor",
    "PreviewApp",
    "collect_image_paths",
    "open_file_dialog",
]
