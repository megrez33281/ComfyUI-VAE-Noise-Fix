"""
Regenerate the before/after figure used in the README.

Runs the default pipeline on ``dataset/GroupA/000.png``, finds the crop
window containing the most detected noise, and writes a three-panel
nearest-neighbour zoom (Original | Detected | Repaired) next to a
downscaled overview with the crop window outlined.

Usage:
    python scripts/make_readme_figure.py [output.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import NoiseFixPipeline  # noqa: E402

SOURCE = REPO_ROOT / "dataset" / "GroupA" / "000.png"
DEFAULT_OUT = REPO_ROOT / "docs" / "images" / "before_after.png"

CROP = 96           # crop window (source pixels)
ZOOM = 4            # nearest-neighbour magnification of the crop
GAP = 8             # spacing between panels
LABEL_H = 28        # label strip height
BG = (24, 24, 24)


# np.fromfile / imdecode instead of cv2.imread so non-ASCII Windows paths
# work (same approach as gui/image_io.py, without pulling in tkinter).
def _imread(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def _imwrite(path: Path, img: np.ndarray) -> bool:
    ok, encoded = cv2.imencode(path.suffix, img)
    if ok:
        encoded.tofile(str(path))
    return ok


def _densest_window(mask: np.ndarray, size: int, step: int = 16) -> tuple[int, int]:
    integral = cv2.integral((mask > 0).astype(np.float32))
    h, w = mask.shape
    best = (-1.0, 0, 0)
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            s = (integral[y + size, x + size] - integral[y, x + size]
                 - integral[y + size, x] + integral[y, x])
            if s > best[0]:
                best = (s, x, y)
    return best[1], best[2]


def _labelled(panel: np.ndarray, text: str) -> np.ndarray:
    strip = np.full((LABEL_H, panel.shape[1], 3), BG, np.uint8)
    cv2.putText(strip, text, (6, LABEL_H - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([strip, panel])


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

    bgr = _imread(SOURCE)
    if bgr is None:
        raise SystemExit(f"Cannot read {SOURCE}")

    result = NoiseFixPipeline(
        gradient_sensitivity=0.35, max_noise_size=6, mask_dilate=2,
    ).run(bgr)

    x, y = _densest_window(result.final_mask, CROP)
    crop = (slice(y, y + CROP), slice(x, x + CROP))

    def zoom(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img[crop], None, fx=ZOOM, fy=ZOOM,
                          interpolation=cv2.INTER_NEAREST)

    panels = [
        _labelled(zoom(result.original_bgr), "Original"),
        _labelled(zoom(result.overlay_bgr), "Detected"),
        _labelled(zoom(result.repaired_bgr), "Repaired"),
    ]
    panel_h = panels[0].shape[0]

    # Overview: downscale the full frame to the panel height and outline the crop.
    overview = result.original_bgr.copy()
    cv2.rectangle(overview, (x, y), (x + CROP, y + CROP), (0, 200, 255), 4)
    scale = (panel_h - LABEL_H) / overview.shape[0]
    overview = cv2.resize(overview, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    panels.insert(0, _labelled(overview, "Full frame"))

    spacer = np.full((panel_h, GAP, 3), BG, np.uint8)
    row = [panels[0]]
    for p in panels[1:]:
        row += [spacer, p]
    figure = np.hstack(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _imwrite(out_path, figure):
        raise SystemExit(f"Failed to write {out_path}")
    print(f"Wrote {out_path}  ({figure.shape[1]}x{figure.shape[0]}, "
          f"crop at x={x}, y={y})")


if __name__ == "__main__":
    main()
