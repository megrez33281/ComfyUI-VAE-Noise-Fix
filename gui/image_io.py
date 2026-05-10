"""
Windows non-ASCII path safe image I/O helpers.

OpenCV's ``cv2.imread`` / ``cv2.imwrite`` go through the C runtime on
Windows and break on Unicode paths (e.g. Chinese, Japanese, emoji).
These helpers route through ``np.fromfile`` / ``ndarray.tofile`` which
use the Windows wide-char API and handle every code page correctly.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def imread_safe(path: str) -> Optional[np.ndarray]:
    """Read an image, supporting non-ASCII paths on Windows."""
    buf = np.fromfile(path, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def imwrite_safe(path: str, img: np.ndarray) -> bool:
    """Write an image, supporting non-ASCII paths on Windows."""
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, img)
    if ok:
        encoded.tofile(path)
    return ok
