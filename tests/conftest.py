"""
Shared synthetic-image helpers.

Every test image is generated in code so the suite is deterministic and
does not depend on the bundled dataset.  Images are OpenCV BGR uint8.
"""

from __future__ import annotations

import numpy as np
import pytest


def gradient_background(h: int = 192, w: int = 192, lo: int = 30, hi: int = 60) -> np.ndarray:
    """Dark horizontal gradient — smooth, so it must never be flagged."""
    ramp = np.linspace(lo, hi, w, dtype=np.float32)
    plane = np.broadcast_to(ramp, (h, w))
    return np.repeat(plane[..., None], 3, axis=2).round().astype(np.uint8)


def add_gaussian_star(
    img: np.ndarray,
    cx: int,
    cy: int,
    sigma: float,
    peak: float = 230.0,
) -> np.ndarray:
    """Blend a soft Gaussian highlight (a 'natural star') into ``img``."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    out = img.astype(np.float32)
    out += (peak - out) * g[..., None]
    return out.clip(0, 255).round().astype(np.uint8)


@pytest.fixture
def clean_image() -> np.ndarray:
    return gradient_background()


@pytest.fixture
def noisy_image() -> np.ndarray:
    """Background with one white and one magenta single-pixel impulse."""
    img = gradient_background()
    img[40, 40] = (255, 255, 255)
    img[150, 60] = (255, 0, 255)
    return img
