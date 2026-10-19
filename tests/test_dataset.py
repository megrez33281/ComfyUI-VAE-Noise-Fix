"""
Regression smoke test on a real SDXL sample from ``dataset/``.

Skipped automatically if the dataset is not present (e.g. in an install
from the Comfy Registry, which excludes it).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core import NoiseFixPipeline

SAMPLE = Path(__file__).resolve().parent.parent / "dataset" / "GroupA" / "000.png"


@pytest.mark.skipif(not SAMPLE.exists(), reason="dataset/ not available")
def test_groupa_sample_detects_and_repairs():
    # np.fromfile + imdecode keeps non-ASCII Windows paths working.
    bgr = cv2.imdecode(np.fromfile(str(SAMPLE), np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None

    result = NoiseFixPipeline(0.35, 6, 2).run(bgr)

    # The black-background portrait is known to contain dozens of fireflies.
    assert result.stats.noise_blobs >= 10
    assert result.stats.coverage_pct < 1.0

    changed = np.any(result.repaired_bgr != bgr, axis=2)
    assert changed.any()
    assert not np.any(changed & (result.final_mask == 0))
