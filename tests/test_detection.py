"""
Algorithm-level tests for ``core``.

Covers the two behaviours the node exists for: isolated impulses must be
detected and repaired, while soft natural highlights (stars) and thin
structures must be left untouched.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import add_gaussian_star, gradient_background
from core import GradientNoiseDetector, NoiseFixPipeline

DEFAULTS = dict(gradient_sensitivity=0.35, max_noise_size=6, mask_dilate=2)
REPO_ROOT = Path(__file__).resolve().parent.parent


def run(img: np.ndarray, **overrides):
    return NoiseFixPipeline(**{**DEFAULTS, **overrides}).run(img)


def mask_hits(mask: np.ndarray, x: int, y: int, r: int = 2) -> bool:
    return bool(mask[y - r:y + r + 1, x - r:x + r + 1].any())


# -- No false positives on clean content --------------------------------------

def test_clean_gradient_has_no_detections(clean_image):
    result = run(clean_image)
    assert not result.final_mask.any()
    assert np.array_equal(result.repaired_bgr, clean_image)
    assert result.stats.noise_blobs == 0


def test_thin_line_is_not_flagged():
    img = gradient_background()
    img[96, 20:80] = 255     # 1 px wide, 60 px long — hair / scratch-like
    assert not run(img).final_mask.any()


# -- Impulse detection ---------------------------------------------------------

def test_white_impulse_detected_and_repaired(noisy_image):
    result = run(noisy_image)
    assert mask_hits(result.final_mask, 40, 40)

    # The repaired pixel should be pulled back to the local background level.
    bg = gradient_background()
    diff = np.abs(result.repaired_bgr[40, 40].astype(int) - bg[40, 40].astype(int))
    assert diff.max() <= 10


def test_chromatic_impulse_detected(noisy_image):
    result = run(noisy_image)
    assert mask_hits(result.final_mask, 60, 150)
    # Magenta cast removed: channels end up close to each other again.
    b, g, r = result.repaired_bgr[150, 60].astype(int)
    assert max(b, g, r) - min(b, g, r) <= 10


def test_pixels_outside_mask_are_untouched(noisy_image):
    result = run(noisy_image)
    outside = result.final_mask == 0
    assert outside.any()
    assert np.array_equal(result.repaired_bgr[outside], noisy_image[outside])


def test_mask_dilate_grows_mask(noisy_image):
    tight = run(noisy_image, mask_dilate=0).final_mask
    wide = run(noisy_image, mask_dilate=4).final_mask
    assert (wide > 0).sum() > (tight > 0).sum()
    assert np.all(wide[tight > 0] > 0)


def test_2x2_block_detected_with_larger_max_noise_size():
    # A 2x2 block's Laplacian seed includes a 1-px ring (16 px total), so it
    # needs max_noise_size above the default of 6 to qualify as noise.
    img = gradient_background()
    img[100:102, 100:102] = 255
    assert mask_hits(run(img, max_noise_size=20).final_mask, 100, 100)


# -- Natural-highlight protection ---------------------------------------------

def test_impulse_passes_lab_verification(noisy_image):
    maps = GradientNoiseDetector(0.35, 6, 192, 192).detect_with_intermediates(noisy_image)
    assert maps.filtered_mask[38:43, 38:43].any()
    assert maps.verified_mask[38:43, 38:43].any()


def test_lab_stage_rejects_soft_star_that_passes_shape_filter():
    # A dim, wide star is small and compact enough to survive the structural
    # filter; only the LAB cliff-steepness check tells it apart from noise.
    img = add_gaussian_star(gradient_background(), 96, 96, sigma=2.0, peak=120)
    maps = GradientNoiseDetector(0.35, 6, 192, 192).detect_with_intermediates(img)
    window = (slice(85, 108), slice(85, 108))
    assert maps.filtered_mask[window].any()
    assert not maps.verified_mask[window].any()


@pytest.mark.parametrize("sigma", [1.0, 1.5, 2.0])
def test_gaussian_star_is_preserved(sigma):
    # End-to-end: bright stars at default settings are rejected by the
    # size/isolation rules before they ever reach the LAB stage.
    img = add_gaussian_star(gradient_background(), 96, 96, sigma)
    result = run(img)
    assert not mask_hits(result.final_mask, 96, 96, r=4)
    assert np.array_equal(result.repaired_bgr[88:105, 88:105], img[88:105, 88:105])


def test_star_kept_while_neighbouring_impulse_removed():
    img = add_gaussian_star(gradient_background(), 130, 130, sigma=2.0, peak=120)
    img[50, 50] = 255
    mask = run(img).final_mask
    assert mask_hits(mask, 50, 50)
    assert not mask_hits(mask, 130, 130, r=4)


# -- Pipeline plumbing ---------------------------------------------------------

def test_skip_inpaint_leaves_image_unchanged(noisy_image):
    result = NoiseFixPipeline(**DEFAULTS).run(noisy_image, skip_inpaint=True)
    assert result.final_mask.any()
    assert np.array_equal(result.repaired_bgr, noisy_image)


def test_result_shapes_and_dtypes(noisy_image):
    result = run(noisy_image)
    h, w = noisy_image.shape[:2]
    assert result.final_mask.shape == (h, w)
    assert result.final_mask.dtype == np.uint8
    assert set(np.unique(result.final_mask)) <= {0, 255}
    assert result.repaired_bgr.shape == noisy_image.shape
    assert result.side_by_side.shape == (h, 2 * w, 3)


@pytest.mark.parametrize(
    ("h", "w", "expected"),
    [(512, 512, 6), (1024, 1024, 6), (2048, 2048, 24)],
)
def test_max_noise_size_scales_above_one_megapixel(h, w, expected):
    det = GradientNoiseDetector(0.35, 6, image_height=h, image_width=w)
    assert det.scaled_max_noise_size == expected


# -- Dependency hygiene -------------------------------------------------------

def test_core_imports_without_torch():
    # Block torch entirely; core (used by the standalone GUI) must still load
    # and run.
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import numpy as np\n"
        "from core import NoiseFixPipeline\n"
        "img = np.full((32, 32, 3), 40, np.uint8); img[16, 16] = 255\n"
        "assert NoiseFixPipeline(0.35, 6, 2).run(img).final_mask.any()\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)
