"""
ComfyUI binding tests: tensor shapes, dtypes and batch handling.

These exercise the node classes directly, without a running ComfyUI.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import gradient_background  # noqa: E402
from comfyui_node import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    VAENoiseDetectorNode,
    VAENoiseFixNode,
    VAENoiseInpainterNode,
    VAENoiseMaskEditorNode,
)
from core import PreviewMode, TensorBridge  # noqa: E402

PARAMS = dict(gradient_sensitivity=0.35, max_noise_size=6, mask_dilate=2)


def to_tensor(*frames_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 frames → ComfyUI IMAGE ``[B, H, W, 3]`` float32 RGB."""
    rgb = [f[..., ::-1].astype(np.float32) / 255.0 for f in frames_bgr]
    return torch.from_numpy(np.stack(rgb))


@pytest.fixture
def batch(noisy_image, clean_image) -> torch.Tensor:
    return to_tensor(noisy_image, clean_image, noisy_image)


# -- Registration --------------------------------------------------------------

def test_mappings_are_consistent():
    assert NODE_CLASS_MAPPINGS.keys() == NODE_DISPLAY_NAME_MAPPINGS.keys()
    for cls in NODE_CLASS_MAPPINGS.values():
        inputs = cls.INPUT_TYPES()
        assert "image" in inputs["required"]
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
        assert callable(getattr(cls, cls.FUNCTION))


def test_detector_preview_modes_exclude_inpaint_views():
    choices = VAENoiseDetectorNode.INPUT_TYPES()["required"]["preview_mode"][0]
    assert PreviewMode.REPAIRED.value not in choices
    assert PreviewMode.SIDE_BY_SIDE.value not in choices
    assert PreviewMode.MASK_OVERLAY.value in choices


# -- VAENoiseFix ---------------------------------------------------------------

def test_fix_node_batch_shapes_and_types(batch):
    image, mask = VAENoiseFixNode().execute(
        image=batch, preview_mode=PreviewMode.REPAIRED.value, **PARAMS
    )
    b, h, w, _ = batch.shape
    assert image.shape == (b, h, w, 3)
    assert mask.shape == (b, h, w)
    assert image.dtype == torch.float32 and mask.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_fix_node_processes_frames_independently(batch):
    _, mask = VAENoiseFixNode().execute(
        image=batch, preview_mode=PreviewMode.REPAIRED.value, **PARAMS
    )
    assert mask[0].any() and mask[2].any()
    assert not mask[1].any()                      # clean frame stays clean
    assert torch.equal(mask[0], mask[2])


@pytest.mark.parametrize("mode", PreviewMode.choices())
def test_fix_node_every_preview_mode(batch, mode):
    image, mask = VAENoiseFixNode().execute(image=batch, preview_mode=mode, **PARAMS)
    b, h, w, _ = batch.shape
    expected_w = 2 * w if mode == PreviewMode.SIDE_BY_SIDE.value else w
    assert image.shape == (b, h, expected_w, 3)
    assert mask.shape == (b, h, w)               # mask is independent of view


def test_fix_node_original_mode_is_passthrough(batch):
    image, _ = VAENoiseFixNode().execute(
        image=batch, preview_mode=PreviewMode.ORIGINAL.value, **PARAMS
    )
    assert torch.allclose(image, batch, atol=1 / 255)


# -- VAENoiseDetector / VAENoiseInpainter ------------------------------------

def test_detector_then_inpainter_matches_fix_node(batch):
    _, mask = VAENoiseDetectorNode().execute(
        image=batch, preview_mode=PreviewMode.MASK_OVERLAY.value, **PARAMS
    )
    (repaired,) = VAENoiseInpainterNode().execute(
        image=batch, mask=mask, max_noise_size=PARAMS["max_noise_size"]
    )
    expected, _ = VAENoiseFixNode().execute(
        image=batch, preview_mode=PreviewMode.REPAIRED.value, **PARAMS
    )
    assert torch.equal(repaired, expected)


def test_inpainter_broadcasts_single_2d_mask(batch):
    b, h, w, _ = batch.shape
    mask = torch.zeros(h, w)
    mask[38:43, 38:43] = 1.0
    (out,) = VAENoiseInpainterNode().execute(image=batch, mask=mask, max_noise_size=6)
    assert out.shape == batch.shape
    # Every frame got the same patch inpainted, including the clean one
    # (its pixels there are background already, so it barely moves).
    assert not torch.allclose(out[0, 40, 40], batch[0, 40, 40], atol=0.05)


def test_inpainter_empty_mask_returns_input(batch):
    b, h, w, _ = batch.shape
    (out,) = VAENoiseInpainterNode().execute(
        image=batch, mask=torch.zeros(b, h, w), max_noise_size=6
    )
    assert torch.allclose(out, batch, atol=1 / 255)


def test_inpainter_resizes_mismatched_mask(batch):
    b, h, w, _ = batch.shape
    mask = torch.zeros(b, h // 2, w // 2)
    mask[:, 20, 20] = 1.0
    (out,) = VAENoiseInpainterNode().execute(image=batch, mask=mask, max_noise_size=6)
    assert out.shape == batch.shape


# -- VAENoiseMaskEditor --------------------------------------------------------

def test_mask_editor_passes_through_without_edits(batch):
    mask = torch.zeros(batch.shape[:3])
    image_out, mask_out = VAENoiseMaskEditorNode().execute(
        image=batch, mask=mask, node_id="test-node"
    )
    assert image_out is batch
    assert torch.equal(mask_out, mask)


# -- TensorBridge --------------------------------------------------------------

def test_tensor_bridge_round_trip():
    bgr = gradient_background(16, 24)
    bgr[3, 5] = (10, 200, 90)
    t = TensorBridge.cv2_to_comfyui(bgr, torch.device("cpu"))
    assert t.shape == (16, 24, 3)
    assert np.array_equal(TensorBridge.comfyui_to_cv2(t), bgr)
    # RGB order on the tensor side.
    assert torch.allclose(t[3, 5], torch.tensor([90, 200, 10]) / 255.0)
