"""
ComfyUI custom nodes — VAE high-frequency noise auto-detection & repair.

Three nodes are registered
--------------------------
* **VAENoiseFix** — all-in-one node: detection → repair in a single graph
  node.  Drop-in backward-compatible with earlier workflows.  Use this
  when no manual touch-up is needed.

* **VAENoiseDetector** — detection only.  Outputs the preview view
  selected by ``preview_mode`` plus a binary ``MASK``.  Use this when you
  want to inspect or edit the mask before inpainting.  Also handy for
  fanning the same image into multiple detectors with different
  sensitivities and then OR-merging the masks.

* **VAENoiseInpainter** — Telea inpainting only.  Takes an ``IMAGE``
  plus a ``MASK`` (typically the edited output of the Detector) and
  returns the repaired image.

Workflow A — fully automatic (no touch-up needed)
--------------------------------------------------
::

    [VAE Decode] ─► [VAENoiseFix] ─► repaired IMAGE + MASK

Workflow B — manual mask editing with original image visible
------------------------------------------------------------
The native ComfyUI ``MaskEditor`` only shows a black/white canvas; the
original image is not visible, making it hard to judge where to add or
erase mask regions.

**Recommended solution**: use **ComfyUI-Advanced-ControlNet** (or any
package that provides a mask-editor node accepting a background IMAGE
input) so the original image is rendered underneath the mask while you
paint.

::

    [VAE Decode] ─► [VAENoiseDetector]
                          ├─► IMAGE  ("Mask Overlay" preview mode)
                          │         └─► [PreviewImage]  ← reference while editing
                          │
                          └─► MASK ─► [AdvancedControlNet / MaskEditor
                                       with IMAGE background]
                                              │
                                              ▼  edited MASK
                                    [VAENoiseInpainter] ◄─ [VAE Decode]
                                              │
                                              ▼
                                        repaired IMAGE

Tip: set ``preview_mode`` on ``VAENoiseDetector`` to ``"Mask Overlay"``
so its IMAGE output is the original frame with red markers — route this
into the mask-editor's background socket to see exactly which pixels are
flagged while you paint.

Adheres to SOLID — this file is pure ComfyUI binding; all algorithm
logic lives in ``core/``.  Adding a new preview mode requires editing
exactly one enum and one dispatch dict in ``core/pipeline.py``; the
nodes here are untouched (OCP).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

# Dual-mode import: relative when loaded by ComfyUI as a package
# (custom_nodes/<dir>/__init__.py → ``from .comfyui_node import …``),
# absolute when run via standalone scripts where this module is top-level.
try:
    from .core import (
        NoiseFixPipeline,
        PreviewMode,
        TeleaInpainter,
        TensorBridge,
    )
except ImportError:
    from core import (
        NoiseFixPipeline,
        PreviewMode,
        TeleaInpainter,
        TensorBridge,
    )


# ---------------------------------------------------------------------------
# Shared input-widget definitions
# ---------------------------------------------------------------------------

_DETECTOR_PARAM_INPUTS = {
    "gradient_sensitivity": (
        "FLOAT",
        {
            "default": 0.35,
            "min": 0.01,
            "max": 1.0,
            "step": 0.01,
            "display": "slider",
            "tooltip": (
                "Laplacian + Median Residual detection threshold (normalised). "
                "Lower = more sensitive; higher = only extreme gradients."
            ),
        },
    ),
    "max_noise_size": (
        "INT",
        {
            "default": 6,
            "min": 1,
            "max": 100,
            "step": 1,
            "display": "slider",
            "tooltip": (
                "Maximum connected-component area (pixels @ 1024×1024) "
                "to classify as noise. Auto-scales with image resolution. "
                "Blobs exceeding the scaled value are ignored."
            ),
        },
    ),
    "mask_dilate": (
        "INT",
        {
            "default": 2,
            "min": 0,
            "max": 10,
            "step": 1,
            "display": "slider",
            "tooltip": (
                "Dilation radius (pixels) applied to the verified mask. "
                "Pads the mask so downstream inpainting does not sample "
                "corrupted edge pixels as source data."
            ),
        },
    ),
}


# Preview modes that the standalone Detector exposes (no Repaired /
# Side-by-Side — those require the inpaint step which lives downstream).
_DETECTOR_PREVIEW_MODES: List[str] = [
    m.value for m in PreviewMode
    if m not in (PreviewMode.REPAIRED, PreviewMode.SIDE_BY_SIDE)
]


class VAENoiseFixNode:
    """ComfyUI node: detect & repair SDXL VAE fireflies via classical CV."""

    # -- ComfyUI registration metadata ---------------------------------------

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "image/postprocessing"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "gradient_sensitivity": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Laplacian energy threshold (normalised). "
                            "Lower = more sensitive; higher = only extreme gradients."
                        ),
                    },
                ),
                "max_noise_size": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "tooltip": (
                            "Maximum connected-component area (pixels @ 1024×1024) "
                            "to classify as noise. Auto-scales with image resolution. "
                            "Blobs exceeding the scaled value are ignored."
                        ),
                    },
                ),
                "mask_dilate": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "display": "slider",
                        "tooltip": (
                            "Dilation radius (pixels) applied to the verified mask "
                            "before inpainting. Prevents Telea from sampling "
                            "corrupted edge pixels as source data."
                        ),
                    },
                ),
                "preview_mode": (
                    PreviewMode.choices(),
                    {
                        "default": PreviewMode.REPAIRED.value,
                        "tooltip": (
                            "Which view to return on the IMAGE output. "
                            "MASK output is unaffected."
                        ),
                    },
                ),
            }
        }

    # -- Entry point ---------------------------------------------------------

    def execute(
        self,
        image: torch.Tensor,
        gradient_sensitivity: float,
        max_noise_size: int,
        mask_dilate: int,
        preview_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterate the batch dimension and return ``(IMAGE, MASK)`` tensors."""
        device = image.device
        mode = PreviewMode(preview_mode)

        pipeline = NoiseFixPipeline(
            gradient_sensitivity=gradient_sensitivity,
            max_noise_size=max_noise_size,
            mask_dilate=mask_dilate,
        )

        image_results: List[torch.Tensor] = []
        mask_results:  List[torch.Tensor] = []

        for idx in range(image.shape[0]):
            frame_tensor = image[idx]                                       # [H,W,C]
            bgr_u8 = TensorBridge.comfyui_to_cv2(frame_tensor)
            view_bgr, final_mask, _ = pipeline.run_view(bgr_u8, mode)

            image_results.append(TensorBridge.cv2_to_comfyui(view_bgr, device))
            mask_results.append(TensorBridge.mask_u8_to_comfyui(final_mask, device))

        # Stack along batch dimension.
        image_batch = torch.stack(image_results, dim=0)                     # [B,H,W,C]
        mask_batch = torch.stack(mask_results, dim=0)                       # [B,H,W]
        return image_batch, mask_batch


# ---------------------------------------------------------------------------
# Detector-only node
# ---------------------------------------------------------------------------

class VAENoiseDetectorNode:
    """Detection-only node.

    Outputs the chosen preview view + a binary MASK.  Skips the Telea
    inpaint step entirely (saves time), so this node is ideal when:

      * The mask will be manually edited downstream.
      * Multiple detectors with different parameters are fanned in
        parallel and their masks combined.
      * You only want a detection preview without spending cycles on
        inpaint.

    Recommended mask-editing workflow
    ----------------------------------
    Set ``preview_mode`` to ``"Mask Overlay"`` so the IMAGE output shows
    the original frame with red noise markers.  Route that IMAGE into the
    background-image socket of a mask-editor node (e.g. from
    **ComfyUI-Advanced-ControlNet**) so you can see exactly which pixels
    are flagged while you paint additional regions or erase false
    positives.  Feed the edited MASK into ``VAENoiseInpainter``.
    """

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "image/postprocessing"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                **_DETECTOR_PARAM_INPUTS,
                "preview_mode": (
                    _DETECTOR_PREVIEW_MODES,
                    {
                        "default": PreviewMode.MASK_OVERLAY.value,
                        "tooltip": (
                            "Which view to return on the IMAGE output. "
                            "The MASK output is unaffected."
                        ),
                    },
                ),
            }
        }

    def execute(
        self,
        image: torch.Tensor,
        gradient_sensitivity: float,
        max_noise_size: int,
        mask_dilate: int,
        preview_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = image.device
        mode = PreviewMode(preview_mode)

        pipeline = NoiseFixPipeline(
            gradient_sensitivity=gradient_sensitivity,
            max_noise_size=max_noise_size,
            mask_dilate=mask_dilate,
        )

        image_results: List[torch.Tensor] = []
        mask_results:  List[torch.Tensor] = []
        for idx in range(image.shape[0]):
            bgr_u8 = TensorBridge.comfyui_to_cv2(image[idx])
            view_bgr, final_mask, _ = pipeline.run_view(
                bgr_u8, mode, skip_inpaint=True
            )
            image_results.append(TensorBridge.cv2_to_comfyui(view_bgr, device))
            mask_results.append(TensorBridge.mask_u8_to_comfyui(final_mask, device))

        return (
            torch.stack(image_results, dim=0),
            torch.stack(mask_results, dim=0),
        )


# ---------------------------------------------------------------------------
# Inpainter-only node
# ---------------------------------------------------------------------------

class VAENoiseInpainterNode:
    """Telea-inpainting-only node.

    Takes an IMAGE plus a MASK (anywhere from 0–255 uint8 or 0.0–1.0
    float — both are normalised internally) and returns the inpainted
    image.  Use after ``VAENoiseDetector`` + optional ``MaskEditor`` to
    repair the manually-curated regions.

    If the supplied mask is empty the original image is returned
    unchanged (no wasted Telea call).
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/postprocessing"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask":  ("MASK",),
                "max_noise_size": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "tooltip": (
                            "Used solely to derive the Telea inpaint radius "
                            "(≈ ⌈√max_noise_size⌉, clamped to [2, 7]). "
                            "Match the value used in the upstream Detector "
                            "for consistent results."
                        ),
                    },
                ),
            }
        }

    def execute(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        max_noise_size: int,
    ) -> Tuple[torch.Tensor]:
        device = image.device
        batch_size = image.shape[0]

        # ComfyUI MASK is [B, H, W] or [H, W] float32.  Normalise to a
        # 3-D [B, H, W] view so we can iterate uniformly.
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        # Broadcast mask batch if a single mask is fed to a batched image.
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)

        results: List[torch.Tensor] = []
        for idx in range(batch_size):
            bgr_u8 = TensorBridge.comfyui_to_cv2(image[idx])
            mask_np = mask[idx].detach().cpu().numpy()
            mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255

            if mask_u8.any():
                out = TeleaInpainter.inpaint(bgr_u8, mask_u8, max_noise_size)
            else:
                out = bgr_u8
            results.append(TensorBridge.cv2_to_comfyui(out, device))

        return (torch.stack(results, dim=0),)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VAENoiseFix":       VAENoiseFixNode,
    "VAENoiseDetector":  VAENoiseDetectorNode,
    "VAENoiseInpainter": VAENoiseInpainterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAENoiseFix":       "VAE Noise Fix (Traditional CV)",
    "VAENoiseDetector":  "VAE Noise Detector",
    "VAENoiseInpainter": "VAE Noise Inpainter (Telea)",
}
