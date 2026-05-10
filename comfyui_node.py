"""
ComfyUI custom node — VAE high-frequency noise auto-detection & repair.

Aligned 1-to-1 with the standalone GUI:

  * Same 11 preview modes (Original / Mask Overlay / Mask Only / Repaired /
    Side-by-Side / Laplacian binary / Median binary / Seed / Context /
    Filtered / Verified).
  * Same parameter triple ``(gradient_sensitivity, max_noise_size, mask_dilate)``.
  * Same resolution-aware auto-scaling of ``max_noise_size`` (handled
    inside ``GradientNoiseDetector``).
  * Same dual-output pipeline: an ``IMAGE`` selected by ``preview_mode``
    plus a ``MASK`` of the final dilated detection that downstream nodes
    can re-use.

Adheres to SOLID:

  * **SRP** — this file only exposes the ComfyUI binding.  All algorithm
    logic lives in ``core/``.
  * **OCP** — adding a new preview mode requires editing exactly one
    enum and one dispatch dict in ``core/pipeline.py``; the node is
    untouched.
  * **DIP** — depends on ``NoiseFixPipeline`` (composition root) rather
    than on individual stage classes.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

# Dual-mode import: relative when loaded by ComfyUI as a package
# (custom_nodes/<dir>/__init__.py → ``from .comfyui_node import …``),
# absolute when run via standalone scripts where this module is top-level.
try:
    from .core import (
        NoiseFixPipeline,
        PreviewMode,
        TensorBridge,
    )
except ImportError:
    from core import (
        NoiseFixPipeline,
        PreviewMode,
        TensorBridge,
    )


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
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VAENoiseFix": VAENoiseFixNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAENoiseFix": "VAE Noise Fix (Traditional CV)",
}
