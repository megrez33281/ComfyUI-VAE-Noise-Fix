"""
Tensor / ndarray bridge.

Single responsibility: marshal data between the ComfyUI PyTorch tensor
representation (``[B, H, W, C]`` float32 RGB ∈ [0, 1]) and the OpenCV
NumPy representation (``[H, W, C]`` uint8 BGR ∈ [0, 255]) with the
minimum number of GPU↔CPU transfers and minimum number of buffer copies.

Putting these adapters in a dedicated module satisfies SRP — none of the
algorithmic modules need to know that the upstream caller is ComfyUI.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


class TensorBridge:
    """Stateless utility class for ComfyUI ↔ OpenCV format conversions."""

    # -- ComfyUI ➜ OpenCV ----------------------------------------------------

    @staticmethod
    def comfyui_to_cv2(tensor: torch.Tensor) -> np.ndarray:
        """Convert a single ComfyUI frame ``[H, W, 3]`` to an OpenCV BGR uint8 array.

        The tensor is moved to CPU exactly once; subsequent quantisation and
        channel swap operate entirely in NumPy to avoid redundant copies.
        """
        rgb_f32: np.ndarray = tensor.detach().cpu().numpy()
        rgb_u8: np.ndarray = np.clip(rgb_f32 * 255.0, 0, 255).astype(np.uint8)
        bgr_u8: np.ndarray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        return bgr_u8

    # -- OpenCV ➜ ComfyUI ----------------------------------------------------

    @staticmethod
    def cv2_to_comfyui(bgr_u8: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert an OpenCV BGR uint8 array back to a ComfyUI ``[H, W, 3]`` tensor."""
        rgb_u8: np.ndarray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)
        rgb_f32: np.ndarray = rgb_u8.astype(np.float32) / 255.0
        return torch.from_numpy(rgb_f32).to(device)

    # -- Mask helpers --------------------------------------------------------

    @staticmethod
    def mask_to_cv2(mask_f32: np.ndarray) -> np.ndarray:
        """Quantise a float32 binary mask ``{0.0, 1.0}`` to uint8 ``{0, 255}``."""
        return (mask_f32 * 255.0).astype(np.uint8)

    @staticmethod
    def mask_u8_to_comfyui(mask_u8: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert a uint8 mask ``{0, 255}`` to a ComfyUI MASK tensor ``[H, W]`` float32."""
        return torch.from_numpy((mask_u8 > 0).astype(np.float32)).to(device)

    # -- Luminance utility ---------------------------------------------------

    @staticmethod
    def grayscale_rec709(bgr_u8: np.ndarray) -> np.ndarray:
        """Compute perceptual luminance following ITU-R BT.709 weighting.

            Y = 0.2126·R + 0.7152·G + 0.0722·B

        OpenCV's ``cvtColor(..., COLOR_BGR2GRAY)`` uses BT.601 weights, which
        slightly under-weight red channel content. BT.709 is preferred for
        SDXL content generated at sRGB gamut.
        """
        b, g, r = cv2.split(bgr_u8)
        y = (0.2126 * r.astype(np.float32)
             + 0.7152 * g.astype(np.float32)
             + 0.0722 * b.astype(np.float32))
        return np.clip(y, 0, 255).astype(np.uint8)
