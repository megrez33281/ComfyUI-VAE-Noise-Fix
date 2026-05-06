"""
ComfyUI-VAE-Noise-Fix

A lightweight custom node that detects and repairs SDXL VAE high-frequency
artifacts using traditional computer vision (Laplacian + CCA + Telea).
Zero neural-network dependency.

Installation:
    Clone or symlink this directory into ``ComfyUI/custom_nodes/``.
    Requires: opencv-python, numpy, torch (bundled with ComfyUI).
"""

from .vae_noise_fix import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
