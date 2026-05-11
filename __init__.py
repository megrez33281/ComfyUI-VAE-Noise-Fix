"""
ComfyUI-VAE-Noise-Fix custom node package.

A lightweight custom node that detects and repairs SDXL VAE
high-frequency artifacts ("fireflies") using traditional computer vision
(Laplacian + Median residual + CCA + LAB chroma + Telea).
Zero neural-network dependency.

Project layout:

    core/             - algorithm package (SOLID-modularised)
    comfyui_node.py   - ComfyUI binding
    mask_editor_server.py - API routes for the interactive mask editor
    web/js/           - frontend JavaScript extensions
    gui/              - standalone OpenCV preview application
    gui_preview.py    - GUI entry point
    vae_noise_fix.py  - backward-compat shim (legacy import path)

Installation as a ComfyUI custom node:
    Clone or symlink this directory into ``ComfyUI/custom_nodes/``.
    Requires: opencv-python, numpy, torch (bundled with ComfyUI).
"""

from .comfyui_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Expose the web/ directory so ComfyUI serves the JS frontend extension.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

