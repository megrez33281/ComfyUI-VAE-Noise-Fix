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
    web/js/           - frontend JavaScript extensions (mask editor popup)
    gui/              - standalone tkinter preview / tuning application
    gui_preview.py    - GUI entry point
    vae_noise_fix.py  - backward-compat shim (legacy import path)

Installation as a ComfyUI custom node:
    Install "VAE Noise Fix" from ComfyUI-Manager / the Comfy Registry, or
    clone this directory into ``ComfyUI/custom_nodes/``.
    Requires: opencv-python, numpy (torch and Pillow come with ComfyUI).
"""

# Dual-mode import: relative when ComfyUI loads this directory as a
# package, absolute when this file is imported as a plain module (pytest
# imports a test tree's parent ``__init__.py`` during collection).
try:
    from .comfyui_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    from comfyui_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Expose the web/ directory so ComfyUI serves the JS frontend extension.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

