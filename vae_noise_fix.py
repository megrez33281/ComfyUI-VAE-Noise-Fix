"""
Backwards-compatibility shim.

Earlier versions of the project distributed every class (TensorBridge,
GradientNoiseDetector, TeleaInpainter, DebugOverlayRenderer,
VAENoiseFixNode, ...) from a single ``vae_noise_fix.py`` module.  The
codebase has since been split into ``core/`` (algorithms) and
``comfyui_node.py`` (ComfyUI binding) following SOLID.

To keep existing imports working — including the bundled
``test_vae_noise_fix.py`` and any third-party scripts — this module
re-exports the public API under its previous names.

New code SHOULD import from ``core`` directly, e.g.

    from core import GradientNoiseDetector, NoiseFixPipeline, PreviewMode
    from comfyui_node import VAENoiseFixNode
"""

from __future__ import annotations

# Dual-mode imports: relative when loaded as part of the ComfyUI package,
# absolute when imported directly (e.g. by standalone test scripts).
try:
    from .core import (
        TensorBridge,
        GradientNoiseDetector,
        TeleaInpainter,
        DebugOverlayRenderer,
        MaskSoloRenderer,
        SideBySideRenderer,
        NoiseStatistics,
        NoiseFixPipeline,
        DetectionResult,
        PreviewMode,
    )
    from .comfyui_node import (
        VAENoiseFixNode,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:
    from core import (
        TensorBridge,
        GradientNoiseDetector,
        TeleaInpainter,
        DebugOverlayRenderer,
        MaskSoloRenderer,
        SideBySideRenderer,
        NoiseStatistics,
        NoiseFixPipeline,
        DetectionResult,
        PreviewMode,
    )
    from comfyui_node import (
        VAENoiseFixNode,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )

__all__ = [
    # Legacy classes (originally in this module)
    "TensorBridge",
    "GradientNoiseDetector",
    "TeleaInpainter",
    "DebugOverlayRenderer",
    "VAENoiseFixNode",
    # New public API (forwarded for convenience)
    "MaskSoloRenderer",
    "SideBySideRenderer",
    "NoiseStatistics",
    "NoiseFixPipeline",
    "DetectionResult",
    "PreviewMode",
    # ComfyUI registration
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
