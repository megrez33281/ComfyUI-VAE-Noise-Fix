"""
Core package — Zero-NN VAE noise detection & repair pipeline.

This package decomposes the pipeline into small, single-responsibility
units following the SOLID principles:

    * tensor_bridge      — ComfyUI tensor / OpenCV ndarray conversion.
    * energy             — Laplacian & median-residual energy extractors.
    * thresholding       — Dual-path binary mask generation.
    * structural_filter  — CCA-based area / shape / isolation filtering.
    * chromatic_filter   — CIE-LAB cliff & chroma verification.
    * morphology         — Mask dilation.
    * detector           — Orchestrates the detection chain.
    * inpainter          — Telea Fast-Marching inpainting.
    * overlay            — Visualization renderers (overlay, mask solo, etc).
    * statistics         — Noise statistics calculator.
    * pipeline           — End-to-end pipeline + ``DetectionResult`` DTO.

Each consumer (ComfyUI node, GUI, test scripts) imports only what it
needs and depends on stable abstractions, not concrete implementations.
"""

from .tensor_bridge import TensorBridge
from .energy import LaplacianEnergyExtractor, MedianResidualExtractor
from .thresholding import DualPathMaskGenerator
from .structural_filter import StructuralFilter
from .chromatic_filter import ChromaticFilter
from .morphology import MaskDilator
from .detector import GradientNoiseDetector
from .inpainter import TeleaInpainter
from .overlay import (
    DebugOverlayRenderer,
    MaskSoloRenderer,
    SideBySideRenderer,
)
from .statistics import NoiseStatistics
from .pipeline import DetectionResult, NoiseFixPipeline, PreviewMode

__all__ = [
    # Bridges
    "TensorBridge",
    # Detection stages
    "LaplacianEnergyExtractor",
    "MedianResidualExtractor",
    "DualPathMaskGenerator",
    "StructuralFilter",
    "ChromaticFilter",
    "MaskDilator",
    # Orchestrators
    "GradientNoiseDetector",
    "TeleaInpainter",
    # Visualization
    "DebugOverlayRenderer",
    "MaskSoloRenderer",
    "SideBySideRenderer",
    "NoiseStatistics",
    # Pipeline
    "DetectionResult",
    "NoiseFixPipeline",
    "PreviewMode",
]
