"""Local generative carrier utilities (BrushNet Phase 3A + RevBridge local).

BrushNetAdapter is lazy-imported so RevBridge inference paths do not pull BrushNet.
"""

from stegoanything.generation.generation_preprocess import GeometryTransform, preprocess_image_mask, invert_geometry
from stegoanything.generation.local_composite import hard_composite, feather_composite

__all__ = [
    "BrushNetAdapter",
    "GeometryTransform",
    "preprocess_image_mask",
    "invert_geometry",
    "hard_composite",
    "feather_composite",
]


def __getattr__(name: str):
    if name == "BrushNetAdapter":
        from stegoanything.generation.brushnet_adapter import BrushNetAdapter

        return BrushNetAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
