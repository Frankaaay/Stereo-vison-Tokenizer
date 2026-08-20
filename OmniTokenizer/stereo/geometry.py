"""Compatibility import while the original tokenizer is migrated in place."""

from ..modules.stereo_geometry import DepthOutput, disparity_to_depth

__all__ = ["DepthOutput", "disparity_to_depth"]
