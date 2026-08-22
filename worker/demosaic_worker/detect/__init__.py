"""Detection. prd.md §5.2."""

from __future__ import annotations

from .regions import Region, extract_regions, iou
from .unet import MosaicUNet

__all__ = ["MosaicUNet", "Region", "extract_regions", "iou"]
