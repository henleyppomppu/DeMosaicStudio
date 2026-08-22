"""Degradation analysis. prd.md §5.4."""

from __future__ import annotations

from .motion import MotionBand, MotionSummary, classify, content_shift, cumulative_content_shifts, summarize
from .profile import DegradationType, GridAnchor, MosaicProfile, band_for

__all__ = [
    "DegradationType",
    "GridAnchor",
    "MosaicProfile",
    "MotionBand",
    "MotionSummary",
    "band_for",
    "classify",
    "content_shift",
    "cumulative_content_shifts",
    "summarize",
]
