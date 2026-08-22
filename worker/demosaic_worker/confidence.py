"""Restoration confidence buckets. prd.md §5.9.4.

Mirror of ``DeMosaicStudio.Domain.Policies.Confidence`` (§13.4).
"""

from __future__ import annotations

from enum import Enum
from typing import Final

#: Lower bound of :attr:`Bucket.HIGH`.
HIGH_THRESHOLD: Final[float] = 0.66

#: Lower bound of :attr:`Bucket.MEDIUM`.
MEDIUM_THRESHOLD: Final[float] = 0.33


class Bucket(str, Enum):
    """Qualitative confidence band."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


def bucket(confidence: float) -> Bucket:
    """Classifies a confidence value."""
    if confidence >= HIGH_THRESHOLD:
        return Bucket.HIGH
    if confidence >= MEDIUM_THRESHOLD:
        return Bucket.MEDIUM
    return Bucket.LOW
