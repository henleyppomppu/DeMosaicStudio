"""Mosaic degradation profile. prd.md §5.4.

This is the *description* of a degradation, used by the worker to decide how to restore it. The
degradation *generator* that produces training data lives in `training/degradation/` and imports
from here, never the other way round (AGENTS.md layer rules).

The field that matters most is :attr:`MosaicProfile.anchor`. §1.4.1 explains why in principle;
`docs/phase0-report.md` §3.2 measured it: object-anchored multi-frame scores **0.79 to 1.50 dB below
single-frame** even with perfect alignment, because the neighbours contribute codec noise and no
information. Getting this field wrong produces output worse than not restoring at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Final


class GridAnchor(str, Enum):
    """How the mosaic grid is anchored. prd.md §1.4.1, §5.4.4.

    Wire and metadata values are ``SCREEN`` / ``OBJECT`` / ``UNKNOWN``. The host mirrors these as
    ``GridAnchor.Screen`` / ``ObjectTracked`` / ``Unknown`` — the C# name differs only because
    ``Object`` collides with a type name there.
    """

    SCREEN = "SCREEN"
    OBJECT = "OBJECT"
    UNKNOWN = "UNKNOWN"


class DegradationType(str, Enum):
    """Degradation families. prd.md §5.4.1."""

    PIXELATION = "PIXELATION"
    GAUSSIAN_BLUR = "GAUSSIAN_BLUR"
    BOX_BLUR = "BOX_BLUR"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class MosaicProfile:
    """One region's estimated degradation parameters. prd.md §5.4.

    ``block_width`` and ``block_height`` are independent: real tools produce non-square blocks often
    enough that assuming square would misestimate the grid on a noticeable fraction of input.
    """

    kind: DegradationType = DegradationType.PIXELATION
    block_width: int = 8
    block_height: int = 8
    grid_offset_x: int = 0
    grid_offset_y: int = 0
    anchor: GridAnchor = GridAnchor.SCREEN
    anchor_confidence: float = 0.0
    blur_sigma: float = 0.0
    opacity: float = 1.0
    degradation_strength: float = 1.0
    temporal_stability: float = 1.0
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.block_width < 1 or self.block_height < 1:
            raise ValueError("block size must be >= 1 px")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be in [0, 1]")

    @property
    def block_size(self) -> int:
        """A single representative block size, for band lookup and window policy."""
        return max(self.block_width, self.block_height)

    def phase_for(self, origin_x: int, origin_y: int) -> tuple[int, int]:
        """The grid phase to use for a region whose box origin sits at ``(origin_x, origin_y)``.

        This one method is where anchoring actually happens.

        * ``SCREEN`` — phase is measured in frame coordinates, so it does not follow the box. As the
          box moves, the subject crosses block boundaries and phase diversity accumulates.
        * ``OBJECT`` / ``UNKNOWN`` — phase is measured relative to the box origin, so the grid rides
          along and the phase the subject sees never changes.
        """
        if self.anchor is GridAnchor.SCREEN:
            return self.grid_offset_x % self.block_width, self.grid_offset_y % self.block_height

        return (
            (self.grid_offset_x - origin_x) % self.block_width,
            (self.grid_offset_y - origin_y) % self.block_height,
        )

    def with_geometry(self, block_width: int, block_height: int, offset_x: int, offset_y: int) -> "MosaicProfile":
        """Returns a copy with new grid geometry, leaving type and anchoring alone."""
        return replace(
            self,
            block_width=block_width,
            block_height=block_height,
            grid_offset_x=offset_x % block_width,
            grid_offset_y=offset_y % block_height,
        )


#: Recoverability bands from prd.md §1.4.2, confirmed by measurement in docs/phase0-report.md §3.2.
RECOVERABILITY_BANDS: Final[tuple[tuple[int, int, str], ...]] = (
    (1, 4, "deblocking; most output pixels evidence-backed"),
    (5, 12, "target band; genuine multi-frame reconstruction"),
    (13, 24, "prior-dominated; report Low confidence"),
    (25, 10_000, "information destroyed; fabrication"),
)


def band_for(block_size: int) -> str:
    """Names the §1.4.2 recoverability band a block size falls into."""
    for low, high, description in RECOVERABILITY_BANDS:
        if low <= block_size <= high:
            return description
    raise ValueError(f"block size {block_size} is outside every band")
