"""Output timing rules. prd.md §5.1.7 (FR-1.7).

v1.0 of the PRD said "processing is PTS-based" and stopped there. The rule that actually prevents
A/V drift is about *output*, and it is small enough to state exactly:

* The output frame for source frame *f* carries **the source PTS of f**, rescaled to the output time
  base. Nothing is dropped, duplicated, or retimed by the restoration pipeline.
* CFR in, CFR out. VFR in, VFR out, preserving per-frame durations.
* ``output_frames == input_frames``. Asserted, never assumed.
* Audio is never resampled or shifted, so A/V sync is preserved by construction rather than by
  correction.

Everything here is pure arithmetic over timestamps, so it is testable without a decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final

#: A source frame with no PTS is a hole in the timeline, not a zero. prd.md §5.1.7.
MISSING_PTS: Final[None] = None


def rescale_pts(pts: int, source_time_base: Fraction, target_time_base: Fraction) -> int:
    """Rescales a timestamp between time bases, rounding half away from zero.

    FFmpeg's own ``av_rescale_q`` rounds to nearest with ties away from zero; matching it keeps our
    timestamps identical to what a straight remux would have produced, which is what makes the
    frame-count and PTS assertions in the tests meaningful.
    """
    numerator = pts * source_time_base.numerator * target_time_base.denominator
    denominator = source_time_base.denominator * target_time_base.numerator

    if denominator == 0:
        raise ValueError("target time base has a zero numerator")

    if (numerator < 0) != (denominator < 0):
        return -((-numerator * 2 + abs(denominator)) // (2 * abs(denominator)))

    return (numerator * 2 + abs(denominator)) // (2 * abs(denominator))


@dataclass(frozen=True, slots=True)
class TimelineCheck:
    """The result of comparing an output timeline against its source. prd.md §5.1.7."""

    source_frames: int
    output_frames: int
    max_pts_error_ticks: int
    monotonic: bool

    @property
    def frame_count_preserved(self) -> bool:
        """``output_frames == input_frames`` -- the invariant, not a guideline."""
        return self.source_frames == self.output_frames

    def is_faithful(self, tolerance_ticks: int = 1) -> bool:
        """True when the output timeline reproduces the source within one time-base tick."""
        return (
            self.frame_count_preserved
            and self.monotonic
            and self.max_pts_error_ticks <= tolerance_ticks
        )

    def describe(self) -> str:
        """A one-line summary for logs and test failures."""
        return (
            f"frames {self.output_frames}/{self.source_frames}, "
            f"max PTS error {self.max_pts_error_ticks} ticks, "
            f"monotonic={self.monotonic}"
        )


def check_timeline(
    source_pts: list[int],
    output_pts: list[int],
    source_time_base: Fraction,
    output_time_base: Fraction,
) -> TimelineCheck:
    """Compares an output timeline against its source.

    Both lists are in **presentation order**. Decode order is not presentation order whenever
    B-frames are involved, and comparing the two orders would produce a spurious failure.
    """
    expected = [rescale_pts(p, source_time_base, output_time_base) for p in source_pts]

    monotonic = all(b > a for a, b in zip(output_pts, output_pts[1:], strict=False))

    max_error = 0
    for want, got in zip(expected, output_pts, strict=False):
        max_error = max(max_error, abs(want - got))

    return TimelineCheck(
        source_frames=len(source_pts),
        output_frames=len(output_pts),
        max_pts_error_ticks=max_error,
        monotonic=monotonic if len(output_pts) > 1 else True,
    )


def is_variable_frame_rate(pts: list[int], tolerance_ticks: int = 1) -> bool:
    """Detects VFR from a presentation-ordered PTS list.

    A container's declared frame rate is a hint, not a fact: a VFR stream frequently declares a
    nominal rate. Measuring the deltas is the only reliable answer, and the answer decides whether
    the output is written CFR or VFR (§5.1.7).
    """
    if len(pts) < 3:
        return False

    deltas = [b - a for a, b in zip(pts, pts[1:], strict=False)]
    return max(deltas) - min(deltas) > tolerance_ticks
