"""Global motion estimation. prd.md §5.6, §1.4.1.

Two places need to know how fast a shot moves:

* The temporal window policy (§5.6) picks ``K`` from motion in pixels per frame.
* The feasibility gate (§1.4.1) depends on motion, because a **screen-anchored** mosaic grid only
  yields phase diversity when the subject moves across it. A static shot under a static grid gives
  the multi-frame path nothing to work with no matter how many frames it is handed.

Phase correlation gives global translation to sub-pixel accuracy using nothing but an FFT, which
keeps this dependency-free and deterministic. It measures *camera* motion, not object motion — good
enough to stratify a corpus and to drive the window policy, and honest about what it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MotionBand(str, Enum):
    """Motion classes from prd.md §5.6."""

    STATIC = "static"
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


#: Boundaries in pixels per frame. The slow/medium and medium/fast edges are §5.6's thresholds;
#: STATIC is split out of SLOW because a truly static shot is the interesting failure case for
#: phase diversity (§1.4.1), not merely a slower version of a slow one.
STATIC_MAX = 0.25
SLOW_MAX = 1.0
MEDIUM_MAX = 6.0


def classify(pixels_per_frame: float) -> MotionBand:
    """Bins a motion magnitude into a band."""
    if pixels_per_frame < STATIC_MAX:
        return MotionBand.STATIC
    if pixels_per_frame < SLOW_MAX:
        return MotionBand.SLOW
    if pixels_per_frame <= MEDIUM_MAX:
        return MotionBand.MEDIUM
    return MotionBand.FAST


def _hann2d(height: int, width: int) -> np.ndarray:
    # Without a window the FFT sees the frame edges as a huge discontinuity, and the correlation
    # peak lands on (0, 0) for every pair.
    return np.outer(np.hanning(height), np.hanning(width))


def estimate_translation(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Estimates global translation from ``previous`` to ``current`` by phase correlation.

    Both inputs are 2-D luma arrays of the same shape. Returns ``(dx, dy)`` in pixels, with
    sub-pixel refinement from a parabolic fit around the correlation peak.
    """
    if previous.shape != current.shape:
        raise ValueError(f"shape mismatch: {previous.shape} vs {current.shape}")
    if previous.ndim != 2:
        raise ValueError("expected 2-D luma arrays")

    height, width = previous.shape
    window = _hann2d(height, width)

    a = np.fft.rfft2(previous.astype(np.float64) * window)
    b = np.fft.rfft2(current.astype(np.float64) * window)

    cross = a * np.conj(b)
    magnitude = np.abs(cross)
    magnitude[magnitude == 0] = 1e-12

    correlation = np.fft.irfft2(cross / magnitude, s=(height, width))

    peak = int(np.argmax(correlation))
    py, px = divmod(peak, width)

    dy = _subpixel(correlation[:, px], py, height)
    dx = _subpixel(correlation[py, :], px, width)

    return dx, dy


def _subpixel(line: np.ndarray, index: int, length: int) -> float:
    """Parabolic refinement around a correlation peak, wrapped to a signed shift."""
    left = line[(index - 1) % length]
    centre = line[index]
    right = line[(index + 1) % length]

    denominator = left - 2.0 * centre + right
    offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (left - right) / denominator

    shift = index + offset
    if shift > length / 2:
        shift -= length

    return float(shift)


@dataclass(frozen=True, slots=True)
class MotionSummary:
    """Per-clip motion, used to stratify the evaluation set (§11.5)."""

    mean_pixels_per_frame: float
    median_pixels_per_frame: float
    max_pixels_per_frame: float
    frames: int

    @property
    def band(self) -> MotionBand:
        """The band the clip belongs to, taken from the median so one whip pan does not reclassify it."""
        return classify(self.median_pixels_per_frame)


def summarize(luma_frames: list[np.ndarray]) -> MotionSummary:
    """Measures motion across a sequence of luma frames."""
    if len(luma_frames) < 2:
        return MotionSummary(0.0, 0.0, 0.0, len(luma_frames))

    magnitudes = []
    for previous, current in zip(luma_frames, luma_frames[1:], strict=False):
        dx, dy = estimate_translation(previous, current)
        magnitudes.append(float(np.hypot(dx, dy)))

    array = np.asarray(magnitudes)

    return MotionSummary(
        mean_pixels_per_frame=float(array.mean()),
        median_pixels_per_frame=float(np.median(array)),
        max_pixels_per_frame=float(array.max()),
        frames=len(luma_frames),
    )


def content_shift(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Displacement of the *content* from ``previous`` to ``current``, in pixels.

    :func:`estimate_translation` reports the registration shift, which is the negation of how the
    content moved. Reconstruction needs the content displacement — feeding it the registration
    shift warps every neighbour the wrong way and turns a working multi-frame solve into one that
    scores *worse* than single-frame. Both signs exist here so neither caller has to remember.
    """
    dx, dy = estimate_translation(previous, current)
    return -dx, -dy


def cumulative_content_shifts(
    luma_frames: list[np.ndarray],
    target_index: int,
) -> list[tuple[float, float]]:
    """Content displacement of every frame relative to ``luma_frames[target_index]``.

    Accumulated pairwise rather than measured directly against the target: over a window of a few
    frames the drift is small, and pairwise correlation stays reliable where a direct
    target-to-far-frame correlation would already have lost the peak.
    """
    if not luma_frames:
        return []
    if not 0 <= target_index < len(luma_frames):
        raise IndexError("target_index outside the frame list")

    shifts: list[tuple[float, float]] = [(0.0, 0.0)] * len(luma_frames)

    dx = dy = 0.0
    for index in range(target_index + 1, len(luma_frames)):
        step_x, step_y = content_shift(luma_frames[index - 1], luma_frames[index])
        dx += step_x
        dy += step_y
        shifts[index] = (dx, dy)

    dx = dy = 0.0
    for index in range(target_index - 1, -1, -1):
        step_x, step_y = content_shift(luma_frames[index + 1], luma_frames[index])
        dx += step_x
        dy += step_y
        shifts[index] = (dx, dy)

    return shifts
