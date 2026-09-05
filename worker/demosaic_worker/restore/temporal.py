"""Temporal smoothing of single-frame restorations. D-43.

A single-frame upscaler invents a slightly different texture every frame, and the eye reads that
as shimmer. Blending each frame's output with the previous one, per track, damps it — the
seven-to-three blend the proposal named, exposed as ``temporalAlpha``.

This is the naive cousin of :class:`~demosaic_worker.restore.accumulator.EvidenceAccumulator`.
That one warps the carried estimate by optical flow before folding; this one does not warp at
all, which is why it costs nothing and why it is only valid where the content under the region
did not move between frames. The re-embed step handles the *region* moving — the crop's bounds
shifting frame to frame — which is a different thing from the *content* moving.

**The blend is applied only where the observation did not change.** Measured on the quality
fixture — a screen-fixed mosaic over a panning picture — an unconditional 7:3 blend scored
**-4.52 dB** inside the region against +1.00 dB for the same restorer with no blend at all: every
frame carried a ghost of the previous one. The mosaicked crop itself is the observation, and it
is deterministic per frame, so comparing this frame's crop with the previous one (re-embedded)
says exactly which pixels the picture moved under. Those take the new frame at full weight; the
rest are blended. A face under a mosaic that follows it is unchanged block to block and blends;
a pan is changed everywhere and does not. The threshold is in luma levels at block resolution,
where codec noise sits well below it and a shift of one block sits well above.

Resets on the same events the accumulator resets on: a scene cut, a gap in the frame sequence, a
region that jumped rather than moved. Without those, the first frame after a cut would be blended
with the last frame before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .accumulator import MIN_ROI_OVERLAP, _reembed

#: The default weight on the *new* frame where the observation is unchanged. 0.3 is the 7:3
#: blend. Where the observation changed the weight is 1.0 regardless. Not swept: D-29 found the
#: accumulator's equivalent knob varies eightfold across content, and this one has no
#: measurement behind it beyond the one that made it motion-aware.
DEFAULT_ALPHA = 0.3

#: Absolute change in the observed crop, in luma levels, above which a pixel is treated as moved
#: and takes the new frame at full weight. A block mean averages away codec noise to well under
#: a level; a one-block shift of any real picture moves it by tens. Swept on the quality fixture
#: (panning picture, 7:3 blend): 6.0 scored +0.20 dB, 2.0 +0.97, **1.0 +1.06**, 0.5 +1.02 -
#: against +1.00 with no blend at all. Above about 2 the flat parts of a moving picture blend
#: with their own ghost; at 1.0 the blend applies only where it helps.
MOTION_TOLERANCE = 1.0


@dataclass
class _Carried:
    output: np.ndarray
    observation: np.ndarray
    bounds: tuple[int, int, int, int]
    frame_index: int


@dataclass
class TemporalSmoother:
    """Per-track exponential blend of restorations, with the resets that make it safe."""

    alpha: float = DEFAULT_ALPHA
    # Read at construction, not at class creation, so a sweep can change the module constant.
    motion_tolerance: float = field(default_factory=lambda: MOTION_TOLERANCE)
    _tracks: dict[int, _Carried] = field(default_factory=dict)

    def smooth(
        self,
        track_id: int,
        output: np.ndarray,
        *,
        observation: np.ndarray,
        bounds: tuple[int, int, int, int],
        frame_index: int,
        same_scene: bool = True,
    ) -> np.ndarray:
        """Blends this frame's restoration with what was carried for the track, and carries the result.

        ``observation`` is the mosaicked crop the restoration was made from: the blend applies
        only where it matches the previous frame's. ``bounds`` are the crop's bounds in frame
        coordinates; a shift is re-embedded, a jump resets. ``frame_index`` must advance by exactly
        one for the carry to be used - a skipped frame is a gap, and blending across it would
        smear whatever happened in between.
        """
        previous = self._tracks.get(track_id)
        fresh = output.astype(np.float64)
        seen = observation.astype(np.float64)

        reset = (
            previous is None
            or not same_scene
            or frame_index != previous.frame_index + 1
            or _overlap(previous.bounds, bounds) < MIN_ROI_OVERLAP
            or seen.shape != fresh.shape
        )

        if reset:
            blended = fresh
        else:
            carried = _reembed(previous.output, previous.bounds, bounds, fresh)
            # Filled with this frame's observation where the old crop did not reach, so those
            # pixels read as unchanged and blend with themselves - i.e. take the fresh value.
            before = _reembed(previous.observation, previous.bounds, bounds, seen)
            moved = np.abs(seen - before) > self.motion_tolerance
            weight = np.where(moved, 1.0, self.alpha)
            blended = (1.0 - weight) * carried + weight * fresh

        self._tracks[track_id] = _Carried(blended, seen, bounds, frame_index)
        return blended

    def reset(self, track_id: int | None = None) -> None:
        """Forgets one track, or every track."""
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    def has(self, track_id: int) -> bool:
        """True when the track has something carried."""
        return track_id in self._tracks


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over the *smaller* rectangle, so a region that grew still counts as the same."""
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (width * height) / smaller if smaller > 0 else 0.0
