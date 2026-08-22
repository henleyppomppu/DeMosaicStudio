"""Evidence carried forward, rather than re-gathered every frame. prd.md §5.6, §5.7, D-28.

The batch form asks, for each frame, "align me to my K neighbours and solve". That costs K
alignments and K times the solver work **per frame**, and the corrected forward model (D-26) needs
K of about 17 before it earns its keep. Measured: 0.23 fps against §6.1's 4 fps target.

This carries one estimate per track instead. Each frame it is warped by a single frame-to-frame
flow and the new observation is folded in. The cost is one alignment and one warp regardless of how
far back the evidence goes, and the history it accumulates is unbounded rather than K.

It is also **better**, not merely cheaper, and for a reason the project already measured:
``docs/phase2-alignment-report.md`` §3 found that shorter baselines align better. The batch form
aligns the target directly to a frame 24 away; this chains 24 one-frame alignments. On the
screen-anchored clip, 24 frames of history:

===============================  ==========  =======
form                             PSNR        vs input
===============================  ==========  =======
mosaicked input                  24.12       -
batch, 24 neighbours             26.04       +1.92
**accumulator**                  **28.96**   **+4.84**
===============================  ==========  =======

and it holds up on real inputs - +4.55 dB with the detector's mask and an estimated grid rather
than ground truth, and +2.46 dB even on object-anchored content, where a batch window of 3 sees
almost nothing but 24 accumulated crescents add up to a third of the region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..analyze.profile import MosaicProfile
from .ibp import forward_and_adjoint

#: Below this overlap between the previous ROI and the current one, the estimate is not carried:
#: the region has moved far enough that most of what was accumulated is about somewhere else.
MIN_ROI_OVERLAP = 0.35

#: How many frames of evidence are worth carrying, as an exponential horizon. **Measured.**
#:
#: Quality does not rise forever with the chain: it peaks and then falls, because the oldest
#: evidence in a chain of N has been warped N times and carries N frames of the flow's error with
#: it. On the screen-anchored clip it peaks at depth 16 and loses 0.8 dB by 48; on the
#: object-anchored one it peaks at 32 and loses 0.6 dB by 48.
#:
#: A depth cap would be a *reset* - it discards everything and starts at zero, which oscillates.
#: Decaying the carried estimate towards what the current frame observed bounds the horizon without
#: ever discarding anything. At depth 48:
#:
#: =========  =========  =========
#: horizon    screen     ladder
#: =========  =========  =========
#: off        +3.27      +2.72
#: 64         +4.49      +2.85
#: **32**     **+5.28**  **+2.81**
#: 16         +5.97      +2.54
#: 8          +6.01      +1.98
#: =========  =========  =========
#:
#: 32 improves both. The optimum itself differs between them - 8-16 for the fast pan, 32-64 for the
#: slow drift - which says the governing quantity is not a frame count. Two clips cannot say what it
#: is; the likely candidate is accumulated warping error, which grows with the motion per frame.
EVIDENCE_HORIZON_FRAMES = 32


@dataclass(slots=True)
class TrackEvidence:
    """One track's running estimate of the scene behind the mosaic."""

    #: Accumulated luma, in the coordinates of the frame it was last updated on.
    estimate: np.ndarray

    #: The ROI the estimate covers, in frame coordinates, as ``(left, top, right, bottom)``.
    bounds: tuple[int, int, int, int]

    #: The frame index it was last updated on. A gap means the chain is broken.
    frame_index: int

    #: How many observations have been folded in. This is the evidence depth the router asks about
    #: -- with an accumulator it is a count of history, not of simultaneously aligned neighbours.
    depth: int = 0


@dataclass(slots=True)
class EvidenceAccumulator:
    """Per-track evidence, carried forward across frames.

    Reset is not an optimisation here: a scene cut, a lost track or a region that jumped means the
    accumulated pixels describe different content, and carrying them would composite one shot's
    picture into another. The resets are listed on :meth:`update` and each returns depth to zero.
    """

    tracks: dict[int, TrackEvidence] = field(default_factory=dict)

    #: Exponential forgetting horizon in frames; zero or below disables it.
    horizon: int = EVIDENCE_HORIZON_FRAMES

    def depth(self, track_id: int) -> int:
        """How many observations this track has absorbed. Zero if it has none."""
        evidence = self.tracks.get(track_id)
        return evidence.depth if evidence is not None else 0

    def forget(self, track_id: int) -> None:
        """Drops a terminated track's evidence."""
        self.tracks.pop(track_id, None)

    def reset(self, track_id: int) -> None:
        """Breaks the chain without dropping the entry, so the next frame starts fresh."""
        self.tracks.pop(track_id, None)

    def update(
        self,
        track_id: int,
        *,
        frame_index: int,
        observation: np.ndarray,
        previous_observation: np.ndarray | None,
        bounds: tuple[int, int, int, int],
        mask: np.ndarray | None,
        spec: MosaicProfile,
        phase: tuple[int, int],
        flow_to_previous: np.ndarray | None,
        same_scene: bool = True,
    ) -> np.ndarray:
        """Folds one frame's observation into this track's estimate and returns it.

        ``observation`` is this ROI's crop of the current frame and ``previous_observation`` the
        **same ROI's** crop of the previous one - the two therefore have identical shapes, which is
        what lets an estimate be carried when the region moves. ``bounds`` is the rectangle the crop
        corresponds to in frame coordinates (:attr:`Roi.crop_bounds`, not :attr:`Roi.bounds`: the
        reflect padding is part of the crop). ``flow_to_previous`` maps each pixel of the current
        crop to where it was in the previous one.

        The chain restarts -- returning the observation unchanged, which is what a single frame of
        evidence can honestly produce -- when any of these holds:

        * the track has no evidence yet;
        * the previous update was not on the immediately preceding frame;
        * a scene cut falls between the two;
        * no usable flow was available;
        * the ROI moved so far that less than :data:`MIN_ROI_OVERLAP` of it is shared.
        """
        previous = self.tracks.get(track_id)

        restart = (
            previous is None
            or previous_observation is None
            or flow_to_previous is None
            or not same_scene
            or previous.frame_index != frame_index - 1
            or _overlap(previous.bounds, bounds) < MIN_ROI_OVERLAP
        )

        if restart:
            self.tracks[track_id] = TrackEvidence(
                estimate=observation.astype(np.float64).copy(),
                bounds=bounds,
                frame_index=frame_index,
                depth=1,
            )
            return self.tracks[track_id].estimate

        assert previous is not None and previous_observation is not None  # narrowed above

        from .flow import warp_by_flow

        carried = _reembed(previous.estimate, previous.bounds, bounds, previous_observation)
        carried = warp_by_flow(carried, flow_to_previous)

        if self.horizon > 0:
            # Forget slowly. Evidence that has been warped this many times carries more of the
            # flow's error than of the scene, and holding it costs more than it brings.
            decay = 1.0 / self.horizon
            carried = (1.0 - decay) * carried + decay * observation.astype(np.float64)

        _, spread = forward_and_adjoint(carried, observation, spec, phase, mask)
        estimate = np.clip(carried + spread, 0.0, 255.0)

        self.tracks[track_id] = TrackEvidence(
            estimate=estimate,
            bounds=bounds,
            frame_index=frame_index,
            depth=previous.depth + 1,
        )
        return estimate


def _overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    """Intersection over the *second* box: how much of the new ROI the old estimate covers."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])

    if right <= left or bottom <= top:
        return 0.0

    area = (second[2] - second[0]) * (second[3] - second[1])
    return ((right - left) * (bottom - top)) / area if area else 0.0


def _reembed(
    estimate: np.ndarray,
    old_bounds: tuple[int, int, int, int],
    new_bounds: tuple[int, int, int, int],
    fill: np.ndarray,
) -> np.ndarray:
    """Moves an estimate from one ROI to another, in the *previous* frame's coordinates.

    The region moves, so the ROI moves with it. Where the two overlap the accumulated estimate is
    carried across; where the new ROI reaches picture the old one never covered, ``fill`` - the
    previous frame seen through the *new* ROI - supplies it. Filling with zeros instead would
    inject a black border that the next fold would then treat as evidence.
    """
    left, top, right, bottom = new_bounds
    carried = fill.astype(np.float64).copy()

    overlap_left = max(old_bounds[0], left)
    overlap_top = max(old_bounds[1], top)
    overlap_right = min(old_bounds[2], right)
    overlap_bottom = min(old_bounds[3], bottom)

    if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
        return carried

    carried[
        overlap_top - top : overlap_bottom - top,
        overlap_left - left : overlap_right - left,
    ] = estimate[
        overlap_top - old_bounds[1] : overlap_bottom - old_bounds[1],
        overlap_left - old_bounds[0] : overlap_right - old_bounds[0],
    ]

    return carried
